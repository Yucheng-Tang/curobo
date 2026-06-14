# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relative tool pose cost: constrain the pose of one tool frame in another.

Two implementations sharing the same interface and configuration:

- :class:`RelativePoseCost` — fused Warp kernel with analytic gradients for
  both tool frames computed in the forward pass (one kernel launch per cost
  evaluation, CUDA-graph safe).
- :class:`TorchRelativePoseCost` — pure-torch quaternion composition relying
  on autograd. Slower (more kernel launches) but useful as a numerical
  reference and for rapid prototyping of error variants.
"""
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, List, Optional, Tuple

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.wp_relative_pose import (
    RelativePoseDistance,
    create_relative_pose_distance_kernel_with_constants,
)
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise, log_warn

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg


class RelativePoseCost(BaseCost):
    """Pose cost on ``T_rel = T_base_frame^-1 * T_tool_frame`` (fused Warp kernel)."""

    def __init__(self, config: RelativePoseCostCfg):
        self.config: RelativePoseCostCfg = config
        super().__init__(config)
        self._link_idx_base = -1
        self._link_idx_tool = -1
        self._num_links = -1
        self._warp_kernel: Optional[wp.kernel] = None

        device = self.device_cfg.device
        if config.goal_pose is not None:
            self._goal_position = config.goal_pose[:3].clone().view(1, 3)
            self._goal_quaternion = config.goal_pose[3:].clone().view(1, 4)
        else:
            log_warn(
                "RelativePoseCost: no goal_pose set; defaulting to identity "
                "(this pulls tool_frame onto base_frame). Set cfg.goal_pose or "
                "call update_goal() before solving."
            )
            self._goal_position = torch.zeros((1, 3), dtype=torch.float32, device=device)
            self._goal_quaternion = torch.zeros((1, 4), dtype=torch.float32, device=device)
            self._goal_quaternion[:, 0] = 1.0
        self._project_distance_to_goal = torch.tensor(
            [1 if config.project_distance_to_goal else 0], dtype=torch.uint8, device=device
        )

    def set_tool_frames(self, tool_frames: List[str]):
        """Resolve base/tool frame indices in the robot's tool frame list."""
        if self.config.base_frame not in tool_frames:
            log_and_raise(
                f"RelativePoseCost: base_frame {self.config.base_frame} not in robot "
                f"tool_frames {tool_frames}"
            )
        if self.config.tool_frame not in tool_frames:
            log_and_raise(
                f"RelativePoseCost: tool_frame {self.config.tool_frame} not in robot "
                f"tool_frames {tool_frames}"
            )
        self._link_idx_base = tool_frames.index(self.config.base_frame)
        self._link_idx_tool = tool_frames.index(self.config.tool_frame)
        self._num_links = len(tool_frames)

    def update_goal(self, goal_pose: Pose):
        """Update the goal relative pose in-place (CUDA-graph safe).

        Only a single relative goal is supported. Per-problem relative goals
        would need a dedicated index buffer in GoalRegistry (the cost manager
        currently forwards ``idxs_link_pose``, which maps the tool_pose
        goalset, not a relative-goal set).
        """
        position = goal_pose.position.view(-1, 3)
        quaternion = goal_pose.quaternion.view(-1, 4)
        if position.shape[0] != 1:
            log_and_raise(
                "RelativePoseCost.update_goal supports a single relative goal "
                f"(got {position.shape[0]} rows). Batched relative goals are "
                "not supported yet."
            )
        # shape is always (1, 3)/(1, 4) → in-place copy keeps captured pointers
        self._goal_position.copy_(position.to(self.device_cfg.device))
        self._goal_quaternion.copy_(quaternion.to(self.device_cfg.device))

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        if batch_size != self._batch_size or horizon != self._horizon:
            device = self.device_cfg.device
            if self._link_idx_base < 0:
                log_and_raise("RelativePoseCost: call set_tool_frames before use")
            num_links = self._num_links
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2), dtype=torch.float32, device=device
            )
            self._out_position_distance = torch.zeros(
                (batch_size, horizon, 1), dtype=torch.float32, device=device
            )
            self._out_rotation_distance = torch.zeros(
                (batch_size, horizon, 1), dtype=torch.float32, device=device
            )
            # gradients cover all stored tool frames; the kernel only writes the
            # two participating slots, the rest stay zero
            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, num_links, 3), dtype=torch.float32, device=device
            )
            self._out_rotation_gradient = torch.zeros(
                (batch_size, horizon, num_links, 4), dtype=torch.float32, device=device
            )
            self._idxs_zero = torch.zeros((batch_size, 1), dtype=torch.int32, device=device)
            # relative pose has no goalset; solver result extraction still
            # expects one goalset index column per pose-tolerance metric
            self._goalset_idx_zero = torch.zeros(
                (batch_size, horizon, 1), dtype=torch.int32, device=device
            )
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the relative pose cost.

        Args:
            current_tool_poses: FK tool poses, position (b, h, L, 3) and
                quaternion (b, h, L, 4) wxyz.
            idxs_goal: Optional (b, 1) int32 mapping problems to goal rows;
                only used when more than one goal relative pose is stored.

        Returns:
            cost (b, h, 2), position_error (b, h, 1), rotation_error (b, h, 1).
        """
        if self._warp_kernel is None:
            self._warp_kernel = create_relative_pose_distance_kernel_with_constants(
                self.config.rotation_method
            )
        if idxs_goal is None or self._goal_position.shape[0] == 1:
            idxs_goal = self._idxs_zero

        cost, position_error, rotation_error = RelativePoseDistance.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            self._goal_position,
            self._goal_quaternion,
            idxs_goal,
            self._weight,
            self.config.terminal_pose_axes_weight_factor,
            self.config.non_terminal_pose_axes_weight_factor,
            self.config.terminal_pose_convergence_tolerance,
            self.config.non_terminal_pose_convergence_tolerance,
            self._project_distance_to_goal,
            self._out_distance,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_position_gradient,
            self._out_rotation_gradient,
            self._link_idx_base,
            self._link_idx_tool,
            self.config.use_grad_input,
            self._warp_kernel,
        )
        return cost, position_error, rotation_error


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


class _FKQuaternionGradAdapter(torch.autograd.Function):
    """Adapt raw autograd quaternion gradients to the FK backward convention.

    The FK backward converts incoming quaternion gradients with
    ``omega = 0.5 * vec(q^-1 * g)`` (a body-frame adjoint) and dots the result
    with world-frame joint axes. That pairing is only frame-consistent for
    gradients packed as ``g = q * (omega_world, 0)`` — the packing used by the
    Warp cost kernels. A raw autograd gradient ``g`` must be conjugated to
    ``q * g * q^-1`` so the same conversion yields the exact world-frame
    angular gradient ``0.5 * vec(g * q^-1)``.
    """

    @staticmethod
    def forward(ctx, quaternion: torch.Tensor):
        ctx.save_for_backward(quaternion)
        return quaternion.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (quaternion,) = ctx.saved_tensors
        return _quat_multiply(
            quaternion, _quat_multiply(grad_output, _quat_conjugate(quaternion))
        )


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q[..., 0:1] * t + torch.cross(q_vec, t, dim=-1)


class TorchRelativePoseCost(BaseCost):
    """Pure-torch implementation of the relative pose cost (autograd backward).

    Implements ONLY the axis-angle error (rotation_method=0); it does not
    implement the Lie-group variants, so it raises if ``use_lie_group`` is set.
    Used as a numerical reference and for prototyping; gradients flow through
    torch autograd into the FK backward instead of being computed analytically
    in-kernel.
    """

    def __init__(self, config: RelativePoseCostCfg):
        self.config: RelativePoseCostCfg = config
        super().__init__(config)
        if config.rotation_method != 0:
            log_and_raise(
                "TorchRelativePoseCost implements only axis-angle rotation error "
                "(rotation_method=0); got rotation_method="
                f"{config.rotation_method} (use_lie_group={config.use_lie_group}). "
                "Use RelativePoseCost (the Warp kernel) for Lie-group rotation."
            )
        self._link_idx_base = -1
        self._link_idx_tool = -1
        self._num_links = -1
        device = self.device_cfg.device
        if config.goal_pose is not None:
            self._goal_position = config.goal_pose[:3].clone().view(1, 3)
            self._goal_quaternion = config.goal_pose[3:].clone().view(1, 4)
        else:
            log_warn(
                "TorchRelativePoseCost: no goal_pose set; defaulting to identity "
                "(pulls tool_frame onto base_frame). Set cfg.goal_pose or call "
                "update_goal() before solving."
            )
            self._goal_position = torch.zeros((1, 3), dtype=torch.float32, device=device)
            self._goal_quaternion = torch.zeros((1, 4), dtype=torch.float32, device=device)
            self._goal_quaternion[:, 0] = 1.0

    def set_tool_frames(self, tool_frames: List[str]):
        if self.config.base_frame not in tool_frames:
            log_and_raise(
                f"TorchRelativePoseCost: base_frame {self.config.base_frame} not in robot "
                f"tool_frames {tool_frames}"
            )
        if self.config.tool_frame not in tool_frames:
            log_and_raise(
                f"TorchRelativePoseCost: tool_frame {self.config.tool_frame} not in robot "
                f"tool_frames {tool_frames}"
            )
        self._link_idx_base = tool_frames.index(self.config.base_frame)
        self._link_idx_tool = tool_frames.index(self.config.tool_frame)
        self._num_links = len(tool_frames)

    def update_goal(self, goal_pose: Pose):
        position = goal_pose.position.view(-1, 3)
        quaternion = goal_pose.quaternion.view(-1, 4)
        if position.shape[0] != 1:
            log_and_raise(
                "TorchRelativePoseCost.update_goal supports a single relative "
                f"goal (got {position.shape[0]} rows)."
            )
        self._goal_position.copy_(position.to(self.device_cfg.device))
        self._goal_quaternion.copy_(quaternion.to(self.device_cfg.device))

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        if batch_size != self._batch_size or horizon != self._horizon:
            device = self.device_cfg.device
            if self._link_idx_base < 0:
                log_and_raise("TorchRelativePoseCost: call set_tool_frames before use")
            # goalset index buffer for the convergence path (no goalset → zeros);
            # parity with RelativePoseCost so the cost manager works for both.
            self._goalset_idx_zero = torch.zeros(
                (batch_size, horizon, 1), dtype=torch.int32, device=device
            )
            # per-timestep axis weights / tolerances: non-terminal for h < H-1
            axes = self.config.non_terminal_pose_axes_weight_factor.view(1, 6).repeat(horizon, 1)
            axes[-1] = self.config.terminal_pose_axes_weight_factor
            tol = self.config.non_terminal_pose_convergence_tolerance.view(1, 2).repeat(
                horizon, 1
            )
            tol[-1] = self.config.terminal_pose_convergence_tolerance
            if horizon == 1:
                axes[0] = self.config.terminal_pose_axes_weight_factor
                tol[0] = self.config.terminal_pose_convergence_tolerance
            self._axes_weight = axes.view(1, horizon, 6).to(device)
            self._tolerance_sq = (tol**2).view(1, horizon, 2).to(device)
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = current_tool_poses.position
        quaternion = current_tool_poses.quaternion
        p_base = position[:, :, self._link_idx_base]
        q_base = quaternion[:, :, self._link_idx_base]
        p_tool = position[:, :, self._link_idx_tool]
        q_tool = quaternion[:, :, self._link_idx_tool]
        if quaternion.requires_grad:
            q_base = _FKQuaternionGradAdapter.apply(q_base)
            q_tool = _FKQuaternionGradAdapter.apply(q_tool)

        goal_p = self._goal_position
        goal_q = self._goal_quaternion
        if goal_p.shape[0] > 1 and idxs_goal is not None:
            goal_p = goal_p[idxs_goal.view(-1).long()]
            goal_q = goal_q[idxs_goal.view(-1).long()]
        goal_p = goal_p.view(-1, 1, 3)
        goal_q = goal_q.view(-1, 1, 4)

        q_base_conj = _quat_conjugate(q_base)
        p_rel = _quat_rotate(q_base_conj, p_tool - p_base)
        q_rel = _quat_multiply(q_base_conj, q_tool)

        if self.config.project_distance_to_goal:
            goal_q_conj = _quat_conjugate(goal_q)
            position_delta = _quat_rotate(goal_q_conj, p_rel - goal_p)
            quat_delta = _quat_multiply(goal_q_conj, q_rel)
        else:
            position_delta = p_rel - goal_p
            quat_delta = _quat_multiply(q_rel, _quat_conjugate(goal_q))

        position_weight = self._weight[0]
        rotation_weight = self._weight[1]

        weighted_delta = position_delta * self._axes_weight[..., :3]
        position_cost = 0.5 * position_weight * (weighted_delta**2).sum(dim=-1)

        q_xyz = quat_delta[..., 1:4] * self._axes_weight[..., 3:6]
        vec_length = torch.sqrt((q_xyz**2).sum(dim=-1) + 1.0e-30)
        angle = 2.0 * torch.atan2(vec_length, torch.abs(quat_delta[..., 0]))
        rotation_cost = rotation_weight * angle**2

        zero = torch.zeros_like(position_cost)
        position_cost = torch.where(
            position_cost < self._tolerance_sq[..., 0], zero, position_cost
        )
        rotation_cost = torch.where(
            rotation_cost < self._tolerance_sq[..., 1], zero, rotation_cost
        )

        cost = torch.stack([position_cost, rotation_cost], dim=-1)
        with torch.no_grad():
            position_error = torch.sqrt(
                torch.clamp(2.0 * position_cost / torch.clamp(position_weight, min=1e-12), min=0)
            ).unsqueeze(-1)
            rotation_error = angle.unsqueeze(-1)
        return cost, position_error, rotation_error
