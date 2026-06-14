# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused relative tool pose cost kernel.

Computes the pose error between the relative transform of two tool frames
(``T_rel = T_base^-1 * T_tool``) and a goal relative pose, with analytic
gradients written for BOTH tool frames in one kernel launch. The gradients
ride the standard multi-tool-frame FK backward, which accumulates them into
joint gradients for both arms.

Gradient math (world-frame left perturbations, ``D_p = p_tool - p_base``,
``g_p``/``g_w`` are the error gradients expressed in the base-tool frame):

    grad_p_tool  =  R_base @ g_p
    grad_p_base  = -R_base @ g_p
    grad_w_tool  =  R_base @ g_w
    grad_w_base  = -R_base @ g_w - D_p x (R_base @ g_p)

The moment-arm term in ``grad_w_base`` accounts for the dependence of the
relative position on the base frame's orientation. The total wrench over both
frames is zero, reflecting invariance of the cost to a rigid motion of both
frames together.
"""
from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.wp_tool_pose import (
    compute_position_error,
    compute_rotation_error,
    convert_angular_velocity_to_quaternion_rate,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, warp_kernel


def create_relative_pose_distance_kernel_with_constants(rotation_method: int = 0):
    """Create the fused relative pose distance kernel.

    Args:
        rotation_method: Method for rotation error computation:
            0 = axis-angle, 1 = lie group, 2 = lie group advanced.
    """

    def _relative_pose_distance_template(
        current_position: wp.array(dtype=wp.vec3),  # [batch * horizon * num_links]
        current_quat: wp.array(dtype=wp.vec4),  # [batch * horizon * num_links], wxyz
        goal_position: wp.array(dtype=wp.vec3),  # [num_goals]
        goal_quat: wp.array(dtype=wp.vec4),  # [num_goals], wxyz
        idxs_goal: wp.array(dtype=wp.int32),  # [batch]
        position_orientation_weight: wp.array(dtype=wp.float32),  # [2]
        terminal_pose_axes_weight_factor: wp.array(dtype=wp.float32),  # [6]
        non_terminal_pose_axes_weight_factor: wp.array(dtype=wp.float32),  # [6]
        terminal_pose_convergence_tolerance: wp.array(dtype=wp.float32),  # [2]
        non_terminal_pose_convergence_tolerance: wp.array(dtype=wp.float32),  # [2]
        project_distance_to_goal: wp.array(dtype=wp.uint8),  # [1]
        out_distance: wp.array(dtype=wp.float32),  # [batch * horizon * 2]
        out_position_distance: wp.array(dtype=wp.float32),  # [batch * horizon]
        out_rotation_distance: wp.array(dtype=wp.float32),  # [batch * horizon]
        out_position_gradient: wp.array(dtype=wp.vec3),  # [batch * horizon * num_links]
        out_rotation_gradient: wp.array(dtype=wp.vec4),  # [batch * horizon * num_links], wxyz
        batch_size: wp.int32,
        horizon: wp.int32,
        num_links: wp.int32,
        link_idx_base: wp.int32,
        link_idx_tool: wp.int32,
    ):
        tid = wp.tid()  # one thread per (batch, horizon)
        if tid >= batch_size * horizon:
            return
        b_idx = tid / horizon
        h_idx = tid - b_idx * horizon

        position_axes_weight = wp.vec3(0.0, 0.0, 0.0)
        rotation_axes_weight = wp.vec3(0.0, 0.0, 0.0)
        convergence_tolerance = wp.vec2(0.0, 0.0)
        if h_idx < (horizon - 1) and horizon > 1:
            position_axes_weight = wp.vec3(
                non_terminal_pose_axes_weight_factor[0],
                non_terminal_pose_axes_weight_factor[1],
                non_terminal_pose_axes_weight_factor[2],
            )
            rotation_axes_weight = wp.vec3(
                non_terminal_pose_axes_weight_factor[3],
                non_terminal_pose_axes_weight_factor[4],
                non_terminal_pose_axes_weight_factor[5],
            )
            convergence_tolerance = wp.vec2(
                non_terminal_pose_convergence_tolerance[0],
                non_terminal_pose_convergence_tolerance[1],
            )
        else:
            position_axes_weight = wp.vec3(
                terminal_pose_axes_weight_factor[0],
                terminal_pose_axes_weight_factor[1],
                terminal_pose_axes_weight_factor[2],
            )
            rotation_axes_weight = wp.vec3(
                terminal_pose_axes_weight_factor[3],
                terminal_pose_axes_weight_factor[4],
                terminal_pose_axes_weight_factor[5],
            )
            convergence_tolerance = wp.vec2(
                terminal_pose_convergence_tolerance[0],
                terminal_pose_convergence_tolerance[1],
            )

        position_weight = position_orientation_weight[0]
        rotation_weight = position_orientation_weight[1]
        convergence_tolerance[0] = convergence_tolerance[0] ** 2.0
        convergence_tolerance[1] = convergence_tolerance[1] ** 2.0

        # read the two tool frame poses (global memory is wxyz, wp.quat is xyzw)
        pose_addr = b_idx * horizon * num_links + h_idx * num_links
        p_base = current_position[pose_addr + link_idx_base]
        q_base_w = current_quat[pose_addr + link_idx_base]
        q_base = wp.quaternion(q_base_w[1], q_base_w[2], q_base_w[3], q_base_w[0])
        p_tool = current_position[pose_addr + link_idx_tool]
        q_tool_w = current_quat[pose_addr + link_idx_tool]
        q_tool = wp.quaternion(q_tool_w[1], q_tool_w[2], q_tool_w[3], q_tool_w[0])

        # relative pose: T_rel = T_base^-1 * T_tool (expressed in base-tool frame)
        t_base = wp.transform(p_base, q_base)
        t_tool = wp.transform(p_tool, q_tool)
        t_rel = wp.transform_multiply(wp.transform_inverse(t_base), t_tool)
        p_rel = wp.transform_get_translation(t_rel)
        q_rel = wp.transform_get_rotation(t_rel)

        # read goal relative pose
        goal_idx = idxs_goal[b_idx]
        g_position = goal_position[goal_idx]
        g_quat_w = goal_quat[goal_idx]
        g_quaternion = wp.quaternion(g_quat_w[1], g_quat_w[2], g_quat_w[3], g_quat_w[0])

        current_position_in_frame = wp.vec3(0.0, 0.0, 0.0)
        current_quaternion_in_frame = wp.quat(0.0, 0.0, 0.0, 1.0)
        goal_position_in_frame = wp.vec3(0.0, 0.0, 0.0)
        goal_quaternion_in_frame = wp.quat(0.0, 0.0, 0.0, 1.0)
        g_transform = wp.transform(g_position, g_quaternion)

        local_project_distance_to_goal = project_distance_to_goal[0]
        if local_project_distance_to_goal == 1:
            # express the error in the goal frame (per-axis weights apply there)
            rel_in_g_frame = wp.transform_multiply(wp.transform_inverse(g_transform), t_rel)
            current_position_in_frame = wp.transform_get_translation(rel_in_g_frame)
            current_quaternion_in_frame = wp.transform_get_rotation(rel_in_g_frame)
            goal_position_in_frame = wp.vec3(0.0, 0.0, 0.0)
            goal_quaternion_in_frame = wp.quat(0.0, 0.0, 0.0, 1.0)
        else:
            current_position_in_frame = p_rel
            current_quaternion_in_frame = q_rel
            goal_position_in_frame = g_position
            goal_quaternion_in_frame = g_quaternion

        position_distance, position_gradient = compute_position_error(
            current_position_in_frame,
            goal_position_in_frame,
            position_axes_weight,
            position_weight,
            convergence_tolerance[0],
        )

        angular_distance, gradient_as_angular_velocity, angle = compute_rotation_error(
            current_quaternion_in_frame,
            goal_quaternion_in_frame,
            rotation_axes_weight,
            rotation_weight,
            convergence_tolerance[1],
            rotation_method,
        )

        # gradients are expressed in the frame the error was computed in; bring
        # them to the base-tool frame first
        if local_project_distance_to_goal == 1:
            position_gradient = wp.transform_vector(g_transform, position_gradient)
            gradient_as_angular_velocity = wp.transform_vector(
                g_transform, gradient_as_angular_velocity
            )

        # map base-tool-frame gradients to world-frame gradients on both links
        grad_p_world = wp.quat_rotate(q_base, position_gradient)
        grad_w_world = wp.quat_rotate(q_base, gradient_as_angular_velocity)
        delta_p_world = p_tool - p_base

        grad_p_tool = grad_p_world
        grad_p_base = -grad_p_world
        # The complete angular gradients are scaled by 2 to cancel the 0.5
        # applied by the FK backward's quaternion_gradient_to_angular_velocity
        # when it unpacks the q*(omega,0) quaternion-rate packing below; this
        # makes the joint gradient finite-difference exact (the upstream tool
        # pose cost omits this factor and produces half-scale rotation
        # gradients).
        grad_w_tool = 2.0 * grad_w_world
        grad_w_base = 2.0 * (-grad_w_world - wp.cross(delta_p_world, grad_p_world))

        # convert angular gradients to quaternion-rate (same convention as
        # the standard tool pose cost), packed wxyz
        quat_rate_tool = convert_angular_velocity_to_quaternion_rate(grad_w_tool, q_tool)
        quat_rate_base = convert_angular_velocity_to_quaternion_rate(grad_w_base, q_base)

        # weight-independent geometric distances for convergence checks
        geometric_position_distance = (
            wp.sqrt(2.0 * position_distance / position_weight) if position_weight > 0.0 else 0.0
        )

        out_distance[2 * (b_idx * horizon + h_idx)] = position_distance
        out_distance[2 * (b_idx * horizon + h_idx) + 1] = angular_distance
        out_position_distance[b_idx * horizon + h_idx] = geometric_position_distance
        out_rotation_distance[b_idx * horizon + h_idx] = angle

        out_position_gradient[pose_addr + link_idx_tool] = grad_p_tool
        out_position_gradient[pose_addr + link_idx_base] = grad_p_base
        out_rotation_gradient[pose_addr + link_idx_tool] = wp.vec4(
            quat_rate_tool[3], quat_rate_tool[0], quat_rate_tool[1], quat_rate_tool[2]
        )
        out_rotation_gradient[pose_addr + link_idx_base] = wp.vec4(
            quat_rate_base[3], quat_rate_base[0], quat_rate_base[1], quat_rate_base[2]
        )

    kernel_name = f"relative_pose_distance_{rotation_method}"
    return warp_kernel(kernel_name)(_relative_pose_distance_template)


class RelativePoseDistance(torch.autograd.Function):
    """Autograd bridge for the fused relative pose cost kernel.

    The analytic gradients for both tool frames are computed in the forward
    kernel launch; backward only hands back the cached buffers, exactly like
    :class:`curobo._src.cost.wp_tool_pose.ToolPoseDistance`.
    """

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,  # (b, h, num_links, 3)
        current_quat: torch.Tensor,  # (b, h, num_links, 4) wxyz
        goal_position: torch.Tensor,  # (num_goals, 3)
        goal_quat: torch.Tensor,  # (num_goals, 4) wxyz
        idxs_goal: torch.Tensor,  # (b, 1) int32
        position_orientation_weight: torch.Tensor,  # (2,)
        terminal_pose_axes_weight_factor: torch.Tensor,  # (6,)
        non_terminal_pose_axes_weight_factor: torch.Tensor,  # (6,)
        terminal_pose_convergence_tolerance: torch.Tensor,  # (2,)
        non_terminal_pose_convergence_tolerance: torch.Tensor,  # (2,)
        project_distance_to_goal: torch.Tensor,  # (1,) uint8
        out_distance: torch.Tensor,  # (b, h, 2)
        out_position_distance: torch.Tensor,  # (b, h, 1)
        out_rotation_distance: torch.Tensor,  # (b, h, 1)
        out_position_gradient: torch.Tensor,  # (b, h, num_links, 3)
        out_rotation_gradient: torch.Tensor,  # (b, h, num_links, 4)
        link_idx_base: int,
        link_idx_tool: int,
        use_grad_input: bool,
        warp_kernel,
    ):
        ctx.set_materialize_grads(False)
        if current_position.ndim != 4:
            log_and_raise("current_position must be a 4D tensor (b, h, num_links, 3)")
        if current_quat.ndim != 4:
            log_and_raise("current_quat must be a 4D tensor (b, h, num_links, 4)")
        b, h, num_links, _ = current_position.shape
        if goal_position.ndim != 2 or goal_position.shape[-1] != 3:
            log_and_raise("goal_position must have shape (num_goals, 3)")
        if goal_quat.ndim != 2 or goal_quat.shape[-1] != 4:
            log_and_raise("goal_quat must have shape (num_goals, 4)")
        if idxs_goal.shape != (b, 1):
            log_and_raise(f"idxs_goal must have shape ({b}, 1), got {idxs_goal.shape}")
        if out_distance.shape != (b, h, 2):
            log_and_raise("out_distance must have shape (b, h, 2)")
        if out_position_gradient.shape != (b, h, num_links, 3):
            log_and_raise("out_position_gradient must have shape (b, h, num_links, 3)")
        if out_rotation_gradient.shape != (b, h, num_links, 4):
            log_and_raise("out_rotation_gradient must have shape (b, h, num_links, 4)")

        ctx.use_grad_input = use_grad_input
        wp_device, wp_stream = get_warp_device_stream(current_position)

        wp.launch(
            kernel=warp_kernel,
            dim=b * h,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(goal_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(goal_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(idxs_goal.detach().view(-1), dtype=wp.int32),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(terminal_pose_axes_weight_factor.view(-1), dtype=wp.float32),
                wp.from_torch(non_terminal_pose_axes_weight_factor.view(-1), dtype=wp.float32),
                wp.from_torch(terminal_pose_convergence_tolerance.view(-1), dtype=wp.float32),
                wp.from_torch(
                    non_terminal_pose_convergence_tolerance.view(-1), dtype=wp.float32
                ),
                wp.from_torch(project_distance_to_goal.view(-1), dtype=wp.uint8),
                wp.from_torch(out_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_rotation_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                b,
                h,
                num_links,
                link_idx_base,
                link_idx_tool,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )

        ctx.mark_non_differentiable(
            out_position_distance,
            out_rotation_distance,
            goal_position,
            goal_quat,
            idxs_goal,
            position_orientation_weight,
            terminal_pose_axes_weight_factor,
            non_terminal_pose_axes_weight_factor,
            terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance,
            project_distance_to_goal,
        )
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)

        return out_distance, out_position_distance, out_rotation_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_distance: Optional[torch.Tensor],  # (b, h, 2)
        grad_position_distance: Optional[torch.Tensor],
        grad_rotation_distance: Optional[torch.Tensor],
    ):
        use_grad_input = ctx.use_grad_input
        pos_grad = None
        quat_grad = None
        if grad_distance is not None:
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                out_position_gradient, out_rotation_gradient = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                if use_grad_input:
                    grad_pos = grad_distance[:, :, 0:1].unsqueeze(-1)
                    pos_grad = out_position_gradient * grad_pos
                else:
                    pos_grad = out_position_gradient
            if ctx.needs_input_grad[1]:
                if use_grad_input:
                    grad_ori = grad_distance[:, :, 1:2].unsqueeze(-1)
                    quat_grad = out_rotation_gradient * grad_ori
                else:
                    quat_grad = out_rotation_gradient

        return (
            pos_grad,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
