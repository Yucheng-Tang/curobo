# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the relative tool pose cost (pose of one tool frame in another)."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class RelativePoseCostCfg(BaseCostCfg):
    """Configuration for the relative tool pose cost.

    Constrains the pose of ``tool_frame`` expressed in the frame of
    ``base_frame`` (``T_rel = T_base^-1 * T_tool``) to a goal relative pose.
    Both frames must be listed in the robot's ``tool_frames``.

    ``weight`` is a 2-vector ``[position_weight, orientation_weight]``.
    """

    #: Class type of the cost; set in __post_init__ (avoids circular import).
    class_type: Type = None

    #: Tool frame whose frame the relative pose is expressed in (e.g. the
    #: object-holding arm's TCP).
    base_frame: Optional[str] = None

    #: Tool frame whose pose is constrained relative to base_frame (e.g. the
    #: probe arm's TCP).
    tool_frame: Optional[str] = None

    #: Initial goal relative pose as [x, y, z, qw, qx, qy, qz]. Can be updated
    #: at runtime through RelativePoseCost.update_goal.
    goal_pose: Optional[Union[torch.Tensor, List[float]]] = None

    #: If true, rotation distance and gradient use the Lie group log map.
    use_lie_group: bool = False

    #: If true, the error is computed in the goal frame so per-axis weights
    #: apply there (e.g. free rotation about an axisymmetric probe axis).
    project_distance_to_goal: bool = False

    #: Per-axis weight factors [px, py, pz, rx, ry, rz] at the last timestep.
    terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Per-axis weight factors at all other timesteps. Unlike the absolute
    #: tool pose cost, a relative constraint usually holds over the whole
    #: horizon, so this defaults to ones as well.
    non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence dead-zone [position, rotation] at the last timestep.
    terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence dead-zone at all other timesteps.
    non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    def __post_init__(self):
        if self.class_type is None:
            from curobo._src.cost.cost_relative_pose import RelativePoseCost

            self.class_type = RelativePoseCost
        if self.base_frame is None or self.tool_frame is None:
            log_and_raise("RelativePoseCostCfg requires base_frame and tool_frame")
        if self.base_frame == self.tool_frame:
            log_and_raise("RelativePoseCostCfg: base_frame and tool_frame must differ")

        def _to_device(value, default, name, length):
            if value is None:
                value = default
            tensor = self.device_cfg.to_device(value)
            if tensor.shape != (length,):
                log_and_raise(f"RelativePoseCostCfg: {name} must have {length} elements")
            return tensor

        self.terminal_pose_axes_weight_factor = _to_device(
            self.terminal_pose_axes_weight_factor,
            [1.0] * 6,
            "terminal_pose_axes_weight_factor",
            6,
        )
        self.non_terminal_pose_axes_weight_factor = _to_device(
            self.non_terminal_pose_axes_weight_factor,
            [1.0] * 6,
            "non_terminal_pose_axes_weight_factor",
            6,
        )
        self.terminal_pose_convergence_tolerance = _to_device(
            self.terminal_pose_convergence_tolerance,
            [0.0, 0.0],
            "terminal_pose_convergence_tolerance",
            2,
        )
        self.non_terminal_pose_convergence_tolerance = _to_device(
            self.non_terminal_pose_convergence_tolerance,
            [0.0, 0.0],
            "non_terminal_pose_convergence_tolerance",
            2,
        )
        if self.goal_pose is not None:
            goal = self.device_cfg.to_device(self.goal_pose)
            if goal.shape != (7,):
                log_and_raise("RelativePoseCostCfg: goal_pose must be [x,y,z,qw,qx,qy,qz]")
            self.goal_pose = goal

        super().__post_init__()
        if self.weight.shape != (2,):
            log_and_raise(
                "RelativePoseCostCfg: weight must be [position_weight, orientation_weight]"
            )

    @property
    def rotation_method(self) -> int:
        return 1 if self.use_lie_group else 0
