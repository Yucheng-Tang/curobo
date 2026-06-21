# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the closed-form point-to-point execution-time cost.

Implements a differentiable surrogate of the synchronized time-optimal PTP
duration (zero boundary velocity, box velocity/acceleration limits) used for
redundancy resolution toward fast-to-execute joint configurations. A
mathematically grounded alternative to the learned execution-time MLP of ETA-IK
(arXiv:2411.14381): on collision-free PTP the closed form matches TOPPRA at
Spearman ~0.999 across 12/13/14-DoF robots, so no network is needed for that
term. An optional learned residual can be plugged in for the collision-induced
excess time (see ExecTimeCost.set_residual_model).
"""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.transition.robot_state_transition import RobotStateTransition


@dataclass
class ExecTimeCostCfg(BaseCostCfg):
    """Configuration for the closed-form execution-time cost.

    ``weight`` is a scalar (1-vector). Per-joint ``vmax``/``amax`` may be given
    explicitly; otherwise they are filled from the robot's joint limits via
    :meth:`initialize_from_transition_model`.
    """

    #: Concrete cost class; resolved in __post_init__ to avoid a circular import.
    class_type: Type = None

    #: logsumexp sharpness for the smooth max over joints. Larger -> closer to
    #: the true max, sharper bottleneck-joint selection. T_an <= T_beta <=
    #: T_an + log(n_joints)/beta.
    beta: float = 8.0

    #: Floor (rad) on the triangular-branch distance inside the sqrt, to bound
    #: the otherwise-unbounded gradient 1/sqrt(a*d) as d -> 0.
    d_floor: float = 1.0e-3

    #: Per-joint velocity limits (rad/s). None -> filled from robot joint limits.
    vmax: Optional[Union[torch.Tensor, List[float]]] = None

    #: Per-joint acceleration limits (rad/s^2). None -> filled from robot limits.
    amax: Optional[Union[torch.Tensor, List[float]]] = None

    #: When True (default) the cost is ``weight*(T_closed_form + residual)``.
    #: When False the closed-form term is dropped and the cost is
    #: ``weight*residual`` — i.e. a learned model (set via set_residual_model)
    #: is used as the FULL execution-time objective rather than a correction on
    #: top of the analytic surrogate. Used to compare a pure-learned ETA-IK MLP
    #: cost against the closed form. vmax/amax are still required (the C0 floor /
    #: switch buffers are built regardless) but are unused when False.
    closed_form: bool = True

    def __post_init__(self):
        if self.class_type is None:
            from curobo._src.cost.cost_exec_time import ExecTimeCost

            self.class_type = ExecTimeCost
        if self.vmax is not None:
            self.vmax = self.device_cfg.to_device(self.vmax).view(-1)
            if not torch.all(torch.isfinite(self.vmax)) or torch.any(self.vmax <= 0):
                log_and_raise("ExecTimeCostCfg: vmax must be finite and strictly positive")
        if self.amax is not None:
            self.amax = self.device_cfg.to_device(self.amax).view(-1)
            if not torch.all(torch.isfinite(self.amax)) or torch.any(self.amax <= 0):
                log_and_raise("ExecTimeCostCfg: amax must be finite and strictly positive")
        super().__post_init__()
        if self.weight.numel() != 1:
            log_and_raise("ExecTimeCostCfg: weight must be a scalar (1 element)")

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        """Fill vmax/amax from the robot's joint limits if not given explicitly."""
        bounds = transition_model.get_state_bounds()
        dof = transition_model.action_dim
        if self.vmax is None:
            self.vmax = bounds.velocity[1].clone().to(self.device_cfg.device)
        if self.amax is None:
            self.amax = bounds.acceleration[1].clone().to(self.device_cfg.device)
        if self.vmax.numel() != dof or self.amax.numel() != dof:
            log_and_raise(
                f"ExecTimeCostCfg: vmax/amax must have {dof} entries, got "
                f"{self.vmax.numel()}/{self.amax.numel()}"
            )
        if torch.any(self.vmax <= 0) or torch.any(self.amax <= 0):
            log_and_raise("ExecTimeCostCfg: vmax/amax must be strictly positive")

    def clone(self):
        return ExecTimeCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
            beta=self.beta,
            d_floor=self.d_floor,
            vmax=self.vmax.clone() if isinstance(self.vmax, torch.Tensor) else self.vmax,
            amax=self.amax.clone() if isinstance(self.amax, torch.Tensor) else self.amax,
            closed_form=self.closed_form,
        )
