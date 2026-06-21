# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-form point-to-point execution-time cost (joint space).

Differentiable surrogate of the synchronized time-optimal PTP duration from the
current configuration ``q0`` to a candidate target ``q`` under box velocity and
acceleration limits with zero boundary velocity:

    d_j   = |q_j - q0_j|
    t_j   = 2*sqrt(d_j / a_j)         if d_j <  v_j^2 / a_j   (triangular)
          = d_j / v_j + v_j / a_j     if d_j >= v_j^2 / a_j   (trapezoidal)
    T     = max_j t_j                 (synchronized: all joints finish together)

The non-smooth ``max`` is replaced by ``logsumexp`` for a differentiable
surrogate ``T_beta = (1/beta) log sum_j exp(beta t_j)`` with
``T <= T_beta <= T + log(n)/beta``; the triangular branch's sqrt cusp at d=0 is
regularized by a floor ``d_floor`` so the gradient stays bounded.

Pure joint-space: reads joint positions directly, so the gradient flows through
the joint positions to the trajectory parameters with NO forward-kinematics
backward and NO quaternion-gradient pitfall (unlike pose costs).

An optional learned residual ``T_residual(q0, q)`` can be added for the
collision-induced excess time that has no closed form (see set_residual_model).
"""
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Callable, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_exec_time_cfg import ExecTimeCostCfg


class ExecTimeCost(BaseCost):
    """Closed-form differentiable PTP execution-time cost for redundancy resolution."""

    def __init__(self, config: ExecTimeCostCfg):
        self.config: ExecTimeCostCfg = config
        super().__init__(config)
        if config.vmax is None or config.amax is None:
            log_and_raise(
                "ExecTimeCost: vmax/amax not set. Pass them in the cfg or call "
                "cfg.initialize_from_transition_model(...)."
            )
        device = self.device_cfg.device
        self._vmax = config.vmax.to(device).view(1, 1, -1)
        self._amax = config.amax.to(device).view(1, 1, -1)
        self._d_switch = self._vmax**2 / self._amax  # triangular/trapezoidal boundary
        self._d_floor = float(config.d_floor)
        self._beta = float(config.beta)
        self._dof = config.vmax.numel()
        self._use_closed_form = bool(config.closed_form)
        self._residual_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def set_closed_form(self, flag: bool):
        """Toggle the analytic closed-form term at runtime. When False the cost
        is ``weight * residual_model(q0, q)`` only (the learned model is the full
        objective); when True it is ``weight * (T_closed_form + residual)``."""
        self._use_closed_form = bool(flag)

    def set_residual_model(
        self, model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ):
        """Plug in a learned collision-excess residual.

        Args:
            model: callable ``(q0, q) -> dT`` where ``q0`` and ``q`` are
                ``(B, H, dof)`` joint positions and ``dT`` is ``(B, H, 1)``
                non-negative extra time (e.g. an MLP). The total cost becomes
                ``weight * (T_closed_form + dT)``. Pass None to disable.
        """
        self._residual_model = model

    def _ptp_time_per_joint(self, d: torch.Tensor) -> torch.Tensor:
        """Per-joint synchronized time t_j(d), shape (..., dof)."""
        # sqrt-cusp floor applied UNIFORMLY (both branches + the switch use the
        # same softened distance), so the trapezoid/triangle pieces still meet
        # C0/C1 at d = v^2/a; the floor only bounds the triangular gradient at
        # d -> 0. (Applying the floor to the triangular branch alone would put a
        # tiny C0 jump at the switch.)
        d = torch.sqrt(d * d + self._d_floor * self._d_floor)
        t_tri = 2.0 * torch.sqrt(d / self._amax)
        t_trap = d / self._vmax + self._vmax / self._amax
        return torch.where(d >= self._d_switch, t_trap, t_tri)

    def time(self, q: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """Smooth execution-time surrogate T_beta(q0, q), shape (B, H)."""
        d = torch.abs(q - q0)
        t_j = self._ptp_time_per_joint(d)
        return torch.logsumexp(self._beta * t_j, dim=-1) / self._beta

    def _broadcast_start(
        self,
        q: torch.Tensor,
        current_joint_state: JointState,
        idxs_current_joint_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Bring the start config q0 to (B, H, dof) aligned with q (B, H, dof)."""
        b, h, dof = q.shape
        q0 = current_joint_state.position
        if q0.ndim == 3:
            q0 = q0[:, 0, :]
        q0 = q0.view(-1, dof)
        if idxs_current_joint_state is not None:
            q0 = q0[idxs_current_joint_state.view(-1).long()]
        if q0.shape[0] == 1:
            q0 = q0.expand(b, dof)
        elif q0.shape[0] != b:
            log_and_raise(
                f"ExecTimeCost: start config batch {q0.shape[0]} does not match "
                f"state batch {b}; pass idxs_current_joint_state."
            )
        return q0.view(b, 1, dof)

    def forward(
        self,
        joint_state: JointState,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the execution-time cost.

        Args:
            joint_state: candidate trajectory/target, position (B, H, dof).
            current_joint_state: the start config q0 (per problem). If None, the
                cost is zero (no reference to move from, e.g. pure batch IK
                without a start state).
            idxs_current_joint_state: (B,) mapping each batch row to its
                problem's start config row.

        Returns:
            cost of shape (B, H, 1).
        """
        q = joint_state.position
        if q.ndim != 3:
            log_and_raise("ExecTimeCost: joint_state.position must be (B, H, dof)")
        b, h, dof = q.shape
        if current_joint_state is None:
            return torch.zeros((b, h, 1), device=self.device_cfg.device, dtype=q.dtype)

        q0 = self._broadcast_start(q, current_joint_state, idxs_current_joint_state)
        if self._use_closed_form:
            cost = self.time(q, q0)  # (B, H)
        else:
            # pure-learned objective: closed-form term dropped, residual is all.
            cost = torch.zeros((b, h), device=self.device_cfg.device, dtype=q.dtype)
        if self._residual_model is not None:
            # q0 is (b, 1, dof); expand+contiguous so a residual MLP that
            # flattens with .view(b*h, dof) does not hit a stride-0 tensor.
            dt = self._residual_model(q0.expand(b, h, dof).contiguous(), q).reshape(b, h)
            cost = cost + dt
        cost = self._weight * cost
        return cost.unsqueeze(-1)
