# Upstream cuRobo v2 issues (found during relative-pose cost development)

Three independent issues found while adding a fused relative-TCP-pose cost to
cuRobo v2 (commit `e0b1030`, v0.8.0.post1.dev34). Each section below is written
so it can be filed verbatim as a GitHub issue at NVlabs/curobo. Issue 1 is the
load-bearing one; issues 2 and 3 are smaller correctness/reporting bugs.

All line numbers are against `curobo/_src/...` at the above commit.

---

## Issue 1 — FK backward computes a wrong VJP for arbitrary quaternion gradients

**Summary.** `KinematicsFusedFunction.backward` (the FK `torch.autograd.Function`)
does not implement the true vector–Jacobian product from a link quaternion to
joint angles. It applies a conversion
`omega = 0.5 * vec(q^{-1} ⊗ g_quat)` and then dots `omega` with **world-frame**
joint axes. That pairing is only self-consistent when the incoming gradient is
packed as `g_quat = q ⊗ (omega_world, 0)` — the exact packing the built-in Warp
cost kernels emit. For a gradient that is the *plain* `∂L/∂q` produced by
`torch.autograd` over a custom cost that differentiates a link quaternion, the
result is a different (generally wrong, sign-flipped) joint gradient. This
contradicts the custom-cost guide (`docs/guides/custom_cost.rst`), which tells
users that gradients of a `BaseCost` written in plain PyTorch "flow
automatically via PyTorch backpropagation."

**Where.**
- `curobo/_src/curobolib/kernels/common/quaternion_util.cuh:86-102`
  (`quaternion_gradient_to_angular_velocity`: `omega = 0.5 * (...)`)
- `curobo/_src/curobolib/kernels/kinematics/kinematics_backward_helper.cuh:140-158`
  (`compute_link_gradients`: converts the per-frame quaternion gradient to
  `omega`, then `xyz_rot_backward` dots it with world-frame axes)
- `curobo/_src/curobolib/cuda_ops/kinematics.py` (`KinematicsFusedFunction.backward`)

**Why it is wrong.** The backward consumes `grad_nlinks_quat` (a `dL/dq_link`,
wxyz) and produces `dL/dq_joint`. The correct VJP for a left-invariant
parameterization is `dL/dθ_j = <axis_j^world, ω>` with
`ω = 2 * vec(g_quat ⊗ q^{-1})` (a world-frame angular gradient). The code
instead computes `0.5 * vec(q^{-1} ⊗ g_quat)`, which is the body-frame adjoint
*of a different packing*. The 0.5 vs 2 and the left/right multiplication order
both differ; they happen to cancel exactly when `g_quat = q ⊗ (ω,0)` (the Warp
kernel convention), so all NVIDIA-authored costs are correct, but a raw
autograd gradient is not.

**Reproduction (isolates cuRobo from torch).** Same scalar cost
`C(p_rel, R_rel)` evaluated three ways, vs central finite differences of the
forward value, on `dual_ur10e` (13/12-DoF dual arm):

| path | max rel err vs FD |
|------|-------------------|
| pure-torch FK (no cuRobo code) + autograd | **8.0e-4** (correct) |
| cuRobo CUDA FK + raw autograd quaternion grad | **13.6** (wrong, sign flips) |
| cuRobo CUDA FK + `q⊗g⊗q^{-1}` adapter | **8.0e-4** (correct) |

Script: a self-contained ~120-line repro (builds a pure-torch FK from the URDF,
compares to `Kinematics.compute_kinematics` backward) is available on request;
the key point is the middle row — feeding cuRobo's FK backward a plain
`∂L/∂q_link` yields a joint gradient with the wrong direction, while a pure
torch FK of the identical cost matches FD. So torch autograd is correct; the
custom CUDA backward is not a true VJP for arbitrary quaternion gradients.

**Impact.** Any user-written `BaseCost` (the supported extension mechanism) that
backprops through a tool-frame *quaternion* gets silently wrong joint gradients
— wrong descent direction, not just wrong scale. Position-only costs are
unaffected (the position VJP is the exact geometric Jᵀ). NVIDIA's own costs are
unaffected (they emit the matching packing).

**Suggested fix.** Either (a) make `compute_link_gradients` implement the true
quaternion→omega VJP `ω = 2 * vec(g_quat ⊗ q^{-1})` so any autograd gradient is
handled, or (b) document explicitly that custom costs must hand the FK backward
gradients packed as `q ⊗ (ω_world, 0)` (i.e. emit angular-velocity-style
gradients, as the Warp kernels do), and provide a helper that converts a raw
`∂L/∂q` into that packing (`q ⊗ g ⊗ q^{-1}`).

---

## Issue 2 — `ToolPoseCost` rotation gradient is half the true gradient

**Summary.** The Warp pose kernel's rotation gradient is exactly 0.5× the true
gradient of the rotation cost it reports. The compensating factor is present in
the code but commented out.

**Where.** `curobo/_src/cost/wp_tool_pose.py:108-126`
(`convert_angular_velocity_to_quaternion_rate`): the line
`# quat_rate = 0.5 * quat_rate` is commented out, and downstream the FK backward
applies another 0.5 (Issue 1's conversion), so the net rotation gradient handed
to the optimizer is half of `∂(rotation_cost)/∂θ`.

**Reproduction.** Finite-difference the joint gradient of a pure rotation cost
(`weight=[0, w_rot]`) on any robot: `fd / analytic == 2.0` for the rotation
joints. (Position joints give `1.0`.)

**Impact.** Low in practice — the position/rotation weights are hand-tuned and
L-BFGS rescales the step — but it means the effective `orientation` weight is
half of what the config says, and it breaks `torch.autograd.gradcheck` against
the reported cost. A custom cost that mixes this kernel's gradient with an
analytically-correct term will be mis-weighted.

**Suggested fix.** Restore the factor (uncomment the `0.5`) or, equivalently,
fold a factor 2 into the documented gradient convention and note it. Our
relative-pose kernel includes the factor and is finite-difference exact.

---

## Issue 3 — `IKSolverResult.position_error` / `rotation_error` mis-paired for multi-link or multi-group convergence

**Summary.** `solver_ik._get_result` recovers per-link `(position, orientation)`
pairs by collecting every convergence metric whose name contains
`position_tolerance`/`orientation_tolerance` in registration order,
concatenating on the last axis, then reshaping to `(num_links, 2)` with
`num_links = total_cols / 2`. This interleave is only correct for a **single
1-link pose group**. With a multi-link `tool_pose` (e.g. `dual_ur10e`
`tool_frames=[tool1, tool0]`, so 2 position columns + 2 orientation columns) the
reshape pairs `(pos_l0, pos_l1)` and `(ori_l0, ori_l1)` as if they were
`(pos, ori)` pairs, so the reported `position_error`/`rotation_error` mix
position and orientation values.

**Where.** `curobo/_src/solver/solver_ik.py:461-472` (collection by substring,
in add order), `:544-557` (`num_links = cols/2`, `.view(batch, seeds,
num_links, 2)`, `max` over column 0/1).

**Reproduction.** Solve IK on `dual_ur10e` with both `tool_frames` goaled
(num_links=2). `IKSolverResult.position_error` becomes
`max(pos_l0, ori_l0, ...)` rather than the true max position error.

**Impact.** Reporting only — solver **success/convergence is unaffected**
(each named metric is thresholded *before* the reshape, in `converge_list`), and
seed ranking uses an order-insensitive sum. But any consumer reading
`IKSolverResult.position_error/rotation_error` for a multi-link problem gets
scrambled values. This is a pre-existing bug independent of any custom cost
(multi-link `tool_pose` alone triggers it); adding a second pose-cost group
(e.g. a relative-pose cost) makes it concretely wrong on the shipped dual-arm
config.

**Suggested fix.** Reshape each named metric to `(..., k_metric, 2)` (or keep a
per-metric `(position, orientation)` channel) before concatenating, instead of
deriving `num_links` from the total column count and assuming a global
`[pos..., ori...]` layout.

---

*Filed from work in `relpose_experiments/` (a fused relative-TCP-pose cost for
dual-arm scanning). The relative-pose cost itself works around Issue 1 with a
`q⊗g⊗q^{-1}` adapter for its pure-torch reference path and emits the Warp-native
packing (with the Issue-2 factor 2) for its analytic kernel.*
