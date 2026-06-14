# Relative TCP Pose in cuRobo v2

Adds a relative tool pose cost (pose of `tool_frame` expressed in `base_frame`,
`T_rel = T_base^-1 * T_tool`) to cuRobo v2, with analytic gradients for BOTH
arms fused into one Warp kernel launch. Developed for the dekon_scan dual-arm
scanning project (arm A holds the object, arm B holds the probe; the task
variable is the probe pose in the object frame).

## Why a single long chain (tool1 as base) used to be much faster

It is NOT forward kinematics: cuRobo's FK composes the cumulative transforms
in one sequential loop over links in topological order (v1
`kinematics_fused_kernel.cu`, v2 `kinematics_forward_helper.cuh:488`), so a
12-joint serial chain and a 2x6 tree cost the same. The gap is entirely on the
cost side of the v1 fork's two-chain implementation:

1. The pytorch3d composition (`quaternion_to_matrix` x2, matmuls,
   `matrix_to_quaternion`) costs ~250 extra CUDA kernel launches per cost
   evaluation forward+backward (measured: 614 vs 10 kernels/iter).
2. `pytorch3d.transforms.matrix_to_quaternion` does boolean-mask indexing ->
   `nonzero()` -> implicit device-host sync per evaluation, which also breaks
   CUDA graph capture for the whole optimizer loop.
3. The goal-side relative pose was recomputed every cost call although it is
   constant during a solve.
4. The raw autograd quaternion gradients fed to the FK backward are converted
   with the wrong frame convention (see below), so the descent directions were
   also subtly wrong.

With the long chain, the relative pose IS the FK output, so the stock fused
pose cost kernel (2 launches, analytic gradients, graph-capturable) applies.

## How the two-chain version gets the same speed

`RelativePoseCost` (Warp kernel `wp_relative_pose.py`): one thread per
(batch, horizon) reads both TCP poses from the FK output, composes
`T_rel`, evaluates the same position/rotation error functions as the stock
`ToolPoseCost`, and writes analytic world-frame gradients for BOTH links:

    grad_p_tool =  R_base g_p          grad_w_tool =  R_base g_w
    grad_p_base = -R_base g_p          grad_w_base = -R_base g_w - Dp x (R_base g_p)

The FK backward natively accumulates multi-tool-frame gradients into joint
gradients for both arms, so nothing else changes.

## Gradient convention findings (important for ANY custom v2 cost)

- The FK backward (`quaternion_gradient_to_angular_velocity`) computes
  `omega = 0.5 * vec(q^-1 x g)` and dots it with world-frame joint axes. This
  is only frame-consistent for gradients packed as `g = q x (omega_world, 0)`
  (what the Warp cost kernels emit).
- The upstream `ToolPoseCost` omits a factor 2 in that packing: its rotation
  gradients are exactly HALF the true gradient (verified by finite
  differences; `wp_tool_pose.py:125` has the compensating `0.5*` commented
  out). Our kernel includes the factor and is finite-difference exact.
- Pure-torch custom costs that differentiate through tool QUATERNIONS get
  silently wrong joint gradients (frame mismatch, not just scale). Wrap the
  quaternions with `_FKQuaternionGradAdapter` (sandwich `q x g x q*` in
  backward) — see `TorchRelativePoseCost`. Position-only costs (e.g.
  sphere-vs-SDF) are unaffected: position gradients are exact geometric J^T.

## Files

Implementation (in `curobo/_src/`):
- `cost/wp_relative_pose.py` — fused Warp kernel + autograd Function
- `cost/cost_relative_pose.py` — `RelativePoseCost` (Warp), `TorchRelativePoseCost`
  (autograd reference), `_FKQuaternionGradAdapter`
- `cost/cost_relative_pose_cfg.py` — config (`base_frame`, `tool_frame`,
  `goal_pose [x,y,z,qw,qx,qy,qz]`, per-axis weights, `project_distance_to_goal`)
- `rollout/cost_manager/cost_manager_robot{,_cfg}.py` — registered as
  `relative_pose_cfg` in cost/constraint/convergence channels; runtime goal
  update via `cost_manager.update_params(relative_pose_goal=Pose(...))`;
  convergence emits `relative_pose_{position,orientation}_tolerance` which
  gates IK/trajopt success automatically.

Long-chain baseline:
- `reroot_urdf.py` — re-roots dual_ur10e.urdf at tool1 (frame-preserving dummy
  link construction; FK verified to 1.3e-15 against `inv(T_tool1) @ T_tool0`)
- generates `dual_ur10e_rerooted_tool1.urdf` + `dual_ur10e_rerooted.yml`

Experiments (run inside the `neural-sdf-v2` container,
`cd ~/ws/neural_sdf/curobo_v2`):
- `validate_relative_pose.py` — forward/gradient/finite-difference validation
- `debug_gradients.py` — pose-level + upstream-convention decomposition
- `prove_fk_backward_issue.py` — isolates the FK-backward convention issue:
  the same cost evaluated through a pure-torch FK (no cuRobo code) matches
  finite differences exactly, while cuRobo's CUDA FK backward fed raw autograd
  quaternion gradients is wrong (sign flips); with the adapter both paths
  agree digit-for-digit. Proves the issue is cuRobo's custom backward, not
  torch autograd.
- `bench_relative_pose.py` — per-iteration cost+grad benchmark
- `demo_ik_relative.py` — end-to-end IK with the relative cost via config dicts

## Results (RTX 5090, eager mode, FK+cost+backward per iteration)

| variant                      | (32,1)   | (512,1)  | (32,16)  | kernels |
|------------------------------|----------|----------|----------|---------|
| abs ToolPoseCost (floor)     | 0.599 ms | 0.576 ms | 0.582 ms | 10      |
| long chain + ToolPoseCost    | 0.603 ms | 0.640 ms | 0.596 ms | 10      |
| tree + fused RelativePoseCost| 0.758 ms | 0.607 ms | 0.628 ms | 10      |
| tree + torch quaternion      | 3.08 ms  | 4.05 ms  | 4.04 ms  | 442     |
| tree + pytorch3d (v1 fork)   | 4.89 ms  | 5.41 ms  | 5.39 ms  | 614     |

End-to-end IK (dual_ur10e, 32 seeds, CUDA graphs on): tree+relative 2.13 ms,
long chain 2.01 ms, stock absolute 2.25 ms — parity.

Validation: warp vs torch joint gradients agree to 2e-7; both finite-
difference exact; cost is zero at the goal configuration; both
`project_distance_to_goal` modes pass.

## Long chain vs two-chain: when to still use the long chain

The long chain remains a neat trick when the ONLY task constraint is the
relative pose (the goal becomes a plain absolute goal, so even the LM seed IK
stage works on it directly). But it expresses the world in the moving tool1
frame, which breaks scene collision against static obstacles, makes inverse
dynamics physically meaningless, and needs a re-rooted URDF. The fused
relative cost keeps the normal world-frame model (collision, dynamics, other
costs intact) at the same per-iteration price.

## Known limitations / caveats

Found in an adversarial review; the ones in OUR code are fixed, the ones in
upstream cuRobo are documented here (not patched, to avoid diverging from
upstream).

Fixed in this implementation:
- `TorchRelativePoseCost` raises if `use_lie_group=True` (it only implements
  axis-angle); previously it silently diverged ~4x from the Warp kernel,
  invalidating it as a numerical reference. The shipped solver always uses the
  Warp `RelativePoseCost`, which supports both.
- Both cost classes now allocate `_goalset_idx_zero` (the convergence path
  needs it); previously the Torch class would `AttributeError` there.
- `update_goal` enforces a single relative goal (in-place copy, CUDA-graph
  safe) and raises on batched goals. Batched per-problem relative goals are
  not supported: the cost manager forwards `idxs_link_pose`, which indexes the
  tool_pose goalset, not a relative-goal set. Forward always uses a zero index,
  so a single goal is correct; multiple would mis-map.
- A `log_warn` fires if the cost is constructed without a goal_pose (identity
  default silently pulls the two frames together).

Upstream cuRobo behaviour to be aware of (NOT changed here):
- `solver_ik._get_result` recovers per-link (position, orientation) pairs by
  reshaping all `*_tolerance` convergence columns to `(num_links, 2)`. That
  layout is only correct for a single 1-link pose group. With a multi-link
  tool_pose (e.g. dual_ur10e tool_frames=[tool1,tool0]) OR tool_pose +
  relative_pose together, the REPORTED `IKSolverResult.position_error/
  rotation_error` are mis-paired (mix position and orientation columns). This
  is a PRE-EXISTING upstream bug (already wrong for multi-link tool_pose alone)
  and affects only the reported scalars — NOT solver success/convergence
  (those are computed per-named-metric before the reshape) and NOT planning.
  For a trustworthy relative-pose error, recompute via FK of the solution as
  `demo_ik_relative.py` does, instead of reading the result fields.
- Convergence success ANDs every `*_position_tolerance`/`*_orientation_tolerance`
  metric. If `relative_pose_cfg` is added to `convergence_cfg` while the
  default `tool_pose_cfg` stays (as `metrics_base.yml` ships it), IK success
  requires BOTH the absolute tool poses AND the relative pose to converge,
  under the SAME `position_tolerance` (0.005 m) / `orientation_tolerance`
  (0.05 rad). For a relative-ONLY task, set `tool_pose_cfg: null` in the
  metrics/convergence config (and supply a consistent absolute goal if you
  keep it). Relative-pose errors are compared against the absolute tolerances;
  there is no separate relative tolerance knob.
