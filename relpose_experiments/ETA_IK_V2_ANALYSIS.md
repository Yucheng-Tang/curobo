# ETA-IK in cuRobo v2: assessment and more-correct alternatives

Assessment of whether a learned execution-time MLP (ETA-IK, arXiv:2411.14381)
helps relative-pose IK in cuRobo v2, and what is mathematically more correct.
Grounded in an in-container measurement and an adversarially-verified analysis.

## The empirical anchor

`eta_analytic_vs_toppra.py` (run in the `neural-sdf` v1 container — TOPPRA needs
numpy<2), 13-DoF dual arm, 400 random PTP pairs, paper limits:

| surrogate for collision-free PTP time | mean rel-err | median | max | Spearman vs TOPPRA |
|---|---|---|---|---|
| analytic `T = max_j t_j(d_j)` | 0.19% | 0.00% | 6.59% | **0.9989** |
| smooth `T_β = (1/β)logΣ exp(β t_j)`, β=8 | 1.04% | 0.54% | 7.07% | 0.9940 |

→ For the collision-free (TOPPRA) execution-time target — one of ETA-IK's two
data generators, and the one where ~50% of its trajectories are collision-free
anyway — the time is a **near-closed-form** function of `(q0, q)`. The MLP is
approximating something already analytic.

## Disambiguation: "faster solution" has two meanings

1. **Faster motion** (shorter execution time). ETA-IK's real contribution. But
   in v2 the exec-time IK cost does **not** time the motion — the B-spline
   trajopt already minimizes execution time by `dt` re-timing under vel/acc/jerk
   limits. The re-timing rule (`util/trajectory.py:234-258`,
   `calculate_dt_no_clamp`) is itself a synchronized max-over-joints:
   `dt ∝ max( max_j |v_j|/v̄_j , (max_j |a_j|/ā_j)^{1/2} , (max_j |j_j|/j̄_j)^{1/3} )`.
   So for a *fixed path* timing is already optimal; the IK cost can only steer
   the solver to a **different redundant branch / target config** whose
   time-optimal motion is shorter. ETA-IK's ~25% gain is **branch selection**,
   not better timing. Phrase it that way.

2. **Faster solve** (optimizer wall-clock / throughput). The exec-time cost does
   NOT help — a cost *adds* an objective, it cannot remove iterations. The
   "compute constant from 20→2000 seeds" result is GPU batching, not the cost
   being free. What cuts iterations / `num_seeds` is a **seeder**, not a cost.

The MLP retains a defensible role **only** for the collision-aware
(cuRobo-TrajOpt) time target, where obstacles reshape the motion and `T` is not
closed-form. Even there, v2 lets you run the real dynamics-aware trajopt on a
small goalset and read the true duration, which dominates a regression.

## Recommended approaches, priority order

**P1 — Low-time seed distribution (biggest lever).** Because IK only selects the
redundant branch, inject the time objective at the *seed*, not as an in-loop
cost. Options: a learned diffusion seeder (DiffusionSeeder, arXiv:2410.16727 —
ETA-IK's own ref [31]; integrated with cuRobo, reports 12–36× planning speedup),
point-cloud-conditioned flow-matching warm-start (arXiv:2510.03460), or even a
cheap **analytic-time-sorted** seed set. A seeder and an exec-time cost are NOT
orthogonal — both push toward the low-PTP-time branch — so a good seeder largely
**subsumes** the cost. Plumbing is free: `seed_ik_solver.py:485-509` already
accepts an external `seed_config` and pads with Halton; no solver change.

**P2 — Pre-screen-then-trajopt (v2-native, no learning).** Generate redundant IK
targets → score each with the closed-form `T_an = max_j t_j` (sub-0.1 ms) →
run the expensive dynamics-aware trajopt only on the top-k (a small goalset) →
keep min-time. For the collision-aware objective this returns the *true*
re-timed duration and beats any regressed MLP. v2 batches trajopt on GPU.

**P3 — If an in-loop cost is still wanted, make it closed-form, not an MLP.**
Per-joint synchronized time, zero boundary velocity, box vel/acc (C0/C1 at the
switch):
```
t_j(d_j) = 2·sqrt(d_j/a_j)        if d_j <  v_j²/a_j   (triangular)
         = d_j/v_j + v_j/a_j      if d_j >= v_j²/a_j   (trapezoidal)
T_an = max_j t_j(d_j),   d_j = |q_j − q0_j|
```
Smooth, differentiable surrogate + gradient (backprops through the B-spline
control points `g ← B(α)^T g`):
```
T_β = (1/β) log Σ_j exp(β t_j),      T_an ≤ T_β ≤ T_an + (log n)/β
∂T_β/∂q_j = w_j · dt_j/dq_j,   w_j = softmax(β t_j)
dt_j/dq_j = sign(q_j−q0_j) · ( 1/sqrt(a_j d_j) triangular ; 1/v_j trapezoidal )
```
This is a **pure joint-space cost**: reads `state.joint_state.position`
`(b,h,dof)` and `goal.current_js` (q0), returns `(b,h,1)`, backprops through
joint positions — **no FK backward, no quaternion adapter** (unlike
`RelativePoseCost`). `v̄/ā` come from `JointLimits`; jerk-aware variant available
via `JerkLimit` to match `calculate_dt_no_clamp`.

### Implementation notes for P3 (verified against v2 source)
- The data is already routed: `compute_costs` passes `state.joint_state` and
  `goal.current_js` into the hardcoded cspace branch
  (`cost_manager_robot.py:258-275`). No `GoalRegistry`/solver wiring needed.
- BUT cost dispatch is per-name and hardcoded — add a dedicated
  `if self.has_cost("exec_time")` branch (mechanical copy of the cspace block)
  calling `forward(state.joint_state, current_joint_state=goal.current_js, ...)`.
- **Guard `current_js is None`** (Optional; None in pure batch-IK).
- **Do NOT model it on `CSpaceDistCost`.** That uses L2 sum-of-squares (dense
  gradient `≈2w(q−q*)`), which ranks redundant targets at only Spearman ~0.59 vs
  the time surrogate. The exec-time objective is an **L∞ bottleneck**: a sparse
  gradient on the single slowest joint, leaving the others free in the 9-dim
  task nullspace. That structure is exactly what produces the ~25% gain.

## SOTA context (2024–2026)

**Direct analytic competitor — validates the closed-form direction.**
*A Bi-Level Optimization Method for Redundant Dual-Arm Minimum-Time Problems*,
Fried & Paternain (RPI), arXiv:2506.03982, 2025. Same domain as ETA-IK
(redundant dual-arm, **relative** formulation, minimum time) but **fully
analytic, no NN**:
- Lower level (Theorem 1): for a fixed joint-trajectory parameterization, the
  time-optimal constant path speed is **convex, closed-form** —
  `V(θ)=max_{j,pts}{ max(p'θ/q̇̄)², max(p''θ/q̈̄) }` — i.e. the tightest
  vel/accel joint constraint. This is the continuous-path analogue of our
  `max_j t_j` and our measured Spearman-0.999 result, now with a convexity proof.
- Upper level: subgradient descent on the redundancy parameters (nullspace),
  s.t. relative Cartesian path error ≤ ε.
- Models both arms as ONE coupled kinematic chain (B-TCP→B-base→A-base→A-TCP,
  Eq.16) with a **relative Jacobian** `J=[-ψΩJ_B, ΩJ_A]` — the same single-chain
  / relative-Jacobian structure as note.md's `J_rel` and the original
  "single long chain" intuition.
- Difference: it is **continuous path following** (spray/deposition); ETA-IK is
  **discrete PTP** (NBV poses). For dekon_scan they are complementary — in-segment
  sweep ≈ continuous (bi-level-style closed form), inter-segment transfer ≈
  discrete PTP (ETA-IK-style).

**Faster-solve via seeds (the throughput axis):**
- DiffusionSeeder (CoRL 2024, NVIDIA, arXiv:2410.16727) — diffusion seeds for
  cuRobo, 12–36× speedup, 86% real success, 26 ms.
- Flow-matching warm-start (arXiv:2510.03460, 2025) — point-cloud-conditioned.
- Neural-IK manifold learners (CycleIK arXiv:2307.11554, Fusion-IK) — learn the
  redundant nullspace manifold to seed an optimizer; one-to-many mapping is the
  core difficulty.

## Bottom line

- The 3-layer exec-time MLP does **not** make relative-pose IK *solve* faster
  (it adds compute); it biases the solution toward redundant branches that
  *execute* faster.
- For the **collision-free** term the MLP is unnecessary — a closed-form
  `max_j t_j` (+ logsumexp for differentiability) matches TOPPRA at Spearman
  0.999 and is what SOTA (Fried-Paternain) uses analytically.
- The MLP is only defensible for **collision-aware** time, and even then
  pre-screen-then-(batched)-trajopt gives the true value without learning.
- If throughput is the real goal (≈1e4 IK/lane in the feasibility graph), invest
  in a **seeder** (DiffusionSeeder / analytic-time-sorted seeds), not a cost.
- Cleanest v2 design: relative-pose cost (task constraint, done) + optional
  closed-form `ExecTimeCost` (redundancy resolution) + analytic-time seed
  ordering. Keep learning for the collision-excess residual only, if at all.
