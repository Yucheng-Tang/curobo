# Global optimality for dual-arm minimum-time relative-pose IK

ETA-IK solves a point-to-point (PTP) problem: among the redundant target joint
configs `q` that satisfy the relative TCP-pose constraint (nullspace dim
`ρ = nA+nB−5 = 9` for a 5-DoF axisymmetric task on two 7-DoF arms), pick the one
minimizing execution time `T(q0, q)` from the current config `q0`. Because the
path is free, the redundancy space is large and the problem is strongly
nonconvex (multiple IK branches/homotopy classes, a nonsmooth min-of-max time
objective, a nonlinear FK constraint manifold). The closed-form `ExecTimeCost`
and a learned MLP, plugged into a local/quasi-Newton optimizer with multi-start
Halton seeds (as ETA-IK does), give only **local** optima — no certificate.

This documents whether/how global optimality (or a bound) is achievable. Verdict
per route, then the rigorous core.

## Two load-bearing facts

1. **Timing and target-selection separate.** In cuRobo v2 the B-spline trajopt
   already re-times any fixed path to time-optimal (`calculate_dt_no_clamp` is a
   synchronized max-over-joints of vel/acc/jerk ratios). So IK only chooses
   *which* redundant target/branch; the global question is **redundant-target
   selection**, not full trajectory optimization.
2. **The closed-form time is ½-Hölder, not Lipschitz.** `T_an = max_j t_j(d_j)`
   with `t_j` trapezoidal/triangular and `d_j = |q_j − q0_j|`. The triangular
   branch `t_j = 2√(d_j/a_j)` has unbounded slope `dt_j/dd_j = 1/√(a_j d_j) → ∞`
   as `d_j → 0`, so `T_an` is **not globally Lipschitz**. But `√d` is globally
   **½-Hölder** with finite constant: `|√d₁−√d₂| ≤ √|d₁−d₂|`, so
   `t_j` is ½-Hölder with constant `2/√a_j`, and `T_an = max_j t_j` is ½-Hölder
   in `q` with constant `C = max_j 2/√a_j`. This finite Hölder constant is what
   makes a deterministic cover bound possible despite the cusp (route 3).

## Route verdicts

### 1. Certifiably-global IK (SOS / QCQP / MIP / distance geometry) — NO for this objective at this scale
The global-IK literature certifies a **convex/quadratic** objective (or
feasibility), not min-of-max PTP time:
- **Votroubek & Kroupa, "Globally Optimal IK as a (non-convex) QCQP"**
  (arXiv:2312.15569, 2024) — minimizes weighted **joint distance** from
  preferred angles; demonstrated to **≤10 revolute joints**. ETA-IK cites it
  (their ref [13]) at **77.6 s for 10-DoF**.
- **Dai, Izatt & Tedrake, "Global IK via mixed-integer convex optimization"**
  (2019) — McCormick/MIP relaxation; feasibility + convex objective; loose/slow
  at high DoF.
- **Marić & Giamou et al.** — distance-geometry: Riemannian local
  (arXiv:2108.13720) is fast but local; the **global** variant CIDGIK
  (arXiv:2109.03374) is an SDP/convex-iteration sequence; objective is a
  distance/feasibility form, not min-time.

Why min-of-max PTP time does not fit: it needs (i) the nonlinear FK lifting,
(ii) a second-order-cone lift per joint for each `√(d_j/a_j)`, (iii)
binaries/disjunction for trapezoid-vs-triangle per joint, (iv) an epigraph for
the `max`. That is a **mixed-integer SOCP/SDP at 12–14 DoF** — strictly harder
than the quadratic case whose baseline is already 77.6 s/10-DoF. At
`≈1e4 IK/lane`, even a flat 77.6 s/query is ~9 days/lane. **Not viable.** (The
velocity-only first-order surrogate `max_j |Δq_j|/v_j` *is* LP-representable and
could be globally optimized, but it drops the acceleration phase that dominates
in the triangular regime — see eta_analytic_vs_toppra.py, where it ranks TOPPRA
at only 0.41 on the accel-limited robdekon — so it is expressible but useless.)

### 2. Single-chain / relative Jacobian + Fried-Paternain — convex lower level enables a global B&B over the *redundancy*
**Fried & Paternain, "A Bi-Level Optimization Method for Redundant Dual-Arm
Minimum-Time Problems"** (arXiv:2506.03982, 2025) models the two arms as one
coupled chain with the relative Jacobian (the "single long chain" / `J_rel`
structure) and gives a **convex, closed-form lower level** (Theorem 1: the
time-optimal path speed for a fixed joint trajectory is the tightest joint
vel/acc constraint = `max_{j,pts}{(p′θ/v̄)², p″θ/ā}`, convex) with a **nonconvex
upper level** over the redundancy parameters solved by **subgradient descent**
(local).

- Re-rooting to one chain reduces the constraint to 5 equalities and gives the
  clean ρ=9 nullspace, but does **not** convexify: the FK manifold nonlinearity
  and the discrete IK-branch multiplicity are intrinsic, not artifacts of the
  two-chain view.
- The convex closed-form lower level is the enabler: for any fixed redundancy
  parameter the cost is evaluated exactly and instantly → the precondition for a
  **Lipschitz/Hölder branch-and-bound (or DIRECT) over the ≤9-dim redundancy
  parameter only**, not the full 12–14 DoF. A valid lower bound per box comes
  from the ½-Hölder constant (route 3) or interval FK.
- **Residual obstacle:** a redundancy chart is valid only within one IK branch;
  the global optimum is therefore **global-per-chart**, and a full certificate
  needs a branch/chart enumeration on top. This is still strictly stronger than
  ETA-IK's pure random multistart (a certificate per chart). **Most promising
  rigorous route.**

### 3. Discretize-the-redundancy + Hölder cover — the practical bound
Cover the (≤9-dim) redundancy manifold within a chart with a δ-net of
`N ≈ (D/δ)^9` points, project each to the relative-pose constraint (GPU-batched
IK), score with the free closed-form `T_an`. Then:
`T* ≥ min_net T_an − C·(δ·‖N_J‖)^{1/2}` — a **deterministic** gap (½-Hölder, so
`r^{1/2}` rate, `C = max_j 2/√a_j`, `N_J` the nullspace-basis Jacobian). The
cusp does **not** kill the certificate; it only degrades the rate from linear to
√r. The `δ^{−9}` density is the cost, but each evaluation is ~free (sub-0.1 ms),
so large `N` is affordable on GPU.
- **Probabilistic version (the sweet spot):** ETA-IK's Halton multistart already
  *is* a (quasi-)Monte-Carlo cover. With `M` low-discrepancy samples and the
  Hölder constant, "global within ε at confidence 1−p" follows from a coverage
  bound — mostly bookkeeping on top of what already runs. Halton/Sobol give a
  deterministic discrepancy bound tighter than i.i.d.

### 4. Global time-optimal *trajectory* with dynamics — NO, and unnecessary
Fixed-path time-optimal control is convex (TOPP/Verscheure; Fried-Paternain
Theorem 1), but jointly optimizing the free path + timing + dynamics is
nonconvex; global MIP/collocation formulations exist only at low DoF and are not
dual-arm-scalable. Not needed: timing for a fixed path is already
near-optimal/closed-form, so the global question reduces to route 2/3.

### 5. Recommendation for the feasibility graph (~1e4 IK/lane)
Use **QMC (Halton) GPU multistart over the re-rooted relative-pose constraint,
scored by the closed-form `T_an`, with an explicit ½-Hölder cover gap** as the
certificate. A feasibility-graph edge weight is a *ranking input*, not a safety
guarantee, so the **probabilistic** bound is the right tier; pay for the
deterministic per-chart B&B (route 2) only if a downstream step demands a hard
certificate. Global-IK QP/SOS **never** beats dense GPU multistart in this
regime (12–14 DoF, min-of-max time objective, ≤low-ms/query budget). Drop the
learned exec-time MLP for the collision-free term (the closed form is already
exact); keep ML only for a collision-excess residual if needed.

## One-line verdicts
1. SOS/QCQP/MIP/distance-geometry global IK: optimizes joint-distance to ~10 DoF
   (77.6 s); cannot express min-of-max PTP time; infeasible at 12–14 DoF × 1e4. **No.**
2. Fried-Paternain convex closed-form lower level → **Hölder/Lipschitz B&B over
   the ≤9-dim redundancy = global per IK-chart.** Most promising rigorous route;
   residual = chart enumeration.
3. `T_an` is **½-Hölder** (`C = max_j 2/√a_j`) → **deterministic `C·r^{1/2}`
   cover gap** + clean probabilistic ε-bound, both layered on Halton GPU
   multistart at ~zero cost. **Recommended.**
4. Global time-optimal trajectory with dynamics: nonconvex, not scalable, and
   unnecessary (timing separates from target selection). **No.**
5. **QMC multistart + closed-form `T_an` + Hölder cover bound**; deterministic
   B&B only on demand.

Sources: arXiv:2312.15569, arXiv:2108.13720, arXiv:2109.03374, arXiv:2506.03982;
Dai-Izatt-Tedrake IJRR 2019.
