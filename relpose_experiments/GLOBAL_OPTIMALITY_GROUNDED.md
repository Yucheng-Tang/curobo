# Global / certified optimality for dual-arm min-time relative-pose IK — web-grounded review

Supersedes the earlier `GLOBAL_OPTIMALITY_ANALYSIS.md` pass, which ran with web access blocked
(its external paper facts were from-memory). This version verified every arXiv id / DoF / runtime
against the actual PDFs (2026-06-17). Verdicts unchanged in direction; several specifics corrected.

**Problem.** Dual-arm (12–14 DoF) minimum-EXECUTION-TIME point-to-point IK; only the RELATIVE TCP
pose is constrained (5-DoF axisymmetric task → ~9-dim redundancy). Objective = synchronized
min-of-max per-joint trapezoidal/triangular PTP time `T = max_j t_j(|dq_j|)` (nonsmooth; √ cusp at
dq=0). ~1e4 IK queries / planning lane → ms-level budget.

## Verdicts

**1. Global IK (SOS/Lasserre, QCQP, distance-geometry, MICP) — NO (fails on objective AND scale).**
- Objective: none can express min-of-max trapezoidal PTP *time*. The cost must be low-degree
  polynomial in cos/sin lift vars (SOS/QCQP), or convex in the Gram matrix (CIDGIK), or in relaxed
  SO(3) entries (MICP). `|dq_j|` is not a trig polynomial of θ and the triangular branch
  `2√(dq/a)` is a non-polynomial cusp; epigraph-lifting the `max` doesn't rescue it. In
  distance-geometry/MICP joint angles are **not decision variables** (recovered post-hoc), so a
  cost in `dq_j` literally isn't a function of the program variables. They optimize: nearest-config
  angle-distance (SOS/QCQP), EDM feasibility/rank (CIDGIK), pure feasibility/infeasibility (MICP).
- Scale: QCQP iCub 7→8→9→10 DoF = 0.2→0.8→6.5→**77.6 s** mean (Table III, 4 threads; 10.1 s on 126
  EPYC threads) — ~1 order of magnitude per DoF; 12–14 DoF extrapolates past minutes. ×1e4 queries =
  hours–days/lane. Authors: "does not apply to real-time control."

**2. Single-chain / relative-Jacobian + Fried-Paternain bi-level — borrow the modeling, NOT a global solver.**
- The relative-Jacobian single-chain coupling is the right way to model the 5-DoF relative-pose
  constraint and cheaply project seeds onto the ~9-dim redundancy manifold — worth reusing.
- Fried-Paternain (arXiv:2506.03982) is **path-following at constant speed, not PTP**: its convex
  closed-form inner value `V(θ)` (min-of-max over joints/path points) has **no accel-decel ramp and
  no √ cusp**, so it is *not* the trapezoidal PTP time. The upper level (Cartesian-error
  redundancy) is nonconvex and solved **locally** by subgradient descent ("solutions are local").
  Demonstrated only on 6-DoF planar arms, offline ~2477 s/solve.
- A global-per-chart Hölder branch-and-bound over the 9-dim redundancy is *theoretically* sound
  (see point 3) but needs ~(1/r)^9 boxes × IK-charts × 1e4 queries → dominated by GPU multistart.
  Use only as a slow offline certified oracle.

**3. Discretize redundancy + Hölder cover bound — YES (recommended production method).**
- `T(z)=max_j t_j(|dq_j(z)|)` is **½-Hölder, not Lipschitz**: triangular gradient `~1/√(a_j dq_j)
  →∞` at the cusp, but a finite Hölder constant `C = max_j 2/√a_j` (composed with the Lipschitz
  `z→dq(z)`). max-of-Hölder is Hölder, so on each chart: deterministic gap
  `|T* − min_sample T| ≤ C·r^{1/2}` (a genuine non-probabilistic certificate) + a probabilistic
  QMC-coverage bound.
- The very nonsmoothness (max + √) that breaks polynomial/convex lifting is **harmless to a sampler**:
  evaluating `T(z)` is the O(DoF) closed-form formula (µs at 14 DoF); GPU runs thousands of
  Halton/QMC seeds per ms.
- Caveats: 9-dim curse → `r^{1/2}` shrinks slowly, so use the deterministic gap as a coarse global
  gate and Newton/SQP-polish the top-K seeds for the final digits; the optimum often sits at a
  `dq_j=0` cusp → sample deliberately near dq=0; compute `C` per chart incl. the Jacobian factor.

**4. Global time-optimal trajectory with dynamics — NOT needed.** In straight-line PTP the path is
fixed, so timing degenerates to the closed-form synchronized trapezoidal `T` (µs). Convex TOPP
(TOPP-RA, Verscheure/Diehl) is global but only for a *fixed* path and ~26–30 ms/call (6-DoF) — solving
a problem you don't have. Jerk/torque-rate or free-path+dynamics is nonconvex/local, seconds/solve.
All residual nonconvexity is in redundant-TARGET selection, not timing.

**5. Recommendation.** Dense **GPU Halton/QMC multistart over the ~9-dim relative-pose null-space +
closed-form trapezoidal `T` oracle + top-K Newton/SQP polish**, certified by the **deterministic
½-Hölder cover gap `C·r^{1/2}` + a probabilistic QMC bound**. Project seeds with the relative-Jacobian
single-chain trick; sample near `dq=0`. Global-IK QP/SOS beats this **essentially never** for this
objective/budget — reserve them as *offline* oracles only: moment-SOS (Trutman) for a 7-DoF certified
ground-truth, MICP (Dai) for a one-off "is this relative pose reachable at all?" infeasibility check.
This is exactly the multi-seed-relpose-IK + closed-form-time pipeline already in use; the learned
exec-time model is justified only as a *collision-aware residual* (see `eta-ik-exectime-findings`),
not for the collision-free term.

## Corrections vs the earlier (web-blocked) analysis
1. **77.6 s is REAL** — it is the genuine 10-DoF iCub mean from Table III of arXiv:2312.15569 (not a
   misquote); prior skepticism was unwarranted.
2. The paper's true title is "Globally Optimal IK as a **Non-Convex Quadratically Constrained**
   Quadratic Program" (arXiv:2312.15569); the arXiv /abs page AND ETA-IK ref [13] both truncate it
   to "…as a Quadratic Program". It is a non-convex QCQP solved by spatial branch-and-bound (Gurobi),
   not a convex QP.
3. **SOS ≠ QCQP**: moment-SOS (Trutman et al., arXiv:2007.12550) is 7-DoF only, offline, ~2.9 s
   reduced, with a TRUE Lasserre rank certificate; the QCQP successor reaches 10 DoF at 0.26–77.6 s
   but only a Gurobi BB bound (no reconstructable certificate). The two must not be conflated.
4. **Fried-Paternain over-optimism corrected**: its `V` superficially looks like min-of-max PTP time
   but is constant-speed path-following (no ramp/cusp); it is local-only, not a global route.
5. Distance-geometry/MICP: joint angles are not decision variables → min-time isn't expressible at
   all (a stronger negative than "wrong objective").

## Verified sources
- arXiv:2312.15569 (Votroubek & Kroupa, CTU Prague) — QCQP global IK; Table III runtimes.
- arXiv:2007.12550 (Trutman, Safey El Din, Henrion, Pajdla, RA-L 2022) — moment-SOS 7-DoF IK.
- arXiv:2108.13720 (Marić et al.) Riemannian-EDM IK; arXiv:2109.03374 (Giamou et al.) CIDGIK SDP.
- Dai, Izatt, Tedrake, "Global IK via Mixed-Integer Convex Optimization," IJRR 2019.
- arXiv:2506.03982 (Fried & Paternain) — bi-level min-time path-following redundancy.
- arXiv:1707.07239 (Pham & Pham, TOPP-RA); Verscheure et al. TAC 2009 — convex TOPP.
- arXiv:2411.14381 (ETA-IK) — cites 2312.15569 as [13]; "77.6 s for 10 DoF".

(Workflow: 7 web-grounded agents, 102 tool uses; see session run wf_6aa292f5-cea.)
