# POTENTIAL IDEA (parked): object-centric learned IK seeder

Status: **parked idea, not built.** Measured headroom exists; net throughput win
is unproven and needs an A/B. Revisit later.

## The idea
Replace cuRobo's pose-blind Halton IK seeds with a **conditional generative
seeder** for the dual-arm relative-pose IK, specialized per grasped object:
- Input / conditioning: the target relative pose `T_rel` (a point on the scan
  trajectory) + the object-frame clearance field `F_O` (already computed).
- Output: K seed joint configs in the **feasible + fast-execution branch** of
  the 9-DoF relative-pose nullspace.
- Model: **flow matching (1–2 sampling steps)**, NOT diffusion (20–50 steps) —
  step count is decisive at ~1e4 IK/lane. ViIK (arXiv:2408.11293) / IKFlow
  (2111.08933) style; IKDiffuser (2506.13087) handles 14-DoF dual-arm trees.
- Training: once per object (or per object class). The task is **object-frame
  static within a grasp**, so there is zero distribution shift inside a scan —
  the cleanest possible learning setting; data comes free from the
  eta_ik_training dataset pipeline (sampled feasible IK solutions + closed-form
  exec-time labels).

## Why this is the strongest learning foothold on the motion side
- Halton is pose-blind; our task is object-frame static and the query family is
  a structured 5D surface manifold → a conditional model can specialize where
  Halton structurally cannot.
- It composes cleanly: seeder = PROPOSER (diverse feasible candidates),
  closed-form exec-time = SELECTOR (exact ranker, Spearman 0.999). The seeder
  does NOT touch the ranker or the certificate.
- `F_O` conditioning gives the seeder collision-awareness for free.

## Measured headroom (seed_headroom.py, no object yet)
Relative-pose IK, 64 Halton seeds, 20 targets:
| | dual_fr3 | robdekon |
|---|---|---|
| feasible-seed yield | 17% | 34.7% |
| exec-time worst/best (branch matters) | 1.73x | 1.74x |
| single-seed time / 64-seed best | 1.35x | 1.48x |
| seeds to reach <3% of best | ~16 | ~16 |

Two headrooms: (a) **feasibility** — 65–83% of Halton seeds are wasted; a
feasibility-aware seeder could cut seed count ~4x for equal yield; (b) **branch**
— need ~16 seeds to hit the fast branch (single seed 35–48% worse).

## Honest caveats
- This is a **throughput** win, not a solution-quality win: cuRobo with enough
  seeds already finds the near-best solution in ~2 ms. The seeder pays off only
  at high query volume (the 1e4/lane feasibility graph).
- Low yield might also be improved by plain cuRobo tuning (more iters / seed
  config) — the unique learning angle is object-centric conditioning.
- Net wall-clock win is **UNPROVEN**: must A/B a trained seeder vs Halton-30 on
  (feasible-seed yield, total L-BFGS iters-to-converge, end-to-end lane time).
- Headroom was measured WITHOUT the grasped object in the collision model; the
  object lowers yield and raises the value of good seeds (see the object/PRM
  experiment) — likely increasing this idea's payoff.

## If pursued
Reuse `eta_ik_training/` data gen (feasible IK solutions per object) → add a
flow-matching head conditioned on (T_rel, F_O) → A/B vs Halton in cuRobo's
IKSolver via the existing `seed_config` hook (`seed_ik_solver.py` accepts
external seeds, pads with Halton). Measure net throughput before committing.
