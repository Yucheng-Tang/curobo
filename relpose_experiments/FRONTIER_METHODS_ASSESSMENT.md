# Frontier methods vs the dekon_scan pipeline — what helps, what doesn't

Assessment of VGGT/feed-forward-3D, NeRF/Gaussian-Splatting, diffusion/flow-matching,
distributed/multi-agent RL, and learned distance fields against the 6 stages of the
dekon_scan dual-arm scanning pipeline. arXiv ids verified live where noted; a few
future-dated ids could not be confirmed and are flagged LOW.

## One-line verdict
The frontier buys **two** things worth engineering, both learned *proposers* (never
*deciders*):
1. **A learned Next-Best-View value signal** on the unknown object (Stage 1).
2. **A learned IK seeder over the 9-DoF relative-pose nullspace** to cut wasted
   L-BFGS work across the ~1e4 IK/lane (Stages 4/5).
Everything else is **redundant-or-worse** for *our* setup. Geometry, certificates,
coverage, and redundancy-ranking are already won by LiDAR-metric SDF, the analytic
clearance bound + anisotropic-Lipschitz certificate, GTSP, and the closed-form
execution-time surrogate.

## Three governing facts that kill most of the hype *for us*
- **We have metric depth (arm LiDAR) + known relative pose** (rigid grasp). Those are
  exactly what the DUSt3R/VGGT feed-forward family exists to *estimate* — and it stays
  scale-ambiguous (VGGT 2503.11651, MASt3R 2406.09756 are similarity-invariant;
  LiDAR-VGGT 2511.01186 exists only to re-inject scale). So image-only 3D is redundant,
  not additive.
- **The object frame makes s_O and F_O static for the whole task.** This dissolves the
  selling point of GS-in-the-loop planners (Splat-Nav 2403.02751, FOCI 2505.08510),
  which exist because the *world* map is live. A one-time VoxelSDF bake amortizes to ~0.
- **We need certifiable collision-free sweeps** (nuclear decommissioning). Every
  learned-field-as-safety paper (CDF 2406.01137, neural C-space barrier 2503.04929,
  FOCI) explicitly disclaims conservatism / has no Lipschitz bound. The analytic
  certificate is strictly stronger → black-box geometry stays out of Stages 2 & 6.

## Stage-by-stage

**Stage 1 — Perception / Reconstruction / NBV → the ONE place to invest.**
- Reconstruction: no frontier method beats LiDAR-fused TSDF/neural-SDF on metric
  accuracy. *Optional* front-end only if the current neural SDF shows floaters/holes:
  LiDAR-supervised 2DGS (2403.17888) or LI-GS (2409.12899), then bake to VoxelSDF.
- **NBV is the real win.** Replace the SDF-frontier heuristic with a **supervised
  view-scorer (VIN-NBV, 2505.06219)**: predicts per-candidate Relative Reconstruction
  Improvement; ~30% better than coverage-maximizing, ~40% better than RL-NBV; no
  per-scene training; **composes with our candidate set + GTSP** (augments, not
  replaces). Alternative: Fisher-information EIG demonstrated **on a robot arm**
  ("Next Best Sense", 2410.04680, ICRA'25) — must be re-targeted from photometric to
  surface/coverage uncertainty. Feed-forward uncertainty (AREA3D 2512.05131) is
  attractive but RGB/sim-only.
- **RL-NBV does NOT help us** (GenNBV 2402.16174): 24h training, sim-only,
  uncertifiable, and its value is learning free-flight views we don't have (our
  axisymmetric probe + fixed standoff shrink the view space). Supervised scorer
  dominates.

**Stage 2 — object-frame s_O and 5D clearance F_O → keep what we have.**
- Keep baked VoxelSDF / hash-grid neural s_O. A neural SDF's only edge is gradient
  smoothness (VoxNeuS 2406.07170); querying GS/MLP per inner-loop step is 1–3 orders
  slower — fatal at 1e4 IK/lane.
- The analytic 5D lower bound `min_i[s_O(p+R(d)c_i) − ρ_i]` is microsecond, strictly
  conservative, zero-training, and *generates the labels* any learned F_O would need.
  A learned F_O only helps as **optimization guidance in deep concave pockets** where
  the sphere union false-positives — gated behind the analytic check, never as safety.

**Stage 3 — coverage planning → no frontier method beats geometric + GTSP.**
On *known* reconstructed geometry, OR-Tools/LKH solve GTSP near-optimally in ms and you
keep the optimality gap. RL-CPP / neural-combinatorial GTSP (diffusion-NCO 2411.00003)
only pay off on unknown/online maps or huge N. Only indirect use: an early *completed*
mesh lets scan-line + GTSP start before the full scan (workflow win, not a method swap).

**Stage 4 — motion generation (cuRobo) → learned SEEDER yes; learned/GS collision no.**
- **Seeder is the second real win.** cuRobo's Halton seeds are pose-blind; our task is
  object-frame static, so a conditional sampler trained once per object specializes
  where Halton can't: **flow-matching / normalizing-flow IK samplers (ViIK 2408.11293
  — 1000 collision-free configs in ~40 ms, 17× vs TRAC-IK+check; IKFlow 2111.08933;
  IKDiffuser 2506.13087 handles 14-DoF dual-arm).** Condition on `T_rel` + the
  object-frame clearance field `F_O`. Use **flow (1–2 step, 2510.03460), not diffusion
  (20–50 step)** — step count is decisive at 1e4 queries. CAVEAT: the win over cuRobo's
  tuned Halton+L-BFGS is **plausible but UNPROVEN — must A/B** feasible-seed yield and
  total L-BFGS iters-to-converge.
- **DiffusionSeeder (2410.16727) does NOT transfer as-is**: its 12–36× is against the
  graph-planner slow path in cluttered goal-reaching (which our static-object probes
  rarely trigger), and its world-frame depth conditioning throws away our object-frame
  advantage. Architecture inspirational only.
- **GS/neural fields in the optimizer loop = net loss.** FOCI (2505.08510) overlap
  integrals are costlier than sphere-vs-voxel-ESDF and break the Lipschitz certificate.
  The only viable GS→planner path is GS→ESDF bake = cuRobo's existing WorldCollisionVoxel.
- **MARL for dual-arm coordination = avoid.** The rigid relative-pose constraint is the
  textbook QMIX failure case (monotonic mixing can't represent tightly-coupled
  inter-arm value, 1803.11485). We solve the coordination analytically.

**Stage 5 — redundancy / execution time → keep the closed-form ranker untouched.**
Closed-form exec-time matches TOPPRA at Spearman 0.999 and cuRobo timing is optimal
per-path. A learned amortizer only *approximates* the exact argmin + adds drift, for
zero gain. Correct split: **generative model = SEEDER (proposes diverse feasible
candidates); closed-form = SELECTOR (exact ranker).**

**Stage 6 — certificates → frontier offers no replacement.**
Exclude all feed-forward/GS/CDF geometry from the certificate (scale-ambiguous /
hallucinated / non-conservative). The only citable upgrade: make s_O **1-Lipschitz by
construction** (2407.09505 + SLL/AOL 2303.03169) as a *provably L-bounded pre-filter*
with the analytic check as the strict arbiter — a 双保险, not a replacement; pays an
accuracy tax, doesn't bound ε or the directional axis.

## Prioritized shortlist
**Try first (medium effort, defensible payoff):**
1. **VIN-NBV-style supervised view-scorer** over existing candidate probe poses
   (Stage 1). Lowest sim-to-real risk, composes with GTSP. *Start here.*
2. **Conditional flow-matching IK seeder over the 9-DoF nullspace** (ViIK/IKFlow-style,
   conditioned on `T_rel`+`F_O`, per-object), A/B'd against Halton-30 on feasible-seed
   yield and total L-BFGS iters (Stages 4/5).

**Try second (conditional):**
3. FisherRF/Next-Best-Sense EIG NBV, re-targeted to surface uncertainty (Stage 1).
4. 1-Lipschitz s_O pre-filter (Stage 6), if upgrading the conformal-margin repair to a
   provable bound is worth the accuracy tax.
5. LiDAR-fused 2DGS/LI-GS reconstruction front-end (Stage 1), only if floaters/holes.

**Ignore / do NOT adopt:** image-only VGGT/MASt3R as geometry source; GS/neural fields
in cuRobo's collision loop (FOCI/Splat-Nav); learned exec-time amortizer replacing the
closed-form ranker; full RL-NBV (GenNBV) and MARL for the two arms; RL coverage /
neural-combinatorial GTSP for Stage 3; DiffusionSeeder as-is.

## Citation hygiene
Verified anchors: 2503.11651 (VGGT), 2505.06219 (VIN-NBV), 2410.04680 (Next Best Sense),
2410.16727 (DiffusionSeeder), 2406.01137 (CDF), 2402.16174 (GenNBV), 2505.08510 (FOCI),
2512.05131 (AREA3D), 2408.11293 (ViIK), 2111.08933 (IKFlow), 2510.03460 (flow warm-start).
LOW confidence / could not verify (do not cite externally without checking): Robo3R
(2602.10101), Subsecond-Mesh (2512.24428), BiCICLe (2604.20348), IKDiffuser exact speeds.
