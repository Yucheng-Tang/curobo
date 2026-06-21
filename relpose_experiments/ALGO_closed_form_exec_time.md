# Closed-form point-to-point execution time — algorithm + diagrams

The differentiable surrogate behind `ExecTimeCost` (and the ranker that beats a
learned MLP at Spearman 0.999 vs TOPPRA). It computes the minimum time to move
from a current config `q0` to a target `q`, rest-to-rest, under per-joint box
velocity/acceleration limits.

---

## 1. The problem

Move joint `j` from `q0_j` to `q_j` (distance `d_j = |q_j − q0_j|`), starting
and ending at REST (zero boundary velocity), respecting `|q̇_j| ≤ v_j` and
`|q̈_j| ≤ a_j`. For a point-to-point move (no path constraint) each joint runs
its own time-optimal 1-D profile; the move finishes when the SLOWEST joint
finishes:

```
T(q0, q) = max_j  t_j(d_j)
```

`t_j` is the time-optimal 1-D rest-to-rest "bang-bang" profile, which takes one
of two shapes depending on whether the joint reaches its top speed.

---

## 2. Per-joint 1-D time: trapezoid vs triangle

### Trapezoidal — long move, reaches v_j
```
 q̇
 v_j |      ________________
     |     /                \
     |    /                  \        area under curve = d_j
     |   /                    \
   0 |__/______________________\____ t
     | accel |   cruise   | decel |
       v/a               d/v - v/a   v/a
```
Accelerate at `a_j` to `v_j` (time `v_j/a_j`, covers `v_j²/(2a_j)`), cruise,
decelerate symmetrically. Total time:
```
t_j = d_j / v_j  +  v_j / a_j          (reaches v_j  ⇔  d_j ≥ v_j²/a_j)
```

### Triangular — short move, never reaches v_j
```
 q̇
 vpk |        /\               vpk = sqrt(d_j · a_j) < v_j
     |       /  \              area = d_j
     |      /    \
   0 |_____/______\_________ t
     | accel | decel |
        sqrt(d/a)  sqrt(d/a)
```
Accelerate at `a_j` for half the distance, then decelerate. Total time:
```
t_j = 2 · sqrt(d_j / a_j)               (d_j < v_j²/a_j)
```

### The switch and continuity
```
        d_j  <  v_j²/a_j   →  triangle:  t_j = 2·sqrt(d_j/a_j)
        d_j  ≥  v_j²/a_j   →  trapezoid: t_j = d_j/v_j + v_j/a_j
```
At the boundary `d_j = v_j²/a_j` both give `t_j = 2·v_j/a_j` (C0) and both have
slope `dt_j/dd_j = 1/v_j` (C1). So the per-joint time is C1-smooth.

ASCII: per-joint time vs distance
```
 t_j
     |                         . trapezoid (slope 1/v_j, offset v_j/a_j)
     |                    .  '
     |               .  '
     |          _,-'        <- triangle 2·sqrt(d/a), slope 1/sqrt(a·d) -> ∞ at 0
     |      _,-'
  0  |__,-'________________________ d_j
     0     v_j²/a_j  (switch)
```

---

## 3. Synchronization: max over joints

```
 joint times t_j(d_j):     j1 ███████ 1.2 s
                           j2 ████ 0.7 s
                           j3 █████████████ 2.1 s   <- bottleneck
                           ...
 T = max_j t_j  =  2.1 s   (all joints arrive together; faster joints idle/slow)
```
For unconstrained PTP this max IS the exact time-optimum (each joint optimal,
synchronized to the slowest). TOPPRA on a *straight joint-space path* couples
all joints to one path-speed profile → slightly slower; empirically the gap is
tiny (the bottleneck joint dominates), which is why the closed form matches
TOPPRA at Spearman 0.999 (eta_analytic_vs_toppra.py).

---

## 4. Gradient (differentiable cost) — THREE separate non-smoothness sources

`T_β = (1/β)logΣ_j exp(β·t_j(d_j))`, `d_j = |q_j − q0_j|`. Differentiating needs
care at three places, with three different handlings.

### (A) Within each piece — smooth analytic derivative
Triangle, `t = 2(d/a)^½ = 2 a^{-½} d^{½}`:
```
dt/dd = 2·a^{-½}·(½)·d^{-½} = 1/sqrt(a·d)
```
Trapezoid, `t = d/v + v/a` (the `v/a` term is constant in d):
```
dt/dd = 1/v
```

### (B) At the trapezoid↔triangle switch (d* = v²/a) — C1 FOR FREE
This is the part that "just works" — the two pieces meet smoothly:
```
value :  triangle 2·sqrt(d*/a) = 2v/a   =   trapezoid d*/v + v/a = 2v/a    (C0 ✓)
slope :  triangle 1/sqrt(a·d*) = 1/v     =   trapezoid 1/v                  (C1 ✓)
```
The trapezoid *degenerates into* the triangle exactly when the peak velocity
reaches v_max, so there is NO kink. `torch.where(d≥v²/a, t_trap, t_tri)` is
differentiable here: it passes the gradient of the selected branch, and both
branches give 1/v at the boundary. (No engineering needed for this seam.)

### (C) The chain to q (abs) + the d→0 cusp — the ONLY engineered smoothing
`d=|Δ|`, `Δ=q−q0` ⇒ `dd/dΔ = sign(Δ)`, so unfloored:
```
dt/dq = sign(q−q0) · g ,   g = { 1/sqrt(a·d) triangle ; 1/v trapezoid }
```
Two problems at `Δ=0`: the abs KINK (sign undefined) AND the triangular CUSP
(`1/sqrt(a·d) → ∞`). Both are killed by ONE softabs floor used uniformly,
`d_eff = sqrt(Δ² + ε²)` (smooth in Δ everywhere; `d_eff(0)=ε`):
```
triangle : t = 2 sqrt(d_eff/a)      dt/dΔ = (1/sqrt(a·d_eff))·(Δ/d_eff) = Δ / (sqrt(a)·d_eff^{3/2})
trapezoid: t = d_eff/v + v/a        dt/dΔ = (1/v)·(Δ/d_eff)             = Δ / (v·sqrt(Δ²+ε²))
```
At `Δ=0` both → 0 (bounded); for `|Δ|≫ε` both recover the unfloored `sign(Δ)·g`.

### (D) The max over joints — logsumexp (softmax)
```
∂T_β/∂q_j = softmax_j(β·t_j) · dt_j/dq_j ,   Σ_j softmax = 1
```
See §3/§ below. So: (A) smooth pieces, (B) C1 seam free, (C) softabs floor for
the abs+cusp at d→0, (D) logsumexp for the joint-max. The trapezoid/triangle
split is the EASY part; only the d→0 cusp and the joint-max need smoothing.

### The max → logsumexp
The `max_j` is non-smooth (subgradient = the bottleneck joint). Smooth it:
```
T_β = (1/β) · log Σ_j exp(β · t_j) ,        T_an ≤ T_β ≤ T_an + (log n)/β
∂T_β/∂q_j = softmax_j(β·t_j) · dt_j/dq_j   (weights sum to 1; β≈8 soft-selects
                                            the bottleneck)
```

```
   t_j:  [3.0, 3.05, 2.9, 1.0]
   softmax(8·t):  [0.34, 0.51, 0.15, 0.00]   <- gradient flows mostly to the
                                                two near-bottleneck joints
```

### Why this is an L∞ bottleneck, NOT an L2 distance
The gradient concentrates on the slowest joint(s); the rest are free. That is
exactly what lets it resolve the 9-DoF nullspace toward fast branches. Plain
joint distance (`Σ(q−q0)²`, what CSpaceDist/Pick-IK use) spreads gradient over
all joints and ranks redundant branches at only Spearman ~0.4–0.6 — measured.

---

## 5. Where the closed form is NOT enough

It assumes a free straight-ish PTP with no collision. The real (collision-aware)
trajopt detours around obstacles, adding time the closed form misses:
```
T_trajopt(q0,q)  =  T_closed_form(q0,q)  +  ΔT_collision(q0,q,world)
                    └ exact, free, Spearman 0.999 ┘  └ learn THIS residual ┘
```
Measured detour overhead `ΔT`: ~1.34× on dual_fr3 (rare/uniform collision),
larger and more variable for close-arm robots / cluttered scenes. So: closed
form = the base + exact ranker; a learned model only earns its keep on the
sparse, non-negative collision-excess residual.

---

## 6. Implementation
`curobo/_src/cost/cost_exec_time.py`:
- `_ptp_time_per_joint(d)`: `d=sqrt(d²+ε²)`; `t_tri=2√(d/a)`, `t_trap=d/v+v/a`;
  `where(d≥v²/a, t_trap, t_tri)`.
- `time(q,q0)`: `logsumexp(β·t_j)/β`.
- pure joint-space → gradient flows directly to joint positions (NO FK backward,
  no quaternion-gradient pitfall, unlike pose costs).
- `forward()`: `weight · time` (+ optional learned residual), shape `(B,H,1)`.
Validated finite-difference exact (`bench_exec_time_cost.py`, grad rel-err 6e-4).
