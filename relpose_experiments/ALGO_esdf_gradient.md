# Sphere-vs-ESDF collision distance & gradient in cuRobo v2 — algorithm + diagrams

Exactly how cuRobo v2 turns a robot collision sphere into a collision cost and a
gradient on the joint angles, against a voxelized signed-distance field (ESDF)
or an analytic primitive. Reconstructed from the source (file:line cited). This
is the machinery our object-frame sphere-vs-SDF custom cost plugs into.

Key design (same as the pose cost): **the gradient is computed analytically in
the forward pass and cached; autograd backward just replays it.**

---

## 0. End-to-end data flow

```
 joint q
   │  KinematicsFusedFunction.forward  (FK, CUDA)
   ▼
 robot_spheres[b,h,n,4]  = (cx,cy,cz, r)   world frame
   │  SphereObstacleColl.forward  (Warp, one thread per (sphere,obstacle))
   ▼
 for each obstacle:
   center ──(world→obstacle-local transform)──▶ local_pt
   local_pt ──(trilinear ESDF lookup)──▶  sdf , ∇sdf(analytic)     [§1,§2]
   penetration = (r + η) − sdf                                     [§3]
   (cost, grad_scale) = smooth_hinge(penetration, η)               [§4]
   grad_world = R · (unit ∇sdf)                                    [§2d]
   atomic_add(distance[sphere]   += w·cost)                        [§5]
   atomic_add(gradient[sphere,:3]+= w·grad_scale·grad_world)
   │  (swept only) CHOMP speed-metric scaling                      [§6]
   │  cache gradient buffer (b,h,n,4)
   ▼  KinematicsFusedFunction.backward  (CUDA)
 grad_in_robot_spheres[b,h,n,4]  ──  Σ_joints Jᵥᵀ·g  ──▶  grad_q   [§7]
```
Files: `wp_collision_kernel.py`, `wp_collision_common.py`, `data_voxel.py`,
`wp_speed_metric.py`, `wp_autograd.py`, `cuda_ops/kinematics.py`,
`kernels/kinematics/kinematics_backward_helper.cuh`.

---

## 1. Sphere → obstacle-local coordinates

Per (sphere, obstacle) thread (`wp_collision_kernel.py:112-146`):
- load sphere `center=(s.x,s.y,s.z)`, `radius=s.w`; **radius_adjusted = r + η**,
  `η = activation_distance` folded into the radius (`wp_collision_common.py:80`).
  Spheres with `r<0` are skipped.
- each obstacle stores its **inverse pose**; `local_pt = transform_point(inv_T,
  center)` brings the sphere center into the obstacle's grid frame.

---

## 2. ESDF lookup: trilinear value + ANALYTIC gradient (`data_voxel.py`)

### 2a. continuous voxel coordinate (align_corners=True, matches grid_sample)
```
vx = local_pt.x / voxel_size + grid_dims_x/2 − 0.5     (data_voxel.py:830-835)
x0 = floor(vx) ; x1 = x0+1 ; fx = vx − x0 ; fx1 = 1−fx   (and y, z)
flat index (C-order): idx = x·(ny·nz) + y·nz + z         (:728-742,:869)
```
The 8 surrounding voxel SDF values `s000..s111` are read (stored float16 →
float32, `:873-880`).

### 2b. the trilinear cell
```
        s011───────s111         f* = fractional position in the cell ∈ [0,1]
       /│          /│           value at local_pt = trilinear blend of the
     s001───────s101│            8 corner SDFs, weighted by (fx,fy,fz)
      │ │         │ │
      │s010───────s110
      │/          │/
     s000───────s100
        ──fx──▶
```
```
sdf = s000·fx1·fy1·fz1 + s100·fx·fy1·fz1 + s010·fx1·fy·fz1 + s001·fx1·fy1·fz
    + s110·fx·fy·fz1   + s101·fx·fy1·fz  + s011·fx1·fy·fz  + s111·fx·fy·fz
                                                          (data_voxel.py:883-892)
```

### 2c. analytic gradient (NOT finite differences)
Exact derivative of the trilinear field = bilinearly-interpolated corner
differences, scaled by `1/voxel_size` (`data_voxel.py:896-915`):
```
∂sdf/∂x = [ (s100−s000)·fy1·fz1 + (s101−s001)·fy1·fz
          + (s110−s010)·fy ·fz1 + (s111−s011)·fy ·fz ] / voxel_size
∂sdf/∂y = [ (s010−s000)·fx1·fz1 + (s011−s001)·fx1·fz
          + (s110−s100)·fx ·fz1 + (s111−s101)·fx ·fz ] / voxel_size
∂sdf/∂z = [ (s001−s000)·fx1·fy1 + (s011−s010)·fx1·fy
          + (s101−s100)·fx ·fy1 + (s111−s110)·fx ·fy ] / voxel_size
```
Boundary path (some corners out of grid): SDF = validity-weighted average,
gradient only sums corner-pairs both valid (`data_voxel.py:919-1069`).
Unobserved (`sdf ≥ max_dist`) → `(max_dist, 0,0,0)`.

### 2d. post-process: negate + unit-normalize, rotate to world
```
∇sdf_local = normalize( (−∂sdf/∂x, −∂sdf/∂y, −∂sdf/∂z) )     (data_voxel.py:1208-1213)
∇sdf_world = R_obstacle · ∇sdf_local           (rotation only; wp_collision_kernel.py:156-158)
```
Negation → points toward INCREASING penetration (the repulsion direction).
Unit-normalize → the ESDF gradient direction is purely geometric; its magnitude
is carried entirely by the activation `grad_scale` (§4).

---

## 3. Penetration test
```
penetration = −sdf + radius_adjusted  =  (r + η) − sdf       (wp_collision_kernel.py:154)
```
ESDF: positive OUTSIDE the obstacle, negative INSIDE. So `penetration > 0`
exactly when the sphere surface is within the activation band `η` of (or inside)
the obstacle.
```
            sphere
            ( r )                obstacle surface (sdf=0)
   center●━━━━━━━┿━━━ η ━━━┃##########  inside (sdf<0)
              r        ↑ activation band
   penetration = (r+η) − sdf
```

---

## 4. Activation: smooth hinge (C1) + speed-metric chain factor
`apply_collision_activation(dist=penetration, η)` → `(cost, grad_scale)`
(`wp_collision_common.py:11-38`):
```
 dist ≤ 0       : cost = 0            grad_scale = 0        (free, no gradient)
 0 < dist ≤ η   : cost = 0.5·dist²/η  grad_scale = dist/η   (QUADRATIC ramp)
 dist > η       : cost = dist − 0.5η  grad_scale = 1        (LINEAR)
```
```
 cost
     |                         ____ linear (slope 1)
     |                    ____/
     |               _,·''   <- quadratic 0.5·d²/η, smooth at d=η
   0 |____________,·'____________ penetration
     0            η
```
`grad_scale = d(cost)/d(penetration)` is the chain-rule factor. Accumulate
(`wp_collision_common.py:84-96`, atomic so multiple obstacles/launches sum):
```
distance[sphere]    += w · cost
gradient[sphere,:3] += w · grad_scale · ∇sdf_world          (4th slot = 0)
```
So **gradient magnitude = w · grad_scale** (the SDF direction is unit). Buffers
pre-zeroed each forward (`wp_autograd.py:76`).

---

## 5. Autograd wrapper (forward-cached gradient)
`SphereObstacleCollision(torch.autograd.Function)` (`wp_autograd.py:37-121`):
- forward: zero buffer → loop obstacle datasets, launch the kernel per type
  (atomic-add into one buffer) → return `buffer.distance`; **save only
  `buffer.gradient`**.
- backward: return the cached `gradient` directly as ∂/∂query_spheres (×
  upstream scalar if `return_loss`). O(1) tensor read — the expensive SDF+grad
  was done once in forward.

---

## 6. CHOMP speed metric (swept-trajectory path only)
Separate kernel after all obstacles accumulate (`wp_speed_metric.py`,
`wp_autograd.py:206-222`), per (b,h,sphere), central differences over
neighbouring timesteps:
```
v = 0.5/dt·(x_next − x_prev) ;  ‖v‖ ;  v̂ = v/‖v‖
acc = 1/dt²·(x_prev + x_next − 2·x_cur) ;  κ = acc/‖v‖²
orth_g    = g    − (v̂·g)·v̂          (project out motion direction: I − v̂v̂ᵀ)
orth_κ    = κ    − (v̂·κ)·v̂
new_grad  = ‖v‖ · ( orth_g − cost · orth_κ )     ← CHOMP obstacle functional
new_cost  = ‖v‖ · cost
```
Scales cost/gradient by arc-length speed so the optimizer escapes collision by
moving FASTER through the band (the CHOMP speed metric the v2 paper mentions).

---

## 7. Per-sphere world gradient → joint angles (exact Jᵥᵀ)
The cached buffer is `∂cost/∂(sphere center, world)`, shape **(b,h,n,4)** =
`[gx,gy,gz,0]`. It becomes `grad_in_robot_spheres` into
`KinematicsFusedFunction.backward` (`cuda_ops/kinematics.py:274,291-292,325`).
The CUDA backward (`kinematics_backward_helper.cuh:15-98`,
`kinematics_joint_util.cuh`) walks only the chain links affecting each sphere
and accumulates the **geometric position-Jacobian transpose**:
```
 revolute joint:  grad_q += axis_sign · ( axis × (p_sphere − p_joint) ) · g
 prismatic joint: grad_q += axis_sign · axis · g
```
This is exact (`Jᵥ = axis×(p−o)` for revolute, `axis` for prismatic), summed
over all spheres and all affecting joints, in one fused kernel. Sparsity: zero
sphere-gradients are skipped.

---

## 8. Primitive comparison: analytic box SDF (`data_cuboid.py`)
Same kernel, different overload. Center → box-local frame; half-extents `h`:
```
q = |local_pt| − h                                  (per axis)
c = max(q, 0)
sdf = ‖c‖ + min(max(qx,qy,qz), 0)                   (+outside, −inside)   (:582-592)
∇sdf: outside = c/‖c‖ with per-axis sign of local_pt (unit normal);
      inside  = ±1 on the axis with largest q, else 0                     (:595-628)
```
Already unit-magnitude → NOT re-normalized/negated, no max_dist clamp; flows
through the same penetration/hinge/accumulate. (Mesh: `data_mesh.py`, wp.Mesh.)

---

## Why this matters for dekon_scan (Stage 2/4)
Our object-frame `s_O` is exactly an ESDF in the OBJECT frame; the probe/gripper
spheres query it via this same trilinear+analytic-gradient path, and the
gradient flows to BOTH arms' joints through the FK backward (object held by arm
A moving in the world is dissolved by the object-frame choice). The analytic
trilinear gradient is microsecond-cheap and exact-for-the-field — which is why a
baked VoxelSDF beats a per-step neural-SDF/GS query at 1e4 IK/lane, and why our
sphere-decomposition lower bound on the 5D clearance field composes directly
with this machinery.
