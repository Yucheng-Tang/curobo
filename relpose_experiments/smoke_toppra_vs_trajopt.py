# Smoke test for the ETA-IK dataset primitive + the TOPPRA-vs-trajopt question.
#
#   mode "gen"  (run in neural-sdf-v2): sample self-collision-free (q1,q2) pairs,
#       run self-collision-aware cspace trajopt, DIRECTLY validate each output
#       trajectory (collision-free over path + within vel/acc/jerk limits +
#       reaches goal), and save the VALID pairs' {q1, q2, trajopt_time} to npz.
#   mode "cmp"  (run in neural-sdf, has TOPPRA / numpy<2): load the npz, compute
#       the collision-IGNORING straight-line TOPPRA time for each pair, and
#       compare trajopt_time (collision-aware) vs toppra_time. The ratio
#       trajopt/toppra >= 1 is the collision-detour overhead — ETA-IK's premise
#       that distance/TOPPRA underestimates the true (collision-aware) time.
#
# Usage:
#   docker exec neural-sdf-v2 bash -c "cd ~/ws/neural_sdf/curobo_v2 && \
#       python relpose_experiments/smoke_toppra_vs_trajopt.py gen dual_ur10e.yml"
#   docker exec neural-sdf bash -c "cd ~/ws/neural_sdf/curobo_v2 && \
#       python relpose_experiments/smoke_toppra_vs_trajopt.py cmp dual_ur10e.yml"
import os
import sys

import numpy as np

NPZ = os.path.join(os.path.dirname(__file__), "_smoke_pairs.npz")


def gen(robot_yml, n_candidates=192, batch=24):
    import warp as wp

    wp.init()
    import torch

    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo._src.solver.solver_trajopt import TrajOptSolver, TrajOptSolverCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    torch.manual_seed(0)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_yml))
    dof = len(kin.joint_names)
    jl = kin.kinematics_config.joint_limits
    lo, hi = jl.position[0], jl.position[1]
    vmax, amax, jmax = jl.velocity[1], jl.acceleration[1], jl.jerk[1]

    sc = SelfCollisionCostCfg(weight=[1.0])
    sc.self_collision_kin_config = kin.get_self_collision_config()
    scost = SelfCollisionCost(sc)

    def free_mask(q):
        n = q.shape[0]
        scost.setup_batch_tensors(n, 1)
        sp = kin.compute_kinematics(
            JointState.from_position(q.view(n, 1, -1), joint_names=kin.joint_names)
        ).robot_spheres
        with torch.no_grad():
            return scost.forward(sp).view(n, -1).max(-1).values <= 1e-6

    cfg = TrajOptSolverCfg.create(
        robot=robot_yml, num_seeds=4, self_collision_check=True, max_batch_size=batch
    )
    tj = TrajOptSolver(cfg)

    q1_all, q2_all, t_all = [], [], []
    done = 0
    while done < n_candidates:
        # free endpoints
        c = lo + (hi - lo) * torch.rand(batch * 4, dof, device="cuda")
        c = c[free_mask(c)]
        if c.shape[0] < 2 * batch:
            continue
        q1 = c[:batch]
        q2 = c[batch : 2 * batch]
        res = tj.solve_cspace(
            JointState.from_position(q2, joint_names=kin.joint_names),
            JointState.from_position(q1, joint_names=kin.joint_names),
        )
        P = res.js_solution.position.reshape(batch, -1, dof)
        H = P.shape[1]
        free = free_mask(P.reshape(batch * H, dof)).view(batch, H).all(1)
        V = res.js_solution.velocity.abs().reshape(batch, -1, dof)
        A = res.js_solution.acceleration.abs().reshape(batch, -1, dof)
        Jk = res.js_solution.jerk.abs().reshape(batch, -1, dof)
        vok = (V <= vmax * 1.01).all(dim=(1, 2))
        aok = (A <= amax * 1.01).all(dim=(1, 2))
        jok = (Jk <= jmax * 1.01).all(dim=(1, 2))
        reached = (P[:, -1, :] - q2).abs().max(1).values < 1e-2
        good = free & vok & aok & jok & reached
        mt = res.motion_time().view(-1)
        for i in torch.where(good)[0].tolist():
            q1_all.append(q1[i].cpu().numpy())
            q2_all.append(q2[i].cpu().numpy())
            t_all.append(float(mt[i]))
        done += batch
        print(f"  candidates {done}/{n_candidates}, valid so far {len(t_all)}")
        del res, P, V, A, Jk
        torch.cuda.empty_cache()

    np.savez(
        NPZ,
        q1=np.array(q1_all),
        q2=np.array(q2_all),
        trajopt_time=np.array(t_all),
        joint_names=np.array(kin.joint_names),
        vmax=vmax.cpu().numpy(),
        amax=amax.cpu().numpy(),
    )
    print(f"saved {len(t_all)} valid collision-free trajopt pairs -> {NPZ}")
    print(f"trajopt motion_time: mean {np.mean(t_all):.3f}s range [{min(t_all):.3f},{max(t_all):.3f}]")


def cmp():
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    d = np.load(NPZ, allow_pickle=True)
    q1, q2, tjt = d["q1"], d["q2"], d["trajopt_time"]
    vmax, amax = d["vmax"], d["amax"]

    def toppra_time(a, b):
        path = ta.SplineInterpolator([0.0, 1.0], np.stack([a, b]))
        pc_v = constraint.JointVelocityConstraint(np.stack([-vmax, vmax], axis=1))
        pc_a = constraint.JointAccelerationConstraint(np.stack([-amax, amax], axis=1))
        jt = algo.TOPPRA([pc_v, pc_a], path).compute_trajectory()
        return float(jt.duration) if jt is not None else np.nan

    topp = np.array([toppra_time(q1[i], q2[i]) for i in range(len(q1))])
    ok = np.isfinite(topp)
    topp, tjt2 = topp[ok], tjt[ok]
    ratio = tjt2 / topp  # collision-aware trajopt / collision-ignoring TOPPRA
    print(f"n={len(topp)} valid collision-free trajopt pairs")
    print(f"TOPPRA (collision-IGNORING straight line): mean {topp.mean():.3f}s")
    print(f"trajopt (collision-AWARE):                 mean {tjt2.mean():.3f}s")
    print(
        f"ratio trajopt/TOPPRA (detour overhead): mean {ratio.mean():.3f} "
        f"median {np.median(ratio):.3f} max {ratio.max():.3f}"
    )
    print(f"  fraction with >10% overhead: {(ratio > 1.1).mean()*100:.1f}%")
    print(f"  fraction with >50% overhead: {(ratio > 1.5).mean()*100:.1f}%")
    print(
        "  NOTE: bare-trajopt valid pairs are biased toward near-straight (big-detour\n"
        "  pairs fail without a graph planner), so this UNDER-states the real gap that\n"
        "  ETA-IK's collision-aware MLP must capture."
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    robot = sys.argv[2] if len(sys.argv) > 2 else "dual_ur10e.yml"
    if mode == "gen":
        gen(robot)
    else:
        cmp()
