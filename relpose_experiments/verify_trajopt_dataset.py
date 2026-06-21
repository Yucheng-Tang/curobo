# Verify the self-collision-aware trajopt dataset-generation primitive works:
# batched cspace trajopt (q1 -> q2), self_collision on, read motion_time.
# Usage: python verify_trajopt_dataset.py <robot.yml>
import sys

import warp as wp

wp.init()

import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver, TrajOptSolverCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

DEVICE = "cuda"


def make_self_collision_checker(kin):
    """Return fn(q [N,dof]) -> bool mask of self-collision-FREE configs."""
    cfg = SelfCollisionCostCfg(weight=[1.0])
    cfg.self_collision_kin_config = kin.get_self_collision_config()
    cost = SelfCollisionCost(cfg)

    def is_free(q):
        n = q.shape[0]
        cost.setup_batch_tensors(n, 1)
        js = JointState.from_position(q.view(n, 1, -1), joint_names=kin.joint_names)
        spheres = kin.compute_kinematics(js).robot_spheres
        with torch.no_grad():
            dist = cost.forward(spheres).view(n, -1).max(dim=-1).values
        return dist <= 1e-6  # 0 penetration = free

    return is_free


def sample_free_pairs(kin, is_free, n_pairs, lo, hi, dof, oversample=16):
    """Rejection-sample n_pairs of self-collision-free (q1, q2)."""
    free_q = []
    while len(free_q) < 2 * n_pairs:
        cand = lo + (hi - lo) * torch.rand(n_pairs * oversample, dof, device=DEVICE)
        free_q.append(cand[is_free(cand)])
        if sum(len(f) for f in free_q) > 200 * n_pairs:  # safety cap
            break
    free = torch.cat(free_q)[: 2 * n_pairs]
    return free[:n_pairs], free[n_pairs : 2 * n_pairs]


def main(robot_yml):
    torch.manual_seed(0)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_yml))
    dof = len(kin.joint_names)
    jl = kin.kinematics_config.joint_limits
    lo, hi = jl.position[0], jl.position[1]
    print(f"robot={robot_yml} dof={dof} joints={kin.joint_names}")

    is_free = make_self_collision_checker(kin)
    probe = lo + (hi - lo) * torch.rand(2000, dof, device=DEVICE)
    frac = float(is_free(probe).float().mean())
    print(f"self-collision-free fraction of uniform samples: {frac*100:.1f}%")

    B = 16
    cfg = TrajOptSolverCfg.create(
        robot=robot_yml, num_seeds=4, self_collision_check=True, max_batch_size=B
    )
    trajopt = TrajOptSolver(cfg)
    q1, q2 = sample_free_pairs(kin, is_free, B, lo, hi, dof)
    start = JointState.from_position(q1, joint_names=kin.joint_names)
    goal = JointState.from_position(q2, joint_names=kin.joint_names)

    res = trajopt.solve_cspace(goal, start)
    mt = res.motion_time().view(-1)
    succ = res.success.view(-1)
    print(f"batch={B}  success={int(succ.sum())}/{B}")
    if succ.any():
        good = mt[succ]
        print(
            f"self-collision trajopt motion_time [s]: mean {good.mean():.3f} "
            f"range [{good.min():.3f}, {good.max():.3f}]"
        )
        print("DATASET PRIMITIVE OK for", robot_yml)
    else:
        print("WARN: no successful trajopt — check config / sampling")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dual_ur10e.yml")
