# How much headroom is there for a learned object-centric IK seeder over Halton?
#
# For a relative-pose target with a 9-DoF nullspace, solve IK with many Halton
# seeds, keep the feasible (success) solutions, and measure:
#   (1) feasible-seed YIELD: fraction of seeds that converge to a feasible soln
#       (low yield => many wasted L-BFGS refinements a seeder could cut).
#   (2) execution-time SPREAD among feasible solutions (best/median/worst):
#       how much picking the right redundant branch matters at all.
#   (3) the HEADROOM CURVE: E[best closed-form exec-time among a random K-subset
#       of seeds] for K=1,2,4,... — if it plateaus near the all-seeds best at
#       small K, Halton already nearly saturates and a seeder buys little; if it
#       needs many K, a seeder that proposes the good branch directly wins.
#
# Run in neural-sdf-v2.
import sys

import warp as wp

wp.init()
import numpy as np
import torch

from curobo._src.cost.cost_relative_pose import _quat_conjugate, _quat_multiply, _quat_rotate
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

torch.manual_seed(0)
DEVICE = "cuda"


def closed_form_time(q, q0, vmax, amax):
    d = (q - q0).abs()
    d_switch = vmax**2 / amax
    t = torch.where(d >= d_switch, d / vmax + vmax / amax, 2 * torch.sqrt(d / amax))
    return t.max(dim=-1).values


def run(robot, n_targets=20, num_seeds=64):
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot))
    dof = len(kin.joint_names)
    jl = kin.kinematics_config.joint_limits
    vmax, amax = jl.velocity[1], jl.acceleration[1]
    tf = kin.tool_frames
    ik = InverseKinematics(InverseKinematicsCfg.create(robot=robot, num_seeds=num_seeds))

    yields, spreads, curves = [], [], []
    Ks = [1, 2, 4, 8, 16, 32, num_seeds]
    for _ in range(n_targets):
        # a reachable relative-pose target = FK of a random config's two TCPs
        q_ref = torch.randn(1, 1, dof, device=DEVICE) * 0.5
        tp = kin.compute_kinematics(JointState.from_position(q_ref, joint_names=kin.joint_names)).tool_poses
        goal = GoalToolPose.from_poses(
            {tf[0]: Pose(position=tp.position[:, 0, 0].clone(), quaternion=tp.quaternion[:, 0, 0].clone()),
             tf[1]: Pose(position=tp.position[:, 0, 1].clone(), quaternion=tp.quaternion[:, 0, 1].clone())},
            tf, num_goalset=1)
        res = ik.solve_pose(goal, return_seeds=num_seeds)
        succ = res.success.view(-1)
        js = res.js_solution
        full = list(js.joint_names)
        idx = torch.tensor([full.index(n) for n in kin.joint_names], device=DEVICE)
        sols = js.position.reshape(-1, len(full))[:, idx][succ]  # (n_feasible, dof)
        yields.append(float(succ.float().mean()))
        if sols.shape[0] < 2:
            continue
        q0 = torch.randn(1, dof, device=DEVICE) * 0.4  # an arbitrary current config
        times = closed_form_time(sols, q0, vmax, amax).cpu().numpy()
        spreads.append((times.min(), np.median(times), times.max()))
        # headroom curve: best-of-K averaged over random orderings
        best = times.min()
        row = []
        for K in Ks:
            if K > len(times):
                row.append(row[-1] if row else times.min())
                continue
            # E[min of random K subset], Monte-Carlo
            mins = [times[np.random.choice(len(times), min(K, len(times)), replace=False)].min() for _ in range(200)]
            row.append(np.mean(mins) / best)  # ratio to all-seeds best (1.0 = saturated)
        curves.append(row)

    print(f"\n=== {robot} | {n_targets} relative-pose targets, {num_seeds} seeds each ===")
    print(f"feasible-seed YIELD: mean {np.mean(yields)*100:.1f}%  (of {num_seeds} Halton seeds per target)")
    sp = np.array(spreads)
    print(f"exec-time among feasible solns [s]: best {sp[:,0].mean():.2f}  median {sp[:,1].mean():.2f}  worst {sp[:,2].mean():.2f}")
    print(f"  -> worst/best ratio {sp[:,2].mean()/sp[:,0].mean():.2f}x  (how much branch choice matters)")
    cur = np.array(curves)
    print("HEADROOM CURVE  E[best-time among K seeds] / all-seeds-best  (1.00 = saturated):")
    for i, K in enumerate(Ks):
        print(f"  K={K:3d}: {cur[:,i].mean():.3f}")
    print("  (if this hits ~1.0 by small K, Halton already saturates -> low seeder headroom)")


if __name__ == "__main__":
    for robot in (sys.argv[1:] or ["dual_fr3.yml", "robdekon_scanning.yml"]):
        run(robot)
