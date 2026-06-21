# Does adding the GRASPED OBJECT tank IK/trajopt success, and does a PRM graph
# planner actually rescue the trajopt failures?
#
# Models the dekon_scan setup: the LEFT FR3 rigidly holds a ~15 cm object
# (a cluster of spheres attached to left_panda_hand); the RIGHT arm must avoid
# it. We measure:
#   (A) relative-pose IK feasible-seed yield AND per-query success, with vs
#       without the object.
#   (B) cspace trajopt connectivity (free q1->q2) WITH the object:
#       bare TrajOptSolver  vs  MotionPlanner (graph-seeded) — does graph help?
#
# Run in neural-sdf-v2.
import copy
import sys
from pathlib import Path

import warp as wp

wp.init()
import numpy as np
import torch
import yaml

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver, TrajOptSolverCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

DEVICE = "cuda"
# relpose_experiments/ lives under curobo_v2/, so the robot config dir is two levels up.
ROBOT_DIR = str(Path(__file__).resolve().parents[1] / "curobo/content/configs/robot")


def make_object_config():
    """Write dual_fr3_object.yml = dual_fr3 + a ~15cm object held by the left
    hand (spheres on left_panda_hand), ignored vs the left wrist/hand so only
    inter-arm (object-vs-right-arm) collisions are checked."""
    d = yaml.safe_load(open(f"{ROBOT_DIR}/dual_fr3.yml"))
    kin = d["robot_cfg"]["kinematics"]
    # object as a cluster of spheres ~0.12 m ahead of the hand along its +z axis
    obj = []
    for dz in (0.06, 0.10, 0.14, 0.18):
        for off in ([0, 0, 0], [0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0], [0, -0.05, 0]):
            obj.append({"center": [off[0], off[1], dz], "radius": 0.045})
    kin["collision_spheres"]["left_panda_hand"] = (
        kin["collision_spheres"]["left_panda_hand"] + obj
    )
    # the object near the hand must not false-collide with the left wrist links
    ig = kin["self_collision_ignore"]
    for L in ["left_panda_link5", "left_panda_link6", "left_panda_link7"]:
        ig.setdefault("left_panda_hand", [])
        if L not in ig["left_panda_hand"]:
            ig["left_panda_hand"].append(L)
    out = f"{ROBOT_DIR}/dual_fr3_object.yml"
    with open(out, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=None, width=120)
    return "dual_fr3_object.yml"


def free_checker(kin):
    sc = SelfCollisionCostCfg(weight=[1.0])
    sc.self_collision_kin_config = kin.get_self_collision_config()
    c = SelfCollisionCost(sc)

    def is_free(q):
        n = q.shape[0]
        c.setup_batch_tensors(n, 1)
        sp = kin.compute_kinematics(JointState.from_position(q.view(n, 1, -1), joint_names=kin.joint_names)).robot_spheres
        with torch.no_grad():
            return c.forward(sp).view(n, -1).max(-1).values <= 1e-6

    return is_free


def part_a_ik_yield(robot, n_targets=20, num_seeds=64):
    torch.manual_seed(0)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot))
    dof = len(kin.joint_names)
    tf = kin.tool_frames
    ik = InverseKinematics(InverseKinematicsCfg.create(robot=robot, num_seeds=num_seeds))
    yields, query_ok = [], 0
    for _ in range(n_targets):
        q_ref = torch.randn(1, 1, dof, device=DEVICE) * 0.5
        tp = kin.compute_kinematics(JointState.from_position(q_ref, joint_names=kin.joint_names)).tool_poses
        goal = GoalToolPose.from_poses(
            {tf[0]: Pose(position=tp.position[:, 0, 0].clone(), quaternion=tp.quaternion[:, 0, 0].clone()),
             tf[1]: Pose(position=tp.position[:, 0, 1].clone(), quaternion=tp.quaternion[:, 0, 1].clone())},
            tf, num_goalset=1)
        res = ik.solve_pose(goal, return_seeds=num_seeds)
        succ = res.success.view(-1)
        yields.append(float(succ.float().mean()))
        query_ok += int(succ.any())
    print(f"  {robot}: feasible-seed yield {np.mean(yields)*100:.1f}% | per-QUERY success {query_ok}/{n_targets}")
    return np.mean(yields), query_ok / n_targets


def part_b_trajopt_connectivity(robot, n_pairs=24, batch=12):
    torch.manual_seed(1)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot))
    dof = len(kin.joint_names)
    jl = kin.kinematics_config.joint_limits
    import math
    lo = torch.clamp(jl.position[0], min=-math.pi); hi = torch.clamp(jl.position[1], max=math.pi)
    is_free = free_checker(kin)
    # collision-free endpoint pairs
    c = lo + (hi - lo) * torch.rand(4000, dof, device=DEVICE)
    c = c[is_free(c)][: 2 * n_pairs]
    Q1, Q2 = c[:n_pairs], c[n_pairs:2 * n_pairs]

    # bare trajopt
    cfg = TrajOptSolverCfg.create(robot=robot, num_seeds=8, self_collision_check=True, max_batch_size=batch)
    tj = TrajOptSolver(cfg)
    bare_succ = 0
    for s in range(0, n_pairs, batch):
        e = min(s + batch, n_pairs)
        r = tj.solve_cspace(JointState.from_position(Q2[s:e], joint_names=kin.joint_names),
                            JointState.from_position(Q1[s:e], joint_names=kin.joint_names))
        bare_succ += int(r.success.view(-1).sum())

    # MotionPlanner with graph (single-problem loop)
    mp = MotionPlanner(MotionPlannerCfg.create(robot=robot, num_trajopt_seeds=8))
    graph_succ = 0
    for i in range(n_pairs):
        r = mp.plan_cspace(JointState.from_position(Q2[i:i+1], joint_names=kin.joint_names),
                           JointState.from_position(Q1[i:i+1], joint_names=kin.joint_names), max_attempts=6)
        if r is not None and int(r.success.view(-1).sum()) > 0:
            graph_succ += 1
    print(f"  {robot}: bare trajopt {bare_succ}/{n_pairs} | MotionPlanner(graph) {graph_succ}/{n_pairs}")
    return bare_succ / n_pairs, graph_succ / n_pairs


def main():
    obj_robot = make_object_config()
    print("=== (A) relative-pose IK yield: no-object vs with-object ===")
    part_a_ik_yield("dual_fr3.yml")
    part_a_ik_yield(obj_robot)
    print("=== (B) trajopt connectivity WITH object: bare vs PRM-graph ===")
    part_b_trajopt_connectivity(obj_robot)


if __name__ == "__main__":
    main()
