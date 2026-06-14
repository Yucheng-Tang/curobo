# End-to-end IK with the relative TCP pose cost on dual_ur10e.
#
# Three solvers:
#   tree_abs      - stock dual-arm IK, absolute goals for both tools (baseline)
#   tree_relative - dual-arm IK with the fused relative pose cost injected into
#                   both optimizer stages via the task config dicts
#   long_chain    - IK on the re-rooted single-chain model; the absolute goal
#                   IS the relative pose (the "long chain trick")
#
# All solve the same task; we report wall time and the achieved relative pose
# error T_rel = T_tool1^-1 * T_tool0 vs the goal.
import time
from pathlib import Path

import torch
import yaml

from curobo._src.cost.cost_relative_pose import (
    _quat_conjugate,
    _quat_multiply,
    _quat_rotate,
)
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

torch.manual_seed(4)
DEVICE = "cuda"
TASK_DIR = Path(__file__).resolve().parent.parent / "curobo/content/configs/task"


def load_task_yml(rel_path):
    with open(TASK_DIR / rel_path) as f:
        return yaml.safe_load(f)


def relative_pose_of(kin, js_position, joint_names):
    tp = kin.compute_kinematics(
        JointState.from_position(js_position, joint_names=joint_names)
    ).tool_poses
    i1 = tp.tool_frames.index("tool1")
    i0 = tp.tool_frames.index("tool0")
    p1, q1 = tp.position[:, :, i1], tp.quaternion[:, :, i1]
    p0, q0 = tp.position[:, :, i0], tp.quaternion[:, :, i0]
    p_rel = _quat_rotate(_quat_conjugate(q1), p0 - p1)
    q_rel = _quat_multiply(_quat_conjugate(q1), q0)
    return p_rel.view(-1, 3), q_rel.view(-1, 4), tp


def rel_error(p_rel, q_rel, goal_p, goal_q):
    perr = (p_rel - goal_p).norm(dim=-1)
    qd = _quat_multiply(q_rel, _quat_conjugate(goal_q))
    angle = 2.0 * torch.atan2(qd[..., 1:4].norm(dim=-1), qd[..., 0].abs())
    return perr, angle


def timed_solve(ik, goal, n=5):
    result = ik.solve_pose(goal)  # warmup + graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        result = ik.solve_pose(goal)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) / n * 1000.0


def main():
    tree_kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
    dof = len(tree_kin.joint_names)

    # task: consistent absolute pose pair derived from a reference config; the
    # relative pose of that pair is the actual constraint of interest
    q_ref = torch.randn(1, 1, dof, device=DEVICE) * 0.5
    with torch.no_grad():
        goal_p_rel, goal_q_rel, tp_ref = relative_pose_of(
            tree_kin, q_ref, tree_kin.joint_names
        )
    i1 = tp_ref.tool_frames.index("tool1")
    i0 = tp_ref.tool_frames.index("tool0")
    abs_goal = GoalToolPose.from_poses(
        {
            "tool1": Pose(
                position=tp_ref.position[:, 0, i1].clone(),
                quaternion=tp_ref.quaternion[:, 0, i1].clone(),
            ),
            "tool0": Pose(
                position=tp_ref.position[:, 0, i0].clone(),
                quaternion=tp_ref.quaternion[:, 0, i0].clone(),
            ),
        },
        ["tool1", "tool0"],
        num_goalset=1,
    )

    results = {}

    # --- tree, absolute goals only ---
    ik = InverseKinematics(
        InverseKinematicsCfg.create(robot="dual_ur10e.yml", num_seeds=32)
    )
    results["tree_abs"] = timed_solve(ik, abs_goal)

    # --- tree, with relative pose cost injected into both optimizer stages ---
    goal_list = [float(v) for v in goal_p_rel.view(-1).tolist()] + [
        float(v) for v in goal_q_rel.view(-1).tolist()
    ]
    relative_pose_cfg = {
        "weight": [10000.0, 500.0],
        "base_frame": "tool1",
        "tool_frame": "tool0",
        "goal_pose": goal_list,
    }
    particle = load_task_yml("ik/particle_ik.yml")
    lbfgs = load_task_yml("ik/lbfgs_ik.yml")
    metrics = load_task_yml("../task/metrics_base.yml") if False else load_task_yml(
        "metrics_base.yml"
    )
    for cfg in (particle, lbfgs):
        cfg["rollout"]["cost_cfg"]["relative_pose_cfg"] = dict(relative_pose_cfg)
    metrics["rollout"].setdefault("convergence_cfg", {})["relative_pose_cfg"] = {
        "weight": [1.0, 1.0],
        "base_frame": "tool1",
        "tool_frame": "tool0",
        "goal_pose": goal_list,
    }
    ik_rel = InverseKinematics(
        InverseKinematicsCfg.create(
            robot="dual_ur10e.yml",
            optimizer_configs=[particle, lbfgs],
            metrics_rollout=metrics,
            num_seeds=32,
        )
    )
    results["tree_relative"] = timed_solve(ik_rel, abs_goal)

    # --- long chain: absolute goal IS the relative pose ---
    ik_chain = InverseKinematics(
        InverseKinematicsCfg.create(robot="dual_ur10e_rerooted.yml", num_seeds=32)
    )
    chain_goal = GoalToolPose.from_poses(
        {"tool0": Pose(position=goal_p_rel.clone(), quaternion=goal_q_rel.clone())},
        ["tool0"],
        num_goalset=1,
    )
    results["long_chain"] = timed_solve(ik_chain, chain_goal)

    print(f"\ngoal relative pose: p={goal_p_rel.view(-1).tolist()}")
    print(f"{'solver':14s} | success | solve ms | rel pos err (mm) | rel rot err (rad)")
    for name, (result, ms) in results.items():
        q_sol = result.js_solution.position.view(1, 1, -1)
        if name == "long_chain":
            chain_kin = ik_chain.kinematics
            tp = chain_kin.compute_kinematics(
                JointState.from_position(
                    q_sol, joint_names=chain_kin.joint_names
                )
            ).tool_poses
            p_rel = tp.position[:, 0, 0]
            q_rel = tp.quaternion[:, 0, 0]
        else:
            p_rel, q_rel, _ = relative_pose_of(tree_kin, q_sol, tree_kin.joint_names)
        perr, aerr = rel_error(
            p_rel.view(-1, 3), q_rel.view(-1, 4), goal_p_rel, goal_q_rel
        )
        print(
            f"{name:14s} | {bool(result.success.view(-1)[0].item())!s:7s} | "
            f"{ms:7.2f}  | {perr.item() * 1000:13.4f}    | {aerr.item():.6f}"
        )


if __name__ == "__main__":
    main()
