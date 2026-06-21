# Validate + benchmark the closed-form ExecTimeCost in cuRobo v2.
#
#  (A) Finite-difference check of the cost gradient w.r.t. joint positions.
#  (B) Ranking quality for redundancy resolution: generate many redundant IK
#      solutions for a relative-pose task, score each candidate target with
#      {ExecTimeCost (closed form), CSpaceDist (L2), [optional learned MLP]},
#      and compare each ranking to the validated ground-truth time T_an (which
#      matches TOPPRA at Spearman ~0.999, see eta_analytic_vs_toppra.py).
#      Also report the execution time saved by picking argmin-ExecTime vs
#      argmin-CSpaceDist.
#
# Run in the neural-sdf-v2 container:
#   docker exec neural-sdf-v2 bash -c "cd ~/ws/neural_sdf/curobo_v2 && \
#       python relpose_experiments/bench_exec_time_cost.py"
import warp as wp

wp.init()

import torch

from curobo._src.cost.cost_exec_time import ExecTimeCost
from curobo._src.cost.cost_exec_time_cfg import ExecTimeCostCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose

torch.manual_seed(0)
DEVICE = "cuda"


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))


def make_exec_cost(vmax, amax, beta=8.0, weight=1.0):
    cfg = ExecTimeCostCfg(weight=[weight], vmax=list(vmax), amax=list(amax), beta=beta)
    cost = ExecTimeCost(cfg)
    return cost


def part_a_fd(kin):
    print("=== (A) finite-difference gradient check ===")
    dof = len(kin.joint_names)
    # use the robot's own joint limits
    bounds = None
    # pull limits via a throwaway transition? Simpler: read from kinematics config.
    jl = kin.kinematics_config.joint_limits
    vmax = jl.velocity[1].tolist()
    amax = jl.acceleration[1].tolist()
    cost = make_exec_cost(vmax, amax)
    cost.setup_batch_tensors(8, 1)

    q0 = (torch.randn(8, 1, dof, device=DEVICE) * 0.3)
    q = (q0 + torch.randn(8, 1, dof, device=DEVICE) * 0.4).detach()
    cjs = JointState.from_position(q0.detach(), joint_names=kin.joint_names)

    qa = q.clone().requires_grad_(True)
    c = cost.forward(JointState.from_position(qa, joint_names=kin.joint_names), current_joint_state=cjs)
    c.sum().backward()
    g = qa.grad.clone()

    # full central-difference gradient, compared by vector norm (robust to the
    # many near-zero-gradient joints that softmax suppresses — per-element rel
    # error is meaningless there).
    eps = 1e-3
    g_fd = torch.zeros_like(g)
    for j in range(dof):
        qp, qm = q.clone(), q.clone()
        qp[:, 0, j] += eps
        qm[:, 0, j] -= eps
        with torch.no_grad():
            lp = cost.forward(JointState.from_position(qp, joint_names=kin.joint_names), current_joint_state=cjs).squeeze(-1).squeeze(-1)
            lm = cost.forward(JointState.from_position(qm, joint_names=kin.joint_names), current_joint_state=cjs).squeeze(-1).squeeze(-1)
        g_fd[:, 0, j] = (lp - lm) / (2 * eps)
    rel = (g - g_fd).norm(dim=-1) / (g_fd.norm(dim=-1) + 1e-9)  # per-sample
    print(f"  per-sample grad-vector rel-err: max {rel.max():.2e}  mean {rel.mean():.2e}")
    # also the dominant (max-weight) joint per sample
    print(f"  => {'OK' if rel.max() < 2e-2 else 'MISMATCH'}\n")
    return vmax, amax


def part_b_ranking(kin, vmax, amax, learned_scorer=None):
    print("=== (B) ranking quality for redundancy resolution (dual_ur10e) ===")
    dof = len(kin.joint_names)
    # ground-truth closed-form time (no logsumexp): exact synchronized PTP time
    vmax_t = torch.tensor(vmax, device=DEVICE)
    amax_t = torch.tensor(amax, device=DEVICE)
    d_switch = vmax_t**2 / amax_t

    def true_time(q, q0):
        d = (q - q0).abs()
        t_tri = 2 * torch.sqrt(d / amax_t)
        t_trap = d / vmax_t + vmax_t / amax_t
        return torch.where(d >= d_switch, t_trap, t_tri).max(dim=-1).values

    # generate redundant IK solutions: many seeds to the SAME relative goal,
    # collected as diverse target configs. Use a single absolute goalset and
    # read back all seed solutions (return_seeds).
    ik = InverseKinematics(
        InverseKinematicsCfg.create(robot="dual_ur10e.yml", num_seeds=128)
    )
    # a reachable pose pair
    q_ref = torch.randn(1, 1, dof, device=DEVICE) * 0.5
    tp = kin.compute_kinematics(
        JointState.from_position(q_ref, joint_names=kin.joint_names)
    ).tool_poses
    goal = GoalToolPose.from_poses(
        {
            "tool1": Pose(position=tp.position[:, 0, 0].clone(), quaternion=tp.quaternion[:, 0, 0].clone()),
            "tool0": Pose(position=tp.position[:, 0, 1].clone(), quaternion=tp.quaternion[:, 0, 1].clone()),
        },
        ["tool1", "tool0"],
        num_goalset=1,
    )
    res = ik.solve_pose(goal, return_seeds=64)  # keep many redundant branches
    sols = res.js_solution.position.view(-1, dof)  # (return_seeds, dof) redundant targets
    success = res.success.view(-1)
    sols = sols[success]
    if sols.shape[0] < 8:
        print(f"  only {sols.shape[0]} successful redundant solutions; skipping")
        return
    q0 = (torch.randn(1, dof, device=DEVICE) * 0.4)  # an arbitrary current config

    gt = true_time(sols, q0)  # (K,) ground-truth PTP time per candidate
    # scorers
    exec_cost = make_exec_cost(vmax, amax)
    exec_cost.setup_batch_tensors(sols.shape[0], 1)
    cjs = JointState.from_position(q0, joint_names=kin.joint_names)
    exec_score = exec_cost.forward(
        JointState.from_position(sols.view(-1, 1, dof), joint_names=kin.joint_names),
        current_joint_state=cjs,
    ).view(-1)
    l2_score = ((sols - q0) ** 2).sum(dim=-1)  # CSpaceDist-style
    wl2_score = ((sols - q0) ** 2 / vmax_t**2).sum(dim=-1)

    print(f"  {sols.shape[0]} redundant IK solutions; ground truth = exact closed-form PTP time")
    print(f"  ExecTimeCost (logsumexp) vs GT : Spearman {spearman(exec_score, gt):.4f}")
    print(f"  CSpaceDist  (L2 sum)     vs GT : Spearman {spearman(l2_score, gt):.4f}")
    print(f"  weighted L2              vs GT : Spearman {spearman(wl2_score, gt):.4f}")
    if learned_scorer is not None:
        ml = learned_scorer(q0.expand(sols.shape[0], dof), sols).view(-1)
        print(f"  learned MLP (ETA-IK)     vs GT : Spearman {spearman(ml, gt):.4f}")

    # time saved by picking argmin-ExecTime vs argmin-CSpaceDist
    t_exec_pick = gt[torch.argmin(exec_score)].item()
    t_l2_pick = gt[torch.argmin(l2_score)].item()
    t_best = gt.min().item()
    print(
        f"  true PTP time of picked target: ExecTime-pick {t_exec_pick:.3f}s, "
        f"CSpaceDist-pick {t_l2_pick:.3f}s, oracle-best {t_best:.3f}s"
    )
    if t_l2_pick > 0:
        print(f"  ExecTime pick is {(1 - t_exec_pick / t_l2_pick) * 100:.1f}% faster than CSpaceDist pick\n")


def main():
    from curobo.kinematics import Kinematics, KinematicsCfg

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
    vmax, amax = part_a_fd(kin)
    part_b_ranking(kin, vmax, amax, learned_scorer=None)


if __name__ == "__main__":
    main()
