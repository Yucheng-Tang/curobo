# Validate the fused Warp relative pose cost against the pure-torch reference
# and finite differences, through the full FK chain on dual_ur10e.
import torch

from curobo._src.cost.cost_relative_pose import RelativePoseCost, TorchRelativePoseCost
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.types.pose import Pose
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

torch.manual_seed(7)
DEVICE = "cuda"


def make_cost(cls, project: bool, weight=(50.0, 30.0)):
    cfg = RelativePoseCostCfg(
        weight=list(weight),
        base_frame="tool1",
        tool_frame="tool0",
        project_distance_to_goal=project,
    )
    cfg.class_type = cls
    cost = cls(cfg)
    cost.set_tool_frames(["tool1", "tool0"])
    return cost


def fk_tool_poses(kin, q):
    js = JointState.from_position(q, joint_names=kin.joint_names)
    state = kin.compute_kinematics(js)
    return state.tool_poses


def total_cost(cost, tool_poses):
    c, _, _ = cost.forward(tool_poses)
    return c.sum()


def main():
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
    dof = len(kin.joint_names)
    B, H = 8, 4

    # goal = relative pose at a reference configuration (reachable by construction)
    q_ref = torch.randn(1, 1, dof, device=DEVICE) * 0.5
    with torch.no_grad():
        tp_ref = fk_tool_poses(kin, q_ref)
        p1 = tp_ref.position[0, 0, 0]
        q1 = tp_ref.quaternion[0, 0, 0]
        p2 = tp_ref.position[0, 0, 1]
        q2 = tp_ref.quaternion[0, 0, 1]
        # T_rel = T_tool1^-1 * T_tool0 via torch helpers
        from curobo._src.cost.cost_relative_pose import (
            _quat_conjugate,
            _quat_multiply,
            _quat_rotate,
        )

        p_rel = _quat_rotate(_quat_conjugate(q1), p2 - p1)
        q_rel = _quat_multiply(_quat_conjugate(q1), q2)
    goal = Pose(position=p_rel.view(1, 3), quaternion=q_rel.view(1, 4))

    q_test = (torch.randn(B, H, dof, device=DEVICE) * 0.4).detach()

    for project in [False, True]:
        warp_cost = make_cost(RelativePoseCost, project)
        torch_cost = make_cost(TorchRelativePoseCost, project)
        warp_cost.update_goal(goal)
        torch_cost.update_goal(goal)
        warp_cost.setup_batch_tensors(B, H)
        torch_cost.setup_batch_tensors(B, H)

        # --- forward agreement ---
        with torch.no_grad():
            tp = fk_tool_poses(kin, q_test)
            c_w, pe_w, re_w = warp_cost.forward(tp)
            c_t, pe_t, re_t = torch_cost.forward(tp)
        fwd_err = (c_w - c_t).abs().max().item() / max(c_t.abs().max().item(), 1e-9)
        print(f"[project={project}] forward rel err: {fwd_err:.2e}")
        assert fwd_err < 1e-4, "forward mismatch"

        # --- zero cost at the reference configuration ---
        warp_cost1 = make_cost(RelativePoseCost, project)
        warp_cost1.update_goal(goal)
        warp_cost1.setup_batch_tensors(1, 1)
        with torch.no_grad():
            c0, _, _ = warp_cost1.forward(fk_tool_poses(kin, q_ref))
        print(f"[project={project}] cost at goal config: {c0.abs().max().item():.3e}")
        assert c0.abs().max().item() < 1e-6, "cost not zero at goal config"

        # --- gradient agreement (warp analytic vs torch autograd, through FK) ---
        q_a = q_test.clone().requires_grad_(True)
        total_cost(warp_cost, fk_tool_poses(kin, q_a)).backward()
        grad_warp = q_a.grad.clone()  # clone: FK backward reuses its grad buffer

        q_b = q_test.clone().requires_grad_(True)
        total_cost(torch_cost, fk_tool_poses(kin, q_b)).backward()
        grad_torch = q_b.grad.clone()

        denom = grad_torch.abs().max().item()
        grad_err = (grad_warp - grad_torch).abs().max().item() / max(denom, 1e-9)
        print(f"[project={project}] grad rel err (warp vs torch): {grad_err:.2e}")

        # --- finite differences (ground truth) ---
        eps = 1e-3
        n_checks, fd_errs = 24, []
        for _ in range(n_checks):
            b = torch.randint(0, B, (1,)).item()
            h = torch.randint(0, H, (1,)).item()
            j = torch.randint(0, dof, (1,)).item()
            qp = q_test.clone()
            qm = q_test.clone()
            qp[b, h, j] += eps
            qm[b, h, j] -= eps
            with torch.no_grad():
                lp = total_cost(warp_cost, fk_tool_poses(kin, qp)).item()
                lm = total_cost(warp_cost, fk_tool_poses(kin, qm)).item()
            fd = (lp - lm) / (2 * eps)
            an = grad_warp[b, h, j].item()
            fd_errs.append(abs(fd - an) / max(abs(fd), abs(an), 1e-6))
        fd_max = max(fd_errs)
        print(f"[project={project}] FD max rel err vs warp grad: {fd_max:.2e}")
        status = "OK" if (grad_err < 1e-3 and fd_max < 2e-2) else "MISMATCH"
        print(f"[project={project}] => {status}\n")


if __name__ == "__main__":
    main()
