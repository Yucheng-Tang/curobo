# Decompose the relative-pose gradient mismatch:
# A) pose-level: analytic world-frame formulas vs torch autograd
# B) joint-level FD for position-only and rotation-only costs
# C) upstream standard ToolPoseCost joint-level FD (convention baseline)
import torch
import warp as wp

wp.init()

from curobo._src.cost.cost_relative_pose import (
    RelativePoseCost,
    TorchRelativePoseCost,
    _quat_conjugate,
    _quat_multiply,
    _quat_rotate,
)
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

torch.manual_seed(3)
DEVICE = "cuda"
B, H = 2, 1


class FakeToolPose:
    def __init__(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion
        self.tool_frames = ["tool1", "tool0"]


def make_cost(cls, weight):
    cfg = RelativePoseCostCfg(
        weight=list(weight), base_frame="tool1", tool_frame="tool0"
    )
    cfg.class_type = cls
    c = cls(cfg)
    c.set_tool_frames(["tool1", "tool0"])
    c.setup_batch_tensors(B, H)
    return c


def rand_pose_tensors():
    p = torch.randn(B, H, 2, 3, device=DEVICE)
    q = torch.randn(B, H, 2, 4, device=DEVICE)
    q = q / q.norm(dim=-1, keepdim=True)
    return p, q


def part_a():
    print("=== A) pose-level: analytic vs autograd ===")
    goal = Pose(
        position=torch.tensor([[0.1, -0.2, 0.3]], device=DEVICE),
        quaternion=torch.nn.functional.normalize(
            torch.tensor([[0.9, 0.2, -0.3, 0.1]], device=DEVICE), dim=-1
        ),
    )
    for weight, label in [((50.0, 0.0), "pos-only"), ((0.0, 30.0), "rot-only")]:
        warp_cost = make_cost(RelativePoseCost, weight)
        torch_cost = make_cost(TorchRelativePoseCost, weight)
        warp_cost.update_goal(goal)
        torch_cost.update_goal(goal)

        p0, q0 = rand_pose_tensors()
        # autograd reference through the torch cost
        p = p0.clone().requires_grad_(True)
        q = q0.clone().requires_grad_(True)
        c, _, _ = torch_cost.forward(FakeToolPose(p, q))
        c.sum().backward()
        g_p_ref, g_q_ref = p.grad.clone(), q.grad.clone()

        # warp analytic buffers
        with torch.no_grad():
            warp_cost.forward(FakeToolPose(p0, q0))
        g_p_warp = warp_cost._out_position_gradient.clone()
        g_q_warp = warp_cost._out_rotation_gradient.clone()

        print(f"[{label}] pos grad: warp vs autograd")
        print("  ref base :", g_p_ref[0, 0, 0].tolist())
        print("  warp base:", g_p_warp[0, 0, 0].tolist())
        print("  ref tool :", g_p_ref[0, 0, 1].tolist())
        print("  warp tool:", g_p_warp[0, 0, 1].tolist())

        # convert autograd raw-quat grads to world angular gradient:
        # left perturbation dq = 0.5*(0,w) x q  =>  g_w[i] = sum_j g_q[j] d(dq_j)/d(w_i)
        def quat_grad_to_omega(g_q_raw, q_cur):
            omega = torch.zeros(B, H, 2, 3, device=DEVICE)
            for i in range(3):
                e = torch.zeros(B, H, 2, 4, device=DEVICE)
                e[..., i + 1] = 1.0
                dq = 0.5 * _quat_multiply(e, q_cur)
                omega[..., i] = (g_q_raw * dq).sum(dim=-1)
            return omega

        g_w_ref = quat_grad_to_omega(g_q_ref, q0)
        # warp rotation grads are quaternion-rate (wxyz) = q x (w,0); invert:
        # w = vec( q^-1 x rate )
        def quat_rate_to_omega(rate, q_cur):
            return _quat_multiply(_quat_conjugate(q_cur), rate)[..., 1:4]

        g_w_warp = quat_rate_to_omega(g_q_warp, q0)
        print(f"[{label}] angular grad (omega): autograd-derived vs warp-derived")
        print("  ref base :", g_w_ref[0, 0, 0].tolist())
        print("  warp base:", g_w_warp[0, 0, 0].tolist())
        print("  ref tool :", g_w_ref[0, 0, 1].tolist())
        print("  warp tool:", g_w_warp[0, 0, 1].tolist())
        print("  ratio tool:", (g_w_warp[0, 0, 1] / g_w_ref[0, 0, 1]).tolist())
        print()


def part_c():
    print("=== C) upstream ToolPoseCost: joint-level FD ===")
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
    dof = len(kin.joint_names)
    q_test = (torch.randn(1, 1, dof, device=DEVICE) * 0.4).detach()

    cfg = ToolPoseCostCfg(weight=[50.0, 30.0], tool_frames=["tool1", "tool0"])
    cost = ToolPoseCost(cfg)
    cost.setup_batch_tensors(1, 1)
    goal = GoalToolPose.from_poses(
        {
            "tool1": Pose(
                position=torch.tensor([[0.4, 0.3, 0.5]], device=DEVICE),
                quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE),
            ),
            "tool0": Pose(
                position=torch.tensor([[0.4, -0.3, 0.5]], device=DEVICE),
                quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE),
            ),
        },
        ["tool1", "tool0"],
        num_goalset=1,
    )
    idxs = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)

    def loss(qq):
        js = JointState.from_position(qq, joint_names=kin.joint_names)
        tp = kin.compute_kinematics(js).tool_poses
        c, _, _, _ = cost.forward(tp, goal, idxs)
        return c.sum()

    q_a = q_test.clone().requires_grad_(True)
    loss(q_a).backward()
    g = q_a.grad.clone()

    errs = []
    eps = 1e-3
    for j in range(dof):
        qp, qm = q_test.clone(), q_test.clone()
        qp[0, 0, j] += eps
        qm[0, 0, j] -= eps
        with torch.no_grad():
            fd = (loss(qp).item() - loss(qm).item()) / (2 * eps)
        an = g[0, 0, j].item()
        errs.append((j, fd, an, fd / an if abs(an) > 1e-8 else float("nan")))
    print("  joint | fd | analytic | fd/analytic")
    for j, fd, an, r in errs:
        print(f"  {j:2d} | {fd: .5f} | {an: .5f} | {r: .4f}")


if __name__ == "__main__":
    part_a()
    part_c()
