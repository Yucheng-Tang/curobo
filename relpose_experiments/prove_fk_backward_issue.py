# Isolate where the wrong joint gradients come from.
#
# Same scalar cost C(p_rel, R_rel), three gradient paths:
#   path1: pure-torch FK (URDF -> torch matrix ops, NO cuRobo code) + autograd
#   path2a: cuRobo CUDA FK -> raw autograd quaternion grads -> cuRobo backward
#   path2b: same, but quaternions wrapped with _FKQuaternionGradAdapter
# Ground truth: central finite differences of the forward value.
#
# If torch were at fault, path1 would disagree with FD.
# If cuRobo's hand-written FK backward is at fault, path1 == FD == path2b,
# while path2a disagrees.
from pathlib import Path

import numpy as np
import torch

from curobo._src.cost.cost_relative_pose import _FKQuaternionGradAdapter
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reroot_urdf import Urdf, joint_axis, origin_to_tf

torch.manual_seed(5)
DEVICE = "cuda"
W_POS, W_ROT = 50.0, 30.0


# ---------- pure-torch FK from the URDF (no cuRobo) ----------
class TorchFK:
    def __init__(self, urdf_path, target_link):
        urdf = Urdf(urdf_path)
        joints, _ = urdf.path_to_root(target_link)
        self.steps = []  # (origin 4x4 tensor, axis tensor or None, joint name)
        for j in reversed(joints):
            origin = torch.tensor(
                origin_to_tf(j.find("origin")), dtype=torch.float32, device=DEVICE
            )
            if j.get("type") in ("revolute", "continuous"):
                axis = torch.tensor(
                    joint_axis(j), dtype=torch.float32, device=DEVICE
                )
                self.steps.append((origin, axis, j.get("name")))
            else:
                self.steps.append((origin, None, None))

    def forward(self, q_by_name):
        b = next(iter(q_by_name.values())).shape[0]
        tf = torch.eye(4, device=DEVICE).expand(b, 4, 4)
        for origin, axis, name in self.steps:
            tf = tf @ origin
            if axis is not None:
                q = q_by_name[name]
                ux, uy, uz = axis
                c, s = torch.cos(q), torch.sin(q)
                one = torch.ones_like(c)
                zero = torch.zeros_like(c)
                k = 1 - c
                rot = torch.stack(
                    [
                        c + ux * ux * k, ux * uy * k - uz * s, ux * uz * k + uy * s, zero,
                        uy * ux * k + uz * s, c + uy * uy * k, uy * uz * k - ux * s, zero,
                        uz * ux * k - uy * s, uz * uy * k + ux * s, c + uz * uz * k, zero,
                        zero, zero, zero, one,
                    ],
                    dim=-1,
                ).view(-1, 4, 4)
                tf = tf @ rot
        return tf


def quat_to_matrix(q):  # wxyz, differentiable, no branching
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(*q.shape[:-1], 3, 3)


def cost_from_rel(p_rel, r_rel, goal_p, goal_r):
    pos = 0.5 * W_POS * ((p_rel - goal_p) ** 2).sum(dim=-1)
    rot = W_ROT * ((r_rel - goal_r) ** 2).sum(dim=(-2, -1))
    return (pos + rot).sum()


def main():
    urdf = str(
        Path(__file__).resolve().parent.parent
        / "curobo/content/assets/robot/ur_description/dual_ur10e.urdf"
    )
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
    names = kin.joint_names
    dof = len(names)
    i1 = kin.tool_frames.index("tool1")
    i0 = kin.tool_frames.index("tool0")

    fk1 = TorchFK(urdf, "tool1")
    fk0 = TorchFK(urdf, "tool0")

    goal_p = torch.tensor([0.1, -0.2, 0.3], device=DEVICE)
    goal_r = quat_to_matrix(
        torch.nn.functional.normalize(
            torch.tensor([0.9, 0.2, -0.3, 0.1], device=DEVICE), dim=-1
        )
    )
    q0 = (torch.randn(1, dof, device=DEVICE) * 0.4).detach()

    def loss_pure_torch(q_flat):
        q_by_name = {n: q_flat[:, k] for k, n in enumerate(names)}
        t1 = fk1.forward(q_by_name)
        t0 = fk0.forward(q_by_name)
        t_rel = torch.linalg.inv(t1) @ t0
        return cost_from_rel(t_rel[:, :3, 3], t_rel[:, :3, :3], goal_p, goal_r)

    def loss_curobo(q_flat, use_adapter):
        js = JointState.from_position(q_flat.view(1, 1, dof), joint_names=names)
        tp = kin.compute_kinematics(js).tool_poses
        p1, q1 = tp.position[:, :, i1].view(-1, 3), tp.quaternion[:, :, i1].view(-1, 4)
        p0t, q0t = tp.position[:, :, i0].view(-1, 3), tp.quaternion[:, :, i0].view(-1, 4)
        if use_adapter:
            q1 = _FKQuaternionGradAdapter.apply(q1)
            q0t = _FKQuaternionGradAdapter.apply(q0t)
        r1, r0 = quat_to_matrix(q1), quat_to_matrix(q0t)
        r_rel = r1.transpose(-2, -1) @ r0
        p_rel = (r1.transpose(-2, -1) @ (p0t - p1).unsqueeze(-1)).squeeze(-1)
        return cost_from_rel(p_rel, r_rel, goal_p, goal_r)

    # gradients
    qa = q0.clone().requires_grad_(True)
    loss_pure_torch(qa).backward()
    g_pure = qa.grad.clone()

    qb = q0.clone().requires_grad_(True)
    loss_curobo(qb, use_adapter=False).backward()
    g_raw = qb.grad.clone()

    qc = q0.clone().requires_grad_(True)
    loss_curobo(qc, use_adapter=True).backward()
    g_adapter = qc.grad.clone()

    # finite differences on the pure-torch forward (values agree across paths)
    eps = 2e-3
    with torch.no_grad():
        v_pure = loss_pure_torch(q0).item()
        v_curo = loss_curobo(q0, use_adapter=False).item()
    print(f"forward values: pure-torch {v_pure:.6f} vs curobo-FK {v_curo:.6f}")
    fd = []
    for j in range(dof):
        qp, qm = q0.clone(), q0.clone()
        qp[0, j] += eps
        qm[0, j] -= eps
        with torch.no_grad():
            fd.append((loss_pure_torch(qp).item() - loss_pure_torch(qm).item()) / (2 * eps))

    print(f"\n j |     FD      | pure-torch FK | curobo raw  | curobo+adapter")
    for j in range(dof):
        print(
            f"{j:2d} | {fd[j]: 10.4f} | {g_pure[0, j]: 12.4f} | "
            f"{g_raw[0, j]: 10.4f} | {g_adapter[0, j]: 12.4f}"
        )

    def maxrel(g):
        return max(
            abs(g[0, j].item() - fd[j]) / max(abs(fd[j]), 1e-6) for j in range(dof)
        )

    print(f"\nmax rel err vs FD:  pure-torch {maxrel(g_pure):.2e}   "
          f"curobo raw {maxrel(g_raw):.2e}   curobo+adapter {maxrel(g_adapter):.2e}")


if __name__ == "__main__":
    main()
