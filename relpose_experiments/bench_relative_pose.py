# Benchmark: relative TCP pose cost evaluation (forward + backward through FK)
#
# Variants:
#   warp_fused   - dual-arm tree + fused Warp relative pose cost (this work)
#   torch_quat   - dual-arm tree + pure-torch quaternion composition (autograd)
#   pytorch3d_v1 - dual-arm tree + pytorch3d matrix composition, replicating the
#                  v1 fork's calculate_relative_pose (incl. its hidden H2D sync)
#   long_chain   - re-rooted single chain (tool1=base) + standard ToolPoseCost
#   abs_tree     - dual-arm tree + standard ToolPoseCost, absolute goals (floor)
#
# Each iteration: FK -> cost -> sum -> backward, timed after warmup.
import time

import torch

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
from curobo._src.types.tool_pose import GoalToolPose
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

torch.manual_seed(0)
DEVICE = "cuda"
WEIGHT = [50.0, 30.0]
SHAPES = [(32, 1), (512, 1), (32, 16)]
WARMUP, ITERS = 30, 200

tree = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e.yml"))
chain = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_ur10e_rerooted.yml"))
DOF = len(tree.joint_names)
I1 = tree.tool_frames.index("tool1")
I0 = tree.tool_frames.index("tool0")

# a reachable relative goal (from a reference configuration)
with torch.no_grad():
    q_ref = torch.randn(1, 1, DOF, device=DEVICE) * 0.5
    tp = tree.compute_kinematics(
        JointState.from_position(q_ref, joint_names=tree.joint_names)
    ).tool_poses
    GOAL_P = _quat_rotate(
        _quat_conjugate(tp.quaternion[:, :, I1]), tp.position[:, :, I0] - tp.position[:, :, I1]
    ).view(1, 3)
    GOAL_Q = _quat_multiply(_quat_conjugate(tp.quaternion[:, :, I1]), tp.quaternion[:, :, I0]).view(
        1, 4
    )
GOAL = Pose(position=GOAL_P, quaternion=GOAL_Q)


def make_relative_cost(cls, b, h):
    cfg = RelativePoseCostCfg(weight=WEIGHT, base_frame="tool1", tool_frame="tool0")
    cfg.class_type = cls
    cost = cls(cfg)
    cost.set_tool_frames(tree.tool_frames)
    cost.setup_batch_tensors(b, h)
    cost.update_goal(GOAL)
    return cost


def make_tool_pose_cost(frames, b, h):
    cfg = ToolPoseCostCfg(weight=WEIGHT, tool_frames=list(frames))
    cost = ToolPoseCost(cfg)
    cost.setup_batch_tensors(b, h)
    return cost


def make_pytorch3d_fn():
    """Replicate the v1 fork: pytorch3d quat->matrix composition -> matrix->quat,
    then the same pose error as TorchRelativePoseCost."""
    import pytorch3d.transforms as p3dt

    w_pos = torch.tensor(WEIGHT[0], device=DEVICE)
    w_rot = torch.tensor(WEIGHT[1], device=DEVICE)

    def fn(tool_poses):
        p1 = tool_poses.position[:, :, I1]
        q1 = tool_poses.quaternion[:, :, I1]
        p2 = tool_poses.position[:, :, I0]
        q2 = tool_poses.quaternion[:, :, I0]
        m2 = p3dt.quaternion_to_matrix(q2)
        m1 = p3dt.quaternion_to_matrix(q1)
        m1_inv = m1.transpose(-2, -1)
        p1_inv = -torch.matmul(m1_inv, p1.unsqueeze(-1)).squeeze(-1)
        rel_rot = torch.matmul(m1_inv, m2)
        rel_p = p1_inv + torch.matmul(m1_inv, p2.unsqueeze(-1)).squeeze(-1)
        rel_q = p3dt.matrix_to_quaternion(rel_rot)  # hidden device sync inside

        position_delta = rel_p - GOAL_P.view(1, 1, 3)
        position_cost = 0.5 * w_pos * (position_delta**2).sum(dim=-1)
        quat_delta = _quat_multiply(rel_q, _quat_conjugate(GOAL_Q.view(1, 1, 4)))
        q_xyz = quat_delta[..., 1:4]
        vec_length = torch.sqrt((q_xyz**2).sum(dim=-1) + 1.0e-30)
        angle = 2.0 * torch.atan2(vec_length, torch.abs(quat_delta[..., 0]))
        rotation_cost = w_rot * angle**2
        return position_cost + rotation_cost

    return fn


def run_case(name, kin, joint_names, loss_fn, b, h, count_kernels=False):
    q0 = (torch.randn(b, h, DOF, device=DEVICE) * 0.4).detach()

    def step():
        q = q0.clone().requires_grad_(True)
        js = JointState.from_position(q, joint_names=joint_names)
        tool_poses = kin.compute_kinematics(js).tool_poses
        loss = loss_fn(tool_poses)
        loss.backward()
        return q.grad

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000.0

    n_kernels = -1
    if count_kernels:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        n_kernels = sum(
            1 for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA
        )
        n_kernels = sum(
            e.count
            for e in prof.key_averages()
            if e.device_type == torch.autograd.DeviceType.CUDA
        )
    return ms, n_kernels


def main():
    results = {}
    for b, h in SHAPES:
        idxs_goal = torch.zeros(b, 1, dtype=torch.int32, device=DEVICE)

        # warp fused relative cost
        warp_cost = make_relative_cost(RelativePoseCost, b, h)
        results[("warp_fused", b, h)] = run_case(
            "warp_fused",
            tree,
            tree.joint_names,
            lambda tp: warp_cost.forward(tp)[0].sum(),
            b,
            h,
            count_kernels=True,
        )

        # torch quaternion relative cost
        torch_cost = make_relative_cost(TorchRelativePoseCost, b, h)
        results[("torch_quat", b, h)] = run_case(
            "torch_quat",
            tree,
            tree.joint_names,
            lambda tp: torch_cost.forward(tp)[0].sum(),
            b,
            h,
            count_kernels=True,
        )

        # pytorch3d v1-fork replica
        p3d_fn = make_pytorch3d_fn()
        results[("pytorch3d_v1", b, h)] = run_case(
            "pytorch3d_v1",
            tree,
            tree.joint_names,
            lambda tp: p3d_fn(tp).sum(),
            b,
            h,
            count_kernels=True,
        )

        # long chain + standard tool pose cost (goal == relative goal)
        chain_cost = make_tool_pose_cost(["tool0"], b, h)
        chain_goal = GoalToolPose.from_poses({"tool0": GOAL}, ["tool0"], num_goalset=1)
        results[("long_chain", b, h)] = run_case(
            "long_chain",
            chain,
            chain.joint_names,
            lambda tp: chain_cost.forward(tp, chain_goal, idxs_goal)[0].sum(),
            b,
            h,
            count_kernels=True,
        )

        # tree + standard absolute tool pose cost on both tools (reference floor)
        abs_cost = make_tool_pose_cost(tree.tool_frames, b, h)
        abs_goal = GoalToolPose.from_poses(
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
            tree.tool_frames,
            num_goalset=1,
        )
        results[("abs_tree", b, h)] = run_case(
            "abs_tree",
            tree,
            tree.joint_names,
            lambda tp: abs_cost.forward(tp, abs_goal, idxs_goal)[0].sum(),
            b,
            h,
            count_kernels=True,
        )

    print(f"\n{'variant':14s}" + "".join(f" | (B={b},H={h})" for b, h in SHAPES) + " | kernels/iter")
    for name in ["abs_tree", "long_chain", "warp_fused", "torch_quat", "pytorch3d_v1"]:
        row = f"{name:14s}"
        for b, h in SHAPES:
            ms, _ = results[(name, b, h)]
            row += f" | {ms:9.3f} ms"
        row += f" | {results[(name, SHAPES[0][0], SHAPES[0][1])][1]}"
        print(row)


if __name__ == "__main__":
    main()
