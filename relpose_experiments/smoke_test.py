# Smoke test: cuRobo v2 on RTX 5090 — dual_ur10e multi-tool-frame IK.
import time

import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose


def main():
    config = InverseKinematicsCfg.create(robot="dual_ur10e.yml", num_seeds=32)
    ik = InverseKinematics(config)
    print("tool_frames:", ik.tool_frames)

    goals = {
        "tool0": Pose(
            position=torch.tensor([[0.6, -0.3, 0.5]], device="cuda"),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device="cuda"),
        ),
        "tool1": Pose(
            position=torch.tensor([[0.6, 0.3, 0.5]], device="cuda"),
            quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device="cuda"),
        ),
    }
    goal = GoalToolPose.from_poses(goals, ik.tool_frames, num_goalset=1)

    t0 = time.time()
    result = ik.solve_pose(goal)
    torch.cuda.synchronize()
    t_first = time.time() - t0

    t0 = time.time()
    result = ik.solve_pose(goal)
    torch.cuda.synchronize()
    t_second = time.time() - t0

    print(f"success: {result.success.item()}")
    print(f"position_error: {result.position_error.item() * 1000:.3f} mm")
    print(f"rotation_error: {result.rotation_error.item():.5f}")
    print(f"first solve (incl. NVRTC compile): {t_first:.2f} s")
    print(f"second solve: {t_second * 1000:.1f} ms")


if __name__ == "__main__":
    main()
