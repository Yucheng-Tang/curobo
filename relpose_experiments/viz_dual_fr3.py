# Visualize the dual FR3 URDF with pure matplotlib (no OpenGL backend needed).
#  (1) mesh render: yourdfpy poses the URDF meshes (FK), we draw the triangles
#      with a matplotlib Poly3DCollection -> dual_fr3_mesh.png
#  (2) cuRobo collision-sphere 3D scatter (the model cuRobo actually uses,
#      confirms the two arms are 0.92 m apart in Y) -> dual_fr3_spheres.png
# Headless, GL-free. Run in neural-sdf-v2.
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(
    OUT, "..", "curobo", "content", "assets", "robot", "franka_description", "dual_fr3.urdf"
)
DEFAULT_ARM = [0.0, -0.6, 0.0, -2.0, 0.0, 1.5, 0.8]  # a spread pose, per arm


def mesh_render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import yourdfpy

    # use COLLISION meshes (coarse -> fast in matplotlib) posed by FK
    urdf = yourdfpy.URDF.load(
        URDF, load_meshes=False, build_collision_scene_graph=True, load_collision_meshes=True
    )
    names = list(urdf.actuated_joint_names)
    cfg = {}
    arm = DEFAULT_ARM + [0.04, 0.04]
    for side in ("left_", "right_"):
        for i, v in enumerate(arm):
            jn = side + (f"panda_joint{i+1}" if i < 7 else f"panda_finger_joint{i-6}")
            if jn in names:
                cfg[jn] = v
    urdf.update_cfg(cfg)

    combined = urdf.collision_scene.dump(concatenate=True)  # world-frame mesh
    V, F = np.asarray(combined.vertices), np.asarray(combined.faces)
    tris = V[F]  # (nfaces, 3, 3)
    face_y = tris[:, :, 1].mean(axis=1)
    facecolors = np.where(face_y[:, None] < 0, [0.85, 0.3, 0.3, 0.9], [0.3, 0.45, 0.85, 0.9])

    fig = plt.figure(figsize=(15, 6))
    for k, (elev, azim, title) in enumerate(
        [(18, -65, "perspective"), (12, 0, "front (looking -X): 0.92 m Y gap"),
         (90, -90, "top (X-Y)")]
    ):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        pc = Poly3DCollection(tris, facecolors=facecolors, edgecolors=(0, 0, 0, 0.08), linewidths=0.1)
        ax.add_collection3d(pc)
        lim = np.stack([V.min(0), V.max(0)])
        ctr = lim.mean(0); rad = (lim[1] - lim[0]).max() / 2
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle("dual_fr3 collision meshes (red=left arm y<0, blue=right arm y>0), bases 0.92 m apart in Y")
    p = os.path.join(OUT, "dual_fr3_mesh.png")
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p, f"({len(F)} triangles)")
    return p


def sphere_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import warp as wp

    wp.init()
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("dual_fr3.yml"))
    dof = len(kin.joint_names)
    q = torch.tensor([DEFAULT_ARM * 2], device="cuda").view(1, 1, dof)
    spheres = (
        kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names))
        .robot_spheres.view(-1, 4)
        .cpu()
        .numpy()
    )
    fig = plt.figure(figsize=(12, 5))
    for k, (elev, azim, title) in enumerate(
        [(20, -60, "perspective"), (90, -90, "top view (X-Y): check 0.92 m Y gap")]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        colors = np.where(spheres[:, 1] < 0, "tab:red", "tab:blue")
        ax.scatter(spheres[:, 0], spheres[:, 1], spheres[:, 2], s=spheres[:, 3] * 800, c=colors, alpha=0.5)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle("dual_fr3 cuRobo collision spheres (red=left y<0, blue=right y>0, 0.92 m apart)")
    p = os.path.join(OUT, "dual_fr3_spheres.png")
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("wrote", p)
    return p


if __name__ == "__main__":
    try:
        mesh_render()
    except Exception as e:
        print("mesh render failed:", repr(e)[:200])
    sphere_plot()
