# Build a dual Franka Research 3 (FR3) URDF: two Panda-geometry arms placed
# parallel, 0.92 m apart in Y (left at y=-0.46, right at y=+0.46), with FR3
# datasheet velocity limits. FR3 ~ Panda kinematics; this is a simulation model.
#
# Produces curobo/content/assets/robot/franka_description/dual_fr3.urdf and
# verifies its FK against single-Panda FK + base offset. Run in the container
# (host has no numpy):
#   docker exec neural-sdf-v2 bash -c "cd ~/ws/neural_sdf/curobo_v2 && \
#       python relpose_experiments/build_dual_fr3.py"
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FR_DIR = str(
    Path(__file__).resolve().parent.parent / "curobo/content/assets/robot/franka_description"
)
SRC = f"{FR_DIR}/franka_panda.urdf"
DST = f"{FR_DIR}/dual_fr3.urdf"
Y_OFFSET = 0.46  # half of 0.92 m
# Franka Research 3 datasheet joint limits (differ from Panda, esp. j4/j6).
FR3_VMAX = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]  # joint1..7 rad/s
FR3_QMIN = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
FR3_QMAX = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
FR3_TAUMAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]  # N*m
ARM_REVOLUTE = [f"panda_joint{i}" for i in range(1, 8)]


# ---- minimal numpy FK walker (mirrors reroot_urdf.py) ----
def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_to_tf(elem):
    xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if elem is not None:
        if elem.get("xyz"):
            xyz = [float(v) for v in elem.get("xyz").split()]
        if elem.get("rpy"):
            rpy = [float(v) for v in elem.get("rpy").split()]
    tf = np.eye(4)
    tf[:3, :3] = rpy_to_matrix(rpy)
    tf[:3, 3] = xyz
    return tf


def axis_of(j):
    a = j.find("axis")
    return np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])


def motion_tf(jtype, axis, q):
    tf = np.eye(4)
    if jtype in ("revolute", "continuous"):
        a = axis / np.linalg.norm(axis)
        c, s = math.cos(q), math.sin(q)
        ux, uy, uz = a
        tf[:3, :3] = np.array([
            [c + ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c), uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)],
        ])
    elif jtype == "prismatic":
        tf[:3, 3] = axis / np.linalg.norm(axis) * q
    return tf


class Urdf:
    def __init__(self, root):
        self.root = root
        self.parent_of = {j.find("child").get("link"): j for j in root.findall("joint")}

    def fk(self, target, q_by_name):
        tf = np.eye(4)
        chain = []
        link = target
        while link in self.parent_of:
            j = self.parent_of[link]
            chain.append(j)
            link = j.find("parent").get("link")
        for j in reversed(chain):
            tf = tf @ origin_to_tf(j.find("origin"))
            if j.get("type") in ("revolute", "continuous", "prismatic"):
                tf = tf @ motion_tf(j.get("type"), axis_of(j), q_by_name.get(j.get("name"), 0.0))
        return tf


def prefix_arm(src_root, prefix):
    """Return (links, joints) deep-copied from src_root with names prefixed."""
    links, joints = [], []
    for l in src_root.findall("link"):
        nl = copy.deepcopy(l)
        nl.set("name", prefix + l.get("name"))
        links.append(nl)
    for j in src_root.findall("joint"):
        nj = copy.deepcopy(j)
        nj.set("name", prefix + j.get("name"))
        nj.find("parent").set("link", prefix + j.find("parent").get("link"))
        nj.find("child").set("link", prefix + j.find("child").get("link"))
        # FR3 limit override (position range, velocity, effort) on arm joints
        base = j.get("name")
        if base in ARM_REVOLUTE:
            k = ARM_REVOLUTE.index(base)
            lim = nj.find("limit")
            if lim is not None:
                lim.set("lower", repr(FR3_QMIN[k]))
                lim.set("upper", repr(FR3_QMAX[k]))
                lim.set("velocity", repr(FR3_VMAX[k]))
                lim.set("effort", repr(FR3_TAUMAX[k]))
        joints.append(nj)
    return links, joints


def build():
    src = ET.parse(SRC).getroot()
    root_link = "base_link"  # single Panda root
    out = ET.Element("robot", {"name": "dual_fr3"})
    ET.SubElement(out, "link", {"name": "world_base_link"})
    for prefix, y in [("left_", -Y_OFFSET), ("right_", Y_OFFSET)]:
        links, joints = prefix_arm(src, prefix)
        for l in links:
            out.append(l)
        for j in joints:
            out.append(j)
        fj = ET.SubElement(out, "joint", {"name": f"{prefix}mount", "type": "fixed"})
        ET.SubElement(fj, "origin", {"xyz": f"0 {y} 0", "rpy": "0 0 0"})
        ET.SubElement(fj, "parent", {"link": "world_base_link"})
        ET.SubElement(fj, "child", {"link": prefix + root_link})
    ET.indent(out)
    ET.ElementTree(out).write(DST, xml_declaration=True)
    print(f"wrote {DST}")


def verify():
    single = Urdf(ET.parse(SRC).getroot())
    dual = Urdf(ET.parse(DST).getroot())
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(20):
        q = {f"panda_joint{i}": rng.uniform(-2.5, 2.5) for i in range(1, 8)}
        for side, y in [("left_", -Y_OFFSET), ("right_", Y_OFFSET)]:
            t_single = single.fk("panda_hand", q)
            offset = np.eye(4)
            offset[1, 3] = y
            expected = offset @ t_single
            qd = {side + k: v for k, v in q.items()}
            t_dual = dual.fk(side + "panda_hand", qd)
            max_err = max(max_err, np.abs(expected - t_dual).max())
    print(f"FK verification (dual vs single+offset, 20 samples): max abs error = {max_err:.3e}")
    assert max_err < 1e-9, "dual FR3 FK does not match single Panda + offset!"
    print("VERIFIED")


if __name__ == "__main__":
    build()
    verify()
