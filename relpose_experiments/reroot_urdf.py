# Re-root dual_ur10e.urdf at tool1, producing a single long serial chain
# tool1 -> (arm-1 reversed) -> world_base_link -> (arm-0 unchanged) -> tool0.
#
# Joint reversal (frame-preserving): a URDF joint is T_parent_child(q) =
# Origin * Rot(axis, q). The reversed transform Rot(axis, -q) * Origin^-1 is
# expressed as TWO URDF joints so that every original link keeps its frame:
#   1. revolute joint (same name, axis negated, identity origin) from the old
#      child to a dummy link  -- Rot(-axis, q) = Rot(axis, -q)
#   2. fixed joint carrying Origin^-1 from the dummy link to the old parent.
# Joint names, values, and limits are unchanged; visuals/collisions/spheres of
# all original links stay valid verbatim.
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_DIR = str(
    Path(__file__).resolve().parent.parent / "curobo/content/assets/robot/ur_description"
)
SRC = f"{URDF_DIR}/dual_ur10e.urdf"
DST = f"{URDF_DIR}/dual_ur10e_rerooted_tool1.urdf"
NEW_ROOT = "tool1"


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(rot):
    # inverse of extrinsic XYZ (Rz @ Ry @ Rx)
    p = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    if abs(rot[2, 0]) < 1.0 - 1e-12:
        r = math.atan2(rot[2, 1], rot[2, 2])
        y = math.atan2(rot[1, 0], rot[0, 0])
    else:  # gimbal lock
        r = math.atan2(-rot[1, 2], rot[1, 1])
        y = 0.0
    return r, p, y


def origin_to_tf(origin_elem):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_elem is not None:
        if origin_elem.get("xyz"):
            xyz = [float(v) for v in origin_elem.get("xyz").split()]
        if origin_elem.get("rpy"):
            rpy = [float(v) for v in origin_elem.get("rpy").split()]
    tf = np.eye(4)
    tf[:3, :3] = rpy_to_matrix(rpy)
    tf[:3, 3] = xyz
    return tf


def tf_to_origin_attrs(tf):
    r, p, y = matrix_to_rpy(tf[:3, :3])
    return (
        " ".join(f"{v:.16g}" for v in tf[:3, 3]),
        f"{r:.16g} {p:.16g} {y:.16g}",
    )


def joint_axis(joint_elem):
    axis_elem = joint_elem.find("axis")
    if axis_elem is None:
        return np.array([1.0, 0.0, 0.0])
    return np.array([float(v) for v in axis_elem.get("xyz").split()])


def motion_tf(joint_type, axis, q):
    tf = np.eye(4)
    if joint_type in ("revolute", "continuous"):
        a = axis / np.linalg.norm(axis)
        ca, sa = math.cos(q), math.sin(q)
        ux, uy, uz = a
        tf[:3, :3] = np.array(
            [
                [ca + ux * ux * (1 - ca), ux * uy * (1 - ca) - uz * sa, ux * uz * (1 - ca) + uy * sa],
                [uy * ux * (1 - ca) + uz * sa, ca + uy * uy * (1 - ca), uy * uz * (1 - ca) - ux * sa],
                [uz * ux * (1 - ca) - uy * sa, uz * uy * (1 - ca) + ux * sa, ca + uz * uz * (1 - ca)],
            ]
        )
    elif joint_type == "prismatic":
        tf[:3, 3] = axis / np.linalg.norm(axis) * q
    return tf


class Urdf:
    def __init__(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        self.parent_of = {}  # child link -> joint elem
        for j in self.root.findall("joint"):
            self.parent_of[j.find("child").get("link")] = j

    def path_to_root(self, link):
        joints = []
        while link in self.parent_of:
            j = self.parent_of[link]
            joints.append(j)
            link = j.find("parent").get("link")
        return joints, link

    def fk(self, target_link, q_by_name):
        tf = np.eye(4)
        joints, root = self.path_to_root(target_link)
        for j in reversed(joints):
            tf = tf @ origin_to_tf(j.find("origin"))
            if j.get("type") in ("revolute", "continuous", "prismatic"):
                tf = tf @ motion_tf(j.get("type"), joint_axis(j), q_by_name[j.get("name")])
        return tf


def reroot():
    src = Urdf(SRC)
    path_joints, old_root = src.path_to_root(NEW_ROOT)
    print(f"re-rooting at {NEW_ROOT}; path to {old_root}: "
          f"{[j.get('name') for j in path_joints]}")

    new_root_elem = ET.Element("robot", {"name": src.root.get("name") + "_rerooted"})
    path_names = {j.get("name") for j in path_joints}

    # all links unchanged
    for link in src.root.findall("link"):
        new_root_elem.append(copy.deepcopy(link))
    # all non-path joints unchanged
    for j in src.root.findall("joint"):
        if j.get("name") not in path_names:
            new_root_elem.append(copy.deepcopy(j))

    # reversed path joints (visited from tool1 toward old root)
    for j in path_joints:
        name = j.get("name")
        jtype = j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin_inv = np.linalg.inv(origin_to_tf(j.find("origin")))
        xyz, rpy = tf_to_origin_attrs(origin_inv)

        if jtype == "fixed":
            rev = ET.SubElement(new_root_elem, "joint", {"name": name + "__rev", "type": "fixed"})
            ET.SubElement(rev, "origin", {"xyz": xyz, "rpy": rpy})
            ET.SubElement(rev, "parent", {"link": child})
            ET.SubElement(rev, "child", {"link": parent})
        elif jtype in ("revolute", "continuous", "prismatic"):
            dummy = f"{parent}__rev"
            ET.SubElement(new_root_elem, "link", {"name": dummy})
            rev = ET.SubElement(new_root_elem, "joint", {"name": name, "type": jtype})
            ET.SubElement(rev, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(rev, "parent", {"link": child})
            ET.SubElement(rev, "child", {"link": dummy})
            axis = joint_axis(j)
            ET.SubElement(
                rev, "axis", {"xyz": " ".join(f"{-v:.16g}" for v in axis)}
            )
            limit = j.find("limit")
            if limit is not None:
                ET.SubElement(rev, "limit", dict(limit.attrib))
            dyn = j.find("dynamics")
            if dyn is not None:
                ET.SubElement(rev, "dynamics", dict(dyn.attrib))
            fix = ET.SubElement(
                new_root_elem, "joint", {"name": name + "__revfix", "type": "fixed"}
            )
            ET.SubElement(fix, "origin", {"xyz": xyz, "rpy": rpy})
            ET.SubElement(fix, "parent", {"link": dummy})
            ET.SubElement(fix, "child", {"link": parent})
        else:
            raise ValueError(f"unsupported joint type {jtype}")

    ET.indent(new_root_elem)
    ET.ElementTree(new_root_elem).write(DST, xml_declaration=True)
    print(f"wrote {DST}")


def verify():
    src = Urdf(SRC)
    dst = Urdf(DST)
    actuated = [
        j.get("name")
        for j in src.root.findall("joint")
        if j.get("type") in ("revolute", "continuous", "prismatic")
    ]
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(50):
        q = {}
        for name in actuated:
            limit = src.joints[name].find("limit")
            lo = float(limit.get("lower", -3.14))
            hi = float(limit.get("upper", 3.14))
            q[name] = rng.uniform(lo, hi)
        t_tool1 = src.fk("tool1", q)
        t_tool0 = src.fk("tool0", q)
        t_rel = np.linalg.inv(t_tool1) @ t_tool0
        t_new = dst.fk("tool0", q)
        max_err = max(max_err, np.abs(t_rel - t_new).max())
    print(f"FK verification over 50 random q: max abs error = {max_err:.3e}")
    assert max_err < 1e-9, "re-rooted FK does not match!"
    print("VERIFIED")


if __name__ == "__main__":
    reroot()
    verify()
