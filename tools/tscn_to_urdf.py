"""Convert the Godot active-ragdoll scene into a URDF articulation for Isaac Lab.

`Scenes/ActiveRagdoll.tscn` is the single source of truth for the rig: every body carries its
own mass, collision primitive and PD gains, and every joint carries its own pivot and per-axis
angular limits. This script reads those numbers directly rather than restating them, so the
Isaac model cannot silently drift from the Godot one - a re-run is the only way to change it.

Two structural translations are unavoidable.

**Spherical joints do not exist in URDF.** Each `Generic6DOFJoint3D` is emitted as three stacked
revolute joints (`<Bone>_rx`, `_ry`, `_rz`) joined by two massless dummy links, all sharing the
same pivot. This is the same decomposition the MuJoCo humanoid shipped with Isaac Lab uses for
its hips (`right_thigh:0/:1/:2`). Fifteen joints become 45 revolute DOF, of which the 36 belonging
to the twelve RL-controlled bones are the policy's action space - matching the existing Godot
`JointLimitedActionSpace` width exactly, one action per URDF joint, no reshaping.

**Godot is Y-up, URDF is Z-up.** The mapping is
``urdf = (-godot.z, godot.x, godot.y)``, i.e. URDF +X forward, +Y left, +Z up. Rotation axes map
the same way, which means a rotation about Godot's Z axis is a rotation about URDF's *negative* X
axis: those limits are negated and swapped. The shoulder and hip limits are mirror-asymmetric in
the source scene, so a sign error here is visible as a left/right asymmetry that survives the
round trip - see `_convert_limits` for the invariant that catches it.

Usage (from the project root):

    python tools/tscn_to_urdf.py

Writes `isaac_lab/assets/dummy.urdf` and `isaac_lab/assets/dummy_rig.json`. The JSON is the
machine-readable rig contract: joint order, which DOF are policy-actuated, limits and gains. The
Isaac task config and the Godot-side ONNX bridge both read it instead of hardcoding the layout,
because a mismatch between the two is silent and costs a full training run to discover.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "Scenes" / "ActiveRagdoll.tscn"
OUTPUT_DIR = PROJECT_ROOT / "isaac_lab" / "assets"

# Bones the RL policy actuates. This is a SET, used only to decide which joints are actuated - the
# order here does NOT reach the action vector.
#
# The action order is `rig["actuated_joints"]`, appended in build_urdf()'s tree-order loop below,
# which produces Spine, Thigh_L, Thigh_R, Chest, ... This comment used to claim the two were the
# same and that the list had to match RagdollRLBridge.ControlledBoneNames; they are not, and only
# index 0 coincides. Anything consuming actions must read `actuated_joints`, never this list.
CONTROLLED_BONES = [
    "Spine", "Chest",
    "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
    "Thigh_L", "Shin_L", "Foot_L",
    "Thigh_R", "Shin_R", "Foot_R",
]

# Bones whose contact state the observation reports, in order (BodyStateObservation.ContactBoneNames).
CONTACT_BONES = ["Hand_L", "Hand_R", "Foot_L", "Foot_R"]

# Mass and inertia given to the two massless links that carry each decomposed joint. PhysX
# articulations solve poorly through genuinely zero-mass links, so these are small rather than
# absent: 30 dummies at 1 g add 0.03 kg to an 80.6 kg body, or 0.04%.
DUMMY_MASS = 0.001
DUMMY_INERTIA = 1.0e-6

# Joint velocity limit. The source scene constrains torque (MaxTorque) but never velocity, so this
# is an Isaac-side safety rail rather than a ported number - high enough not to bind during normal
# motion, low enough to stop a solver blow-up from launching a limb.
JOINT_VELOCITY_LIMIT = 20.0


@dataclass
class Shape:
    """A Godot collision primitive, still in Godot's local axes."""

    kind: str  # "box" | "capsule" | "sphere"
    size: tuple[float, float, float] | None = None  # box only
    radius: float = 0.0  # capsule / sphere
    height: float = 0.0  # capsule only, TOTAL height including both hemispherical caps


@dataclass
class Body:
    name: str
    mass: float
    pos: tuple[float, float, float]  # Godot world position
    parent: str | None
    shape: Shape | None
    kp: float = 0.0
    kd: float = 0.0
    max_torque: float = 0.0


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    pivot: tuple[float, float, float]  # Godot world position
    # Per-axis (lower, upper) in Godot axes; None means the axis is unlimited.
    limits: dict[str, tuple[float, float] | None] = field(default_factory=dict)


def _floats(text: str) -> list[float]:
    """Pull the numbers out of a Godot property value.

    Identifiers are stripped before matching because Godot's type names embed digits -
    `Vector3(...)`, `Transform3D(...)`, `Color(...)`. Left in, that leading `3` is matched as a
    component and shifts every subsequent index by one, which does not fail loudly: it produces a
    structurally valid URDF describing a body of the wrong size. Callers assert their expected
    arity for the same reason.
    """
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", re.sub(r"[A-Za-z_]+\d*", "", text))]


def _vector3(text: str) -> tuple[float, float, float]:
    nums = _floats(text)
    if len(nums) != 3:
        raise ValueError(f"expected 3 components in Vector3, parsed {len(nums)} from {text!r}")
    return (nums[0], nums[1], nums[2])


def _origin(transform: str) -> tuple[float, float, float]:
    """Godot Transform3D stores the 3x3 basis first, then the translation."""
    if not transform:
        return (0.0, 0.0, 0.0)
    nums = _floats(transform)
    if len(nums) != 12:
        raise ValueError(f"expected 12 components in Transform3D, parsed {len(nums)}")
    return (nums[9], nums[10], nums[11])


def _node_path(value: str) -> str:
    """`NodePath("../Pelvis")` -> `Pelvis`."""
    match = re.search(r'NodePath\("\.\./([^"]+)"\)', value)
    return match.group(1) if match else ""


def parse_scene(path: Path) -> tuple[dict[str, Body], list[Joint]]:
    text = path.read_text(encoding="utf-8")

    # --- sub-resources: the collision primitives, keyed by their SubResource id ---
    shapes: dict[str, Shape] = {}
    for block in re.split(r"^\[sub_resource ", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        match = re.match(r'type="([^"]+)" id="([^"]+)"', header)
        if not match:
            continue
        res_type, res_id = match.groups()
        props = _properties(body)
        if res_type == "BoxShape3D":
            shapes[res_id] = Shape("box", size=_vector3(props.get("size", "")))
        elif res_type == "CapsuleShape3D":
            shapes[res_id] = Shape(
                "capsule", radius=float(props.get("radius", 0)), height=float(props.get("height", 0))
            )
        elif res_type == "SphereShape3D":
            shapes[res_id] = Shape("sphere", radius=float(props.get("radius", 0)))

    # --- nodes ---
    nodes = []
    for block in re.split(r"^\[node ", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        match = re.match(r'name="([^"]+)" type="([^"]+)"', header)
        if not match:
            continue
        parent_match = re.search(r'parent="([^"]*)"', header)
        nodes.append({
            "name": match.group(1),
            "type": match.group(2),
            "parent": parent_match.group(1) if parent_match else None,
            "props": _properties(body),
        })

    # CollisionShape3D children carry the shape reference; none of them are offset from their
    # body in this scene, which _assert_no_collision_offsets re-checks on every run.
    shape_of_body: dict[str, Shape] = {}
    for node in nodes:
        if node["type"] != "CollisionShape3D":
            continue
        if "transform" in node["props"]:
            raise ValueError(
                f"CollisionShape3D under '{node['parent']}' has a local transform; the converter "
                "assumes collision geometry is centred on its body and would place it wrongly."
            )
        ref = re.search(r'SubResource\("([^"]+)"\)', node["props"].get("shape", ""))
        if ref and node["parent"]:
            shape_of_body[node["parent"]] = shapes[ref.group(1)]

    bodies: dict[str, Body] = {}
    for node in nodes:
        if node["type"] != "RigidBody3D":
            continue
        props = node["props"]
        basis = _floats(props.get("transform", ""))[:9]
        if basis and basis != [1, 0, 0, 0, 1, 0, 0, 0, 1]:
            raise ValueError(
                f"Body '{node['name']}' has a rotated basis; the converter assumes an axis-aligned "
                "rest pose and would need to compose that rotation into every child joint origin."
            )
        bodies[node["name"]] = Body(
            name=node["name"],
            mass=float(props.get("mass", 1.0)),
            pos=_origin(props.get("transform", "")),
            parent=_node_path(props.get("ParentBone", "")) or None,
            shape=shape_of_body.get(node["name"]),
            kp=float(props.get("ProportionalGain", 0.0)),
            kd=float(props.get("DerivativeGain", 0.0)),
            max_torque=float(props.get("MaxTorque", 0.0)),
        )

    joints: list[Joint] = []
    for node in nodes:
        if node["type"] != "Generic6DOFJoint3D":
            continue
        props = node["props"]
        limits: dict[str, tuple[float, float] | None] = {}
        for axis in "xyz":
            if props.get(f"angular_limit_{axis}/enabled") == "true":
                limits[axis] = (
                    float(props[f"angular_limit_{axis}/lower_angle"]),
                    float(props[f"angular_limit_{axis}/upper_angle"]),
                )
            else:
                limits[axis] = None
        joints.append(Joint(
            name=node["name"],
            parent=_node_path(props.get("node_a", "")),
            child=_node_path(props.get("node_b", "")),
            pivot=_origin(props.get("transform", "")),
            limits=limits,
        ))

    return bodies, joints


def _properties(body: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for line in body.splitlines():
        if line.startswith("["):
            break
        key, sep, value = line.partition("=")
        if sep:
            props[key.strip()] = value.strip()
    return props


# --------------------------------------------------------------------------------------------
# Coordinate conversion
# --------------------------------------------------------------------------------------------

def to_urdf(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Godot (X left, Y up, Z back) -> URDF (X forward, Y left, Z up)."""
    return (-p[2], p[0], p[1])


# Which URDF axis each Godot rotation axis becomes, and whether the sense flips. Godot's Z maps
# to URDF's -X, so rotations about it reverse: the limit pair is negated and swapped.
AXIS_MAP = {
    "x": ((0.0, 1.0, 0.0), False),   # Godot X (lateral)  -> URDF Y : pitch
    "y": ((0.0, 0.0, 1.0), False),   # Godot Y (vertical) -> URDF Z : yaw
    "z": ((1.0, 0.0, 0.0), True),    # Godot Z (forward)  -> URDF X : roll, reversed
}


def _convert_limits(limit: tuple[float, float] | None, flip: bool) -> tuple[float, float]:
    if limit is None:
        return (-math.pi, math.pi)
    lower, upper = limit
    return (-upper, -lower) if flip else (lower, upper)


def _box_size(shape: Shape) -> tuple[float, float, float]:
    """Godot box extents (x, y, z) -> URDF (z, x, y), matching the position mapping."""
    assert shape.size is not None
    return (shape.size[2], shape.size[0], shape.size[1])


def inertia(shape: Shape | None, mass: float) -> tuple[float, float, float]:
    """Principal moments about the body's own centre of mass, in URDF axes.

    The rest pose is axis-aligned and every Godot primitive is axis-aligned within its body, so
    the inertia tensor stays diagonal and the axis remap is a permutation of its three entries.
    Godot capsules run along local +Y, which is URDF +Z.
    """
    if shape is None:
        return (DUMMY_INERTIA, DUMMY_INERTIA, DUMMY_INERTIA)

    if shape.kind == "box":
        sx, sy, sz = _box_size(shape)
        return (
            mass / 12.0 * (sy * sy + sz * sz),
            mass / 12.0 * (sx * sx + sz * sz),
            mass / 12.0 * (sx * sx + sy * sy),
        )

    if shape.kind == "sphere":
        i = 0.4 * mass * shape.radius**2
        return (i, i, i)

    # Capsule, axis along URDF Z. Godot's `height` spans the whole capsule, so the cylindrical
    # section is height - 2*radius; mass is split between cylinder and caps by volume.
    r = shape.radius
    length = max(shape.height - 2.0 * r, 0.0)
    v_cyl = math.pi * r * r * length
    v_caps = 4.0 / 3.0 * math.pi * r**3
    total = v_cyl + v_caps
    m_cyl = mass * (v_cyl / total)
    m_caps = mass * (v_caps / total)
    m_hemi = m_caps / 2.0

    i_axial = 0.5 * m_cyl * r * r + 2.0 * (0.4 * m_hemi * r * r)
    i_trans = (
        m_cyl * (length * length / 12.0 + r * r / 4.0)
        + 2.0 * m_hemi * (0.4 * r * r + 0.25 * length * length + 0.375 * length * r)
    )
    return (i_trans, i_trans, i_axial)


def collision_xml(shape: Shape, indent: str) -> str:
    if shape.kind == "box":
        sx, sy, sz = _box_size(shape)
        return f'{indent}<box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'
    if shape.kind == "sphere":
        return f'{indent}<sphere radius="{shape.radius:.6f}"/>'
    # Emitted as a cylinder of the capsule's cylindrical section; the Isaac URDF importer turns it
    # back into a true capsule when replace_cylinders_with_capsules is set, which the asset config
    # does. A bare cylinder would give the limbs sharp rims and change every ground contact.
    length = max(shape.height - 2.0 * shape.radius, 0.0)
    return f'{indent}<cylinder radius="{shape.radius:.6f}" length="{length:.6f}"/>'


# --------------------------------------------------------------------------------------------
# URDF emission
# --------------------------------------------------------------------------------------------

def build_urdf(bodies: dict[str, Body], joints: list[Joint]) -> tuple[str, dict]:
    joint_by_child = {j.child: j for j in joints}
    root = next(b for b in bodies.values() if b.parent is None)

    lines = [
        '<?xml version="1.0"?>',
        '<!-- Generated by tools/tscn_to_urdf.py from Scenes/ActiveRagdoll.tscn. Do not edit. -->',
        '<robot name="p4f_dummy">',
    ]

    def link_frame(body: Body) -> tuple[float, float, float]:
        """A link's frame sits on its own joint pivot; the root sits on its body centre."""
        joint = joint_by_child.get(body.name)
        return to_urdf(joint.pivot) if joint else to_urdf(body.pos)

    def emit_link(body: Body) -> None:
        centre = to_urdf(body.pos)
        frame = link_frame(body)
        offset = tuple(centre[i] - frame[i] for i in range(3))
        ixx, iyy, izz = inertia(body.shape, body.mass)
        lines.append(f'  <link name="{body.name}">')
        lines.append('    <inertial>')
        lines.append(f'      <origin xyz="{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{body.mass:.6f}"/>')
        lines.append(f'      <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>')
        lines.append('    </inertial>')
        for tag in ("visual", "collision"):
            lines.append(f'    <{tag}>')
            lines.append(f'      <origin xyz="{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}" rpy="0 0 0"/>')
            lines.append('      <geometry>')
            lines.append(collision_xml(body.shape, "        "))
            lines.append('      </geometry>')
            lines.append(f'    </{tag}>')
        lines.append('  </link>')

    def emit_dummy(name: str) -> None:
        lines.append(f'  <link name="{name}">')
        lines.append('    <inertial>')
        lines.append(f'      <mass value="{DUMMY_MASS}"/>')
        lines.append(
            f'      <inertia ixx="{DUMMY_INERTIA}" ixy="0" ixz="0" '
            f'iyy="{DUMMY_INERTIA}" iyz="0" izz="{DUMMY_INERTIA}"/>'
        )
        lines.append('    </inertial>')
        lines.append('  </link>')

    rig: dict = {
        "source_scene": "Scenes/ActiveRagdoll.tscn",
        "root_link": root.name,
        "controlled_bones": CONTROLLED_BONES,
        "contact_bones": CONTACT_BONES,
        "actuated_joints": [],
        "passive_joints": [],
        "joints": {},
        "total_mass": round(sum(b.mass for b in bodies.values()), 6),
    }

    emit_link(root)

    # Emit in tree order so the URDF reads top-down and Isaac's DOF ordering is predictable.
    order: list[Body] = []
    frontier = [root.name]
    while frontier:
        current = frontier.pop(0)
        for body in bodies.values():
            if body.parent == current:
                order.append(body)
                frontier.append(body.name)

    for body in order:
        joint = joint_by_child[body.name]
        parent_frame = link_frame(bodies[joint.parent])
        pivot = to_urdf(joint.pivot)
        offset = tuple(pivot[i] - parent_frame[i] for i in range(3))
        is_actuated = body.name in CONTROLLED_BONES

        chain = [
            (f"{body.name}_rx", "x"),
            (f"{body.name}_ry", "y"),
            (f"{body.name}_rz", "z"),
        ]
        parent_link = joint.parent
        for index, (joint_name, godot_axis) in enumerate(chain):
            axis, flip = AXIS_MAP[godot_axis]
            lower, upper = _convert_limits(joint.limits[godot_axis], flip)
            child_link = body.name if index == 2 else f"{body.name}_dummy{index}"
            if index < 2:
                emit_dummy(child_link)
            # Only the first joint in the chain carries the offset; all three share one pivot.
            xyz = offset if index == 0 else (0.0, 0.0, 0.0)
            lines.append(f'  <joint name="{joint_name}" type="revolute">')
            lines.append(f'    <parent link="{parent_link}"/>')
            lines.append(f'    <child link="{child_link}"/>')
            lines.append(f'    <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="0 0 0"/>')
            lines.append(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>')
            lines.append(
                f'    <limit lower="{lower:.6f}" upper="{upper:.6f}" '
                f'effort="{body.max_torque:.3f}" velocity="{JOINT_VELOCITY_LIMIT}"/>'
            )
            lines.append('  </joint>')

            rig["joints"][joint_name] = {
                "bone": body.name,
                "godot_axis": godot_axis,
                "urdf_axis": list(axis),
                "reversed": flip,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "effort": body.max_torque,
                "stiffness": body.kp,
                "damping": body.kd,
                "actuated": is_actuated,
            }
            (rig["actuated_joints"] if is_actuated else rig["passive_joints"]).append(joint_name)
            parent_link = child_link

        emit_link(body)

    lines.append('</robot>')
    return "\n".join(lines) + "\n", rig


def main() -> None:
    bodies, joints = parse_scene(SCENE_PATH)
    urdf, rig = build_urdf(bodies, joints)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dummy.urdf").write_text(urdf, encoding="utf-8")
    (OUTPUT_DIR / "dummy_rig.json").write_text(json.dumps(rig, indent=2) + "\n", encoding="utf-8")

    actuated = len(rig["actuated_joints"])
    passive = len(rig["passive_joints"])
    print(f"links     : {len(bodies)} real + {2 * len(joints)} dummy")
    print(f"joints    : {actuated + passive} revolute ({actuated} actuated, {passive} passive)")
    print(f"total mass: {rig['total_mass']:.2f} kg (+{2 * len(joints) * DUMMY_MASS:.3f} kg of dummies)")
    print(f"written   : {(OUTPUT_DIR / 'dummy.urdf').relative_to(PROJECT_ROOT)}")
    print(f"            {(OUTPUT_DIR / 'dummy_rig.json').relative_to(PROJECT_ROOT)}")

    # Mirror invariant: the shoulder and hip roll limits are deliberately asymmetric in the source
    # scene, so a sign slip in the Godot-Z -> URDF-X flip shows up here rather than as a limp
    # eight hours into a training run.
    for left, right in (("UpperArm_L", "UpperArm_R"), ("Thigh_L", "Thigh_R")):
        l_lim = rig["joints"][f"{left}_rz"]
        r_lim = rig["joints"][f"{right}_rz"]
        assert math.isclose(l_lim["lower"], -r_lim["upper"]), f"{left}/{right} roll limits not mirrored"
        assert math.isclose(l_lim["upper"], -r_lim["lower"]), f"{left}/{right} roll limits not mirrored"
    print("mirror    : shoulder and hip roll limits verified symmetric")


if __name__ == "__main__":
    main()
