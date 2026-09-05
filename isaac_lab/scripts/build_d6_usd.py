"""Generate a USD articulation from `ActiveRagdoll.tscn` using native D6 joints.

Replaces the URDF path for the Isaac rig. Same source scene, same skeleton, different encoding -
and the difference is the whole point.

**Why not URDF.** URDF has no 3-DOF joint. Godot's `Generic6DOFJoint3D` is a ball joint with
per-axis limits, and the only way to say that in URDF is three hinges in series separated by two
massless links. That turned a 16-body skeleton into 46 bodies and 45 hinges, and PhysX solves so
poorly through genuinely zero-mass links that the converter had to invent a small fake mass for
them. The result trains fine in Isaac and does not transfer to Godot: measured, a Stand policy that
holds 98.4% standing in Isaac puts the Godot body on the floor in under two seconds, and no
actuator model, gain scale or observation randomisation closed the gap. The two engines were
simulating different machines.

USD has the joint type URDF lacks. Verified by `isaac_lab/scripts/probe_d6.py`: a D6 joint with
drives on all three rotational axes stays inside the reduced-coordinate articulation, reports three
DOF, and tracks a commanded target exactly. So this emits 16 bodies and 15 joints - Godot's actual
mechanism.

**The DOF count is unchanged.** 15 joints x 3 rotational axes = 45, exactly what the URDF rig
produced, so the 143-float observation and 36-float action contract carry over untouched. Only the
rig and the DOF ORDERING change.

**Axis handling.** The Godot-to-USD rotation is carried on the JOINT frames, not the body frames.
Bodies stay conventionally oriented (USD Z-up), so everything on the Isaac side that reads "up" as
component 2 - projected gravity, the upright reward, the fall termination, the observation's
gravity slice - keeps working untouched. Each joint's `localRot0`/`localRot1` carries the rotation
instead, which aligns its rotX/rotY/rotZ with Godot's local x/y/z, so the limits copy across
verbatim with no sign conversion.

Putting the rotation on the bodies instead was tried first and is subtly wrong: it makes each body
frame equal Godot's frame, and then `projected_gravity_b` reads (0,-1,0) rather than (0,0,-1), so
`upright` computes as exactly 0.0 for a body that is standing perfectly. Nothing errors; the reward
just silently measures the wrong axis.

That is a deliberate departure from the URDF converter, which mapped positions with
`(x,y,z) -> (-z, +x, y)`. That matrix has determinant -1: it is a REFLECTION, so the URDF dummy was
a mirror image of the Godot one, and rotations about the mapped axes came out inverted. The
`reversed` flag and the negated-and-swapped roll limits in the old rig were compensating for a
mirror that should never have been introduced. The rotation used here, `(x,y,z) -> (-z, -x, y)`,
has determinant +1 and needs no compensation anywhere.

    python isaac_lab/scripts/build_d6_usd.py

Writes `isaac_lab/assets/dummy_d6.usd` and `isaac_lab/assets/dummy_d6_rig.json`.

Lives here rather than in `tools/` because `pxr` only exists once Isaac Sim's runtime has
bootstrapped - `tools/tscn_to_urdf.py` stays pure text generation and is importable anywhere, which
is why the scene PARSING is reused from it rather than duplicated.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# **Only boot Kit if `pxr` is not already importable.** This script writes a USD and needs nothing
# else from Isaac Sim, but `AppLauncher` requires `EXP_PATH` and so fails outright in the Newton
# environment (env_isaaclab3), which has `pxr` standalone. Booting Kit for a file write also costs
# ~40 s. Falling back keeps one script usable from both environments.
simulation_app = None
PhysxSchema = None
try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

    try:
        from pxr import PhysxSchema  # noqa: E402
    except ImportError:
        # PhysX-specific extension schema, absent outside Isaac Sim. The Newton backend does not
        # consume it and `UsdPhysics.ArticulationRootAPI` below is the standard marker, so the USD
        # is still complete for this pipeline. Warned rather than silently dropped.
        print("[build_d6_usd] PhysxSchema unavailable; skipping PhysxArticulationAPI "
              "(not used by the Newton backend).")
except ImportError:  # pragma: no cover - only on an installation without standalone USD
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from tscn_to_urdf import CONTACT_BONES, CONTROLLED_BONES, Body, Joint, Shape, parse_scene  # noqa: E402
SCENE_PATH = PROJECT_ROOT / "Scenes" / "ActiveRagdoll.tscn"
OUTPUT_DIR = PROJECT_ROOT / "isaac_lab" / "assets"
ROOT_PATH = "/p4f_dummy"

# Joint velocity ceiling, matching the URDF rig so the two are comparable.
JOINT_VELOCITY_LIMIT = 20.0

# Rotational axes of a D6 joint, in the order their DOF are reported.
AXES = ("rotX", "rotY", "rotZ")
GODOT_AXES = ("x", "y", "z")


def to_usd(p: tuple[float, float, float]) -> Gf.Vec3f:
    """Godot (X right, Y up, Z back) -> USD (X forward, Y left, Z up).

    A proper rotation, determinant +1. The URDF converter used `(-z, +x, y)`, determinant -1, which
    mirrored the body - see the module docstring.
    """
    return Gf.Vec3f(-p[2], -p[0], p[1])


def joint_frame_rotation() -> Gf.Quatf:
    """The Godot-to-USD rotation as a quaternion, applied to every JOINT frame.

    This is what makes a joint's rotX/rotY/rotZ mean Godot's local x/y/z, so the per-axis limits
    from the scene transfer with no permutation and no sign flips.
    """
    # USD uses the row-vector convention (v' = v * M), so the ROWS are the images of the basis
    # vectors: godot X -> usd -Y, godot Y -> usd +Z, godot Z -> usd -X. Determinant +1.
    m = Gf.Matrix3d(
        0.0, -1.0, 0.0,
        0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
    )
    assert math.isclose(m.GetDeterminant(), 1.0, abs_tol=1e-9), "frame map must be a rotation"
    q = Gf.Matrix4d().SetRotate(m).ExtractRotationQuat()
    return Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(x) for x in q.GetImaginary()]))


def add_shape(stage: Usd.Stage, parent_path: str, shape: Shape | None) -> None:
    """The collider, permuted into USD axes.

    The rotation lives on the JOINT frames, not the bodies, so a body's own frame is standard USD
    and its collider has to be expressed there: `to_usd` sends Godot (x, y, z) to (-z, -x, y), so a
    box's extents permute the same way and a Godot capsule running along local +Y runs along USD
    +Z. Sign is irrelevant for extents; only the axis assignment matters.
    """
    if shape is None:
        return

    path = f"{parent_path}/collision"
    if shape.kind == "box":
        assert shape.size is not None
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
        sx, sy, sz = shape.size
        # **A `UsdGeom.Cube` with `size = 1.0` spans -0.5..+0.5, so a scale of k gives a side of k,
        # not 2k.** Halving here made every BOX collider exactly half its Godot size - measured
        # 2026-09-05: foot (0.22, 0.12, 0.08) authored, (0.11, 0.06, 0.04) in the USD, and the same
        # factor on the pelvis, chest and hands. Capsules and spheres were unaffected because they
        # take radius and height directly, which is why it survived every rig check: the limbs were
        # right and only the four boxes were wrong.
        #
        # The foot is the one that matters. Isaac trained on a support polygon half as long
        # fore-aft as Godot's, which is the single most important parameter a biped balances on.
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(sz, sx, sy))
    elif shape.kind == "sphere":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(shape.radius)
    elif shape.kind == "capsule":
        prim = UsdGeom.Capsule.Define(stage, path)
        prim.CreateRadiusAttr(shape.radius)
        # Godot's `height` spans the whole capsule including both caps; USD's is the cylinder only.
        prim.CreateHeightAttr(max(shape.height - 2.0 * shape.radius, 1e-4))
        # Godot capsules run along local +Y, which is USD +Z under to_usd.
        prim.CreateAxisAttr("Z")
    else:
        raise ValueError(f"unhandled shape kind {shape.kind!r}")

    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())


def add_body(stage: Usd.Stage, body: Body) -> str:
    path = f"{ROOT_PATH}/{body.name}"
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in to_usd(body.pos)]))

    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(body.mass)

    add_shape(stage, path, body.shape)
    return path


def add_joint(stage: Usd.Stage, joint: Joint, bodies: dict[str, Body], rig: dict) -> None:
    """One D6 joint: translations locked, three rotations limited and driven.

    The direct analogue of `Generic6DOFJoint3D`, which is what URDF could not express.
    """
    child = bodies[joint.child]
    parent = bodies[joint.parent]
    path = f"{ROOT_PATH}/joint_{joint.child}"

    prim = UsdPhysics.Joint.Define(stage, path)
    prim.CreateBody0Rel().SetTargets([f"{ROOT_PATH}/{joint.parent}"])
    prim.CreateBody1Rel().SetTargets([f"{ROOT_PATH}/{joint.child}"])

    # Anchor both frames at the shared pivot, expressed in each body's own local axes. Both bodies
    # carry the same rest orientation, so this is a plain difference of Godot world positions.
    pivot = joint.pivot
    rot = joint_frame_rotation()
    prim.CreateLocalPos0Attr().Set(to_usd(tuple(pivot[i] - parent.pos[i] for i in range(3))))
    prim.CreateLocalPos1Attr().Set(to_usd(tuple(pivot[i] - child.pos[i] for i in range(3))))
    # Both frames carry the Godot-to-USD rotation, so rotX/rotY/rotZ are Godot's local x/y/z.
    prim.CreateLocalRot0Attr().Set(rot)
    prim.CreateLocalRot1Attr().Set(rot)

    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim.GetPrim(), axis)
        # USD encodes a locked axis as low > high.
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)

    for usd_axis, godot_axis in zip(AXES, GODOT_AXES):
        bounds = joint.limits.get(godot_axis)
        lower, upper = bounds if bounds is not None else (-math.pi, math.pi)

        limit = UsdPhysics.LimitAPI.Apply(prim.GetPrim(), usd_axis)
        limit.CreateLowAttr(math.degrees(lower))
        limit.CreateHighAttr(math.degrees(upper))

        drive = UsdPhysics.DriveAPI.Apply(prim.GetPrim(), usd_axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(child.kp)
        drive.CreateDampingAttr(child.kd)
        drive.CreateMaxForceAttr(child.max_torque)

        # PhysX names the DOF of a multi-axis joint `<joint>:0/:1/:2`, in rotX/rotY/rotZ order -
        # confirmed by probe_d6.py. Recorded so nothing downstream has to guess.
        dof_name = f"joint_{joint.child}:{AXES.index(usd_axis)}"
        rig["joints"][dof_name] = {
            "bone": joint.child,
            "godot_axis": godot_axis,
            # Kept for schema compatibility with the URDF rig contract. Always false here: the
            # frames coincide, so no axis is inverted.
            "reversed": False,
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "effort": child.max_torque,
            "stiffness": child.kp,
            "damping": child.kd,
            "actuated": joint.child in CONTROLLED_BONES,
        }
        target = rig["actuated_joints"] if joint.child in CONTROLLED_BONES else rig["passive_joints"]
        target.append(dof_name)


def build(bodies: dict[str, Body], joints: list[Joint]) -> tuple[Usd.Stage, dict]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    stage.SetDefaultPrim(root.GetPrim())

    rig: dict = {
        "source_scene": str(SCENE_PATH.relative_to(PROJECT_ROOT)),
        "encoding": "usd-d6",
        "root_link": next(b.name for b in bodies.values() if b.parent is None),
        "controlled_bones": CONTROLLED_BONES,
        "contact_bones": CONTACT_BONES,
        "actuated_joints": [],
        "passive_joints": [],
        "joints": {},
        "total_mass": 0.0,
    }

    # Emit in the same tree order the URDF converter used, so the two rigs are comparable.
    joint_by_child = {j.child: j for j in joints}
    order: list[Body] = []
    queue = [b for b in bodies.values() if b.parent is None]
    while queue:
        body = queue.pop(0)
        order.append(body)
        queue.extend(b for b in bodies.values() if b.parent == body.name)

    for body in order:
        add_body(stage, body)
        rig["total_mass"] += body.mass

    root_prim = stage.GetPrimAtPath(f"{ROOT_PATH}/{rig['root_link']}")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    if PhysxSchema is not None:
        PhysxSchema.PhysxArticulationAPI.Apply(root_prim)

    for body in order:
        if body.name in joint_by_child:
            add_joint(stage, joint_by_child[body.name], bodies, rig)

    rig["total_mass"] = round(rig["total_mass"], 4)
    return stage, rig


# **`physx_dof_order` is NOT written by this script and a rebuild silently drops it.**
# The key records the DOF ordering PhysX assigned at USD load and is read by Godot's
# `IsaacRigContract`; it was produced by the 2.3.2 tooling, not here. Rebuilding on 2026-09-05
# removed it from `dummy_d6_rig.json` and it had to be merged back from a backup by hand.
# If you rebuild, check for it afterwards - everything else round-trips byte-identically
# (all 45 joints, masses, limits), so this is the only field at risk.
def main() -> None:
    bodies, joints = parse_scene(SCENE_PATH)
    stage, rig = build(bodies, joints)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    usd_path = OUTPUT_DIR / "dummy_d6.usd"
    stage.GetRootLayer().Export(str(usd_path))
    (OUTPUT_DIR / "dummy_d6_rig.json").write_text(json.dumps(rig, indent=2) + "\n", encoding="utf-8")

    actuated = len(rig["actuated_joints"])
    passive = len(rig["passive_joints"])
    print(f"bodies    : {len(bodies)} (no dummy links - the URDF rig needed 46 for this skeleton)")
    print(f"joints    : {len(joints)} D6, {actuated + passive} DOF ({actuated} actuated, {passive} passive)")
    print(f"mass      : {rig['total_mass']:.2f} kg")
    print(f"written   : {usd_path.relative_to(PROJECT_ROOT)}")
    print(f"            {(OUTPUT_DIR / 'dummy_d6_rig.json').relative_to(PROJECT_ROOT)}")

    # The old rig needed this assertion because its reflection could silently swap L and R. This
    # one cannot mirror, so the check is a regression guard rather than a correctness proof.
    for left, right in (("UpperArm_L", "UpperArm_R"), ("Thigh_L", "Thigh_R")):
        lo = rig["joints"][f"joint_{left}:2"]
        ro = rig["joints"][f"joint_{right}:2"]
        assert math.isclose(lo["lower"], -ro["upper"], abs_tol=1e-6), f"{left}/{right} roll not mirrored"
    print("[OK] shoulder and hip roll limits stay mirror-symmetric across L/R")


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
    # Isaac Sim's teardown deadlocks on Windows after USD work; everything is written by here.
    os._exit(0)
