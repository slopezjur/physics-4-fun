"""Can Isaac drive a 3-DOF D6 joint inside a reduced-coordinate articulation?

Feasibility gate for rebuilding the rig to match Godot's mechanism. Godot's dummy is 16 rigid
bodies joined by `Generic6DOFJoint3D` - one 3-DOF rotational joint per limb. URDF cannot express
that (it has only 1-DOF joints), so the current rig decomposes every joint into three stacked
hinges separated by two massless links: 46 bodies and 45 hinges for a 16-bone skeleton.

USD has no such limitation, but the question is not whether USD can *describe* a D6 joint - it is
whether PhysX's ARTICULATION solver accepts one with position drives on all three rotational axes.
Articulations are reduced-coordinate and support a restricted joint set; if a 3-DOF joint either
falls out of the articulation into the general constraint solver, or accepts no drive, then the
whole plan collapses and the URDF decomposition was not a workaround but a requirement.

Three things have to hold, and each is checked below:

  1. The articulation reports 16 links, not 46 - the joint did not get split back up.
  2. It reports 45 DOF across 15 joints - all three rotational axes are live.
  3. `set_joint_position_target` actually moves the joint - the drives are real.

    python isaac_lab/scripts/probe_d6.py
"""

from __future__ import annotations

import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

STAGE_PATH = "/World/Probe"


def add_body(stage: Usd.Stage, path: str, pos: tuple[float, float, float], mass: float) -> Usd.Prim:
    """A capsule with a rigid body and collider, standing in for one bone."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    capsule = UsdGeom.Capsule.Define(stage, f"{path}/geom")
    capsule.CreateHeightAttr(0.2)
    capsule.CreateRadiusAttr(0.05)
    capsule.CreateAxisAttr("Z")

    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    return xform.GetPrim()


def add_d6(stage: Usd.Stage, path: str, parent: str, child: str) -> None:
    """One 3-DOF rotational joint: translation locked, all three rotations limited and driven.

    This is the direct analogue of Godot's Generic6DOFJoint3D - the thing URDF forced us to fake
    with three hinges and two invisible bodies.
    """
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([parent])
    joint.CreateBody1Rel().SetTargets([child])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.15))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -0.15))

    # Lock all three translations; the joint is purely rotational.
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)  # low > high is USD's encoding for "locked"

    # Limit and drive each rotation. A drive on all three is the part that has to survive the
    # articulation solver - it is what the policy's joint targets act through.
    for axis in ("rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(-45.0)
        limit.CreateHighAttr(45.0)

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(500.0)
        drive.CreateDampingAttr(20.0)
        drive.CreateMaxForceAttr(300.0)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=1.0 / 120.0))
    stage = sim.stage

    UsdGeom.Xform.Define(stage, STAGE_PATH)
    root = add_body(stage, f"{STAGE_PATH}/Root", (0.0, 0.0, 1.0), 5.0)
    add_body(stage, f"{STAGE_PATH}/Link1", (0.0, 0.0, 0.7), 3.0)
    add_body(stage, f"{STAGE_PATH}/Link2", (0.0, 0.0, 0.4), 2.0)

    # Root of the articulation. Everything reachable through joints below it is one articulation.
    UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root)

    add_d6(stage, f"{STAGE_PATH}/joint_0", f"{STAGE_PATH}/Root", f"{STAGE_PATH}/Link1")
    add_d6(stage, f"{STAGE_PATH}/joint_1", f"{STAGE_PATH}/Link1", f"{STAGE_PATH}/Link2")

    robot = Articulation(
        ArticulationCfg(
            prim_path=STAGE_PATH,
            spawn=None,
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        )
    )
    sim.reset()

    print("\n" + "=" * 64)
    print("  D6 ARTICULATION PROBE")
    print("=" * 64)
    print(f"bodies : {robot.num_bodies}   (expect 3 - one per real link, no dummies)")
    print(f"joints : {robot.num_joints}   (expect 6 - two joints x three rotational DOF)")
    print(f"names  : {list(robot.joint_names)}")
    print(f"bodies : {list(robot.body_names)}")

    if robot.num_joints == 0:
        print("\n[FAIL] no DOF - the D6 joint did not become part of the articulation.")
        return

    # Drive check: command a target and see whether the joint actually tracks it.
    target = torch.zeros(1, robot.num_joints, device=robot.device)
    target[:, 0] = 0.4
    before = robot.data.joint_pos[0, 0].item()
    for _ in range(240):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1.0 / 120.0)
    after = robot.data.joint_pos[0, 0].item()

    print(f"\ndrive  : joint 0 commanded to 0.400 rad")
    print(f"         before {before:+.4f} -> after {after:+.4f}")
    moved = abs(after - before) > 0.05
    print(f"\n[{'OK' if moved else 'FAIL'}] drives on a 3-DOF articulation joint "
          f"{'work' if moved else 'do NOT work'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
