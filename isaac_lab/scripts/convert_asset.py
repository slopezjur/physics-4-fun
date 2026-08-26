"""Bake `assets/dummy.urdf` into a USD articulation Isaac Lab can spawn.

Run after `tools/tscn_to_urdf.py`, and re-run whenever the Godot scene changes:

    python isaac_lab/scripts/convert_asset.py

Isaac Lab ships `scripts/tools/convert_urdf.py`, but it exposes neither of the two settings this
rig actually depends on, so the conversion lives here instead:

* **Cylinders must become capsules.** The Godot limbs are `CapsuleShape3D`, and URDF has no
  capsule primitive, so the converter emits cylinders. Imported as literal cylinders they gain
  sharp rims, and every knee-on-floor and foot-edge contact - the entire get-up problem - changes
  character. `replace_cylinders_with_capsules` restores the intended shape at import.

* **Gains are per joint, not global.** The scene tunes stiffness across a 30x range (the ankle at
  1200 and the knee at 1800 against the hand at 60); collapsing that to one number would discard
  the tuning that makes the procedural controller work, and it is the closest thing to a prior we
  have for what the learned policy needs.

Self-collision is left OFF deliberately. The rest pose has the arms hanging inside the torso
capsule's swept volume, so enabling it makes the articulation start interpenetrating and PhysX
spends the first frames pushing the body apart. It belongs in the task config once the limits are
confirmed to keep the limbs out of the torso, not in the asset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=ASSETS_DIR / "dummy.urdf")
parser.add_argument("--rig", type=Path, default=ASSETS_DIR / "dummy_rig.json")
parser.add_argument("--output", type=Path, default=ASSETS_DIR / "dummy.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main() -> None:
    rig = json.loads(args.rig.read_text(encoding="utf-8"))

    # Per-joint gains, lifted from each bone's ActiveBone ProportionalGain / DerivativeGain. All
    # three axes of a decomposed joint inherit the bone's gains, matching Godot, where one PD
    # controller drives the whole 3-DOF joint.
    stiffness = {name: spec["stiffness"] for name, spec in rig["joints"].items()}
    damping = {name: spec["damping"] for name, spec in rig["joints"].items()}

    cfg = UrdfConverterCfg(
        asset_path=str(args.urdf),
        usd_dir=str(args.output.parent),
        usd_file_name=args.output.name,
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
        ),
        force_usd_conversion=True,
    )

    converter = UrdfConverter(cfg)
    print(f"[OK] wrote {converter.usd_path}")

    # Read the articulation back and check it against the rig contract rather than trusting the
    # import. A URDF that parses can still land a different DOF count or drop a link, and finding
    # that out here costs seconds instead of a training run.
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))
    stage_utils.add_reference_to_stage(converter.usd_path, "/World/Dummy")
    # Gains are left as None so the actuator adopts whatever the USD drives carry; this check is
    # about structure, and restating them here would only prove they match themselves.
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Dummy",
            spawn=None,
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        )
    )
    sim.reset()

    expected_dof = len(rig["joints"])
    print(f"bodies : {robot.num_bodies}")
    print(f"joints : {robot.num_joints} (expected {expected_dof})")
    print(f"mass   : {robot.root_physx_view.get_masses().sum():.2f} kg (expected {rig['total_mass']:.2f})")

    missing = [name for name in rig["joints"] if name not in robot.joint_names]
    if missing:
        raise SystemExit(f"[FAIL] joints missing from the imported articulation: {missing}")
    print("[OK] every joint in the rig contract is present in the USD")


if __name__ == "__main__":
    main()
    simulation_app.close()
    # Isaac Sim's shutdown hangs on Windows after a converter run - the Kit plugin unload deadlocks
    # ("USD stage detach not called", "Recursive unloadAllPlugins") and the process never returns.
    # Everything above has already been written and verified by this point, so leaving through
    # os._exit skips the stuck teardown rather than papering over unfinished work.
    os._exit(0)
