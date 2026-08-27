"""Print every body's rest position for whichever rig is loaded.

Forward kinematics, not the authored file, decides where a body actually ends up: Isaac places the
articulation ROOT at `init_state.pos` and derives everything else from the joint frames with all
joint angles at zero. So a USD whose bodies are authored in the right places can still assemble
into a heap if a joint's local anchors are wrong, and nothing about that shows up as an error.

    P4F_RIG=d6 python isaac_lab/scripts/inspect_rig.py

Compare against the Godot scene: pelvis 0.82, head 1.54, feet 0.04.
"""

from __future__ import annotations

import os
import pathlib
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_isaac.tasks  # noqa: F401, E402
from p4f_isaac.tasks.stand.stand_env_cfg import StandEnvCfg  # noqa: E402


def main() -> None:
    cfg = StandEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("P4F-Dummy-Stand-Direct-v0", cfg=cfg)
    env.reset()

    base = env.unwrapped
    robot = base.robot
    origin = base.scene.env_origins[0]

    print("\n" + "=" * 58)
    print(f"  RIG INSPECTION - {os.environ.get('P4F_RIG', 'urdf')}")
    print("=" * 58)
    print(f"bodies {robot.num_bodies}, joints {robot.num_joints}, "
          f"mass {robot.root_physx_view.get_masses().sum():.2f} kg")
    print(f"\n{'body':<16}{'x':>9}{'y':>9}{'z (up)':>10}")

    pos = robot.data.body_pos_w[0] - origin
    for name, p in sorted(zip(robot.body_names, pos.tolist()), key=lambda r: -r[1][2]):
        print(f"{name:<16}{p[0]:>9.3f}{p[1]:>9.3f}{p[2]:>10.3f}")

    # Record the DOF ordering into the rig contract.
    #
    # Forward kinematics is not the only thing the authored file fails to determine: PhysX assigns
    # its own DOF order at import, and the observation indexes `data.joint_pos` directly, so slices
    # [10:55] and [55:100] follow THIS order and not the declaration order. On the URDF rig the two
    # differed in 32 of 45 slots, which nothing recorded and nothing would have caught.
    import json

    is_d6 = os.environ.get("P4F_RIG", "").lower() == "d6"
    rig_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "assets"
        / ("dummy_d6_rig.json" if is_d6 else "dummy_rig.json")
    )
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    order = list(robot.joint_names)
    declared = list(rig["joints"])
    rig["physx_dof_order"] = order
    rig_path.write_text(json.dumps(rig, indent=2) + "\n", encoding="utf-8")

    differing = sum(1 for a, b in zip(order, declared) if a != b)
    print(f"\nphysx_dof_order recorded in {rig_path.name}")
    print(f"  differs from declaration order in {differing} of {len(order)} slots")
    print(f"  first 9: {order[:9]}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
