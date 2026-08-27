"""Which half of Newton's root twist is linear, and which is angular?

`joint_qd`'s root FREE joint carries six entries. Whether they are ordered (linear, angular) or
(angular, linear) decides whether a commanded push shoves the body or spins it, and **both
orderings run without error**. warp's spatial vectors are conventionally (angular, linear) while
most robotics APIs are the reverse, so this is exactly the kind of thing the project has already
been burned by twice - the root quaternion silently changed wxyz to xyzw between Isaac Lab
releases, and `physx_dof_order` disagreed with Newton's in 42 of 45 slots.

So it is measured causally rather than read out of a docstring: write a pure +X value into the
first three entries, step once, and see whether the pelvis translates or rotates.

    python isaac_lab_3/scripts/probe_twist.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium

PROBE_SPEED = 2.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--wrench",
        action="store_true",
        help="Probe State.body_f's angular/linear split instead of the root twist's.",
    )
    args = p.parse_args()

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 8
    env_cfg.sim.device = args.device
    # The push under test must be the only source of motion.
    env_cfg.push_velocity = 0.0

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()

    pelvis = base.robot.body_names.index("Pelvis")
    ids = torch.arange(base.num_envs, device=base.device)

    if args.wrench:
        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager

        # A large torque so one step of integration is unambiguous against gravity.
        for label, slot in (("body_f [0:3]", 0), ("body_f [3:6]", 3)):
            env.reset()
            for _ in range(3):
                state = NewtonManager._state_0
                f = wp.to_torch(state.body_f).view(base.num_envs, -1, 6)
                f[:, pelvis, slot] += 400.0
                env.step(torch.zeros(base.num_envs, base.cfg.action_space, device=base.device))
            ang = base.robot.data.body_ang_vel_w.torch[:, pelvis].mean(dim=0)
            lin = base.robot.data.body_com_lin_vel_w.torch[:, pelvis].mean(dim=0)
            print(f"  wrote 400 into {label:>12}  ->  pelvis ang ({ang[0]:+.3f}, {ang[1]:+.3f}, "
                  f"{ang[2]:+.3f})  lin ({lin[0]:+.3f}, {lin[1]:+.3f}, {lin[2]:+.3f})")
        print("  The half that moved `ang` is the ANGULAR half. Set WRENCH_ANGULAR accordingly.")
        env.close()
        return

    for label, slot in (("entries [0:3]", 0), ("entries [3:6]", 3)):
        twist = torch.zeros(base.num_envs, 6, device=base.device)
        twist[:, slot] = PROBE_SPEED  # pure +X in whichever half is under test

        joint_pos = base._default_joint_pos[ids].clone()
        root_pose = base.robot.data.default_root_pose.torch[ids].clone()
        root_pose[:, :3] += base.scene.env_origins[ids]
        base.rig.reset_to(
            env_ids=ids,
            root_pos=root_pose[:, :3],
            root_quat_xyzw=root_pose[:, 3:7],
            joint_pos=joint_pos,
            joint_vel=torch.zeros_like(joint_pos),
            root_vel=twist,
        )

        # One step, so the written state has propagated into the live per-body buffers without the
        # body having had time to fall and confuse the reading.
        env.step(torch.zeros(base.num_envs, base.cfg.action_space, device=base.device))
        lin = base.robot.data.body_com_lin_vel_w.torch[:, pelvis].mean(dim=0)
        ang = base.robot.data.body_ang_vel_w.torch[:, pelvis].mean(dim=0)
        print(f"  wrote {PROBE_SPEED} into {label:>13}  ->  "
              f"pelvis lin ({lin[0]:+.3f}, {lin[1]:+.3f}, {lin[2]:+.3f})  "
              f"ang ({ang[0]:+.3f}, {ang[1]:+.3f}, {ang[2]:+.3f})")

    print("\n  The half that moved `lin` is the LINEAR half. Set ROOT_TWIST_LINEAR accordingly.")
    env.close()


if __name__ == "__main__":
    main()
