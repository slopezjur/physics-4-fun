"""Drop the dummy into the scene, hold the rest pose, and print what happens.

A humanoid standing with its feet flat and its centre of mass inside the support polygon should
stay up indefinitely under a PD drive holding that pose. If it does not, the problem is in the
asset or the actuators, not in the policy - and no amount of training will find its way around it.
This prints the telemetry needed to tell those apart: heights, tilt, contact forces and how far the
joints have drifted from the pose they are being commanded to hold.

    python isaac_lab/scripts/diagnose_rest_pose.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from p4f_isaac.tasks.stand.stand_env import StandEnv  # noqa: E402
from p4f_isaac.tasks.stand.stand_env_cfg import StandEnvCfg  # noqa: E402


def main() -> None:
    cfg = StandEnvCfg()
    cfg.scene.num_envs = 4
    env = StandEnv(cfg)
    robot = env.unwrapped.robot

    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

    head_id = robot.find_bodies("Head")[0][0]
    foot_ids = robot.find_bodies(["Foot_L", "Foot_R"])[0]

    print(f"\n{'t(s)':>6}{'pelvis_z':>10}{'head_z':>9}{'upright':>9}"
          f"{'|jpos|':>9}{'footL_z':>9}{'footR_z':>9}{'contactN':>10}")
    steps = int(args.seconds / env.unwrapped.step_dt)
    for i in range(steps):
        env.step(action)
        if i % 6 and i != steps - 1:
            continue
        data = robot.data
        t = i * env.unwrapped.step_dt
        pelvis_z = data.root_pos_w[0, 2].item() - env.unwrapped.scene.env_origins[0, 2].item()
        head_z = data.body_com_pos_w[0, head_id, 2].item() - env.unwrapped.scene.env_origins[0, 2].item()
        upright = -data.projected_gravity_b[0, 2].item()
        # How far the joints have been pushed off the pose the drives are commanding. A large
        # value with the body still upright means the actuators are too weak to hold it.
        jerr = data.joint_pos[0].abs().max().item()
        foot_z = [data.body_com_pos_w[0, f, 2].item() - env.unwrapped.scene.env_origins[0, 2].item() for f in foot_ids]
        forces = env.unwrapped.contact_sensor.data.net_forces_w[0]
        contact_n = forces.norm(dim=-1).sum().item()
        print(f"{t:>6.2f}{pelvis_z:>10.3f}{head_z:>9.3f}{upright:>9.3f}"
              f"{jerr:>9.3f}{foot_z[0]:>9.3f}{foot_z[1]:>9.3f}{contact_n:>10.1f}")

    print("\nexpected at rest: pelvis_z 0.820, head_z 1.540, upright 1.000, foot_z 0.040")
    print(f"total mass {robot.root_physx_view.get_masses()[0].sum():.2f} kg "
          f"=> weight {robot.root_physx_view.get_masses()[0].sum() * 9.81:.0f} N should appear as contact")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
