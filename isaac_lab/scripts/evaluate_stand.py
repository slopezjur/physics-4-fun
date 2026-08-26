"""Score a checkpoint against Godot's standing criteria, not against episode length.

Mean episode length is the wrong gate and it is easy to be fooled by. Stand terminates only below
0.9 m of head height or past 70 degrees of tilt, so a policy can run a near-full episode while
crouching, wobbling, or balancing on one leg. `UprightTermination` makes exactly this point: a
policy can "lunge upward through the threshold and collapse", which momentarily satisfies a height
check while being the opposite of the intended behaviour - and the Godot Stand run scored 36% on
the strict criterion while looking healthy on coarse metrics.

So this measures the criterion that file actually defines, ported verbatim:

    head height  >= 1.35 m
    torso tilt   <= 30 deg
    CoM speed    <= 0.60 m/s
    held continuously for 1.5 s

and reports the posture and smoothness telemetry needed to tell a real stand from a survivable
one. Run before promoting a checkpoint or handing it to Walk:

    python isaac_lab/scripts/evaluate_stand.py --checkpoint <abs path to model_*.pt>
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to a model_*.pt.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2, help="Full episode windows to run.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_isaac.tasks  # noqa: F401, E402
from p4f_isaac.tasks.stand.agents.rsl_rl_ppo_cfg import StandPPORunnerCfg  # noqa: E402
from p4f_isaac.tasks.stand.stand_env_cfg import (  # noqa: E402
    STANDING_HEAD_HEIGHT,
    STANDING_MAX_SPEED,
    STANDING_TILT_DEG,
    StandEnvCfg,
)

STANDING_HOLD_SECONDS = 1.5  # UprightTermination.StandingHoldSeconds


def main() -> None:
    env_cfg = StandEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = StandPPORunnerCfg()
    env = gym.make("P4F-Dummy-Stand-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.robot
    head_id = robot.find_bodies("Head")[0][0]
    masses = robot.data.default_mass.to(base.device)          # (num_envs, num_bodies)
    total_mass = masses.sum(dim=1, keepdim=True)
    tilt_cos = math.cos(math.radians(STANDING_TILT_DEG))
    hold_steps = int(STANDING_HOLD_SECONDS / base.step_dt)

    n = base.num_envs
    dev = base.device
    hold = torch.zeros(n, device=dev)                 # consecutive qualifying steps
    succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
    head_sum = torch.zeros(n, device=dev)
    tilt_sum = torch.zeros(n, device=dev)
    speed_sum = torch.zeros(n, device=dev)
    rate_sum = torch.zeros(n, device=dev)
    effort_sum = torch.zeros(n, device=dev)
    alive_steps = torch.zeros(n, device=dev)

    obs = env.get_observations()
    steps = int(args.episodes * base.max_episode_length)
    for _ in range(steps):
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, _ = env.step(action)

        data = robot.data
        head = data.body_com_pos_w[:, head_id, 2] - base.scene.env_origins[:, 2]
        upright = -data.projected_gravity_b[:, 2]
        # Whole-body CoM speed, mass-weighted: the criterion is about the body settling, and the
        # pelvis alone can be near-stationary while the limbs are still swinging.
        com_vel = (data.body_com_vel_w[:, :, :3] * masses.unsqueeze(-1)).sum(dim=1) / total_mass
        speed = com_vel.norm(dim=-1)

        qualifies = (head >= STANDING_HEAD_HEIGHT) & (upright >= tilt_cos) & (speed <= STANDING_MAX_SPEED)
        hold = torch.where(qualifies, hold + 1.0, torch.zeros_like(hold))
        succeeded |= hold >= hold_steps

        head_sum += head
        tilt_sum += torch.rad2deg(torch.acos(upright.clamp(-1.0, 1.0)))
        speed_sum += speed
        rate_sum += (base._action - base._previous_action).abs().mean(dim=1)
        effort_sum += (data.applied_torque[:, base._actuated_ids] / base._effort_limit).abs().mean(dim=1)
        alive_steps += 1.0

        ever_fell |= base._fell
        hold = torch.where(dones.bool(), torch.zeros_like(hold), hold)

    k = alive_steps.clamp(min=1.0)
    print("\n" + "=" * 62)
    print(f"  STAND EVALUATION  -  {pathlib.Path(args.checkpoint).name}")
    print("=" * 62)
    print(f"  criterion: head >= {STANDING_HEAD_HEIGHT} m, tilt <= {STANDING_TILT_DEG} deg,")
    print(f"             CoM speed <= {STANDING_MAX_SPEED} m/s, held {STANDING_HOLD_SECONDS} s")
    print("-" * 62)
    print(f"  standing success   : {succeeded.float().mean().item() * 100:6.1f} %   <- the gate")
    print(f"  ever fell          : {ever_fell.float().mean().item() * 100:6.1f} %")
    print("-" * 62)
    print(f"  mean head height   : {(head_sum / k).mean().item():6.3f} m   (rest 1.540)")
    print(f"  mean torso tilt    : {(tilt_sum / k).mean().item():6.2f} deg (limit {STANDING_TILT_DEG})")
    print(f"  mean CoM speed     : {(speed_sum / k).mean().item():6.3f} m/s (limit {STANDING_MAX_SPEED})")
    print(f"  mean |action rate| : {(rate_sum / k).mean().item():6.4f}     (smoothness; lower transfers better)")
    print(f"  mean effort used   : {(effort_sum / k).mean().item() * 100:6.1f} %   of per-joint torque limit")
    print("=" * 62)

    # A crouch is the failure this is most likely to miss: it satisfies tilt and speed while
    # sitting well under the standing height, and reads as a long episode.
    mean_head = (head_sum / k).mean().item()
    if mean_head < STANDING_HEAD_HEIGHT:
        print(f"  WARNING: mean head height {mean_head:.3f} is below the standing threshold - the")
        print("           policy is surviving in a crouch rather than standing.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
