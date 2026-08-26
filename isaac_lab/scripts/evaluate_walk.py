"""Score a Walk checkpoint on whether it tracks the command and whether it actually steps.

Two things can look like walking and are not, and mean reward separates neither:

* **Sliding.** Velocity tracking alone is satisfiable by shuffling with both feet in permanent
  ground contact. The reward pays out; the gait is wrong; it will not survive a solver change.
* **Saturation.** A policy whose raw output has drifted outside the clip range still tracks well
  in-sim - the environment clamps - but has no graded control left. This is what the first Stand
  run did, and it was invisible to every quality metric until the exported network was run
  directly.

So this reports gait and output range next to the tracking error, and fixes the command rather
than sampling it, so the number means "how well does it hold 0.6 m/s" instead of averaging over an
unknown mix of easy and hard commands.

    python isaac_lab/scripts/evaluate_walk.py --checkpoint <abs path to model_*.pt> --speed 0.6
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--speed", type=float, default=0.6, help="Commanded forward speed (m/s).")
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
from p4f_isaac.tasks.walk.agents.rsl_rl_ppo_cfg import WalkPPORunnerCfg  # noqa: E402
from p4f_isaac.tasks.walk.walk_env_cfg import WalkEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = WalkEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = WalkPPORunnerCfg()
    env = gym.make("P4F-Dummy-Walk-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.robot
    n, dev = base.num_envs, base.device

    speed_err = torch.zeros(n, device=dev)
    speed_sum = torch.zeros(n, device=dev)
    upright_sum = torch.zeros(n, device=dev)
    head_sum = torch.zeros(n, device=dev)
    steps_taken = torch.zeros(n, device=dev)
    air_time_sum = torch.zeros(n, device=dev)
    both_feet_down = torch.zeros(n, device=dev)
    no_feet_down = torch.zeros(n, device=dev)
    alive = torch.zeros(n, device=dev)
    ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
    raw_max = torch.zeros((), device=dev)
    raw_sat = torch.zeros((), device=dev)
    raw_count = 0.0

    obs = env.get_observations()
    total = int(args.episodes * base.max_episode_length)
    for _ in range(total):
        with torch.inference_mode():
            action = policy(obs)
            # Overwrite the sampled command every step: the evaluation asks a specific question -
            # can it hold this speed - not "how did it do on whatever it happened to be given".
            base._command[:, 0] = args.speed
            base._command[:, 1] = 0.0
            base._command[:, 2] = 0.0
            obs, _, dones, _ = env.step(action)

        raw_max = torch.maximum(raw_max, action.abs().max())
        raw_sat += (action.abs() > 1.0).float().mean()
        raw_count += 1.0

        data = robot.data
        fwd = data.root_lin_vel_b[:, 0]
        speed_sum += fwd
        speed_err += (fwd - args.speed).abs()
        upright_sum += -data.projected_gravity_b[:, 2]
        head_sum += data.body_com_pos_w[:, base._head_id[0], 2] - base.scene.env_origins[:, 2]

        first_contact = base.contact_sensor.compute_first_contact(base.step_dt)[:, base._feet_ids]
        steps_taken += first_contact.sum(dim=1)
        air_time_sum += (base.contact_sensor.data.last_air_time[:, base._feet_ids] * first_contact).sum(dim=1)
        contacts = base.contact_sensor.data.net_forces_w[:, base._feet_ids, :].norm(dim=-1) > 1.0
        both_feet_down += contacts.all(dim=1).float()
        # Flight phase: neither foot loaded. This is the definition of running as opposed to
        # walking - a walk always has at least one foot down, however fast it gets - so without
        # measuring it, "it runs" is an assertion rather than a result.
        no_feet_down += (~contacts.any(dim=1)).float()

        alive += 1.0
        ever_fell |= base._fell

    k = alive.clamp(min=1.0)
    duration = alive.mean().item() * base.step_dt
    steps_per_s = (steps_taken / k / base.step_dt).mean().item()
    mean_air = (air_time_sum / steps_taken.clamp(min=1.0)).mean().item()

    print("\n" + "=" * 62)
    print(f"  WALK EVALUATION  -  {pathlib.Path(args.checkpoint).name}")
    print(f"  commanded forward speed: {args.speed} m/s")
    print("=" * 62)
    print(f"  ever fell           : {ever_fell.float().mean().item() * 100:6.1f} %")
    print(f"  mean forward speed  : {(speed_sum / k).mean().item():6.3f} m/s  (target {args.speed})")
    print(f"  mean |speed error|  : {(speed_err / k).mean().item():6.3f} m/s")
    print("-" * 62)
    print(f"  steps per second    : {steps_per_s:6.2f}     <- 0 means sliding, not walking")
    print(f"  mean air time/step  : {mean_air:6.3f} s")
    print(f"  both feet grounded  : {(both_feet_down / k).mean().item() * 100:6.1f} % of the time")
    print(f"  FLIGHT (no foot down): {(no_feet_down / k).mean().item() * 100:6.1f} % of the time"
          f"   <- >0 means running, not walking")
    print("-" * 62)
    print(f"  mean uprightness    : {(upright_sum / k).mean().item():6.3f}     (1.0 = vertical)")
    print(f"  mean head height    : {(head_sum / k).mean().item():6.3f} m   (rest 1.540)")
    print(f"  raw |action| max    : {raw_max.item():6.2f}     <- must be near 1, not 10+")
    print(f"  actions saturated   : {(raw_sat / raw_count).item() * 100:6.1f} %")
    print("=" * 62)

    if steps_per_s < 0.5:
        print("  WARNING: almost no touchdowns - this is sliding, not walking.")
    if raw_max.item() > 3.0:
        print("  WARNING: policy output far outside the clip range - degenerate, do not export.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
