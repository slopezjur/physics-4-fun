"""Sweep impulse magnitude and report what the policy survives.

A single "success rate" for a perturbation task is close to meaningless, because it averages over
whatever impulse distribution training happened to use. What matters is the *curve*: how hard a
shove can this policy absorb, and where does it fall over. That also makes it directly comparable
to the arithmetic in `BallGun.cs`, which works in impulse and tipping energy rather than in
success percentages.

Reference points from that file, for this rig (80.6 kg, CoM 0.840 m, I about the toe line
69.5 kg.m^2, chest impact at 1.250 m; 6.74 J tips a passive body over its toe edge):

    0.75 kg @ 6 m/s  =  4.5 N.s   <- the Godot BallGun shot
    1.5  kg @ 6 m/s  =  9.0 N.s
                       15.0 N.s   <- training ceiling used here

Each row forces every environment to the same impulse, so the number means "of bodies hit this
hard, how many stayed up".

    python isaac_lab/scripts/evaluate_perturb.py --checkpoint <abs path to model_*.pt>
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--impulses",
    type=float,
    nargs="+",
    default=[0.0, 4.5, 9.0, 15.0, 22.0, 30.0],
    help="Impulse magnitudes (N.s) to sweep. 4.5 is the Godot BallGun shot.",
)
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
from p4f_isaac.tasks.perturb.agents.rsl_rl_ppo_cfg import PerturbPPORunnerCfg  # noqa: E402
from p4f_isaac.tasks.perturb.perturb_env_cfg import PerturbEnvCfg  # noqa: E402
from p4f_isaac.tasks.stand.stand_env_cfg import STANDING_HEAD_HEIGHT, STANDING_TILT_DEG  # noqa: E402


def main() -> None:
    env_cfg = PerturbEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    agent_cfg = PerturbPPORunnerCfg()
    env = gym.make("P4F-Dummy-Perturb-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.robot
    n, dev = base.num_envs, base.device
    tilt_cos = math.cos(math.radians(STANDING_TILT_DEG))
    episode_steps = int(base.max_episode_length)

    print("\n" + "=" * 72)
    print(f"  PERTURBATION SWEEP  -  {pathlib.Path(args.checkpoint).name}")
    print(f"  random target bone, random compass direction; {n} bodies per row")
    print("=" * 72)
    print(f"{'impulse':>9}{'x BallGun':>11}{'stayed up':>12}{'still standing':>16}{'max tilt':>11}{'recovery':>11}")
    print(f"{'(N.s)':>9}{'':>11}{'':>12}{'at episode end':>16}{'(deg)':>11}{'(s)':>11}")
    print("-" * 72)

    obs = env.get_observations()
    for impulse in args.impulses:
        # No explicit reset between rungs: reset() through the gymnasium wrapper stack is not valid
        # mid-episode, and it is not needed. Environments auto-reset on termination, and each rung
        # runs two full episode windows, so every body is hit at this magnitude at least once with
        # a clean start. The magnitude is re-forced every step because _resample_commands redraws
        # it on each reset.
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        max_tilt = torch.zeros(n, device=dev)
        recovered_step = torch.full((n,), float("nan"), device=dev)
        hit_step = torch.zeros(n, device=dev)

        for step in range(2 * episode_steps):
            with torch.inference_mode():
                # Direction is whatever the env drew; only the magnitude is pinned.
                mag = base._push_impulse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                base._push_impulse = base._push_impulse / mag * impulse
                action = policy(obs)
                obs, _, _, _ = env.step(action)

            data = robot.data
            upright = -data.projected_gravity_b[:, 2]
            tilt = torch.rad2deg(torch.acos(upright.clamp(-1.0, 1.0)))
            just_hit = base._was_hit & (hit_step == 0)
            hit_step = torch.where(just_hit, float(step), hit_step)

            after_hit = base._was_hit
            max_tilt = torch.where(after_hit, torch.maximum(max_tilt, tilt), max_tilt)
            head = data.body_com_pos_w[:, base._head_id[0], 2] - base.scene.env_origins[:, 2]
            standing = (head >= STANDING_HEAD_HEIGHT) & (upright >= tilt_cos)
            newly_ok = after_hit & standing & torch.isnan(recovered_step)
            recovered_step = torch.where(newly_ok, float(step), recovered_step)
            fell |= base._fell

        stayed_up = (~fell).float().mean().item() * 100.0
        head = robot.data.body_com_pos_w[:, base._head_id[0], 2] - base.scene.env_origins[:, 2]
        upright = -robot.data.projected_gravity_b[:, 2]
        still_standing = ((head >= STANDING_HEAD_HEIGHT) & (upright >= tilt_cos)).float().mean().item() * 100.0
        rec = (recovered_step - hit_step) * base.step_dt
        rec = rec[torch.isfinite(rec) & (rec >= 0)]
        rec_s = rec.mean().item() if rec.numel() else float("nan")

        print(f"{impulse:>9.1f}{impulse / 4.5:>10.1f}x{stayed_up:>11.1f}%{still_standing:>15.1f}%"
              f"{max_tilt.mean().item():>11.1f}{rec_s:>11.2f}")

    print("=" * 72)
    print("  'stayed up' = never tripped the fall termination during the episode")
    print("  'still standing' = meets the strict criterion at the final step")

    # Per-target breakdown at a fixed hard impulse. An aggregate hides the failure that matters:
    # a policy solid on torso shoves and helpless when a shin is clipped mid-swing scores well
    # overall, because ten of the twelve targets are easy.
    probe = max(args.impulses)
    fell_by_body = {name: [0, 0] for name in base._target_names}
    id_to_name = {int(i): n for i, n in zip(base._target_ids.tolist(), base._target_names)}
    # Denominator must be "shots actually delivered at this body", counted as each hit fires -
    # not a snapshot of what happens to be targeted at the end, which would be a per-body sample of
    # 1/12th of one instant and make every rate noise.
    was_hit_prev = base._was_hit.clone()
    for _ in range(2 * episode_steps):
        with torch.inference_mode():
            mag = base._push_impulse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            base._push_impulse = base._push_impulse / mag * probe
            obs, _, _, _ = env.step(policy(obs))
        newly_hit = base._was_hit & (~was_hit_prev)
        for env_i in newly_hit.nonzero(as_tuple=False).squeeze(-1).tolist():
            fell_by_body[id_to_name[int(base._push_body_idx[env_i])]][1] += 1
        for env_i in base._fell.nonzero(as_tuple=False).squeeze(-1).tolist():
            fell_by_body[id_to_name[int(base._push_body_idx[env_i])]][0] += 1
        was_hit_prev = base._was_hit.clone()

    print()
    print("=" * 72)
    print(f"  PER-TARGET at {probe:.0f} N.s")
    print("=" * 72)
    print(f"  {'target':<12}{'shots':>8}{'falls':>8}{'fall rate':>12}")
    print("-" * 72)
    for name, (fell_n, shots) in sorted(
        fell_by_body.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1))
    ):
        rate = 100.0 * fell_n / shots if shots else float("nan")
        print(f"  {name:<12}{shots:>8}{fell_n:>8}{rate:>11.1f}%")
    print("=" * 72)
    print("  A target far above the others is the weak case - that is what this table is for.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
