"""How much exploration noise does this plant tolerate?

The v2 training run started from the zero-action controller that stands 40 s at 100% upright, and
still fell after 1.7 s of every episode - before the first ball, which fires at 2-4 s. So the falls
were caused by the exploration noise itself, and the policy was being taught to survive PPO rather
than to survive an impact.

This measures the tolerated std directly instead of guessing it a third time: zero-mean Gaussian
actions at a range of std values, quiet room, no ball.

The arithmetic that motivates it: an action of `a` moves a joint target by `a * span * authority`,
and the position actuator answers with `kp` up to 1800 N.m/rad. At std 0.15 and authority 0.25 that
is a fresh random target ~3.75% of the joint range away, on all 36 joints, 60 times a second.

    python mujoco_rig/rl/probe_noise.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from perturb_env import PerturbEnv  # noqa: E402

SECONDS = 15.0
CASES = [(0.25, 0.03, 4), (0.25, 0.10, 4),
         (0.25, 0.03, 8), (0.25, 0.10, 8), (0.25, 0.20, 8),
         (0.25, 0.10, 12), (0.25, 0.20, 12), (0.25, 0.35, 12),
         (0.25, 0.20, 20), (0.25, 0.35, 20), (0.25, 0.50, 20)]
ENVS = 6


def main() -> int:
    print(f"zero-mean Gaussian actions, no ball, {ENVS} envs x {SECONDS:.0f} s")
    print(f"{'std':>6}  {'auth':>5}  {'Hz':>5}  {'upright %':>9}  {'fell':>6}  {'fall s':>7}")
    for authority, std, dec in CASES:
        env = PerturbEnv(num_envs=ENVS, episode_seconds=SECONDS, seed=5, model="dummy.xml")
        env.authority = authority
        # Policy rate is the second axis. The plant tolerates LARGE slow deviations - the trained
        # deterministic policy uses the full 0.25 authority without falling - and rejects fast random
        # ones, so the question is whether a slower policy step buys back the exploration magnitude
        # that a protective step needs.
        env.decimation = dec
        env.max_episode_length = int(SECONDS / (env.dt * dec))
        env.reset_all()
        env.auto_reset = False
        steps = env.max_episode_length
        up = np.zeros(ENVS)
        alive = np.full(ENVS, steps, dtype=float)
        fallen = np.zeros(ENVS, dtype=bool)
        g = torch.Generator().manual_seed(5)
        for t in range(steps):
            env.step(torch.randn(ENVS, env.num_actions, generator=g) * std)
            for i, d in enumerate(env.datas):
                u = d.xmat[env.pelvis].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
                if d.xpos[env.pelvis][2] >= 0.55 and u[2] >= 0.5:
                    up[i] += 1.0
                elif not fallen[i]:
                    fallen[i], alive[i] = True, t
        dt = env.dt * env.decimation
        print(f"{std:6.2f}  {authority:5.2f}  {1.0 / dt:5.0f}  "
              f"{100.0 * up.mean() / steps:9.1f}  {int(fallen.sum()):3d}/{ENVS}  "
              f"{np.mean(alive) * dt:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
