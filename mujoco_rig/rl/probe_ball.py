"""Is the training perturbation survivable at all?

Run 3 preserved the rest-pose baseline and then flatlined - return 2506 at iteration 65, 2667 at 375
- and scored 32.9% under fire against the unaided plant's 32.3%. A flat return is what a WORKING
algorithm does on an impossible task, so before changing the algorithm again, check the task.

The projectile carries 10 kg. `build_mjcf.py` says why: Godot's BallGun fires 3.0 kg at 6 m/s, which
this body absorbs with 3.5 degrees of tilt - correct, but invisible to watch - so it was raised to
make the hit read on screen. That display decision then became the training perturbation. 10 kg at
6 m/s is 60 N.s into an 80.6 kg body.

This sweeps the mass with NO policy, so the number is the plant's own survivability. Godot's real
3.0 kg is the value that matters for transfer; anything the unaided body cannot survive at all
leaves a policy nothing to learn.

    python mujoco_rig/rl/probe_ball.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from perturb_env import PerturbEnv  # noqa: E402

SECONDS = 40.0
ENVS = 8


def main() -> int:
    print(f"zero action (rest-pose hold), ball every 2-4 s at 6 m/s, {ENVS} envs x {SECONDS:.0f} s")
    print(f"{'ball kg':>8}  {'N.s':>6}  {'upright %':>9}  {'fell':>6}  {'fall s':>7}")
    for mass in (2.0, 3.0, 5.0, 7.0, 10.0):
        env = PerturbEnv(num_envs=ENVS, episode_seconds=SECONDS, seed=5, model="dummy_ball.xml")
        authored = float(env.model.body_mass[env.ball])
        scale = mass / authored
        env.model.body_mass[env.ball] = mass
        env.model.body_inertia[env.ball] *= scale      # inertia scales with mass at fixed radius
        env.reset_all()
        env.auto_reset = False
        steps = env.max_episode_length
        up = np.zeros(ENVS)
        alive = np.full(ENVS, steps, dtype=float)
        fallen = np.zeros(ENVS, dtype=bool)
        zero = torch.zeros(ENVS, env.num_actions)
        for t in range(steps):
            env.step(zero)
            for i, d in enumerate(env.datas):
                u = d.xmat[env.pelvis].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
                if d.xpos[env.pelvis][2] >= 0.55 and u[2] >= 0.5:
                    up[i] += 1.0
                elif not fallen[i]:
                    fallen[i], alive[i] = True, t
        dt = env.dt * env.decimation
        print(f"{mass:8.1f}  {mass * 6.0:6.1f}  {100.0 * up.mean() / steps:9.1f}  "
              f"{int(fallen.sum()):3d}/{ENVS}  {np.mean(alive) * dt:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
