"""Is a SINGLE impact recoverable? The mass sweep says the ball is not what makes this hard.

`probe_ball.py`, 2026-09-09: with no policy, the body toppled 8/8 in 5-7 s at every ball mass. That
run's mass sweep is RETRACTED - it wrote `model.body_mass` on a live model, which leaves MuJoCo's
precomputed `body_invweight0` / `dof_invweight0` constants stale, and those scale the contact solver.
It reported a 0.5 kg ball as more destructive than a 10 kg one. Mass is changed in the MJCF and
recompiled here instead, which is the only way to get a consistent body.

What is left is the RATE. Balls arrive every 2-4 s from a random heading at one of twelve bones,
and a recovery takes on the order of a second. A body that is hit again mid-recovery never finishes
one, and no controller can be scored on a task where survival is impossible for every policy.

This fires exactly ONE ball and watches what happens afterwards, which is the quantity a curriculum
needs: the largest impact the unaided plant can absorb, and how long it takes to settle.

    python mujoco_rig/rl/probe_single.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from perturb_env import PerturbEnv  # noqa: E402

SECONDS = 12.0
ENVS = 12
FIRE_AT = 3.0
# The ball needs ~0.33 s to cross the 2 m to its target. Measuring "settled" from the moment of
# FIRING reported 0.00 s for every mass - it was reading the untouched body before the impact.
LANDED = FIRE_AT + 0.5


def main() -> int:
    print(f"zero action, ONE ball at t={FIRE_AT:.0f}s, {ENVS} envs x {SECONDS:.0f} s")
    print(f"{'ball kg':>8}  {'N.s':>6}  {'survived':>9}  {'max tilt':>9}  {'settled':>8}")
    source = (pathlib.Path(__file__).resolve().parent.parent / "dummy_ball.xml").read_text("utf-8")
    for mass in (0.5, 1.0, 2.0, 3.0, 6.0, 10.0):
        variant = pathlib.Path(__file__).resolve().parent.parent / f"_ball_{mass:g}.xml"
        variant.write_text(re.sub(r'(name="g_ball"[^>]*?mass=")[\d.]+',
                                  rf'\g<1>{mass}', source), encoding="utf-8")
        env = PerturbEnv(num_envs=ENVS, episode_seconds=SECONDS, seed=11, model=variant.name)
        assert abs(float(env.model.body_mass[env.ball]) - mass) < 1e-6, "mass did not take"
        env.reset_all()
        env.auto_reset = False
        env.ball_every = (1.0e6, 1.0e6)          # one shot only
        env.next_ball[:] = FIRE_AT
        steps = env.max_episode_length
        dt = env.dt * env.decimation

        fallen = np.zeros(ENVS, dtype=bool)
        max_tilt = np.zeros(ENVS)
        settled = np.full(ENVS, np.nan)
        zero = torch.zeros(ENVS, env.num_actions)
        for t in range(steps):
            env.step(zero)
            for i, d in enumerate(env.datas):
                u = d.xmat[env.pelvis].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
                tilt = float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))
                if t * dt > LANDED:
                    max_tilt[i] = max(max_tilt[i], tilt)
                if d.xpos[env.pelvis][2] < 0.55 or u[2] < 0.5:
                    fallen[i] = True
                # settled = upright and slow again, after the hit
                elif (t * dt > LANDED and np.isnan(settled[i]) and tilt < 5.0
                      and np.linalg.norm(d.cvel[env.pelvis][3:6]) < 0.1):
                    settled[i] = t * dt - LANDED
        ok = ~fallen
        s = settled[ok & ~np.isnan(settled)]
        print(f"{mass:8.1f}  {mass * 6.0:6.1f}  {100.0 * ok.mean():8.0f}%  "
              f"{max_tilt.mean():8.1f}d  "
              + (f"{np.mean(s):7.2f}s" if len(s) else "       -"))
        variant.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
