"""Score a walk policy on CPU MuJoCo - the engine Godot drives.

Answers the three questions actually asked of a walk, one manoeuvre at a time with the command
PINNED, because a policy that tracks a randomly changing command well on average can still be
unable to hold a straight line:

  * **Does it go straight?** Forward command, zero lateral, zero turn. Reports achieved speed
    against commanded, and lateral drift - the thing that makes a walk look wrong.
  * **Does it turn?** Yaw command. Reports achieved turn rate and the heading actually swept.
  * **Does it stop?** Zero command. A walk that cannot stop is not controllable from Godot.

**Every gait number is reported beside uprightness, and the run is 40 seconds.** Both rules were
bought with retractions on this project: stride, foot height and excursion are all manufactured by
a falling body, and a 20 s window passed a body that was already toppling three separate times.

Single-support fraction separates the three failure modes this project keeps producing:
a *walk* alternates (single support most of the time), a *hop* has both feet off together, and a
*flamingo* parks one foot in the air and never puts it down.

    python mujoco_rig/rl/eval_walk.py --checkpoint logs/mujoco/<run>/model_N.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ppo import ActorCritic  # noqa: E402
from walk_env import WalkEnv  # noqa: E402
from env_config import WALK_FOOT_CLEAR  # noqa: E402

UPRIGHT_HEIGHT = 0.55
UPRIGHT_COSINE = 0.50
STEP_TRAVEL = 0.05


def manoeuvre(policy, num_envs, seconds, seed, cmd, label):
    env = WalkEnv(num_envs=num_envs, episode_seconds=seconds, seed=seed)
    env.reset_all()
    env.set_command(*cmd)
    env.auto_reset = False          # score the fall, never undo it

    steps = env.max_episode_length
    dt = env.dt * env.decimation
    n = num_envs

    upright = np.zeros(n)
    fallen = np.zeros(n, dtype=bool)
    alive = np.full(n, steps, dtype=float)
    start_xy = np.stack([d.xpos[env.pelvis][:2].copy() for d in env.datas])
    start_yaw = np.zeros(n)
    vx = [[] for _ in range(n)]
    vy = [[] for _ in range(n)]
    wz = [[] for _ in range(n)]
    single = np.zeros(n)
    both_air = np.zeros(n)
    step_count = np.zeros(n)
    foot_air = [np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)]
    foot_from = [np.zeros((n, 2)), np.zeros((n, 2))]
    obs = env.get_observations()

    for i in range(n):
        start_yaw[i] = env.heading_of(i)
    yaw_prev = start_yaw.copy()
    yaw_total = np.zeros(n)

    for t in range(steps):
        with torch.no_grad():
            action = policy.actor(obs) if policy is not None else torch.zeros(n, env.num_actions)
        obs, _, _, _ = env.step(action)

        for i, d in enumerate(env.datas):
            r = d.xmat[env.pelvis].reshape(3, 3)
            up = r @ np.array([0.0, 0.0, 1.0])
            tall = d.xpos[env.pelvis][2] >= UPRIGHT_HEIGHT and up[2] >= UPRIGHT_COSINE
            if tall:
                upright[i] += 1.0
            elif not fallen[i]:
                fallen[i], alive[i] = True, t
            if fallen[i]:
                continue

            lin = r.T @ d.cvel[env.pelvis][3:6]
            ang = r.T @ d.cvel[env.pelvis][:3]
            vx[i].append(float(lin[0])); vy[i].append(float(lin[1])); wz[i].append(float(ang[2]))

            y = env.heading_of(i)
            dy = np.arctan2(np.sin(y - yaw_prev[i]), np.cos(y - yaw_prev[i]))
            yaw_total[i] += dy
            yaw_prev[i] = y

            fz = np.array([d.xpos[env.foot_l][2], d.xpos[env.foot_r][2]])
            air = fz > env.rest_foot_z + WALK_FOOT_CLEAR     # above rest, as the reward reads it
            if air.sum() == 1:
                single[i] += 1.0
            elif air.sum() == 2:
                both_air[i] += 1.0
            for k, body in enumerate((env.foot_l, env.foot_r)):
                p = d.xpos[body]
                if air[k] and not foot_air[k][i]:
                    foot_from[k][i] = p[:2]
                elif foot_air[k][i] and not air[k]:
                    if np.linalg.norm(p[:2] - foot_from[k][i]) > STEP_TRAVEL:
                        step_count[i] += 1.0
                foot_air[k][i] = air[k]

    end_xy = np.stack([d.xpos[env.pelvis][:2].copy() for d in env.datas])
    travel = end_xy - start_xy
    # Distance along the commanded heading, and perpendicular to it - "does it go straight".
    forward = np.array([np.cos(start_yaw), np.sin(start_yaw)]).T
    lateral = np.array([-np.sin(start_yaw), np.cos(start_yaw)]).T
    along = (travel * forward).sum(axis=1)
    across = (travel * lateral).sum(axis=1)

    mean = lambda xs: float(np.mean([np.mean(x) for x in xs if len(x)])) if any(xs) else float("nan")
    live = np.maximum(alive, 1)
    pct = 100.0 * upright / steps
    print(f"\n=== {label}   cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f} ===")
    print(f"  upright        {pct.mean():6.1f} %      never fell {100.0 * np.mean(~fallen):5.1f} %"
          f"   time to fall {np.mean(alive) * dt:5.2f} s")
    print(f"  speed          vx {mean(vx):+6.3f} (cmd {cmd[0]:+.2f})   "
          f"vy {mean(vy):+6.3f} (cmd {cmd[1]:+.2f})   yaw {mean(wz):+6.3f} (cmd {cmd[2]:+.2f})")
    print(f"  travel         along {along.mean():+6.2f} m   across {across.mean():+6.2f} m"
          f"   heading swept {np.degrees(yaw_total.mean()):+7.1f} deg")
    print(f"  gait           single support {100.0 * single.mean() / live.mean():5.1f} %   "
          f"both feet airborne {100.0 * both_air.mean() / live.mean():5.1f} %   "
          f"steps {step_count.mean():5.1f}")
    return {"cmd": list(cmd), "upright": float(pct.mean()),
            "survived": 100.0 * float(np.mean(~fallen)), "fall_s": float(np.mean(alive) * dt),
            "vx": mean(vx), "vy": mean(vy), "wz": mean(wz),
            "along": float(along.mean()), "across": float(across.mean()),
            "heading_swept_deg": float(np.degrees(yaw_total.mean())),
            "single": float(100.0 * single.mean() / live.mean()),
            "airborne": float(100.0 * both_air.mean() / live.mean()),
            "steps": float(step_count.mean())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--seconds", type=float, default=40.0)
    # The straight line is scored at the speed the policy was TRAINED for. At a fixed 0.6 m/s a
    # stage-1 policy (0.15-0.35 m/s) is judged on a command it has never seen.
    p.add_argument("--speed", type=float, default=0.6,
                   help="forward command for the STRAIGHT LINE manoeuvre")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--json", default="",
                   help="also write every manoeuvre's numbers here, as data (scripts/scoring.py)")
    args = p.parse_args()

    policy = None
    if not args.baseline_only:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        assert ck.get("task", "walk") == "walk", f"checkpoint is a {ck.get('task')} policy"
        policy = ActorCritic(ck["num_obs"], ck["num_actions"])
        policy.load_state_dict(ck["model"])
        policy.eval()
        print(f"[eval_walk] {args.checkpoint}  {ck['num_obs']} obs / {ck['num_actions']} actions, "
              f"{args.num_envs} envs x {args.seconds:.0f} s on CPU MuJoCo")
    else:
        print(f"[eval_walk] NO POLICY (the unaided plant), {args.num_envs} envs x {args.seconds:.0f} s")

    manoeuvres = [("STRAIGHT LINE", (args.speed, 0.0, 0.0)),
                  ("TURN LEFT", (0.4, 0.0, 0.6)),
                  ("TURN RIGHT", (0.4, 0.0, -0.6)),
                  # Turning on the spot is what a joystick does when the stick is only rotated, and
                  # it was not scored at all - the two turns above both carry forward speed, so a
                  # policy that can only turn while walking passed them.
                  ("TURN IN PLACE", (0.0, 0.0, 0.8)),
                  ("STAND STILL", (0.0, 0.0, 0.0))]
    results = {label.lower().replace(" ", "_"): manoeuvre(policy, args.num_envs, args.seconds,
                                                          args.seed, cmd, label)
               for label, cmd in manoeuvres}
    if args.json:
        # The result as DATA - what promote.py and overnight.py decide on, via scripts/scoring.py.
        doc = {"checkpoint": args.checkpoint,
               "settings": dict(num_envs=args.num_envs, seconds=args.seconds, seed=args.seed,
                                speed=args.speed),
               "manoeuvres": results}
        pathlib.Path(args.json).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
