"""Score a trained policy on the assist-free MuJoCo plant.

**Gates this obeys, all of them bought with retractions earlier in the project.**

- *40 seconds, not 20.* A 20 s window has passed a body that was already toppling three separate
  times on this project. Travel is also reported per quarter, because a fall produces distance.
- *Uprightness is reported beside every gait number.* Stride, foot height and excursion are all
  manufactured by a falling body; a step count without an uprightness column means nothing.
- *The zero-action baseline runs in the same process.* A task nothing can fail teaches nothing, and
  the only way to know the policy did the work is to see what the plant does without it.

Auto-reset is disabled here. Training wants a fallen env back in service immediately; scoring wants
to know exactly when it fell and never to see it stand back up for free.

    python mujoco_rig/rl/eval.py --checkpoint logs/mujoco/<run>/model_3000.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from perturb_env import PerturbEnv
from env_config import FALL_FRACTION, FOOT_CLEAR  # noqa: E402
from ppo import ActorCritic  # noqa: E402
from recovery_metrics import (RecoveryMetrics, HeightProxyRecoveryMetrics,
                              LEGACY_METRIC_VERSION, CONTACT_METRIC_VERSION)  # noqa: E402
from env_config import BALL_SPAWN_DISTANCE  # noqa: E402

# The same pelvis height the training fall penalty uses - read from the ENVIRONMENT, which
# derives it from the model. Hardcoding it at 0.55 silently changed what "upright" meant
# every time the body's height changed.
UPRIGHT_COSINE = 0.50      # pelvis up-axis vs world up; 60 deg of tilt
STEP_TRAVEL = 0.05         # horizontal travel during one flight that counts as a step


def rollout(policy, num_envs, seconds, seed, ball, label, ball_every=(4.0, 7.0), ball_speed=6.0,
            single_hit=False, observation_version="legacy_v1"):
    """One scored run. `policy` of None is the zero-action baseline."""
    env = PerturbEnv(num_envs=num_envs, episode_seconds=seconds, seed=seed,
                     ball_every=ball_every, ball_speed=ball_speed, observation_version=observation_version,
                     model="dummy_ball.xml" if ball else "dummy.xml")
    env.reset_all()
    env.auto_reset = False          # score the fall, do not undo it

    steps = env.max_episode_length
    dt = env.dt * env.decimation
    n = num_envs
    flight_seconds = BALL_SPAWN_DISTANCE / max(ball_speed, 1e-6) * 1.2
    recovery = HeightProxyRecoveryMetrics(n, dt, flight_seconds, env.rest_foot_z)
    contact_recovery = RecoveryMetrics(n, dt, flight_seconds)
    previous_action = np.zeros((n, env.num_actions))

    # Episode behavior is configuration; reset and firing methods retain their normal contracts.
    env.max_shots_per_episode = 1 if single_hit else None
    hits = env.shots_fired

    upright = np.zeros(n)
    alive = np.full(n, steps, dtype=float)     # step index of the first fall
    fallen = np.zeros(n, dtype=bool)
    quarter = np.zeros((4, n))
    step_count = np.zeros(n)
    foot_air = [np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)]
    foot_from = [np.zeros((n, 2)), np.zeros((n, 2))]
    foot_want = [np.zeros((n, 2)), np.zeros((n, 2))]   # where the COM had escaped at liftoff
    aim_cos: list[float] = []
    com_prev = np.stack([env.com(d)[:2] for d in env.datas])
    obs = env.get_observations()

    for t in range(steps):
        with torch.no_grad():
            action = policy.actor(obs) if policy is not None else torch.zeros(n, env.num_actions)
        obs, _, _, _ = env.step(action)
        clipped_action = action.clamp(-1.0, 1.0).cpu().numpy()

        for i, d in enumerate(env.datas):
            up = d.xmat[env.pelvis].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            tall = (d.xpos[env.pelvis][2] >= FALL_FRACTION * env.rest_pelvis_z
                    and up[2] >= UPRIGHT_COSINE)
            if tall:
                upright[i] += 1.0
            elif not fallen[i]:
                fallen[i], alive[i] = True, t

            features = env.recovery_features(i)
            for metrics in (recovery, contact_recovery):
                metrics.record(i, features, d.cvel[env.pelvis, :3],
                               not fallen[i], hits[i], clipped_action[i] - previous_action[i])

            com = env.com(d)[:2]
            if not fallen[i]:
                quarter[min(3, (t * 4) // steps), i] += float(np.linalg.norm(com - com_prev[i]))
            com_prev[i] = com

            # A protective step is a foot that leaves the floor and lands somewhere else.
            # Historical protective-step diagnostic: rest height + 6 cm. The new
            # recovery gate separately uses the tighter contact proxy and foot speed;
            # a low shuffle must not disappear from the quality measurements.
            for k, body in enumerate((env.foot_l, env.foot_r)):
                p = d.xpos[body]
                air = p[2] > env.rest_foot_z[k] + FOOT_CLEAR
                if air and not foot_air[k][i]:
                    foot_from[k][i] = p[:2]
                    mid = 0.5 * (d.xpos[env.foot_l][:2] + d.xpos[env.foot_r][:2])
                    foot_want[k][i] = com - mid
                elif foot_air[k][i] and not air and not fallen[i]:
                    moved = p[:2] - foot_from[k][i]
                    if np.linalg.norm(moved) > STEP_TRAVEL:
                        step_count[i] += 1.0
                        # Aimed at the fall: the step's direction against where the COM had
                        # escaped when the foot left the floor. +1 is a capture step.
                        want = foot_want[k][i]
                        if np.linalg.norm(want) > 1e-6:
                            aim_cos.append(float(moved @ want / (np.linalg.norm(moved)
                                                                 * np.linalg.norm(want))))
                foot_air[k][i] = air

        previous_action = clipped_action

    pct = 100.0 * upright / steps
    survived = 100.0 * np.mean(~fallen)
    quality = recovery.result(~fallen, require_shot=ball)
    contact_quality = contact_recovery.result(~fallen, require_shot=ball)
    print(f"\n=== {label} ===")
    print(f"  upright        {pct.mean():6.1f} %   (best env {pct.max():.1f}, worst {pct.min():.1f})")
    print(f"  never fell     {survived:6.1f} %   of {n} envs over {seconds:.0f} s")
    print(f"  recovered      {quality['recovered']:6.1f} %   stable for the final 0.5 s")
    print(f"  contact audit  {contact_quality['recovered']:6.1f} % recovered; "
          f"{contact_quality['foot_sliding_m']:.3f} m/world RMS contact slip (diagnostic only)")
    print(f"  foot sliding   {quality['foot_sliding_m']:6.3f} m/world while alive; "
          f"rapid replants {quality['rapid_replants']:.2f}; "
          f"mean action change {quality['action_change']:.4f}")
    print(f"  time to fall   {np.mean(alive) * dt:6.2f} s  (median {np.median(alive) * dt:.2f})")
    print(f"  travel/quarter " + "  ".join(f"{q:.3f}" for q in quarter.mean(axis=1)) + "  m")
    print(f"  steps taken    {step_count.mean():6.2f}   (max {step_count.max():.0f})")
    aim = np.array(aim_cos)
    stepped = step_count > 0
    aimed = 100.0 * float(np.mean(aim > 0)) if len(aim) else float("nan")
    with_step = 100.0 * float(np.mean(~fallen[stepped])) if stepped.any() else float("nan")
    no_step = 100.0 * float(np.mean(~fallen[~stepped])) if (~stepped).any() else float("nan")
    print(f"  aimed at fall  {aimed:6.1f} %   of {len(aim)} steps (mean cos "
          f"{aim.mean() if len(aim) else float('nan'):+.3f})")
    print(f"  survived       stepped {with_step:5.1f} % of {int(stepped.sum())}, "
          f"no step {no_step:5.1f} % of {int((~stepped).sum())}")
    print(f"  hits per env   {hits.mean():6.2f}   (max {hits.max()})")
    # **How it ends up standing**, for the worlds still up at the end. Survival alone cannot tell a
    # body back in its stance from one left standing with its feet crossed.
    ends = [env.stance_of(d) for d, up in zip(env.datas, ~fallen) if up]
    stance = {k: float(np.mean([e[k] for e in ends])) if ends else float("nan")
              for k in ("score", "width", "split", "leg_rms", "trunk_rms", "at_limit")}
    crossed = 100.0 * float(np.mean([e["width"] < 0.0 for e in ends])) if ends else float("nan")
    print(f"  end stance     {stance['score']:6.2f}   width {stance['width']:.2f} m "
          f"(rest {env.rest_stance:.2f})   split {stance['split']:.2f} m   crossed {crossed:.0f}%   "
          f"leg error {stance['leg_rms']:.2f} rad   trunk {stance['trunk_rms']:.2f} rad   "
          f"joints on a limit {stance['at_limit']:.1f}   (survivors)")
    return {"upright": pct.mean(), "survived": survived, "steps": step_count.mean(),
            "fall_s": float(np.mean(alive) * dt), "survived_mask": ~fallen,
            "aimed": aimed, "no_step": no_step, "hits": float(hits.mean()),
            "stance": stance["score"], "stance_width": stance["width"],
            "stance_split": stance["split"], "stance_crossed": crossed,
            "leg_rms": stance["leg_rms"], "trunk_rms": stance["trunk_rms"],
            "at_limit": stance["at_limit"], **quality,
            "metric_version": LEGACY_METRIC_VERSION,
            "contact_metric_version": CONTACT_METRIC_VERSION,
            **{'contact_' + key: value for key, value in contact_quality.items()}}


def write_json(path, checkpoint, settings, sections):
    """The result as DATA - what promote.py and overnight.py decide on, through scripts/scoring.py.

    The printed text is for people. Parsing it is how the quiet room's block was once read as the
    under-fire one, and a policy that fell to every hit was reported 100% upright.
    """
    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v.item() if isinstance(v, np.generic) else v
    doc = {"checkpoint": str(checkpoint), "settings": settings,
           "sections": {name: {k: plain(v) for k, v in r.items()} for name, r in sections.items()}}
    pathlib.Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--quiet_seconds", type=float, default=None,
                   help="separate quiet-room duration (promotion uses 40 seconds)")
    p.add_argument("--quiet_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    # Scored at the SAME firing rate it was trained at; 2-4 s is impossible for any controller.
    p.add_argument("--ball_every", type=float, nargs=2, default=(4.0, 7.0))
    # Defaults to the curriculum stage the checkpoint actually reached, which travels with it.
    p.add_argument("--ball_speed", type=float, default=0.0)
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--baseline_only", action="store_true",
                   help="score the unaided plant alone; --checkpoint is ignored")
    p.add_argument("--single_hit", action="store_true",
                   help="fire only the first scheduled shot per env")
    p.add_argument("--under_fire_only", action="store_true",
                   help="diagnostic only: skip the quiet-room rollout required by promotion")
    p.add_argument("--json", default="",
                   help="also write every section's numbers here, as data (scripts/scoring.py)")
    args = p.parse_args()
    settings = dict(num_envs=args.num_envs, seconds=args.seconds, seed=args.seed,
                    ball_every=list(args.ball_every), single_hit=args.single_hit,
                    quiet_seconds=args.quiet_seconds or args.seconds,
                    quiet_envs=args.quiet_envs or args.num_envs)

    if args.baseline_only:
        speed = args.ball_speed or 6.0
        sections = {
            "baseline": rollout(None, args.num_envs, args.seconds, args.seed, ball=True,
                                label="ZERO ACTION, under fire  (the unaided plant)",
                                ball_every=tuple(args.ball_every), ball_speed=speed,
                                single_hit=args.single_hit),
            "baseline_no_ball": rollout(None, args.num_envs, args.seconds, args.seed, ball=False,
                                        label="ZERO ACTION, no ball")}
        if args.json:
            write_json(args.json, "", dict(settings, ball_speed=speed), sections)
        return 0

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from observation_contract import checkpoint_version
    observation_version = checkpoint_version(ck)
    net = ActorCritic(ck["num_obs"], ck["num_actions"])
    net.load_state_dict(ck["model"])
    net.eval()
    speed = args.ball_speed or float(ck.get("ball_speed", 6.0))
    print(f"[eval] {args.checkpoint}  {ck['num_obs']} obs / {ck['num_actions']} actions, "
          f"{args.num_envs} envs x {args.seconds:.0f} s, ball {speed:.2f} m/s every "
          f"{args.ball_every[0]:.1f}-{args.ball_every[1]:.1f} s"
          f"{', single hit' if args.single_hit else ''}, deterministic (mean action)")

    # The policy is scored quiet AND under fire. Without the quiet run a policy that merely stands
    # still cannot be told apart from one that rejects an impact.
    # Diagnostic callers can skip quiet trials; promotion and chaining require both sections.
    sections = {}
    if not args.under_fire_only:
        sections["no_ball"] = rollout(net, args.quiet_envs or args.num_envs,
                                      args.quiet_seconds or args.seconds, args.seed, ball=False,
                                      label="POLICY, no ball", ball_every=tuple(args.ball_every),
                                      ball_speed=speed, observation_version=observation_version)
    hit = sections["under_fire"] = rollout(net, args.num_envs, args.seconds, args.seed, ball=True,
                                           label="POLICY, under fire",
                                           ball_every=tuple(args.ball_every), ball_speed=speed,
                                           single_hit=args.single_hit, observation_version=observation_version)
    if not args.no_baseline:
        base = sections["baseline"] = rollout(
            None, args.num_envs, args.seconds, args.seed, ball=True,
            label="ZERO ACTION, under fire  (what the plant does unaided)",
            ball_every=tuple(args.ball_every), ball_speed=speed, single_hit=args.single_hit)
        print(f"\n  policy vs unaided under fire: upright {hit['upright']:.1f} % vs "
              f"{base['upright']:.1f} %, steps {hit['steps']:.2f} vs {base['steps']:.2f}")
    if args.json:
        write_json(args.json, args.checkpoint,
                   dict(settings, ball_speed=speed, observation_version=observation_version), sections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
