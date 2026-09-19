"""Is this task WINNABLE, and does the environment do what it claims? Run before long sessions.

**Why this exists.** Everything else in this repo checks that the code does what it was told:
`test_env_parity.py` that both backends agree, `validate.py` that the model is the body we meant,
`batch_table.py` that the batch is worth its wall-clock. None of them asks whether what the code was
told to do is POSSIBLE. Four failures cost hours of training each, and every one was a task-level
error a few minutes of measurement would have caught:

  * the projectile could not reach the dummy - a ballistic ball's range is `v^2/g`, which is 0.60 m
    at the 2.42 m/s the curriculum had reached, from a 2 m standoff;
  * the arms hung 5.5 cm inside the thighs at the start pose, 331 N.m per shoulder to hold;
  * exploration was inherited at std 0.08 from a brain trained to stand perfectly still, so a step
    was undiscoverable, and eleven walk sessions drove it lower still;
  * the ball was 15 kg - 90 N.s, 1.29 m/s of centre-of-mass velocity, which no controller with
    human strength survives. The curriculum was climbing toward a target that did not exist.

None of these is a learning problem and no amount of training fixes any of them.

    python mujoco_rig/scripts/preflight.py --task perturb --speed_end 6.0
    python mujoco_rig/scripts/preflight.py --task walk
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rl"))

import mujoco                                                      # noqa: E402
import torch                                                       # noqa: E402

from env_config import (CAPTURE_V, FALL_FRACTION, FOOT_CLEAR,      # noqa: E402
                        OFF_BALANCE)
from perturb_env import PerturbEnv                                 # noqa: E402
from walk_env import WalkEnv                                       # noqa: E402

FAIL, WARN, OK = "FAIL", "WARN", "ok  "
results: list[tuple[str, str, str]] = []


def report(level, name, detail):
    results.append((level, name, detail))
    print(f"  [{level}] {name}: {detail}", flush=True)


def check_start_pose(env):
    """The body must not begin inside itself. Every episode starts here."""
    m, d = env.model, env.datas[0]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "rest")
    mujoco.mj_resetDataKeyframe(m, d, key if key >= 0 else 0)
    mujoco.mj_forward(m, d)
    bad = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
           for c in d.contact[:d.ncon]
           if m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
           and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE]
    if bad:
        report(FAIL, "start pose", f"self-intersects: {sorted(set(bad))}")
    else:
        report(OK, "start pose", f"{d.ncon} contacts, none body-on-body")


def check_perturbation(env, speeds):
    """Does the projectile CONNECT, and how many newton-seconds does it deliver?"""
    if env.ball < 0:
        report(WARN, "perturbation", "this model carries no projectile")
        return
    m = env.model
    ball_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == env.ball]
    kg = float(m.body_mass[env.ball])
    for speed in speeds:
        env.ball_speed = speed
        peak = 0.0
        # **Several shots, not one.** The gun picks a bone at random from twelve, and a single
        # shot at a limb can legitimately miss - reading that as "the projectile cannot reach"
        # would be the check inventing a failure. The best of a handful answers the question the
        # check is actually asking: CAN it connect?
        for shot in range(6):
            env.reset_all()
            d = env.datas[0]
            for k in range(int(3.0 / (env.dt * env.decimation))):
                if k == 1:
                    env.fire_ball(d, 0)
                before = env.com_velocity(d) * env.total_mass
                env.step(torch.zeros(env.num_envs, env.num_actions))
                after = env.com_velocity(d) * env.total_mass
                if any((c.geom1 in ball_geoms or c.geom2 in ball_geoms)
                       and m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
                       and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE
                       for c in d.contact[:d.ncon]):
                    peak = max(peak, float(np.linalg.norm(after - before)))
        carried = kg * speed
        level = OK if peak > 0.25 * carried else FAIL
        report(level, f"projectile at {speed:.2f} m/s",
               f"carries {carried:5.1f} N.s, delivers {peak:5.1f} N.s "
               f"({peak/max(carried,1e-9):.0%})"
               + ("" if level == OK else "  <- it is MISSING"))


def check_feasible(env, speed_end, trials=12):
    """Can ANY controller survive one impact at the curriculum's end? Zero-action is the floor.

    This is not a policy test. If a single hit at `speed_end` puts the body on the floor whatever
    it does, the curriculum's target does not exist and every hour spent climbing toward it is
    wasted - which is precisely what happened with a 15 kg ball.
    """
    if env.ball < 0:
        return
    m = env.model
    kg = float(m.body_mass[env.ball])
    env.ball_speed = speed_end
    env.reset_all()
    env.auto_reset = False
    dv = np.zeros(env.num_envs)
    dt = env.dt * env.decimation
    for k in range(int(6.0 / dt)):
        if k == int(0.5 / dt):
            for i, d in enumerate(env.datas):
                env.fire_ball(d, i)
        before = [env.com_velocity(d)
                  for d in env.datas]
        env.step(torch.zeros(env.num_envs, env.num_actions))
        after = [env.com_velocity(d)
                 for d in env.datas]
        # **Per step, not the maximum absolute speed.** Taking the largest COM speed over the
        # window counts the FALL as if the ball had caused it - the same conflation that made
        # MjBallGun report 168 N.s for a shot that merely knocked the dummy down. What the impact
        # did is the change it produced in one step; gravity contributes g*dt = 0.04 m/s to that.
        dv = np.maximum(dv, [float(np.linalg.norm(a - b)) for a, b in zip(after, before)])
    # **The bound is the carried impulse over the body mass, not anything measured.** A measured
    # centre-of-mass change mixes in the ground reaction and the body's own dynamics - the first
    # version of this check read 2.02 m/s and the second 1.06 m/s for the same 30 N.s shot whose
    # unambiguous value is 30/69.6 = 0.43. Momentum is momentum; use it.
    #
    # A standing person arrests roughly 0.3-0.5 m/s with the ankles and about 1.0 m/s by stepping.
    # Beyond that, going down is the correct outcome and no controller changes it.
    dv_bound = kg * speed_end / env.total_mass
    upright = float(np.mean([d.xpos[env.pelvis][2] > FALL_FRACTION * env.rest_pelvis_z
                             for d in env.datas]))
    level = OK if dv_bound < 1.0 else FAIL
    report(level, f"feasibility at {speed_end:.2f} m/s",
           f"{kg * speed_end:.0f} N.s / {env.total_mass:.0f} kg = {dv_bound:.2f} m/s of COM "
           f"velocity; unaided body upright after {100*upright:.0f}%"
           + ("" if level == OK else "  <- beyond a stepping recovery (~1.0 m/s); UNWINNABLE"))


def _step_fixture(env):
    """A body tipped off balance with its left foot lifted, as a function of how fast that foot is
    swinging - and the hip rate that swings it TOWARD the escaping centre of mass.

    "Toward" comes from finite-differencing the foot's position, independently of the velocity the
    reward reads. MuJoCo's cvel is referenced to the subtree centre of mass, and read raw it reports
    -0.14 m/s for a foot moving at +0.70 - the wrong SIGN - so a reward built on it fails here.
    Returns None when this model cannot produce the fixture.
    """
    m, d = env.model, env.datas[0]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "rest")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Thigh_L_rx")
    qa, va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]

    def at(tilt, hip_rate=0.0):
        mujoco.mj_resetDataKeyframe(m, d, key if key >= 0 else 0)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0]),
                           d.qpos[3:7].copy())
        d.qpos[3:7] = q
        d.qpos[qa] = -0.5
        d.qvel[va] = hip_rate
        mujoco.mj_forward(m, d)

    def escape():
        mid = 0.5 * (d.xpos[env.foot_l][:2] + d.xpos[env.foot_r][:2])
        return env.com(d)[:2] - mid

    tilt, widest = 0.0, -1.0
    for t in (0.15, -0.15):
        at(t)
        if np.linalg.norm(escape()) > widest:
            tilt, widest = t, float(np.linalg.norm(escape()))
    at(tilt)
    want = escape()
    lift = np.array([d.xpos[env.foot_l][2], d.xpos[env.foot_r][2]]) - env.rest_foot_z
    if np.linalg.norm(want) < OFF_BALANCE or (lift > FOOT_CLEAR).sum() != 1:
        return None

    # The lifted foot's horizontal travel per radian of hip, from positions alone.
    p0 = d.xpos[env.foot_l][:2].copy()
    d.qpos[qa] += 1e-5
    mujoco.mj_kinematics(m, d)
    per_rad = (d.xpos[env.foot_l][:2] - p0) / 1e-5
    toward = 1.0 if per_rad @ want > 0 else -1.0
    rate = toward * 1.5 * CAPTURE_V / max(float(np.linalg.norm(per_rad)), 1e-6)
    return (lambda hip_rate: at(tilt, hip_rate)), rate


def check_reward_ranking(env, capture=False):
    """Does the reward rank the intended behaviours in the intended order?

    The two reward failures that each cost a 15-minute session were both visible here without any
    training: a flight penalty so strong that standing still outscored walking, and a stance term
    that made a protective step cost reward. With `capture` the step is a real capture - a foot
    swinging toward the escaping centre of mass, see _step_fixture - rather than a lifted foot.
    """
    m, d = env.model, env.datas[0]
    a = np.zeros(env.num_actions)

    def at(fn):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        fn()
        mujoco.mj_forward(m, d)
        return env.reward(0, a)

    def lift_one():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Thigh_L_rx")
        d.qpos[m.jnt_qposadr[j]] = -0.5

    def lying():
        # **Actually fallen: on its back, on the floor.** This used to be the standing pose sunk
        # 45 cm, which keeps every balance term and scored only 1.0 below standing.
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]),
                           d.qpos[3:7].copy())
        d.qpos[3:7] = q
        d.qpos[2] = 0.12

    rest = at(lambda: None)
    airborne = at(lambda: d.qpos.__setitem__(2, d.qpos[2] + 0.25))
    fallen = at(lying)
    # A perturb step is a CAPTURE. A foot lifted and held still is the stomp the reward no longer
    # pays for; "a stomp beats falling" only ever passed because of that defect.
    fixture = _step_fixture(env) if capture else None
    if fixture:
        set_state, rate = fixture
        set_state(rate)
        stepping = env.reward(0, a)
    else:
        stepping = at(lift_one)

    report(OK if rest > fallen else FAIL, "reward: standing beats fallen",
           f"{rest:+.2f} vs {fallen:+.2f}")
    report(OK if rest > airborne else WARN, "reward: grounded beats airborne",
           f"{rest:+.2f} vs {airborne:+.2f}")
    report(OK if stepping > fallen else FAIL, "reward: a step beats falling",
           f"{stepping:+.2f} vs {fallen:+.2f}")


def check_step_is_worth_taking(env):
    """Does the reward pay more for a CAPTURE step than for a stomp?

    The perturb reward once paid for single support while off balance, and a stomp satisfies that
    completely. Measured on 2026-09-10 at 6 m/s: 78% of foot lifts travelled under 5 cm, the rest
    were directionally random (mean cos -0.069 to the COM escape), and four sessions learned
    nothing. Three states identical in POSITION that differ only in how the lifted foot is moving:
    not at all, toward the escaping centre of mass, and away from it. Only the capture should earn.
    """
    fixture = _step_fixture(env)
    if fixture is None:
        report(WARN, "reward: capture beats stomp",
               "could not build an off-balance, one-foot fixture on this model; not judged")
        return
    set_state, rate = fixture
    a = np.zeros(env.num_actions)
    scores = {}
    for label, w in (("stomp", 0.0), ("capture", rate), ("away", -rate)):
        set_state(w)
        scores[label] = env.reward(0, a)

    gain = scores["capture"] - scores["stomp"]
    report(OK if gain > 0.1 else FAIL, "reward: capture beats stomp",
           f"capture {scores['capture']:+.2f} vs stomp {scores['stomp']:+.2f}"
           + ("" if gain > 0.1 else "  <- a stomp earns what a step earns; the policy will farm it"))
    # **Within 1% of what a capture earns, not exactly zero.** The placement term pays the swing
    # foot's closeness to the capture point, which moves with the COM velocity - and swinging a leg
    # moves the COM. On this fixture that couples a swing away to +0.0003 of placement, 0.01% of the
    # capture's advantage; a real leak is the size of recover_step itself.
    leak = scores["away"] - scores["stomp"]
    clean = leak <= 0.01 * max(gain, 0.0)
    report(OK if clean else FAIL, "reward: stepping away is not paid",
           f"away {scores['away']:+.4f} vs stomp {scores['stomp']:+.4f}"
           + ("" if clean else "  <- a step AWAY from the falling COM is rewarded"))


def check_walk_beats_statue(env):
    """Does walking out-earn standing still, at every speed the walk is commanded?

    The walk reward's velocity kernel was calibrated at 0.6 m/s, where a statue collects 9% of the
    tracking reward. Stage 1 was later slowed to 0.15-0.35 m/s, where the same kernel paid a statue
    44-86% of it: walking, which risks a fall on every step, then lost to standing still, and the
    one session trained there stood still at 0.03 m/s. Two states per speed, identical but for
    motion - standing, and the whole body gliding at exactly the command. A real walker earns less
    than the glide (sway, stride oscillation, falls), so the margin has to be comfortable.
    """
    m, d = env.model, env.datas[0]
    a = np.zeros(env.num_actions)

    def at(cmd, moving):
        env.set_command(cmd, 0.0, 0.0)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        if moving:
            d.qvel[0:3] = d.xmat[env.pelvis].reshape(3, 3) @ np.array([cmd, 0.0, 0.0])
            mujoco.mj_forward(m, d)
        return env.reward(0, a)

    for cmd in (0.15, 0.25, 0.35, 0.60):
        statue, moving = at(cmd, False), at(cmd, True)
        margin = moving - statue
        report(OK if margin >= 2.0 else FAIL, f"reward: walking beats a statue at {cmd:.2f} m/s",
               f"moving {moving:+.2f} vs statue {statue:+.2f} (margin {margin:+.2f}, the statue "
               f"keeps {100 * statue / max(moving, 1e-9):.0f}%)"
               + ("" if margin >= 2.0 else "  <- standing still is the rational policy"))


def check_blowup_is_contained(env):
    """Does ONE exploding world stay harmless? It must never hand PPO an unbounded reward.

    Walk training diverged to NaN twice on 2026-09-10 because one world in 4,096 blew up: the walk
    reward's velocity-squared penalties turned it into -5.6e3, then -3.6e15, then NaN, and nothing
    sanitised the reward before the update saw it. The perturb reward, bounded, gave -8.2 for the
    same blow-up. A finite, bounded reward for a broken world is the whole requirement here.
    """
    env.reset_all()
    a = torch.zeros(env.num_envs, env.num_actions)
    for _ in range(3):
        env.step(a)
    env.datas[0].qvel[:] = 50.0               # every DOF at 50 rad/s or m/s: a blow-up
    worst = 0.0
    for _ in range(5):
        _, r, _, _ = env.step(a)
        r0 = float(r[0])
        worst = max(worst, abs(r0)) if np.isfinite(r0) else float("inf")
    ok = worst <= 20.0
    report(OK if ok else FAIL, "an exploding world stays harmless",
           f"its worst reward over 5 steps: {worst:.3g}"
           + ("" if ok else "  <- one world like this poisons a whole PPO update"))
    env.reset_all()


def check_exploration(seed):
    """Exploration must suit the task, not the one the seed came from."""
    if not seed:
        report(OK, "exploration", "no seed; --init_std applies")
        return
    ck = torch.load(seed, map_location="cpu", weights_only=False)
    sd = ck["model"].get("log_std")
    if sd is None:
        report(WARN, "exploration", "checkpoint has no log_std")
        return
    std = float(sd.exp().mean())
    level = OK if std > 0.15 else WARN
    report(level, "exploration", f"seed std {std:.3f} (task {ck.get('task', '?')})"
           + ("" if level == OK else "  <- collapsed; use --reset_std for a cross-task seed"))


def perturb_checks(args):
    """Perturb: a capture step must pay, and the ball must connect and be survivable at all."""
    env = PerturbEnv(num_envs=1, model="dummy_ball.xml", ball_every=(1e6, 1e6))
    return [lambda: check_start_pose(env),
            lambda: check_reward_ranking(env, capture=True),
            lambda: check_step_is_worth_taking(env),
            lambda: check_blowup_is_contained(env),
            lambda: check_exploration(args.seed),
            lambda: check_perturbation(env, [args.speed_end * 0.5, args.speed_end]),
            lambda: check_feasible(PerturbEnv(num_envs=12, episode_seconds=12.0,
                                              model="dummy_ball.xml", ball_every=(1e6, 1e6)),
                                   args.speed_end)]


def walk_checks(args):
    """Walk: walking must out-earn a statue at every commanded speed."""
    env = WalkEnv(num_envs=1, seed=0)
    env.set_command(0.6, 0.0, 0.0)
    return [lambda: check_start_pose(env),
            lambda: check_reward_ranking(env),
            lambda: check_walk_beats_statue(env),
            lambda: check_blowup_is_contained(env),
            lambda: check_exploration(args.seed)]


# Each task's checks, in the order they run. A new task is a new entry, nothing else.
CHECKS = {"perturb": perturb_checks, "walk": walk_checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=tuple(CHECKS), default="perturb")
    p.add_argument("--speed_end", type=float, default=6.0)
    p.add_argument("--seed", default="", help="the checkpoint a run would start from")
    args = p.parse_args()

    print(f"\n  PREFLIGHT - task '{args.task}', curriculum end {args.speed_end:.2f} m/s\n")
    for check in CHECKS[args.task](args):
        check()

    bad = sum(1 for lvl, _, _ in results if lvl == FAIL)
    warn = sum(1 for lvl, _, _ in results if lvl == WARN)
    print(f"\n  -> {bad} failures, {warn} warnings")
    if bad:
        print("     A task that fails preflight cannot be fixed by training. Do not start a run.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
