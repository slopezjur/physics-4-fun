"""Chain short training sessions unattended, scoring each one before deciding what to chain from.

**Why sessions rather than one long run.** A 15-minute session that is scored is a measurement; a
nine-hour run that is not is a hope. Every session ends with a scorer on the CPU engine - the one
Godot ships - and the result decides whether the next session continues from the new checkpoint or
falls back to the last one that actually scored better. A run that ends worse than it started is not
a foundation to build on, and this project has spent whole evenings on exactly that.

What a task is scored on is `scoring.py`'s. What a chain does for a task beyond training - its extra
train.py arguments, when a running session is hopeless, how a finished one is judged and where the
next one starts - is a `SessionPolicy` here.

    python mujoco_rig/scripts/overnight.py --task perturb --sessions 12 --minutes 15 \
        --seed logs/mujoco/<run>/model_N.pt
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scoring import (PY, RL, ROOT, TASKS, newest_checkpoint, newest_run,  # noqa: E402
                     run, run_concurrently)

# One training-log line: iteration, return, episode length, and the walk's forward speed.
ROW = re.compile(r"^it\s+(\d+)\s+return\s+(\S+)\s+ep_len\s+(\S+)/\d+\s+"
                 r"(?:vx\s+(\S+)|speed\s+\S+)", re.M)


class SessionPolicy(ABC):
    """What a chain does for one task beyond training itself."""

    task = ""
    tolerance = 0.0    # how far below the best a session may score and still be chained from

    def train_args(self, speed_now, args):
        """Extra train.py arguments for the next session."""
        return []

    def abort_verdict(self, rows, text, args):
        """A reason to stop a running session early, or None."""
        return None

    @abstractmethod
    def score(self, ck, speed, speed_now, name, scratch, args):
        """Judge a finished session: (metric, table detail, next session's difficulty, why it
        cannot seed the next session or None)."""
        raise NotImplementedError


class PerturbSessions(SessionPolicy):
    task = "perturb"
    # Only a CLEAR regression is rejected - about four standard errors of single-hit survival at the
    # default 512 envs. A session whose episode length went 252 -> 735 was once discarded over a
    # 0.9-point difference, which is noise, and the chain then could not advance at all.
    tolerance = 8.0

    def train_args(self, speed_now, args):
        # **The curriculum position travels with the chain.** train.py starts at --speed_start, so
        # without this every session would quietly go back to the start of the curriculum.
        #
        # **At most one step per session.** Every session on 2026-09-10 climbed two whether or not
        # the policy had mastered the stage, and handed the next a difficulty it had never beaten.
        # Where the next session starts is decided from measurement, in `score`.
        return ["--ball_every", args.ball_every[0], args.ball_every[1],
                "--speed_start", speed_now,
                "--speed_end", min(args.speed_end, speed_now * args.speed_step),
                "--speed_step", args.speed_step,
                "--stage_min_episodes", 3000, "--stage_min_iters", 15]

    def abort_verdict(self, rows, text, args):
        # The collapse signature: far below its own peak, and staying there.
        ep = [float(r[2]) for r in rows]
        peak = max(ep)
        if peak < 120:
            return None
        recent = ep[-3:]
        if sum(recent) / 3 < 0.5 * peak and max(recent) < 0.8 * peak:
            return f"ep_len {sum(recent) / 3:.0f} against a peak of {peak:.0f}"
        return None

    def score(self, ck, speed, speed_now, name, scratch, args):
        # Judged on one hit at the FIXED reference, and at the session's own difficulty - the
        # second number decides where the next session trains. Both run at once.
        scorer = TASKS[self.task]
        ref_json, stage_json = scratch / f"{name}.score.json", scratch / f"{name}.stage.json"
        codes = run_concurrently([
            (scorer.command(ck, ref_json, dict(envs=args.score_envs, ball_speed=args.speed_ref)),
             scratch / f"{name}.score"),
            (scorer.command(ck, stage_json, dict(envs=args.score_envs, ball_speed=float(speed))),
             scratch / f"{name}.score_stage")])
        ref = scorer.read(ref_json) if codes[0] == 0 else None
        here = scorer.read(stage_json) if codes[1] == 0 else None
        metric = scorer.metric(ref) if ref else float("nan")
        detail = (f"{ref['survived']:.1f}% one hit at {args.speed_ref:.2f} "
                  f"(aimed {ref['aimed']:.0f}%, no-step {ref['no_step']:.0f}%, "
                  f"stance {ref.get('stance', float('nan')):.2f}) | "
                  + (f"{here['survived']:.1f}% at {speed}" if here else "-")
                  if ref else "- | -")
        if here is None:
            # Nothing measured at the session's own difficulty: keep the stage the trainer reached.
            import torch
            reached = torch.load(ck, map_location="cpu", weights_only=False).get("ball_speed")
            return metric, detail, float(reached or speed_now), None
        # **The difficulty follows measured competence, in both directions.** It used to be
        # inherited from the checkpoint, so it could only ratchet: four sessions climbed
        # 2.66 -> 5.70 m/s while survival at the training difficulty fell 64.5% -> 25.2%.
        reached = float(speed)
        if here["survived"] >= args.advance_at:
            nxt, move = min(args.speed_end, reached * args.speed_step), "advance"
        elif here["survived"] < args.descend_below:
            nxt, move = max(args.speed_min, reached / args.speed_step), "descend"
        else:
            nxt, move = reached, "hold"
        return metric, detail + f" -> {move} to {nxt:.2f}", nxt, None


class WalkSessions(SessionPolicy):
    task = "walk"
    tolerance = 5.0    # points of the joystick score

    def abort_verdict(self, rows, text, args):
        # **A statue is standing AND not moving.** Episode length cannot see a statue - three walk
        # sessions raised it 573 -> 736 while travelling zero metres - and a body that keeps
        # falling, or is still learning to stand under a new command, also reads vx ~ 0. Only a
        # body that holds long episodes without moving is cut.
        vx = [abs(float(r[3])) for r in rows[-4:] if r[3]]
        ep = [float(r[2]) for r in rows[-4:]]
        m = re.search(r"ep_len\s+\S+/(\d+)", text)
        limit = float(m.group(1)) if m else float("inf")
        if len(vx) == 4 and max(vx) < args.min_vx and min(ep) > 0.5 * limit:
            return (f"a statue: forward speed {max(vx):.3f} m/s over the last 4 reports, below "
                    f"{args.min_vx}, while episodes last {min(ep):.0f} of {limit:.0f}")
        return None

    def score(self, ck, speed, speed_now, name, scratch, args):
        scorer = TASKS[self.task]
        out = scratch / f"{name}.score.json"
        code = run(scorer.command(ck, out, dict(envs=8, walk_speed=args.walk_speed)),
                   scratch / f"{name}.score")
        result = scorer.read(out) if code == 0 else None
        if result is None:
            return float("nan"), "- | -", speed_now, None
        # Two cells, score and detail, like every other row of the table.
        metric = scorer.metric(result)
        m = result["manoeuvres"]
        s = m["straight_line"]
        detail = (f"{metric:.1f}% of the joystick | straight {s['along']:+.2f} m "
                  f"{s['upright']:.0f}% up; turned L {m['turn_left']['heading_swept_deg']:+.0f} "
                  f"R {m['turn_right']['heading_swept_deg']:+.0f} "
                  f"spin {m['turn_in_place']['heading_swept_deg']:+.0f} deg; stand drift "
                  f"{np.hypot(m['stand_still']['along'], m['stand_still']['across']):.2f} m")
        return metric, detail, speed_now, scorer.collapsed(result)


POLICIES = {p.task: p for p in (PerturbSessions(), WalkSessions())}


def run_watched(cmd, log, policy, args, poll=20.0):
    """Run a session, killing it early once its task says it is going nowhere."""
    with open(log, "w") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    started = time.time()
    verdict = None
    try:
        while proc.poll() is None:
            time.sleep(poll)
            if (time.time() - started) / 60.0 < args.abort_at:
                continue
            text = pathlib.Path(log).read_text(errors="replace")
            # Enough rows to be a judgement rather than a coin flip, and none of the seeded start's NaN.
            rows = [r for r in ROW.findall(text) if "nan" not in r[1]]
            verdict = policy.abort_verdict(rows, text, args) if len(rows) >= 6 else None
            if verdict:
                print(f"[night] ABORTED at {(time.time() - started) / 60:.1f} min: {verdict}",
                      flush=True)
                proc.kill()
                proc.wait(timeout=30)
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return verdict or (f"trainer exited with code {proc.returncode}" if proc.returncode else None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=tuple(POLICIES), required=True)
    p.add_argument("--seed", default="", help="checkpoint to start the first session from")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--tag", default="")
    p.add_argument("--envs", type=int, default=16384)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--ball_speed", type=float, default=None,
                   help="the first session's difficulty. Given explicitly it wins over the seed's "
                        "stored one; it used to be silently replaced by it.")
    p.add_argument("--ball_every", type=float, nargs=2, default=(4.0, 7.0))
    p.add_argument("--speed_end", type=float, default=6.0)
    p.add_argument("--score_envs", type=int, default=512,
                   help="envs in the scoring run. At 128 the same checkpoint read 19.5%% and 32.0%% "
                        "on two runs; 512 puts one standard error near 2 points on a survival rate.")
    p.add_argument("--tolerance", type=float, default=None,
                   help="how far below the best a session may score and still be chained from; "
                        "defaults per task. Perturb is in percentage points, walk in "
                        "upright-weighted metres - one number sized for both is how a "
                        "backwards-walking session once got chained.")
    p.add_argument("--speed_ref", type=float, default=6.0,
                   help="the FIXED difficulty every session is judged at, on ONE hit. Scoring at "
                        "its own curriculum stage makes the sessions incomparable: a session that "
                        "earns a promotion is then scored on a harder task and looks worse than "
                        "the one before it, and a chain-only-if-improved rule rejects exactly the "
                        "sessions that made progress.")
    p.add_argument("--speed_step", type=float, default=1.10,
                   help="one curriculum step, passed to train.py so the two agree what it is")
    p.add_argument("--speed_min", type=float, default=2.0,
                   help="the curriculum never descends below this")
    p.add_argument("--advance_at", type=float, default=50.0,
                   help="single-hit survival %% at the session's own difficulty needed to advance")
    p.add_argument("--descend_below", type=float, default=25.0,
                   help="below this the next session trains one step EASIER. Nothing could lower "
                        "the difficulty before: four sessions climbed 2.66 -> 5.70 m/s while "
                        "survival at the training difficulty fell 64.5%% -> 25.2%%.")
    p.add_argument("--extra", default="", help="passed through to train.py")
    p.add_argument("--reset_std", action="store_true",
                   help="re-inflate exploration on the FIRST session, for a cross-task seed")
    p.add_argument("--entropy_coef", type=float, default=0.0005)
    p.add_argument("--abort_at", type=float, default=5.0,
                   help="minutes into a session before the early-abort check starts. 0 disables. "
                        "Three statue sessions in a row cost 45 minutes; this cuts them at 5.")
    p.add_argument("--walk_speed", type=float, default=0.6,
                   help="walk only: the straight-line command sessions and promotion are scored "
                        "at. Score the speed the stage trains - stage 1 is 0.15-0.35 m/s.")
    p.add_argument("--min_vx", type=float, default=0.05,
                   help="walk only: forward speed below which a standing session is a statue")
    p.add_argument("--no_promote", action="store_true",
                   help="do not offer the run's best checkpoint to the scenes at the end")
    p.add_argument("--note", default="", help="one line written into the log header")
    args = p.parse_args()
    policy = POLICIES[args.task]
    if args.tolerance is None:
        args.tolerance = policy.tolerance

    scratch = pathlib.Path(os.environ.get("P4F_NIGHT", ROOT / "logs" / "night"))
    scratch.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"night_{args.task}"
    table = scratch / f"{tag}.md"
    with open(table, "a") as fh:
        fh.write(f"\n## {tag} - {time.strftime('%Y-%m-%d %H:%M')}   {args.note}\n\n")
        fh.write("| session | minutes | iters | ep_len | difficulty | score | detail | note |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")

    seed = args.seed
    best = None
    best_checkpoint = None
    speed_now = args.ball_speed
    if speed_now is None:
        speed_now = 1.5
        if seed:
            import torch
            speed_now = float(torch.load(seed, map_location="cpu",
                                         weights_only=False).get("ball_speed", speed_now))
    if seed:
        metric, _, _, collapsed = policy.score(seed, str(speed_now), speed_now,
                                               f"{tag}_seed", scratch, args)
        if not math.isfinite(metric) or collapsed:
            print(f"[night] initial seed has no usable score: {collapsed or 'scoring failed'}")
            return 1
        best, best_checkpoint = metric, seed
    for s in range(1, args.sessions + 1):
        name = f"{tag}_s{s}"
        log = scratch / f"{name}.log"
        cmd = [PY, "-u", RL / "train.py", "--task", args.task, "--backend", "warp",
               "--num_envs", args.envs, "--steps", args.steps, "--iterations", 1000000,
               "--seconds", 20, "--max_minutes", args.minutes, "--init_std", 0.2,
               "--epochs", 5, "--minibatches", 4, "--entropy_coef", args.entropy_coef,
               "--run_name", name] + policy.train_args(speed_now, args)
        if seed:
            cmd += ["--init_from", seed]
            # Only the FIRST session re-inflates: after that the chain is continuing its own task
            # and its std should carry.
            if s == 1 and args.reset_std:
                cmd += ["--reset_std"]
        cmd += args.extra.split()

        print(f"\n[night] {name}  {args.minutes:.0f} min  seed="
              f"{pathlib.Path(seed).name if seed else 'none'}", flush=True)
        t0 = time.time()
        if args.abort_at > 0:
            aborted = run_watched(cmd, log, policy, args)
        else:
            code = run(cmd, log)
            aborted = f"trainer exited with code {code}" if code else None
        text = log.read_text(errors="replace")
        if aborted:
            with open(table, "a") as fh:
                fh.write(f"| {s} | {(time.time()-t0)/60:.0f} | - | - | - | - | - | "
                         + f"ABORTED: {aborted} |" + chr(10))
            continue                      # same seed, next session
        rows = re.findall(r"^it\s+(\d+)\s+return\s+(\S+)\s+ep_len\s+(\S+)/\d+\s+"
                          r"(?:speed\s+(\S+)|vx\s+(\S+))", text, re.M)
        rows = [(a, b, c, d or "-") for a, b, c, d, _ in rows]
        rows = [r for r in rows if "nan" not in r[2]]
        if not rows:
            print(f"[night] {name} produced no usable iterations - stopping", flush=True)
            with open(table, "a") as fh:
                fh.write(f"| {s} | {args.minutes:.0f} | - | - | - | - | - | FAILED |\n")
            break
        it, _, ep, speed = rows[-1]

        run_dir = newest_run(name)
        ck = newest_checkpoint(run_dir) if run_dir else None
        if ck is None:
            print(f"[night] {name} wrote no checkpoint - stopping", flush=True)
            break

        metric, detail, next_speed, collapsed = policy.score(ck, speed, speed_now, name, scratch,
                                                            args)
        # Chain unless this is a CLEAR regression. `best` still tracks the high-water mark, so a
        # slow drift downward cannot walk the chain away from a good policy one tolerance at a time.
        # **Never chain a crashed or collapsed session.** A tolerance of 3 metres waved through
        # walk_e_s3 - NaN-degraded, 30% upright, hopping backwards - so the next session trained
        # from a broken checkpoint. A session whose trainer died, or whose result the task calls
        # collapsed, is never the seed, whatever its score.
        crashed = "Traceback" in text
        ok = (math.isfinite(metric) and not crashed and not collapsed
              and (best is None or metric >= best - args.tolerance))
        note = ("chained" if ok
                else "REJECTED: the trainer crashed" if crashed
                else f"REJECTED: {collapsed}" if collapsed
                else f"REGRESSED from {best:.1f}, re-chaining from previous" if best is not None
                else "no usable score")
        if ok:
            seed = str(ck)
            speed_now = next_speed
            if best is None or metric > best:
                best = metric
                best_checkpoint = str(ck)
        with open(table, "a") as fh:
            fh.write(f"| {s} | {(time.time()-t0)/60:.0f} | {it} | {ep} | {speed} | "
                     f"{detail} | {note} |\n")
        print(f"[night] {name}: it {it}, ep_len {ep}, difficulty {speed} -> {detail}  ({note})",
              flush=True)

    # Ship the best of the run, but only if it beats what the scenes already load. The decision
    # belongs to a measurement, not to whoever remembers to run export_onnx.py: a manual promotion
    # once put a policy that walked BACKWARDS into the scenes.
    if best_checkpoint and not args.no_promote:
        promoted = subprocess.run([str(PY), "-u", str(ROOT / "mujoco_rig" / "scripts" / "promote.py"),
                        "--task", args.task, "--checkpoint", best_checkpoint,
                        "--walk_speed", str(args.walk_speed)],
                       cwd=str(ROOT), env=dict(os.environ, PYTHONIOENCODING="utf-8"))

        if promoted.returncode:
            return promoted.returncode

    print(f"\n[night] table: {table}")
    if best_checkpoint:
        print(f"[night] best checkpoint: {best_checkpoint}")
    return 0 if best_checkpoint else 1


if __name__ == "__main__":
    raise SystemExit(main())
