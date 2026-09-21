"""How each task is scored - in one place, for every tool that decides on a score.

Two tools make decisions from a scorer's result: `promote.py` (does a checkpoint reach the game?)
and `overnight.py` (which checkpoint does the next session start from?). Each used to carry its own
copy of how to run `eval.py` / `eval_walk.py`, its own regular expressions over their PRINTED text,
and its own idea of what the number meant. Printed text is a poor interface: reading the quiet
room's block for the under-fire one produced this project's worst retraction - "100% upright at
6 m/s" for a policy that fell to every hit. The scorers now write their results as JSON (`--json`),
and a `Scorer` here is the only thing that reads them.

A task is one `Scorer`; adding a task means adding one here, and neither tool changes.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from contextlib import ExitStack

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RL = ROOT / "mujoco_rig" / "rl"
PY = sys.executable
LOGS = ROOT / "logs" / "mujoco"

sys.path.insert(0, str(RL))
from env_config import POLICY_FAMILY  # noqa: E402
from seed_validation import quiet_rejection  # noqa: E402


# ---------------------------------------------------------------- processes
def run(cmd, log):
    """Run `cmd` to completion with its output in `log`; return the exit code."""
    with open(log, "w") as fh:
        return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode


def run_concurrently(jobs):
    """run() for several (command, log) pairs at once. Scoring is single-threaded and CPU-bound, so
    two scorings side by side cost what one did."""
    processes = []
    with ExitStack() as files:
        try:
            for cmd, log in jobs:
                fh = files.enter_context(open(log, "w", encoding="utf-8"))
                processes.append(subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                    stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8")))
            return [proc.wait() for proc in processes]
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()


# ---------------------------------------------------------------- runs on disk
def newest_run(tag):
    """The most recent `logs/mujoco/*_<tag>` run directory, or None."""
    runs = sorted(LOGS.glob(f"*_{tag}"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def newest_checkpoint(run_dir):
    """The highest-numbered `model_N.pt` in a run directory, or None."""
    ck = sorted(pathlib.Path(run_dir).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ck[-1] if ck else None


# ---------------------------------------------------------------- statistics
def paired_z(new_mask, old_mask):
    """McNemar on the worlds where two policies that faced the SAME shots disagree.

    `b` survived only by the challenger, `c` only by the incumbent; `z = (b - c) / sqrt(b + c)`,
    and 0 when they never disagree.
    """
    new_mask, old_mask = np.asarray(new_mask, dtype=bool), np.asarray(old_mask, dtype=bool)
    if new_mask.ndim != 1 or new_mask.shape != old_mask.shape or not new_mask.size:
        raise ValueError("Paired outcomes must be non-empty vectors of the same size")
    b = int(np.sum(new_mask & ~old_mask))
    c = int(np.sum(~new_mask & old_mask))
    return b, c, ((b - c) / np.sqrt(b + c) if b + c else 0.0)


# ---------------------------------------------------------------- tasks
class Scorer(ABC):
    """How one task is scored, and what its score means."""

    task = ""
    unit = ""
    paired = False              # whether per-world outcomes allow a paired test
    defaults: dict = {}         # the scoring run's settings; a caller overrides any of them

    @property
    def family(self):
        """The name its shipped brain carries: `<family>_policy.onnx`."""
        return POLICY_FAMILY[self.task]

    @abstractmethod
    def command(self, checkpoint, json_out, settings):
        """The scorer invocation for `checkpoint`, writing its result to `json_out`."""
        raise NotImplementedError

    @abstractmethod
    def read(self, json_path):
        """The result a decision is made on, or None if the scorer produced none."""
        raise NotImplementedError

    @abstractmethod
    def metric(self, result):
        """The one number promotion and chaining compare."""
        raise NotImplementedError

    def outcomes(self, result):
        """Per-world outcomes for a paired test, where the task has them."""
        return None

    def collapsed(self, result):
        """Why a result is unusable as a seed whatever its metric says, or None."""
        return None

    def regression(self, new, old):
        """An outcome that must not be traded away to improve the primary metric."""
        return None

    def summary(self, result):
        """A few words about a result beyond its metric, for the promotion log."""
        return ""

    @abstractmethod
    def describe(self, settings):
        """One line saying what the scoring run measures."""
        raise NotImplementedError

    def _settings(self, settings):
        return {**self.defaults, **settings}

    @staticmethod
    def _load(json_path, *keys):
        try:
            doc = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
            for k in keys:
                doc = doc[k]
            return doc
        except (OSError, KeyError, ValueError, TypeError):
            return None


class PerturbScorer(Scorer):
    """Survival plus a stable final hold after ONE shot at a fixed impulse.

    **One hit, then time to recover.** Forty seconds of hits every 4-7 s compounds seven of them and
    floors at 0% for every policy, and an easy reference read 81-89% for checkpoints no better than
    the incumbent - the ball had gone 15 kg -> 5 kg, so the "fixed" reference had quietly become an
    easy test. **And it has to BE one hit.** Until 2026-09-10 the "one hit" test fired two: the env
    schedules the next shot after each one, and 6 s at 2.0-2.4 s holds two. `--single_hit`
    suppresses the second, and eval.py reports hits per env so the label is checked, not trusted.

    512 envs puts one standard error near 2 points; launch occurs at 1.5-1.8 s, followed by
    flight time. Success requires a stable final hold and no fall. The margin is 0 because
    the paired z-test sizes it from how much two policies
    actually disagree, instead of a guessed constant that turned out to sit inside the noise.
    """

    task, unit, paired = "perturb", "% settled after one shot", True
    defaults = dict(envs=512, seconds=6.0, ball_speed=6.0, ball_every=(1.5, 1.8),
                            margin=0.0)

    def command(self, checkpoint, json_out, settings):
        s = self._settings(settings)
        return [PY, "-u", RL / "eval.py", "--checkpoint", checkpoint, "--num_envs", s["envs"],
                "--seconds", s["seconds"], "--ball_speed", s["ball_speed"],
                "--ball_every", s["ball_every"][0], s["ball_every"][1],
                "--single_hit", "--quiet_seconds", 40, "--quiet_envs", 16,
                "--no_baseline", "--json", json_out]

    def read(self, json_path):
        # **The section matters.** eval.py can score the quiet room too, and reading that block
        # instead of this one is what produced "100% upright at 6 m/s" for a policy that fell to
        # every hit.
        sections = self._load(json_path, "sections")
        if not sections or 'under_fire' not in sections:
            return None
        quiet = sections.get('no_ball', {})
        return dict(sections['under_fire'], quiet_survived=quiet.get('survived', float('nan')),
                    quiet_recovered=quiet.get('recovered', float('nan')))

    def collapsed(self, result):
        return quiet_rejection(result.get('quiet_survived', float('nan')),
                               result.get('quiet_recovered', float('nan')))

    def metric(self, result):
        return result.get("recovered", float("nan"))

    def outcomes(self, result):
        return np.asarray(result["recovered_mask"], dtype=bool)

    def regression(self, new, old):
        before, after = old.get("survived", float("nan")), new.get("survived", float("nan"))
        if not np.isfinite([before, after]).all():
            return "missing finite survival measurements"
        if after < before:
            return f"survival regressed {before:.1f}% -> {after:.1f}%"
        return None

    def summary(self, result):
        nan = float("nan")
        return (f"(survived {result.get('survived', nan):.1f}%, "
                f"sliding {result.get('foot_sliding_m', nan):.2f} m, "
                f"replants {result.get('rapid_replants', nan):.1f}, "
                f"end stance {result.get('stance', nan):.2f}, width "
                f"{result.get('stance_width', nan):.2f} m, crossed "
                f"{result.get('stance_crossed', nan):.0f}%, trunk {result.get('trunk_rms', nan):.2f} "
                f"rad, {result.get('at_limit', nan):.1f} joints on a limit)")

    def describe(self, settings):
        s = self._settings(settings)
        return (f"{s['envs']} envs x {s['seconds']:.0f} s, ONE ball at {s['ball_speed']:.1f} m/s "
                f"after {s['ball_every'][0]:.1f}-{s['ball_every'][1]:.1f} s, final stable hold 0.5 s")


class WalkScorer(Scorer):
    """How much of the JOYSTICK the policy delivers: eval_walk's five manoeuvres, 0-100.

    Each manoeuvre scores 0-1 for doing what it was told, times the fraction of it spent upright,
    and the score is their mean:

        straight line   1 - |metres along the starting heading / the commanded distance - 1|
        turn left/right 1 - |heading turned / the commanded angle - 1|
        turn in place   the same, with no forward command
        stand still     1 - metres drifted / 2

    **Too much is as wrong as too little.** Each part used to be clipped at 1.0, which made
    overshoot free: a session walked 14.26 m against a commanded 10 m - 43% too fast - and scored a
    perfect straight line.

    **Metres ALONG the heading, never speed.** Raw vx rewarded a body that ran backwards - a session
    scored vx -0.32 against a +0.60 command and was chained - and the pelvis-frame vx cannot see a
    circle either. **All five, not the straight line alone** (which was the score until 2026-09-11):
    the straight line cannot see a turn, so a session that learned to turn and gave up a little
    distance read as a regression.
    """

    task, unit = "walk", "% of the joystick"
    defaults = dict(envs=32, seconds=40.0, walk_speed=0.6, margin=0.0)

    def command(self, checkpoint, json_out, settings):
        s = self._settings(settings)
        return [PY, "-u", RL / "eval_walk.py", "--checkpoint", checkpoint, "--num_envs", s["envs"],
                "--seconds", s["seconds"], "--speed", s["walk_speed"], "--json", json_out]

    def read(self, json_path):
        return self._load(json_path)

    @staticmethod
    def parts(result):
        """Each manoeuvre's 0-1 score, before weighting by uprightness."""
        seconds = result["settings"]["seconds"]
        m = result["manoeuvres"]

        def clip(x):
            return float(min(max(x, 0.0), 1.0))

        def followed(achieved, commanded):
            return clip(1.0 - abs(achieved / commanded - 1.0))

        def turned(r):
            return followed(np.radians(r["heading_swept_deg"]), r["cmd"][2] * seconds)

        s, still = m["straight_line"], m["stand_still"]
        return {"straight_line": followed(s["along"], s["cmd"][0] * seconds),
                "turn_left": turned(m["turn_left"]),
                "turn_right": turned(m["turn_right"]),
                "turn_in_place": turned(m["turn_in_place"]),
                "stand_still": clip(1.0 - np.hypot(still["along"], still["across"]) / 2.0)}

    def metric(self, result):
        m = result["manoeuvres"]
        return 100.0 * float(np.mean([p * max(m[k]["upright"], 0.0) / 100.0
                                      for k, p in self.parts(result).items()]))

    def collapsed(self, result):
        # A walk that cannot stay upright on the straight line is never a seed: a 3-metre tolerance
        # once waved through a NaN-degraded, 30%-upright checkpoint that hopped backwards.
        straight = result["manoeuvres"]["straight_line"]
        return "upright under 50% on the straight line" if straight["upright"] < 50.0 else None

    def describe(self, settings):
        s = self._settings(settings)
        return (f"{s['envs']} envs x {s['seconds']:.0f} s, five manoeuvres (straight line at "
                f"{s['walk_speed']:.2f} m/s, two arcs, turn in place, stand still)")


TASKS = {s.task: s for s in (PerturbScorer(), WalkScorer())}


def score_all(scorer, checkpoints, settings):
    """Score several checkpoints CONCURRENTLY with the same settings and seed - so perturb candidates
    face the same shots - and return each one's result, None where the scorer produced none."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="score_"))
    try:
        jobs = [(scorer.command(ck, tmp / f"{k}.json", settings), tmp / f"{k}.log")
                for k, ck in enumerate(checkpoints)]
        codes = run_concurrently(jobs)
        results = [scorer.read(tmp / f"{k}.json") if code == 0 else None
                   for k, code in enumerate(codes)]
        for k, result in enumerate(results):
            if result is None:
                tail = (tmp / f"{k}.log").read_text(errors="replace")[-1500:]
                print(f"[score] no result for {checkpoints[k]}:\n{tail}", flush=True)
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
