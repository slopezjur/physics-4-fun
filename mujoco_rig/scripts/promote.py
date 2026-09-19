"""Ship a checkpoint to the Godot scenes only if it actually scores better than what is there.

**Why this is a script and not a habit.** Promotion had been a manual `export_onnx.py` call, and the
manual version put `walk_n_s2` into the scenes instead of `walk_m_s2` - one letter, two entirely
different policies, and the one that shipped walked BACKWARDS at -1.07 m against a forward command
while the intended one travelled +0.58 m. Nothing caught it except scoring them side by side
afterwards. A checkpoint reaches the game by beating the incumbent on a measurement or not at all.

The incumbent is read from the contract that ships beside the policy, so "what is currently in the
scenes" is never a guess. Both are scored in the SAME invocation, same seed, same settings, because
a comparison between two differently-configured runs is what produced the wrong conclusion this
project has retracted most often. How each task is scored, and on what, lives in `scoring.py`.

**Perturb is judged on ONE hit at a hard, fixed impulse, on whether the body survived it, and on a
paired significance test.** The first version scored upright-over-40-s at 2.2 m/s = 11 N.s, where
every candidate reads ~100%: it compared +99.90 against +100.00 and "decided" on the rounding. The
second fired two shots while calling it one, and shipped on +32.0 vs +25.4 at 256 envs two
checkpoints that score 30.1% and 30.5% at 512. Both checkpoints now face the SAME shots (same
seed), so the decision uses the worlds where they disagree: `b` survived only by the challenger,
`c` only by the incumbent, and `z = (b - c) / sqrt(b + c)` must reach 2.

    python mujoco_rig/scripts/promote.py --task perturb --checkpoint logs/mujoco/<run>/model_N.pt
    python mujoco_rig/scripts/promote.py --task walk --checkpoint <ck> --dry_run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scoring import PY, RL, ROOT, TASKS, paired_z, score_all  # noqa: E402


def incumbent_of(scorer):
    """The checkpoint the scenes load now, read from the contract shipped beside the policy."""
    contract = ROOT / "mujoco_rig" / f"{scorer.family}_policy.contract.json"
    if not contract.exists():
        if (contract.parent / f"{scorer.family}_policy.onnx").exists():
            raise ValueError("Shipped policy has no contract; cannot establish the incumbent")
        return None
    shipped = json.loads(contract.read_text(encoding="utf-8")).get("source_checkpoint", "")
    if shipped and (ROOT / shipped).exists():
        return str(ROOT / shipped)
    raise ValueError(f"Incumbent checkpoint {shipped!r} is missing; cannot compare policies")


def ship(checkpoint, family):
    """Export `checkpoint` over the family's policy and contract. Returns the exit code."""
    out = ROOT / "mujoco_rig" / f"{family}_policy.onnx"
    r = subprocess.run([str(PY), "-u", str(RL / "export_onnx.py"),
                        "--checkpoint", checkpoint, "--out", str(out)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode != 0:
        print(r.stdout[-2000:] + r.stderr[-2000:])
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=tuple(TASKS), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--margin", type=float, default=None,
                   help="how much better the challenger must be; defaults per task (scoring.py)")
    p.add_argument("--z", type=float, default=2.0,
                   help="the paired significance a challenger's win must reach, where the task has "
                        "one (perturb)")
    # Scoring settings; each defaults to the task's own (scoring.py).
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--ball_speed", type=float, default=None)
    p.add_argument("--ball_every", type=float, nargs=2, default=None)
    p.add_argument("--walk_speed", type=float, default=None,
                   help="walk: the straight-line command both checkpoints are scored at")
    p.add_argument("--dry_run", action="store_true", help="score and report, ship nothing")
    args = p.parse_args()

    scorer = TASKS[args.task]
    settings = {k: (getattr(args, k) if getattr(args, k, None) is not None else v)
                for k, v in scorer.defaults.items()}
    print(f"[promote] scoring: {scorer.describe(settings)}", flush=True)

    try:
        incumbent = incumbent_of(scorer)
    except (OSError, ValueError) as error:
        print(f"[promote] {error}")
        return 1
    results = score_all(scorer, [args.checkpoint] + ([incumbent] if incumbent else []), settings)
    new = results[0]
    if new is None:
        print("[promote] could not score the challenger")
        return 1
    new_value = scorer.metric(new)
    if not math.isfinite(new_value) or scorer.collapsed(new):
        print("[promote] challenger is non-finite or collapsed")
        return 1
    print(f"[promote] challenger {pathlib.Path(args.checkpoint).name}: "
          f"{new_value:+.2f} {scorer.unit}  {scorer.summary(new)}")

    old_value, win = None, True
    if incumbent:
        old = results[1]
        old_value = scorer.metric(old) if old is not None else None
        if old_value is None or not math.isfinite(old_value):
            print("[promote] incumbent could not be scored; leaving the shipped policy unchanged")
            return 1
        print(f"[promote] incumbent  {pathlib.Path(incumbent).name}: "
              + (f"{old_value:+.2f} {scorer.unit}  {scorer.summary(old)}" if old_value is not None
                 else "unscorable"))
        if old_value is not None:
            win = new_value > old_value + settings["margin"]
            if scorer.paired:
                b, c, z = paired_z(scorer.outcomes(new), scorer.outcomes(old))
                print(f"[promote] paired: challenger alone survived {b}, incumbent alone {c}, "
                      f"z {z:+.2f} (needs {args.z:+.2f})")
                win = win and z >= args.z

    if not win:
        print(f"[promote] NOT shipped: {new_value:+.2f} vs {old_value:+.2f}")
        return 0
    if args.dry_run:
        print("[promote] dry run - would have shipped")
        return 0
    if ship(args.checkpoint, scorer.family) != 0:
        return 1
    print(f"[promote] SHIPPED to {scorer.family}_policy.onnx"
          + (f" ({new_value:+.2f} vs {old_value:+.2f})" if old_value is not None
             else " (slot was empty)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
