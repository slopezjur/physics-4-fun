"""Rank checkpoints by how they behave in GODOT, not by their Isaac score.

**The Isaac score stopped predicting the transfer, and that is why this exists.** Measured on the
assist chain: a checkpoint scoring 92.2% in Isaac holds the Godot dummy at an action authority of
0.10, while a later one scoring 93.8% only holds it at 0.07. The training metric went up while the
thing we care about went down, so every "best checkpoint" decision made from `SUMMARY.md` is being
made on the wrong basis.

For each checkpoint this exports it, promotes it into `Models/`, and runs the Godot check scene
headless at a descending ladder of authorities, reporting the highest one at which the dummy still
stands. Optionally it then runs the push test at that authority, which is the measurement that
separates a policy the body TOLERATES from one that is doing work - both a trained policy and a
zero-action baseline survive 4 N.s and fall at 8, so a checkpoint that beats that is the first one
genuinely contributing.

    python isaac_lab_3/scripts/select_by_godot.py --experiment p4f_newton_stand_assist --last 6

Each checkpoint costs roughly one Godot run per authority tried, about 20 s each, so a six-way
comparison is a few minutes and needs no GPU training at all.

**It rewrites `Models/balance_policy.onnx` as it goes**, finishing by re-promoting the winner - so the
project is left holding the best checkpoint rather than the last one tested.
"""

from __future__ import annotations

import argparse
import pathlib
import re

from godot_check import ISAAC3_ROOT, add_common_args, measure, set_scene


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=str, default="p4f_newton_stand_assist")
    p.add_argument("--last", type=int, default=6, help="How many recent checkpoints to compare.")
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    add_common_args(p)
    return p.parse_args()


def checkpoints(experiment: str, last: int) -> list[pathlib.Path]:
    root = ISAAC3_ROOT / "logs" / "rsl_rl" / experiment
    found: list[pathlib.Path] = []
    for run in sorted(root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not run.is_dir():
            continue
        models = sorted(run.glob("model_*.pt"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
        if models:
            found.append(models[-1])
        if len(found) >= last:
            break
    return found


def main() -> None:
    args = parse_args()
    found = checkpoints(args.experiment, args.last)
    if not found:
        raise SystemExit(f"no checkpoints under logs/rsl_rl/{args.experiment}")

    print(f"Ranking {len(found)} checkpoints by GODOT behaviour, not Isaac score.\n")
    results: list[tuple[float, float, pathlib.Path]] = []

    for checkpoint in found:
        print(f"  {checkpoint.parent.name}/{checkpoint.name}")
        best, held = measure(checkpoint, args.task, args.authorities, args.push)

        results.append((best, held, checkpoint))
        print(f"    highest authority that stands: {best:.2f}"
              + (f"   survives {held:.0f} N.s push" if held else ""))

    if not results:
        raise SystemExit("nothing exported successfully")

    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    print("\n  ranked:")
    for authority, held, checkpoint in results:
        print(f"    {authority:.2f}  {held:>4.0f} N.s  {checkpoint.parent.name}/{checkpoint.name}")

    winner = results[0]
    print(f"\n  promoting the winner: {winner[2].parent.name}/{winner[2].name}")
    measure(winner[2], args.task, [winner[0]], push=0.0)
    set_scene(winner[0], push=0.0)
    print(f"  scene left at ActionScaleOverride = {winner[0]:.2f}")


if __name__ == "__main__":
    main()
