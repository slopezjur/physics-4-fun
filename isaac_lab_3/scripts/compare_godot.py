"""Measure two checkpoints against each other IN GODOT, and leave `Models/` exactly as it was.

**The Isaac score does not decide this.** On the assist chain a checkpoint scoring 92.2% held the
Godot dummy at an action authority of 0.10 while a later one scoring 93.8% only held it at 0.07, and
`stand_assist/night06` at 3,087 iterations beats `night10` at 6,272 in Godot despite half the
training. So "is the new one better" is only answerable by running both in the engine that has to
ship them.

    python isaac_lab_3/scripts/compare_godot.py --new <model_N.pt> --current <model_M.pt>

For each checkpoint it reports two numbers:

* **authority** - the highest `ActionScaleOverride` at which the dummy still stands. Higher is
  better: it means the body tolerates more of what the policy asks for.
* **push** - the impulse it survives at that authority. This is the one that separates a policy the
  body TOLERATES from one doing work, because a zero-action baseline also survives 4 N.s and falls
  at 8.

**It restores `Models/` on the way out, including on Ctrl+C.** Measuring a checkpoint means
promoting it so Godot loads it, so the comparison necessarily overwrites the very brain it is
comparing against. Nothing under `Models/` is tracked by git, so a comparison that died halfway used
to be unrecoverable - hence the `finally`.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib

from godot_check import PROJECT_ROOT, add_common_args, measure, scene_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new", required=True, help="Checkpoint under test.")
    p.add_argument("--current", default="", help="Checkpoint to beat. Empty = measure --new alone.")
    p.add_argument("--task", default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--brain", default="balance", help="Artifact stem, from export.py's BRAIN map.")
    add_common_args(p)
    return p.parse_args()


@contextlib.contextmanager
def models_restored(brain: str):
    """Put `Models/<brain>_policy.*` back exactly as found, whatever happens in between.

    Not optional. Measuring a checkpoint means promoting it, so this comparison overwrites the
    working brain as its very first act - and `Models/` is untracked, so there is no other copy.
    """
    models = PROJECT_ROOT / "Models"
    originals = [p for p in models.glob(f"{brain}_policy.*") if p.is_file()]
    backup = {p: p.read_bytes() for p in originals}
    try:
        yield
    finally:
        for path, blob in backup.items():
            path.write_bytes(blob)
        # A brain that did not exist before must not be left behind by the comparison.
        for path in models.glob(f"{brain}_policy.*"):
            if path not in backup:
                path.unlink()
        if backup:
            print(f"\n  Models/{brain}_policy.* restored to what it was before the comparison.")


def main() -> None:
    args = parse_args()
    new = pathlib.Path(args.new).resolve()
    current = pathlib.Path(args.current).resolve() if args.current else None

    scene = scene_path()
    scene_backup = scene.read_text(encoding="utf-8")

    rows: list[tuple[str, pathlib.Path, float, float]] = []
    try:
        with models_restored(args.brain):
            for label, checkpoint in (("new", new), ("current", current)):
                if checkpoint is None:
                    continue
                print(f"  measuring {label}: {checkpoint.parent.name}/{checkpoint.name}")
                authority, push = measure(checkpoint, args.task, args.authorities, args.push)
                rows.append((label, checkpoint, authority, push))
                print(f"    authority {authority:.2f}" + (f", survives {push:.0f} N.s" if push else ""))
    finally:
        scene.write_text(scene_backup, encoding="utf-8", newline="\n")

    print(f"\n  {'':<9} {'authority':>9} {'push':>7}   checkpoint")
    for label, checkpoint, authority, push in rows:
        print(f"  {label:<9} {authority:>9.2f} {push:>5.0f} N.s   "
              f"{checkpoint.parent.name}/{checkpoint.name}")

    if len(rows) == 2:
        (_, _, a_new, p_new), (_, _, a_cur, p_cur) = rows
        better = (a_new, p_new) > (a_cur, p_cur)
        same = (a_new, p_new) == (a_cur, p_cur)
        verdict = "BETTER" if better else ("EQUAL" if same else "WORSE")
        print(f"\n  verdict: the new checkpoint is {verdict} than the one currently in Godot.")
        # Exit code is the machine-readable half: 0 better, 1 equal, 2 worse. resume.ps1 reads it
        # to colour the prompt, and a caller that wants to gate automatically can branch on it.
        raise SystemExit(0 if better else (1 if same else 2))


if __name__ == "__main__":
    main()
