"""Score a checkpoint in Godot by whether it WALKS, not by whether it survives.

**The authority ladder measures the wrong thing.** A brain that stands perfectly still for 20 s
passes it; measured 2026-09-04, the promoted walk checkpoint "stood" at authority 0.10 while
covering -0.05 m in 20 s with 2 foot strikes, against Isaac's 10.31 m and 38. Survival is necessary
and nowhere near sufficient, and every checkpoint this project has promoted was selected on it.

Reports displacement, foot strikes and flight fraction from the driver's own observation trace, so
the number cannot be satisfied by a statue.

    python isaac_lab_3/scripts/godot_walk_score.py --checkpoints <model.pt> [<model.pt> ...]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import os
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
ISAAC_PY = r"D:\Programas\anaconda3\envs\env_isaaclab3\python.exe"


def score(trace: pathlib.Path) -> dict:
    rows = list(csv.DictReader(open(trace, encoding="utf-8")))
    if len(rows) < 10:
        return {"displacement": 0.0, "steps": 0, "flight": 0.0, "upright": 0.0, "rows": len(rows)}
    dt = 1.0 / 60.0
    vx = [float(r["linVel_x"]) for r in rows]
    feet = [(float(r["c_FootL"]), float(r["c_FootR"])) for r in rows]
    steps = sum(
        1
        for i in range(1, len(feet))
        if (feet[i][0] > 0.5 and feet[i - 1][0] < 0.5) or (feet[i][1] > 0.5 and feet[i - 1][1] < 0.5)
    )
    flight = sum(1 for a, b in feet if a < 0.5 and b < 0.5) / len(feet)
    upright = sum(1 for r in rows if float(r["height"]) > 0.6) / len(rows)
    # **Distance only counts while the dummy is still standing.** Measured 2026-09-05, a checkpoint
    # scored +1.01 m at authority 0.15 with 6.1% uprightness: it fell over and slid a metre. Total
    # displacement is satisfiable by a forward tumble, which is exactly the kind of number that
    # reads as progress and is not.
    upright_dx = sum(
        v for v, r in zip(vx, rows) if float(r["height"]) > 0.6
    ) * dt

    return {
        "upright_displacement": upright_dx,
        "displacement": sum(vx) * dt,
        "mean_vx": statistics.mean(vx),
        "steps": steps,
        "flight": flight,
        "upright": upright,
        "rows": len(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--task", default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--authorities", nargs="+", type=float, default=[0.10, 0.15])
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--load_compensation", type=float, default=0.0)
    args = p.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="walkscore_"))
    print(f"{'checkpoint':>34s} {'auth':>5s} {'dist(m)':>8s} {'UPdist(m)':>9s} {'steps':>6s} "
          f"{'flight':>7s} {'upright':>8s}")
    best = []
    for ckpt in args.checkpoints:
        # export.py prints a U+2705 on success; captured output on Windows defaults to cp1252
        # and the encode raises INSIDE the child, so the export dies for a tick character.
        child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
        export = subprocess.run(
            [ISAAC_PY, str(HERE / "export.py"), "--task", args.task,
             "--checkpoint", str(pathlib.Path(ckpt).resolve()), "--promote"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
        if export.returncode != 0:
            print(f"{pathlib.Path(ckpt).name:>34s}  EXPORT FAILED: "
                  f"{(export.stderr or '').strip().splitlines()[-1][:70]}")
            continue

        label = f"{pathlib.Path(ckpt).parent.name[-8:]}/{pathlib.Path(ckpt).stem}"
        for a in args.authorities:
            trace = tmp / f"{pathlib.Path(ckpt).stem}_{a}.csv"
            subprocess.run(
                [sys.executable, str(HERE / "godot_run.py"),
                 "--scene", "Walk/IsaacWalkCheckNewton.tscn",
                 "--set", f"ActionScaleOverride = {a}",
                 "--set", f"LoadCompensation = {args.load_compensation}",
                 "--set", f'DofTracePath = "{trace.as_posix()}"',
                 "--seconds", str(args.seconds), "--timeout", "240"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if not trace.is_file():
                print(f"{label:>34s} {a:5.2f}   no trace")
                continue
            s = score(trace)
            print(f"{label:>34s} {a:5.2f} {s['displacement']:8.2f} "
                  f"{s['upright_displacement']:9.2f} {s['steps']:6d} {s['flight']:6.1%} "
                  f"{s['upright']:7.1%}")
            # Rank on distance covered WHILE UPRIGHT, which a tumble cannot collect.
        best.append((s["upright_displacement"] if s["upright"] > 0.9 else -1.0, label, a, s))

    best.sort(reverse=True)
    if best:
        d, label, a, s = best[0]
        print(f"\nBEST that stayed upright: {label} at authority {a} - "
              f"{s['displacement']:.2f} m, {s['steps']} strikes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
