"""Where is the useful batch on THIS plant? One table, measured, same seed, same wall-clock.

**This is calibration, not training.** It exists to give `train.py --steps_schedule` its endpoints:
the lowest rung has to be a batch that still learns rather than collapsing, and the highest has to
be one that is actually buying something. Both numbers are properties of the gradient, and the
gradient changed when the plant did - the batch-collapse threshold currently written into
`config.ps1` was measured on the POSITION-actuated body, which no longer exists.

Every cell is the same seed, the same minutes and one trainer at a time; the only difference is the
envs x steps split. Cells are scored on CPU MuJoCo like everything else.

**Two questions, and the grid answers both:**

  * cutting STEPS at a fixed env count - the cheap axis, since `ms per policy step = 35.8 +
    0.04157 x envs` makes envs nearly free and steps cost 36 ms each, sequentially;
  * one pair at equal batch and different split, because at the old threshold the splits
    DISAGREED (32,768x4 kept 98% of its peak, 16,384x8 kept 7%), so batch alone did not predict
    stability and it may not here either.

    python mujoco_rig/scripts/batch_table.py --task walk --minutes 10 --seed <checkpoint>
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RL = ROOT / "mujoco_rig" / "rl"
PY = sys.executable

GRID = [(16384, 4), (16384, 8), (16384, 16), (4096, 16)]


def run(cmd, log):
    with open(log, "w") as fh:
        subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                       stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("perturb", "walk"), default="walk")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--seed", required=True)
    p.add_argument("--extra", default="")
    args = p.parse_args()

    out = pathlib.Path(os.environ.get("P4F_NIGHT", ROOT / "logs" / "night"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for envs, steps in GRID:
        tag = f"bt_{envs}x{steps}"
        log = out / f"{tag}.log"
        cmd = [PY, "-u", RL / "train.py", "--task", args.task, "--backend", "warp",
               "--num_envs", envs, "--steps", steps, "--iterations", 1000000,
               "--seconds", 20, "--max_minutes", args.minutes, "--init_std", 0.2,
               "--epochs", 5, "--minibatches", 4, "--entropy_coef", 0.005,
               "--init_from", args.seed, "--run_name", tag] + args.extra.split()
        print(f"\n[table] {envs:,} x {steps} = batch {envs*steps:,}", flush=True)
        t0 = time.time()
        run(cmd, log)
        text = log.read_text(errors="replace")
        it = re.findall(r"^it\s+(\d+)\s+return\s+(\S+)\s+ep_len\s+(\S+)/", text, re.M)
        it = [r for r in it if "nan" not in r[1]]
        if not it:
            rows.append((envs, steps, None)); print("[table]   no usable iterations"); continue
        iters = int(it[-1][0])
        # Peak and end, because a collapsed cell has a fine peak - that is the whole signature.
        eps = [float(r[2]) for r in it]
        vx = re.findall(r"vx\s+([+-][0-9.]+)", text)
        rows.append((envs, steps, dict(
            iters=iters, ipm=iters / max((time.time() - t0) / 60.0 - 0.4, 0.1),
            peak=max(eps), end=sum(eps[-3:]) / 3,
            vx=float(vx[-1]) if vx else float("nan"))))
        r = rows[-1][2]
        print(f"[table]   {r['iters']} iters, {r['ipm']:.1f} it/min, peak {r['peak']:.0f}, "
              f"end {r['end']:.0f} ({r['end']/max(r['peak'],1):.0%} retained), vx {r['vx']:+.2f}",
              flush=True)

    print("\n  envs x steps      batch    it/min   updates/h   peak ep_len   end   retained   vx")
    for envs, steps, r in rows:
        if r is None:
            print(f"  {envs:>6}x{steps:<3} {envs*steps:>10,}        FAILED"); continue
        print(f"  {envs:>6}x{steps:<3} {envs*steps:>10,}   {r['ipm']:6.1f}   {r['ipm']*60:8.0f}   "
              f"{r['peak']:11.0f}   {r['end']:5.0f}   {r['end']/max(r['peak'],1):7.0%}   "
              f"{r['vx']:+.2f}")
    print("\n  **Read the RETAINED column, not the peak.** A collapsed cell reaches a fine peak and")
    print("  then loses it; that is what 'the batch is below the critical size' looks like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
