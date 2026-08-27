"""Chained short training segments with an evaluation after each one.

**Why segments rather than one long run.** At ~570k steps/s a 15-minute segment is already ~500M
environment steps, which is twice the entire 2.3.2 Stand run that reached 100% success. Splitting
the night into many short segments instead of a few long ones costs nothing in throughput and buys
the thing that actually matters here: a scored checkpoint every quarter hour, so a plateau, a
divergence or a regression is visible within minutes instead of at the end.

Each segment resumes from the previous one's final checkpoint, so this is one continuous training
run - not independent restarts. Progress accumulates; only the observation cadence changes.

After every segment it scores the checkpoint with `evaluate_stand.py` against the strict criterion
(head, tilt, speed, held 1.5 s, unbounded arena) and appends one row to a session log. The
zero-action baseline is measured once at the start, because a policy is only meaningful relative to
it - the 2.3.2 track recorded a case where a trained policy scored WORSE than doing nothing while
every quality gate read healthy.

    python isaac_lab_3/scripts/night.py --until 10:00 --minutes 15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ISAAC3_ROOT = HERE.parent
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--experiment", type=str, default="p4f_newton_stand")
    p.add_argument("--minutes", type=float, default=15.0, help="Wall clock per segment.")
    p.add_argument("--num_envs", type=int, default=16384)
    p.add_argument("--until", type=str, default="10:00", help="Local HH:MM to stop by.")
    p.add_argument("--max_segments", type=int, default=100)
    p.add_argument("--eval_envs", type=int, default=256)
    p.add_argument("--resume", type=str, default="", help="Checkpoint to start the chain from.")
    p.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Seed the FIRST segment from another experiment's checkpoint (weights only, fresh "
        "optimizer and exploration noise). Later segments resume normally from their own chain.",
    )
    p.add_argument(
        "--push",
        type=float,
        default=-1.0,
        help="Spawn push in m/s, applied to BOTH training and scoring so the two cannot drift "
        "apart. -1 keeps the task default. A balance policy scored without a disturbance is not "
        "being asked to balance - a statue passes every criterion.",
    )
    p.add_argument("--xpbd_iterations", type=int, default=2)
    return p.parse_args()


def deadline_from(hhmm: str) -> float:
    now = dt.datetime.now()
    hour, minute = (int(x) for x in hhmm.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target.timestamp()


def newest_checkpoint(experiment: str) -> pathlib.Path | None:
    """Highest-numbered checkpoint in the most recently written run directory.

    Highest-numbered, not newest by mtime: `exported/` is written after the last checkpoint and
    would win on timestamp.
    """
    root = ISAAC3_ROOT / "logs" / "rsl_rl" / experiment
    if not root.is_dir():
        return None
    runs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    for run in runs:
        ckpts = list(run.glob("model_*.pt"))
        if not ckpts:
            continue
        return max(ckpts, key=lambda p: int(re.sub(r"\D", "", p.stem)))
    return None


def run(cmd: list[str], env: dict, log: pathlib.Path) -> tuple[int, str]:
    """Run a subprocess, tee its output to `log`, and return (returncode, text)."""
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n\n===== {dt.datetime.now():%H:%M:%S}  {' '.join(cmd[1:])}\n")
        proc = subprocess.Popen(
            cmd, cwd=str(ISAAC3_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        )
        chunks = []
        for line in proc.stdout:  # type: ignore[union-attr]
            chunks.append(line)
            fh.write(line)
        proc.wait()
    return proc.returncode, "".join(chunks)


def main() -> None:
    args = parse_args()
    stop_at = deadline_from(args.until)
    session = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ISAAC3_ROOT / "logs" / "night" / session
    out_dir.mkdir(parents=True, exist_ok=True)
    console = out_dir / "console.log"
    results = out_dir / "results.jsonl"
    summary = out_dir / "SUMMARY.md"

    env = dict(os.environ, P4F_XPBD_ITERATIONS=str(args.xpbd_iterations), PYTHONIOENCODING="utf-8")

    def note(msg: str) -> None:
        line = f"[night {dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(console, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    note(f"session {session}; segments of {args.minutes} min until {args.until}; "
         f"{args.num_envs} envs, xpbd iterations {args.xpbd_iterations}")

    # Baseline first. Every later row is meaningless without it.
    note("scoring the zero-action baseline ...")
    code, text = run(
        [PYTHON, str(HERE / "evaluate_stand.py"), "--task", args.task, "--zero_action",
         "--num_envs", str(args.eval_envs), "--json", str(results)],
        env, console)
    for line in text.splitlines():
        if "standing success" in line or "mean head height" in line:
            note("  baseline " + line.strip())

    rows: list[dict] = []
    resume = args.resume or (newest_checkpoint(args.experiment) or "")
    if resume:
        note(f"chain starts from {pathlib.Path(resume).name}")
    elif args.init_from:
        note(f"chain seeds from {pathlib.Path(args.init_from).name} (weights only)")

    for segment in range(1, args.max_segments + 1):
        remaining_min = (stop_at - time.time()) / 60.0
        if remaining_min <= 2.0:
            note(f"{remaining_min:.1f} min left - stopping.")
            break
        minutes = min(args.minutes, remaining_min - 1.0)

        note(f"--- segment {segment}: training {minutes:.1f} min "
             f"({remaining_min:.0f} min left in the session)")
        cmd = [PYTHON, str(HERE / "train.py"), "--task", args.task,
               "--experiment", args.experiment,
               "--num_envs", str(args.num_envs), "--max_minutes", f"{minutes:.2f}",
               "--run_name", f"night{segment:02d}", "--push", f"{args.push:.3f}"]
        if resume:
            cmd += ["--resume", str(resume)]
        elif segment == 1 and args.init_from:
            cmd += ["--init_from", args.init_from]
        code, text = run(cmd, env, console)
        if code != 0:
            note(f"training exited {code} - see {console}. Stopping the chain.")
            break

        ckpt = newest_checkpoint(args.experiment)
        if ckpt is None:
            note("no checkpoint produced - stopping.")
            break
        resume = str(ckpt)

        note(f"    evaluating {ckpt.name} ...")
        code, text = run(
            [PYTHON, str(HERE / "evaluate_stand.py"), "--task", args.task,
             "--checkpoint", str(ckpt), "--num_envs", str(args.eval_envs), "--json", str(results),
             "--push", f"{args.push:.3f}"],
            env, console)

        row: dict = {"segment": segment, "checkpoint": ckpt.name}
        for key, pattern in (
            ("standing", r"standing success\s*:\s*([0-9.]+)"),
            ("fell", r"ever fell\s*:\s*([0-9.]+)"),
            ("head", r"mean head height\s*:\s*([0-9.]+)"),
            ("action", r"mean \|action\|\s*:\s*([0-9.]+)"),
        ):
            m = re.search(pattern, text)
            row[key] = float(m.group(1)) if m else float("nan")
        # Training-side numbers, taken from the segment that just ran.
        for key, pattern in (
            ("ep_len", r"Mean episode length:\s*([0-9.]+)"),
            ("reward", r"Mean reward:\s*(-?[0-9.]+)"),
            ("divergences", r"Diagnostics/divergences:\s*([0-9.]+)"),
        ):
            found = re.findall(pattern, text)
            row[key] = float(found[-1]) if found else float("nan")
        rows.append(row)

        note(f"    standing {row['standing']:.1f}%  head {row['head']:.3f}  "
             f"|a| {row['action']:.3f}  fell {row['fell']:.1f}%")

        write_summary(summary, session, args, rows)

    write_summary(summary, session, args, rows)
    note(f"done. {len(rows)} segments. summary -> {summary}")


def write_summary(path: pathlib.Path, session: str, args, rows: list[dict]) -> None:
    lines = [
        f"# Night session {session}",
        "",
        f"Task `{args.task}`, {args.num_envs} envs, {args.minutes:.0f}-minute segments, "
        f"XPBD iterations {args.xpbd_iterations}.",
        "",
        "Scored with the strict criterion from `UprightTermination`: head >= 1.35 m, tilt <= 30 deg,",
        "CoM speed <= 0.6 m/s, **held continuously for 1.5 s**, in an unbounded arena (no resets).",
        "The zero-action baseline for this rig is **0.0% standing** - it is on the floor in under 2 s.",
        "",
        f"Spawn push while scoring: **{'the task default' if args.push < 0 else f'{args.push:.2f} m/s'}**.",
        "A balance policy scored WITHOUT a disturbance is not being asked to balance: a statue",
        "satisfies every one of those four conditions, and one did - 100% here, 0% in Godot.",
        "",
        "| seg | checkpoint | standing % | ever fell % | head (m) | mean \\|a\\| | ep_len | reward | div |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['segment']} | `{r['checkpoint']}` | **{r['standing']:.1f}** | {r['fell']:.1f} | "
            f"{r['head']:.3f} | {r['action']:.3f} | {r['ep_len']:.0f} | {r['reward']:.2f} | "
            f"{r['divergences']:.0f} |"
        )
    lines += [
        "",
        "`head` near 1.540 means balancing; materially below means surviving in a crouch.",
        "`mean |a|` near 1.0 means bracing rather than balancing, and bracing is what stops",
        "transferring when the solver underneath changes. `div` is non-finite environments caught",
        "by the NaN guard - a rising count means the physics needs attention, not the policy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
