"""Throughput and resource cost per environment count, measured rather than assumed.

Answers one question: what should `$Envs` be in `config.ps1`? The 2.3.2 track records that its own
value was set from a table taken on a different task and was wrong by 2x, so this exists to make
re-measuring cheap whenever the task, the rig or the solver changes.

Each row runs a **fresh subprocess**, so one env count cannot inherit another's allocator state or
fragmented VRAM. The parent samples while it runs:

* **steps/s** — from rsl_rl's own per-iteration report, averaged over the timed iterations with the
  first few skipped (CUDA graph capture and kernel compilation land there and are not steady state).
* **peak process RAM** and **peak system RAM committed** — RAM is the binding constraint on this
  machine, not VRAM, which is not obvious. The 2.3.2 table fits ~18.7 GB fixed for Isaac Sim's
  process; a kit-less Newton run should be far below that, and this measures whether it is.
* **peak VRAM** — from `nvidia-smi`, so it includes the desktop, which is what actually decides
  whether a long run dies on an allocation.
* **peak CPU** — percent of one core, summed across cores.

    python isaac_lab_3/scripts/benchmark.py
    python isaac_lab_3/scripts/benchmark.py --envs 4096 8192 16384 --iterations 30
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ISAAC3_ROOT = HERE.parent

STEPS_RE = re.compile(r"Steps per second:\s*([0-9.]+)")
EPLEN_RE = re.compile(r"Mean episode length:\s*([0-9.]+)")


def gpu_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return float("nan")


class Sampler(threading.Thread):
    """Polls resource use while a child process runs."""

    def __init__(self, proc: subprocess.Popen, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self._proc = proc
        self._interval = interval
        self.stop_flag = threading.Event()
        self.peak_rss = 0.0
        self.peak_sys = 0.0
        self.peak_vram = 0.0
        self.peak_cpu = 0.0

    def run(self) -> None:
        try:
            p = psutil.Process(self._proc.pid)
        except psutil.NoSuchProcess:
            return
        p.cpu_percent(None)
        while not self.stop_flag.is_set() and self._proc.poll() is None:
            try:
                # Include children: warp/torch may spawn workers whose memory is real.
                rss = p.memory_info().rss
                cpu = p.cpu_percent(None)
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                        cpu += child.cpu_percent(None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_rss = max(self.peak_rss, rss / 2**30)
                self.peak_cpu = max(self.peak_cpu, cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self.peak_sys = max(self.peak_sys, psutil.virtual_memory().used / 2**30)
            self.peak_vram = max(self.peak_vram, gpu_used_mib() / 1024.0)
            time.sleep(self._interval)


def run_one(num_envs: int, iterations: int, warmup: int, task: str, xpbd_iters: int) -> dict:
    env = dict(os.environ, P4F_XPBD_ITERATIONS=str(xpbd_iters))
    cmd = [
        sys.executable,
        str(HERE / "train.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--iterations", str(iterations),
        "--run_name", f"bench_{num_envs}",
    ]
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(ISAAC3_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    sampler = Sampler(proc)
    sampler.start()

    rates: list[float] = []
    ep_lens: list[float] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        m = STEPS_RE.search(line)
        if m:
            rates.append(float(m.group(1)))
        m = EPLEN_RE.search(line)
        if m:
            ep_lens.append(float(m.group(1)))
    proc.wait()
    sampler.stop_flag.set()
    sampler.join(timeout=2.0)

    # Drop the warm-up iterations: CUDA graph capture and kernel compilation land in the first few
    # and are one-off costs, not throughput.
    steady = rates[warmup:] or rates
    return {
        "envs": num_envs,
        "ok": proc.returncode == 0,
        "steps_s": sum(steady) / len(steady) if steady else float("nan"),
        "peak_steps_s": max(steady) if steady else float("nan"),
        "ep_len": ep_lens[-1] if ep_lens else float("nan"),
        "rss": sampler.peak_rss,
        "sys_ram": sampler.peak_sys,
        "vram": sampler.peak_vram,
        "cpu": sampler.peak_cpu,
        "wall": time.time() - started,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--xpbd_iterations", type=int, default=2)
    args = p.parse_args()

    idle_vram = gpu_used_mib() / 1024.0
    idle_ram = psutil.virtual_memory().used / 2**30
    print(f"idle baseline: {idle_ram:.1f} GB RAM, {idle_vram:.1f} GB VRAM "
          f"(desktop and anything else already running)\n")

    rows = []
    for n in args.envs:
        print(f"[bench] {n} envs, {args.iterations} iterations ...", flush=True)
        row = run_one(n, args.iterations, args.warmup, args.task, args.xpbd_iterations)
        rows.append(row)
        status = "" if row["ok"] else "  <- FAILED"
        print(f"        {row['steps_s']:,.0f} steps/s, {row['sys_ram']:.1f} GB RAM, "
              f"{row['vram']:.1f} GB VRAM, {row['wall']:.0f}s{status}", flush=True)

    print(f"\n{'envs':>8}{'steps/s':>12}{'vs 4096':>10}{'proc RAM':>11}{'sys RAM':>10}"
          f"{'VRAM':>9}{'CPU%':>8}{'ep len':>9}")
    base = rows[0]["steps_s"] if rows else float("nan")
    for r in rows:
        rel = r["steps_s"] / base if base == base else float("nan")
        print(f"{r['envs']:>8}{r['steps_s']:>12,.0f}{rel:>9.2f}x{r['rss']:>10.1f}G"
              f"{r['sys_ram']:>9.1f}G{r['vram']:>8.1f}G{r['cpu']:>8.0f}{r['ep_len']:>9.1f}")

    print("\nRead the throughput column against the RAM and VRAM ones, not on its own: the 2.3.2")
    print("track's ceiling was system RAM, and an env count that fits at startup can still die on")
    print("an allocation hours later. Leave headroom.")


if __name__ == "__main__":
    main()
