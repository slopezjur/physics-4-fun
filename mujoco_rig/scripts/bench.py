"""Find the fastest training configuration on THIS machine, and prove nothing is degrading.

Throughput alone is the wrong target, which is why this reports two rates side by side:

  * **samples/s** - how much experience the sim produces.
  * **updates/s** - how many times PPO actually moves the policy.

Measured 2026-09-09, those two disagree and the disagreement decides the setting. 4,096 envs x 24
steps reached episode length 897 in 1.6 min while the 96-env CPU run reached 1,102 in 1.5 min: 42x
the samples, no faster learning, because PPO advances only as far as its KL cap allows and a bigger
batch buys a cleaner gradient rather than a longer step. A configuration that maximises samples/s
while starving updates/s trains more slowly than the CPU it replaced.

Each configuration runs in its own SUBPROCESS. Measuring several in one process would let CUDA
fragmentation and warp's cached allocations from an earlier configuration contaminate a later one,
and peak VRAM would be meaningless.

Resources are sampled while each run is in flight - process CPU and RSS via psutil, GPU utilisation
and VRAM via nvidia-smi - so a configuration that only looks fast because it is about to exhaust
memory is visible as such.

    python mujoco_rig/scripts/bench.py --seconds 45
    python mujoco_rig/scripts/bench.py --seconds 60 --grid full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RL = ROOT / "mujoco_rig" / "rl"

# (backend, envs, steps). The CPU row is the control the GPU has to beat, not a candidate.
GRID_QUICK = [("cpu", 96, 8)] + [("warp", e, s) for e in (2048, 4096, 8192) for s in (4, 8, 16)]
GRID_FULL = [("cpu", 96, 8), ("cpu", 96, 24)] + \
            [("warp", e, s) for e in (2048, 4096, 8192, 16384) for s in (4, 8, 16, 24)]


def gpu_sample():
    """(utilisation %, VRAM MiB) or (None, None). nvidia-smi because pynvml is not installed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        util, mem = (x.strip() for x in out.split(","))
        return float(util), float(mem)
    except Exception:
        return None, None


def run_single(backend, envs, steps, seconds, python):
    """One configuration, in a clean process, sampled while it runs."""
    import psutil

    cmd = [python, "-u", str(pathlib.Path(__file__).resolve()), "--single",
           "--backend", backend, "--envs", str(envs), "--steps", str(steps),
           "--seconds", str(seconds)]
    env_vars = {"PYTHONIOENCODING": "utf-8"}
    import os
    full_env = dict(os.environ, **env_vars)

    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=full_env)
    ps = psutil.Process(proc.pid)
    ncpu = psutil.cpu_count()
    peak = {"cpu": 0.0, "rss": 0.0, "gpu": 0.0, "vram": 0.0, "sys_ram": 0.0}
    samples = {"gpu": [], "cpu": []}
    stop = threading.Event()

    def monitor():
        ps.cpu_percent(None)                       # prime the counter
        while not stop.is_set():
            try:
                # Process CPU can exceed 100% (it is per-core); normalise to % of the machine so a
                # single-threaded loop on 16 threads reads ~6% rather than a misleading 100%.
                cpu = ps.cpu_percent(None) / max(ncpu, 1)
                rss = ps.memory_info().rss / 2 ** 30
                peak["cpu"] = max(peak["cpu"], cpu)
                peak["rss"] = max(peak["rss"], rss)
                samples["cpu"].append(cpu)
            except Exception:
                pass
            peak["sys_ram"] = max(peak["sys_ram"], psutil.virtual_memory().used / 2 ** 30)
            util, vram = gpu_sample()
            if util is not None:
                peak["gpu"] = max(peak["gpu"], util)
                peak["vram"] = max(peak["vram"], vram)
                samples["gpu"].append(util)
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    out, _ = proc.communicate()
    stop.set()
    thread.join(timeout=2)

    result = None
    for line in out.splitlines():
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                pass
    if result is None:
        tail = " | ".join(l for l in out.splitlines()[-4:] if l.strip())
        return {"backend": backend, "envs": envs, "steps": steps, "failed": tail[:200]}

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0        # noqa: E731
    result.update({
        "peak_cpu": peak["cpu"], "mean_cpu": mean(samples["cpu"]),
        "peak_rss": peak["rss"], "peak_sys_ram": peak["sys_ram"],
        "peak_gpu": peak["gpu"], "mean_gpu": mean(samples["gpu"]),
        "peak_vram": peak["vram"],
    })
    return result


def single(args) -> int:
    """Run one configuration and print a JSON line. Executed in the child process."""
    import torch
    sys.path.insert(0, str(RL))
    from ppo import PPO

    if args.backend == "warp":
        from perturb_env_warp import PerturbEnvWarp
        device = "cuda"
        env = PerturbEnvWarp(num_envs=args.envs, device=device, episode_seconds=20.0, seed=0)
    else:
        from perturb_env import PerturbEnv
        device = "cpu"
        env = PerturbEnv(num_envs=args.envs, device=device, episode_seconds=20.0, seed=0)
    algo = PPO(env.num_obs, env.num_actions, device=device)

    def iteration():
        nonlocal obs
        z = dict(device=device)
        obs_buf = torch.zeros(args.steps, env.num_envs, env.num_obs, **z)
        act_buf = torch.zeros(args.steps, env.num_envs, env.num_actions, **z)
        logp_buf = torch.zeros(args.steps, env.num_envs, **z)
        rew_buf = torch.zeros(args.steps, env.num_envs, **z)
        done_buf = torch.zeros(args.steps, env.num_envs, **z)
        val_buf = torch.zeros(args.steps, env.num_envs, **z)
        for t in range(args.steps):
            action, logp, value = algo.act(obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action, logp, value
            obs, reward, done, _ = env.step(action)
            rew_buf[t], done_buf[t] = reward, done.float()
        with torch.no_grad():
            last_value = algo.net.value(obs)
        adv, ret = algo.compute_returns(rew_buf, done_buf, val_buf, last_value)
        algo.update((obs_buf.reshape(-1, env.num_obs), act_buf.reshape(-1, env.num_actions),
                     logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1), val_buf.reshape(-1)))

    obs = env.reset_all()
    # Warm up OUTSIDE the timer. Warp compiles CUDA kernels on first use - up to ~20 s cold - and
    # timing that would report the compiler, not the simulator.
    for _ in range(3):
        iteration()
    if args.backend == "warp":
        torch.cuda.synchronize()

    started = time.perf_counter()
    iters = 0
    while time.perf_counter() - started < args.seconds:
        iteration()
        iters += 1
    if args.backend == "warp":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    buffers = {}
    if args.backend == "warp":
        try:
            buffers = env.assert_buffers_ok()
        except RuntimeError as e:
            buffers = {"overflow": str(e)}

    batch = env.num_envs * args.steps
    print(json.dumps({
        "backend": args.backend, "envs": env.num_envs, "steps": args.steps,
        "batch": batch, "iters": iters, "elapsed": elapsed,
        "sps": iters * batch / elapsed,
        "updates_per_s": iters / elapsed,
        "phys_per_s": iters * batch * env.decimation / elapsed,
        "buffers": buffers,
    }))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend", default="warp")
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--grid", choices=("quick", "full"), default="quick")
    # An explicit grid, e.g. --envs-list 12288,24576,32768 --steps-list 4,8,16,24. Overrides
    # --grid. The saturation point is a property of the GPU and the rig, not something to guess:
    # at 2,048 worlds the 4080 sits at 51% utilisation on 1.2 GiB of 16, so the interesting range
    # is well above where a hand-picked grid tends to stop.
    p.add_argument("--envs-list", default="", help="comma-separated env counts (GPU only)")
    p.add_argument("--steps-list", default="", help="comma-separated rollout lengths")
    p.add_argument("--no-cpu", action="store_true", help="skip the CPU control rows")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    if args.single:
        return single(args)

    if args.envs_list:
        envs = [int(x) for x in args.envs_list.split(",") if x.strip()]
        steps = [int(x) for x in (args.steps_list or "4,8,16,24").split(",") if x.strip()]
        grid = [] if args.no_cpu else [("cpu", 96, 8)]
        grid += [("warp", e, s) for e in envs for s in steps]
    else:
        grid = GRID_FULL if args.grid == "full" else GRID_QUICK
    print(f"[bench] {len(grid)} configurations x {args.seconds:.0f} s "
          f"(plus warm-up), one subprocess each")
    print(f"[bench] python: {args.python}\n")
    header = (f"{'backend':>7} {'envs':>6} {'steps':>5} {'batch':>8} | "
              f"{'samples/s':>10} {'updates/s':>9} | {'GPU%':>5} {'VRAM':>7} | "
              f"{'CPU%':>5} {'proc RAM':>9} {'sys RAM':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for backend, envs, steps in grid:
        r = run_single(backend, envs, steps, args.seconds, args.python)
        rows.append(r)
        if "failed" in r:
            print(f"{backend:>7} {envs:>6} {steps:>5} {'-':>8} | FAILED: {r['failed'][:80]}")
            continue
        print(f"{r['backend']:>7} {r['envs']:>6} {r['steps']:>5} {r['batch']:>8} | "
              f"{r['sps']:>10,.0f} {r['updates_per_s']:>9.2f} | "
              f"{r['peak_gpu']:>4.0f}% {r['peak_vram'] / 1024:>6.1f}G | "
              f"{r['peak_cpu']:>4.0f}% {r['peak_rss']:>8.1f}G {r['peak_sys_ram']:>7.1f}G")

    ok = [r for r in rows if "failed" not in r]
    if not ok:
        print("\n[bench] every configuration failed.")
        return 1

    print()
    best_sps = max(ok, key=lambda r: r["sps"])
    best_upd = max((r for r in ok if r["backend"] == "warp"), key=lambda r: r["updates_per_s"],
                   default=None)
    print(f"[bench] most samples/s : {best_sps['backend']} {best_sps['envs']}x{best_sps['steps']} "
          f"-> {best_sps['sps']:,.0f} samples/s, {best_sps['updates_per_s']:.2f} updates/s")
    if best_upd:
        print(f"[bench] most updates/s : warp {best_upd['envs']}x{best_upd['steps']} "
              f"-> {best_upd['updates_per_s']:.2f} updates/s, {best_upd['sps']:,.0f} samples/s")

    # Degradation checks. A row that is fast while sitting on the memory ceiling is not a setting
    # anyone should run for an hour.
    warnings = []
    for r in ok:
        if r["peak_vram"] / 1024 > 13.0:
            warnings.append(f"{r['envs']}x{r['steps']}: VRAM peaked at {r['peak_vram']/1024:.1f}G of 16G")
        if r["peak_sys_ram"] > 27.0:
            warnings.append(f"{r['envs']}x{r['steps']}: system RAM peaked at {r['peak_sys_ram']:.1f}G")
        if isinstance(r.get("buffers"), dict) and "overflow" in r["buffers"]:
            warnings.append(f"{r['envs']}x{r['steps']}: constraint buffer overflow")
    # Non-monotonic scaling: more envs producing fewer samples/s means the GPU is saturated and the
    # extra worlds are only spending memory.
    warp = sorted((r for r in ok if r["backend"] == "warp"), key=lambda r: (r["steps"], r["envs"]))
    for a, b in zip(warp, warp[1:]):
        if a["steps"] == b["steps"] and b["envs"] > a["envs"] and b["sps"] < a["sps"]:
            warnings.append(f"steps {a['steps']}: {b['envs']} envs is SLOWER than {a['envs']} "
                            f"({b['sps']:,.0f} vs {a['sps']:,.0f} samples/s) - past the plateau")
    if warnings:
        print("\n[bench] degradation:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[bench] no degradation detected in the sampled range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
