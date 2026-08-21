"""Report the scalars of a training run straight from its TensorBoard event file.

Exists because stdout is not a reliable channel for judging a run. Python block-buffers when its
output is piped rather than attached to a terminal, so the SB3 tables can sit unflushed for
minutes - and for a background run there is no terminal at all. The event file is written
continuously by the SB3 logger either way, so it is the honest source.

Usage:
    python tools/tbreport.py                 # newest run under rl/runs
    python tools/tbreport.py perturbation_v1_1
    python tools/tbreport.py --gates walk    # apply the pass/fail gates for a task
"""

import argparse
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "rl", "runs")

# Scalars worth printing, grouped so a report reads top-down from "is it training at all" to
# "is the task being solved". Missing tags are skipped rather than fatal: which tags exist depends
# on the task, and a get-up run legitimately has no ball/* series.
GROUPS = {
    "training": ["rollout/ep_rew_mean", "rollout/ep_len_mean", "train/explained_variance",
                 "train/approx_kl", "train/entropy_loss", "train/n_updates"],
    "task": ["start_standing/success", "start_prone/success", "standing/all", "standing/grounded",
             "standing/head", "standing/tilt", "standing/speed", "standing/icp"],
    "walk": ["walk/forward_speed", "walk/distance", "walk/lateral_drift", "walk/fell",
             "walk/alive_fraction"],
    "ball": ["ball/hit_rate", "ball/small_share", "ball/shots"],
    "reward": ["reward/shaping", "reward/upright", "reward/effort", "reward/terminal",
               "reward/progress", "reward/velocity", "reward/alive", "reward/heading"],
}


def load(run_dir):
    accumulator = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def series(accumulator, tag):
    """(first, last, last_step, count) for a tag, or None if it is absent."""
    if tag not in accumulator.Tags()["scalars"]:
        return None
    events = accumulator.Scalars(tag)
    if not events:
        return None
    return events[0].value, events[-1].value, events[-1].step, len(events)


def newest_run():
    if not os.path.isdir(RUNS):
        return None
    candidates = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
                  if os.path.isdir(os.path.join(RUNS, d))]
    return max(candidates, key=os.path.getmtime) if candidates else None


def report(run_dir):
    accumulator = load(run_dir)
    print(f"=== {os.path.basename(run_dir)} ===")

    available = accumulator.Tags()["scalars"]
    if not available:
        print("  no scalars yet - the run has not completed its first rollout")
        return {}

    values = {}
    for group, tags in GROUPS.items():
        rows = [(t, series(accumulator, t)) for t in tags]
        rows = [(t, s) for t, s in rows if s is not None]
        if not rows:
            continue
        print(f"\n  [{group}]")
        for tag, (first, last, step, count) in rows:
            values[tag] = last
            arrow = "->" if count > 1 else "  "
            print(f"    {tag:<32} {first:>10.4f} {arrow} {last:>10.4f}   (step {step:,}, n={count})")

    # Any tag the groups above do not mention, so a newly added metric is never silently invisible.
    known = {t for tags in GROUPS.values() for t in tags}
    extra = sorted(t for t in available if t not in known)
    if extra:
        print(f"\n  [other] {', '.join(extra)}")
    return values


# Gates are expressed as (tag, predicate, human description). A gate whose tag is missing FAILS
# rather than passing quietly - an absent metric is exactly the case that hid a broken run before.
GATES = {
    "perturbation": [
        ("ball/hit_rate", lambda v: v > 0.8, "ball connects on >80% of shots"),
        ("ball/small_share", lambda v: 0.3 <= v <= 0.7, "both ball profiles firing (share 0.3-0.7)"),
        ("rollout/ep_len_mean", lambda v: v > 60, "episodes reach most of the 5s window (>60 steps)"),
        ("standing/all", lambda v: v > 0.2, "standing achieved on >20% of ticks"),
        ("train/approx_kl", lambda v: v < 0.1, "policy updates stable (approx_kl < 0.1)"),
    ],
    # The speed and distance gates are set to catch the STANDING-STILL COLLAPSE specifically, which
    # is the failure this task actually exhibits. An earlier version asked only that forward speed
    # be non-negative, and it passed cheerfully while the policy stood perfectly still at
    # 0.007 m/s - a gate that cannot fail on the one thing going wrong is not a gate. The
    # thresholds sit roughly an order of magnitude above that collapse, and are about "is it moving
    # at all", not "has it learned to walk" - five minutes cannot answer the latter.
    "walk": [
        ("rollout/ep_len_mean", lambda v: v > 10, "episodes are not terminating instantly"),
        ("walk/forward_speed", lambda v: v > 0.05, "actually moving, not standing still"),
        ("walk/distance", lambda v: v > 0.30, "covers ground over the episode"),
        ("walk/alive_fraction", lambda v: v > 0.1, "survives past the first ticks"),
        ("train/approx_kl", lambda v: v < 0.1, "policy updates stable (approx_kl < 0.1)"),
    ],
}


def check(values, task):
    print(f"\n  [gates: {task}]")
    failed = 0
    for tag, predicate, description in GATES[task]:
        if tag not in values:
            print(f"    MISSING {tag:<28} {description}")
            failed += 1
            continue
        ok = predicate(values[tag])
        print(f"    {'PASS   ' if ok else 'FAIL   '}{tag:<28} {values[tag]:>9.4f}  {description}")
        if not ok:
            failed += 1
    print(f"\n  {'ALL GATES PASS' if failed == 0 else f'{failed} GATE(S) FAILED'}")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", help="run directory name under rl/runs (default: newest)")
    parser.add_argument("--gates", choices=sorted(GATES), help="apply this task's pass/fail gates")
    args = parser.parse_args()

    run_dir = os.path.join(RUNS, args.run) if args.run else newest_run()
    if not run_dir or not os.path.isdir(run_dir):
        print(f"No such run: {run_dir}")
        return 2

    values = report(run_dir)
    if args.gates:
        return 1 if check(values, args.gates) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
