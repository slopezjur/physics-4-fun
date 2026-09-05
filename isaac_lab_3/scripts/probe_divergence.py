"""WHEN and WHERE do the two engines part company under motion?

**Every other tool here measures either a static pose or a closed-loop outcome.** The static probes
say the bodies agree; the ladder says anything that moves falls. Neither can say what happens in
between, and "the plants diverge under motion" has been the standing explanation without a single
measurement behind it.

This reads an open-loop replay - the SAME recorded action sequence driven into both engines, no
policy in either loop - and reports, per joint: the time its position error first crosses a
threshold, whether the error is led by position or by velocity, and whether the onset coincides with
a foot-contact transition. The ordering of those onset times is the thing worth having: a fault that
starts at a foot and climbs is a contact problem, one that starts at the pelvis and descends is a
root/balance problem, and one where everything crosses at once is neither.

    python isaac_lab_3/scripts/probe_divergence.py --godot god.csv --isaac iso.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONTRACT = PROJECT_ROOT / "Models" / "locomotion_policy.contract.json"

# Rough proximal->distal depth, for reading the onset ordering as a propagation direction.
DEPTH = {
    "Pelvis": 0, "Spine": 1, "Chest": 2, "Head": 3,
    "UpperArm": 3, "Forearm": 4, "Hand": 5,
    "Thigh": 1, "Shin": 2, "Foot": 3,
}


def bone_of(dof: str) -> str:
    """`joint_Foot_L:0` -> `Foot`. The side suffix has to go or every lookup misses."""
    bone = dof.removeprefix("joint_").rpartition(":")[0]
    return bone.removesuffix("_L").removesuffix("_R")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--godot", required=True)
    p.add_argument("--isaac", required=True)
    p.add_argument("--threshold", type=float, default=0.10, help="rad of position error")
    p.add_argument("--seconds", type=float, default=4.0)
    args = p.parse_args()

    g = list(csv.DictReader(open(args.godot, encoding="utf-8")))
    i = list(csv.DictReader(open(args.isaac, encoding="utf-8")))
    order = json.load(open(CONTRACT, encoding="utf-8"))["newton_dof_order"]
    gp = [k for k in g[0] if k.startswith("pos_")]
    ip = [k for k in i[0] if k.startswith("pos_")]
    gv = [k for k in g[0] if k.startswith("vel_")]
    iv = [k for k in i[0] if k.startswith("vel_")]
    n = min(len(g), len(i), int(args.seconds * 60))

    # Foot-contact transitions in each engine, for the coincidence test.
    def strikes(rows):
        out = []
        for k in range(1, n):
            for foot in ("c_FootL", "c_FootR"):
                if (float(rows[k][foot]) > 0.5) != (float(rows[k - 1][foot]) > 0.5):
                    out.append(k / 60.0)
        return sorted(set(out))

    g_strikes = strikes(g)

    rows = []
    for idx, (a, b, va, vb) in enumerate(zip(gp, ip, gv, iv)):
        perr = [abs(float(g[k][a]) - float(i[k][b])) for k in range(n)]
        verr = [abs(float(g[k][va]) - float(i[k][vb])) for k in range(n)]
        # **Separate "started apart" from "drifted apart".** A joint whose error is already at the
        # threshold on step 0 has not diverged - it was never together, and reporting it as an onset
        # of 0.02 s reads as an instant dynamic fault that is not there.
        start = perr[0]
        grew = [max(0.0, e - start) for e in perr]
        onset = next((k / 60.0 for k in range(n) if grew[k] > args.threshold), None)
        # Velocity-led means the rate disagreed before the angle did.
        vonset = next((k / 60.0 for k in range(n) if verr[k] > args.threshold * 60 * 0.25), None)
        rows.append((onset if onset is not None else 1e9, vonset, order[idx], max(perr), start))

    rows.sort(key=lambda r: r[0])
    print(f"\n[divergence] {n} steps ({n/60:.1f}s), threshold {args.threshold} rad")
    print(f"[divergence] Godot foot-contact transitions at: "
          f"{', '.join(f'{t:.2f}' for t in g_strikes[:12])}{' ...' if len(g_strikes) > 12 else ''}")
    print(f"\n{'joint':22s} {'depth':>5s} {'pos onset':>10s} {'vel onset':>10s} {'peak pos':>9s} {'near strike':>12s}")
    never = 0
    for onset, vonset, name, pmax, start in rows[:16]:
        if onset >= 1e9:
            never += 1
            continue
        near = min((abs(onset - t) for t in g_strikes), default=9.9)
        print(f"{name:22s} {DEPTH.get(bone_of(name), 9):5d} {onset:11.2f} "
              f"{(f'{vonset:.2f}' if vonset is not None else '-'):>10s} {pmax:7.3f} {start:8.3f} "
              f"{(f'{near:+.2f}s' if near < 9 else '-'):>10s}")
    never += sum(1 for r in rows if r[0] >= 1e9) - never
    print(f"\n{sum(1 for r in rows if r[0] >= 1e9)} of 45 joints never diverge past {args.threshold} rad")

    first = [r for r in rows if r[0] < 1e9][:6]
    if first:
        depths = [DEPTH.get(bone_of(r[2]), 9) for r in first]
        print(f"\nfirst six to diverge have depths {depths} "
              f"(0 = pelvis, 5 = hand/foot tip): "
              f"{'DISTAL-led - look at contact' if sum(depths) / len(depths) > 2.5 else 'PROXIMAL-led - look at the root/balance'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
