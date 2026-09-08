"""Is it a WALK, a flamingo, or a hop? Scores gait quality in Isaac across a command sweep.

**This exists because "94.3% single support, 0% flight" was read as walking and was not.** Measured
2026-09-05, `walk_clean/model_28941` at command 0.60 held 95.3% single support with 0% flight and
**one foot strike in twelve seconds** while covering -0.03 m. A body standing on one leg maximises
single-support fraction perfectly. Support fraction alone cannot tell a gait from a statue, and
`rew_single_support` is maximised by exactly the wrong behaviour.

So the verdict here is driven by STRIDE COUNT and DISTANCE first, with support fractions used only to
classify what kind of failure it is:

    FALLEN     upright < 90% - checked first; a corpse still reports contact fractions
    WALK       strikes >= 12, flight < 10%, alternating feet
    HOP        strikes >= 12, flight >= 10% or feet mostly in phase
    FLAMINGO   strikes < 12, single support high      <- reads as a walk on support fraction alone
    STATUE     strikes < 12, double support high

`alternation` is the fraction of strikes that land on the opposite foot from the previous strike; a
real gait is near 1.0, a two-footed hop near 0.0. That is the number `rew_single_support` should have
been shaped against.

    python isaac_lab_3/scripts/gait_score.py --checkpoint <model.pt> --commands 0.3 0.6 1.0
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import csv

HERE = pathlib.Path(__file__).resolve().parent

# Matched to `IsaacObservation.ContactHeight` and `stand_env.CONTACT_HEIGHT`, which agree since the
# collider half-size fix moved both rigs' planted foot COM to ~0.039 m.
CONTACT_HEIGHT = 0.05


def classify(strikes: int, flight: float, single: float, double: float, alternation: float,
             upright: float) -> str:
    # Checked FIRST: a fallen body reports whatever contact pattern its corpse happens to hold, and
    # without this it scores as a STATUE - measured on a Godot replay that fell to 0.12 m and was
    # labelled "STATUE, 100% double support" while lying on the floor.
    if upright < 0.9:
        return "FALLEN"
    if strikes >= 12:
        return "HOP" if (flight >= 0.10 or alternation < 0.5) else "WALK"
    return "FLAMINGO" if single > double else "STATUE"


def analyse(path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    if n < 10:
        return {}
    dt = 1.0 / 60.0
    # **Prefer the true foot HEIGHT over the reported contact flags.** The flags are an observation
    # channel, and anything that rewrites that channel - `obs_contact_stuck_prob`, a mask, a
    # different threshold per engine - falsifies the score while the body may be doing something
    # else entirely. Measured 2026-09-06: a policy trained with the flags forced to double support
    # scored "0 strikes, 99.9% double support" because the TRACE carried the forced flags, not
    # because the feet stayed down. `footZ_*` is ground truth in both engines and is written by both
    # `dump_obs.py` and `IsaacPolicyDriver.DofTracePath`.
    if "footZ_L" in rows[0] and "footZ_R" in rows[0]:
        feet = [(float(r["footZ_L"]) < CONTACT_HEIGHT, float(r["footZ_R"]) < CONTACT_HEIGHT)
                for r in rows]
    else:
        feet = [(float(r["c_FootL"]) > 0.5, float(r["c_FootR"]) > 0.5) for r in rows]
    # Strike = a foot transitioning from air to ground. Record WHICH foot, so alternation is
    # measurable: a two-footed hop strikes constantly and alternates never.
    order: list[int] = []
    for i in range(1, n):
        if feet[i][0] and not feet[i - 1][0]:
            order.append(0)
        if feet[i][1] and not feet[i - 1][1]:
            order.append(1)
    alternation = (
        sum(1 for a, b in zip(order, order[1:]) if a != b) / max(1, len(order) - 1)
        if len(order) > 1 else 0.0
    )
    single = sum(1 for a, b in feet if a != b) / n
    double = sum(1 for a, b in feet if a and b) / n
    flight = sum(1 for a, b in feet if not a and not b) / n
    upright = sum(1 for r in rows if float(r["height"]) > 0.6) / n
    dist = sum(float(r["linVel_x"]) for r in rows) * dt
    return {
        "strikes": len(order), "alternation": alternation, "single": single,
        "double": double, "flight": flight, "upright": upright, "dist": dist,
        "vx": dist / (n * dt),
        "verdict": classify(len(order), flight, single, double, alternation, upright),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--commands", nargs="+", type=float, default=[0.3, 0.6, 1.0])
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--no_playback", action="store_true",
                   help="Leave the per-episode actuator randomisation ON. Default is OFF (pinned), "
                        "because it flips the verdict between runs of one checkpoint.")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gait_"))
    name = f"{pathlib.Path(args.checkpoint).parent.parent.name[-14:]}/{pathlib.Path(args.checkpoint).stem}"
    print(f"{name}")
    print(f"{'cmd':>5s} {'verdict':>9s} {'dist(m)':>8s} {'vx':>6s} {'strikes':>8s} "
          f"{'altern':>7s} {'single':>7s} {'double':>7s} {'flight':>7s} {'upright':>8s}")
    for c in args.commands:
        out = tmp / f"c{c}.csv"
        r = subprocess.run(
            [args.python, str(HERE / "dump_obs.py"), "--task", args.task,
             "--checkpoint", args.checkpoint, "--out", str(out),
             "--steps", str(args.steps), "--no_reset_noise", "--command", str(c)]
            + ([] if args.no_playback else ["--playback"]),
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not out.is_file():
            print(f"{c:5.2f}   FAILED: {(r.stderr or '').strip().splitlines()[-1][:60] if r.stderr else '?'}")
            continue
        s = analyse(out)
        print(f"{c:5.2f} {s['verdict']:>9s} {s['dist']:8.2f} {s['vx']:6.3f} {s['strikes']:8d} "
              f"{s['alternation']:6.0%} {s['single']:6.1%} {s['double']:6.1%} {s['flight']:6.1%} "
              f"{s['upright']:7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
