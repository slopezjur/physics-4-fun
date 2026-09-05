"""Generate `p4f_newton/godot_plant.py` from a Godot run log.

Godot drives every joint through `ActiveBone` -> `PidController3D`, which uses the Tan-Liu-Turk
Stable PD form and divides BOTH gains by

    denominator = 1 + Kd*dt/I + Kp*dt^2/I

Isaac's `ImplicitActuatorCfg` applies the authored gains at full value, so a policy trained in Isaac
has been driving a joint several times stiffer than the one it meets in Godot. Measured live at
120 Hz the mean is `kEff=0.15`, and the per-bone spread is 8.4x - from 0.064 at the foot to 0.540 at
the chest - so a single averaged factor is wrong for almost every joint. That is the trap the lone
"0.176 = 155/882" figure in `assets.py` sets: 0.176 is the THIGH's factor, and nothing else's.

This script turns a measured table into a generated module rather than a hand-copied constant, so
the numbers carry their provenance and can be regenerated when the rig, the gains or the physics
rate change.

Usage
-----
Produce a log with the `[PLANT]` lines (any Isaac3 check scene emits them once, at the first step)::

    Godot_v4.7.1-stable_mono_win64.exe --path . \
        "res://Scenes/RL/Isaac3/Walk/IsaacWalkCheckNewton.tscn" > run.log 2>&1

    python isaac_lab_3/scripts/derive_plant.py run.log

**Use `Godot_v4.7.1-stable_mono_win64.exe`, not the `_console` build.** An export written into the
Godot install folder on 2026-08-26 left a `.pck` beside - and name-matching - the console binary, so
that executable boots as a self-contained game and ignores `--path` for project settings. The
`[PLANT]` header line records `physicsHz`; if it disagrees with project.godot, that is the cause.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

from plant_template import TEMPLATE

HERE = pathlib.Path(__file__).resolve().parent
ISAAC3_ROOT = HERE.parent
OUT_PATH = ISAAC3_ROOT / "p4f_newton" / "godot_plant.py"

# `[PLANT] bone=Spine kp=600 kd=18 inertia=0,0486439 denom=4,9402 kpEff=121,453 ...`
# Godot formats floats with the system decimal separator, which is a COMMA on this machine. Parsing
# with `float()` alone silently truncates "0,0486439" to a ValueError at best and to 0 at worst, and
# a zero inertia would inverte the denominator's meaning rather than raising.
_BONE_RE = re.compile(
    r"\[PLANT\] bone=(?P<bone>\S+) kp=(?P<kp>\S+) kd=(?P<kd>\S+) inertia=(?P<inertia>\S+)"
    r" denom=(?P<denom>\S+) kpEff=(?P<kp_eff>\S+) kdEff=(?P<kd_eff>\S+) scale=(?P<scale>\S+)"
    r" maxTorque=(?P<max_torque>\S+) parent=(?P<parent>\S+) mass=(?P<mass>\S+)"
    r" pivotOffset=(?P<pivot>\S+)"
)
_HEADER_RE = re.compile(r"\[PLANT\] physicsHz=(?P<hz>\d+) dt=(?P<dt>\S+)")


def _num(text: str) -> float:
    """Parse a Godot-formatted float, tolerating a comma decimal separator."""
    return float(text.replace(",", "."))


def _to_usd(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Godot (x, y, z) -> USD, matching `isaac_lab/scripts/build_d6_usd.py::to_usd` exactly.

    Godot is Y-up and USD here is Z-up, and the builder's mapping is the ONLY definition of how the
    two frames correspond on this rig. Restating it by eye is how a pivot ends up pointing sideways
    instead of up - measured as a gravity feed-forward that drove the body into the floor.
    """
    return (-p[2], -p[0], p[1])


def parse(log_text: str) -> tuple[int, dict[str, dict[str, float]]]:
    header = _HEADER_RE.search(log_text)
    if header is None:
        raise SystemExit(
            "no '[PLANT] physicsHz=' line in the log. The table is emitted once, with the first-step "
            "log, so the run must reach the first policy step - and LogFirstStep must be enabled."
        )

    bones: dict[str, dict[str, float]] = {}
    for match in _BONE_RE.finditer(log_text):
        bone = match.group("bone")
        bones[bone] = {
            "kp": _num(match.group("kp")),
            "kd": _num(match.group("kd")),
            "inertia": _num(match.group("inertia")),
            "denominator": _num(match.group("denom")),
            "kp_eff": _num(match.group("kp_eff")),
            "kd_eff": _num(match.group("kd_eff")),
            "scale": _num(match.group("scale")),
            "max_torque": _num(match.group("max_torque")),
            "parent": match.group("parent"),
            "mass": _num(match.group("mass")),
            "pivot": _to_usd(tuple(_num(v) for v in match.group("pivot").split("|"))),
        }

    if not bones:
        raise SystemExit("header found but no '[PLANT] bone=' rows - the format has drifted.")

    for bone, row in bones.items():
        if row["inertia"] <= 0.0 or row["denominator"] <= 0.0:
            raise SystemExit(f"{bone}: non-positive inertia/denominator, refusing to generate.")
        # The denominator is 1 + non-negative terms, so a scale above 1 means the formula or the
        # parse is wrong, not that Godot is stiffer than authored.
        if not (0.0 < row["scale"] <= 1.0):
            raise SystemExit(f"{bone}: scale {row['scale']} outside (0, 1].")

    return int(header.group("hz")), bones


def render(physics_hz: int, bones: dict[str, dict[str, float]], source: pathlib.Path) -> str:
    stamp = datetime.date.today().isoformat()
    nl = chr(10)
    rows = nl.join(
        f'    "{bone}": {row["scale"]:.6g},'
        f'  # kp {row["kp"]:g} -> {row["kp_eff"]:.4g}, I={row["inertia"]:.4g}, denom={row["denominator"]:.4g}'
        for bone, row in bones.items()
    )
    parents = nl.join(
        f'    "{bone}": ' + (f'"{row["parent"]}",' if row["parent"] != "-" else "None,")
        for bone, row in bones.items()
    )
    pivots = nl.join(
        f'    "{bone}": ({row["pivot"][0]:.6g}, {row["pivot"][1]:.6g}, {row["pivot"][2]:.6g}),'
        for bone, row in bones.items()
    )
    max_torques = nl.join(
        f'    "{bone}": {row["max_torque"]:.6g},' for bone, row in bones.items()
    )
    masses = nl.join(f'    "{bone}": {row["mass"]:.6g},' for bone, row in bones.items())
    lowest = min(bones.items(), key=lambda kv: kv[1]["scale"])
    highest = max(bones.items(), key=lambda kv: kv[1]["scale"])
    spread = highest[1]["scale"] / lowest[1]["scale"]

    return TEMPLATE.format(
        stamp=stamp,
        source=source.name,
        physics_hz=physics_hz,
        spread=f"{spread:.1f}",
        low_bone=lowest[0],
        low_scale=f'{lowest[1]["scale"]:.4g}',
        high_bone=highest[0],
        high_scale=f'{highest[1]["scale"]:.4g}',
        rows=rows,
        parents=parents,
        masses=masses,
        pivots=pivots,
        max_torques=max_torques,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path, help="Godot run log containing [PLANT] lines")
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = parser.parse_args()

    physics_hz, bones = parse(args.log.read_text(encoding="utf-8", errors="replace"))
    args.out.write_text(render(physics_hz, bones, args.log), encoding="utf-8", newline="\n")

    print(f"[derive_plant] {len(bones)} bones at {physics_hz} Hz -> {args.out}")
    for bone, row in sorted(bones.items(), key=lambda kv: kv[1]["scale"]):
        print(f"  {bone:<12} kp {row['kp']:>6g} -> {row['kp_eff']:>8.3f}   scale {row['scale']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
