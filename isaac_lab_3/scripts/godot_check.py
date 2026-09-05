"""Measuring a checkpoint's behaviour IN GODOT: the one implementation both rankers use.

`select_by_godot.py` ranks a lineage's checkpoints; `compare_godot.py` weighs a new one against
whatever is currently promoted. Different questions, identical mechanics - export the checkpoint,
promote it so Godot loads it, walk a descending ladder of action authorities until the dummy stands,
then push it.

**Extracted because the copies had already drifted and a bug had to be fixed twice.** Both files
carried their own `set_scene`, and both needed the same `newline="\\n"` correction on the same day -
without it a bare `write_text` rewrites the whole scene as CRLF on Windows, and Godot renders the
`\\n` inside multi-line Label strings as an extra line break. Two copies means the next such fix
lands in one of them.

Nothing here decides anything. It runs Godot and reports what happened; the callers own the policy
about what the numbers mean.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ISAAC3_ROOT = HERE.parent
PROJECT_ROOT = ISAAC3_ROOT.parent

# The scene both rankers drive. It carries the Godot-side settings a policy needs - BalanceAssist,
# HillVelocityFilter - so measuring through it measures the real deployment plant rather than a
# stripped one.
SCENE = "res://Scenes/RL/Isaac3/Stand/IsaacStandCheckNewton.tscn"

# NOT the `_console` build, deliberately. An export of this project was written into the Godot
# INSTALL folder on 2026-08-26, and it took the console executable's own basename:
# `Godot_v4.7.1-stable_mono_win64_console.pck`. Godot auto-mounts a pack whose basename matches the
# running executable, so that binary boots as a SELF-CONTAINED GAME and ignores `--path` for project
# settings - `physics_ticks_per_second`, the Jolt solver steps, everything.
#
# It is nearly invisible: scenes, ONNX models and the C# assembly still load from disk, so code
# edits take effect while project.godot edits do nothing. On 2026-09-03 that made a 120 -> 480 Hz
# change produce output byte-identical to the baseline, which reads as "no effect" rather than
# "never applied". `ProjectSettings.GlobalizePath("res://")` returns EMPTY under the hijacked binary;
# `IsaacPolicyDriver` prints it as `root ...` in its config line, so check there first.
GODOT = pathlib.Path(
    os.environ.get(
        "P4F_GODOT",
        r"D:\Programas\Godot_v4.7.1-stable_mono_win64\Godot_v4.7.1-stable_mono_win64.exe",
    )
)

# Tried high to low; the first that stands is the score. Higher is better - it means the body
# tolerates more of what the policy asks for.
DEFAULT_AUTHORITIES = (0.20, 0.15, 0.10, 0.07, 0.05)

# Separates a policy the body TOLERATES from one doing work: a zero-action baseline also survives
# 4 N.s and falls at 8, so anything that beats that is contributing.
DEFAULT_PUSH = 8.0


def scene_path() -> pathlib.Path:
    return PROJECT_ROOT / SCENE.removeprefix("res://")


def set_scene(authority: float, push: float) -> None:
    """Rewrite only the two properties under test, leaving the rest of the scene alone.

    **`newline="\\n"` is not optional on Windows.** A bare `write_text` applies universal-newline
    translation and rewrites the WHOLE file as CRLF - including the `\\n` inside multi-line Label
    strings, which Godot renders as an extra line break. That is not a diff artefact: it silently
    double-spaced every line of the TestChamber HUD, and this function touches the scene on every
    single rung of the authority ladder.
    """
    path = scene_path()
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"ActionScaleOverride = [0-9.]+\n|PushImpulse = [0-9.]+\n|PushAtSeconds = [0-9.]+\n", "", text)
    block = f"ActionScaleOverride = {authority}\n"
    if push > 0.0:
        block += f"PushImpulse = {push}\nPushAtSeconds = 4.0\n"
    path.write_text(
        text.replace("BalanceAssist = 1.0\n", f"BalanceAssist = 1.0\n{block}"),
        encoding="utf-8", newline="\n",
    )


def stands() -> bool:
    """Run the check scene headless. True when the arena reports the dummy still upright."""
    out = subprocess.run(
        [str(GODOT), "--headless", "--path", str(PROJECT_ROOT), SCENE],
        capture_output=True, text=True, errors="replace", timeout=180,
    ).stdout
    return "=> STANDING" in out


def export(checkpoint: pathlib.Path, task: str) -> bool:
    """Export and promote, so the Godot scene loads this checkpoint on its next run."""
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        P4F_XPBD_ITERATIONS=os.environ.get("P4F_XPBD_ITERATIONS", "2"),
    )
    done = subprocess.run(
        [sys.executable, str(HERE / "export.py"), "--task", task,
         "--checkpoint", str(checkpoint), "--promote"],
        cwd=str(ISAAC3_ROOT), env=env, capture_output=True, text=True, errors="replace", timeout=900,
    )
    if done.returncode != 0:
        print(f"    export failed: {done.stdout[-300:]}")
    return done.returncode == 0


def measure(
    checkpoint: pathlib.Path,
    task: str,
    authorities=DEFAULT_AUTHORITIES,
    push: float = DEFAULT_PUSH,
) -> tuple[float, float]:
    """(highest authority that stands, push survived there). (0, 0) if it never stands.

    The ladder descends and stops at the first rung that stands, so it is deliberately sequential -
    running the rungs concurrently would mostly compute results that are then discarded.
    """
    if not export(checkpoint, task):
        return 0.0, 0.0

    best = 0.0
    for authority in sorted(authorities, reverse=True):
        set_scene(authority, push=0.0)
        if stands():
            best = authority
            break

    held = 0.0
    if best > 0.0 and push > 0.0:
        set_scene(best, push=push)
        held = push if stands() else 0.0
    return best, held


def add_common_args(p: argparse.ArgumentParser) -> None:
    """The ladder options every ranker exposes, so they cannot describe them differently."""
    p.add_argument(
        "--authorities", type=float, nargs="+", default=list(DEFAULT_AUTHORITIES),
        help="Action-scale ladder, tried high to low; the first that stands is the score.",
    )
    p.add_argument(
        "--push", type=float, default=DEFAULT_PUSH,
        help="Push impulse for the tie-break, N.s. 0 skips.",
    )
