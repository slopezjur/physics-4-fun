"""Run one Godot check scene with properties overridden, and return its stdout.

**Every sim-to-sim test tonight is 'set two scene properties, run 20 s, read the log'.** Doing that
by hand rewrites a `.tscn` each time, and a bare `write_text` on Windows silently converts the whole
file to CRLF - which has already corrupted a HUD label on this project. This does the rewrite with
`newline="\n"`, restores the original file afterwards no matter what, and never leaves the scene
dirty in git.

    python isaac_lab_3/scripts/godot_run.py --scene Walk/IsaacWalkCheckNewton.tscn \
        --set ActionScaleOverride=0.05 --set 'DofTracePath="C:/tmp/t.csv"' --seconds 20
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GODOT = pathlib.Path(os.environ.get(
    "P4F_GODOT",
    r"D:\Programas\Godot_v4.7.1-stable_mono_win64\Godot_v4.7.1-stable_mono_win64.exe"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="Walk/IsaacWalkCheckNewton.tscn")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   help="KEY=VALUE written verbatim into the scene's root node block")
    p.add_argument("--seconds", type=float, default=0.0, help="overrides AutoQuitSeconds")
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    scene = PROJECT_ROOT / "Scenes/RL/Isaac3" / args.scene
    if not scene.is_file():
        raise SystemExit(f"no such scene: {scene}")

    original = scene.read_text(encoding="utf-8")
    text = original
    pairs = list(args.sets)
    if args.seconds > 0.0:
        pairs.append(f"AutoQuitSeconds = {args.seconds}")

    for pair in pairs:
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        # Replace in place when present, append to the root node block when not, so a property the
        # scene never declared still takes effect rather than silently doing nothing.
        pattern = rf"^{re.escape(key)} = .*$"
        if re.search(pattern, text, flags=re.M):
            text = re.sub(pattern, f"{key} = {value}", text, flags=re.M)
        else:
            text = text.rstrip("\n") + f"\n{key} = {value}\n"

    try:
        scene.write_text(text, encoding="utf-8", newline="\n")
        res = subprocess.run(
            [str(GODOT), "--headless", "--path", str(PROJECT_ROOT),
             "res://" + str(scene.relative_to(PROJECT_ROOT)).replace("\\", "/")],
            capture_output=True, text=True, timeout=args.timeout, encoding="utf-8", errors="replace")
        sys.stdout.write(res.stdout or "")
        sys.stdout.write(res.stderr or "")
    except subprocess.TimeoutExpired:
        print(f"[godot_run] TIMEOUT after {args.timeout}s")
        return 2
    finally:
        # Unconditional restore. A test that leaves the scene edited makes the NEXT test a different
        # experiment, which is the single most common way a night's results become fiction here.
        scene.write_text(original, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
