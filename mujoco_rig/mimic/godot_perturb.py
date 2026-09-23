"""Validate actual Godot push replay against native MuJoCo ONNX baseline trials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

import mujoco
import numpy as np

from .perturb import RECOVERY


def replay(args):
    protocol = json.loads((args.run / "protocol.json").read_text())
    if protocol["recovery"] != RECOVERY:
        raise ValueError("Unknown recovery measurement contract")
    output = args.run.resolve() / f"godot-{uuid.uuid4().hex}.json"
    env = {**os.environ, "P4F_MIMIC_PUSH_PROTOCOL": str((args.run / "protocol.json").resolve()),
           "P4F_MIMIC_REPLAY_OUTPUT": str(output), "P4F_MUJOCO_LIBRARY": str(Path(mujoco.__file__).parent)}
    root = Path(__file__).resolve().parents[2]
    with (args.run / "godot.log").open("w") as log:
        subprocess.run([str(args.godot), "--headless", "--path", str(root),
                        "res://Scenes/RL/Isaac3/MuJoCo/MimicPerturbReplay.tscn"], env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=180, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    actual = json.loads(output.read_text())
    expected = json.loads((args.run / "native-episodes.json").read_text())
    if [e["name"] for e in actual] != [e["name"] for e in expected]:
        raise ValueError("Godot did not execute the same trials in order")
    initial = dict(observation=0., action=0., qpos_after=0.)
    push_window = dict(observation=0., action=0., qpos_after=0.)
    timing = True
    for a, b in zip(actual, expected):
        a_trace = {v["step"]: v for v in a["trace"]}
        timing &= a_trace.keys() == {v["step"] for v in b["trace"]}
        for frame in b["trace"]:
            other = a_trace.get(frame["step"])
            if other is None:
                continue
            timing &= other["force"] == frame["force"]
            errors = initial if frame["step"] < 2 else push_window
            for key in errors:
                errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(other[key]) - frame[key]))))
    duration_error = max(abs(a["metrics"]["survival_seconds"] - b["survival_seconds"]) for a, b in zip(actual, expected))
    mismatches = [a["name"] for a, b in zip(actual, expected) if a["metrics"]["survived"] != b["survived"]]
    recovery_mismatches = [a["name"] for a, b in zip(actual, expected) if a["metrics"]["recovered"] != b["recovered"]]
    metric_errors = {key: max(abs(a["metrics"][key] - b[key]) for a, b in zip(actual, expected))
                     for key in ("max_horizontal_displacement_m", "foot_travel_m", "contact_switches")}
    recovery_time_error = max((abs(a["metrics"]["recovery_seconds"] - b["recovery_seconds"])
                               for a, b in zip(actual, expected) if a["metrics"]["recovered"] and b["recovered"]), default=0.)
    checks = {"force_and_timing": bool(timing), "initial_observations": initial["observation"] < .003,
              "initial_actions": initial["action"] < 1e-4, "initial_poses": initial["qpos_after"] < 1e-4,
              "push_window_pose": push_window["qpos_after"] < .01,
              "survival_outcomes": not mismatches, "survival_duration": duration_error < .1,
              "measured_recovery": not recovery_mismatches and recovery_time_error < 2 * protocol["control_timestep"],
              "measured_movement": metric_errors["max_horizontal_displacement_m"] < .005
                  and metric_errors["foot_travel_m"] < .01 and metric_errors["contact_switches"] == 0}
    report = {"passed": all(checks.values()), "checks": checks, "trials": len(actual), "initial_errors": initial,
              "push_window_errors": push_window, "max_survival_difference_seconds": duration_error,
              "survival_mismatches": mismatches, "settled_recovery_mismatches": recovery_mismatches,
              "metric_errors": metric_errors, "recovery_time_error_seconds": recovery_time_error,
              "scope": "Fixed force schedule and survival parity; settled threshold is a diagnostic, not a training gate"}
    output.replace(args.run / "godot-episodes.json")
    (args.run / "godot-parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--godot", type=Path,
                        default=Path(os.environ.get("P4F_GODOT_EXE") or
                                     Path(__file__).resolve().parents[2] / "tools/Godot/Godot.exe"))
    raise SystemExit(0 if replay(parser.parse_args()) else 1)
