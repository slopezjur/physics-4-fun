"""Score a fixed ball validation suite and verify actual Godot policy/contact parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .ball_protocol import VALIDATION_SCHEMA, VALIDATION_EPISODES
from .checkpoints import validate_contract
from .evaluate_stand import evaluate
from .runtime import activate
from .rig import Rig
from .contact_contract import mode_from_contract


def compare(metrics, traces, actual, dt):
    expected = metrics["cases"]
    if [e["name"] for e in expected] != [e["name"] for e in actual]:
        raise ValueError("Godot replay case ordering/count mismatch")
    errors = dict(observation=0., action=0., qpos_after=0., ball_position=0.)
    keys = dict(observation="observations", action="actions", qpos_after="qpos_after", ball_position="ball_position")
    complete = True
    for i, (reference, result) in enumerate(zip(expected, actual)):
        end = int(round(reference["survival_seconds"] / dt))
        first = reference["first_hit_step"]
        wanted = {step for step in range(end) if step < 2
                  or reference["launch_step"] - 2 <= step < reference["launch_step"] + 6
                  or first >= 0 and first - 1 <= step < first + 8}
        complete &= [frame["step"] for frame in result["trace"]] == sorted(wanted)
        for frame in result["trace"]:
            step = frame["step"]
            if not 0 <= step < end:
                complete = False
                continue
            for field, key in keys.items():
                value = np.asarray(frame[field])
                target = traces[key][step, i]
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid Godot {field} trace")
                errors[field] = max(errors[field], float(np.max(np.abs(value - target))))
    checks = dict(
        trace_coverage=bool(complete),
        observations=errors["observation"] < .003,
        actions=errors["action"] < 1e-4,
        poses=errors["qpos_after"] < .001,
        ball_positions=errors["ball_position"] < .001,
        hit_timing=all(a["first_hit_step"] == b["first_hit_step"] for a, b in zip(expected, actual)),
        survival=all(a["survived"] == b["survived"] for a, b in zip(expected, actual)),
        duration=all(abs(a["survival_seconds"] - b["survival_seconds"]) < .02 for a, b in zip(expected, actual)),
    )
    return dict(passed=all(checks.values()), checks=checks, max_errors=errors)


def run(args):
    activate(args.mimickit)
    from .ball_task import BallTask
    root = Path(__file__).resolve().parents[2]
    bundle = args.bundle.resolve()
    contract = json.loads((bundle / "contract.json").read_text())
    if sha256(bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    rig = Rig.load(root / "mujoco_rig/dummy.xml")
    task = BallTask(rig, bundle / "stand_reference.npz", VALIDATION_EPISODES, "cpu",
                    speed_min=args.speed_min, speed_max=args.speed_max, contact_mode=mode_from_contract(contract))
    validate_contract(contract, task.observation_contract())
    args.out.mkdir(parents=True, exist_ok=False)
    session = ort.InferenceSession(str(bundle / "stand.onnx"), providers=["CPUExecutionProvider"])
    began = time.monotonic()
    metrics, traces = evaluate(task, lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0]),
                               capture_actions=True)
    native_seconds = time.monotonic() - began
    (args.out / "native.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(args.out / "native-traces.npz", **traces)
    protocol = dict(schema=VALIDATION_SCHEMA, bundle=str(bundle), cases=task.validation_cases,
                    world_model_sha256=sha256(task.engine.rig.path), actor_sha256=contract["actor_sha256"],
                    reference_sha256=contract["reference_sha256"], control_timestep=task.dt)
    path = (args.out / "protocol.json").resolve()
    path.write_text(json.dumps(protocol, indent=2))
    output = (args.out / "godot.json").resolve()
    import mujoco
    env = {**os.environ, "P4F_MIMIC_PUSH_PROTOCOL": str(path), "P4F_MIMIC_REPLAY_OUTPUT": str(output),
           "P4F_MUJOCO_LIBRARY": str(Path(mujoco.__file__).parent)}
    with (args.out / "godot.log").open("w") as log:
        subprocess.run([str(args.godot), "--headless", "--path", str(root),
                        "res://Scenes/RL/Isaac3/MuJoCo/MimicPerturbReplay.tscn"],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    result = compare(metrics, traces, json.loads(output.read_text()), task.dt)
    result.update(native_evaluation_seconds=native_seconds, episodes=metrics["episodes"],
                  confirmed_hits=metrics["confirmed_hits"], survived_hits=metrics["survived_hits"],
                  recovered_hits=metrics["recovered_hits"],
                  scope="Five-second checkpoint-selection suite; parity does not establish long-term recovery")
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed-min", type=float, default=1.)
    parser.add_argument("--speed-max", type=float, default=2.5)
    parser.add_argument("--godot", type=Path, default=Path(os.environ.get("P4F_GODOT_EXE") or
                        Path(__file__).resolve().parents[2] / "tools/Godot/Godot.exe"))
    raise SystemExit(0 if run(parser.parse_args()) else 1)
