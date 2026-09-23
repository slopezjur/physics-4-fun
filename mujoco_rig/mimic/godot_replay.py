"""Run the actual Godot scene, then compare observations/actions and episode outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np
import onnxruntime as ort
import torch

from .runtime import activate
from .rig import Rig
from .godot_bundle import prepare
from .contact_contract import mode_from_contract


def replay(args):
    activate(args.mimickit)
    from envs.base_env import EnvMode, DoneFlags
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine

    bundle = args.bundle.resolve()
    prepare(bundle)
    # A new path prevents a successful but misconfigured engine launch from reusing stale evidence.
    output = bundle.parent / f"godot-replay-{uuid.uuid4().hex}.json"
    environment = os.environ.copy()
    import mujoco
    environment.update(P4F_MIMIC_BUNDLE=str(bundle), P4F_MIMIC_REPLAY_OUTPUT=str(output),
                       P4F_MUJOCO_LIBRARY=str(Path(mujoco.__file__).parent))
    root = Path(__file__).resolve().parents[2]
    with (bundle.parent / "godot-replay.log").open("w") as log:
        subprocess.run([str(args.godot), "--headless", "--path", str(root),
                        "res://Scenes/RL/Isaac3/MuJoCo/MimicStandReplay.tscn"], env=environment,
                       stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    actual = json.loads(output.read_text())["episodes"]
    if len(actual) != 8:
        raise ValueError("Godot did not evaluate all eight phases")
    output.replace(bundle.parent / "godot-replay.json")
    rig = Rig.load(root / "mujoco_rig/dummy.xml")
    contract = json.loads((bundle / "contract.json").read_text())
    mode = "target_pd" if contract["action_mode"] == "reference_relative_target_pd" else "torque"
    task = StandTask(rig, bundle / "stand_reference.npz", 1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode=mode, contact_mode=mode_from_contract(contract))
    expected_contract = task.observation_contract()
    if any(contract.get(key) != value for key, value in expected_contract.items() if key != "deployment_status"):
        raise ValueError("Replay implementation does not match bundle control/observation contract")
    task.set_mode(EnvMode.TEST)
    session = ort.InferenceSession(str(bundle / "stand.onnx"), providers=["CPUExecutionProvider"])
    errors = dict(observation=0.0, action=0.0, pose=0.0)
    expected = []
    for i, episode in enumerate(actual):
        task.reset()
        task.offset[:] = i / 7 * (task.reference.duration - 3)
        q, v = task.reference.sample(task.offset)
        task.engine.reset_envs(task.ids, q, v)
        tracking = []
        for step in range(int(np.ceil(3 / task.dt))):
            observation = task.observations()
            action = session.run(None, {"observation": observation.numpy()})[0]
            _, _, done, _ = task.step(torch.tensor(action))
            target, _ = task.reference.sample(task.offset + task.steps * task.dt)
            tracking.append(float(torch.linalg.vector_norm(task.engine.get_root_pos(0) - target[:, :3])))
            if step < len(episode["prefix"]):
                prefix = episode["prefix"][step]
                for name, value, other in (("observation", observation.numpy()[0], prefix["observation"]),
                                           ("action", action[0], prefix["action"]),
                                           ("pose", task.engine.native_qpos().numpy()[0], prefix["qpos_after"])):
                    errors[name] = max(errors[name], float(np.abs(value - np.asarray(other)).max()))
            if int(done[0]) != DoneFlags.NULL.value:
                break
        expected.append({"success": int(done[0]) == DoneFlags.TIME.value,
                         "survival_seconds": (step + 1) * task.dt,
                         "mean_root_tracking_error_m": float(np.mean(tracking))})
    duration_error = max(abs(a["survival_seconds"] - b["survival_seconds"]) for a, b in zip(actual, expected))
    checks = {"prefix_observations": errors["observation"] < 0.003,
              "prefix_actions": errors["action"] < 1e-4, "prefix_poses": errors["pose"] < 1e-4,
              "episode_outcomes": all(a["success"] == b["success"] for a, b in zip(actual, expected)),
              "episode_duration": duration_error < 0.1}
    report = {"passed": all(checks.values()), "checks": checks, "prefix_errors": errors,
              "max_survival_difference_seconds": duration_error, "python_onnx": expected,
              "godot": [{k: v for k, v in episode.items() if k != "prefix"} for episode in actual]}
    (bundle.parent / "godot-parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--godot", type=Path,
                        default=Path(os.environ.get("P4F_GODOT_EXE") or
                                     Path(__file__).resolve().parents[2] / "tools/Godot/Godot.exe"))
    raise SystemExit(0 if replay(parser.parse_args()) else 1)
