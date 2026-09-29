"""Compare one Godot recovery scene trial with the same native MuJoCo episode."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ball import set_curriculum_ball_mass
from .ball_native import BallNativeEngine
from .probe_recovery_robustness import ForcedTeacher, disturbed_recovery_passed
from .probe_single_support_entry import EntryReference
from .recovery_policy_server import load
from .recovery_step_motion import evaluate
from .rig import Rig
from .runtime import activate


def validate(bundle, trace_path, mimickit, out):
    activate(mimickit)
    from .stand_task import StandTask

    torch.set_num_threads(1)
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if not records or records[0] != {"reset": True}:
        raise ValueError("Godot trace must begin with a recovery reset")
    scene = [row for row in records[1:] if "observation" in row]
    scene_obs = np.asarray([row["observation"] for row in scene], dtype=np.float32)
    scene_act = np.asarray([row["action"] for row in scene], dtype=np.float32)
    manifest = json.loads((bundle / "recovery.json").read_text(encoding="utf-8"))
    rig = Rig.load(bundle / "dummy.xml")
    with np.load(bundle / "motion.npz", allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}

    def reference_factory(_, model_rig, device):
        return EntryReference(motion["qpos"], model_rig, float(motion["dt"]), device)

    task = StandTask(rig, bundle / "motion.npz", 1, "cpu", engine_factory=BallNativeEngine,
                     control_mode="target_pd", contact_mode="support_v2",
                     reference_factory=reference_factory)
    task.episode_seconds = task.reference.duration
    set_curriculum_ball_mass(task.engine, manifest["ball_mass_kg"])
    controller, _ = load(bundle)
    push = dict(onset_seconds=1.75, duration_seconds=.1, force_world_n=[1., 0., 0.])
    forced = ForcedTeacher(task, controller, push,
                           {phase: [0, 0] for phase in ("transfer", "swing", "landing")}, 1.)
    metrics, _, samples = evaluate(task, forced, .15, 2.55)
    native_obs, native_act = samples["observation"], samples["action"]
    count = min(len(scene_obs), len(native_obs))
    differences = np.abs(scene_obs[:count] - native_obs[:count])
    actions = np.abs(scene_act[:count] - native_act[:count])
    observation_worst = np.unravel_index(np.argmax(differences), differences.shape)
    action_worst = np.unravel_index(np.argmax(actions), actions.shape)
    report = dict(scene_ball_mass_kg=manifest["ball_mass_kg"],
                  trained_ball_mass_kg=manifest.get("trained_ball_mass_kg", .25),
                  scene_steps=len(scene_obs), native_steps=len(native_obs),
                  ball_hit=bool(task.engine.hit[0]),
                  native_recovery_passed=disturbed_recovery_passed(metrics, 2.55),
                  strict_touchdown_passed=bool(metrics["passed"]),
                  native_survived=bool(metrics["survived"]),
                  native_survival_seconds=metrics["survival_seconds"],
                  final_loaded_landing_seconds=metrics["final_loaded_landing_seconds"],
                  final_loaded_landing_start_seconds=metrics["final_loaded_landing_start_seconds"],
                  final_slow=bool(metrics["final_slow"]),
                  observation_max_abs=float(differences.max()),
                  observation_worst_step=int(observation_worst[0]),
                  observation_worst_column=int(observation_worst[1]),
                  action_max_abs=float(actions.max()),
                  action_worst_step=int(action_worst[0]),
                  action_worst_column=int(action_worst[1]),
                  first_second_observation_max_abs=float(differences[:60].max()),
                  first_second_action_max_abs=float(actions[:60].max()),
                  first_divergent_step=(int(np.flatnonzero(differences.max(axis=1) > .01)[0])
                                        if np.any(differences.max(axis=1) > .01) else None))
    report["parity_passed"] = bool(len(scene_obs) == len(native_obs)
                                   and report["observation_max_abs"] < .005
                                   and report["action_max_abs"] < .0001)
    report["passed"] = bool(report["parity_passed"] and report["ball_hit"]
                            and report["native_recovery_passed"])
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "trace", "mimickit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.bundle, args.trace, args.mimickit, args.out)))
