"""Localize the first untrackable pre-lift pose in native MuJoCo."""
import argparse
import copy
import json
from pathlib import Path

import onnxruntime as ort
import torch

from .baseline import sha256
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .target_control import DelayedTargetControl
from .track_transfer import CalibrationReference


CASES = (
    ("original", None, None, False, 4.),
    ("source_phase", None, None, False, 4.),
    ("freeze_3_25", 3.25, None, False, 4.),
    ("freeze_3_50", 3.5, None, False, 4.),
    ("static_3_25", None, 3.25, False, 4.),
    ("static_3_50", None, 3.5, False, 4.),
    ("floor_only", None, None, True, 4.),
    ("gain_8", None, None, False, 8.),
)


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    contract = json.loads((args.bundle / "contract.json").read_text())
    if contract.get("diagnostic_only") or sha256(args.bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Require an intact trained Stand export")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, freeze, static, floor_only, gain in CASES:
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        class Control(DelayedTargetControl):
            pass
        Control.gain = gain

        def engine_factory(*engine_args, **_):
            return NativeDummyEngine(*engine_args, control_factory=Control)

        task = StandTask(rig, args.reference, 1, "cpu", engine_factory=engine_factory,
                         control_mode="target_pd", contact_mode=mode_from_contract(contract),
                         reference_factory=CalibrationReference)
        # A changed PD gain is an explicit diagnostic contract intervention.
        expected_contract = {**contract, "pd_gain_per_torque_limit": gain}
        validate_reference_substitution(expected_contract, task.observation_contract())
        if not task.reference.placed_step:
            raise ValueError("Require the checked single-step reference")
        task.episode_seconds = 6.
        if freeze is not None or static is not None:
            reference = copy.copy(task.reference)
            index = round((freeze if freeze is not None else static) / reference.dt)
            reference.qpos = reference.qpos.clone()
            reference.qvel = reference.qvel.clone()
            if freeze is not None:
                reference.qpos[index:] = reference.qpos[index].clone()
                reference.qvel[index:] = 0.
            else:
                reference.qpos[:] = reference.qpos[index].clone()
                reference.qvel[:] = 0.
            task.reference = reference
        if floor_only:
            # Private diagnostic model: floor contacts remain; body/body contacts are disabled.
            rig.model.geom_contype[:] = 2
            rig.model.geom_conaffinity[:] = 1
            rig.model.geom_contype[0], rig.model.geom_conaffinity[0] = 1, 2

        def actor(obs):
            obs = obs.clone()
            if name == "source_phase":
                obs[:, -1] *= task.reference.duration / contract["reference_duration"]
            return torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])

        metrics, _ = evaluate(task, actor, episodes=1)
        rows.append(dict(case=name, survived=metrics["successes"] == 1,
                         survival_seconds=metrics["mean_survival_seconds"],
                         mean_root_tracking_error_m=metrics["mean_root_tracking_error_m"],
                         intervention=dict(freeze_after_seconds=freeze, static_pose_seconds=static,
                                           floor_only=floor_only, pd_gain=gain,
                                           actor_phase="source duration" if name == "source_phase" else "reference duration")))
        (args.out / "progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps(rows[-1]), flush=True)
    report = dict(schema="mimic_prelift_native_audit_v1",
                  model_sha256=sha256(rig.path),
                  reference_sha256=task.reference.sha256,
                  actor_sha256=contract["actor_sha256"],
                  duration_seconds=6., cases=rows,
                  scope="Native diagnostic interventions only. No reference, controller, training or viewer promotion.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
