"""Measure how a passing native step replay tolerates student action error."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .distill_recovery_step import StepStudent
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import EntryReference
from .recovery_step_motion import evaluate
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_step_playback_sensitivity_v3"
SCENARIOS = ((0., 0.), (.02, 0.), (.02, 2.), (.05, 2.), (1., 2.))


class PlaybackBlend:
    def __init__(self, task, actions, observations, student, q_columns, blend, position_gain):
        self.task = task
        self.actions = actions
        self.observations = observations
        self.student = student
        self.q_columns = q_columns
        self.blend = blend
        self.position_gain = position_gain

    def __call__(self, observation):
        index = min(int(self.task.steps[0]), len(self.actions) - 1)
        nominal = self.actions[index:index + 1]
        correction = self.position_gain * (
            self.observations[index:index + 1, self.q_columns]
            - observation[:, self.q_columns])
        return (nominal + self.blend * (self.student(observation) - nominal)
                + correction).clamp(-1., 1.)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    report_path = args.teacher_run / "report.json"
    teacher = json.loads(report_path.read_text(encoding="utf-8"))
    reference_path = args.teacher_run / "step_reference.npz"
    search = None
    if args.teacher_search is not None:
        search_path = args.teacher_search / "report.json"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        samples_path = args.teacher_search / "best_teacher_samples.npz"
        native = search["best"]["native"]
        if (not native.get("touchdown_retained", False) or not native["passed"]
                or search["source_report_sha256"] != sha256(report_path)
                or search["source_motion_sha256"] != sha256(reference_path)
                or search["teacher_samples_sha256"] != sha256(samples_path)):
            raise ValueError("Require a passing, hash-bound teacher search")
    else:
        samples_path = args.teacher_run / "teacher_samples.npz"
        native = teacher["native_evaluation"]
        if teacher["teacher_samples_sha256"] != sha256(samples_path):
            raise ValueError("Teacher samples hash mismatch")
    if (not native.get("touchdown_retained", False) or not native["passed"]
            or teacher["model_sha256"] != sha256(rig.path)
            or teacher["reference_sha256"] != sha256(reference_path)):
        raise ValueError("Require the intact historical step teacher run")
    source_best = json.loads(args.initialize_from.with_name("best.json").read_text(encoding="utf-8"))
    if source_best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    with np.load(samples_path, allow_pickle=False) as arrays:
        observations = torch.as_tensor(arrays["observation"].copy())
        actions = torch.as_tensor(arrays["action"].copy())
    with np.load(reference_path, allow_pickle=False) as arrays:
        reference = EntryReference(arrays["qpos"], rig, float(arrays["dt"]))
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    if (len(actions) != round(task.episode_seconds / task.dt)
            or len(observations) != len(actions)
            or actions.shape[1] != len(rig.actuators)
            or observations.shape[1] != task.observations().shape[1]):
        raise ValueError("Teacher sample dimensions differ from the native step")
    student = StepStudent(args.initialize_from).eval()
    student.load_state_dict(torch.load(args.student_checkpoint, map_location="cpu",
                                       weights_only=True))
    q_columns = task.engine.policy_control.q_indices - 7 + 15
    results = []
    for blend, position_gain in SCENARIOS:
        controller = PlaybackBlend(task, actions, observations, student, q_columns,
                                   blend, position_gain)
        native, _, _ = evaluate(task, controller, teacher["requested_placement_m"],
                                teacher["phase_seconds"]["touchdown"])
        results.append(dict(student_blend=blend, position_gain=position_gain,
                            native={key: value for key, value in native.items()
                                    if key != "trace"}))
        print(json.dumps(results[-1]), flush=True)
    args.out.mkdir(parents=True)
    report = dict(schema=SCHEMA, teacher_report_sha256=sha256(report_path),
                  teacher_search_sha256=(sha256(args.teacher_search / "report.json")
                                         if search is not None else None),
                  teacher_samples_sha256=sha256(samples_path),
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  student_checkpoint_sha256=sha256(args.student_checkpoint),
                  scenarios=results, admitted_for_training=False,
                  scope="Nominal native playback sensitivity only; no impact robustness, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--teacher-search", type=Path)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
