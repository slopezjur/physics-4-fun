"""Native DAgger pilot for the loaded step, retaining quiet Stand behavior."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .checkpoints import standing_passed
from .distill_recovery_step import StepStudent, TrajectoryStudent
from .evaluate_stand import evaluate as evaluate_stand
from .native_engine import NativeDummyEngine
from .optimize_landing_actions import JOINTS, LandingActionCorrection
from .probe_single_support_entry import CONTROLLERS, EntryReference, ScheduledStabilizer
from .recovery_step_motion import (FootAnchorController, TwoFootBalanceController,
                                   evaluate as evaluate_step, support_schedule)
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_step_dagger_pilot_v2"
MIXTURES = (.75, .5, .25, 0.)


def collect(task, actor, motion, schedule, report, mixture, search=None,
            teacher_override=None):
    """Visit student states while the native teacher supplies corrective labels."""
    from envs.base_env import DoneFlags, EnvMode

    times, torques, centers, _ = schedule
    adjusted_centers = centers.copy()
    if search is not None:
        offsets = np.asarray(search["best"]["offsets"])
        transfer = np.clip((times - 2.35) / .20, 0., 1.)
        release = np.clip((search["com_release_end"] - times) / .50, 0., 1.)
        adjusted_centers[:, 1] += offsets[-2] * np.minimum(transfer, release)
        adjusted_centers[:, 0] += offsets[-1] * np.clip((times - 2.55) / .50, 0., 1.)
    balance = ScheduledStabilizer(task, times, torques, adjusted_centers,
                                  CONTROLLERS["balanced_axes"])
    balance.stabilizer.k_i_hr = report.get("hip_integral_ki", 3.)
    balance = TwoFootBalanceController(
        task, balance, report["phase_seconds"]["touchdown"],
        report["requested_placement_m"],
        com_kp=report["landing_com_kp"], com_kd=report["landing_com_kd"])
    teacher = FootAnchorController(
        task, balance, report["phase_seconds"]["swing_start"], None,
        report["foot_anchor_blend"], motion["foot_targets"][:, 0])
    if search is not None:
        teacher = LandingActionCorrection(task, teacher, offsets[:len(JOINTS)],
                                          search["correction_end"])
    if teacher_override is not None:
        teacher = teacher_override
    task.set_mode(EnvMode.TEST)
    task.reset()
    observations, labels = [], []
    actor.eval()
    for _ in range(round(task.episode_seconds / task.dt)):
        with torch.no_grad():
            observation = task.observations()
            label = teacher(observation)
            proposal = actor(observation)
            action = (mixture * label + (1. - mixture) * proposal).clamp(-1., 1.)
        observations.append(observation[0].numpy().copy())
        labels.append(label[0].numpy().copy())
        _, _, done, _ = task.step(action)
        if done[0] != DoneFlags.NULL.value:
            break
    return (np.asarray(observations), np.asarray(labels),
            float(task.steps[0] * task.dt), bool(done[0] == DoneFlags.TIME.value))


def run(args):
    if (args.out.exists() or args.epochs_per_round < 1
            or args.critical_action_weight < 1 or args.transition_weight < 1
            or args.teacher_dataset is not None
            and (args.observation_contract is None or args.teacher_search is None)):
        raise ValueError("Require a new output directory and positive training budget")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    teacher_report_path = args.teacher_run / "report.json"
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    search = None
    if args.teacher_search is not None:
        search_path = args.teacher_search / "report.json"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        samples_path = args.teacher_search / "best_teacher_samples.npz"
        native = search["best"]["native"]
        if (not native.get("touchdown_retained", False) or not native["passed"]
                or search["source_report_sha256"] != sha256(teacher_report_path)
                or search["source_motion_sha256"] != sha256(args.teacher_run / "step_reference.npz")
                or search["teacher_samples_sha256"] != sha256(samples_path)):
            raise ValueError("Require a passing, hash-bound teacher search")
    else:
        samples_path = args.teacher_run / "teacher_samples.npz"
        native = teacher_report["native_evaluation"]
    motion_path = args.teacher_run / "step_reference.npz"
    if (not native.get("touchdown_retained", False)
            or not native["passed"]
            or not teacher_report["track_swing_foot"]
            or teacher_report["two_foot_balance"] != "ankle"
            or teacher_report["model_sha256"] != sha256(rig.path)
            or (search is None and teacher_report["teacher_samples_sha256"] != sha256(samples_path))
            or teacher_report["reference_sha256"] != sha256(motion_path)):
        raise ValueError("Require the intact native-passing step teacher")
    robust = None
    observation_contract = None
    if args.teacher_dataset is not None:
        dataset_path = args.teacher_dataset / "teacher_samples.npz"
        robust = json.loads((args.teacher_dataset / "report.json").read_text(encoding="utf-8"))
        observation_contract = json.loads(args.observation_contract.read_text(encoding="utf-8"))
        if (not robust.get("recovery_gate_passed", False)
                or robust.get("teacher_samples_sha256") != sha256(dataset_path)
                or robust.get("teacher_report_sha256") != sha256(teacher_report_path)
                or search is None
                or robust.get("controller_search_sha256")
                != sha256(args.teacher_search / "report.json")):
            raise ValueError("Require a passing, hash-bound observation teacher dataset")
        samples_path = dataset_path
    best = json.loads(args.initialize_from.with_name("best.json").read_text(encoding="utf-8"))
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    with np.load(samples_path, allow_pickle=False) as arrays:
        step_x = arrays["observation"].copy()
        step_y = arrays["action"].copy()
    with np.load(motion_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
    step = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    step.reference = reference
    step.episode_seconds = reference.duration
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                      8, "cpu", engine_factory=NativeDummyEngine,
                      control_mode="target_pd", contact_mode="support_v2")
    source = StepStudent(args.initialize_from).eval()
    quiet_metrics, quiet_trace = evaluate_stand(quiet, source, capture_actions=True)
    if not standing_passed(quiet_metrics):
        raise ValueError("Source actor fails quiet standing")
    quiet_x = quiet_trace["observations"].reshape(-1, step_x.shape[1])
    quiet_y = quiet_trace["actions"].reshape(-1, step_y.shape[1])
    schedule = support_schedule(rig, motion)
    actuator_names = [rig.model.actuator(int(index)).name for index in rig.actuators]
    action_weights = torch.ones(len(actuator_names))
    for name in ("Shin_L_rx", "Foot_L_rz", "Thigh_L_rx", "Foot_R_rx"):
        action_weights[actuator_names.index(name)] = args.critical_action_weight
    student = copy.deepcopy(source)
    if args.trajectory_feedback:
        with np.load(samples_path, allow_pickle=False) as arrays:
            nominal = arrays["case_id"] == 0
        student = TrajectoryStudent(student, torch.as_tensor(step_x[nominal]),
                                     torch.as_tensor(step_y[nominal]), torch.as_tensor(step_x))
    if args.resume_student is not None:
        student.load_student_state(torch.load(args.resume_student, map_location="cpu",
                                              weights_only=True))
    if args.calibrated_features and not hasattr(student, "feature_std"):
        student.calibrate_features(torch.as_tensor(np.concatenate((step_x, quiet_x))))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    student.to(device)
    action_weights = action_weights.to(device)
    quiet_x = torch.as_tensor(quiet_x, dtype=torch.float32, device=device)
    quiet_y = torch.as_tensor(quiet_y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    history = []
    args.out.mkdir(parents=True)

    for round_index, mixture in enumerate((None, *MIXTURES)):
        rollout = None
        if mixture is not None:
            actor = copy.deepcopy(student).cpu().eval()
            teacher_override = None
            if robust is not None:
                from .observation_step_teacher import ObservationStepTeacher
                teacher_override = ObservationStepTeacher(
                    rig, observation_contract, motion, teacher_report, search, step.dt,
                    adaptive_root_xy=robust["adaptive_root_xy"],
                    nominal_capture_xy=robust["nominal_capture_xy"],
                    capture_target_gain=robust["capture_target_gain"],
                    right_arm_clearance=robust["right_arm_clearance"],
                    right_arm_clearance_scale=robust["right_arm_clearance_scale"],
                    negative_y_roll_gains=robust["negative_y_roll_gains"],
                    negative_x_recovery=robust["negative_x_recovery"])
            visited_x, visited_y, seconds, survived = collect(
                step, actor, motion, schedule, teacher_report, mixture, search,
                teacher_override)
            step_x = np.concatenate((step_x, visited_x))
            step_y = np.concatenate((step_y, visited_y))
            rollout = dict(mixture=mixture, samples=len(visited_x),
                           survival_seconds=seconds, survived=survived)
        x = torch.as_tensor(step_x, dtype=torch.float32, device=device)
        y = torch.as_tensor(step_y, dtype=torch.float32, device=device)
        seconds = x[:, -1] * reference.duration
        phase_weights = torch.ones_like(seconds)
        phase_weights[(seconds >= 1.35) & (seconds < 4.)] = args.transition_weight
        for _ in range(args.epochs_per_round):
            student.train()
            for indices in torch.randperm(len(x), device=device).split(128):
                quiet_indices = torch.randint(len(quiet_x), (len(indices),), device=device)
                error = (student(x[indices]) - y[indices]).square()
                loss_step = (error * action_weights * phase_weights[indices, None]).mean()
                loss_quiet = (student(quiet_x[quiet_indices]) - quiet_y[quiet_indices]).square().mean()
                loss = loss_step + .5 * loss_quiet
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                optimizer.step()
        actor = copy.deepcopy(student).cpu().eval()
        with torch.no_grad():
            imitation_mse = float((actor(torch.as_tensor(step_x))
                                   - torch.as_tensor(step_y)).square().mean())
            native, _, _ = evaluate_step(step, actor,
                                         teacher_report["requested_placement_m"],
                                         teacher_report["phase_seconds"]["touchdown"])
            standing, _ = evaluate_stand(quiet, actor)
        row = dict(round=round_index, rollout=rollout,
                   step_samples=len(step_x), imitation_mse=imitation_mse,
                   native={key: value for key, value in native.items() if key != "trace"},
                   quiet=standing, quiet_passed=standing_passed(standing))
        history.append(row)
        checkpoint = args.out / f"round-{round_index:02d}.pt"
        torch.save(actor.state_dict(), checkpoint)
        row["checkpoint_sha256"] = sha256(checkpoint)
        (args.out / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(dict(round=round_index, rollout=rollout,
                              imitation_mse=imitation_mse,
                              native_passed=native["passed"],
                              native_survival=native["survival_seconds"],
                              quiet_passed=row["quiet_passed"])), flush=True)
    passing = [row for row in history if row["native"]["passed"] and row["quiet_passed"]]
    report = dict(schema=SCHEMA, teacher_report_sha256=sha256(teacher_report_path),
                  teacher_search_sha256=(sha256(args.teacher_search / "report.json")
                                         if search is not None else None),
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  teacher_dataset_sha256=(sha256(args.teacher_dataset / "report.json")
                                          if args.teacher_dataset is not None else None),
                  resume_student_sha256=(sha256(args.resume_student)
                                         if args.resume_student is not None else None),
                  rounds=len(MIXTURES), epochs_per_round=args.epochs_per_round,
                  critical_action_weight=args.critical_action_weight,
                  transition_weight=args.transition_weight,
                  input_features=("trajectory_feedback_v1" if args.trajectory_feedback else
                                  "stand_plus_calibrated_all_v1"
                                  if hasattr(student, "feature_std") else "frozen_stand"),
                  learning_rate=args.learning_rate, seed=args.seed,
                  source_quiet=quiet_metrics, history=history,
                  passing_rounds=[row["round"] for row in passing],
                  passed=bool(passing), admitted_for_deployment=False,
                  scope="Diagnostic native DAgger step pilot; no PPO, ball-hit training, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--teacher-search", type=Path)
    parser.add_argument("--teacher-dataset", type=Path)
    parser.add_argument("--observation-contract", type=Path)
    parser.add_argument("--resume-student", type=Path)
    parser.add_argument("--calibrated-features", action="store_true")
    parser.add_argument("--trajectory-feedback", action="store_true")
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs-per-round", type=int, default=25)
    parser.add_argument("--critical-action-weight", type=float, default=1.)
    parser.add_argument("--transition-weight", type=float, default=1.)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=260926)
    run(parser.parse_args())
