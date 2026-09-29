"""Bounded native teacher robustness probe with motor-demand diagnostics."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .observation_step_teacher import ObservationStepTeacher
from .optimize_landing_actions import JOINTS, LandingActionCorrection
from .probe_single_support_entry import CONTROLLERS, EntryReference, ScheduledStabilizer
from .recovery_step_motion import (FootAnchorController, TwoFootBalanceController,
                                   evaluate, support_schedule)
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_step_teacher_robustness_v4"
PUSH_NEWTONS = 10.
PUSH_SECONDS = .1
VELOCITY_M_S = .05


def cases():
    result = [dict(name="nominal", initial_velocity=(0., 0.), push=None)]
    for axis in (0, 1):
        for sign in (-1, 1):
            velocity = [0., 0.]
            velocity[axis] = sign * VELOCITY_M_S
            result.append(dict(name=f"velocity-{'xy'[axis]}-{sign:+d}",
                               initial_velocity=velocity, push=None))
    for onset in (1.75, 2.75):
        for axis in (0, 1):
            for sign in (-1, 1):
                force = [0., 0., 0.]
                force[axis] = sign * PUSH_NEWTONS
                result.append(dict(name=f"push-{onset:g}-{'xy'[axis]}-{sign:+d}",
                                   initial_velocity=(0., 0.),
                                   push=dict(onset_seconds=onset, duration_seconds=PUSH_SECONDS,
                                             force_world_n=force)))
    return result


def disturbed_recovery_passed(native, touchdown):
    """A hit may break contact briefly if both feet then settle and stay loaded."""
    final_start = native["final_loaded_landing_start_seconds"]
    return bool(native["survived"] and native["final_slow"]
                and native["peak_swing_lift_m"] >= .03
                and native["final_left_foot_shift_m"] >= .13
                and final_start is not None and final_start <= touchdown + 1.
                and native["final_loaded_landing_seconds"] >= 2.)


class ForcedTeacher:
    def __init__(self, task, teacher, push, action_counts, ball_speed=None):
        self.task, self.teacher, self.push = task, teacher, push
        self.action_counts = action_counts
        self.chest = task.rig.model.body("Chest").id
        self.ball_speed = ball_speed
        self.launched = False

    def __call__(self, observation):
        seconds = float(self.task.steps[0] * self.task.dt)
        if (self.ball_speed is not None and self.push is not None and not self.launched
                and seconds >= self.push["onset_seconds"] - .65 / self.ball_speed):
            from .ball import launch_state
            position, velocity = launch_state(
                self.task.engine.datas[0].xpos[self.chest][None],
                np.asarray(self.push["force_world_n"])[None],
                [self.ball_speed], self.task.rig.model.opt.gravity)
            self.task.engine.launch_ball(torch.tensor([0]), position, velocity)
            self.launched = True
        if (self.ball_speed is None and self.push is not None
                and self.push["onset_seconds"] <= seconds
                < self.push["onset_seconds"] + self.push["duration_seconds"]):
            self.task.engine.set_external_force(self.chest,
                                                [self.push["force_world_n"]])
        action = self.teacher(observation)
        phase = "transfer" if seconds < 1.35 else "swing" if seconds < 2.55 else "landing"
        self.action_counts[phase][0] += int((action.abs() >= .99).sum())
        self.action_counts[phase][1] += action.numel()
        return action


class ObservationPlayback:
    """Use recorded actions and observed joint-position error only."""
    def __init__(self, task, actions, observations, gain):
        self.task, self.actions, self.observations = task, actions, observations
        self.gain = gain
        self.columns = task.engine.policy_control.q_indices - 7 + 15

    def __call__(self, observation):
        index = min(int(self.task.steps[0]), len(self.actions) - 1)
        correction = self.gain * (self.observations[index:index + 1, self.columns]
                                  - observation[:, self.columns])
        return (self.actions[index:index + 1] + correction).clamp(-1., 1.)


class RecoveryHandoff:
    def __init__(self, task, base, standing_actor, blend, start):
        self.task, self.base = task, base
        self.standing_actor, self.blend, self.start = standing_actor, blend, start

    def __call__(self, observation):
        action = self.base(observation)
        seconds = float(self.task.steps[0] * self.task.dt)
        weight = self.blend * np.clip((seconds - self.start) / .75, 0., 1.)
        return ((1. - weight) * action
                + weight * self.standing_actor(observation)).clamp(-1., 1.)


def run(args):
    if (args.out.exists() or sum(value is not None for value in
                                  (args.controller_search, args.playback_search,
                                   args.observation_search)) > 1
            or not np.isfinite((args.playback_gain, args.latch_press_m,
                                args.sole_level_gain,
                                args.landing_com_kp if args.landing_com_kp is not None else 10.,
                                args.landing_com_kd if args.landing_com_kd is not None else 2.)).all()
            or args.playback_gain < 0 or not 0. <= args.latch_press_m <= .04
            or not 0. <= args.sole_level_gain <= 1.
            or args.landing_com_kp is not None and not 0. <= args.landing_com_kp <= 20.
            or args.landing_com_kd is not None and not 0. <= args.landing_com_kd <= 5.
            or not np.isfinite(args.capture_target_gain)
            or not 0. <= args.capture_target_gain <= 3.
            or args.latch_world_at_contact and args.observation_search is None
            or args.sole_level_gain and args.observation_search is None
            or args.adaptive_ankle and args.observation_search is None
            or args.capture_target_gain and args.observation_search is None
            or args.right_arm_clearance and args.observation_search is None
            or not 0. <= args.right_arm_clearance_scale <= 1.
            or args.negative_y_roll_kp is not None and not 0. <= args.negative_y_roll_kp <= 20.
            or args.negative_y_roll_kd is not None and not 0. <= args.negative_y_roll_kd <= 10.
            or (args.negative_y_roll_kp is None) != (args.negative_y_roll_kd is None)
            or args.negative_y_roll_kp is not None and not args.adaptive_ankle
            or args.negative_x_recovery and not args.adaptive_ankle
            or not .5 <= args.push_scale <= 2.
            or not -.2 <= args.early_onset_shift <= .2
            or (args.landing_com_kp is not None or args.landing_com_kd is not None)
            and args.observation_search is None
            or args.left_ankle_offset is not None and args.observation_search is None
            or args.left_ankle_offset is not None
            and not -1. <= args.left_ankle_offset <= 1.
            or args.com_x_bias is not None and args.observation_search is None
            or args.com_x_bias is not None and not 0. <= args.com_x_bias <= .1
            or not 0. <= args.handoff_blend <= .5
            or args.handoff_blend and args.handoff_checkpoint is None
            or (args.student is None) != (args.student_source is None)
            or args.ball_speed is not None and not 1. <= args.ball_speed <= 8.
            or args.residual_checkpoint is not None and (args.observation_search is None or args.student is not None)):
        raise ValueError("Require a new output directory, one controller, and a valid playback gain")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    report_path = args.teacher_run / "report.json"
    source = json.loads(report_path.read_text(encoding="utf-8"))
    search = None
    selected_search = args.controller_search or args.observation_search
    if selected_search is not None:
        search_path = selected_search / "report.json"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        if (not search["best"]["native"]["passed"]
                or search["model_sha256"] != sha256(rig.path)
                or search["source_report_sha256"] != sha256(report_path)):
            raise ValueError("Require a passing, hash-bound native controller search")
        if args.left_ankle_offset is not None or args.com_x_bias is not None:
            search = {**search, "best": {**search["best"],
                      "offsets": [*search["best"]["offsets"]]}}
            if args.left_ankle_offset is not None:
                search["best"]["offsets"][3] = args.left_ankle_offset
            if args.com_x_bias is not None:
                search["best"]["offsets"][-1] = args.com_x_bias
    if args.playback_search is not None:
        playback_path = args.playback_search / "report.json"
        playback = json.loads(playback_path.read_text(encoding="utf-8"))
        samples_path = args.playback_search / "best_teacher_samples.npz"
        if (not playback["best"]["native"]["passed"]
                or playback["model_sha256"] != sha256(rig.path)
                or playback["source_report_sha256"] != sha256(report_path)
                or playback["teacher_samples_sha256"] != sha256(samples_path)):
            raise ValueError("Require a passing, hash-bound playback source")
        with np.load(samples_path, allow_pickle=False) as arrays:
            playback_observations = torch.as_tensor(arrays["observation"].copy())
            playback_actions = torch.as_tensor(arrays["action"].copy())
    reference_path = args.teacher_run / "step_reference.npz"
    if (not source["native_evaluation"]["passed"]
            or source["model_sha256"] != sha256(rig.path)
            or source["reference_sha256"] != sha256(reference_path)
            or not source["track_swing_foot"]
            or source["two_foot_balance"] != "ankle"):
        raise ValueError("Require the intact historical step teacher")
    with np.load(reference_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    if args.observation_search is not None and args.contract is None:
        raise ValueError("Observation controller requires the actor contract")
    contract = (json.loads(args.contract.read_text(encoding="utf-8"))
                if args.observation_search is not None else None)
    standing_actor = None
    student = None
    residual_actor = None
    if args.residual_checkpoint is not None:
        from .distill_recovery_step import StepStudent
        protocol = json.loads(args.residual_checkpoint.with_name("protocol.json").read_text(encoding="utf-8"))
        if (protocol["schema"] != "mimic_stateful_recovery_residual_v1"
                or protocol["model_sha256"] != sha256(rig.path)
                or protocol["teacher_report_sha256"] != sha256(report_path)
                or protocol["teacher_search_sha256"] != sha256(args.observation_search / "report.json")
                or protocol["observation_contract_sha256"] != sha256(args.contract)):
            raise ValueError("Residual checkpoint controller contract mismatch")
        residual_actor = StepStudent(args.residual_checkpoint).eval()
    if args.student is not None:
        from .distill_recovery_step import load_student
        student = load_student(args.student, args.student_source)
    if args.handoff_checkpoint is not None:
        from .distill_recovery_step import StepStudent
        best = json.loads(args.handoff_checkpoint.with_name("best.json").read_text(
            encoding="utf-8"))
        if best["checkpoint_sha256"] != sha256(args.handoff_checkpoint):
            raise ValueError("Standing handoff checkpoint hash mismatch")
        standing_actor = StepStudent(args.handoff_checkpoint).eval()
    adaptive_root_xy = None
    nominal_capture_xy = None
    if args.adaptive_ankle or args.capture_target_gain or args.negative_x_recovery:
        samples_path = selected_search / "best_teacher_samples.npz"
        if search["teacher_samples_sha256"] != sha256(samples_path):
            raise ValueError("Adaptive ankle calibration requires intact teacher samples")
        with np.load(samples_path, allow_pickle=False) as arrays:
            index = round(2.15 / float(arrays["dt"]))
            row = arrays["observation"][index].copy()
            if args.adaptive_ankle:
                adaptive_root_xy = row[:2].tolist()
            if args.capture_target_gain or args.negative_x_recovery:
                from .observation_com import ObservationCom
                observer = ObservationCom(rig, contract)
                observer(torch.as_tensor(row[None]))
                data = observer.data
                nominal_capture_xy = (data.subtree_com[1, :2]
                    + data.subtree_linvel[1, :2]
                    * np.sqrt(data.subtree_com[1, 2] / 9.81)).copy()
    if residual_actor is not None and "controller_settings" in protocol:
        actual_settings = dict(
            adaptive_root_xy=adaptive_root_xy,
            nominal_capture_xy=nominal_capture_xy.tolist() if nominal_capture_xy is not None else None,
            capture_target_gain=args.capture_target_gain,
            right_arm_clearance=args.right_arm_clearance,
            right_arm_clearance_scale=args.right_arm_clearance_scale,
            negative_y_roll_gains=([args.negative_y_roll_kp, args.negative_y_roll_kd]
                                   if args.negative_y_roll_kp is not None else None),
            negative_x_recovery=args.negative_x_recovery)
        for key, value in protocol["controller_settings"].items():
            actual = actual_settings[key]
            if ((value is None) != (actual is None)
                    or value is not None and not np.allclose(value, actual, atol=1e-8, rtol=1e-6)):
                raise ValueError(f"Residual baseline controller setting mismatch: {key}")
    reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
    from .ball_native import BallNativeEngine
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=(BallNativeEngine if args.ball_speed else NativeDummyEngine),
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    if args.ball_mass is not None:
        if args.ball_speed is None:
            raise ValueError("A curriculum ball mass requires physical ball evaluation")
        from .ball import set_curriculum_ball_mass
        set_curriculum_ball_mass(task.engine, args.ball_mass)
    schedule = support_schedule(rig, motion)
    times, torques, centers, _ = schedule
    control = task.engine.policy_control
    base_reset = task.reset
    base_physics_control = control.physics_control
    rows = []
    captured_samples = []

    selected_cases = cases()
    if args.case:
        names = {case["name"] for case in selected_cases}
        if set(args.case) - names:
            raise ValueError(f"Unknown cases: {sorted(set(args.case) - names)}")
        selected_cases = [case for case in selected_cases if case["name"] in args.case]
    for case in selected_cases:
        if case["push"] is not None:
            case["push"]["force_world_n"] = [value * args.push_scale for value in
                                              case["push"]["force_world_n"]]
            if case["push"]["onset_seconds"] == 1.75:
                case["push"]["onset_seconds"] += args.early_onset_shift
    for case in selected_cases:
        velocity = case["initial_velocity"]

        def reset(ids=None):
            base_reset(ids)
            data = task.engine.datas[0]
            data.qvel[:2] += velocity
            mujoco.mj_forward(task.engine.rig.model, data)
            task.engine._ground_force_cache[0] = task.engine._read_ground_forces(data)
            return task.observations(), {}

        task.reset = reset
        motor_counts = {phase: [0, 0, 0.] for phase in ("transfer", "swing", "landing")}
        peak_motor = {phase: None for phase in motor_counts}
        first_motor_saturation = {phase: None for phase in motor_counts}
        motor_names = [rig.model.actuator(int(index)).name for index in rig.actuators]
        action_counts = {phase: [0, 0] for phase in motor_counts}

        def physics_control(qpos, qvel):
            result = base_physics_control(qpos, qvel)
            if control.active[0, 60] > 0:
                demand = (control.gain * (control.active[:, :30] - qpos[:, control.q_indices])
                          + control.gain * control.damping_time
                          * (control.active[:, 30:60] - qvel[:, control.v_indices]))
                seconds = float(task.steps[0] * task.dt)
                phase = "transfer" if seconds < 1.35 else "swing" if seconds < 2.55 else "landing"
                motor_counts[phase][0] += int((demand.abs() >= 1.).sum())
                motor_counts[phase][1] += demand.numel()
                if first_motor_saturation[phase] is None and (demand.abs() >= 1.).any():
                    index = int(torch.argmax(demand.abs()[0]))
                    first_motor_saturation[phase] = dict(joint=motor_names[index],
                                                         seconds=seconds)
                maximum, joint = torch.max(demand.abs()[0], dim=0)
                if float(maximum) > motor_counts[phase][2]:
                    motor_counts[phase][2] = float(maximum)
                    peak_motor[phase] = dict(joint=motor_names[int(joint)], seconds=seconds,
                                             normalized_demand=float(demand[0, joint]))
            return result

        control.physics_control = physics_control
        adjusted_centers = centers.copy()
        if search is not None:
            offsets = np.asarray(search["best"]["offsets"])
            transfer = np.clip((times - 2.35) / .20, 0., 1.)
            release = np.clip((search["com_release_end"] - times) / .50, 0., 1.)
            adjusted_centers[:, 1] += offsets[-2] * np.minimum(transfer, release)
            adjusted_centers[:, 0] += offsets[-1] * np.clip((times - 2.55) / .50, 0., 1.)
        balance = ScheduledStabilizer(task, times, torques, adjusted_centers,
                                      CONTROLLERS["balanced_axes"])
        balance.stabilizer.k_i_hr = source.get("hip_integral_ki", 3.)
        balance = TwoFootBalanceController(
            task, balance, source["phase_seconds"]["touchdown"],
            source["requested_placement_m"],
            com_kp=source["landing_com_kp"], com_kd=source["landing_com_kd"])
        teacher = FootAnchorController(
            task, balance, source["phase_seconds"]["swing_start"], None,
            source["foot_anchor_blend"], motion["foot_targets"][:, 0])
        corrected = (ObservationStepTeacher(rig, contract, motion, source, search,
                                            task.dt, args.latch_world_at_contact,
                                            args.latch_press_m, args.sole_level_gain,
                                            adaptive_root_xy, args.landing_com_kp,
                                            args.landing_com_kd, nominal_capture_xy,
                                            args.capture_target_gain,
                                            args.right_arm_clearance,
                                            args.right_arm_clearance_scale,
                                            ((args.negative_y_roll_kp, args.negative_y_roll_kd)
                                             if args.negative_y_roll_kp is not None else None),
                                            args.negative_x_recovery)
                     if args.observation_search is not None else
                     ObservationPlayback(task, playback_actions, playback_observations,
                                         args.playback_gain) if args.playback_search is not None else
                     LandingActionCorrection(task, teacher, offsets[:len(JOINTS)],
                                              search["correction_end"])
                     if search is not None else teacher)
        controlled = (RecoveryHandoff(task, corrected, standing_actor,
                                      args.handoff_blend, args.handoff_start)
                      if args.handoff_blend else corrected)
        if student is not None:
            controlled = student
        if residual_actor is not None:
            from .train_recovery_residual import ResidualController
            controlled = ResidualController(corrected, residual_actor, protocol["residual_scale"])
        forced = ForcedTeacher(task, controlled, case["push"], action_counts, args.ball_speed)
        native, _, samples = evaluate(task, forced, source["requested_placement_m"],
                                      source["phase_seconds"]["touchdown"])
        if args.save_samples:
            captured_samples.append(samples)
        recovery_passed = (native["passed"] if case["push"] is None else
                           disturbed_recovery_passed(
                               native, source["phase_seconds"]["touchdown"]))
        ball_hit = bool(task.engine.hit[0]) if args.ball_speed and case["push"] else None
        if ball_hit is False:
            recovery_passed = False
        row = dict(**case, native={key: value for key, value in native.items()
                                   if key != "trace"},
                   recovery_passed=recovery_passed,
                   ball_hit=ball_hit,
                   ball_impulse_world_ns=(task.engine.hit_impulse[0].tolist() if args.ball_speed else None),
                   ball_hit_bodies=([rig.model.body(index).name for index in sorted(task.engine.hit_bodies[0])]
                                    if args.ball_speed else None),
                   action_saturation={phase: counts[0] / max(1, counts[1])
                                      for phase, counts in action_counts.items()},
                   motor_saturation={phase: dict(fraction=counts[0] / max(1, counts[1]),
                                                 maximum_unclipped_demand=counts[2],
                                                 peak=peak_motor[phase],
                                                 first_saturation=first_motor_saturation[phase])
                                     for phase, counts in motor_counts.items()},
                   foot_anchor_saturated_fraction=(None if args.playback_search is not None
                                                   or args.observation_search is not None else
                                                   teacher.saturated / max(1, teacher.samples)))
        if args.observation_search is not None and student is None:
            row["selected_left_ankle_offset"] = float(corrected.controller.offsets[3])
            row["ankle_selection"] = corrected.ankle_selection
            row["phase_snapshots"] = corrected.phase_snapshots
            row["capture_target_adjustment_xy"] = corrected.capture_adjustment
            row["roll_gains_selected"] = corrected.roll_gains_selected
            row["negative_x_selected"] = corrected.negative_x_selected
        if args.include_trace:
            row["trace"] = native["trace"]
        if args.ball_speed and case["push"] is not None:
            row["push"] = None
            row["ball_launch"] = dict(
                nominal_contact_seconds=case["push"]["onset_seconds"],
                direction=case["push"]["force_world_n"], speed_m_s=args.ball_speed,
                mass_kg=float(task.engine.rig.model.body_mass[task.engine.ball_body]))
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items()
                          if key not in ("trace", "phase_snapshots")}),
              flush=True)

    task.reset = base_reset
    control.physics_control = base_physics_control
    args.out.mkdir(parents=True)
    samples_path = args.out / "teacher_samples.npz"
    if args.save_samples:
        np.savez_compressed(
            samples_path,
            observation=np.concatenate([item["observation"] for item in captured_samples]),
            action=np.concatenate([item["action"] for item in captured_samples]),
            case_id=np.concatenate([np.full(len(item["action"]), index, dtype=np.int16)
                                    for index, item in enumerate(captured_samples)]),
            dt=captured_samples[0]["dt"])
    report = dict(schema=SCHEMA, teacher_report_sha256=sha256(report_path),
                  student_checkpoint_sha256=sha256(args.student) if args.student else None,
                  student_source_sha256=sha256(args.student_source) if args.student_source else None,
                  residual_checkpoint_sha256=(sha256(args.residual_checkpoint)
                                              if args.residual_checkpoint else None),
                  controller_search_sha256=(sha256(selected_search / "report.json")
                                            if selected_search is not None else None),
                  observation_only=args.observation_search is not None,
                  latch_world_at_contact=args.latch_world_at_contact,
                  latch_press_m=args.latch_press_m,
                  sole_level_gain=args.sole_level_gain,
                  adaptive_ankle=args.adaptive_ankle,
                  adaptive_root_xy=adaptive_root_xy,
                  landing_com_kp=args.landing_com_kp,
                  landing_com_kd=args.landing_com_kd,
                  capture_target_gain=args.capture_target_gain,
                  right_arm_clearance=args.right_arm_clearance,
                  right_arm_clearance_scale=args.right_arm_clearance_scale,
                  negative_y_roll_gains=([args.negative_y_roll_kp, args.negative_y_roll_kd]
                                         if args.negative_y_roll_kp is not None else None),
                  negative_x_recovery=args.negative_x_recovery,
                  nominal_capture_xy=(nominal_capture_xy.tolist()
                                      if nominal_capture_xy is not None else None),
                  left_ankle_offset=args.left_ankle_offset,
                  com_x_bias=args.com_x_bias,
                  handoff_checkpoint_sha256=(sha256(args.handoff_checkpoint)
                                             if args.handoff_checkpoint is not None else None),
                  handoff_blend=args.handoff_blend,
                  handoff_start=args.handoff_start,
                  selected_cases=[case["name"] for case in selected_cases],
                  push_scale=args.push_scale,
                  ball_speed_m_s=args.ball_speed,
                  ball_mass_kg=(float(task.engine.rig.model.body_mass[task.engine.ball_body])
                                if args.ball_speed else None),
                  world_model_sha256=sha256(task.engine.rig.path),
                  early_onset_shift=args.early_onset_shift,
                  observation_contract_sha256=(sha256(args.contract)
                                               if args.observation_search is not None else None),
                  playback_search_sha256=(sha256(args.playback_search / "report.json")
                                          if args.playback_search is not None else None),
                  playback_gain=args.playback_gain if args.playback_search is not None else None,
                  model_sha256=sha256(rig.path), reference_sha256=sha256(reference_path),
                  cases=rows, passed_cases=sum(row["native"]["passed"] for row in rows),
                  recovery_passed_cases=sum(row["recovery_passed"] for row in rows),
                  recovery_gate_passed=(len(rows) == len(cases())
                                        and all(row["recovery_passed"] for row in rows)),
                  teacher_samples_sha256=sha256(samples_path) if args.save_samples else None,
                  admitted_for_training=False,
                  scope="Native step controller under recorded disturbances; no export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--student-source", type=Path)
    parser.add_argument("--ball-speed", type=float)
    parser.add_argument("--ball-mass", type=float)
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--controller-search", type=Path)
    parser.add_argument("--observation-search", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--playback-search", type=Path)
    parser.add_argument("--playback-gain", type=float, default=2.)
    parser.add_argument("--case", action="append")
    parser.add_argument("--latch-world-at-contact", action="store_true")
    parser.add_argument("--latch-press-m", type=float, default=0.)
    parser.add_argument("--left-ankle-offset", type=float)
    parser.add_argument("--com-x-bias", type=float)
    parser.add_argument("--handoff-checkpoint", type=Path)
    parser.add_argument("--handoff-blend", type=float, default=0.)
    parser.add_argument("--handoff-start", type=float, default=4.)
    parser.add_argument("--sole-level-gain", type=float, default=0.)
    parser.add_argument("--adaptive-ankle", action="store_true")
    parser.add_argument("--landing-com-kp", type=float)
    parser.add_argument("--landing-com-kd", type=float)
    parser.add_argument("--capture-target-gain", type=float, default=0.)
    parser.add_argument("--right-arm-clearance", action="store_true")
    parser.add_argument("--right-arm-clearance-scale", type=float, default=1.)
    parser.add_argument("--negative-y-roll-kp", type=float)
    parser.add_argument("--negative-y-roll-kd", type=float)
    parser.add_argument("--negative-x-recovery", action="store_true")
    parser.add_argument("--push-scale", type=float, default=1.)
    parser.add_argument("--early-onset-shift", type=float, default=0.)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
