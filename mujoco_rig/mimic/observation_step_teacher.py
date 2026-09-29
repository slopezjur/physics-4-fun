"""Replay the native step controller using only the support-v2 actor observation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .optimize_landing_actions import JOINTS, LandingActionCorrection
from .probe_single_support_entry import CONTROLLERS, EntryReference, ScheduledStabilizer
from .recovery_step_motion import (FootAnchorController, TwoFootBalanceController,
                                   evaluate, support_schedule)
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_observation_step_teacher_v1"
RIGHT_ARM_CLEARANCE = (0.032089785225964874, -0.010444048926304402,
                       -0.09595030766848994, 0.0108705773625742)


class ObservationEngine:
    """Expose reconstructed pose and observation loads to the existing controller."""

    def __init__(self, observer):
        self.observer = observer
        self.datas = [observer.data]
        self.observation = None

    def get_root_vel(self, _):
        return self.observation[:, 9:12]

    def get_root_ang_vel(self, _):
        return self.observation[:, 12:15]

    def get_ground_contact_forces(self, _):
        loads = torch.zeros((1, 4, 3), dtype=self.observation.dtype)
        loads[0, :2, 2] = self.observation[0, 93:95] * 1000.
        return loads


class ObservationStepTeacher:
    def __init__(self, rig, contract, motion, source, search, dt,
                 latch_world_at_contact=False, latch_press_m=0.,
                 sole_level_gain=0., adaptive_root_xy=None,
                 landing_com_kp=None, landing_com_kd=None,
                 nominal_capture_xy=None, capture_target_gain=0.,
                 right_arm_clearance=False, right_arm_clearance_scale=1.,
                 negative_y_roll_gains=None, negative_x_recovery=False,
                 cached_support=None):
        if source["hip_integral_ki"] != 0.:
            raise ValueError("An observation-only controller cannot recover hidden integral state")
        self.observer = ObservationCom(rig, contract)
        self.engine = ObservationEngine(self.observer)
        reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
        self.task = SimpleNamespace(rig=rig, reference=reference, engine=self.engine,
                                    dt=dt, offset=torch.zeros(1), steps=torch.zeros(1),
                                    allowed_ground=np.arange(4))
        if cached_support is None:
            times, torques, centers, _ = support_schedule(rig, motion)
        else:
            times = np.asarray(cached_support["times"], dtype=np.float64).copy()
            torques = np.asarray(cached_support["torques"], dtype=np.float64).copy()
            centers = np.asarray(cached_support["centers"], dtype=np.float64).copy()
            interval = round(.25 / float(motion["dt"]))
            expected_times = np.arange(0, len(motion["qpos"]), interval) * float(motion["dt"])
            if (times.shape != expected_times.shape
                    or torques.shape != (len(times), len(rig.actuators))
                    or centers.shape != (len(times), 3)
                    or not np.allclose(times, expected_times, atol=1e-9, rtol=0)
                    or not np.isfinite(torques).all() or not np.isfinite(centers).all()):
                raise ValueError("Invalid cached recovery support schedule")
        offsets = np.asarray(search["best"]["offsets"])
        transfer = np.clip((times - 2.35) / .20, 0., 1.)
        release = np.clip((search["com_release_end"] - times) / .50, 0., 1.)
        centers[:, 1] += offsets[-2] * np.minimum(transfer, release)
        centers[:, 0] += offsets[-1] * np.clip((times - 2.55) / .50, 0., 1.)
        balance = ScheduledStabilizer(self.task, times, torques, centers,
                                      CONTROLLERS["balanced_axes"])
        balance.stabilizer.k_i_hr = 0.
        landing = TwoFootBalanceController(
            self.task, balance, 2.55, .15,
            com_kp=source["landing_com_kp"] if landing_com_kp is None else landing_com_kp,
            com_kd=source["landing_com_kd"] if landing_com_kd is None else landing_com_kd)
        anchor = FootAnchorController(self.task, landing, 1.35,
                                      .15 if latch_world_at_contact else None,
                                      source["foot_anchor_blend"],
                                      motion["foot_targets"][:, 0].copy(),
                                      latch_world_at_contact, 2.55, latch_press_m)
        self.balance, self.landing, self.anchor = balance, landing, anchor
        self.controller = LandingActionCorrection(self.task, anchor,
                                                   offsets[:len(JOINTS)],
                                                   search["correction_end"])
        self.duration = reference.duration
        self.adaptive_root_xy = adaptive_root_xy
        self.nominal_capture_xy = nominal_capture_xy
        self.capture_target_gain = capture_target_gain
        self.right_arm_clearance = right_arm_clearance
        self.right_arm_clearance_scale = right_arm_clearance_scale
        self.negative_y_roll_gains = negative_y_roll_gains
        self.negative_x_recovery = negative_x_recovery
        self.nominal_roll_gains = (balance.stabilizer.k_hr, balance.stabilizer.d_hr)
        self.nominal_landing_gains = (landing.com_kp, landing.com_kd)
        self.nominal_capture_gain = capture_target_gain
        self.roll_gains_selected = False
        self.negative_x_selected = False
        self.original_targets = anchor.targets.copy()
        self.capture_adjustment = None
        self.original_ankle_offset = float(self.controller.offsets[3])
        self.ankle_offset_selected = False
        self.ankle_selection = None
        self.phase_snapshots = {}
        self.sole_level_gain = sole_level_gain
        self.left_foot_geom = rig.model.geom("g_Foot_L").id
        self.rig_foot_body = rig.model.body("Foot_L").id
        self.left_ankle_joint = rig.model.joint("Foot_L_rx").id
        actuator_names = [rig.model.actuator(int(index)).name for index in rig.actuators]
        self.left_ankle_action = actuator_names.index("Foot_L_rx")
        self.right_arm_actions = [actuator_names.index(name) for name in
                                  ("UpperArm_R_rx", "UpperArm_R_ry",
                                   "UpperArm_R_rz", "Forearm_R_rx")]

    def reset(self):
        self.balance.stabilizer.integral_y = [0.]
        self.balance.stabilizer.k_ap = CONTROLLERS["balanced_axes"]["k_ankle_pitch"]
        self.balance.stabilizer.d_ap = CONTROLLERS["balanced_axes"]["d_ankle_pitch"]
        self.landing.start_x = None
        self.landing.landed = False
        self.anchor.start = None
        self.anchor.latched_world = None
        self.anchor.latched_at_seconds = None
        self.anchor.samples = 0
        self.anchor.saturated = 0
        self.anchor.targets[:] = self.original_targets
        self.capture_adjustment = None
        self.controller.offsets[3] = self.original_ankle_offset
        self.ankle_offset_selected = False
        self.ankle_selection = None
        self.roll_gains_selected = False
        self.negative_x_selected = False
        self.balance.stabilizer.k_hr, self.balance.stabilizer.d_hr = self.nominal_roll_gains
        self.landing.com_kp, self.landing.com_kd = self.nominal_landing_gains
        self.capture_target_gain = self.nominal_capture_gain
        self.phase_snapshots = {}

    def __call__(self, observation):
        if observation.shape != (1, self.observer.num_obs):
            raise ValueError("Observation step teacher requires one support-v2 observation")
        self.observer(observation)
        self.engine.observation = observation
        self.task.steps = observation[:, 361] * (self.duration / self.task.dt)
        seconds = float(self.task.steps[0] * self.task.dt)
        for phase in (2.15, 2.35):
            if phase not in self.phase_snapshots and seconds >= phase:
                self.phase_snapshots[phase] = dict(seconds=seconds,
                    root_position=observation[0, :3].tolist(),
                    root_velocity=observation[0, 9:15].tolist(),
                    left_foot_position=self.observer.data.xpos[
                        self.rig_foot_body].tolist())
        if (self.negative_x_recovery and self.adaptive_root_xy is not None
                and self.capture_adjustment is None and seconds >= 2.15):
            root_xy = observation[0, :2].detach().cpu().numpy()
            error = root_xy - self.adaptive_root_xy
            if error[0] < -.002 and abs(error[1]) < .0015:
                self.negative_x_selected = True
                self.capture_target_gain = 3.
                self.landing.com_kp, self.landing.com_kd = 15., 3.
        if (self.nominal_capture_xy is not None and self.capture_adjustment is None
                and seconds >= 2.15):
            data = self.observer.data
            capture = (data.subtree_com[1, :2]
                       + data.subtree_linvel[1, :2]
                       * np.sqrt(data.subtree_com[1, 2] / 9.81))
            adjustment = np.clip(self.capture_target_gain *
                                 (capture - self.nominal_capture_xy), -.05, .05)
            ramp = np.clip((self.anchor.target_times - 2.15) / .35, 0., 1.)
            self.anchor.targets[:, :2] += ramp[:, None] * adjustment
            self.capture_adjustment = adjustment.tolist()
        if (self.adaptive_root_xy is not None and not self.ankle_offset_selected
                and seconds >= 2.15):
            root_xy = observation[0, :2].detach().cpu().numpy()
            if (self.negative_y_roll_gains is not None
                    and root_xy[1] < self.adaptive_root_xy[1] - .002):
                (self.balance.stabilizer.k_hr,
                 self.balance.stabilizer.d_hr) = self.negative_y_roll_gains
                self.roll_gains_selected = True
            lag = max(self.adaptive_root_xy[0] - root_xy[0],
                      4. * (root_xy[1] - self.adaptive_root_xy[1]))
            blend = np.clip((lag - .003) / .007, 0., 1.)
            self.controller.offsets[3] = self.original_ankle_offset - .265 * float(blend)
            self.ankle_offset_selected = True
            self.ankle_selection = dict(seconds=float(self.task.steps[0] * self.task.dt),
                                        root_xy=root_xy.tolist(), lag_m=float(lag),
                                        blend=float(blend))
        action = self.controller(observation)
        if self.right_arm_clearance and 2.25 <= seconds <= 2.85:
            phase = (seconds - 2.25) / .60
            envelope = np.sin(np.pi * phase) ** 2
            delta = torch.as_tensor(RIGHT_ARM_CLEARANCE, dtype=action.dtype)
            action[0, self.right_arm_actions] = torch.clamp(
                action[0, self.right_arm_actions]
                + self.right_arm_clearance_scale * envelope * delta / .25,
                -1., 1.)
        if self.sole_level_gain and self.landing.landed:
            data = self.observer.data
            normal = data.geom_xmat[self.left_foot_geom].reshape(3, 3)[:, 2]
            derivative = np.cross(data.xaxis[self.left_ankle_joint], normal)
            denominator = float(np.dot(derivative[:2], derivative[:2]))
            if denominator > 1e-8:
                delta = -float(np.dot(normal[:2], derivative[:2])) / denominator
                correction = self.sole_level_gain * np.clip(delta / .25, -2., 2.)
                action[0, self.left_ankle_action] = torch.clamp(
                    action[0, self.left_ankle_action] + correction, -1., 1.)
        return action


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source_path = args.source / "report.json"
    search_path = args.search / "report.json"
    motion_path = args.source / "step_reference.npz"
    samples_path = args.search / "best_teacher_samples.npz"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    search = json.loads(search_path.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (not search["passed"] or search["model_sha256"] != sha256(rig.path)
            or source["model_sha256"] != sha256(rig.path)
            or search["source_report_sha256"] != sha256(source_path)
            or search["source_motion_sha256"] != sha256(motion_path)
            or search["teacher_samples_sha256"] != sha256(samples_path)):
        raise ValueError("Observation controller sources differ from the passing native search")
    with np.load(motion_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    with np.load(samples_path, allow_pickle=False) as arrays:
        observations = torch.as_tensor(arrays["observation"].copy())
        actions = torch.as_tensor(arrays["action"].copy())
        dt = float(arrays["dt"])

    teacher = ObservationStepTeacher(rig, contract, motion, source, search, dt)
    predicted = torch.cat([teacher(row[None]) for row in observations])
    errors = (predicted - actions).abs().numpy()
    parity = dict(max_abs_action_error=float(errors.max()),
                  mean_abs_action_error=float(errors.mean()),
                  samples=len(observations))

    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
    task.episode_seconds = task.reference.duration
    live_teacher = ObservationStepTeacher(rig, contract, motion, source, search, task.dt)
    native, _, _ = evaluate(task, live_teacher, .15, 2.55)
    args.out.mkdir(parents=True)
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  source_report_sha256=sha256(source_path),
                  search_report_sha256=sha256(search_path),
                  observation_contract_sha256=sha256(args.contract),
                  teacher_samples_sha256=sha256(samples_path), parity=parity,
                  native={key: value for key, value in native.items() if key != "trace"},
                  passed=bool(native["passed"] and errors.max() < 1e-4),
                  admitted_for_training=False,
                  scope="Observation-only scripted controller; no physical reference admission, learned actor, ball or Godot parity")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
