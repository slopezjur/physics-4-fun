"""Native diagnostic of a supported transfer into the right-leg hold pose."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .force_retarget import reference_velocities
from .foundation_audit import settle_passive_joints
from .native_engine import NativeDummyEngine
from .probe_single_support import CoronalSagittalStabilizer, support_body_names
from .reference import StandReference
from .rig import Rig
from .runtime import activate
from .static_support import solve_support
from .track_transfer import CalibrationReference


SCHEMA = "mimic_single_support_entry_probe_v2"
PREHOLD_SECONDS = .5
TARGET_SOURCE_SECONDS = 9.
POSTHOLD_SECONDS = 3.
FEEDFORWARD_INTERVAL_SECONDS = .25
CONTROLLERS = {
    "original": {},
    "balanced_axes": {"k_ankle_pitch": 5., "d_ankle_pitch": 1.,
                      "d_hip_roll": 2.},
}


class EntryReference:
    def __init__(self, poses, rig, dt):
        self.dt = dt
        self.qpos = torch.as_tensor(poses, dtype=torch.float32)
        self.qvel = torch.as_tensor(reference_velocities(rig.model, poses, dt),
                                    dtype=torch.float32)
        self.duration = (len(poses) - 1) * dt

    sample = StandReference.sample


class ScheduledStabilizer:
    def __init__(self, task, times, torques, com, controller_parameters=None):
        self.task = task
        self.times = np.asarray(times)
        self.torques = np.asarray(torques)
        self.com = np.asarray(com)
        self.stabilizer = CoronalSagittalStabilizer(
            task, torques[0], com[0], **(controller_parameters or {}))

    def __call__(self, observation):
        time = float(self.task.steps[0] * self.task.dt)
        torque = [np.interp(time, self.times, self.torques[:, j])
                  for j in range(self.torques.shape[1])]
        target_com = [np.interp(time, self.times, self.com[:, j]) for j in range(3)]
        self.stabilizer.set_target(torque, target_com)
        return self.stabilizer(observation)


def build_entry(rig, arrays, start_seconds):
    dt = float(arrays["dt"])
    if not 0 < start_seconds < TARGET_SOURCE_SECONDS or abs(start_seconds / dt - round(start_seconds / dt)) > 1e-6:
        raise ValueError("Require a source-aligned start before 9 seconds")
    source = arrays["qpos"]
    first, last = round(start_seconds / dt), round(TARGET_SOURCE_SECONDS / dt)
    if last >= len(source):
        raise ValueError("Source motion ends before the requested pose")
    settled = []
    for frame in range(first, last + 1):
        pose, passive = settle_passive_joints(rig, source[frame])
        if not passive["passed"]:
            raise ValueError(f"Passive joint equilibrium fails at source frame {frame}")
        settled.append(pose)
    settled = np.asarray(settled)
    pre = round(PREHOLD_SECONDS / dt)
    post = round(POSTHOLD_SECONDS / dt)
    poses = np.concatenate((np.repeat(settled[:1], pre, axis=0), settled,
                            np.repeat(settled[-1:], post, axis=0)))
    samples, times, torque, com = [], [], [], []
    interval = round(FEEDFORWARD_INTERVAL_SECONDS / dt)
    for frame in sorted(set(range(0, len(poses), interval)) | {len(poses) - 1}):
        source_frame = min(max(frame - pre, 0), len(settled) - 1) + first
        names = support_body_names(arrays["contact"][source_frame])
        try:
            support = solve_support(rig, poses[frame], passive_tolerance=.05,
                                    minimum_norm=True, support_names=names)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Support solve fails at {frame * dt:.2f}s: {error}") from error
        if not support["feasible"]:
            raise ValueError(f"Support infeasible at {frame * dt:.2f}s")
        samples.append(dict(seconds=frame * dt, source_seconds=source_frame * dt,
                            max_torque_fraction=support["max_torque_budget_fraction"],
                            contact=arrays["contact"][source_frame].tolist()))
        times.append(frame * dt)
        torque.append(support["normalized_motor_torque"])
        com.append(support["com_world"])
    return EntryReference(poses, rig, dt), samples, np.asarray(times), np.asarray(torque), np.asarray(com)


def evaluate(task, controller, terminal_start):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    records = []
    left_unloaded_since = None
    settled_since = None
    window_start = None
    longest_window = (0., None, None)
    passed = False
    for _ in range(round(task.episode_seconds / task.dt)):
        with torch.no_grad():
            action = controller(task.observations())
        _, _, flags, _ = task.step(action)
        time = float(task.steps[0] * task.dt)
        q = task.engine.native_qpos()[0]
        loads = task.engine.get_ground_contact_forces(0)[0, task.allowed_ground, 2].abs()
        left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
        upright = bool(q[2] >= .75 and 1 - 2 * (q[4].square() + q[5].square()) >= .9659)
        slow = bool(torch.linalg.vector_norm(task.engine.get_root_vel(0)[0, :2]) <= .2 and
                    torch.linalg.vector_norm(task.engine.get_root_ang_vel(0)[0]) <= 1.)
        supported = left < 5. and right >= 5.
        if (time >= PREHOLD_SECONDS and supported and upright
                and flags[0] != DoneFlags.FAIL.value):
            window_start = time if window_start is None else window_start
            duration = time - window_start + task.dt
            if duration > longest_window[0]:
                longest_window = (duration, window_start, time)
        else:
            window_start = None
        if time >= terminal_start:
            left_unloaded_since = (time if left_unloaded_since is None else left_unloaded_since) if supported else None
            settled_since = (time if settled_since is None else settled_since) if supported and upright and slow else None
        if len(records) == 0 or time - records[-1]["seconds"] >= .25 or flags[0] != DoneFlags.NULL.value:
            records.append(dict(seconds=time, pelvis_height_m=float(q[2]),
                                left_load_n=left, right_load_n=right,
                                upright=upright, slow=slow,
                                max_action_fraction=float(action.abs().max())))
        if flags[0] != DoneFlags.NULL.value:
            passed = bool(flags[0] == DoneFlags.TIME.value and settled_since is not None
                          and time - settled_since + task.dt >= .5 and left_unloaded_since is not None
                          and time - left_unloaded_since >= POSTHOLD_SECONDS - task.dt)
            break
    return dict(passed=passed, survival_seconds=time,
                reached_terminal_pose=bool(time >= terminal_start),
                longest_upright_right_support_seconds=longest_window[0],
                longest_upright_right_support_window_seconds=list(longest_window[1:]),
                final_settled_seconds=0. if settled_since is None else time - settled_since + task.dt,
                final_unloaded_seconds=0. if left_unloaded_since is None else time - left_unloaded_since + task.dt,
                trace=records)


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = CalibrationReference(args.reference, rig)
    with np.load(source.path, allow_pickle=False) as arrays:
        reference, samples, times, torques, com = build_entry(rig, arrays, args.start_seconds)
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    controller = ScheduledStabilizer(task, times, torques, com,
                                    CONTROLLERS[args.controller])
    terminal_start = PREHOLD_SECONDS + TARGET_SOURCE_SECONDS - args.start_seconds
    result = evaluate(task, controller, terminal_start)
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  source_reference_sha256=source.sha256,
                  start_source_seconds=args.start_seconds,
                  target_source_seconds=TARGET_SOURCE_SECONDS,
                  controller=args.controller,
                  controller_parameters=CONTROLLERS[args.controller],
                  terminal_start_seconds=terminal_start,
                  episode_seconds=reference.duration,
                  feedforward_samples=samples, evaluation=result,
                  scope="Native oracle-COM transfer diagnostic; no training, export or viewer selection")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(start=args.start_seconds, **{k: v for k, v in result.items() if k != "trace"})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=7.)
    parser.add_argument("--controller", choices=tuple(CONTROLLERS), default="original")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
