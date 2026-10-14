"""Bounded native motor/limit traces; no training or Arena selection changes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


LIMIT_MARGIN_RAD = np.deg2rad(2.)
SATURATION = .98


class TracedEngine(NativeDummyEngine):
    """Observe the production solver hooks without changing control or physics."""

    def __init__(self, rig, num_envs, device='cpu', **kwargs):
        super().__init__(rig, num_envs, device, **kwargs)
        self.active = np.ones(num_envs, dtype=bool)
        self.joints = rig.model.actuator_trnid[rig.actuators, 0]
        self.qadr = rig.model.jnt_qposadr[self.joints]
        self.dadr = rig.model.jnt_dofadr[self.joints]
        self.starts = {}
        self.samples = {name: [] for name in (
            'case', 'time_start_s', 'joint_position_start_rad', 'joint_position_end_rad',
            'joint_velocity_start_rad_s', 'delayed_control_nm', 'actuator_force_nm',
            'joint_actuator_torque_nm', 'bias_torque_nm', 'passive_torque_nm',
            'constraint_torque_nm')}

    def _before_physics_step(self, world, data):
        if self.active[world]:
            self.starts[world] = (float(data.time), data.qpos[self.qadr].copy(),
                                  data.qvel[self.dadr].copy(), data.ctrl[self.rig.actuators].copy())

    def _after_physics_step(self, world, data):
        if not self.active[world]:
            return
        time, qpos, qvel, ctrl = self.starts.pop(world)
        row = dict(case=world, time_start_s=time, joint_position_start_rad=qpos,
            joint_position_end_rad=data.qpos[self.qadr].copy(),
            joint_velocity_start_rad_s=qvel, delayed_control_nm=ctrl,
            actuator_force_nm=data.actuator_force[self.rig.actuators].copy(),
            joint_actuator_torque_nm=data.qfrc_actuator[self.dadr].copy(),
            bias_torque_nm=data.qfrc_bias[self.dadr].copy(),
            passive_torque_nm=data.qfrc_passive[self.dadr].copy(),
            constraint_torque_nm=data.qfrc_constraint[self.dadr].copy())
        for name, value in row.items():
            self.samples[name].append(value)


def load_bundle(run):
    bundle = run / 'arena-preview'
    contract = json.loads((bundle / 'contract.json').read_text())
    selection = json.loads((bundle / 'arena-selection.json').read_text())
    paths = dict(checkpoint=run / 'model.pt', onnx=bundle / 'stand.onnx',
                 model=bundle / 'dummy.xml', reference=bundle / 'stand_reference.npz')
    hashes = {name: sha256(path) for name, path in paths.items()}
    expected = dict(checkpoint=selection['checkpoint_sha256'], onnx=selection['onnx_sha256'],
                    model=contract['model_sha256'], reference=contract['reference_sha256'])
    if hashes != expected:
        raise ValueError(f'Export provenance mismatch: {run}')
    if contract['observation_contract'] != 'mimic_stationary_stand_torque_v2':
        raise ValueError('Diagnostic requires the 33-motor stationary torque contract')
    return bundle, hashes


def summarize(rig, arrays, terminal):
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    ranges = rig.model.jnt_range[joints]
    limited = rig.model.jnt_limited[joints].astype(bool)
    q = arrays['joint_position_end_rad']
    lower = (q <= ranges[:, 0] + LIMIT_MARGIN_RAD) & limited
    upper = (q >= ranges[:, 1] - LIMIT_MARGIN_RAD) & limited
    overshoot = np.maximum(np.maximum(ranges[:, 0] - q, q - ranges[:, 1]), 0.) * limited
    commands = arrays['command_action']
    delayed = arrays['delayed_control_nm'] / rig.action_scale
    rows = []
    for case in terminal:
        case_id = case['case']
        physics = arrays['case'] == case_id
        control = arrays['command_case'] == case_id
        motors = []
        for index, actuator in enumerate(rig.actuators):
            mean_abs = lambda name: float(np.abs(arrays[name][physics, index]).mean())
            motors.append(dict(
                motor=rig.model.actuator(int(actuator)).name,
                joint=rig.model.joint(int(joints[index])).name,
                limited=bool(limited[index]),
                joint_range_degrees=np.rad2deg(ranges[index]).tolist() if limited[index] else None,
                final_position_degrees=float(np.rad2deg(q[physics, index][-1])),
                lower_limit_fraction=float(lower[physics, index].mean()),
                upper_limit_fraction=float(upper[physics, index].mean()),
                maximum_limit_overshoot_degrees=float(np.rad2deg(overshoot[physics, index].max())),
                command_saturation_fraction=float((np.abs(commands[control, index]) >= SATURATION).mean()),
                delayed_control_saturation_fraction=float((np.abs(delayed[physics, index]) >= SATURATION).mean()),
                policy_torque_authority_nm=float(rig.action_scale[index]),
                mean_absolute_control_nm=mean_abs('delayed_control_nm'),
                mean_absolute_actuator_force_nm=mean_abs('actuator_force_nm'),
                mean_absolute_joint_actuator_torque_nm=mean_abs('joint_actuator_torque_nm'),
                mean_absolute_bias_torque_nm=mean_abs('bias_torque_nm'),
                mean_absolute_passive_torque_nm=mean_abs('passive_torque_nm'),
                mean_absolute_constraint_torque_nm=mean_abs('constraint_torque_nm'),
                maximum_actuator_force_control_difference_nm=float(np.abs(
                    arrays['actuator_force_nm'][physics, index] - arrays['delayed_control_nm'][physics, index]).max())))
        rows.append(dict(**case, physics_samples=int(physics.sum()), control_samples=int(control.sum()), motors=motors))
    return rows


def measure(run, bundle, hashes, reference, out, seconds):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask

    rig = Rig.load(bundle / 'dummy.xml', policy_exclude=())
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Expected spring-free torque plant without actuator bias')
    if len(set(rig.model.actuator_trnid[rig.actuators, 0])) != 33:
        raise ValueError('Expected one motor per controlled hinge')
    task = TorqueStandTask(rig, reference, engine_factory=TracedEngine,
                           reward_mode='whole_pose_v6')
    task.set_mode(EnvMode.TEST)
    task.episode_seconds = seconds
    task.reset()
    session = ort.InferenceSession(str(bundle / 'stand.onnx'), providers=['CPUExecutionProvider'])
    commands, command_cases, command_times = [], [], []
    active = task.engine.active
    terminal = {}
    for _ in range(int(np.ceil(seconds / task.dt))):
        observation = task.observations().numpy()
        action = session.run(None, {'observation': observation})[0]
        if action.shape != (8, 33) or not np.isfinite(action).all():
            raise ValueError('Expected eight finite 33-motor command vectors')
        if (np.abs(action) > 1.000001).any():
            raise ValueError('Export returned commands outside its normalized torque contract')
        for case in np.flatnonzero(active):
            commands.append(action[case].copy())
            command_cases.append(int(case))
            command_times.append(float(task.engine.datas[case].time))
        _, _, done, _ = task.step(torch.from_numpy(action))
        state = task.standing_measurements()
        ended = active & (done.numpy() != DoneFlags.NULL.value)
        for case in np.flatnonzero(ended):
            terminal[int(case)] = dict(case=int(case), seconds=float(task.engine.datas[case].time),
                termination='physical_failure' if done[case] == DoneFlags.FAIL.value else 'time_budget',
                pelvis_height_m=float(state['pelvis_height_m'][case]),
                upper_body_tilt_degrees=state['upper_body_tilt_degrees'][case].tolist(),
                foot_loads_n=state['foot_loads_n'][case].tolist(),
                reference_error_degrees=dict(zip(task.imitation_body_names,
                    state['whole_body_reference_error_degrees'][case].tolist())))
        active[ended] = False
        if not active.any():
            break
    if active.any():
        raise RuntimeError('Control budget ended before native time termination')
    arrays = {name: np.asarray(values) for name, values in task.engine.samples.items()}
    arrays.update(command_action=np.asarray(commands), command_case=np.asarray(command_cases),
                  command_time_start_s=np.asarray(command_times))
    np.savez_compressed(out / 'trace.npz', **arrays)
    report = dict(schema='torque_stand_motor_diagnostic_v1', run=str(run), hashes=hashes,
        requested_seconds=seconds, physics_timestep=rig.model.opt.timestep, control_timestep=task.dt,
        limit_proximity_degrees=2., saturation_threshold=SATURATION,
        scope='Eight fixed quiet poses; deterministic exported final actor; trace stops per case at physical failure. '
              'No training, engine comparison, naturalness admission or Arena selection.',
        force_sampling='Solver force fields captured after every mj_step and before mj_forward; '
                       'joint positions captured before and after integration. Bias includes gravity and velocity terms; '
                       'constraint torque includes contacts and limits, not isolated joint-limit force.',
        cases=summarize(rig, arrays, [terminal[i] for i in sorted(terminal)]))
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return dict(run=str(run), hashes=hashes, report=str(out / 'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path,
                        default=Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz'))
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2 or len({p.resolve() for p in args.run}) != 2:
        parser.error('Supply exactly two distinct completed runs for a paired comparison')
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error('Diagnostic duration must be in (0, 30] seconds per case')
    if args.out.exists():
        parser.error('Output already exists; preserve earlier diagnostics')
    bundles = [load_bundle(run.resolve()) for run in args.run]
    for name in ('model', 'reference'):
        if bundles[0][1][name] != bundles[1][1][name]:
            parser.error(f'Paired comparison requires identical {name} hashes')
    if sha256(args.reference) != bundles[0][1]['reference']:
        parser.error('Licensed reference with its JSON provenance must match both exports')
    activate(args.mimickit)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    request = dict(schema='torque_stand_motor_diagnostic_request_v1', seconds=args.seconds,
        runs=[str(run.resolve()) for run in args.run], hashes=[hashes for _, hashes in bundles],
        diagnostic_source_sha256=sha256(Path(__file__)), training_started=False, arena_changed=False)
    (args.out / 'request.json').write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
    results = []
    for index, (run, (bundle, hashes)) in enumerate(zip(args.run, bundles)):
        destination = args.out / str(index)
        destination.mkdir()
        results.append(measure(run.resolve(), bundle, hashes, args.reference, destination, args.seconds))
    (args.out / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
