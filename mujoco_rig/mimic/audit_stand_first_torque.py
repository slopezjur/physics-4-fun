"""Three-interval native intervention on saved first torque vectors; no learning."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import balance_gate, digest, read, save
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


FRACTIONS = (0., .25, .5, .75, 1.)
INTERVALS = 3


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_endpoint(arrays, recorded):
    checks = {}
    for key, tolerance in dict(qpos=1e-7, qvel=1e-9, foot_loads_n=1e-4,
                               pelvis_height_m=1e-7, pelvis_tilt_degrees=1e-5).items():
        expected = recorded[key][:INTERVALS]
        error = float(np.abs(arrays[key]-expected).max())
        assert error <= tolerance, f'{key} endpoint replay differs: {error}'
        checks[key] = dict(exact=bool(np.array_equal(arrays[key], expected)), max_absolute_error=error,
            absolute_tolerance=tolerance)
    for key in ('balance', 'posture', 'done'):
        assert np.array_equal(arrays[key], recorded[key][:INTERVALS]), f'{key} endpoint replay differs'
        checks[key] = dict(exact=True)
    return checks


def execute(task, teacher, first_command, actor_first, label, fraction, out):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.reset()
    engine, rig = task.engine, task.rig
    reset_qpos = engine.native_qpos().numpy().copy()
    reset_qvel = np.stack([data.qvel.copy() for data in engine.datas])
    assert np.array_equal(reset_qpos, teacher['initial_qpos'])
    assert np.array_equal(reset_qvel, teacher['initial_qvel'])
    commands = teacher['action'][:INTERVALS].copy()
    commands[0] = first_command
    assert np.isfinite(commands).all() and (np.abs(commands) <= 1).all()
    records = []
    for command in commands:
        _, _, done, _ = task.step(torch.from_numpy(command.copy()))
        state = task.standing_measurements()
        balance_flags, posture_flags = task.balance_criteria(state), task.posture_criteria(state)
        row = dict(action=command.copy(), held_torque_nm=engine.native_ctrl[:, rig.actuators].numpy().copy(),
            qpos=engine.native_qpos().numpy().copy(), qvel=np.stack([data.qvel.copy() for data in engine.datas]),
            root_linear_velocity_world_m_s=engine.get_root_vel(0).numpy().copy(),
            root_angular_velocity_world_rad_s=engine.get_root_ang_vel(0).numpy().copy(),
            balance=torch.stack(list(balance_flags.values())).all(0).numpy().copy(),
            posture=torch.stack(list(posture_flags.values())).all(0).numpy().copy(),
            done=done.numpy().copy(), solver_warning=engine.solver_warning.copy())
        row.update({key: value.numpy().copy() for key,value in state.items()})
        row.update({f'balance_gate_{key}': value.numpy().copy() for key,value in balance_flags.items()})
        row.update({f'posture_gate_{key}': value.numpy().copy() for key,value in posture_flags.items()})
        assert all(np.isfinite(value).all() for value in row.values())
        assert not row['solver_warning'].any()
        records.append(row)
    arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
    arrays.update(initial_qpos=reset_qpos, initial_qvel=reset_qvel, time_s=np.arange(1, INTERVALS+1)*task.dt)
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2] = first_command*np.asarray(rig.action_scale, dtype=np.float32)
    assert np.array_equal(arrays['held_torque_nm'], expected), 'Delayed held torque mismatch'
    assert all(np.array_equal(arrays[key][:2], teacher[key][:2]) for key in ('qpos', 'qvel'))
    rows = []
    teacher_first = teacher['action'][0]
    for case in range(8):
        difference = first_command[case]-teacher_first[case]
        actor_delta = actor_first[case]-teacher_first[case] if actor_first is not None else difference
        balanced = bool(arrays['balance'][-1, case])
        posture = bool(arrays['posture'][-1, case])
        null = bool(arrays['done'][-1, case] == DoneFlags.NULL.value)
        rows.append(dict(case=case, balanced=balanced, posture_valid=posture,
            balanced_and_posture_valid=balanced and posture and null,
            terminal_flag=DoneFlags(int(arrays['done'][-1, case])).name,
            root_angular_speed_rad_s=float(np.linalg.norm(arrays['qvel'][-1, case, 3:6])),
            root_linear_velocity_world_m_s=arrays['root_linear_velocity_world_m_s'][-1, case].tolist(),
            root_angular_velocity_world_rad_s=arrays['root_angular_velocity_world_rad_s'][-1, case].tolist(),
            horizontal_speed_m_s=float(arrays['horizontal_speed_m_s'][-1, case]),
            pelvis_height_m=float(arrays['pelvis_height_m'][-1, case]),
            pelvis_tilt_degrees=float(arrays['pelvis_tilt_degrees'][-1, case]),
            foot_loads_n=arrays['foot_loads_n'][-1, case].tolist(),
            upper_body_tilt_degrees=arrays['upper_body_tilt_degrees'][-1, case].tolist(),
            upper_body_clearance_deficit_m=arrays['upper_body_clearance_deficit_m'][-1, case].tolist(),
            non_foot_ground_contact_max_n=float(arrays['non_foot_ground_contacts_n'][-1, case].max()),
            balance_gates={key: bool(arrays[f'balance_gate_{key}'][-1, case]) for key in task.balance_criteria(state)},
            posture_gates={key: bool(arrays[f'posture_gate_{key}'][-1, case]) for key in task.posture_criteria(state)},
            normalized_command_rms_distance_from_teacher=float(np.sqrt(np.mean(difference.astype(np.float64)**2))),
            torque_rms_distance_from_teacher_nm=float(np.sqrt(np.mean((difference*rig.action_scale)**2))),
            motors={name:dict(teacher_command=float(teacher_first[case,index]), executed_command=float(first_command[case,index]),
                actor_command=float(actor_first[case,index]) if actor_first is not None else None,
                actor_minus_teacher=float(actor_delta[index]), held_torque_nm=float(arrays['held_torque_nm'][-1,case,index]),
                teacher_torque_nm=float(teacher_first[case,index]*rig.action_scale[index])) for index,name in enumerate(task.observation_contract()['actuator_names'])}))
    summary = dict(episodes=8, endpoint_balanced_cases=sum(row['balanced'] for row in rows),
        endpoint_balanced_and_posture_valid_cases=sum(row['balanced_and_posture_valid'] for row in rows),
        mean_angular_speed_rad_s=float(np.mean([row['root_angular_speed_rad_s'] for row in rows])),
        angular_speed_min_rad_s=float(min(row['root_angular_speed_rad_s'] for row in rows)),
        angular_speed_max_rad_s=float(max(row['root_angular_speed_rad_s'] for row in rows)),
        failed_balance_gate_cases={key: sum(not row['balance_gates'][key] for row in rows) for key in rows[0]['balance_gates']},
        mean_command_rms_distance_from_teacher=float(np.mean([row['normalized_command_rms_distance_from_teacher'] for row in rows])))
    out.mkdir()
    np.savez_compressed(out/'trace.npz', **arrays)
    report = dict(label=label, fraction=fraction, cases=rows, summary=summary, intervals=3,
        endpoint_seconds=3*task.dt, initial_states_identical=True, zero_torque_prefix_exact=True,
        delayed_torque_mapping_verified=True, solver_warnings=0, training_updates=0)
    save(out/'report.json', report)
    print(dict(label=label, fraction=fraction, **summary), flush=True)
    return report, arrays


def run(args):
    activate(args.mimickit)
    from .torque_stand_task import TorqueStandTask
    torch.set_num_threads(1)
    manifest_path = args.root/'comparison-inputs.json'
    protected = dict(read(manifest_path)['inputs'])
    sources = [Path(__file__), Path('mujoco_rig/mimic/rig.py'), Path('mujoco_rig/mimic/native_engine.py'),
        Path('mujoco_rig/mimic/diagnose_stand_foundations.py'), Path('mujoco_rig/mimic/stand_task.py'),
        Path('mujoco_rig/mimic/torque_stand_task.py'), Path('mujoco_rig/mimic/kinematics.py'),
        Path('mujoco_rig/mimic/contact_contract.py'), Path('mujoco_rig/rl/env_config.py'),
        Path('mujoco_rig/mimic/runtime.py'), args.mimickit/'mimickit/envs/base_env.py']
    for path in sources+[manifest_path, args.root/'comparison.json']:
        protected[str(path)] = digest(path)
    teacher_path = Path('logs/mimickit-training/stand-reward-audit-20261002-01/teacher')
    teacher = load_trace(teacher_path/'trace.npz')
    assert read(teacher_path/'report.json')['standing_passes'] == 8
    actors = {weight: load_trace(args.root/f'audit-weight-{weight}/deterministic-native/trace.npz') for weight in ('0','0.25')}
    run_path = args.root/'weight-0'
    contract = read(run_path/'contract.json')
    rig = Rig.load(run_path/'dummy.xml', policy_exclude=())
    assert len(rig.actuators) == 33 and contract['latency_steps'] == 2
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    assert digest(rig.path) == contract['model_sha256']
    assert np.array_equal(rig.action_scale, np.asarray(contract['torque_scale_nm']))
    for original in actors.values():
        assert all(np.array_equal(original[key], teacher[key]) for key in ('initial_qpos','initial_qvel'))
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Protected input/source mismatch'
    task = TorqueStandTask(rig, run_path/'stationary_reference.npz', 8, 'cpu', seed=301010,
        reward_mode='reference_v9', episode_seconds=12., engine_factory=DiagnosticNativeEngine)
    assert task.dt == contract['control_timestep']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'sources').mkdir()
    snapshots = []
    for path in sources:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        snapshots.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    save(args.out/'protocol.json', dict(inputs=protected, sources=snapshots, contract=contract,
        fractions=list(FRACTIONS), command_settings=9, native_cases_per_setting=8,
        control_intervals=INTERVALS, endpoint_seconds=INTERVALS*task.dt, balance_gate=balance_gate(sources[5]),
        no_neural_network_loaded=True, no_optimizer_created=True, mimickit_revision=MIMICKIT_REVISION,
        scoring='Fixed physical gates; no reward, AMP, critic or learning comparison.',
        training_updates=0, arena_changed=False))
    teacher_report, teacher_arrays = execute(task, teacher, teacher['action'][0], None, 'teacher', 0., args.out/'teacher')
    teacher_report['saved_endpoint_replay'] = verify_endpoint(teacher_arrays, teacher)
    save(args.out/'teacher/report.json', teacher_report)
    results = {'teacher':teacher_report}
    for weight, original in actors.items():
        for fraction in FRACTIONS[1:]:
            first = original['action'][0].copy() if fraction == 1 else teacher['action'][0]+np.float32(fraction)*(original['action'][0]-teacher['action'][0])
            label = f'weight-{weight}-fraction-{fraction:g}'
            report, arrays = execute(task, teacher, first, original['action'][0], label, fraction, args.out/label)
            if fraction == 1:
                report['saved_endpoint_replay'] = verify_endpoint(arrays, original)
                save(args.out/label/'report.json', report)
            results[label] = report
    boundaries = {}
    for weight in actors:
        ordered = [teacher_report]+[results[f'weight-{weight}-fraction-{fraction:g}'] for fraction in FRACTIONS[1:]]
        rows = []
        for case in range(8):
            passed = [entry['cases'][case]['balanced_and_posture_valid'] for entry in ordered]
            failing = [index for index,value in enumerate(passed) if not value]
            first_fail = failing[0] if failing else None
            rows.append(dict(case=case, passing_fractions=[fraction for fraction,valid in zip(FRACTIONS,passed) if valid],
                first_failing_fraction=FRACTIONS[first_fail] if first_fail is not None else None,
                preceding_passing_fraction=FRACTIONS[first_fail-1] if first_fail is not None and first_fail > 0 else None,
                monotonic_on_tested_grid=bool(not any(passed[first_fail+1:])) if first_fail is not None else True))
        boundaries[weight] = rows
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Protected input changed'
    assert all(digest(Path(row['snapshot'])) == row['sha256'] for row in snapshots)
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=results,
        per_pose_tested_boundaries=boundaries, cases_executed=72, physics_control_intervals=216,
        saved_endpoints_verified=True, resets_identical=True, zero_torque_prefixes_exact=True,
        actual_delayed_torques_verified=True, inputs_unchanged=True, source_snapshots_verified=True,
        solver_warnings=0, training_updates=0, neural_networks_loaded=0, optimizers_created=0, arena_changed=False,
        interpretation='Three-interval first-action effect only. Grid brackets do not establish an exact threshold, global monotonicity or closed-loop standing.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
