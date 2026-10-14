"""Causal replacement of three saved first-command channels; no learning."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .audit_stand_first_torque import execute, load_trace, verify_endpoint
from .audit_stand_motor_sensitivity import predict_actor
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate


REPLACED_MOTORS = ('Thigh_L_rx', 'Thigh_R_rx', 'Spine_rx')


def run(args):
    activate(args.mimickit)
    from .torque_stand_task import TorqueStandTask
    torch.set_num_threads(1)
    prior, completion = read(args.root/'protocol.json'), read(args.root/'completion-verification.json')
    protected = dict(prior['inputs'])
    protected.update(completion['output_hashes'])
    for source in prior['sources']:
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Historical input/source changed'
    sources = [Path(__file__)]+[Path(source['path']) for source in prior['sources']]
    for path in sources+[args.root/'completion-verification.json']:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    teacher = load_trace(args.root/'teacher/trace.npz')
    actor_root = Path('logs/mimickit-training/stand-first-torque-20261002-01')
    actors = {weight:load_trace(actor_root/f'weight-{weight}-fraction-1/trace.npz') for weight in ('0','0.25')}
    with np.load(args.root/'jacobians.npz', allow_pickle=False) as archive:
        jacobians = {quantity:archive[f'{quantity}_jacobian_per_normalized_command'] for quantity in ('angular','linear')}
    contract = prior['contract']
    run_path = Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0')
    rig = Rig.load(run_path/'dummy.xml', policy_exclude=())
    assert digest(rig.path) == contract['model_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    assert len(rig.actuators) == 33 and contract['latency_steps'] == 2
    assert np.array_equal(rig.action_scale, np.asarray(contract['torque_scale_nm']))
    indices = [contract['actuator_names'].index(name) for name in REPLACED_MOTORS]
    remaining = [index for index in range(33) if index not in indices]
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
        replaced_motors=list(REPLACED_MOTORS), settings=4, cases_per_setting=8, control_intervals_per_case=3,
        endpoint_seconds=3*task.dt, unchanged_actor_coordinates=30,
        intervention='Only first command channels replaced; intervals 2 and 3 contain saved teacher commands and do not reach the motors before the probe ends.',
        scoring='Fixed balance/posture gates and world velocity; local prediction is a separate diagnostic.',
        training_updates=0, neural_networks_loaded=0, optimizers_created=0, arena_changed=False))
    results = {}
    for weight, original in actors.items():
        for corrected in (False, True):
            arm = 'three-teacher' if corrected else 'control'
            label = f'actor-{weight}-{arm}'
            first = original['action'][0].copy()
            if corrected:
                first[:, indices] = teacher['action'][0, :, indices].T
            assert np.array_equal(first[:, remaining], original['action'][0, :, remaining].T)
            if corrected:
                assert np.array_equal(first[:, indices], teacher['action'][0, :, indices].T)
            report, arrays = execute(task, teacher, first, original['action'][0], label, None, args.out/label)
            if corrected:
                assert np.array_equal(arrays['held_torque_nm'][-1, :, indices].T, teacher['held_torque_nm'][-1, :, indices].T)
            else:
                checks = verify_endpoint(arrays, original)
                assert all(row['exact'] for row in checks.values()), 'Control replay must be exact'
                assert all(np.array_equal(arrays[key], original[key]) for key in
                    ('root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s','held_torque_nm'))
                report['saved_control_endpoint_replay'] = checks
            prediction, predicted_arrays = predict_actor(teacher, arrays, jacobians, contract['actuator_names'])
            report.update(weight=weight, arm=arm, replaced_motors=list(REPLACED_MOTORS) if corrected else [],
                remaining_coordinates_identical=True, local_prediction=prediction)
            save(args.out/label/'report.json', report)
            np.savez_compressed(args.out/label/'prediction.npz', **predicted_arrays)
            results[label] = report
            save(args.out/'progress.json', dict(settings_completed=len(results), total_settings=4))
    comparisons = {}
    for weight in actors:
        control, corrected = results[f'actor-{weight}-control'], results[f'actor-{weight}-three-teacher']
        comparisons[weight] = dict(control=control['summary'], three_teacher=corrected['summary'],
            corrected_prediction=corrected['local_prediction']['metrics'],
            cases=[dict(case=index, original_angular_speed_rad_s=before['root_angular_speed_rad_s'],
                corrected_angular_speed_rad_s=after['root_angular_speed_rad_s'],
                original_balance_and_posture=before['balanced_and_posture_valid'],
                corrected_balance_and_posture=after['balanced_and_posture_valid']) for index,(before,after) in enumerate(zip(control['cases'],corrected['cases']))])
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Protected input changed'
    assert all(digest(Path(source[key])) == source['sha256'] for source in snapshots for key in ('path','snapshot'))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=results,
        comparisons=comparisons, cases_executed=32, physics_control_intervals=96,
        controls_reproduced_exactly=True, resets_identical=True, zero_torque_prefixes_exact=True,
        actual_delayed_torques_verified=True, unchanged_thirty_actor_coordinates_verified=True,
        inputs_unchanged=True, source_snapshots_verified=True, solver_warnings=0, training_updates=0,
        neural_networks_loaded=0, optimizers_created=0, arena_changed=False,
        interpretation='Only the first delivered torque is intervened on. Endpoint balance does not establish later closed-loop stability or learned Stand.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/mimickit-training/stand-motor-sensitivity-20261002-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
