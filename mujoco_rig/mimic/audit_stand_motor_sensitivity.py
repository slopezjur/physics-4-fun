"""Local first-torque velocity derivatives; recorded actors, no optimizer updates."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .audit_stand_first_torque import execute, load_trace, verify_endpoint
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate


PERTURBATION = .001
VECTOR_FIELDS = dict(angular='root_angular_velocity_world_rad_s', linear='root_linear_velocity_world_m_s')


def derivatives(teacher, perturbations, names, scale):
    jacobians, midpoints, spans = {}, {}, {}
    for quantity, field in VECTOR_FIELDS.items():
        base = teacher[field][-1].astype(np.float64)
        columns, errors, differences = [], [], []
        for index, name in enumerate(names):
            plus, minus = perturbations[name]['plus'], perturbations[name]['minus']
            positive, negative = plus[field][-1].astype(np.float64), minus[field][-1].astype(np.float64)
            gap = plus['action'][0, :, index].astype(np.float64)-minus['action'][0, :, index].astype(np.float64)
            assert (gap > 0).all()
            columns.append((positive-negative)/gap[:, None])
            errors.append((positive+negative)*.5-base)
            differences.append((positive-negative)*.5)
        jacobians[quantity] = np.stack(columns, axis=-1)
        midpoints[quantity] = np.stack(errors, axis=-1)
        spans[quantity] = np.stack(differences, axis=-1)
    rows = []
    for index, name in enumerate(names):
        rows.append(dict(name=name, torque_scale_nm=float(scale[index]),
            angular_gain_mean_rad_s_per_normalized_command=float(np.linalg.norm(jacobians['angular'][:, :, index], axis=-1).mean()),
            angular_gain_mean_rad_s_per_nm=float(np.linalg.norm(jacobians['angular'][:, :, index]/scale[index], axis=-1).mean()),
            linear_gain_mean_m_s_per_normalized_command=float(np.linalg.norm(jacobians['linear'][:, :, index], axis=-1).mean()),
            angular_midpoint_error_max_rad_s=float(np.linalg.norm(midpoints['angular'][:, :, index], axis=-1).max()),
            linear_midpoint_error_max_m_s=float(np.linalg.norm(midpoints['linear'][:, :, index], axis=-1).max()),
            angular_midpoint_to_half_span_max_ratio=float((np.linalg.norm(midpoints['angular'][:, :, index], axis=-1)/np.maximum(np.linalg.norm(spans['angular'][:, :, index], axis=-1), 1e-12)).max())))
    return jacobians, midpoints, rows


def predict_actor(teacher, actor, jacobians, names):
    command_delta = actor['action'][0].astype(np.float64)-teacher['action'][0].astype(np.float64)
    metrics, arrays, contribution_rows = {}, {}, []
    for quantity, field in VECTOR_FIELDS.items():
        base, actual = teacher[field][-1].astype(np.float64), actor[field][-1].astype(np.float64)
        contributions = jacobians[quantity]*command_delta[:, None, :]
        prediction = base+contributions.sum(-1)
        residual = prediction-actual
        observed_delta = actual-base
        direction = observed_delta/np.maximum(np.linalg.norm(observed_delta, axis=-1, keepdims=True), 1e-12)
        projection = np.einsum('eij,ei->ej', contributions, direction)
        arrays.update({f'{quantity}_{key}': value for key,value in dict(actual=actual, predicted=prediction,
            residual=residual, contribution_by_motor=contributions, projection_on_observed_delta=projection).items()})
        metrics[quantity] = dict(mean_actual_magnitude=float(np.linalg.norm(actual, axis=-1).mean()),
            mean_predicted_magnitude=float(np.linalg.norm(prediction, axis=-1).mean()),
            mean_vector_prediction_error=float(np.linalg.norm(residual, axis=-1).mean()),
            maximum_vector_prediction_error=float(np.linalg.norm(residual, axis=-1).max()),
            mean_relative_vector_error_vs_observed_delta=float((np.linalg.norm(residual, axis=-1)/np.maximum(np.linalg.norm(observed_delta, axis=-1), 1e-12)).mean()))
    for index, name in enumerate(names):
        contribution_rows.append(dict(name=name, mean_actor_minus_teacher=float(command_delta[:, index].mean()),
            angular_contribution_norm_mean_rad_s=float(np.linalg.norm(arrays['angular_contribution_by_motor'][:, :, index], axis=-1).mean()),
            angular_projection_on_observed_delta_mean_rad_s=float(arrays['angular_projection_on_observed_delta'][:, index].mean()),
            linear_contribution_norm_mean_m_s=float(np.linalg.norm(arrays['linear_contribution_by_motor'][:, :, index], axis=-1).mean())))
    cases = [dict(case=case, angular_actual_rad_s=arrays['angular_actual'][case].tolist(),
        angular_predicted_rad_s=arrays['angular_predicted'][case].tolist(),
        angular_prediction_error_rad_s=float(np.linalg.norm(arrays['angular_residual'][case])),
        linear_actual_m_s=arrays['linear_actual'][case].tolist(), linear_predicted_m_s=arrays['linear_predicted'][case].tolist(),
        linear_prediction_error_m_s=float(np.linalg.norm(arrays['linear_residual'][case]))) for case in range(8)]
    return dict(cases=cases, metrics=metrics,
        motor_contributions=sorted(contribution_rows, key=lambda row: row['angular_contribution_norm_mean_rad_s'], reverse=True)), arrays


def run(args):
    activate(args.mimickit)
    from .torque_stand_task import TorqueStandTask
    torch.set_num_threads(1)
    prior = read(args.root/'protocol.json')
    completion = read(args.root/'completion-verification.json')
    protected = dict(prior['inputs'])
    protected.update(completion['output_hashes'])
    for source in prior['sources']:
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Historical input/source changed'
    sources = [Path(__file__), Path('mujoco_rig/mimic/audit_stand_failures.py')]+[Path(source['path']) for source in prior['sources']]
    for path in sources+[args.root/'completion-verification.json']:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    teacher = load_trace(args.root/'teacher/trace.npz')
    teacher['initial_qpos'] = load_trace(Path('logs/mimickit-training/stand-reward-audit-20261002-01/teacher/trace.npz'))['initial_qpos']
    actors = {weight:load_trace(args.root/f'weight-{weight}-fraction-1/trace.npz') for weight in ('0','0.25')}
    run_path = Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0')
    contract = prior['contract']
    rig = Rig.load(run_path/'dummy.xml', policy_exclude=())
    assert digest(rig.path) == contract['model_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    assert len(rig.actuators) == 33 and contract['latency_steps'] == 2
    assert np.array_equal(rig.action_scale, np.asarray(contract['torque_scale_nm']))
    names = contract['actuator_names']
    task = TorqueStandTask(rig, run_path/'stationary_reference.npz', 8, 'cpu', seed=301010,
        reward_mode='reference_v9', episode_seconds=12., engine_factory=DiagnosticNativeEngine)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'sources').mkdir()
    snapshots = []
    for path in sources:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        snapshots.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    save(args.out/'protocol.json', dict(inputs=protected, sources=snapshots, contract=contract,
        perturbation_normalized=PERTURBATION, settings=67, cases_per_setting=8, control_intervals_per_case=3,
        derivative_denominator='Actual plus-minus float32 command gap per pose; per-Nm derivative divides by original torque scale.',
        prediction='Teacher endpoint world velocity + sum_j(local centered derivative_j * recorded actor-minus-teacher command_j).',
        scoring='Fixed physical gates, local velocity response and prediction error; no AMP/critic scores.',
        training_updates=0, neural_networks_loaded=0, optimizers_created=0, arena_changed=False))
    reports, perturbations = {}, {}

    def probe(first, label, motor=None, sign=None):
        report, arrays = execute(task, teacher, first, None, label, None, args.out/label)
        for case in report['cases']:
            for row in case['motors'].values():
                row['command_minus_teacher'] = row.pop('actor_minus_teacher')
                row.pop('actor_command')
        report.update(perturbed_motor=motor, perturbation_sign=sign, perturbation_normalized=PERTURBATION if motor else 0.)
        reports[label] = report
        save(args.out/label/'report.json', report)
        save(args.out/'progress.json', dict(settings_completed=len(reports), total_settings=67))
        return report, arrays

    teacher_report, teacher_arrays = probe(teacher['action'][0].copy(), 'teacher')
    teacher_report['saved_endpoint_replay'] = verify_endpoint(teacher_arrays, teacher)
    save(args.out/'teacher/report.json', teacher_report)
    for index, name in enumerate(names):
        pair = {}
        for sign, direction in (('minus', -1), ('plus', 1)):
            command = teacher['action'][0].copy()
            command[:, index] += np.float32(direction*PERTURBATION)
            label = f'motor-{index:02d}-{name}-{sign}'
            _, pair[sign] = probe(command, label, name, sign)
            assert all(np.array_equal(command[:, other], teacher['action'][0, :, other]) for other in range(33) if other != index)
        perturbations[name] = pair
    jacobians, midpoints, motor_rows = derivatives(teacher_arrays, perturbations, names, rig.action_scale)
    matrix_arrays = dict(teacher_command=teacher['action'][0], torque_scale_nm=rig.action_scale)
    for quantity in jacobians:
        matrix_arrays[f'{quantity}_jacobian_per_normalized_command'] = jacobians[quantity]
        matrix_arrays[f'{quantity}_jacobian_per_nm'] = jacobians[quantity]/rig.action_scale[None,None,:]
        matrix_arrays[f'{quantity}_midpoint_residual'] = midpoints[quantity]
    assert all(np.isfinite(value).all() for value in matrix_arrays.values())
    np.savez_compressed(args.out/'jacobians.npz', **matrix_arrays)
    predictions = {}
    for weight, actor in actors.items():
        predictions[weight], arrays = predict_actor(teacher_arrays, actor, jacobians, names)
        np.savez_compressed(args.out/f'actor-{weight}-prediction.npz', **arrays)
    assert all(digest(Path(path)) == expected for path,expected in protected.items()), 'Protected input changed'
    assert all(digest(Path(source[key])) == source['sha256'] for source in snapshots for key in ('path','snapshot'))
    all_cases = [case for report in reports.values() for case in report['cases']]
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=reports,
        cases_executed=536, physics_control_intervals=1608,
        endpoint_balanced_and_posture_valid_cases=sum(case['balanced_and_posture_valid'] for case in all_cases),
        motor_sensitivities=sorted(motor_rows, key=lambda row:row['angular_gain_mean_rad_s_per_normalized_command'], reverse=True),
        actor_predictions=predictions, teacher_endpoint_verified=True, resets_identical=True,
        zero_torque_prefixes_exact=True, actual_delayed_torques_verified=True, source_snapshots_verified=True,
        inputs_unchanged=True, solver_warnings=0, training_updates=0, neural_networks_loaded=0,
        optimizers_created=0, arena_changed=False,
        interpretation='Local one-channel interventions around the teacher at eight resets. Linear attribution of large actor errors is an approximation checked against saved actor endpoint vectors, not a tested torque fix or exploration schedule.'))
    print(dict(completed_settings=len(reports), endpoint_balanced_cases=sum(case['balanced_and_posture_valid'] for case in all_cases),
        predictions={weight:result['metrics'] for weight,result in predictions.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/mimickit-training/stand-first-torque-20261002-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
