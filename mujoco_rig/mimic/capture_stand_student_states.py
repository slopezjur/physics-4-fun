"""Frozen autonomous trajectories with unexecuted private teacher queries."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.distance import cdist
import torch

from .admit_stand_controller import gpu_native_state
from .audit_stand_failures import digest, longest, read, save
from .audit_stand_student_errors import load_actor, normalizers
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash


PRIOR = Path('logs/mimickit-training/stand-state-coverage-20261002-01')
TEACHER = Path('logs/mimickit-training/stand-teacher-20261001-01')
CORRECTIONS = Path('logs/mimickit-training/stand-corrections-20261001-01')
MIXTURES = Path('logs/mimickit-training/stand-whole-body-blend-20261002-01')
ADDED_GROUPS = ('actor-0-fraction-0.01', 'actor-0.25-fraction-0.01')
BAD_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC)


def archive_inputs(out):
    prior, complete = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = dict(prior['inputs'])
    protected.update(complete['output_hashes'])
    protected[str(PRIOR/'final-verification.json')] = digest(PRIOR/'final-verification.json')
    paths = [Path(__file__), Path('mujoco_rig/mimic/admit_stand_controller.py'),
             Path('mujoco_rig/mimic/newton_adapter.py')]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    assert all(digest(Path(path)) == expected for path, expected in protected.items())
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources, prior['contract']


def cpu(value):
    return value.detach().cpu().numpy().copy()


def pending_commands(queue):
    return torch.cat([queue.pending[(queue.cursor+i) % len(queue.pending)]
                      for i in range(len(queue.pending))], dim=-1)


def gates(task):
    state = task.standing_measurements()
    balance, posture = task.balance_criteria(state), task.posture_criteria(state)
    return (dict(balance=torch.stack(list(balance.values())).all(0),
                 posture=torch.stack(list(posture.values())).all(0)),
            {**state, **{f'balance_gate_{key}': value for key, value in balance.items()},
             **{f'posture_gate_{key}': value for key, value in posture.items()}})


def check_query_nonmutation(task, reader, qpos, qvel, pending, cursor, control, observation):
    q, v = reader()
    assert np.array_equal(q, qpos) and np.array_equal(v, qvel), 'Teacher changed live state'
    assert task.engine.policy_control.cursor == cursor
    assert torch.equal(pending_commands(task.engine.policy_control), pending)
    assert torch.equal(task.engine.native_ctrl, control)
    assert torch.equal(task.observations(), observation), 'Teacher changed policy inputs'


@torch.no_grad()
def capture(rig, reference, actor, backend, historical, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9', episode_seconds=12., **options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    actor.to(device).eval()
    active = np.ones(8, dtype=bool)
    controller = PredictiveStandController(task)
    reader = controller._native_state if backend == 'native' else lambda: gpu_native_state(task)
    initial_q, initial_v = reader()
    rows, private_reports, query_errors = [], [], []
    started = time.perf_counter()
    for step in range(int(np.ceil(12./task.dt))):
        if backend == 'native':
            task.engine.active = active.copy()
        observation = task.observations()
        qpos, qvel = reader()
        pre_flags, pre_state = gates(task)
        queue = task.engine.policy_control
        pending, cursor, control = pending_commands(queue).clone(), queue.cursor, task.engine.native_ctrl.clone()
        assert torch.equal(observation[:, 95:161], pending)
        action = actor(observation)
        action[torch.tensor(~active, device=device)] = 0.
        teacher_command = np.zeros((8, 33), dtype=np.float32)
        succeeded, private_warning, private_finite = (np.zeros(8, dtype=bool) for _ in range(3))
        # Query each live world independently; failed labels must not terminate the actor's trial.
        controller.state_reader = lambda: (qpos, qvel)
        for case in np.flatnonzero(active):
            mask = np.zeros(8, dtype=bool)
            mask[case] = True
            begin = len(controller.reports)
            try:
                teacher_command[case] = controller.command(mask).numpy()[case]
                succeeded[case] = True
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                query_errors.append(dict(frame=step, case=int(case), error=str(error)))
            data = controller.data[case]
            private_warning[case] = any(data.warning[warning].number for warning in BAD_WARNINGS)
            private_finite[case] = all(np.isfinite(value).all() for value in
                (teacher_command[case], data.qpos, data.qvel, data.qacc))
            for diagnostic in controller.reports[begin:]:
                values = {key: (float(value) if isinstance(value, (float, np.floating)) and np.isfinite(value)
                               else value if isinstance(value, (int, bool)) else str(value))
                          for key, value in diagnostic.items()}
                private_reports.append(dict(frame=step, **values))
        check_query_nonmutation(task, reader, qpos, qvel, pending, cursor, control, observation)
        standardized = (observation-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
        _, _, done, _ = task.step(action)
        held = task.engine.native_ctrl[:, torch.as_tensor(rig.actuators, device=device)]
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings, device=device)] = DoneFlags.FAIL.value
        flags, state = gates(task)
        after_q, after_v = reader()
        finite_label = succeeded & private_finite & ~private_warning & cpu(pre_state['finite_state'])
        row = dict(alive_before=active.copy(), obs=cpu(observation), qpos_before=qpos.copy(), qvel_before=qvel.copy(),
            qpos=after_q.copy(), qvel=after_v.copy(), command=cpu(action), actor_only_command=cpu(action),
            teacher_only_command=teacher_command, pending_commands_oldest_first=cpu(pending),
            held_torque_nm=cpu(held), done=cpu(done), balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            pre_balance=cpu(pre_flags['balance']), pre_posture=cpu(pre_flags['posture']),
            teacher_query_succeeded=succeeded, teacher_private_solver_warning=private_warning,
            teacher_private_state_finite=private_finite, diagnostic_label_numerically_valid=finite_label & active,
            solver_warning=warnings, plant_solver_warning_known=np.full(8, backend == 'native', dtype=bool),
            clipped_features=cpu(standardized.abs()>actor.obs_norm._clip), standardized_obs=cpu(standardized),
            obs_clip_fraction=cpu((standardized.abs()>actor.obs_norm._clip).float().mean(-1)),
            outside_clip10_fraction=cpu((standardized.abs()>10.).float().mean(-1)))
        row.update({key: cpu(value) for key, value in state.items()})
        row.update({f'pre_{key}': cpu(value) for key, value in pre_state.items()})
        rows.append(row)
        active &= cpu(done) == DoneFlags.NULL.value
        if (step+1) % 30 == 0:
            print(dict(setting=out.name, seconds=(step+1)*task.dt, live_cases=int(active.sum()),
                       wall_seconds=time.perf_counter()-started), flush=True)
        if not active.any():
            break
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays.update(time_s=(np.arange(len(rows))+1)*task.dt,
                  initial_qpos=initial_q.copy(), initial_qvel=initial_v.copy())
    live = arrays['alive_before']
    assert np.array_equal(arrays['command'], arrays['actor_only_command'])
    assert np.array_equal(arrays['obs'][..., 95:161], arrays['pending_commands_oldest_first'])
    expected_held = np.zeros_like(arrays['held_torque_nm'])
    expected_held[2:] = arrays['command'][:-2]*np.asarray(rig.action_scale, dtype=np.float32)
    assert np.array_equal(arrays['held_torque_nm'], expected_held)
    assert not arrays['teacher_query_succeeded'][~live].any()
    assert np.isfinite(arrays['obs'][live]).all() and np.isfinite(arrays['command']).all()
    comparable = ('alive_before', 'done', 'balance', 'posture', 'pelvis_height_m', 'pelvis_tilt_degrees',
                  'foot_loads_n', 'command', 'obs_clip_fraction', 'outside_clip10_fraction', 'solver_warning', 'time_s')
    with np.load(historical/'trace.npz', allow_pickle=False) as old:
        replay = {}
        for key in comparable:
            matching_shape = arrays[key].shape == old[key].shape
            replay[key] = dict(exact=bool(np.array_equal(arrays[key], old[key])), same_shape=matching_shape,
                max_absolute_difference=float(np.max(np.abs(arrays[key].astype(float)-old[key].astype(float))))
                if matching_shape else None)
        if backend == 'native':
            assert all(value['exact'] for value in replay.values()), 'Native actor trajectory differs from Phase 32'
    out.mkdir()
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', private_reports)
    save(out/'teacher-query-errors.json', query_errors)
    report = dict(backend=backend, cases=8, requested_horizon_seconds=12.,
        live_control_intervals=int(live.sum()), plant_live_physics_steps=int(live.sum())*rig.decimation,
        teacher_private_prediction_physics_steps=int(live.sum())*2*rig.decimation,
        teacher_queries=int(live.sum()), teacher_query_failures=len(query_errors),
        numerically_valid_diagnostic_labels=int(arrays['diagnostic_label_numerically_valid'].sum()),
        teacher_private_warning_labels=int((arrays['teacher_private_solver_warning'] & live).sum()),
        plant_warning_measurement='Native BADQPOS/BADQVEL/BADQACC' if backend == 'native' else 'Unavailable',
        prior_replay=replay, entire_prior_trace_exact=all(value['exact'] for value in replay.values()),
        teacher_changed_state_queue_or_inputs=False, executed_only_actor=True, delayed_torques_exact=True,
        labels_admitted_as_recovery_data=False, new_amp_clips=0, training_updates=0,
        wall_seconds=time.perf_counter()-started)
    save(out/'capture-report.json', report)
    actor.cpu()
    return arrays, report


def reference_pools():
    with np.load(TEACHER/'native/dataset.npz', allow_pickle=False) as data:
        teacher = data['observation'].copy()
    with np.load(CORRECTIONS/'native/dataset.npz', allow_pickle=False) as data:
        corrections, cases = data['observation'].copy(), data['case'].copy()
    mixtures = []
    for group in ADDED_GROUPS:
        with np.load(MIXTURES/group/'trace.npz', allow_pickle=False) as data:
            mixtures.append(data['obs'].copy())
    assert teacher.shape == (3600, 8, 306) and len(corrections) == 18720
    assert ((cases >= 0) & (cases < 8)).all()
    return [(np.concatenate((teacher[:, case], corrections[cases == case])),
             np.concatenate([obs[:, case] for obs in mixtures])) for case in range(8)]


def feature_name(index, contract):
    group = next(item for item in contract['observation_layout']
                 if item['offset'] <= index < item['offset']+item['width'])
    return f"{group['name']}[{index-group['offset']}]"


def ranked(values, names, count=8):
    return [dict(index=int(index), name=names[index], value=float(values[index]))
            for index in np.argsort(values)[::-1][:count] if values[index] > 0]


def error_summary(error, clip, mask, scale, names):
    if not mask.any():
        return dict(samples=0)
    e, c = error[mask].astype(np.float64), clip[mask]
    return dict(samples=int(mask.sum()), normalized_command_rmse=float(np.sqrt(np.mean(e**2))),
        torque_rmse_nm=float(np.sqrt(np.mean((e*scale)**2))), max_absolute_command_error=float(np.abs(e).max()),
        clipped_feature_fraction=float(c.mean()), frames_with_clipping_fraction=float(c.any(-1).mean()),
        per_motor_rmse=np.sqrt(np.mean(e**2, axis=0)).tolist(),
        top_motor_squared_error=ranked(np.mean(e**2, axis=0), names),
        top_motor_squared_torque_error_nm2=ranked(np.mean((e*scale)**2, axis=0), names))


def analyse(arrays, actor, pools, contract, out):
    dt, scale = contract['control_timestep'], np.asarray(contract['torque_scale_nm'])
    mean, std = cpu(actor.obs_norm.get_mean()).astype(float), cpu(actor.obs_norm.get_std()).astype(float)
    names = contract['actuator_names']
    feature_names = [feature_name(index, contract) for index in range(306)]
    errors = arrays['actor_only_command']-arrays['teacher_only_command']
    issue = arrays['time_s']-dt
    diagnostics = dict(command_error=errors, frame_command_rmse=np.sqrt(np.mean(errors**2, axis=-1)),
        frame_torque_rmse_nm=np.sqrt(np.mean((errors*scale)**2, axis=-1)), issue_time_s=issue)
    for pool in ('original', 'mixture', 'combined'):
        diagnostics[f'{pool}_nearest_dynamic_standardized_rms'] = np.zeros_like(arrays['time_s'][:, None]+np.zeros((1, 8)))
    cases = []
    for case in range(8):
        live = arrays['alive_before'][:, case]
        n = int(live.sum())
        indices = np.arange(n)
        valid_label = arrays['diagnostic_label_numerically_valid'][:n, case]
        balance, posture = arrays['balance'][:n, case], arrays['posture'][:n, case]
        pre_valid = arrays['pre_balance'][:n, case] & arrays['pre_posture'][:n, case]
        losses = {key: np.flatnonzero(~flag) for key, flag in
                  dict(balance=balance, posture=posture, combined=balance & posture).items()}
        first = int(losses['combined'][0]) if len(losses['combined']) else n
        clipped = arrays['clipped_features'][:n, case]
        error = errors[:n, case]
        masks = dict(all_live=np.ones(n, dtype=bool), reset=indices == 0,
            first_50ms=issue[:n] < .05, issued_before_first_departure=indices <= first,
            already_applied_by_first_departure=indices <= first-2,
            issued_after_first_departure=indices > first, pre_balanced_valid_posture=pre_valid)
        summaries = {key: error_summary(error, clipped, mask & valid_label, scale, names)
                     for key, mask in masks.items()}
        coverage = {}
        normalized = (arrays['obs'][:n, case].astype(float)-mean)/std
        nearest_deltas = {}
        for pool, values in zip(('original', 'mixture'), pools[case]):
            normalized_pool = (values.astype(float)-mean)/std
            distance = cdist(normalized[:, :161], normalized_pool[:, :161], metric='sqeuclidean')
            nearest = distance.argmin(-1)
            minimum = np.sqrt(distance[indices, nearest]/161.)
            diagnostics[f'{pool}_nearest_dynamic_standardized_rms'][:n, case] = minimum
            nearest_deltas[pool] = normalized-normalized_pool[nearest]
        old = diagnostics['original_nearest_dynamic_standardized_rms'][:n, case]
        added = diagnostics['mixture_nearest_dynamic_standardized_rms'][:n, case]
        combined = np.minimum(old, added)
        diagnostics['combined_nearest_dynamic_standardized_rms'][:n, case] = combined
        for pool, distance in dict(original=old, mixture=added, combined=combined).items():
            coverage[pool] = {key: dict(samples=int(mask.sum()), mean=float(distance[mask].mean()),
                maximum=float(distance[mask].max())) if mask.any() else dict(samples=0)
                for key, mask in masks.items()}
        closest = np.where((old <= added)[:, None], nearest_deltas['original'], nearest_deltas['mixture'])
        feature_contributions = {key: ranked(np.mean(closest[mask, :161]**2, axis=0), feature_names[:161])
                                 for key, mask in masks.items() if mask.any()}
        first_clipped = np.flatnonzero(clipped.any(-1))
        ending_gates = [key for key in arrays if key.startswith(('balance_gate_', 'posture_gate_'))]
        at_loss = None
        if first < n:
            applied = first-2
            at_loss = dict(ending_time_s=float(arrays['time_s'][first]),
                command_issue_time_s=float(issue[first]),
                first_violating_gates=[key for key in ending_gates if not arrays[key][first, case]],
                issued_command_rmse=float(diagnostics['frame_command_rmse'][first, case]) if valid_label[first] else None,
                held_actor_source_frame=applied if applied >= 0 else None,
                held_actor_source_issue_time_s=float(issue[applied]) if applied >= 0 else None,
                held_source_command_rmse=float(diagnostics['frame_command_rmse'][applied, case])
                    if applied >= 0 and valid_label[applied] else None,
                clipped_features_at_issue=int(clipped[first].sum()),
                nearest_original_dynamic_rms=float(old[first]), nearest_combined_dynamic_rms=float(combined[first]))
        cases.append(dict(case=case, survival_seconds=float(arrays['time_s'][n-1]),
            terminal_flag=int(arrays['done'][n-1, case]), longest_balanced_valid_posture_seconds=longest(balance & posture)*dt,
            first_losses={key: float(arrays['time_s'][lost[0]]) if len(lost) else None for key, lost in losses.items()},
            first_clipping_issue_time_s=float(issue[first_clipped[0]]) if len(first_clipped) else None,
            first_clipped_feature_names=[feature_names[index] for index in np.flatnonzero(clipped[first_clipped[0]])]
                if len(first_clipped) else [],
            command_error_subsets=summaries, first_departure=at_loss, coverage=coverage,
            coverage_squared_distance_feature_contributions=feature_contributions,
            feature_clipping_fractions=clipped.mean(0).tolist(),
            top_clipped_features=ranked(clipped.mean(0), feature_names),
            numerically_valid_teacher_labels=int(valid_label.sum()), unavailable_labels=int((~valid_label).sum())))
    subset_means = {}
    for subset in cases[0]['command_error_subsets']:
        rows = [case['command_error_subsets'][subset] for case in cases if case['command_error_subsets'][subset]['samples']]
        subset_means[subset] = dict(poses=len(rows), samples=sum(row['samples'] for row in rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in
               ('normalized_command_rmse', 'torque_rmse_nm', 'clipped_feature_fraction', 'frames_with_clipping_fraction')}) if rows else dict(poses=0, samples=0)
    report = dict(cases=cases, command_error_equal_pose_means=subset_means,
        coverage_metric='Nearest same-pose sample; unmodified shared standardization; RMS over dynamic features 0:161; no clipping or universal out-of-distribution threshold',
        coverage_original_pairs=47520, coverage_added_pairs=11520,
        feature_contribution_scope='Squared nearest-state distance and clipping frequency, not causal network attribution',
        motor_contribution_scope='Squared teacher-proposal difference, not a torque intervention or causal ranking',
        teacher_label_scope='Finite private-controller proposal with warning masks; recovery success unverified',
        time_scope='Physical gates at ending time; decision observations at ending time minus dt; held action source two frames earlier')
    np.savez_compressed(out/'diagnostics.npz', **diagnostics)
    save(out/'analysis.json', report)
    return report


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    protected, sources, contract = archive_inputs(args.out)
    rig = Rig.load(PRIOR/'dummy.xml', policy_exclude=())
    reference = PRIOR/'stationary_reference.npz'
    assert digest(rig.path) == contract['model_sha256'] and digest(reference) == contract['reference_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, contract=contract,
        prior=str(PRIOR), actors=['control', 'augmented'], backends=['native', 'gpu'], total_episodes=32,
        cases_per_setting=8, max_seconds=12., teacher='Original full unconstrained controller on private state, actual actor queue',
        control='Frozen deterministic actor only; inactive commands zero',
        state_reader='Exact native double state; GPU existing free-joint world-to-local velocity conversion',
        teacher_query_failures='Mark unavailable diagnostic labels and continue the autonomous plant',
        training_updates=0, optimizers_created=0, normalizer_samples_added=0, new_amp_clips=0, arena_changed=False))
    pools = reference_pools()
    results, frozen = {}, {}
    for label in ('control', 'augmented'):
        actor, full_state = load_actor(PRIOR/label)
        before, counts = tensor_hash(actor.state_dict()), normalizers(actor)
        assert before == read(PRIOR/label/'report.json')['final_actor_hash']
        for backend in ('native', 'gpu'):
            name = f'{label}-{backend}'
            print(dict(starting=name, autonomous=True, frozen=True), flush=True)
            arrays, capture_report = capture(rig, reference, actor, backend, PRIOR/label/backend, args.out/name)
            analysis = analyse(arrays, actor, pools, contract, args.out/name)
            assert tensor_hash(actor.state_dict()) == before and normalizers(actor) == counts
            results[name] = dict(capture=capture_report, analysis=analysis)
            save(args.out/'progress.json', dict(settings_completed=len(results), total_settings=4, last_setting=name))
            print(dict(completed=name, live_intervals=capture_report['live_control_intervals'],
                exact_prior_replay=capture_report['entire_prior_trace_exact'],
                error=analysis['command_error_equal_pose_means']), flush=True)
        frozen[label] = dict(actor_tensor_hash_before=before, actor_tensor_hash_after=tensor_hash(actor.state_dict()),
            full_saved_checkpoint_tensor_hash=tensor_hash(full_state), normalizer_counts_before=counts,
            normalizer_counts_after=normalizers(actor), portable_sidecar_match=True)
    assert all(digest(Path(path)) == expected for path, expected in protected.items())
    assert all(digest(Path(source[key])) == source['sha256'] for source in sources for key in ('path', 'snapshot'))
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=results, frozen=frozen,
        episodes_executed=32, plant_live_control_intervals=sum(row['capture']['live_control_intervals'] for row in results.values()),
        teacher_queries=sum(row['capture']['teacher_queries'] for row in results.values()),
        private_teacher_prediction_physics_steps=sum(row['capture']['teacher_private_prediction_physics_steps'] for row in results.values()),
        training_updates=0, optimizers_created=0, normalizer_samples_added=0, labels_admitted_as_recovery_data=False,
        new_amp_clips=0, protected_inputs_unchanged=True, sources_and_snapshots_unchanged=True,
        frozen_actors_and_normalizers_verified=True, arena_changed=False)
    save(args.out/'report.json', report)
    save(args.out/'completion-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        inputs_unchanged=True, protected_input_count=len(protected), source_count=len(sources),
        sources_and_snapshots_unchanged=True, frozen_actors_and_normalizers_verified=True,
        episodes_executed=32, optimizers_created=0, training_updates=0,
        output_hashes={str(path): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))
    print(dict(completed=True, episodes=32, output=str(args.out)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
