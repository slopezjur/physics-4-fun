"""Bounded current-policy trajectories, clipping and private teacher diagnostics."""
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


PRIOR = Path('logs/mimickit-training/stand-live-recovery-supervision-20261003-01')
EARLY = Path('logs/mimickit-training/stand-early-recovery-20261002-01')
LIVE = Path('logs/mimickit-training/stand-live-takeover-20261003-01')
FRAMES = 36
FIELDS = dict(full=(0, 306), physical=(0, 95), pending=(95, 161), reference=(161, 306))
TEACHER = Path('logs/mimickit-training/stand-teacher-20261001-01')
CORRECTIONS = Path('logs/mimickit-training/stand-corrections-20261001-01')
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
    for step in range(FRAMES):
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
        prediction_steps = np.zeros(8, dtype=np.int64)
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
            prediction_steps[case] = int(round(data.time/rig.model.opt.timestep))
            private_warning[case] = any(data.warning[warning].number for warning in BAD_WARNINGS)
            private_finite[case] = all(np.isfinite(value).all() for value in
                (teacher_command[case], data.qpos, data.qvel, data.qacc))
            private_finite[case] &= all(np.isfinite(value) for diagnostic in controller.reports[begin:]
                for value in diagnostic.values() if isinstance(value, (float, int, np.floating, np.integer)))
            for diagnostic in controller.reports[begin:]:
                values = {key: (float(value) if isinstance(value, (float, np.floating)) and np.isfinite(value)
                               else value if isinstance(value, (int, bool)) else str(value))
                          for key, value in diagnostic.items()}
                private_reports.append(dict(frame=step, **values))
        check_query_nonmutation(task, reader, qpos, qvel, pending, cursor, control, observation)
        standardized = (observation-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
        before_time = np.asarray([data.time for data in task.engine.datas]) if backend == 'native' else None
        _, _, done, _ = task.step(action)
        held = task.engine.native_ctrl[:, torch.as_tensor(rig.actuators, device=device)]
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings, device=device)] = DoneFlags.FAIL.value
        flags, state = gates(task)
        after_q, after_v = reader()
        physics_steps = (np.rint((np.asarray([data.time for data in task.engine.datas])-before_time)/rig.model.opt.timestep).astype(int)
            if backend == 'native' else active.astype(int)*rig.decimation)
        finite_label = succeeded & (np.abs(teacher_command)<=1.).all(-1) & private_finite & ~private_warning & cpu(pre_state['finite_state'])
        row = dict(alive_before=active.copy(), obs=cpu(observation), qpos_before=qpos.copy(), qvel_before=qvel.copy(),
            qpos=after_q.copy(), qvel=after_v.copy(), command=cpu(action), actor_only_command=cpu(action),
            teacher_only_command=teacher_command, pending_commands_oldest_first=cpu(pending),
            held_torque_nm=cpu(held), done=cpu(done), balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            pre_balance=cpu(pre_flags['balance']), pre_posture=cpu(pre_flags['posture']),
            teacher_query_succeeded=succeeded, teacher_private_solver_warning=private_warning,
            teacher_private_state_finite=private_finite, diagnostic_label_numerically_valid=finite_label & active,
            executed_action=cpu(held)/np.asarray(rig.action_scale, dtype=np.float32),
            queue_cursor_before=np.full(8, cursor, dtype=np.int64), plant_live_physics_steps=physics_steps,
            teacher_private_prediction_physics_steps=prediction_steps,
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
                  issue_time_s=np.arange(len(rows))*task.dt,
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
            prior_prefix = old[key][:len(rows)]
            matching_shape = arrays[key].shape == prior_prefix.shape
            replay[key] = dict(exact=bool(np.array_equal(arrays[key], prior_prefix)), same_shape=matching_shape,
                max_absolute_difference=float(np.max(np.abs(arrays[key].astype(float)-prior_prefix.astype(float))))
                if matching_shape else None)
        if backend == 'native':
            assert all(value['exact'] for value in replay.values()), 'Native actor prefix differs from Phase 39'
    out.mkdir()
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', private_reports)
    save(out/'teacher-query-errors.json', query_errors)
    report = dict(backend=backend, cases=8, requested_control_intervals=FRAMES, requested_horizon_seconds=FRAMES*task.dt, task_episode_seconds=12.,
        live_control_intervals=int(live.sum()), plant_live_physics_steps=int(arrays['plant_live_physics_steps'].sum()),
        teacher_private_prediction_physics_steps=int(arrays['teacher_private_prediction_physics_steps'].sum()),
        teacher_queries=int(live.sum()), teacher_query_failures=len(query_errors),
        numerically_valid_diagnostic_labels=int(arrays['diagnostic_label_numerically_valid'].sum()),
        teacher_private_warning_labels=int((arrays['teacher_private_solver_warning'] & live).sum()),
        plant_warning_measurement='Native BADQPOS/BADQVEL/BADQACC' if backend == 'native' else 'Unavailable',
        prior_replay=replay, entire_prior_prefix_exact=all(value['exact'] for value in replay.values()),
        teacher_changed_state_queue_or_inputs=False, executed_only_actor=True, delayed_torques_exact=True,
        labels_admitted_as_recovery_data=False, new_amp_clips=0, training_updates=0,
        wall_seconds=time.perf_counter()-started)
    save(out/'capture-report.json', report)
    actor.cpu()
    return arrays, report


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


def load_catalog():
    observations, actions, metadata, sources = [], [], [], []

    def append(path, obs, action, rows, cases, frames, episodes):
        source_id = len(sources)
        sources.append(dict(path=str(path), sha256=digest(path)))
        observations.append(obs)
        actions.append(action)
        metadata.append(dict(source_id=np.full(len(obs), source_id, dtype=np.int64),
            row=np.asarray(rows), case=np.asarray(cases), frame=np.asarray(frames), episode=np.asarray(episodes)))

    path = TEACHER/'native/dataset.npz'
    with np.load(path, allow_pickle=False) as data:
        obs, action = data['observation'].reshape(-1, 306), data['action'].reshape(-1, 33)
        rows = np.arange(len(obs), dtype=np.int64)
        append(path, obs, action, rows, rows % 8, rows//8, np.zeros(len(obs), dtype=np.int64))
    path = CORRECTIONS/'native/dataset.npz'
    with np.load(path, allow_pickle=False) as data:
        rows = np.arange(len(data['observation']), dtype=np.int64)
        append(path, data['observation'], data['action'], rows, data['case'], data['step'], data['episode'])
    for root in (EARLY, LIVE):
        for actor_name in ('control', 'augmented'):
            path = root/f'{actor_name}-native'/'admitted-dataset.npz'
            report = read(path.parent/'report.json')
            with np.load(path, allow_pickle=False) as data:
                rows = []
                for case in report['accepted_cases']:
                    selected = np.flatnonzero(data['case'] == case)
                    selected = selected[np.argsort(data['step'][selected])]
                    assert len(selected) == 720 and np.array_equal(data['step'][selected], np.arange(720))
                    rows.extend(selected.tolist())
                rows = np.asarray(rows, dtype=np.int64)
                append(path, data['observation'][rows], data['action'][rows], rows,
                    data['case'][rows], data['step'][rows], np.zeros(len(rows), dtype=np.int64))
    obs, action = np.concatenate(observations), np.concatenate(actions)
    catalog = {key: np.concatenate([row[key] for row in metadata]) for key in metadata[0]}
    assert len(obs) == 69120
    assert [len(values) for values in observations] == [28800, 18720, 5760, 4320, 5760, 5760]
    assert np.isfinite(obs).all() and np.isfinite(action).all() and (np.abs(action)<=1.).all()
    return obs, action, catalog, sources


def partitions(trace):
    live = trace['alive_before']
    frame = np.broadcast_to(np.arange(len(live))[:, None], live.shape)
    before = np.zeros_like(live)
    at_departure = np.zeros_like(live)
    after_state = np.zeros_like(live)
    last = np.zeros_like(live)
    for case in range(8):
        indices = np.flatnonzero(live[:, case])
        if not len(indices):
            continue
        last[indices[-1], case] = True
        violations = np.flatnonzero(live[:, case] & ~(trace['balance'][:, case] & trace['posture'][:, case]))
        first = int(violations[0]) if len(violations) else len(live)
        before[:first+1, case] = True
        if first < len(live):
            at_departure[first, case] = True
            if first+1 < len(live):
                after_state[first+1, case] = True
    return dict(all_live=live, reset=live & (frame == 0), at_0_2s=live & (frame == 12),
        early_noninitial=live & (frame > 0) & (frame < 12),
        late_valid_before_departure=live & before & trace['pre_balance'] & trace['pre_posture'] & (frame >= 12),
        decision_before_or_at_first_departure=live & before,
        first_departure_decision=live & at_departure, first_departed_state=live & after_state,
        after_first_departure=live & ~before, remaining_prefix_from_0_2s=live & (frame >= 12), last_live=live & last)


def nearest(query, pool, begin, end):
    results = {name: dict(index=[], rms=[]) for name in ('baseline', 'live_only', 'union')}
    for offset in range(0, len(query), 64):
        distance = cdist(query[offset:offset+64, begin:end], pool[:, begin:end], metric='sqeuclidean')
        old_index = distance[:, :57600].argmin(-1)
        new_index = distance[:, 57600:].argmin(-1)+57600
        rows = np.arange(len(distance))
        old, new = distance[rows, old_index], distance[rows, new_index]
        indices = dict(baseline=old_index, live_only=new_index, union=np.where(new < old, new_index, old_index))
        for name, index in indices.items():
            results[name]['index'].append(index)
            results[name]['rms'].append(np.sqrt(distance[rows, index]/(end-begin)))
    return {name: {key: np.concatenate(values) for key, values in row.items()} for name, row in results.items()}


@torch.no_grad()
def coverage(actor, traces, pool_obs, pool_actions, out):
    def standardize(values):
        return cpu((torch.from_numpy(values)-actor.obs_norm.get_mean())/actor.obs_norm.get_std()).astype(np.float64)
    pool = standardize(pool_obs)
    query = standardize(np.concatenate([trace['obs'][trace['alive_before']] for trace in traces.values()]))
    results = {}
    for space in ('unclipped', 'inference_clipped'):
        x, y = (query, pool) if space == 'unclipped' else (np.clip(query, -10., 10.), np.clip(pool, -10., 10.))
        for field, (begin, end) in FIELDS.items():
            result = nearest(x, y, begin, end)
            for name, values in result.items():
                for key, value in values.items():
                    results[f'{space}_{field}_{name}_{key}'] = value
            print(dict(coverage_space=space, field=field, queries=len(query), completed=True), flush=True)
    (out/'coverage').mkdir()
    offset = 0
    for name, trace in traces.items():
        live = trace['alive_before']
        count = int(live.sum())
        target, valid = trace['teacher_only_command'][live], trace['diagnostic_label_numerically_valid'][live]
        arrays = dict(frame=np.nonzero(live)[0], case=np.nonzero(live)[1])
        for key, value in results.items():
            arrays[key] = value[offset:offset+count]
            if key.endswith('_index'):
                error = (pool_actions[arrays[key]]-target).astype(np.float64)
                arrays[key.removesuffix('_index')+'_nearest_label_rmse'] = np.where(valid, np.sqrt(np.mean(error**2, axis=-1)), 0.)
        np.savez_compressed(out/'coverage'/f'{name}.npz', **arrays)
        offset += count
    assert offset == len(query)


def describe(trace, mask, contract, cov):
    live, clip = trace['alive_before'], trace['clipped_features']
    selected = mask & live
    count = int(selected.sum())
    if not count:
        return dict(samples=0)
    error = trace['command']-trace['teacher_only_command']
    valid = selected & trace['diagnostic_label_numerically_valid']
    scale = np.asarray(contract['torque_scale_nm'])
    feature_names = [feature_name(index, contract) for index in range(306)]
    clipping = dict(feature_fraction=float(clip[selected].mean()),
        frames_with_clipping_fraction=float(clip[selected].any(-1).mean()),
        per_feature_fraction=clip[selected].mean(0).tolist(),
        top_features=ranked(clip[selected].mean(0), feature_names, 12), blocks={})
    for block in contract['observation_layout']:
        start, end = block['offset'], block['offset']+block['width']
        flags = clip[selected, start:end]
        values = trace['standardized_obs'][selected, start:end].astype(np.float64)
        clipping['blocks'][block['name']] = dict(width=block['width'], feature_fraction=float(flags.mean()),
            frames_with_clipping_fraction=float(flags.any(-1).mean()),
            max_absolute_standardized_value=float(np.abs(values).max()))
    command = error_summary(error, clip, valid, scale, contract['actuator_names'])
    selected_flat, valid_flat = selected[live], valid[live]
    coverage_report = {}
    for space in ('unclipped', 'inference_clipped'):
        coverage_report[space] = {}
        for field in FIELDS:
            coverage_report[space][field] = {}
            for pool in ('baseline', 'live_only', 'union'):
                prefix = f'{space}_{field}_{pool}'
                values = cov[prefix+'_rms'][selected_flat]
                label_errors = cov[prefix+'_nearest_label_rmse'][valid_flat]
                coverage_report[space][field][pool] = dict(samples=count, mean_rms=float(values.mean()),
                    p95_rms=float(np.quantile(values,.95)), maximum_rms=float(values.max()),
                    valid_label_samples=int(valid.sum()),
                    mean_nearest_label_rmse=float(label_errors.mean()) if len(label_errors) else None)
    return dict(samples=count, valid_teacher_queries=int(valid.sum()), unavailable_teacher_queries=count-int(valid.sum()),
        command_error=command, clipping=clipping, coverage=coverage_report)


def analyse(trace, contract, out, coverage_path):
    with np.load(coverage_path, allow_pickle=False) as data:
        cov = {key: data[key] for key in data.files}
    masks = partitions(trace)
    scale, live = np.asarray(contract['torque_scale_nm']), trace['alive_before']
    valid = trace['diagnostic_label_numerically_valid'] & live
    error = np.where(valid[...,None], trace['command']-trace['teacher_only_command'], 0.)
    diagnostics = dict(command_error=error, valid_label=valid,
        frame_command_rmse=np.sqrt(np.mean(error.astype(np.float64)**2, axis=-1)),
        frame_torque_rmse_nm=np.sqrt(np.mean((error.astype(np.float64)*scale)**2, axis=-1)),
        **{f'subset_{name}': mask for name, mask in masks.items()})
    np.savez_compressed(out/'diagnostics.npz', **diagnostics)
    cases = []
    for case in range(8):
        selected = np.zeros_like(live)
        selected[:,case] = live[:,case]
        indices = np.flatnonzero(live[:,case])
        violations = np.flatnonzero(selected[:,case] & ~(trace['balance'][:,case] & trace['posture'][:,case]))
        clipped = np.flatnonzero(selected[:,case] & trace['clipped_features'][:,case].any(-1))
        first = int(violations[0]) if len(violations) else None
        end = int(indices[-1])
        cases.append(dict(case=case, captured_intervals=len(indices), last_ending_time_s=float(trace['time_s'][end]),
            terminal_flag=int(trace['done'][end,case]), stopped_by_budget=bool(len(indices)==FRAMES and trace['done'][end,case]==0),
            first_departure_ending_time_s=float(trace['time_s'][first]) if first is not None else None,
            first_clipping_issue_time_s=float(trace['issue_time_s'][clipped[0]]) if len(clipped) else None,
            first_departure_gates=[key for key in trace if key.startswith(('balance_gate_','posture_gate_')) and not trace[key][first,case]] if first is not None else [],
            partitions={name: describe(trace, selected & mask, contract, cov) for name,mask in masks.items()}))
    temporal = []
    for frame in range(len(live)):
        selected = np.zeros_like(live)
        selected[frame] = live[frame]
        temporal.append(dict(frame=frame, issue_time_s=float(trace['issue_time_s'][frame]), ending_time_s=float(trace['time_s'][frame]),
            **describe(trace, selected, contract, cov)))
    pooled = {name: describe(trace, mask, contract, cov) for name,mask in masks.items()}
    equal_pose = {}
    for name in masks:
        populated = [case['partitions'][name] for case in cases if case['partitions'][name]['samples']]
        errors = [row['command_error'] for row in populated if row['command_error']['samples']]
        equal_pose[name] = dict(poses=len(populated), valid_label_poses=len(errors),
            normalized_command_rmse=float(np.mean([row['normalized_command_rmse'] for row in errors])) if errors else None,
            torque_rmse_nm=float(np.mean([row['torque_rmse_nm'] for row in errors])) if errors else None,
            clipped_feature_fraction=float(np.mean([row['clipping']['feature_fraction'] for row in populated])) if populated else None)
    report = dict(cases=cases, temporal=temporal, pooled_partitions=pooled, equal_pose_partition_means=equal_pose,
        distance_scope='Exact nearest across all original poses in native pools; each block independently minimized; no universal OOD threshold',
        spaces='Frozen torch float32 standardization cast to float64; raw unclipped and inference clip10 distances',
        pools='baseline=57600 actual control training pairs; live_only=11520 added records; union=69120 actual augmented training pairs',
        label_scope='Warning-free finite bounded private teacher proposal only; no physical recovery or data admission',
        attribution_scope='Clipping frequency, nearest-state distance and command disagreement; no causal intervention',
        time_scope='Observation/action issue time versus ending gate time; command executes two intervals later; 0.6-second capture is not survival or Stand success')
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
    pool_obs, pool_actions, catalog, pool_sources = load_catalog()
    resolved = {str(Path(path).resolve()): value for path,value in protected.items()}
    assert all(resolved[str(Path(row['path']).resolve())] == row['sha256'] for row in pool_sources)
    np.savez_compressed(args.out/'pool-catalog.npz', **catalog)
    save(args.out/'pool-sources.json',pool_sources)
    save(args.out/'protocol.json', dict(inputs=protected,sources=sources,contract=contract,prior=str(PRIOR),
        actors=['control','augmented'],backends=['native','gpu'],total_episodes=32,cases_per_setting=8,
        max_control_intervals=FRAMES,max_seconds=FRAMES*contract['control_timestep'],
        task_episode_seconds=12.,budget_terminal_semantics='Stop capture at 36 intervals without introducing TIME into unchanged prior 12-second task',
        teacher='Unchanged one-Hz full PredictiveStandController; two queued intervals replayed privately and two forward-torque refinements',
        control='Frozen deterministic actor only; teacher never drives live plant',
        query_failures='Mark diagnostic label unavailable and continue actor trial',
        fields=FIELDS,spaces=['unclipped','inference_clipped'],baseline_pairs=57600,added_live_pairs=11520,union_pairs=69120,
        coverage='Exact float64 scipy sqeuclidean over float32-standardized inputs; full catalog across all poses; independent minima by block; first row tie, union tie favors baseline',
        training_updates=0,optimizers_created=0,normalizer_samples_added=0,new_amp_clips=0,dataset_admissions=0,arena_changed=False))
    results, frozen, traces, models = {}, {}, {}, {}
    for label in ('control','augmented'):
        actor, full_state = load_actor(PRIOR/label)
        before, counts = tensor_hash(actor.state_dict()), normalizers(actor)
        assert before == read(PRIOR/label/'report.json')['final_actor_hash']
        models[label] = actor
        for backend in ('native','gpu'):
            name = f'{label}-{backend}'
            print(dict(starting=name,autonomous=True,frozen=True,max_intervals=FRAMES),flush=True)
            arrays,capture_report = capture(rig,reference,actor,backend,PRIOR/label/backend,args.out/name)
            assert tensor_hash(actor.state_dict()) == before and normalizers(actor) == counts
            traces[name] = arrays
            results[name] = dict(capture=capture_report)
            save(args.out/'progress.json',dict(settings_captured=len(results),total_settings=4,last_setting=name))
            print(dict(captured=name,live_intervals=capture_report['live_control_intervals'],
                exact_prior_prefix=capture_report['entire_prior_prefix_exact'],query_failures=capture_report['teacher_query_failures']),flush=True)
        frozen[label] = dict(actor_tensor_hash_before=before,actor_tensor_hash_after=tensor_hash(actor.state_dict()),
            full_saved_checkpoint_tensor_hash=tensor_hash(full_state),normalizer_counts_before=counts,
            normalizer_counts_after=normalizers(actor),portable_sidecar_match=True)
    assert all(torch.equal(value,models['augmented'].obs_norm.state_dict()[key])
        for key,value in models['control'].obs_norm.state_dict().items())
    coverage(models['control'],traces,pool_obs,pool_actions,args.out)
    for name,trace in traces.items():
        results[name]['analysis'] = analyse(trace,contract,args.out/name,args.out/'coverage'/f'{name}.npz')
    assert all(digest(Path(path)) == expected for path,expected in protected.items())
    assert all(digest(Path(source[key])) == source['sha256'] for source in sources for key in ('path','snapshot'))
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(),settings=results,frozen=frozen,
        episodes_executed=32,plant_live_control_intervals=sum(row['capture']['live_control_intervals'] for row in results.values()),
        teacher_queries=sum(row['capture']['teacher_queries'] for row in results.values()),
        private_teacher_prediction_physics_steps=sum(row['capture']['teacher_private_prediction_physics_steps'] for row in results.values()),
        plant_live_physics_steps=sum(row['capture']['plant_live_physics_steps'] for row in results.values()),
        training_updates=0,optimizers_created=0,normalizer_samples_added=0,labels_admitted_as_recovery_data=False,
        dataset_admissions=0,new_amp_clips=0,protected_inputs_unchanged=True,sources_and_snapshots_unchanged=True,
        frozen_actors_and_normalizers_verified=True,arena_changed=False)
    save(args.out/'report.json',report)
    save(args.out/'completion-verification.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        inputs_unchanged=True,protected_input_count=len(protected),source_count=len(sources),
        sources_and_snapshots_unchanged=True,frozen_actors_and_normalizers_verified=True,
        episodes_executed=32,optimizers_created=0,training_updates=0,
        output_hashes={str(path):digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))
    print(dict(completed=True,episodes=32,output=str(args.out)),flush=True)


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
