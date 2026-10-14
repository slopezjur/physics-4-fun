"""Bounded teacher recovery from frozen student states with committed controls."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback

import mujoco
import numpy as np
import torch

from .admit_stand_controller import gpu_native_state
from .audit_stand_failures import digest, longest, read, save
from .capture_stand_student_states import BAD_WARNINGS, check_query_nonmutation, cpu, gates, pending_commands
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash


PRIOR = Path('logs/mimickit-training/stand-student-states-20261002-01')
ACTORS = Path('logs/mimickit-training/stand-state-coverage-20261002-01')
SECONDS = 12.


def protect_sources(out):
    protocol, completed = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = dict(protocol['inputs'])
    protected.update(completed['output_hashes'])
    protected[str(PRIOR/'final-verification.json')] = digest(PRIOR/'final-verification.json')
    paths = [Path(__file__), Path('mujoco_rig/mimic/collect_stand_corrections.py')]
    for source in protocol['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        value = digest(path)
        assert str(path) not in protected or protected[str(path)] == value
        protected[str(path)] = value
    assert all(digest(Path(path)) == value for path, value in protected.items())
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources, protocol['contract']


def select_states(name):
    trace_path = PRIOR/name/'trace.npz'
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    selected, records = [], []
    for case in range(8):
        live = trace['alive_before'][:, case]
        first = int(np.flatnonzero(live & ~(trace['balance'][:, case] & trace['posture'][:, case]))[0])
        eligible = live & trace['pre_balance'][:, case] & trace['pre_posture'][:, case]
        eligible &= np.arange(len(live)) <= first
        index = int(np.flatnonzero(eligible)[-1])
        assert index > 0 and trace['diagnostic_label_numerically_valid'][index, case]
        selected.append(index)
        records.append(dict(case=case, source_trace=str(trace_path), source_trace_sha256=digest(trace_path),
            source_frame=index, source_issue_time_s=float(trace['time_s'][index]-trace['time_s'][0]),
            source_first_departure_ending_time_s=float(trace['time_s'][first]),
            source_pre_balance=True, source_pre_posture=True, source_numerically_valid_teacher_label=True))
    ids = np.arange(8)
    indices = np.asarray(selected)
    start = dict(qpos=trace['qpos_before'][indices, ids], qvel=trace['qvel_before'][indices, ids],
        pending_commands_oldest_first=trace['pending_commands_oldest_first'][indices, ids],
        held_torque_nm=trace['held_torque_nm'][indices-1, ids],
        previous_issued_action=trace['command'][indices-1, ids], source_observation=trace['obs'][indices, ids],
        source_teacher_proposal=trace['teacher_only_command'][indices, ids])
    measurements = {key[4:]: value[indices, ids] for key, value in trace.items()
                    if key.startswith('pre_')}
    return start, measurements, records


def restore(task, start, backend):
    device = task.device
    dtype = torch.float64 if backend == 'native' else torch.float32
    task.engine.reset_envs(task.ids, torch.as_tensor(start['qpos'], dtype=dtype, device=device),
                           torch.as_tensor(start['qvel'], dtype=dtype, device=device))
    queue = task.engine.policy_control
    queue.cursor = 0
    pending = torch.as_tensor(start['pending_commands_oldest_first'], device=device).reshape(8, 2, 33)
    queue.pending.copy_(pending.permute(1, 0, 2))
    queue.ctrl.zero_()
    queue.ctrl[:, queue.indices] = torch.as_tensor(start['held_torque_nm'], device=device)
    task.engine.native_ctrl.copy_(queue.ctrl)
    task.previous_action.copy_(torch.as_tensor(start['previous_issued_action'], device=device))
    task.steps.zero_()
    task.offset.zero_()
    # Rebuild the reset contact snapshot under the restored held torque; no time advance.
    if backend == 'native':
        for case, data in enumerate(task.engine.datas):
            data.ctrl[:] = task.engine.native_ctrl[case].numpy()
            mujoco.mj_forward(task.rig.model, data)
            task.engine._ground_force_cache[case] = task.engine._read_ground_forces(data)
    else:
        import mujoco_warp
        import warp as wp
        engine = task.engine
        wp.to_torch(engine._solver.mjw_data.ctrl).copy_(engine.native_ctrl)
        mujoco_warp.forward(engine._solver.mjw_model, engine._solver.mjw_data)
        engine._solver.update_contacts(engine._contacts, engine._sim_state.raw_state)
        engine._update_contact_sensors()
    task.previous_com = task.center_of_mass().clone()
    assert np.array_equal(cpu(pending_commands(queue)), start['pending_commands_oldest_first'])
    assert np.array_equal(cpu(task.engine.native_ctrl[:, queue.indices]), start['held_torque_nm'])


def numeric_diagnostic(row):
    return {key: float(value) if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else int(value) if isinstance(value, (int, np.integer)) else str(value)
            for key, value in row.items()}


@torch.no_grad()
def recovery(rig, reference, name, contract, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    backend = name.rsplit('-', 1)[1]
    device = 'cpu' if backend == 'native' else 'cuda:0'
    start, source_measurements, selections = select_states(name)
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9', episode_seconds=SECONDS, **options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    restore(task, start, backend)
    controller = PredictiveStandController(task)
    reader = controller._native_state if backend == 'native' else lambda: gpu_native_state(task)
    initial_q, initial_v = reader()
    initial_flags, initial_state = gates(task)
    initial_observation = cpu(task.observations())
    if backend == 'native':
        assert np.array_equal(initial_q, start['qpos']) and np.array_equal(initial_v, start['qvel'])
    out.mkdir()
    np.savez_compressed(out/'source-states.npz', **start,
        **{f'source_measurement_{key}': value for key, value in source_measurements.items()},
        restored_qpos=initial_q, restored_qvel=initial_v, restored_observation=initial_observation,
        **{f'restored_measurement_{key}': cpu(value) for key, value in initial_state.items()},
        restored_balance=cpu(initial_flags['balance']), restored_posture=cpu(initial_flags['posture']))
    restore_report = dict(cases=selections, queue_preserved_exactly=True, held_torque_preserved_exactly=True,
        solver_contact_history='Reset; initial snapshot recomputed under the restored held torque; not an exact continuation',
        max_qpos_difference=float(np.abs(initial_q-start['qpos']).max()),
        max_qvel_difference=float(np.abs(initial_v-start['qvel']).max()),
        max_observation_difference=float(np.abs(initial_observation-start['source_observation']).max()),
        measurements={key: dict(source=value.tolist(), restored=cpu(initial_state[key]).tolist(),
            max_absolute_difference=float(np.abs(cpu(initial_state[key]).astype(float)-value.astype(float)).max()))
            for key, value in source_measurements.items() if key in initial_state},
        initially_balanced_cases=int(initial_flags['balance'].sum()),
        initially_posture_valid_cases=int(initial_flags['posture'].sum()))
    save(out/'restoration.json', restore_report)
    active = np.ones(8, dtype=bool)
    rows, errors, private_reports = [], [], []
    started = time.perf_counter()
    for step in range(int(np.ceil(SECONDS/task.dt))):
        observation = task.observations()
        q, v = reader()
        queue = task.engine.policy_control
        pending, cursor, control = pending_commands(queue).clone(), queue.cursor, task.engine.native_ctrl.clone()
        assert torch.equal(observation[:, 95:161], pending)
        proposed = np.zeros((8, 33), dtype=np.float32)
        valid, warning, private_finite = (np.zeros(8, dtype=bool) for _ in range(3))
        prediction_steps = np.zeros(8, dtype=int)
        controller.state_reader = lambda: (q, v)
        for case in np.flatnonzero(active):
            mask = np.zeros(8, dtype=bool)
            mask[case] = True
            begin = len(controller.reports)
            try:
                proposed[case] = controller.command(mask).numpy()[case]
                valid[case] = bool(np.isfinite(proposed[case]).all() and (np.abs(proposed[case]) <= 1.).all())
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                errors.append(dict(frame=step, case=int(case), error=str(error)))
            data = controller.data[case]
            prediction_steps[case] = int(round(data.time/rig.model.opt.timestep))
            warning[case] = any(data.warning[item].number for item in BAD_WARNINGS)
            private_finite[case] = all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc))
            private_finite[case] &= all(np.isfinite(value) for row in controller.reports[begin:]
                for value in row.values() if isinstance(value, (float, int, np.floating, np.integer)))
            private_reports.extend(dict(frame=step, **numeric_diagnostic(row)) for row in controller.reports[begin:])
        check_query_nonmutation(task, reader, q, v, pending, cursor, control, observation)
        integrated = active & valid
        if backend == 'native':
            task.engine.active = integrated.copy()
            before_time = np.asarray([data.time for data in task.engine.datas])
        _, _, done, _ = task.step(torch.from_numpy(proposed).to(device))
        plant_warning = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(plant_warning | (active & ~valid), device=device)] = DoneFlags.FAIL.value
        flags, state = gates(task)
        after_q, after_v = reader()
        physics_steps = (np.rint((np.asarray([data.time for data in task.engine.datas])-before_time)/rig.model.opt.timestep).astype(int)
                         if backend == 'native' else integrated.astype(int)*rig.decimation)
        row = dict(alive_before=active.copy(), integrated_before=integrated, observation=cpu(observation),
            action=proposed, executed_action=proposed.copy(), pending_actions=cpu(pending),
            held_torque_nm=cpu(task.engine.native_ctrl[:, queue.indices]), qpos_before=q.copy(), qvel_before=v.copy(),
            qpos_after=after_q, qvel_after=after_v, done=cpu(done), balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            command_valid=valid, teacher_private_solver_warning=warning, teacher_private_state_finite=private_finite,
            teacher_private_prediction_physics_steps=prediction_steps, plant_live_physics_steps=physics_steps,
            solver_warning=plant_warning, plant_solver_warning_known=np.full(8, backend == 'native', dtype=bool))
        row.update({key: cpu(value) for key, value in state.items()})
        rows.append(row)
        active &= cpu(done) == DoneFlags.NULL.value
        if (step+1) % 120 == 0 or not active.any() or step+1 == int(np.ceil(SECONDS/task.dt)):
            progress = dict(setting=name, seconds=(step+1)*task.dt, live_cases=int(active.sum()),
                            wall_seconds=time.perf_counter()-started)
            save(out/'progress.json', progress)
            print(progress, flush=True)
        if not active.any():
            break
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays['time_s'] = (np.arange(len(rows))+1)*task.dt
    arrays['issue_time_s'] = np.arange(len(rows))*task.dt
    expected_held = np.zeros_like(arrays['held_torque_nm'])
    expected_held[:min(2, len(rows))] = start['pending_commands_oldest_first'].reshape(8, 2, 33).transpose(1, 0, 2)[:len(rows)]*np.asarray(rig.action_scale, dtype=np.float32)
    expected_held[2:] = arrays['action'][:-2]*np.asarray(rig.action_scale, dtype=np.float32)
    assert np.array_equal(expected_held, arrays['held_torque_nm'])
    assert np.array_equal(arrays['observation'][..., 95:161], arrays['pending_actions'])
    assert np.array_equal(arrays['action'], arrays['executed_action'])
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', private_reports)
    save(out/'teacher-errors.json', errors)
    cases, accepted = [], []
    for case in range(8):
        live = arrays['alive_before'][:, case]
        end = int(np.flatnonzero(live)[-1])
        command_failure = bool((~arrays['command_valid'][live, case]).any())
        survived = bool(arrays['done'][end, case] == DoneFlags.TIME.value)
        final = live & (arrays['time_s'] > arrays['time_s'][end]-2.)
        full_window = arrays['time_s'][end] >= 2.
        balanced = bool(survived and full_window and arrays['balance'][final, case].all())
        standing = bool(balanced and arrays['posture'][final, case].all())
        finite = all(np.isfinite(arrays[key][live, case]).all() for key in
            ('observation', 'action', 'pending_actions', 'held_torque_nm', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after'))
        numeric = bool(finite and arrays['command_valid'][live, case].all() and arrays['teacher_private_state_finite'][live, case].all()
            and not arrays['teacher_private_solver_warning'][live, case].any() and not arrays['solver_warning'][live, case].any())
        admitted = bool(standing and numeric)
        if admitted:
            accepted.append(case)
        violations = np.flatnonzero(~(arrays['balance'][live, case] & arrays['posture'][live, case]))
        restored = restore_report['cases'][case]
        cases.append(dict(case=case, source_frame=restored['source_frame'], source_issue_time_s=restored['source_issue_time_s'],
            survived=survived, balance_pass=balanced, standing_pass=standing, accepted=admitted,
            numeric_and_private_warning_checks_pass=numeric,
            duration_seconds=float(arrays['issue_time_s'][end] if command_failure else arrays['time_s'][end]),
            terminal_reason='teacher_command_error' if command_failure else 'TIME' if survived else 'native_solver_warning' if arrays['solver_warning'][end, case] else 'physical_FAIL',
            first_balance_or_posture_violation_seconds=float(arrays['time_s'][violations[0]]) if len(violations) else None,
            longest_balanced_valid_posture_seconds=longest(arrays['balance'][live, case] & arrays['posture'][live, case])*task.dt,
            balance_violation_seconds=float((~arrays['balance'][live, case]).sum()*task.dt),
            posture_violation_seconds=float((~arrays['posture'][live, case]).sum()*task.dt),
            first_teacher_vs_source_proposal_max_difference=float(np.abs(arrays['action'][0, case]-start['source_teacher_proposal'][case]).max()),
            native_plant_warning=bool(arrays['solver_warning'][live, case].any()) if backend == 'native' else None,
            private_teacher_warning=bool(arrays['teacher_private_solver_warning'][live, case].any())))
    if accepted:
        keep = np.zeros_like(arrays['alive_before'])
        keep[:, accepted] = arrays['alive_before'][:, accepted]
        keys = ('observation', 'action', 'executed_action', 'pending_actions', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after')
        admitted_arrays = {key: arrays[key][keep] for key in keys}
        admitted_arrays.update(case=np.broadcast_to(np.arange(8), keep.shape)[keep],
            step=np.broadcast_to(np.arange(len(rows))[:, None], keep.shape)[keep], teacher_executed=np.ones(int(keep.sum()), dtype=bool))
        np.savez_compressed(out/'admitted-dataset.npz', **admitted_arrays)
    report = dict(setting=name, backend=backend, cases=cases, accepted_cases=accepted,
        survivors=sum(row['survived'] for row in cases), balance_passes=sum(row['balance_pass'] for row in cases),
        standing_passes=sum(row['standing_pass'] for row in cases), admitted_samples=len(rows)*len(accepted),
        admitted_scope='Native supervised candidates' if backend == 'native' else 'Independent GPU validation; not training samples',
        admission='TIME at full horizon; final two seconds pass balance/posture; finite live samples and bounded commands; native plant and private prediction warnings absent; GPU plant warnings unmeasured',
        live_control_intervals=int(arrays['integrated_before'].sum()),
        plant_live_physics_steps=int(arrays['plant_live_physics_steps'].sum()),
        private_teacher_prediction_physics_steps=int(arrays['teacher_private_prediction_physics_steps'].sum()),
        teacher_command_errors=len(errors), initial_committed_actor_torques_preserved=True,
        teacher_did_not_mutate_live_state_or_queue=True, new_amp_clips=0, learning_updates=0,
        wall_seconds=time.perf_counter()-started, restoration=restore_report)
    save(out/'report.json', report)
    print(dict(completed=name, survivors=report['survivors'], stand_passes=report['standing_passes'], accepted=accepted), flush=True)
    return report


def run(args):
    activate(args.mimickit)
    from .amp_recovery import validate_sidecar
    torch.set_num_threads(2)
    protected, sources, contract = protect_sources(args.out)
    frozen = {}
    for actor in ('control', 'augmented'):
        state = torch.load(ACTORS/actor/'model.amp.pt', map_location='cpu', weights_only=True)
        validate_sidecar(torch.load(ACTORS/actor/'model.pt', map_location='cpu', weights_only=True), state)
        frozen[actor] = dict(full_checkpoint_tensor_hash=tensor_hash(state), observation_count=int(state['_obs_norm._count']),
                             action_count=int(state['_a_norm._count']), portable_sidecar_match=True)
    rig, reference = Rig.load(ACTORS/'dummy.xml', policy_exclude=()), ACTORS/'stationary_reference.npz'
    assert digest(rig.path) == contract['model_sha256'] and digest(reference) == contract['reference_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, contract=contract, prior=str(PRIOR),
        total_episodes=32, cases_per_setting=8, max_seconds=SECONDS, settings=['control-native', 'control-gpu', 'augmented-native', 'augmented-gpu'],
        source_selection='Last live balanced/posture-valid pre-command state at or before first ending-state departure',
        restoration='Preserve q/v, reference, held torque and oldest-first two-command actor queue; reset solver/contact history; no continuation claim',
        controller='Unchanged one-Hz full predictive controller, inverse contacts, two refinements, actual pending queue',
        admission='Only complete successful numeric/physical recovery cases; native training candidates and separate GPU validation',
        frozen=frozen, neural_forward_passes=0, learning_optimizers_created=0, training_updates=0,
        new_amp_clips=0, original_demonstrations_changed=False, arena_changed=False))
    results = {}
    for actor in ('control', 'augmented'):
        for backend in ('native', 'gpu'):
            name = f'{actor}-{backend}'
            print(dict(starting=name, teacher_recovery=True, actor_inference=False), flush=True)
            results[name] = recovery(rig, reference, name, contract, args.out/name)
            save(args.out/'progress.json', dict(settings_completed=len(results), total_settings=4, last_setting=name))
    for actor, before in frozen.items():
        state = torch.load(ACTORS/actor/'model.amp.pt', map_location='cpu', weights_only=True)
        assert tensor_hash(state) == before['full_checkpoint_tensor_hash']
    assert all(digest(Path(path)) == value for path, value in protected.items())
    assert all(digest(Path(source[key])) == source['sha256'] for source in sources for key in ('path', 'snapshot'))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=results,
        frozen=frozen, episodes_executed=32, survivors=sum(row['survivors'] for row in results.values()),
        teacher_stand_passes=sum(row['standing_passes'] for row in results.values()),
        native_admitted_samples=sum(row['admitted_samples'] for name, row in results.items() if name.endswith('native')),
        gpu_validation_admitted_samples=sum(row['admitted_samples'] for name, row in results.items() if name.endswith('gpu')),
        live_control_intervals=sum(row['live_control_intervals'] for row in results.values()),
        plant_live_physics_steps=sum(row['plant_live_physics_steps'] for row in results.values()),
        private_teacher_prediction_physics_steps=sum(row['private_teacher_prediction_physics_steps'] for row in results.values()),
        inputs_and_sources_unchanged=True, actors_and_normalizers_frozen=True, neural_forward_passes=0,
        training_updates=0, learning_optimizers_created=0, new_amp_clips=0, arena_changed=False))
    save(args.out/'completion-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_input_count=len(protected), source_count=len(sources), inputs_and_source_snapshots_verify=True,
        frozen_checkpoints_verify=True, episodes=32, neural_forward_passes=0, training_updates=0,
        output_hashes={str(path): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))


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
