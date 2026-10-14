"""Bounded live teacher takeover after twelve frozen student control intervals."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .admit_stand_controller import gpu_native_state
from .admit_stand_student_recovery import numeric_diagnostic
from .audit_stand_failures import digest, longest, read, save
from .audit_stand_student_errors import load_actor, normalizers
from .capture_stand_student_states import BAD_WARNINGS, check_query_nonmutation, cpu, gates, pending_commands
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash


PRIOR = Path('logs/mimickit-training/stand-recovery-transfer-audit-20261002-02')
MODELS = Path('logs/mimickit-training/stand-recovery-supervision-20261002-01')
PREFIX = 12
TEACHER_SECONDS = 12.


def protect(out):
    protocol, complete = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = {**protocol['inputs'], **complete['output_hashes']}
    protected[str(PRIOR/'final-verification.json')] = digest(PRIOR/'final-verification.json')
    paths = [Path(__file__)]+[Path(source['path']) for source in protocol['sources']]
    for source in protocol['sources']:
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


@torch.no_grad()
def evaluate(rig, reference, actor, label, backend, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    dt = rig.model.opt.timestep*rig.decimation
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
        episode_seconds=PREFIX*dt+TEACHER_SECONDS, **options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    actor.to(device).eval()
    before, counts = tensor_hash(actor.state_dict()), normalizers(actor)
    controller = PredictiveStandController(task)
    reader = controller._native_state if backend == 'native' else lambda: gpu_native_state(task)
    out.mkdir()
    active = np.ones(8, dtype=bool)
    rows, errors, diagnostics = [], [], []
    takeover = None
    started = time.perf_counter()
    teacher_frames = int(np.ceil(TEACHER_SECONDS/task.dt))
    for step in range(PREFIX+teacher_frames):
        observation = task.observations()
        q, v = reader()
        pre_flags, pre_state = gates(task)
        queue = task.engine.policy_control
        pending, cursor, control = pending_commands(queue).clone(), queue.cursor, task.engine.native_ctrl.clone()
        assert torch.equal(observation[:, 95:161], pending)
        if step == PREFIX:
            # Observe the live handoff. No reset, forward recomputation or queue edit.
            takeover = dict(qpos=q.copy(), qvel=v.copy(), observation=cpu(observation),
                pending_actions=cpu(pending), held_torque_nm=cpu(control[:, queue.indices]),
                reached=active.copy(), balance=cpu(pre_flags['balance']), posture=cpu(pre_flags['posture']),
                **{key: cpu(value) for key, value in pre_state.items()})
            np.savez_compressed(out/'takeover-state.npz', **takeover)
        proposed = np.zeros((8, 33), dtype=np.float32)
        valid, warning, private_finite = (np.zeros(8, dtype=bool) for _ in range(3))
        prediction_steps = np.zeros(8, dtype=np.int64)
        actor_issued = active.copy() if step < PREFIX else np.zeros(8, dtype=bool)
        teacher_issued = active.copy() if step >= PREFIX else np.zeros(8, dtype=bool)
        if step < PREFIX:
            proposed = cpu(actor(observation))
            proposed[~active] = 0.
            valid = active & np.isfinite(proposed).all(-1) & (np.abs(proposed)<=1.).all(-1)
        else:
            controller.state_reader = lambda: (q, v)
            for case in np.flatnonzero(active):
                mask = np.zeros(8, dtype=bool)
                mask[case] = True
                begin = len(controller.reports)
                try:
                    proposed[case] = controller.command(mask).numpy()[case]
                    valid[case] = bool(np.isfinite(proposed[case]).all() and (np.abs(proposed[case])<=1.).all())
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    errors.append(dict(frame=step, case=int(case), error=str(error)))
                data = controller.data[case]
                prediction_steps[case] = int(round(data.time/rig.model.opt.timestep))
                warning[case] = any(data.warning[item].number for item in BAD_WARNINGS)
                private_finite[case] = all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc))
                private_finite[case] &= all(np.isfinite(value) for row in controller.reports[begin:]
                    for value in row.values() if isinstance(value, (float, int, np.floating, np.integer)))
                diagnostics.extend(dict(frame=step, **numeric_diagnostic(row)) for row in controller.reports[begin:])
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
        row = dict(alive_before=active.copy(), integrated_before=integrated,
            actor_issued=actor_issued, teacher_issued=teacher_issued, observation=cpu(observation),
            action=proposed, pending_actions=cpu(pending), held_torque_nm=cpu(task.engine.native_ctrl[:, queue.indices]),
            qpos_before=q.copy(), qvel_before=v.copy(), qpos_after=after_q.copy(), qvel_after=after_v.copy(),
            done=cpu(done), balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            pre_balance=cpu(pre_flags['balance']), pre_posture=cpu(pre_flags['posture']), command_valid=valid,
            teacher_private_solver_warning=warning, teacher_private_state_finite=private_finite,
            teacher_private_prediction_physics_steps=prediction_steps, plant_live_physics_steps=physics_steps,
            solver_warning=plant_warning, plant_solver_warning_known=np.full(8, backend == 'native', dtype=bool))
        row.update({key: cpu(value) for key, value in state.items()})
        rows.append(row)
        active &= cpu(done) == DoneFlags.NULL.value
        if (step+1)%120 == 0 or not active.any() or step+1 == PREFIX+teacher_frames:
            progress = dict(setting=out.name, reset_seconds=(step+1)*task.dt,
                teacher_seconds=max(0, step+1-PREFIX)*task.dt, live_cases=int(active.sum()), wall_seconds=time.perf_counter()-started)
            save(out/'progress.json', progress)
            print(progress, flush=True)
        if not active.any():
            break
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays.update(time_s=(np.arange(len(rows))+1)*task.dt, issue_time_s=np.arange(len(rows))*task.dt)
    expected_held = np.zeros_like(arrays['held_torque_nm'])
    expected_held[2:] = arrays['action'][:-2]*np.asarray(rig.action_scale, dtype=np.float32)
    assert np.array_equal(arrays['held_torque_nm'], expected_held)
    assert np.array_equal(arrays['observation'][..., 95:161], arrays['pending_actions'])
    assert tensor_hash(actor.state_dict()) == before and normalizers(actor) == counts
    actor.cpu()
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', diagnostics)
    save(out/'teacher-errors.json', errors)
    cases, accepted = [], []
    for case in range(8):
        live = arrays['alive_before'][:, case]
        expert = arrays['teacher_issued'][:, case]
        end = int(np.flatnonzero(live)[-1])
        count = int(expert.sum())
        survived = bool(arrays['done'][end, case] == DoneFlags.TIME.value)
        final = expert & (arrays['time_s'] > arrays['time_s'][end]-2.)
        balanced = bool(survived and count == teacher_frames and arrays['balance'][final, case].all())
        standing = bool(balanced and arrays['posture'][final, case].all())
        finite = all(np.isfinite(arrays[key][live, case]).all() for key in
            ('observation', 'action', 'pending_actions', 'held_torque_nm', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after'))
        numeric = bool(finite and arrays['command_valid'][live, case].all() and
            arrays['teacher_private_state_finite'][expert, case].all() and
            not arrays['teacher_private_solver_warning'][expert, case].any() and not arrays['solver_warning'][live, case].any())
        admitted = bool(standing and numeric)
        if admitted:
            accepted.append(case)
        violations = np.flatnonzero(expert & ~(arrays['balance'][:, case] & arrays['posture'][:, case]))
        prefix_failed = count == 0
        command_failure = bool((~arrays['command_valid'][live, case]).any())
        reset_duration = float(arrays['issue_time_s'][end] if command_failure else arrays['time_s'][end])
        cases.append(dict(case=case, prefix_live_intervals=int(arrays['actor_issued'][:, case].sum()),
            takeover_reached=not prefix_failed, survived=survived, balance_pass=balanced, standing_pass=standing,
            accepted=admitted, teacher_live_intervals=count, numeric_checks_pass=numeric,
            reset_duration_seconds=reset_duration, teacher_duration_seconds=max(0., reset_duration-PREFIX*task.dt),
            takeover_balance=bool(takeover['balance'][case]) if not prefix_failed else None,
            takeover_posture=bool(takeover['posture'][case]) if not prefix_failed else None,
            terminal_reason='command_error' if command_failure else 'TIME' if survived else
                'native_solver_warning' if arrays['solver_warning'][end, case] else 'prefix_physical_FAIL' if prefix_failed else 'physical_FAIL',
            first_teacher_balance_or_posture_violation_seconds=float(arrays['time_s'][violations[0]]-PREFIX*task.dt) if len(violations) else None,
            teacher_longest_balanced_valid_posture_seconds=longest(arrays['balance'][expert, case] & arrays['posture'][expert, case])*task.dt,
            teacher_balance_violation_seconds=float((~arrays['balance'][expert, case]).sum()*task.dt),
            teacher_posture_violation_seconds=float((~arrays['posture'][expert, case]).sum()*task.dt)))
    if accepted:
        keep = np.zeros_like(arrays['alive_before'])
        keep[:, accepted] = arrays['teacher_issued'][:, accepted]
        keys = ('observation', 'action', 'pending_actions', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after')
        dataset = {key: arrays[key][keep] for key in keys}
        dataset.update(executed_action=arrays['action'][keep],
            case=np.broadcast_to(np.arange(8), keep.shape)[keep],
            step=np.broadcast_to((np.arange(len(rows))-PREFIX)[:, None], keep.shape)[keep],
            teacher_executed=np.ones(int(keep.sum()), dtype=bool))
        np.savez_compressed(out/'admitted-dataset.npz', **dataset)
    with np.load(MODELS/label/backend/'trace.npz', allow_pickle=False) as old:
        comparison = {}
        for new, prior in (('action', 'command'), ('balance', 'balance'), ('posture', 'posture'),
            ('pelvis_height_m', 'pelvis_height_m'), ('pelvis_tilt_degrees', 'pelvis_tilt_degrees'), ('foot_loads_n', 'foot_loads_n')):
            n = min(PREFIX, len(rows), len(old[prior]))
            comparison[new] = dict(exact=bool(np.array_equal(arrays[new][:n], old[prior][:n])),
                frames_compared=n, max_absolute_difference=float(np.abs(arrays[new][:n].astype(float)-old[prior][:n].astype(float)).max()))
    summary = dict(setting=out.name, actor=label, backend=backend, cases=cases, accepted_cases=accepted,
        survivors=sum(row['survived'] for row in cases), standing_passes=sum(row['standing_pass'] for row in cases),
        admitted_samples=teacher_frames*len(accepted), prefix_failed_cases=sum(not row['takeover_reached'] for row in cases),
        takeover_balanced_cases=int(takeover['balance'][takeover['reached']].sum()) if takeover is not None else 0,
        takeover_posture_valid_cases=int(takeover['posture'][takeover['reached']].sum()) if takeover is not None else 0,
        prefix_replay_comparison=comparison, actor_live_predictions=int(arrays['actor_issued'].sum()),
        live_control_intervals=int(arrays['integrated_before'].sum()), plant_live_physics_steps=int(arrays['plant_live_physics_steps'].sum()),
        private_teacher_prediction_physics_steps=int(arrays['teacher_private_prediction_physics_steps'].sum()),
        teacher_command_errors=len(errors), actor_hash_before=before, actor_hash_after=tensor_hash(actor.state_dict()),
        normalizers=counts, wall_seconds=time.perf_counter()-started,
        solver_contact_history='Preserved in the same live task; no restore or forward recomputation at handoff',
        admission='Full teacher horizon TIME and final two seconds valid; all live finite/bounded, no native or private warnings; GPU plant warnings unavailable',
        learning_updates=0, learning_optimizers_created=0, new_amp_clips=0, arena_changed=False)
    save(out/'report.json', summary)
    print(dict(completed=out.name, stand_passes=summary['standing_passes'], accepted=accepted), flush=True)
    return summary


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    protected, sources, contract = protect(args.out)
    rig, reference = Rig.load(MODELS/'dummy.xml', policy_exclude=()), MODELS/'stationary_reference.npz'
    assert digest(rig.path) == contract['model_sha256'] and digest(reference) == contract['reference_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    actors, frozen = {}, {}
    for label in ('control', 'augmented'):
        actor, full = load_actor(MODELS/label)
        actors[label] = actor
        frozen[label] = dict(full_checkpoint_tensor_hash=tensor_hash(full), actor_hash=tensor_hash(actor.state_dict()),
            normalizers=normalizers(actor), portable_sidecar_match=True)
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, contract=contract, models=str(MODELS),
        episodes=32, cases_per_setting=8, prefix_intervals=PREFIX, prefix_seconds=PREFIX*contract['control_timestep'],
        teacher_seconds=TEACHER_SECONDS, first_teacher_effect_ending_time_s=(PREFIX+3)*contract['control_timestep'],
        frozen=frozen, controller='Unchanged one-Hz full predictive inverse-contact controller with two refinements',
        handoff='Live task/contact/solver history and actual two-slot actor queue preserved; no reset at takeover',
        native_candidates_and_separate_gpu_validation=True, prefix_actor_commands_are_not_expert_targets=True,
        learning_updates=0, learning_optimizers_created=0, new_amp_clips=0, arena_changed=False))
    results = {}
    for label, actor in actors.items():
        for backend in ('native', 'gpu'):
            name = f'{label}-{backend}'
            print(dict(starting=name, prefix_intervals=PREFIX, teacher_seconds=TEACHER_SECONDS), flush=True)
            results[name] = evaluate(rig, reference, actor, label, backend, args.out/name)
    for label in frozen:
        full = torch.load(MODELS/label/'model.amp.pt', map_location='cpu', weights_only=True)
        assert tensor_hash(full) == frozen[label]['full_checkpoint_tensor_hash']
        assert tensor_hash(actors[label].state_dict()) == frozen[label]['actor_hash']
    assert all(digest(Path(path)) == value for path, value in protected.items())
    assert all(digest(Path(source[key])) == source['sha256'] for source in sources for key in ('path', 'snapshot'))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), settings=results,
        episodes_executed=32, teacher_stand_passes=sum(row['standing_passes'] for row in results.values()),
        native_admitted_samples=sum(row['admitted_samples'] for name, row in results.items() if name.endswith('native')),
        gpu_validation_samples=sum(row['admitted_samples'] for name, row in results.items() if name.endswith('gpu')),
        actor_live_predictions=sum(row['actor_live_predictions'] for row in results.values()),
        live_control_intervals=sum(row['live_control_intervals'] for row in results.values()),
        plant_live_physics_steps=sum(row['plant_live_physics_steps'] for row in results.values()),
        private_teacher_prediction_physics_steps=sum(row['private_teacher_prediction_physics_steps'] for row in results.values()),
        frozen=frozen, learning_updates=0, learning_optimizers_created=0, new_amp_clips=0, arena_changed=False,
        inputs_and_sources_unchanged=True, actors_and_normalizers_unchanged=True))
    save(args.out/'completion-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_input_count=len(protected), source_count=len(sources), inputs_and_sources_verify=True,
        frozen_actors_verify=True, output_hashes={str(path): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))


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
