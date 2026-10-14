"""Native frozen-policy probe with a single velocity clipping intervention."""
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
from .audit_stand_current_boundary import partitions, error_summary
from .audit_stand_student_errors import load_actor, normalizers
from .diagnose_stand_foundations import DiagnosticNativeEngine
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash


PRIOR = Path('logs/mimickit-training/stand-current-boundary-20261003-01')
MODELS = Path('logs/mimickit-training/stand-live-recovery-supervision-20261003-01')
FRAMES = 36
VELOCITY = ((9, 15), (54, 93))
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
    assert np.array_equal(q, qpos, equal_nan=True) and np.array_equal(v, qvel, equal_nan=True), 'Teacher changed live state'
    assert task.engine.policy_control.cursor == cursor
    assert torch.equal(pending_commands(task.engine.policy_control), pending)
    assert torch.equal(task.engine.native_ctrl, control)
    assert np.array_equal(cpu(task.observations()), cpu(observation), equal_nan=True), 'Teacher changed policy inputs'


@torch.no_grad()
def capture(rig, reference, actor, velocity_unclipped, historical, out):
    backend = "native"
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
        standardized = (observation-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
        policy_input = actor.obs_norm.normalize(observation)
        would_clip = standardized.abs()>10.
        applied_clip = would_clip.clone()
        if velocity_unclipped:
            for begin, end in VELOCITY:
                policy_input[:, begin:end] = standardized[:, begin:end]
                applied_clip[:, begin:end] = False
        raw_mean = actor.mean(actor.actor(policy_input))
        raw_command = actor.action_norm.unnormalize(raw_mean).clamp(-1., 1.)
        finite_input = cpu(torch.isfinite(observation).all(-1) & torch.isfinite(standardized).all(-1)
            & torch.isfinite(policy_input).all(-1) & pre_state['finite_state'])
        finite_output = cpu(torch.isfinite(raw_mean).all(-1) & torch.isfinite(raw_command).all(-1))
        command_valid = finite_input & finite_output & (np.abs(cpu(raw_command))<=1.).all(-1)
        integrated = active & command_valid
        nonfinite_failure = active & ~(finite_input & finite_output)
        action = raw_command.clone()
        action[torch.tensor(~integrated, device=device)] = 0.
        task.engine.active = integrated.copy()
        teacher_command = np.zeros((8, 33), dtype=np.float32)
        prediction_steps = np.zeros(8, dtype=np.int64)
        succeeded, private_warning, private_finite = (np.zeros(8, dtype=bool) for _ in range(3))
        # Query each live world independently; failed labels must not terminate the actor's trial.
        controller.state_reader = lambda: (qpos, qvel)
        attempted_query = integrated.copy()
        for case in np.flatnonzero(attempted_query):
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
        before_time = np.asarray([data.time for data in task.engine.datas]) if backend == 'native' else None
        _, _, done, _ = task.step(action)
        held = task.engine.native_ctrl[:, torch.as_tensor(rig.actuators, device=device)]
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings | (active & ~command_valid), device=device)] = DoneFlags.FAIL.value
        flags, state = gates(task)
        after_q, after_v = reader()
        physics_steps = (np.rint((np.asarray([data.time for data in task.engine.datas])-before_time)/rig.model.opt.timestep).astype(int)
            if backend == 'native' else active.astype(int)*rig.decimation)
        finite_label = succeeded & (np.abs(teacher_command)<=1.).all(-1) & private_finite & ~private_warning & cpu(pre_state['finite_state'])
        row = dict(alive_before=active.copy(), integrated_before=integrated, command_valid=command_valid,
            nonfinite_failure=nonfinite_failure, finite_input=finite_input, finite_output=finite_output,
            policy_input=cpu(policy_input), raw_mean=cpu(raw_mean), raw_command=cpu(raw_command),
            original_would_clip_features=cpu(would_clip), applied_clipped_features=cpu(applied_clip),
            applied_clip_fraction=cpu(applied_clip.float().mean(-1)), teacher_query_attempted=attempted_query, obs=cpu(observation), qpos_before=qpos.copy(), qvel_before=qvel.copy(),
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
    assert np.isfinite(arrays['command']).all() and (np.abs(arrays['command'])<=1.).all()
    assert np.array_equal(arrays['nonfinite_failure'], live & ~(arrays['finite_input'] & arrays['finite_output']))
    comparable = ('alive_before', 'done', 'balance', 'posture', 'pelvis_height_m', 'pelvis_tilt_degrees',
                  'foot_loads_n', 'command', 'obs_clip_fraction', 'outside_clip10_fraction', 'solver_warning', 'time_s')
    with np.load(historical/'trace.npz', allow_pickle=False) as old:
        replay = {}
        for key in comparable:
            prior_prefix = old[key][:len(rows)]
            matching_shape = arrays[key].shape == prior_prefix.shape
            replay[key] = dict(exact=bool(np.array_equal(arrays[key], prior_prefix)), same_shape=matching_shape,
                max_absolute_difference=finite_number(np.max(np.abs(arrays[key].astype(float)-prior_prefix.astype(float))))
                if matching_shape else None)
        if not velocity_unclipped:
            assert all(value['exact'] for value in replay.values()), 'Baseline native prefix differs from Phase 40'
    out.mkdir()
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', private_reports)
    save(out/'teacher-query-errors.json', query_errors)
    report = dict(backend=backend, velocity_unclipped=velocity_unclipped, cases=8, requested_control_intervals=FRAMES, requested_horizon_seconds=FRAMES*task.dt, task_episode_seconds=12.,
        live_control_intervals=int(live.sum()), plant_live_physics_steps=int(arrays['plant_live_physics_steps'].sum()),
        teacher_private_prediction_physics_steps=int(arrays['teacher_private_prediction_physics_steps'].sum()),
        teacher_queries=int(arrays['teacher_query_attempted'].sum()),
        integrated_control_intervals=int(arrays['integrated_before'].sum()), nonfinite_failures=int(arrays['nonfinite_failure'].sum()), teacher_query_failures=len(query_errors),
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



def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def measures(trace, mask, contract):
    selected = mask & trace['alive_before']
    if not selected.any():
        return dict(samples=0)
    valid = selected & trace['diagnostic_label_numerically_valid'] & trace['command_valid']
    commands = selected & trace['command_valid']
    states = selected & trace['integrated_before'] & trace['finite_state']
    error = trace['command']-trace['teacher_only_command']
    return dict(samples=int(selected.sum()), integrated_intervals=int((selected & trace['integrated_before']).sum()),
        nonfinite_failures=int((selected & trace['nonfinite_failure']).sum()),
        valid_teacher_queries=int(valid.sum()), unavailable_teacher_queries=int(selected.sum()-valid.sum()),
        original_would_clip_fraction=float(trace['original_would_clip_features'][selected].mean()),
        applied_clip_fraction=float(trace['applied_clipped_features'][selected].mean()),
        command_saturation_fraction=float((np.abs(trace['command'][commands])>=.98).mean()) if commands.any() else None,
        angular_speed_mean_rad_s=finite_number(trace['angular_speed_rad_s'][states].mean()) if states.any() else None,
        angular_speed_max_rad_s=finite_number(trace['angular_speed_rad_s'][states].max()) if states.any() else None,
        command_error=error_summary(error,trace['applied_clipped_features'],valid,np.asarray(contract['torque_scale_nm']),contract['actuator_names']))


def analyse(trace, contract, out=None):
    dt, live = contract['control_timestep'], trace['alive_before']
    valid = live & trace['diagnostic_label_numerically_valid'] & trace['command_valid']
    error = np.where(valid[...,None],trace['command']-trace['teacher_only_command'],0.)
    masks = partitions(trace)
    diagnostics = dict(command_error=error,valid_label=valid,
        frame_command_rmse=np.sqrt(np.mean(error.astype(np.float64)**2,axis=-1)),
        frame_torque_rmse_nm=np.sqrt(np.mean((error.astype(np.float64)*np.asarray(contract['torque_scale_nm']))**2,axis=-1)),
        **{f'subset_{key}':mask for key,mask in masks.items()})
    cases=[]
    for case in range(8):
        selected=np.zeros_like(live)
        selected[:,case]=live[:,case]
        indices=np.flatnonzero(live[:,case])
        end=int(indices[-1])
        first_losses={}
        for name,flags in dict(angular=trace['balance_gate_angular_speed'],balance=trace['balance'],
                               posture=trace['posture'],combined=trace['balance'] & trace['posture']).items():
            violations=np.flatnonzero(live[:,case] & trace['integrated_before'][:,case] & ~flags[:,case])
            first_losses[name]=float(trace['time_s'][violations[0]]) if len(violations) else None
        validity=live[:,case] & trace['integrated_before'][:,case] & trace['balance'][:,case] & trace['posture'][:,case]
        state_mask=live[:,case] & trace['integrated_before'][:,case] & trace['finite_state'][:,case]
        failure_reason='nonfinite_input_or_command' if trace['nonfinite_failure'][end,case] else \
            'native_solver_warning' if trace['solver_warning'][end,case] else 'physical_FAIL' if trace['done'][end,case]==1 else 'budget_stop'
        duration=float(trace['issue_time_s'][end] if not trace['integrated_before'][end,case] else trace['time_s'][end])
        growth=finite_number((trace['angular_speed_rad_s'][end,case]-trace['pre_angular_speed_rad_s'][0,case])/duration) if duration>0 else None
        cases.append(dict(case=case,captured_intervals=len(indices),integrated_intervals=int(trace['integrated_before'][live[:,case],case].sum()),
            terminal_flag=int(trace['done'][end,case]),terminal_reason=failure_reason,
            capture_duration_seconds=duration,stopped_by_budget=bool(len(indices)==FRAMES and trace['done'][end,case]==0),
            first_departures_ending_time_s=first_losses,
            longest_balanced_valid_posture_seconds=longest(validity)*dt,
            total_balanced_valid_posture_seconds=int(validity.sum())*dt,
            angular_speed_end_rad_s=finite_number(trace['angular_speed_rad_s'][end,case]),
            angular_speed_peak_rad_s=finite_number(trace['angular_speed_rad_s'][state_mask,case].max()) if state_mask.any() else None,
            angular_speed_growth_rad_s2=growth,
            partitions={name:measures(trace,selected & mask,contract) for name,mask in masks.items()}))
    temporal=[]
    for frame in range(len(live)):
        selected=np.zeros_like(live)
        selected[frame]=live[frame]
        temporal.append(dict(frame=frame,issue_time_s=float(trace['issue_time_s'][frame]),ending_time_s=float(trace['time_s'][frame]),
            **measures(trace,selected,contract)))
    pooled={name:measures(trace,mask,contract) for name,mask in masks.items()}
    equal_pose={}
    for name in masks:
        rows=[row['partitions'][name] for row in cases if row['partitions'][name]['samples']]
        errors=[row['command_error'] for row in rows if row['command_error']['samples']]
        equal_pose[name]=dict(poses=len(rows),valid_label_poses=len(errors),
            normalized_command_rmse=float(np.mean([row['normalized_command_rmse'] for row in errors])) if errors else None,
            torque_rmse_nm=float(np.mean([row['torque_rmse_nm'] for row in errors])) if errors else None,
            applied_clip_fraction=float(np.mean([row['applied_clip_fraction'] for row in rows])) if rows else None)
    report=dict(cases=cases,temporal=temporal,pooled_partitions=pooled,equal_pose_partition_means=equal_pose,
        mean_first_departure_seconds={key:float(np.mean([row['first_departures_ending_time_s'][key] for row in cases if row['first_departures_ending_time_s'][key] is not None]))
            if any(row['first_departures_ending_time_s'][key] is not None for row in cases) else None for key in ('angular','balance','posture','combined')},
        mean_longest_balanced_valid_posture_seconds=float(np.mean([row['longest_balanced_valid_posture_seconds'] for row in cases])),
        physical_failures=sum(row['terminal_flag']==1 for row in cases),budget_stops=sum(row['stopped_by_budget'] for row in cases),
        nonfinite_failures=int(trace['nonfinite_failure'].sum()),native_solver_warnings=int((trace['solver_warning'] & live).sum()),
        scope='Matched frozen native-only prefix probe; no training or learned Stand admission',
        teacher_scope='Counterfactual unchanged private proposals, not executed or physically admitted',
        clipping_scope='Would-clip always uses original clip10; applied-clip excludes velocity features only in intervention',
        command_saturation_threshold=.98)
    if out is not None:
        np.savez_compressed(out/'diagnostics.npz',**diagnostics)
        save(out/'analysis.json',report)
    return report,diagnostics


def paired(baseline, intervention):
    rows=[]
    for before,after in zip(baseline['cases'],intervention['cases']):
        assert before['case']==after['case']
        a,b=before['first_departures_ending_time_s']['combined'],after['first_departures_ending_time_s']['combined']
        rows.append(dict(case=before['case'],baseline_first_departure_seconds=a,intervention_first_departure_seconds=b,
            first_departure_difference_seconds=b-a if a is not None and b is not None else None,
            baseline_longest_valid_seconds=before['longest_balanced_valid_posture_seconds'],
            intervention_longest_valid_seconds=after['longest_balanced_valid_posture_seconds'],
            longest_valid_difference_seconds=after['longest_balanced_valid_posture_seconds']-before['longest_balanced_valid_posture_seconds'],
            baseline_peak_angular_speed_rad_s=before['angular_speed_peak_rad_s'],intervention_peak_angular_speed_rad_s=after['angular_speed_peak_rad_s'],
            baseline_terminal=before['terminal_reason'],intervention_terminal=after['terminal_reason']))
    return dict(cases=rows,later_departures=sum(row['first_departure_difference_seconds']>0 for row in rows if row['first_departure_difference_seconds'] is not None),
        earlier_departures=sum(row['first_departure_difference_seconds']<0 for row in rows if row['first_departure_difference_seconds'] is not None),
        unchanged_departures=sum(row['first_departure_difference_seconds']==0 for row in rows if row['first_departure_difference_seconds'] is not None),
        intervention_no_departure_cases=sum(row['intervention_first_departure_seconds'] is None for row in rows),
        mean_valid_duration_difference_seconds=float(np.mean([row['longest_valid_difference_seconds'] for row in rows])),
        matched_scope='Same frozen actor and initial state; only velocity clipping rule differs')


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    protected,sources,contract=archive_inputs(args.out)
    rig=Rig.load(MODELS/'dummy.xml',policy_exclude=())
    reference=MODELS/'stationary_reference.npz'
    assert digest(rig.path)==contract['model_sha256'] and digest(reference)==contract['reference_sha256']
    assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    save(args.out/'protocol.json',dict(inputs=protected,sources=sources,contract=contract,prior=str(PRIOR),models=str(MODELS),
        actors=['control','augmented'],conditions=['baseline','velocity-unclipped'],backend='native',total_episodes=32,cases_per_setting=8,
        max_control_intervals=FRAMES,max_seconds=FRAMES*contract['control_timestep'],task_episode_seconds=12.,
        velocity_feature_slices=VELOCITY,normalization='Exact frozen Phase32 means/std/counts; baseline clip10 everywhere; intervention velocity 9:15 and 54:93 raw, all other features clip10',
        intervention='Only diagnostic inference input clipping; saved policies/global normalizer untouched',
        teacher='Unchanged predictive private teacher with actual delayed queue; never drives live plant',
        failure_handling='Keep nonfinite input/output and physical/solver failures; no integration of invalid commands; no omitted cases',
        training_updates=0,optimizers_created=0,normalizer_samples_added=0,new_amp_clips=0,dataset_admissions=0,arena_changed=False))
    results,frozen,comparisons={}, {}, {}
    for label in ('control','augmented'):
        actor,full_state=load_actor(MODELS/label)
        before,counts=tensor_hash(actor.state_dict()),normalizers(actor)
        assert before==read(MODELS/label/'report.json')['final_actor_hash']
        traces={}
        for condition in ('baseline','velocity-unclipped'):
            name=f'{label}-{condition}'
            print(dict(starting=name,backend='native',max_intervals=FRAMES,frozen=True),flush=True)
            trace,capture_report=capture(rig,reference,actor,condition=='velocity-unclipped',PRIOR/f'{label}-native',args.out/name)
            analysis,_=analyse(trace,contract,args.out/name)
            assert tensor_hash(actor.state_dict())==before and normalizers(actor)==counts
            results[name]=dict(capture=capture_report,analysis=analysis)
            traces[condition]=trace
            save(args.out/'progress.json',dict(completed_settings=len(results),total_settings=4,last_setting=name))
            print(dict(completed=name,first_departure=analysis['mean_first_departure_seconds'],physical_failures=analysis['physical_failures'],nonfinite_failures=analysis['nonfinite_failures']),flush=True)
        for key in ('initial_qpos','initial_qvel'):
            assert np.array_equal(traces['baseline'][key],traces['velocity-unclipped'][key])
        assert np.array_equal(traces['baseline']['obs'][0],traces['velocity-unclipped']['obs'][0])
        assert np.array_equal(traces['baseline']['command'][0],traces['velocity-unclipped']['command'][0])
        comparisons[label]=paired(results[f'{label}-baseline']['analysis'],results[f'{label}-velocity-unclipped']['analysis'])
        frozen[label]=dict(actor_tensor_hash_before=before,actor_tensor_hash_after=tensor_hash(actor.state_dict()),
            full_saved_checkpoint_tensor_hash=tensor_hash(full_state),normalizer_counts_before=counts,normalizer_counts_after=normalizers(actor),portable_sidecar_match=True)
    assert all(digest(Path(path))==expected for path,expected in protected.items())
    assert all(digest(Path(source[key]))==source['sha256'] for source in sources for key in ('path','snapshot'))
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),settings=results,frozen=frozen,paired_comparison=comparisons,
        episodes_executed=32,live_control_intervals=sum(row['capture']['live_control_intervals'] for row in results.values()),
        integrated_control_intervals=sum(row['capture']['integrated_control_intervals'] for row in results.values()),
        plant_live_physics_steps=sum(row['capture']['plant_live_physics_steps'] for row in results.values()),
        teacher_queries=sum(row['capture']['teacher_queries'] for row in results.values()),
        private_teacher_prediction_physics_steps=sum(row['capture']['teacher_private_prediction_physics_steps'] for row in results.values()),
        training_updates=0,optimizers_created=0,normalizer_samples_added=0,new_amp_clips=0,dataset_admissions=0,arena_changed=False,
        protected_inputs_unchanged=True,sources_and_snapshots_unchanged=True,frozen_actors_and_normalizers_verified=True)
    save(args.out/'report.json',report)
    save(args.out/'completion-verification.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_input_count=len(protected),source_count=len(sources),inputs_and_sources_unchanged=True,episodes_executed=32,
        output_hashes={str(path):digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))
    print(dict(completed=True,episodes=32,output=str(args.out)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
