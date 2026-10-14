"""Frozen actor feedback with exactly three live teacher command substitutions."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_critic import effective_config
from .audit_stand_failures import digest, longest, read, save
from .audit_stand_first_torque import load_trace, verify_endpoint
from .audit_stand_rewards import trial
from .audit_stand_sampling import SamplingNativeEngine
from .audit_stand_three_motors import REPLACED_MOTORS
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


def recording_task(weight):
    from .angular_speed_reward import task_factory

    class RecordedTask(task_factory(weight, record_components=True)):
        def __init__(self, *args, **kwargs):
            self.measurement_records = []
            super().__init__(*args, **kwargs)

        def reset(self, env_ids=None):
            self.measurement_records.clear()
            return super().reset(env_ids)

        def step(self, action):
            result = super().step(action)
            state = self.standing_measurements()
            row = {key:value.numpy().copy() for key,value in state.items()}
            row.update({f'balance_gate_{key}':value.numpy().copy()
                        for key,value in self.balance_criteria(state).items()})
            row.update({f'posture_gate_{key}':value.numpy().copy()
                        for key,value in self.posture_criteria(state).items()})
            row.update(root_linear_velocity_world_m_s=self.engine.get_root_vel(0).numpy().copy(),
                       root_angular_velocity_world_rad_s=self.engine.get_root_ang_vel(0).numpy().copy())
            self.measurement_records.append(row)
            return result

    return RecordedTask


class MixedActor:
    """Preserve actor proposals and evaluate the existing teacher on private states."""
    def __init__(self, actor, task, indices, assisted):
        self.actor, self.task = actor, task
        self.indices, self.assisted = indices, assisted
        self.teacher = PredictiveStandController(task)
        self.records = []
        self.prediction_physics_steps = 0

    @torch.no_grad()
    def __call__(self, observation):
        task = self.task
        queue = task.engine.policy_control
        pending = torch.cat([queue.pending[(queue.cursor+i) % len(queue.pending)]
                             for i in range(len(queue.pending))], dim=-1)
        assert torch.equal(observation[:,95:161], pending), 'Observation has wrong pending controls'
        proposal = self.actor(observation)
        qpos = np.stack([data.qpos.copy() for data in task.engine.datas])
        qvel = np.stack([data.qvel.copy() for data in task.engine.datas])
        control = task.engine.native_ctrl.clone()
        teacher = self.teacher.command(task.engine.active)
        assert np.array_equal(qpos, np.stack([data.qpos for data in task.engine.datas]))
        assert np.array_equal(qvel, np.stack([data.qvel for data in task.engine.datas]))
        assert torch.equal(control, task.engine.native_ctrl)
        assert torch.equal(pending, torch.cat([queue.pending[(queue.cursor+i) % len(queue.pending)]
                                              for i in range(len(queue.pending))], dim=-1))
        mixed = proposal.clone()
        if self.assisted:
            mixed[:,self.indices] = teacher[:,self.indices]
        self.prediction_physics_steps += int(task.engine.active.sum())*len(queue.pending)*task.rig.decimation
        self.records.append(dict(actor_only_command=proposal.numpy().copy(),
            teacher_only_command=teacher.numpy().copy(), mixed_command=mixed.numpy().copy(),
            pending_commands_oldest_first=pending.numpy().copy()))
        return mixed


def normalizer_counts(scorer, actor):
    norms = dict(scorer_obs=scorer._obs_norm, scorer_disc=scorer._disc_obs_norm,
                 scorer_action=scorer._a_norm, actor_obs=actor.obs_norm, actor_action=actor.action_norm)
    return {key:dict(count=int(norm.get_count()), pending=int(getattr(norm, '_pending_count', 0)))
            for key,norm in norms.items()}


def gate_events(valid, times):
    lost = np.flatnonzero(~valid)
    recovery = np.flatnonzero(valid & (np.arange(len(valid)) > lost[0])) if len(lost) else []
    return dict(first_loss_seconds=float(times[lost[0]]) if len(lost) else None,
                first_recovery_after_loss_seconds=float(times[recovery[0]]) if len(recovery) else None,
                valid_intervals=int(valid.sum()), longest_valid_intervals=longest(valid),
                loss_count=int(np.sum(valid[:-1] & ~valid[1:]))+int(not valid[0]),
                recovery_count=int(np.sum(~valid[:-1] & valid[1:])))


def feedback_metrics(arrays, dt):
    rows = []
    gates = [key for key in arrays if key.startswith(('balance_gate_', 'posture_gate_'))]
    for case in range(8):
        n = int(arrays['alive_before'][:,case].sum())
        times = arrays['time_s'][:n]
        balanced, posture = arrays['balance'][:n,case], arrays['posture'][:n,case]
        rows.append(dict(case=case, total_balanced_seconds=float(balanced.sum()*dt),
            longest_balanced_seconds=longest(balanced)*dt,
            total_valid_posture_seconds=float(posture.sum()*dt),
            longest_balanced_and_posture_seconds=longest(balanced & posture)*dt,
            first_effect_balanced=bool(balanced[2]),
            first_effect_angular_speed_rad_s=float(np.linalg.norm(arrays['qvel'][2,case,3:6])),
            balance_events=gate_events(balanced,times),
            posture_events=gate_events(posture,times),
            gate_events={key:gate_events(arrays[key][:n,case],times) for key in gates}))
    keys = ('total_balanced_seconds','longest_balanced_seconds','total_valid_posture_seconds',
            'longest_balanced_and_posture_seconds','first_effect_angular_speed_rad_s')
    return dict(cases=rows, means={key:float(np.mean([row[key] for row in rows])) for key in keys})


def execute(task, scorer, actor, contract, assisted, previous, historical, out):
    wrapper = MixedActor(actor, task, [contract['actuator_names'].index(n) for n in REPLACED_MOTORS], assisted)
    out.mkdir()
    report = trial(task, scorer, wrapper, 'trained_actor', 12., out)
    arrays = load_trace(out/'trace.npz')
    for records in (wrapper.records, task.measurement_records, task.reward_records):
        arrays.update({key:np.stack([row[key] for row in records]) for key in records[0]})
    arrays['held_torque_nm'] = np.stack(task.engine.held_torques)
    alive = arrays['alive_before']
    assert np.array_equal(arrays['action'][alive],arrays['mixed_command'][alive])
    selected = wrapper.indices
    remaining = [i for i in range(33) if i not in selected]
    assert np.array_equal(arrays['action'][:,:,remaining][alive],arrays['actor_only_command'][:,:,remaining][alive])
    if assisted:
        assert np.array_equal(arrays['action'][:,:,selected][alive],arrays['teacher_only_command'][:,:,selected][alive])
    else:
        assert np.array_equal(arrays['action'][alive],arrays['actor_only_command'][alive])
    assert np.array_equal(arrays['obs'][:,:,95:161],arrays['pending_commands_oldest_first'])
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2:] = arrays['action'][:-2].clip(-1,1)*np.asarray(contract['torque_scale_nm'],dtype=np.float32)
    assert np.array_equal(expected, arrays['held_torque_nm']), 'Actual delayed torque mismatch'
    assert all(np.array_equal(arrays[key],previous[key]) for key in ('initial_qpos','initial_qvel'))
    replay = verify_endpoint({key:value[:3] for key,value in arrays.items()}, previous)
    assert all(row['exact'] for row in replay.values()), 'First effect must match Phase 27 exactly'
    for key in ('root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s','held_torque_nm'):
        assert np.array_equal(arrays[key][:3],previous[key]), key
    control_replay = None
    if not assisted:
        control_replay = {key:bool(np.array_equal(arrays[key],value)) for key,value in historical.items()}
        assert all(control_replay.values()), 'Unassisted Phase 24 replay differs'
    assert not arrays['solver_warning'].any()
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(out/'trace.npz', **arrays)
    save(out/'teacher-private-diagnostics.json', wrapper.teacher.reports)
    plant_intervals = int(alive.sum())
    report.update(assisted=assisted, replaced_motors=list(REPLACED_MOTORS) if assisted else [],
        feedback=feedback_metrics(arrays,task.dt), phase27_first_effect_exact_replay=replay,
        phase24_control_exact_replay=control_replay, command_substitution_verified=True,
        pending_mixed_commands_observed_exactly=True, actual_delayed_torques_verified=True,
        plant_control_intervals=plant_intervals, plant_physics_steps=plant_intervals*task.rig.decimation,
        private_teacher_evaluations=len(wrapper.teacher.reports),
        private_teacher_prediction_physics_steps=wrapper.prediction_physics_steps,
        private_teacher_refinement_advances_time=False, teacher_did_not_mutate_plant_or_queue=True,
        critic_scope='Frozen actor critic; assisted trajectories are off-policy diagnostics only.')
    save(out/'report.json', report)
    print(dict(assisted=assisted,feedback=report['feedback']['means'],
               plant_control_intervals=plant_intervals),flush=True)
    return report


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    mp_util.init(0,1,'cpu',0)
    torch.set_num_threads(1)
    prior = read(args.prior/'protocol.json')
    completed = read(args.prior/'completion-verification.json')
    protected = dict(prior['inputs'])
    protected.update(completed['output_hashes'])
    protected[str(args.prior/'completion-verification.json')] = digest(args.prior/'completion-verification.json')
    source_paths = [Path(__file__)]
    for s in prior['sources']:
        source_paths.append(Path(s['path']))
        for key in ('path','snapshot'): protected[s[key]] = s['sha256']
    audits = {}
    for weight in ('0','0.25'):
        audit = args.runs/f'audit-weight-{weight}'
        ap = read(audit/'protocol.json')
        audits[weight] = ap
        protected.update(ap['inputs'])
        for filename in ('protocol.json','report.json','deterministic-native/report.json','deterministic-native/trace.npz'):
            path = audit/filename
            protected[str(path)] = digest(path)
        for s in ap['sources']:
            source_paths.append(Path(s['path']))
            for key in ('path','snapshot'): protected[s[key]] = s['sha256']
    source_paths.extend(Path('mujoco_rig/mimic')/name for name in
        ('inverse_contact_control.py','stand_foundation_followup.py','policy.py','angular_speed_reward.py',
         'audit_stand_three_motors.py','audit_stand_first_torque.py'))
    source_paths = list({str(path.resolve()):path for path in source_paths}.values())
    for path in source_paths:
        current = digest(path)
        assert str(path) not in protected or protected[str(path)] == current
        protected[str(path)] = current
    assert all(digest(Path(path)) == value for path,value in protected.items()), 'Historical input/source changed'
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'sources').mkdir()
    sources = []
    for path in source_paths:
        snapshot = args.out/'sources'/path.name
        assert not snapshot.exists(), 'Source snapshot name collision'
        shutil.copy2(path,snapshot)
        sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=digest(path)))
    save(args.out/'protocol.json',dict(inputs=protected,sources=sources,prior=str(args.prior),
        runs=str(args.runs),replaced_motors=list(REPLACED_MOTORS),contract=prior['contract'],
        settings=4,cases_per_setting=8,total_episodes=32,max_seconds=12.,max_control_intervals_per_episode=720,
        intervention='Continuous live teacher correction of only three channels; frozen deterministic actor controls remaining 30.',
        private_teacher_scope='Original controller on private states in both arms; prediction steps separate from plant.',
        training_updates=0,gaussian_trials=0,arena_changed=False))
    results, frozen = {}, {}
    for weight in ('0','0.25'):
        run_path = args.runs/f'weight-{weight}'
        rp = read(run_path/'protocol.json')
        state = torch.load(run_path/'model.amp.pt',map_location='cpu',weights_only=True)
        validate_sidecar(torch.load(run_path/'model.pt',map_location='cpu',weights_only=True),state)
        rig = Rig.load(run_path/'dummy.xml',policy_exclude=())
        contract = read(run_path/'contract.json')
        assert digest(rig.path)==prior['contract']['model_sha256']==contract['model_sha256']
        assert digest(run_path/'stationary_reference.npz')==contract['reference_sha256']==prior['contract']['reference_sha256']
        assert not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
        assert contract['latency_steps']==2 and len(rig.actuators)==33
        task = recording_task(float(weight))(rig,run_path/'stationary_reference.npz',8,'cpu',seed=rp['seed'],
            reward_mode='reference_v9',episode_seconds=12.,engine_factory=SamplingNativeEngine,
            demonstrations=[Path(record['path']) for record in rp['demonstrations']],demonstration_rig=rig)
        task.mirror_demos = False
        assert task.dt==contract['control_timestep']
        scorer = build_agent(effective_config(run_path/'agent.yaml'),task,'cpu')
        scorer.load_state_dict(state)
        scorer.eval().requires_grad_(False)
        actor = StandPolicy(scorer).eval().requires_grad_(False)
        before = dict(scorer=tensor_hash(scorer.state_dict()),actor=tensor_hash(actor.state_dict()))
        assert before==audits[weight]['frozen_tensor_hashes']
        counts = normalizer_counts(scorer,actor)
        assert not any(row['pending'] for row in counts.values())
        historical = load_trace(args.runs/f'audit-weight-{weight}/deterministic-native/trace.npz')
        for assisted in (False,True):
            arm = 'three-teacher' if assisted else 'control'
            label = f'actor-{weight}-{arm}'
            previous = load_trace(args.prior/label/'trace.npz')
            results[label] = execute(task,scorer,actor,contract,assisted,previous,historical,args.out/label)
            assert before==dict(scorer=tensor_hash(scorer.state_dict()),actor=tensor_hash(actor.state_dict()))
            assert counts==normalizer_counts(scorer,actor)
            save(args.out/'progress.json',dict(settings_completed=len(results),total_settings=4,last_setting=label))
        frozen[weight] = dict(tensor_hashes_before=before,tensor_hashes_after=before,
                             normalizer_counts_before=counts,normalizer_counts_after=normalizer_counts(scorer,actor))
    assert all(digest(Path(path))==value for path,value in protected.items())
    assert all(digest(Path(s[key]))==s['sha256'] for s in sources for key in ('path','snapshot'))
    save(args.out/'report.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),settings=results,
        frozen=frozen,episodes_executed=32,plant_control_intervals=sum(r['plant_control_intervals'] for r in results.values()),
        private_teacher_prediction_physics_steps=sum(r['private_teacher_prediction_physics_steps'] for r in results.values()),
        protected_inputs_unchanged=True,sources_and_snapshots_unchanged=True,normalizer_samples_added=0,
        frozen_tensors_unchanged=True,solver_warnings=0,training_updates=0,gaussian_trials=0,arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,default=Path('logs/mimickit-training/stand-three-motor-substitution-20261002-01'))
    parser.add_argument('--runs',type=Path,default=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir(): save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
