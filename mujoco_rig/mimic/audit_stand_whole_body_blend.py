"""Frozen whole-body actor/teacher blends; no learning or deployment."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_critic import effective_config
from .audit_stand_failures import digest, read, save
from .audit_stand_first_torque import load_trace
from .audit_stand_rewards import trial
from .audit_stand_sampling import SamplingNativeEngine
from .audit_stand_three_motor_feedback import MixedActor, feedback_metrics, normalizer_counts, recording_task
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


FRACTIONS = (0., .01, .05, .10)
PHYSICAL_FIELDS = ('alive_before','obs','next_obs','qpos','qvel','action','done','balance','posture',
    'pelvis_height_m','pelvis_tilt_degrees','foot_loads_n','solver_warning','time_s',
    'initial_qpos','initial_qvel','disc_obs','task_reward')


class WholeBodyActor(MixedActor):
    """Reuse queue/state guards and blend all 33 recorded command coordinates."""
    def __init__(self, actor, task, fraction):
        super().__init__(actor,task,list(range(33)),True)
        self.fraction = fraction

    @torch.no_grad()
    def __call__(self, observation):
        teacher = super().__call__(observation)
        proposal = torch.from_numpy(self.records[-1]['actor_only_command'])
        blended = teacher+np.float32(self.fraction)*(proposal-teacher)
        assert torch.isfinite(blended).all() and (blended.abs() <= 1).all()
        self.records[-1].update(mixed_command=blended.numpy().copy(),
            executed_minus_teacher_command=(blended-teacher).numpy().copy())
        return blended


def command_error_metrics(arrays, scale):
    rows = []
    for case in range(8):
        live = arrays['alive_before'][:,case]
        error = arrays['executed_minus_teacher_command'][live,case].astype(np.float64)
        actor_error = (arrays['actor_only_command'][live,case].astype(np.float64)
                       -arrays['teacher_only_command'][live,case].astype(np.float64))
        rows.append(dict(case=case,normalized_command_rmse=float(np.sqrt(np.mean(error**2))),
            normalized_command_max_absolute_error=float(np.abs(error).max()),
            torque_rmse_nm=float(np.sqrt(np.mean((error*scale)**2))),
            actor_minus_teacher_command_rmse=float(np.sqrt(np.mean(actor_error**2)))))
    return dict(cases=rows,means={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key!='case'})


def execute(task,scorer,actor,contract,fraction,historical,out):
    wrapper = WholeBodyActor(actor,task,fraction)
    out.mkdir()
    report = trial(task,scorer,wrapper,'trained_actor',12.,out)
    arrays = load_trace(out/'trace.npz')
    for records in (wrapper.records,task.measurement_records,task.reward_records):
        arrays.update({key:np.stack([row[key] for row in records]) for key in records[0]})
    arrays['held_torque_nm'] = np.stack(task.engine.held_torques)
    live = arrays['alive_before']
    expected_command = (arrays['teacher_only_command']+np.float32(fraction)
                        *(arrays['actor_only_command']-arrays['teacher_only_command']))
    assert np.array_equal(arrays['action'][live],expected_command[live])
    assert np.array_equal(arrays['action'][live],arrays['mixed_command'][live])
    assert np.array_equal(arrays['obs'][:,:,95:161],arrays['pending_commands_oldest_first'])
    assert (np.abs(arrays['action']) <= 1).all()
    scale = np.asarray(contract['torque_scale_nm'],dtype=np.float32)
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2:] = arrays['action'][:-2].clip(-1,1)*scale
    assert np.array_equal(expected,arrays['held_torque_nm'])
    assert all(np.array_equal(arrays[key],historical[key]) for key in ('initial_qpos','initial_qvel'))
    assert all(np.array_equal(arrays[key][:2],historical[key][:2]) for key in ('qpos','qvel'))
    replay = None
    if fraction==0:
        replay = {key:bool(np.array_equal(arrays[key],historical[key])) for key in PHYSICAL_FIELDS}
        assert all(replay.values()), 'Full-teacher physical trajectory must replay exactly'
        assert report['standing_passes']==8 and report['survivors']==8
    assert not arrays['solver_warning'].any()
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(out/'trace.npz',**arrays)
    save(out/'teacher-private-diagnostics.json',wrapper.teacher.reports)
    intervals = int(live.sum())
    report.update(actor_fraction=fraction,teacher_fraction=1-fraction,
        feedback=feedback_metrics(arrays,task.dt),command_error=command_error_metrics(arrays,scale),
        full_teacher_exact_physical_replay=replay,whole_body_blend_verified=True,
        pending_mixed_commands_observed_exactly=True,actual_delayed_torques_verified=True,
        reset_and_zero_torque_prefix_identical=True,teacher_did_not_mutate_plant_or_queue=True,
        plant_control_intervals=intervals,plant_physics_steps=intervals*task.rig.decimation,
        private_teacher_evaluations=len(wrapper.teacher.reports),
        private_teacher_prediction_physics_steps=wrapper.prediction_physics_steps,
        private_refinement_advances_time=False,
        critic_scope='Frozen actor critic; all blended trajectories are off-policy diagnostics only.',
        scope='Teacher-assisted control tolerance along frozen actor error, not learned or autonomous Stand.')
    save(out/'report.json',report)
    print(dict(actor_fraction=fraction,feedback=report['feedback']['means'],
               command_error=report['command_error']['means'],plant_control_intervals=intervals),flush=True)
    return report,arrays


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    mp_util.init(0,1,'cpu',0)
    torch.set_num_threads(1)
    prior = read(args.prior/'protocol.json')
    complete = read(args.prior/'completion-verification.json')
    protected = dict(prior['inputs'])
    protected.update(complete['output_hashes'])
    protected[str(args.prior/'completion-verification.json')] = digest(args.prior/'completion-verification.json')
    paths = [Path(__file__)]+[Path(s['path']) for s in prior['sources']]
    for s in prior['sources']:
        for key in ('path','snapshot'): protected[s[key]] = s['sha256']
    for path in paths+[args.teacher/'trace.npz',args.teacher/'report.json']:
        current = digest(path)
        assert str(path) not in protected or protected[str(path)]==current
        protected[str(path)] = current
    assert all(digest(Path(path))==value for path,value in protected.items()), 'Historical source/input changed'
    historical = load_trace(args.teacher/'trace.npz')
    assert read(args.teacher/'report.json')['standing_passes']==8
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'sources').mkdir()
    sources = []
    for path in paths:
        snapshot = args.out/'sources'/path.name
        assert not snapshot.exists()
        shutil.copy2(path,snapshot)
        sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=digest(path)))
    save(args.out/'protocol.json',dict(inputs=protected,sources=sources,prior=str(args.prior),
        runs=str(args.runs),teacher=str(args.teacher),contract=prior['contract'],fractions=list(FRACTIONS),
        settings=8,cases_per_setting=8,total_episodes=64,max_seconds=12.,max_control_intervals_per_episode=720,
        command='teacher + actor_fraction * (actor - teacher); float32; all 33 channels',
        teacher_controller='Existing unconstrained full-body controller, actual mixed-state queue prediction',
        full_teacher_replay_fields=list(PHYSICAL_FIELDS),training_updates=0,gaussian_trials=0,arena_changed=False))
    results, frozen, zero_control = {}, {}, None
    for weight in ('0','0.25'):
        run_path = args.runs/f'weight-{weight}'
        rp = read(run_path/'protocol.json')
        state = torch.load(run_path/'model.amp.pt',map_location='cpu',weights_only=True)
        validate_sidecar(torch.load(run_path/'model.pt',map_location='cpu',weights_only=True),state)
        rig = Rig.load(run_path/'dummy.xml',policy_exclude=())
        contract = read(run_path/'contract.json')
        assert digest(rig.path)==contract['model_sha256']==prior['contract']['model_sha256']
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
        previous_frozen = read(args.prior/'report.json')['frozen'][weight]
        assert before==previous_frozen['tensor_hashes_before']
        counts = normalizer_counts(scorer,actor)
        assert counts==previous_frozen['normalizer_counts_before']
        for fraction in FRACTIONS:
            label = f'actor-{weight}-fraction-{fraction:g}'
            report,arrays = execute(task,scorer,actor,contract,fraction,historical,args.out/label)
            results[label] = report
            if fraction==0:
                if zero_control is None:
                    zero_control = {key:arrays[key].copy() for key in PHYSICAL_FIELDS+('held_torque_nm',)}
                else:
                    assert all(np.array_equal(arrays[key],value) for key,value in zero_control.items())
            after = dict(scorer=tensor_hash(scorer.state_dict()),actor=tensor_hash(actor.state_dict()))
            assert after==before and normalizer_counts(scorer,actor)==counts
            assert not any(row['pending'] for row in counts.values())
            save(args.out/'progress.json',dict(settings_completed=len(results),total_settings=8,last_setting=label))
        frozen[weight] = dict(tensor_hashes_before=before,tensor_hashes_after=after,
            normalizer_counts_before=counts,normalizer_counts_after=normalizer_counts(scorer,actor))
    assert all(digest(Path(path))==value for path,value in protected.items())
    assert all(digest(Path(s[key]))==s['sha256'] for s in sources for key in ('path','snapshot'))
    save(args.out/'report.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),settings=results,frozen=frozen,
        episodes_executed=64,zero_fraction_controls_physically_identical=True,
        plant_control_intervals=sum(r['plant_control_intervals'] for r in results.values()),
        private_teacher_prediction_physics_steps=sum(r['private_teacher_prediction_physics_steps'] for r in results.values()),
        protected_inputs_unchanged=True,sources_and_snapshots_unchanged=True,frozen_tensors_unchanged=True,
        normalizer_samples_added=0,solver_warnings=0,training_updates=0,gaussian_trials=0,arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,default=Path('logs/mimickit-training/stand-conditional-feedback-20261002-01'))
    parser.add_argument('--runs',type=Path,default=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01'))
    parser.add_argument('--teacher',type=Path,default=Path('logs/mimickit-training/stand-reward-audit-20261002-01/teacher'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir(): save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
