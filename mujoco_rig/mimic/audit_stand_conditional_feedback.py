"""Frozen three-motor feedback with all other commands fixed inside the solve."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import mujoco
import numpy as np
import torch

from .audit_stand_critic import effective_config
from .audit_stand_failures import digest, read, save
from .audit_stand_first_torque import load_trace
from .audit_stand_rewards import trial
from .audit_stand_sampling import SamplingNativeEngine
from .audit_stand_three_motor_feedback import MixedActor, feedback_metrics, normalizer_counts, recording_task
from .audit_stand_three_motors import REPLACED_MOTORS
from .inverse_contact_control import inverse_contact_map, refine_forward_torque, solve_box
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


class ConditionalController(PredictiveStandController):
    """Use the admitted teacher objective with exactly three free coordinates."""
    def __init__(self, task, indices):
        super().__init__(task)
        self.indices = indices
        self.fixed = [i for i in range(len(self.actuators)) if i not in indices]
        self.actor_command = None

    def command(self, active):
        actor = self.actor_command
        assert actor is not None and actor.shape == (self.task.get_num_envs(),len(self.actuators))
        assert np.isfinite(actor).all() and (np.abs(actor) <= 1).all()
        controls = np.zeros_like(actor)
        queue = self.task.engine.policy_control
        qpos, qvel = self.state_reader()
        pending = queue.pending.detach().cpu().numpy()
        for world in np.flatnonzero(active):
            data = self.data[world]
            mujoco.mj_resetData(self.model,data)
            data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
            for lag in range(len(queue.pending)):
                data.ctrl[self.actuators] = pending[(queue.cursor+lag)%len(pending),world]*self.scale
                for _ in range(self.task.rig.decimation):
                    mujoco.mj_step(self.model,data)
            mujoco.mj_forward(self.model,data)
            error = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model,error,1.,data.qpos,self.target_qpos[world])
            frequency = 2*np.pi*1.
            desired = np.clip(frequency**2*error-2*frequency*data.qvel,-20.,20.)
            mapping, offset, *_ = inverse_contact_map(self.model,data,self.dofs,self.scale)
            reduced = self.tracking @ mapping
            source = data.ctrl[self.actuators]/self.scale
            regularizer = .15**2
            hessian = reduced.T@reduced+np.eye(len(source))*(regularizer+1e-7)
            linear = -reduced.T@(self.tracking@(desired-offset))-regularizer*source
            lower, upper = actor[world].astype(np.float64), actor[world].astype(np.float64)
            lower[self.indices], upper[self.indices] = -1., 1.
            assert np.array_equal(lower[self.fixed],upper[self.fixed])
            proposed, rank = solve_box(hessian,linear,lower,upper,source)
            if rank < 0 or not np.isfinite(proposed).all():
                raise RuntimeError('Conditional motor QP failed')
            assert np.array_equal(proposed[self.fixed],actor[world,self.fixed])
            control, acceleration, diagnostics = refine_forward_torque(self.model,data,
                self.actuators,self.dofs,self.scale,self.tracking,self.tracking@desired,
                proposed,lower,upper,iterations=2)
            assert np.array_equal(control[self.fixed],actor[world,self.fixed])
            assert np.all(control >= lower) and np.all(control <= upper)
            controls[world] = control
            self.reports.append(dict(world=int(world),free_motor_count=3,fixed_motor_count=30,
                fixed_commands_exact_before_and_after_refinement=True,
                tracking_residual=float(np.linalg.norm(self.tracking@(acceleration-desired))),**diagnostics))
        return torch.tensor(controls)


class ActorCommandBinding:
    """Provide the current actor proposal to the conditional controller once."""
    def __init__(self, actor, controller):
        self.actor, self.controller = actor, controller

    def __call__(self, observation):
        proposal = self.actor(observation)
        self.controller.actor_command = proposal.detach().cpu().numpy().copy()
        return proposal


def execute(task,scorer,actor,contract,conditional,historical,out):
    selected = [contract['actuator_names'].index(n) for n in REPLACED_MOTORS]
    remaining = [i for i in range(33) if i not in selected]
    wrapper = MixedActor(actor,task,selected,True)
    if conditional:
        wrapper.teacher = ConditionalController(task,selected)
        wrapper.actor = ActorCommandBinding(actor,wrapper.teacher)
    out.mkdir()
    report = trial(task,scorer,wrapper,'trained_actor',12.,out)
    arrays = load_trace(out/'trace.npz')
    for records in (wrapper.records,task.measurement_records,task.reward_records):
        arrays.update({key:np.stack([row[key] for row in records]) for key in records[0]})
    arrays['held_torque_nm'] = np.stack(task.engine.held_torques)
    live = arrays['alive_before']
    assert np.array_equal(arrays['action'][live],arrays['mixed_command'][live])
    assert np.array_equal(arrays['action'][:,:,remaining][live],arrays['actor_only_command'][:,:,remaining][live])
    assert np.array_equal(arrays['action'][:,:,selected][live],arrays['teacher_only_command'][:,:,selected][live])
    if conditional:
        assert np.array_equal(arrays['teacher_only_command'][:,:,remaining][live],arrays['actor_only_command'][:,:,remaining][live])
        assert np.array_equal(arrays['action'][live],arrays['teacher_only_command'][live])
    assert np.array_equal(arrays['obs'][:,:,95:161],arrays['pending_commands_oldest_first'])
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2:] = arrays['action'][:-2].clip(-1,1)*np.asarray(contract['torque_scale_nm'],dtype=np.float32)
    assert np.array_equal(expected,arrays['held_torque_nm'])
    assert all(np.array_equal(arrays[key],historical[key]) for key in ('initial_qpos','initial_qvel'))
    assert all(np.array_equal(arrays[key][:2],historical[key][:2]) for key in ('qpos','qvel'))
    replay = None
    if not conditional:
        replay = {key:bool(np.array_equal(arrays[key],value)) for key,value in historical.items()}
        assert all(replay.values()), 'Unconstrained Phase 28 replay differs'
    assert not arrays['solver_warning'].any()
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(out/'trace.npz',**arrays)
    save(out/'teacher-private-diagnostics.json',wrapper.teacher.reports)
    intervals = int(live.sum())
    report.update(conditional=conditional,replaced_motors=list(REPLACED_MOTORS),
        feedback=feedback_metrics(arrays,task.dt),phase28_unconstrained_exact_replay=replay,
        other_30_actor_commands_exact=True,selected_3_controller_commands_exact=True,
        conditional_fixed_commands_exact=conditional,pending_mixed_commands_observed_exactly=True,
        actual_delayed_torques_verified=True,reset_and_zero_torque_prefix_identical=True,
        teacher_did_not_mutate_plant_or_queue=True,plant_control_intervals=intervals,
        plant_physics_steps=intervals*task.rig.decimation,private_teacher_evaluations=len(wrapper.teacher.reports),
        private_teacher_prediction_physics_steps=wrapper.prediction_physics_steps,
        private_refinement_advances_time=False,
        controller_proposal_trace_field='teacher_only_command; conditional arm is a full proposal with 30 actor-fixed channels',
        critic_scope='Frozen actor critic; both assisted arms are off-policy diagnostics only.')
    save(out/'report.json',report)
    print(dict(conditional=conditional,feedback=report['feedback']['means'],plant_control_intervals=intervals),flush=True)
    return report


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
    for path in paths:
        current = digest(path)
        assert str(path) not in protected or protected[str(path)]==current
        protected[str(path)] = current
    assert all(digest(Path(path))==value for path,value in protected.items()), 'Historical source/input changed'
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'sources').mkdir()
    sources = []
    for path in paths:
        snapshot = args.out/'sources'/path.name
        assert not snapshot.exists()
        shutil.copy2(path,snapshot)
        sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=digest(path)))
    save(args.out/'protocol.json',dict(inputs=protected,sources=sources,prior=str(args.prior),
        runs=str(args.runs),contract=prior['contract'],replaced_motors=list(REPLACED_MOTORS),
        settings=4,cases_per_setting=8,total_episodes=32,max_seconds=12.,max_control_intervals_per_episode=720,
        conditional_bounds='30 lower=upper=current actor proposal; three selected channels free in [-1,1]',
        unchanged_controller='One Hz feedback; full inverse contact map; regularizer 0.15 squared; two forward refinements; actual two-command queue prediction',
        unassisted_reference='Reuse Phase 28 original controls, without additional plant episodes',
        training_updates=0,gaussian_trials=0,arena_changed=False))
    results, frozen = {}, {}
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
        historical_frozen = read(args.prior/'report.json')['frozen'][weight]
        assert before==historical_frozen['tensor_hashes_before']
        counts = normalizer_counts(scorer,actor)
        assert counts==historical_frozen['normalizer_counts_before']
        historical = load_trace(args.prior/f'actor-{weight}-three-teacher/trace.npz')
        for conditional in (False,True):
            arm = 'conditional' if conditional else 'unconstrained'
            label = f'actor-{weight}-{arm}'
            results[label] = execute(task,scorer,actor,contract,conditional,historical,args.out/label)
            after = dict(scorer=tensor_hash(scorer.state_dict()),actor=tensor_hash(actor.state_dict()))
            assert after==before and normalizer_counts(scorer,actor)==counts
            assert not any(row['pending'] for row in counts.values())
            save(args.out/'progress.json',dict(settings_completed=len(results),total_settings=4,last_setting=label))
        frozen[weight] = dict(tensor_hashes_before=before,tensor_hashes_after=after,
            normalizer_counts_before=counts,normalizer_counts_after=normalizer_counts(scorer,actor))
    assert all(digest(Path(path))==value for path,value in protected.items())
    assert all(digest(Path(s[key]))==s['sha256'] for s in sources for key in ('path','snapshot'))
    save(args.out/'report.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),settings=results,frozen=frozen,
        episodes_executed=32,plant_control_intervals=sum(r['plant_control_intervals'] for r in results.values()),
        private_teacher_prediction_physics_steps=sum(r['private_teacher_prediction_physics_steps'] for r in results.values()),
        protected_inputs_unchanged=True,sources_and_snapshots_unchanged=True,frozen_tensors_unchanged=True,
        normalizer_samples_added=0,solver_warnings=0,training_updates=0,gaussian_trials=0,arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,default=Path('logs/mimickit-training/stand-three-motor-feedback-20261002-01'))
    parser.add_argument('--runs',type=Path,default=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir(): save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
