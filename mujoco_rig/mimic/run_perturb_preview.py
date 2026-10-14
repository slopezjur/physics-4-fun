"""Bounded learned recovery continuation and matched Arena-preview assessments."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate

BASE = Path('logs/mimickit-training')
PARENT = BASE / 'stand-ball-stage33-focused-learning-20261007-02'
OLD = BASE / 'stand-ball-stage12-task-reward-20261006-01'
RETENTION = BASE / 'stand-ball-stage37-varied-retention-20261007-01'
PARENT_MODEL = PARENT / '301028/varied/final/model.pt'
OLD_MODEL = OLD / '301028/single_ball/final/model.pt'


def sealed(root, paths):
    proof = read(root / 'final-verification.json')
    assert proof['verified']
    known = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
    for path in paths:
        assert known[str(path.resolve())] == digest(path), path


@contextmanager
def repeated_curriculum():
    from .chest_ball_curriculum import curriculum
    from . import stand_ball_task, run_stand_ball_pilot as pilot
    from .representation_task import RepresentationMixin
    from .replicate_stand_representations import cpu
    settings = dict(launch_distance_m=.25, training_speed_m_s=1.6, launch_seconds=5, seconds=12)
    with curriculum(settings):
        base, evidence_base = stand_ball_task.StandBallAMPTask, pilot.TrainingEvidence

        class RepeatedTask(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.episode_seconds = 20.
                self.direction.zero_()
                self.direction[:, 0] = 1.
                self.onsets = torch.tensor([int(np.ceil(t/self.dt)) for t in (1, 3, 5, 9)], device=self.device)
                self.schedule_phase = torch.full((64,), -1, device=self.device, dtype=torch.long)
                self.launch_step = self.onsets[(self.ids//8) % 4]
                self.repeat_gap = int(np.ceil(4/self.dt))

            def reset(self, env_ids=None):
                result = super().reset(env_ids)
                if hasattr(self, 'schedule_phase'):
                    ids = self.ids if env_ids is None else env_ids
                    self.schedule_phase[ids] += 1
                    self.launch_step[ids] = self.onsets[(ids//8+self.schedule_phase[ids]) % 4]
                return result

            def step(self, action):
                enabled = (self.steps == self.launch_step) | (self.steps == self.launch_step+self.repeat_gap)
                if self.arm == 'quiet':
                    enabled[:] = False
                ids = self.ids[enabled]
                if len(ids):
                    target = self.engine.get_body_pos(0)[ids, self.rig.model.body('Chest').id-1]
                    direction = self.direction[ids]
                    gravity = torch.as_tensor(self.rig.model.opt.gravity, device=self.device, dtype=torch.float32)
                    d, v = self.launch_distance_m, self.launch_speed_m_s
                    self.engine.launch_ball(ids, target-d*direction, v*direction-.5*gravity*(d/v))
                self.launch_records.append(enabled.detach().clone())
                return RepresentationMixin.step(self, action)

        class Evidence(evidence_base):
            def step(self, action):
                first, phase = self.task.launch_step.clone(), self.task.schedule_phase.clone()
                result = super().step(action)
                self.rows[-1].update(first_launch=first, schedule_phase=phase)
                return result

            def flush(self, path):
                assert len(self.rows) == 64
                values = {k:cpu(torch.stack([r[k] for r in self.rows])) for k in self.rows[0]}
                self.rows.clear()
                assert all(np.isfinite(v).all() for v in values.values())
                expected_first = cpu(self.task.onsets)[(np.arange(64)[None]//8+values['schedule_phase']) % 4]
                assert np.array_equal(expected_first, values['first_launch'])
                expected = (values['steps'] == expected_first) | (values['steps'] == expected_first+self.task.repeat_gap)
                if self.task.arm == 'quiet':
                    expected[:] = False
                assert np.array_equal(expected, values['launched'])
                assert (values['position_error'] <= 2e-6).all() and (values['velocity_error'] <= 2e-6).all()
                assert (values['normal_impulse'] >= 0).all()
                assert ((values['steps'][1:] == values['steps'][:-1]+1) | (values['steps'][1:] == 0)).all()
                if self.task.arm == 'quiet':
                    assert not values['normal_impulse'].any() and not values['contact_impulse'].any()
                np.savez_compressed(path, **values)
                return dict(launches=int(expected.sum()),
                            contact_world_intervals=int(np.count_nonzero(values['normal_impulse'])),
                            task_reward_mean=float(values['task_reward'].mean()))

        with patch.object(stand_ball_task, 'StandBallAMPTask', RepeatedTask), patch.object(pilot, 'TrainingEvidence', Evidence):
            yield


def attach_rehearsal(agent, observations, actions):
    obs = torch.as_tensor(observations, device=agent._device)
    target = torch.as_tensor(actions, device=agent._device)
    original = agent._compute_actor_loss
    state = dict(calls=0, examples=0)
    index = torch.arange(256, device=agent._device)

    def predict(x):
        return agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(x)).mode)

    with torch.no_grad():
        assert float((predict(obs[:256]).clamp(-1,1)-target[:256]).abs().max()) < 2e-6

    def loss(batch):
        result = original(batch)
        ids = (index*31+state['calls']*257) % len(obs)
        mse = (predict(obs[ids])-target[ids]).square().mean()
        result['rehearsal_mse'] = mse.detach()
        result['actor_loss'] = result['actor_loss']+100*mse
        state['calls'] += 1
        state['examples'] += 256
        return result

    agent._compute_actor_loss = loss
    return state


def train(args):
    from . import ball_study_support as support, run_stand_ball_pilot as pilot
    from .prepare_stand_ball_pair import prepare_pair
    from .rig import Rig
    from .stand_ball_native import write_world
    from .stand_experiment_evidence import seal
    from .replicate_stand_representations import tensor_hash
    from .prepare_chest_normalizer_swap import policy_state
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    quiet = RETENTION / '301028-native'
    sealed(PARENT, [PARENT_MODEL, PARENT_MODEL.with_suffix('.amp.pt')])
    sealed(RETENTION, [quiet / n for n in ('trace.npz', 'report.json', 'contract.json')])
    assert read(quiet/'report.json')['standing_passes'] == 8
    full = torch.load(PARENT_MODEL.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    assert tensor_hash(policy_state(full)) == read(quiet/'report.json')['actor_hash']
    with np.load(quiet/'trace.npz') as z:
        keep = z['time_s'] <= 16
        assert (z['alive_before'][keep] & z['balance'][keep] & z['posture'][keep] & z['pre_balance'][keep] & z['pre_posture'][keep]).all()
        observations, actions = z['observation'][keep].reshape(-1,374).copy(), z['action'][keep].reshape(-1,33).copy()
    settings = dict(seeds=[301028], actors=['parent'], updates_per_arm=100,
                    maximum_training_seconds_per_arm=420, checkpoint_interval=1000,
                    envs=64, rollout_steps=64, actor_learning_rate=3e-5,
                    first_onsets=[1,3,5,9], second_shot_gap=4, episode_seconds=20,
                    optimization_seed=381028, rehearsal_weight=100,
                    evidence=[str(PARENT/'final-verification.json'), str(RETENTION/'final-verification.json'),
                              *[str(quiet/n) for n in ('trace.npz','report.json','contract.json')]])
    paths = [Path(__file__), Path(pilot.__file__), Path('mujoco_rig/mimic/prepare_stand_ball_pair.py'),
             Path('mujoco_rig/mimic/chest_ball_curriculum.py'), Path('mujoco_rig/mimic/stand_ball_training_evidence.py')]
    with patch.object(support, 'checkpoint', lambda *_: PARENT_MODEL):
        parent, inputs, sources = support.protect_used(args.out, paths, Path('mujoco_rig/mimic/PERTURB_PREVIEW_PLAN.md'), settings)
    parent = deepcopy(parent)
    parent['config']['actor_optimizer']['learning_rate'] = 3e-5
    assert parent['config']['steps_per_iter'] == 64 and parent['config']['actor_epochs'] == 3
    save(args.out/'protocol.json', dict(settings=settings, inputs=inputs, sources=sources,
         config=parent['config'], demonstrations=parent['demonstrations'], parent=str(PARENT_MODEL),
         scope='Single-parent experimental preview; learned PPO, no scripted teacher; fresh optimizers'))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    write_world(args.out/'dummy.xml', support.FOUNDATION/'ball-5kg.xml', world, 5)
    with repeated_curriculum():
        pair = prepare_pair(rig, reference, world, parent, PARENT_MODEL, 381028, args.out/'starts')
        assert torch.equal(pair['quiet'][1].launch_step, pair['single_ball'][1].launch_step)
        rehearsal = {arm:attach_rehearsal(agent, observations, actions) for arm,(agent,_,_) in pair.items()}
        results = {}
        for arm in ('single_ball', 'quiet'):
            print('Training', arm, flush=True)
            results[arm] = pilot.train_arm(pair, arm, 381028, args.out/'starts', args.out/arm, settings)
            assert rehearsal[arm]['calls'] == 4800 and rehearsal[arm]['examples'] == 1228800
            save(args.out/'progress.json', dict(results=results, rehearsal=rehearsal))
            print('Completed', arm, results[arm], flush=True)
    save(args.out/'report.json', dict(results=results, rehearsal=rehearsal, training_updates=200,
         transitions=819200, promoted=False, evaluation_cases=0))
    seal(args.out, inputs, sources, verified=True, training_updates=200, evaluation_cases=0)


def evaluate(args):
    from .rig import Rig
    from .stand_experiment_evidence import load_actor, seal
    from .evaluate_stand_sustained import evaluate_duration, verify_duration
    from .evaluate_stand_ball_boundary import evaluate as native
    from .evaluate_stand_ball_gpu import evaluate as gpu, compare
    from .run_chest_repeated_screen import wrap_evaluator
    from .ball_study_support import body_observer, save_bodies
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    actors = dict(stage12=(OLD, OLD_MODEL), stage33=(PARENT, PARENT_MODEL),
                  quiet=(args.training,args.training/'quiet/final/model.pt'),
                  repeated=(args.training,args.training/'single_ball/final/model.pt'))
    for root, model in actors.values():
        sealed(root, [model, model.with_suffix('.amp.pt')])
    args.out.mkdir(parents=True, exist_ok=False)
    source = read(args.training/'protocol.json')
    inputs = {str(p):digest(p) for root, model in actors.values()
              for p in (root/'final-verification.json', model, model.with_suffix('.amp.pt'))}
    for p in (Path(__file__), args.training/'protocol.json', args.training/'dummy.xml',
              args.training/'stationary_reference.npz', args.training/'ball-5kg.xml'):
        inputs[str(p)] = digest(p)
    save(args.out/'protocol.json', dict(inputs=inputs, maximum_cases=400, training_updates=0,
         conditions='Single 1.4/1.6m/s +/-X at 5s, repeated 1.6m/s +X at 5/9s; 8 poses, both backends',
         preview_rule='New repeated endpoint only if all quiet/admission gates pass, no primary loss versus Stage33 and strictly better repeated recovery; else Stage33'))
    rig = Rig.load(args.training/'dummy.xml', policy_exclude=())
    ref, world = args.training/'stationary_reference.npz', args.training/'ball-5kg.xml'
    results = {}
    for name, (root, checkpoint) in actors.items():
        actor = load_actor(rig, ref, read(root/'protocol.json'), checkpoint)
        folder = args.out/name
        folder.mkdir()
        result = dict(quiet={}, impacts={}, admitted=True)
        for backend in ('native', 'gpu'):
            trial = folder/f'quiet-{backend}'
            result['quiet'][backend] = evaluate_duration(rig,ref,'target_pd',backend,actor,trial,seconds=30)
            verify_duration(trial,30)
        for label, speed, sign, repeated in [('easy-plus',1.4,1,False),('easy-minus',1.4,-1,False),
                ('strong-plus',1.6,1,False),('strong-minus',1.6,-1,False),('repeat',1.6,1,True)]:
            cfg = dict(seconds=16 if repeated else 12, launch_seconds=5, launch_distance_m=.25)
            rows = {}
            for backend, function in (('native',native),('gpu',gpu)):
                trial = folder/f'hit-{label}-{backend}'
                if repeated:
                    function = wrap_evaluator(function,backend,dict(conditions=[dict(name=label,launch_times_seconds=[5,9])]))
                if backend == 'native':
                    journal = {}
                    with body_observer(journal,int(np.ceil(cfg['seconds']/(rig.model.opt.timestep*rig.decimation)))):
                        rows[backend] = function(rig,ref,actor,world,trial,cfg,speed,sign)
                    save_bodies(trial,journal)
                    bodies = read(trial/'body-report.json')
                    rows['chest_first'] = all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies']
                                              and 'Head' not in c['clusters'][0]['bodies'] for c in bodies['cases'])
                else:
                    rows[backend] = function(rig,ref,actor,world,trial,cfg,speed,sign)
            rows['parity'] = compare(rows['native'],rows['gpu'],rig.model.opt.timestep*rig.decimation)
            if repeated:
                rows['parity']['passed'] &= rows['native']['native_repeat_launch_clear']
                rows['parity']['passed'] &= all(a['passed']==b['passed'] for x,y in zip(rows['native']['cases'],rows['gpu']['cases'])
                                                for a,b in zip(x['encounters'],y['encounters']))
            result['admitted'] &= rows['parity']['passed'] and rows['chest_first']
            result['impacts'][label] = rows
            print(name,label,{b:rows[b]['recovery_passes'] for b in ('native','gpu')},flush=True)
        results[name] = result
        save(args.out/'progress.json',results)
    baseline, candidate = results['stage33'],results['repeated']
    quiet_pass = all(q['standing_passes']==8 for q in candidate['quiet'].values())
    retain = all(candidate['impacts'][k][b]['recovery_passes']>=baseline['impacts'][k][b]['recovery_passes']
                 for k in baseline['impacts'] for b in ('native','gpu'))
    gain = all(candidate['impacts']['repeat'][b]['recovery_passes']>baseline['impacts']['repeat'][b]['recovery_passes'] for b in ('native','gpu'))
    selected = 'repeated' if quiet_pass and retain and gain and candidate['admitted'] else 'stage33'
    assert results[selected]['admitted'] and all(q['standing_passes']==8 for q in results[selected]['quiet'].values())
    root, model = actors[selected]
    actor = load_actor(rig,ref,read(root/'protocol.json'),model)
    retention = {}
    for backend in ('native','gpu'):
        trial = args.out/f'selected-retention-{backend}'
        retention[backend] = evaluate_duration(rig,ref,'target_pd',backend,actor,trial,seconds=120)
        verify_duration(trial,120)
    assert all(r['standing_passes']==8 for r in retention.values())
    save(args.out/'report.json',dict(results=results,selected=selected,checkpoint=str(model),
         selected_root=str(root),retention=retention,physical_cases=400,training_updates=0,
         candidate_retains=retain,candidate_repeated_gain=gain,experimental_preview=True))
    seal(args.out,inputs,[],verified=True,physical_cases=400,training_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','evaluate'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training',type=Path)
    args = parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    (train if args.mode == 'train' else evaluate)(args)
