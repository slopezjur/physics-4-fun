"""Fresh matched PPO arm: quiet standing versus a bounded chest-push curriculum."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .audit_stand_failures import read, save
from .compare_stand_actor_rate import build_task_type, REWARD_COLUMNS
from .evaluate_stand_challenges import evaluate_challenge, verify_challenge
from .evaluate_stand_sustained import verify_duration
from .measure_stand_learning_curve import assess, rng_state, restore_rng, training_fingerprint, tree_hash
from .replicate_stand_representations import tensor_hash
from .train_stand_sustained import assess_sustained
from .stand_experiment_evidence import protect, seal
from .train_torque_stand import build_agent
from .policy import StandPolicy
from .runtime import activate
from .rig import Rig


def amplitude(arm, update):
    if arm == 'quiet' or update < 600:
        return 0.
    return 10.+170.*min((update-600)/300.,1.)


def build_push_task():
    class PushTask(build_task_type()):
        def __init__(self, *args, arm, **kwargs):
            self.arm, self.learning_update = arm, 0
            self.force_records = []
            super().__init__(*args, **kwargs)
            self.push_body = self.rig.model.body('Chest').id
            self.push_start = int(np.ceil(5./self.dt))
            self.push_period = int(np.ceil(10./self.dt))
            self.push_width = int(np.ceil(.1/self.dt))
            directions = torch.tensor([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0]], device=self.device)
            self.push_direction = directions[(self.ids//8)%4]

        def step(self, action):
            strength = amplitude(self.arm,self.learning_update)
            active = (self.ids<32)&(self.steps>=self.push_start)&((self.steps-self.push_start)%self.push_period<self.push_width)
            force = self.push_direction*active[:,None]*strength
            self.engine.set_external_force(self.push_body,force)
            actual = self.engine._sim_state.body_force[0][:,self.push_body-1,:3]
            self.force_records.append(torch.cat((self.steps[:,None].float(),actual),dim=1).detach().clone())
            return super().step(action)

    return PushTask


def verify_force(values, arm, update, dt):
    assert values.shape==(64,64,4) and np.isfinite(values).all()
    steps=values[:,:,0]
    assert np.array_equal(steps,np.rint(steps))
    start,period,width=(int(np.ceil(x/dt)) for x in (5.,10.,.1))
    active=(np.arange(64)[None,:]<32)&(steps>=start)&((steps-start)%period<width)
    directions=np.asarray([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0]],dtype=np.float32)[(np.arange(64)//8)%4]
    expected=directions[None,:,:]*active[:,:,None]*np.float32(amplitude(arm,update))
    assert np.array_equal(values[:,:,1:],expected)
    return int(np.count_nonzero(np.any(expected!=0,axis=2)))


def verify_saved(out):
    protocol,report=read(out/'protocol.json'),read(out/'report.json')
    assert protocol['initialize_from'] is None and report['updates']==1200
    assert report['transitions']==4915200 and report['training_seconds']<=2405.
    rows=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    assert [r['iteration'] for r in rows]==list(range(1,1201))
    assert rows[-1]['actor_optimizer_steps']==57600 and rows[-1]['critic_optimizer_steps']==76800
    with np.load(out/'training-records.npz') as data:
        rewards,forces=data['reward'],data['force']
        assert rewards.shape==(1200,64,6) and np.isfinite(rewards).all()
        assert np.array_equal(rewards[:,:,0],rewards[:,:,1]) and (rewards[:,:,2]==1).all()
        intervals=sum(verify_force(forces[i],protocol['arm'],i,protocol['control_timestep']) for i in range(1200))
    assert intervals==report['forced_world_control_intervals']
    from .amp_recovery import validate_sidecar
    from .stand_experiment_evidence import verify_rollout
    for stage in ('initial','updates-600','updates-1200'):
        path=out/stage
        portable=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
        full=torch.load(path/'model.amp.pt',map_location='cpu',weights_only=True)
        validate_sidecar(portable,full)
        assert tensor_hash(full)==report['assessments'][stage]['full_tensor_hash']
        isolation=read(path/'isolation.json')
        assert isolation['before']==isolation['after'] and isolation['rng_before']==isolation['rng_after']
        for backend in ('native','gpu'):
            verify_rollout(path/backend)
    isolation=read(out/'sustained-1200'/'isolation.json')
    assert isolation['before']==isolation['after'] and isolation['rng_before']==isolation['rng_after']
    assert all(r['transitions']==4096*r['iteration'] for r in rows)
    for backend in ('native','gpu'):
        verify_duration(out/'sustained-1200'/backend,60.)
        assert read(out/'sustained-1200'/backend/'report.json')['actor_hash']==read(out/'updates-1200'/backend/'report.json')['actor_hash']
    isolation=read(out/'push-isolation.json')
    assert isolation['before']==isolation['after'] and isolation['rng_before']==isolation['rng_after']
    for scenario in protocol['assessment_scenarios']:
        for backend in ('native','gpu'):
            _,result=verify_challenge(out/f"{scenario['name']}-{backend}",12.,scenario)
            assert result['actor_hash']==read(out/'updates-1200'/backend/'report.json')['actor_hash']
    return dict(verified=True,cases=192,training_updates=1200,fresh_initialization=True,
        applied_force_schedule_verified=True,reward_identity_verified=True,checkpoint_pairs_verified=True,
        evaluation_isolation_verified=True,sustained_stand_verified=report['sustained_stand_verified'])


def run(args):
    assert args.seed in (301027,301028) and args.arm in ('quiet','curriculum')
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    torch.set_num_threads(1)
    inputs,sources=protect(args.prior,args.out,[Path(__file__)])
    prior=read(args.prior/'protocol.json')
    config=copy.deepcopy(prior['config'])
    assert config['actor_optimizer']['learning_rate']==3e-4
    rig,reference=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    scenarios=[]
    for strength in (90,180):
        for name,unit in [('xp',(1,0,0)),('xm',(-1,0,0)),('yp',(0,1,0)),('ym',(0,-1,0))]:
            scenarios.append(dict(name=f'{strength}N-{name}',body='Chest',pulses=[dict(onset_seconds=5.,duration_seconds=.1,
                force_world_n=[strength*x for x in unit])]))
    protocol=dict(inputs=inputs,sources=sources,config=config,seed=args.seed,sampling_seed=args.seed+701,arm=args.arm,
        demonstrations=prior['demonstrations'],model_sha256=prior['model_sha256'],reference_sha256=prior['reference_sha256'],
        initialize_from=None,maximum_updates=1200,maximum_training_seconds=2400.,maximum_cases=192,
        control_timestep=rig.model.opt.timestep*rig.decimation,assessment_scenarios=scenarios,
        curriculum='Updates 1-600 quiet; 601-900 ramp 10 to 180 N; 901-1200 180 N. First 32 worlds receive four directions equally across all eight poses; remaining 32 stay quiet. Pulses start at 5 s and repeat every 10 s for 0.1 s.',
        scope='Only training force exposure differs; all rewards, PPO, plant and controller settings remain identical')
    save(args.out/'protocol.json',protocol)
    task=build_push_task()(rig,reference,64,'cuda:0',arm=args.arm,representation='target_pd',reward_mode='reference_v9',
        episode_seconds=60.,seed=args.seed,demonstrations=[Path(r['path']) for r in prior['demonstrations']],demonstration_rig=rig)
    task.mirror_demos=False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent=build_agent(config,task,'cuda:0')
    assert int(agent._obs_norm._count)==int(agent._a_norm._count)==0
    initial_hash=tensor_hash(agent.state_dict())
    save(args.out/'contract.json',task.observation_contract())
    assessments={'initial':assess(agent,task,rig,reference,args.out/'initial',0.,0.)}
    elapsed,reward_records,force_records,forced=0.,[],[],0
    agent._init_train()
    torch.manual_seed(args.seed+701)
    np.random.seed(args.seed+701)
    agent._curr_obs,agent._curr_info=agent._reset_envs()
    for target in (600,1200):
        while agent._iter<target:
            if elapsed>=2400.:
                raise RuntimeError('Training cap reached before endpoint; preserve incomplete arm')
            task.learning_update=agent._iter
            task.reward_records.clear()
            task.force_records.clear()
            torch.cuda.synchronize()
            started=time.perf_counter()
            info=agent._train_iter()
            torch.cuda.synchronize()
            elapsed+=time.perf_counter()-started
            force=torch.stack(task.force_records).cpu().numpy()
            forced+=verify_force(force,args.arm,agent._iter,task.dt)
            force_records.append(force)
            reward=torch.stack(task.reward_records).cpu().numpy()
            assert reward.shape==(64,6) and np.isfinite(reward).all()
            assert np.array_equal(reward[:,0],reward[:,1]) and (reward[:,2]==1).all()
            reward_records.append(reward)
            agent._sample_count=agent._update_sample_count()
            agent._iter+=1
            row=dict(iteration=agent._iter,transitions=agent._sample_count,training_seconds=elapsed,
                force_amplitude=amplitude(args.arm,agent._iter-1),forced_world_control_intervals=forced,
                actor_optimizer_steps=agent._actor_optimizer.get_steps(),critic_optimizer_steps=agent._critic_optimizer.get_steps(),
                **{k:float(v) for k,v in info.items()})
            assert all(np.isfinite(v) for v in row.values())
            assert all(torch.isfinite(v).all() for v in agent.state_dict().values())
            with (args.out/'updates.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            save(args.out/'progress.json',row)
            if agent._iter%50==0:
                print(dict(seed=args.seed,arm=args.arm,updates=agent._iter,training_seconds=elapsed),flush=True)
        stage=f'updates-{target}'
        assessments[stage]=assess(agent,task,rig,reference,args.out/stage,elapsed,elapsed)
        save(args.out/'assessments.json',assessments)
    sustained=assess_sustained(agent,task,rig,reference,args.out/'sustained-1200',assessments['updates-1200']['checkpoint'])
    before,state=training_fingerprint(agent,task),rng_state(task)
    results={}
    try:
        for scenario in scenarios:
            results[scenario['name']]={backend:evaluate_challenge(rig,reference,'target_pd',backend,StandPolicy(agent),
                args.out/f"{scenario['name']}-{backend}",seconds=12.,scenario=scenario) for backend in ('native','gpu')}
            save(args.out/'push-progress.json',results)
    finally:
        restore_rng(task,state)
    after,restored=training_fingerprint(agent,task),rng_state(task)
    assert before==after and tree_hash(state)==tree_hash(restored)
    save(args.out/'push-isolation.json',dict(before=before,after=after,rng_before=tree_hash(state),rng_after=tree_hash(restored)))
    np.savez_compressed(args.out/'training-records.npz',reward=np.stack(reward_records),force=np.stack(force_records),reward_columns=np.asarray(REWARD_COLUMNS))
    report=dict(seed=args.seed,arm=args.arm,initial_hash=initial_hash,assessments=assessments,push_results=results,
        sustained_stand_verified=sustained['full_interval_stand'],sustained_assessment=sustained,
        final_checkpoint=assessments['updates-1200']['checkpoint'],updates=agent._iter,transitions=agent._sample_count,
        training_seconds=elapsed,forced_world_control_intervals=forced,arena_changed=False)
    save(args.out/'report.json',report)
    verification=verify_saved(args.out)
    seal(args.out,inputs,sources,**verification)
    print(dict(completed=True,seed=args.seed,arm=args.arm,sustained_stand_verified=report['sustained_stand_verified']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--arm',choices=('quiet','curriculum'),required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
