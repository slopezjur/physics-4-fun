"""Bounded PPO for a new residual; the original Stand network remains frozen."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .runtime import activate
from . import run_support_skill_study as study
from .support_skill_experiments import metric_adapter


def build(config,env,initialize=None):
    from learning.ppo_agent import PPOAgent
    from .stable_normalizer import install

    class ResidualPPO(PPOAgent):
        def _build_normalizers(self):
            super()._build_normalizers();install(self)

    agent=ResidualPPO(config,env,env.device)
    if initialize:
        agent.load_state_dict(torch.load(initialize,map_location=env.device,weights_only=True))
    else:
        with torch.no_grad():
            for parameter in agent._model._action_dist._mean_net.parameters(): parameter.zero_()
    agent._sync_optimizer()
    agent._init_train()
    agent._curr_obs,agent._curr_info=agent._reset_envs()
    return agent


def demonstrations(agent,paths,weight):
    observations,actions=[],[]
    for path in paths:
        path=Path(path);report=json.loads((path/'report.json').read_text())
        with np.load(path/'trace.npz') as z:
            valid=np.array([r['transfer_pass'] and r['final_stand'] for r in report['cases']])
            keep=z['alive'] & z['support_valid'] & valid[None,:]
            observations.append(z['observation'][keep]);actions.append(z['action'][keep])
    obs=torch.tensor(np.concatenate(observations),device=agent._device)
    target=torch.tensor(np.concatenate(actions),device=agent._device)
    if not len(obs): raise ValueError('No physically successful demonstration examples')
    base=agent._compute_actor_loss;counts=dict(dataset_examples=len(obs),presentations=0,weight=weight)

    def loss(batch):
        result=base(batch)
        ids=torch.randint(len(obs),(256,),device=agent._device)
        mean=agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs[ids])).mode)
        mse=(mean-target[ids]).square().mean()
        result['actor_loss']=result['actor_loss']+weight*mse
        result['teacher_mse']=mse.detach();counts['presentations']+=len(ids)
        return result

    agent._compute_actor_loss=loss
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.protocol.read_text())
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from util import mp_util
    from .support_skill_core import SkillEnvironment,Goal
    from .policy import StandPolicy
    mp_util.init(0,1,'cuda:0',None);torch.set_num_threads(1)
    rig,original,actors=study.setup(spec,args.out)
    results={}
    for parent,anchor in actors.items():
        for arm in spec['arms']:
            for rng in spec.get('optimization_seeds',[71001]):
                label=f"{parent}-{arm['name']}-{rng}";out=args.out/label;out.mkdir()
                random.seed(rng);np.random.seed(rng);torch.manual_seed(rng)
                goals=[Goal(**dict(arm['goal'],side=(i//8)%2)) for i in range(64)]
                env=SkillEnvironment(rig,study.FOUNDATION/'stationary_reference.npz',deepcopy(anchor),goals,'cuda:0',
                    seconds=arm.get('seconds',10.),action_mode=arm.get('action_mode','basis'),radius=arm.get('radius',.5))
                config=deepcopy(original['config']);config['agent_name']='PPO'
                config['actor_optimizer']['learning_rate']=arm.get('learning_rate',3e-4)
                config['model']['action_std']=arm.get('action_std',.1)
                save_extra={}
                if arm.get('initialize'):
                    save_extra[arm['initialize']]=study.digest(arm['initialize'])
                for source in arm.get('demonstrations',[]):
                    for name in ('trace.npz','report.json'): save_extra[str(Path(source)/name)]=study.digest(Path(source)/name)
                study.save(out/'protocol.json',dict(config=config,contract=env.contract(),arm=arm,optimization_seed=rng,extra_inputs=save_extra))
                agent=build(config,env,arm.get('initialize'))
                initial=StandPolicy(agent).eval()
                if not arm.get('initialize'):
                    assert torch.equal(initial(env.observations()),torch.zeros((64,env.get_action_space().shape[0]),device='cuda:0'))
                torch.save(agent.state_dict(),out/'initial.pt')
                teacher=demonstrations(agent,arm['demonstrations'],arm.get('supervision_weight',1.)) if arm.get('demonstrations') else None
                rows=[];start=time.perf_counter()
                for update in range(arm.get('updates',60)):
                    if time.perf_counter()-start>600: raise RuntimeError('Training budget exceeded; comparison incomplete')
                    info=agent._train_iter();agent._sample_count=agent._update_sample_count();agent._iter+=1
                    row=dict(update=agent._iter,transitions=agent._sample_count,elapsed_seconds=time.perf_counter()-start,
                        actor_optimizer_steps=agent._actor_optimizer.get_steps(),critic_optimizer_steps=agent._critic_optimizer.get_steps(),
                        **{k:float(v) for k,v in info.items()})
                    assert row['transitions']==4096*agent._iter
                    if not all(np.isfinite(v) for v in row.values()): raise RuntimeError('Nonfinite training diagnostics')
                    rows.append(row)
                    with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if (update+1)%10==0: print(label,update+1,row['transitions'],round(row['elapsed_seconds'],1),flush=True)
                torch.save(agent.state_dict(),out/'final.pt')
                policy=StandPolicy(agent).eval().requires_grad_(False)
                torch.save(policy.state_dict(),out/'actor.pt')
                results[label]=dict(training=rows[-1],supervision=teacher,evaluations={})
                with metric_adapter():
                    for case in spec['evaluations']:
                        for backend in case.get('backends',['native']):
                            name=case['name']+'-'+backend
                            results[label]['evaluations'][name]=study.evaluate(rig,anchor,
                                dict(case,action_mode=arm.get('action_mode','basis'),radius=arm.get('radius',.5)),
                                backend,out/name,policy=policy)
                            print(label,name,{k:v for k,v in results[label]['evaluations'][name].items() if k!='cases'},flush=True)
                for path,expected in save_extra.items():
                    assert study.digest(path)==expected
                study.save(out/'report.json',results[label]);study.save(args.out/'progress.json',results)
                del agent,env,policy;gc.collect();torch.cuda.empty_cache()
    protocol=json.loads((args.out/'protocol.json').read_text())
    for path,expected in {**protocol['sources'],**protocol['inputs']}.items(): assert study.digest(path)==expected
    study.save(args.out/'report.json',dict(stage=spec['stage'],completed=True,results=results))
    study.save(args.out/'final-verification.json',dict(inputs_unchanged=True,outputs={str(p.relative_to(args.out)):study.digest(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
