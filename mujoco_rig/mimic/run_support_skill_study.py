"""Prospective compact physical comparisons for joint-only support skills."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate


FOUNDATION = Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01')
PARENTS = Path('logs/mimickit-training/stand-ball-stage33-focused-learning-20261007-02')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def longest(value):
    padded = np.r_[False, value, False].astype(int)
    return int(np.max(np.flatnonzero(np.diff(padded)==-1)-np.flatnonzero(np.diff(padded)==1), initial=0))


def setup(spec, out):
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = {}
    # Snapshot all local Python helpers; preserve each exact per-stage implementation.
    for path in sorted(Path(__file__).parent.glob('*.py')):
        target = out/'sources'/path.name
        shutil.copy2(path, target)
        sources[str(path)] = digest(path)
    inputs = [FOUNDATION/'dummy.xml', FOUNDATION/'stationary_reference.npz',
              FOUNDATION/'stationary_reference.json', FOUNDATION/'protocol.json']
    for seed in spec['parents']:
        inputs.extend([PARENTS/str(seed)/'varied/final'/name for name in ('model.pt','model.amp.pt')])
    spec = dict(spec, sources=sources, inputs={str(p):digest(p) for p in inputs},
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(out/'protocol.json',spec)
    rig = Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    original = json.loads((FOUNDATION/'protocol.json').read_text())
    actors = {seed:load_actor(rig,FOUNDATION/'stationary_reference.npz',original,
        PARENTS/str(seed)/'varied/final/model.pt') for seed in spec['parents']}
    return rig, original, actors


def summarize(z, goals, dt):
    from .corrective_step_audit import measure_steps
    rows=[]
    for i,g in enumerate(goals):
        live=z['alive'][:,i]
        good=live & z['support_valid'][:,i]
        share=z['loads'][:,i,0]/np.maximum(z['loads'][:,i].sum(-1),1.)
        target=.5+.5*z['goal'][:,i,0]
        transfer=good & (z['coefficients'][:,i,0]>.95) & (np.abs(share-target)<=.1)
        lift=good & (z['loads'][:,i,g.side]<5) & (z['loads'][:,i,1-g.side]>=30)
        actual=lift & (z['clearance'][:,i,g.side]>=max(.004,.8*g.lift)) & (z['coefficients'][:,i,1]>.95)
        strict=lift & (z['clearance'][:,i,g.side]>=.015)
        settled=live & z['stand_valid'][:,i]
        events=measure_steps(z['time'],z['xy'][:,i],z['clearance'][:,i],z['loads'][:,i],
            live,settled,g.onset)
        survivors=bool(live[-1] and z['done'][-1,i]!=1)
        rows.append(dict(case=i%8,side=g.side,survived=survivors,
            transfer_seconds=longest(transfer)*dt,transfer_pass=bool(longest(transfer)*dt>=.5),
            requested_lift_seconds=longest(actual)*dt,strict_lift_seconds=longest(strict)*dt,
            final_stand=bool(survivors and settled[-int(np.ceil(2/dt)):].all()),
            whole_interval_stand=bool(survivors and settled.all()),
            mean_reward=float(z['reward'][live,i].mean()),
            maximum_foot_clearance_m=float(z['clearance'][good,i,g.side].max(initial=0)),
            mechanical_steps=len(events),settled_steps=sum(e['settled_window_start'] is not None for e in events),events=events))
    return dict(cases=rows,case_count=len(rows),survivors=sum(r['survived'] for r in rows),
        transfer_passes=sum(r['transfer_pass'] for r in rows),
        lifts_15mm_50ms=sum(r['strict_lift_seconds']>=.05 for r in rows),
        final_stand=sum(r['final_stand'] for r in rows),
        whole_interval_stand=sum(r['whole_interval_stand'] for r in rows),
        mechanical_steps=sum(r['mechanical_steps'] for r in rows),settled_steps=sum(r['settled_steps'] for r in rows),
        mean_reward=float(np.mean([r['mean_reward'] for r in rows])))


@torch.no_grad()
def evaluate(rig, anchor, arm, backend, out, policy=None):
    from envs.base_env import EnvMode
    from .support_skill_core import SkillEnvironment, Goal
    device='cpu' if backend=='native' else 'cuda:0'
    goals=[Goal(**dict(arm['goal'],side=side)) for side in arm.get('sides',[0,1]) for _ in range(8)]
    env=SkillEnvironment(rig,FOUNDATION/'stationary_reference.npz',deepcopy(anchor),goals,device,
        seconds=arm.get('seconds',10.),action_mode=arm.get('action_mode','basis'),radius=arm.get('radius',.5))
    env.set_mode(EnvMode.TEST);env.reset()
    out.mkdir()
    save(out/'contract.json',dict(env.contract(),goals=[asdict(g) for g in goals],fits=env.fits))
    if policy is not None: policy=deepcopy(policy).to(device).eval()
    active=np.ones(len(goals),dtype=bool);rows=[]
    feedback=torch.zeros(len(goals),device=device)
    for frame in range(int(np.ceil(arm.get('seconds',10.)/env.dt))):
        if backend=='native': env.engine.active=active.copy()
        obs=env.observations();coefficients=env.coefficients().clone()
        pre=env.measurements()
        preflags={**env.balance_criteria(pre),**env.posture_criteria(pre)}
        prestand=torch.stack(list(preflags.values())).all(0)
        if policy is not None: action=policy(obs)
        elif arm.get('controller','teacher')=='zero': action=torch.zeros((len(goals),env.get_action_space().shape[0]),device=device)
        else:
            action=env.teacher_action()
            if arm.get('feedback',0):
                share=pre['foot_loads_n'][:,0]/pre['foot_loads_n'].sum(-1).clamp_min(1.)
                error=(share-(.5+.5*env.goal_features()[:,0]))*torch.where(env.side==0,1.,-1.)
                feedback=(feedback+arm['feedback']*error*env.dt).clamp(-.5,.5)
                action[:,0]=(action[:,0]+feedback*coefficients[:,0]).clamp(-1,1)
        _,reward,done,_=env.step(action)
        state=env.last
        flags={**env.balance_criteria(state),**env.posture_criteria(state)}
        stand=torch.stack(list(flags.values())).all(0)&prestand
        flags.pop('two_foot_support');preflags.pop('two_foot_support')
        supported=torch.stack(list(flags.values())+list(preflags.values())).all(0)
        supported &= state['foot_loads_n'].amax(-1)>=30
        if backend=='native' and env.engine.solver_warning[active].any(): raise RuntimeError('Native solver warning')
        q=env.engine.native_qpos().detach().cpu().numpy().copy()
        if not np.isfinite(q[active]).all(): raise RuntimeError('Nonfinite live state')
        row=dict(alive=active.copy(),done=done.cpu().numpy().copy(),qpos=q,
            loads=state['foot_loads_n'],clearance=state['clearance_m'],xy=state['foot_xy_m'],com=state['com_m'],
            support_valid=supported,stand_valid=stand,coefficients=coefficients,goal=obs[:,-6:],
            action=action,reward=reward,issued=env.engine.policy_control.pending[(env.engine.policy_control.cursor-1)%2,:,:33])
        if arm.get('save_observations',False): row['observation']=obs
        rows.append({k:v.detach().cpu().numpy().copy() if isinstance(v,torch.Tensor) else v for k,v in row.items()})
        active &= done.cpu().numpy()==0
        if not active.any(): break
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    z['time']=(np.arange(len(rows))+1)*env.dt
    np.savez_compressed(out/'trace.npz',**z)
    report=summarize(z,goals,env.dt)
    report['trace_sha256']=digest(out/'trace.npz')
    save(out/'report.json',report)
    return report


def stage_one(rig,actors,out):
    from . import representation_task,stand_ball_native,stand_ball_gpu
    from .support_skill_core import AddedTargetControl
    from .evaluate_stand_sustained import evaluate_duration
    from .evaluate_stand_ball_boundary import evaluate as native
    from .evaluate_stand_ball_gpu import evaluate as gpu
    results={}
    settings=dict(seconds=12.,launch_seconds=5.,launch_distance_m=.25)
    prior=Path('logs/mimickit-training/corrective-step-pilot-20261007-01')
    ball=FOUNDATION/'ball-5kg.xml'
    if not ball.exists():
        ball=next(prior.glob('ball*.xml'))
    with ExitStack() as stack:
        for module in (representation_task,stand_ball_native,stand_ball_gpu):
            stack.enter_context(patch.object(module,'DelayedTargetControl33',AddedTargetControl))
        for seed,actor in actors.items():
            for backend,evaluator in [('native',native),('gpu',gpu)]:
                device='cpu' if backend=='native' else 'cuda:0'
                for condition in ['quiet','1.6-+1']:
                    label=f'{seed}-{backend}-{condition}';folder=out/label
                    actor=deepcopy(actor).to(device).eval()
                    report=(evaluate_duration(rig,FOUNDATION/'stationary_reference.npz','target_pd',backend,actor,folder,seconds=12.)
                        if condition=='quiet' else evaluator(rig,FOUNDATION/'stationary_reference.npz',actor,ball,folder,settings,1.6,1))
                    old=prior/str(seed)/'initial-0.25'/f'{condition}-{backend}'/'trace.npz'
                    with np.load(old) as a,np.load(folder/'trace.npz') as b:
                        keys=['qpos_after','action','issued_entry','done']
                        errors={k:float(np.abs(a[k].astype(float)-b[k].astype(float)).max()) for k in keys}
                    results[label]=dict(exact_match=all(x==0 for x in errors.values()),maximum_errors=errors,report=report,
                        baseline_path=str(old),baseline_sha256=digest(old))
                    save(out/'progress.json',results);print(label,results[label]['exact_match'],flush=True)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.protocol.read_text())
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None);torch.set_num_threads(1)
    rig,config,actors=setup(spec,args.out)
    if spec['stage']==1: results=stage_one(rig,actors,args.out)
    else:
        results={}
        for seed,actor in actors.items():
            for arm in spec['arms']:
                for backend in spec.get('backends',['gpu']):
                    label=f"{seed}-{arm['name']}-{backend}"
                    results[label]=evaluate(rig,actor,arm,backend,args.out/label)
                    save(args.out/'progress.json',results)
                    print(label,{k:v for k,v in results[label].items() if k!='cases'},flush=True)
    protocol=json.loads((args.out/'protocol.json').read_text())
    for path,expected in {**protocol['sources'],**protocol['inputs']}.items():
        if digest(path)!=expected: raise RuntimeError(f'Sealed input changed: {path}')
    save(args.out/'report.json',dict(stage=spec['stage'],completed=True,results=results))
    save(args.out/'final-verification.json',dict(inputs_unchanged=True,outputs={str(p.relative_to(args.out)):digest(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__': main()
