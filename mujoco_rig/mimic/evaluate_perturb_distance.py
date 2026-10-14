"""Frozen, compact evidence for the matched launch-distance continuations."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .audit_stand_failures import digest,read,save
from .runtime import activate
from .run_perturb_preview import sealed
from .run_perturb_distance import ASSESSMENT,PARENT,MODEL,compact


def impact(rig,reference,actor,world,folder,label,speed,sign,*,schedule=None,distance=.25):
    from .evaluate_stand_ball_boundary import evaluate as native
    from .evaluate_stand_ball_gpu import evaluate as gpu,compare
    from .run_chest_repeated_screen import wrap_evaluator
    from .ball_study_support import body_observer,save_bodies
    schedule=schedule or [5]
    dt=rig.model.opt.timestep*rig.decimation
    cfg=dict(seconds=schedule[-1]+7,launch_seconds=schedule[0],launch_distance_m=distance)
    rows={}
    for backend,function in (('native',native),('gpu',gpu)):
        path=folder/f'hit-{label}-{backend}'
        if len(schedule)>1:
            function=wrap_evaluator(function,backend,dict(conditions=[dict(name=label,launch_times_seconds=schedule)]))
        if backend=='native':
            journal={}
            with body_observer(journal,int(np.ceil(cfg['seconds']/dt))):
                rows[backend]=function(rig,reference,actor,world,path,cfg,speed,sign)
            save_bodies(path,journal)
            bodies=read(path/'body-report.json')
            checks=[]
            for case,body in zip(rows[backend]['cases'],bodies['cases']):
                encounters=case.get('encounters',[dict(hit=case['hit'],launch_frame=int(np.ceil(schedule[0]/dt)))])
                for i,event in enumerate(encounters):
                    end=int(np.ceil(schedule[i+1]/dt)) if i+1<len(schedule) else 10**9
                    clusters=[c for c in body['clusters'] if event['launch_frame']<=c['first_frame']<end]
                    checks.append(not event['hit'] or bool(clusters and 'Chest' in clusters[0]['bodies'] and 'Head' not in clusters[0]['bodies']))
            rows['chest_first']=all(checks)
        else:
            rows[backend]=function(rig,reference,actor,world,path,cfg,speed,sign)
        compact(path)
    parity=compare(rows['native'],rows['gpu'],dt)
    if len(schedule)>1:
        parity['passed'] &= rows['native']['native_repeat_launch_clear']
        for a,b in zip(rows['native']['cases'],rows['gpu']['cases']):
            for x,y in zip(a['encounters'],b['encounters']):
                parity['passed'] &= all(x[k]==y[k] for k in ('hit','prequalified','survived','final_valid','passed'))
                parity['passed'] &= all((x[k] is None and y[k] is None) or (x[k] is not None and y[k] is not None and abs(x[k]-y[k])<=2*dt)
                                        for k in ('first_contact_seconds','last_contact_seconds'))
                parity['passed'] &= abs(x['total_normal_impulse_ns']-y['total_normal_impulse_ns'])<=.25+.1*x['total_normal_impulse_ns']
    rows.update(parity=parity,trace_label=label,launch_distance_m=distance,launch_times_seconds=schedule)
    print(folder.name,label,{b:rows[b]['recovery_passes'] for b in ('native','gpu')},'admitted',bool(parity['passed'] and rows['chest_first']),flush=True)
    return rows


def run(args):
    from .rig import Rig
    from .stand_experiment_evidence import load_actor,seal
    from .evaluate_stand_sustained import evaluate_duration,verify_duration
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    checkpoints={label:args.training/label/'final/model.pt' for label in ('fixed','mixed')}
    for checkpoint in checkpoints.values():
        sealed(args.training,[checkpoint,checkpoint.with_suffix('.amp.pt')])
    sealed(ASSESSMENT,[ASSESSMENT/'report.json'])
    baseline_report=read(ASSESSMENT/'report.json')
    assert Path(baseline_report['checkpoint'])==MODEL
    baseline=deepcopy(baseline_report['results']['repeated'])
    assert baseline['admitted'] and all(r['standing_passes']==8 for r in baseline_report['retention'].values())
    args.out.mkdir(parents=True,exist_ok=False)
    input_paths=[Path(__file__),Path('mujoco_rig/mimic/run_perturb_distance.py'),args.training/'protocol.json',
                 args.training/'final-verification.json',ASSESSMENT/'report.json',ASSESSMENT/'final-verification.json',
                 MODEL,MODEL.with_suffix('.amp.pt'),*[p for model in checkpoints.values() for p in (model,model.with_suffix('.amp.pt'))]]
    inputs={str(p):digest(p) for p in input_paths}
    save(args.out/'protocol.json',dict(inputs=inputs,maximum_cases=240,training_updates=0,baseline=str(ASSESSMENT),
         primary_cases=192,conditional_held_out_cases=32,conditional_retention_cases=16,
         held_out_schedule_seconds=[2,6],
         prospective_timing_correction='The plan named 3/7, which is already a training schedule. Use genuinely untrained 2/6 before evaluating either candidate; budgets and gates unchanged.',
         plan='mujoco_rig/mimic/PERTURB_DISTANCE_PLAN.md'))
    rig=Rig.load(args.training/'dummy.xml',policy_exclude=())
    ref,world=args.training/'stationary_reference.npz',args.training/'ball-5kg.xml'
    results=dict(parent=baseline)
    candidates=[]
    for label,checkpoint in checkpoints.items():
        actor=load_actor(rig,ref,read(args.training/'protocol.json'),checkpoint)
        folder=args.out/label
        folder.mkdir()
        result=dict(quiet={},impacts={},admitted=True)
        for backend in ('native','gpu'):
            path=folder/f'quiet-{backend}'
            result['quiet'][backend]=evaluate_duration(rig,ref,'target_pd',backend,actor,path,seconds=30)
            verify_duration(path,30)
            compact(path)
        for name,speed,sign in (('easy-plus',1.4,1),('easy-minus',1.4,-1),('strong-plus',1.6,1),('strong-minus',1.6,-1)):
            result['impacts'][name]=impact(rig,ref,actor,world,folder,name,speed,sign)
        result['impacts']['repeat']=impact(rig,ref,actor,world,folder,'repeat-clear',1.6,1,schedule=[5,9],distance=.46)
        result['admitted']=all(r['parity']['passed'] and r['chest_first'] for r in result['impacts'].values())
        retained=all(result['impacts'][key][b]['recovery_passes']>=baseline['impacts'][key][b]['recovery_passes']
                     for key in baseline['impacts'] for b in ('native','gpu'))
        improved=all(result['impacts']['repeat'][b]['recovery_passes']>baseline['impacts']['repeat'][b]['recovery_passes'] for b in ('native','gpu'))
        result['qualifies']=bool(result['admitted'] and retained and improved and all(r['standing_passes']==8 for r in result['quiet'].values()))
        if result['qualifies']:
            candidates.append(label)
        results[label]=result
        save(args.out/'progress.json',results)
    selected=candidates[0] if candidates else None
    count=192
    held={}
    retention={}
    if selected:
        for name,root,checkpoint in (('parent',PARENT,MODEL),(selected,args.training,checkpoints[selected])):
            folder=args.out/name
            folder.mkdir(exist_ok=True)
            actor=load_actor(rig,ref,read(root/'protocol.json'),checkpoint)
            held[name]=impact(rig,ref,actor,world,folder,'held-onset',1.6,1,schedule=[2,6],distance=.46)
        count+=32
        held_pass=all(row['parity']['passed'] and row['chest_first'] for row in held.values())
        held_pass &= all(held[selected][b]['recovery_passes']>=held['parent'][b]['recovery_passes'] for b in ('native','gpu'))
        if not held_pass:
            selected=None
        else:
            actor=load_actor(rig,ref,read(args.training/'protocol.json'),checkpoints[selected])
            for backend in ('native','gpu'):
                path=args.out/f'selected-retention-{backend}'
                retention[backend]=evaluate_duration(rig,ref,'target_pd',backend,actor,path,seconds=120)
                verify_duration(path,120)
                compact(path)
            count+=16
            if not all(r['standing_passes']==8 for r in retention.values()):
                selected=None
    save(args.out/'report.json',dict(results=results,selected=selected,
        checkpoint=str(checkpoints[selected]) if selected else None,selected_root=str(args.training) if selected else None,
        retention=retention,held_out=held,physical_cases=count,training_updates=0,
        arena_change_eligible=bool(selected),experimental_preview=True,baseline_checkpoint=str(MODEL)))
    seal(args.out,inputs,[],verified=True,physical_cases=count,training_updates=0)
    print('Final selection',selected,'new evaluation cases',count,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run(args)
