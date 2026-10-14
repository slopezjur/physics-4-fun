"""Repair a contaminated repeat-shot comparison using prospectively clear geometry."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate


def main(args):
    from .run_perturb_preview import OLD,OLD_MODEL,PARENT,PARENT_MODEL,sealed
    from .rig import Rig
    from .stand_experiment_evidence import load_actor,seal
    from .evaluate_stand_ball_boundary import evaluate as native
    from .evaluate_stand_ball_gpu import evaluate as gpu,compare
    from .run_chest_repeated_screen import wrap_evaluator
    from .ball_study_support import body_observer,save_bodies
    from .evaluate_stand_sustained import evaluate_duration,verify_duration
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    assert not (args.out/'report.json').exists()
    original = read(args.out/'protocol.json')
    for path,expected in original['inputs'].items():
        assert digest(Path(path))==expected,path
    results = read(args.out/'progress.json')
    assert set(results)=={'stage12','stage33','quiet','repeated'}
    initial_hashes = {str(p):digest(p) for p in args.out.rglob('*') if p.is_file()}
    inputs = dict(original['inputs'],**initial_hashes)
    inputs[str(Path(__file__))] = digest(Path(__file__))
    save(args.out/'completion-protocol.json',dict(inputs=inputs,
        reason='Stage33 repeat case 7: native second insertion overlaps dummy; native/GPU total impulse admission fails. Preserve and exclude original Stage33 repeat group.',
        change='Repeat all four actors at common 0.46 m clear launch distance; same 5 kg,1.6 m/s,+X,5/9 s,16 s end,8 poses,both backends.',
        additional_impact_cases=64,remaining_retention_cases=16,maximum_total_physical_cases=464,
        selection='Original rule on uncontaminated single-hit controls plus replacement repeated condition; no success threshold changes.',
        training_updates=0))
    actors=dict(stage12=(OLD,OLD_MODEL),stage33=(PARENT,PARENT_MODEL),
                quiet=(args.training,args.training/'quiet/final/model.pt'),
                repeated=(args.training,args.training/'single_ball/final/model.pt'))
    rig=Rig.load(args.training/'dummy.xml',policy_exclude=())
    ref,world=args.training/'stationary_reference.npz',args.training/'ball-5kg.xml'
    for name,(root,checkpoint) in actors.items():
        sealed(root,[checkpoint,checkpoint.with_suffix('.amp.pt')])
        actor=load_actor(rig,ref,read(root/'protocol.json'),checkpoint)
        rows={}
        cfg=dict(seconds=16,launch_seconds=5,launch_distance_m=.46)
        for backend,function in (('native',native),('gpu',gpu)):
            folder=args.out/name/f'hit-repeat-clear-{backend}'
            function=wrap_evaluator(function,backend,dict(conditions=[dict(name='repeat-clear',launch_times_seconds=[5,9])]))
            if backend=='native':
                journal={}
                with body_observer(journal,int(np.ceil(16/(rig.model.opt.timestep*rig.decimation)))):
                    rows[backend]=function(rig,ref,actor,world,folder,cfg,1.6,1)
                save_bodies(folder,journal)
                bodies=read(folder/'body-report.json')
                checks=[]
                for case,body in zip(rows[backend]['cases'],bodies['cases']):
                    for encounter in case['encounters']:
                        start=encounter['launch_frame']
                        end=int(np.ceil(9/(rig.model.opt.timestep*rig.decimation))) if encounter['index']==0 else 10**9
                        clusters=[c for c in body['clusters'] if start<=c['first_frame']<end]
                        checks.append(not encounter['hit'] or bool(clusters and 'Chest' in clusters[0]['bodies'] and 'Head' not in clusters[0]['bodies']))
                rows['chest_first']=all(checks)
            else:
                rows[backend]=function(rig,ref,actor,world,folder,cfg,1.6,1)
        dt=rig.model.opt.timestep*rig.decimation
        rows['parity']=compare(rows['native'],rows['gpu'],dt)
        for x,y in zip(rows['native']['cases'],rows['gpu']['cases']):
            for a,b in zip(x['encounters'],y['encounters']):
                exact=all(a[k]==b[k] for k in ('hit','prequalified','survived','final_valid','passed'))
                timing=all((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and abs(a[k]-b[k])<=2*dt)
                           for k in ('first_contact_seconds','last_contact_seconds'))
                impulse=abs(a['total_normal_impulse_ns']-b['total_normal_impulse_ns'])<=.25+.1*a['total_normal_impulse_ns']
                rows['parity']['passed'] &= exact and timing and impulse
        rows['parity']['passed'] &= rows['native']['native_repeat_launch_clear']
        rows['trace_label']='repeat-clear'
        results[name]['original_repeat_025m']=deepcopy(results[name]['impacts']['repeat'])
        results[name]['impacts']['repeat']=rows
        results[name]['admitted']=all(r['chest_first'] and r['parity']['passed'] for r in results[name]['impacts'].values())
        save(args.out/'completion-progress.json',results)
        print(name,'clear-repeat',{b:rows[b]['recovery_passes'] for b in ('native','gpu')},'admitted',results[name]['admitted'],flush=True)
    baseline,candidate=results['stage33'],results['repeated']
    quiet_pass=all(q['standing_passes']==8 for q in candidate['quiet'].values())
    retain=all(candidate['impacts'][k][b]['recovery_passes']>=baseline['impacts'][k][b]['recovery_passes']
               for k in baseline['impacts'] for b in ('native','gpu'))
    gain=all(candidate['impacts']['repeat'][b]['recovery_passes']>baseline['impacts']['repeat'][b]['recovery_passes'] for b in ('native','gpu'))
    selected='repeated' if quiet_pass and retain and gain and candidate['admitted'] else 'stage33'
    assert results[selected]['admitted'] and all(q['standing_passes']==8 for q in results[selected]['quiet'].values())
    root,model=actors[selected]
    actor=load_actor(rig,ref,read(root/'protocol.json'),model)
    retention={}
    print('Selected',selected,'starting 120-second retention',flush=True)
    for backend in ('native','gpu'):
        folder=args.out/f'selected-retention-{backend}'
        retention[backend]=evaluate_duration(rig,ref,'target_pd',backend,actor,folder,seconds=120)
        verify_duration(folder,120)
    assert all(r['standing_passes']==8 for r in retention.values())
    save(args.out/'report.json',dict(results=results,selected=selected,checkpoint=str(model),selected_root=str(root),
        retention=retention,physical_cases=464,excluded_original_repeat_cases=16,admitted_cases=448,training_updates=0,
        candidate_retains=retain,candidate_repeated_gain=gain,experimental_preview=True,
        repeated_comparison_distance_m=.46,original_025m_stage33_repeat_admitted=False))
    seal(args.out,inputs,[],verified=True,physical_cases=464,excluded_cases=16,training_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    main(args)
