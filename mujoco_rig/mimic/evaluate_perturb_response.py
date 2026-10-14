"""Primary learned recovery comparison with unchanged physical admission gates."""
import argparse
from pathlib import Path

import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .run_perturb_distance import ASSESSMENT, MODEL, compact
from .run_perturb_preview import sealed
from .evaluate_perturb_distance import impact


def run(args):
    from .rig import Rig
    from .stand_experiment_evidence import load_actor, seal
    from .evaluate_stand_sustained import evaluate_duration, verify_duration
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    models={label:args.training/label/'final/model.pt' for label in ('control','long_return')}
    for model in models.values():
        sealed(args.training,[model,model.with_suffix('.amp.pt')])
    sealed(ASSESSMENT,[ASSESSMENT/'report.json'])
    baseline=read(ASSESSMENT/'report.json')['results']['repeated']
    args.out.mkdir(exist_ok=False)
    paths=[Path(__file__),Path('mujoco_rig/mimic/evaluate_perturb_distance.py'),
        Path('mujoco_rig/mimic/PERTURB_RESPONSE_LEARNING.md'),args.training/'protocol.json',
        args.training/'final-verification.json',ASSESSMENT/'report.json',
        *[p for m in models.values() for p in (m,m.with_suffix('.amp.pt'))]]
    inputs={str(p):digest(p) for p in paths}
    save(args.out/'protocol.json',dict(inputs=inputs,training=str(args.training),physical_cases=192,
        qualification='All primary single counts retained, repeated 8/8, quiet 8/8, both backends admitted',
        held_out_and_long_retention='Separate comparison after both seeds finish; no primary-only export'))
    rig=Rig.load(args.training/'dummy.xml',policy_exclude=())
    ref,world=args.training/'stationary_reference.npz',args.training/'ball-5kg.xml'
    results={}
    for label,model in models.items():
        actor=load_actor(rig,ref,read(args.training/'protocol.json'),model)
        folder=args.out/label
        folder.mkdir()
        row=dict(impacts={},quiet={},checkpoint=str(model))
        row['impacts']['repeat']=impact(rig,ref,actor,world,folder,'repeat-clear',1.6,1,schedule=[5,9],distance=.46)
        for name,speed,sign in (('easy-plus',1.4,1),('easy-minus',1.4,-1),('strong-plus',1.6,1),('strong-minus',1.6,-1)):
            row['impacts'][name]=impact(rig,ref,actor,world,folder,name,speed,sign)
        for backend in ('native','gpu'):
            path=folder/f'quiet-{backend}'
            row['quiet'][backend]=evaluate_duration(rig,ref,'target_pd',backend,actor,path,seconds=30)
            verify_duration(path,30)
            compact(path)
        row['admitted']=all(r['parity']['passed'] and r['chest_first'] for r in row['impacts'].values())
        retained=all(row['impacts'][k][b]['recovery_passes']>=baseline['impacts'][k][b]['recovery_passes']
            for k in baseline['impacts'] for b in ('native','gpu'))
        row['qualifies']=bool(row['admitted'] and retained and
            all(row['impacts']['repeat'][b]['recovery_passes']==8 for b in ('native','gpu')) and
            all(q['standing_passes']==8 for q in row['quiet'].values()))
        results[label]=row
        save(args.out/'progress.json',results)
        print('Primary',label,'qualifies',row['qualifies'],flush=True)
    save(args.out/'report.json',dict(results=results,physical_cases=192,training_updates=0,
        primary_qualifiers=[k for k,r in results.items() if r['qualifies']],arena_change_eligible=False,
        baseline_checkpoint=str(MODEL),training=str(args.training)))
    seal(args.out,inputs,[],verified=True,physical_cases=192,training_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run(args)
