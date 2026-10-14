"""Untrained-timing and retention gates after both matched optimizer seeds."""
from pathlib import Path

import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .run_perturb_distance import PARENT, MODEL, compact
from .evaluate_perturb_distance import impact
from .run_perturb_preview import sealed

BASE=Path('logs/mimickit-training')
OUT=BASE/'perturb-response-final-20261008-01'


def run():
    from .rig import Rig
    from .stand_experiment_evidence import load_actor, seal
    from .evaluate_stand_sustained import evaluate_duration, verify_duration
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    candidates={}
    sources=[]
    primary={}
    for index in (1,2):
        root=BASE/f'perturb-response-evaluation-20261008-{index:02d}'
        sealed(root,[root/'report.json'])
        source=read(root/'report.json')
        primary[str(index)]=source
        sources.extend([root/'report.json',root/'final-verification.json'])
        for label in source['primary_qualifiers']:
            candidates[f'{index}-{label}']=dict(training=source['training'],evaluation=str(root),
                model=source['results'][label]['checkpoint'],label=label,seed_index=index)
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),Path('mujoco_rig/mimic/PERTURB_RESPONSE_LEARNING.md'),
        PARENT/'protocol.json',MODEL,MODEL.with_suffix('.amp.pt'),*sources]
    for value in candidates.values():
        model=Path(value['model'])
        sealed(Path(value['training']),[model,model.with_suffix('.amp.pt')])
        paths.extend([model,model.with_suffix('.amp.pt')])
    inputs={str(p):digest(p) for p in paths}
    save(OUT/'protocol.json',dict(inputs=inputs,candidates=candidates,schedule=[4,8],
        ranking='Highest timing recovery count, then earlier seed, then control before treatment',
        parent_timing='Measured even without a qualifier to characterize retained boundary',
        maximum_cases=144))
    rig=Rig.load(PARENT/'dummy.xml',policy_exclude=())
    ref,world=PARENT/'stationary_reference.npz',PARENT/'ball-5kg.xml'
    timing={}
    entries=dict(parent=dict(training=str(PARENT),model=str(MODEL)),**candidates)
    for name,item in entries.items():
        actor=load_actor(rig,ref,read(Path(item['training'])/'protocol.json'),Path(item['model']))
        folder=OUT/name
        folder.mkdir()
        timing[name]=impact(rig,ref,actor,world,folder,'untrained-onset',1.6,1,schedule=[4,8],distance=.46)
        save(OUT/'timing-progress.json',timing)
    count=16*len(entries)
    qualified=[]
    retention={}
    for name,item in candidates.items():
        admitted=all(timing[k]['parity']['passed'] and timing[k]['chest_first'] for k in ('parent',name))
        retained=all(timing[name][b]['recovery_passes']>=timing['parent'][b]['recovery_passes'] for b in ('native','gpu'))
        if not admitted or not retained:
            continue
        actor=load_actor(rig,ref,read(Path(item['training'])/'protocol.json'),Path(item['model']))
        retention[name]={}
        for backend in ('native','gpu'):
            path=OUT/name/f'retention-{backend}'
            retention[name][backend]=evaluate_duration(rig,ref,'target_pd',backend,actor,path,seconds=120)
            verify_duration(path,120)
            compact(path)
        count+=16
        if all(r['standing_passes']==8 for r in retention[name].values()):
            qualified.append(name)
    qualified.sort(key=lambda k:(-timing[k]['native']['recovery_passes'],candidates[k]['seed_index'],candidates[k]['label']!='control'))
    selected=qualified[0] if qualified else None
    recipe_replicated=all(
        source['results']['long_return']['qualifies'] and
        source['results']['long_return']['impacts']['repeat']['native']['recovery_passes']>
        source['results']['control']['impacts']['repeat']['native']['recovery_passes']
        for source in primary.values())
    save(OUT/'report.json',dict(selected=selected,candidates=candidates,timing=timing,retention=retention,
        primary=primary,recipe_improvement_replicated=recipe_replicated,physical_cases=count,
        training_updates=0,arena_change_eligible=bool(selected),current_arena_checkpoint=str(MODEL)))
    seal(OUT,inputs,[],verified=True,physical_cases=count,training_updates=0)
    print(dict(selected=selected,recipe_replicated=recipe_replicated,physical_cases=count),flush=True)


if __name__=='__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run()
