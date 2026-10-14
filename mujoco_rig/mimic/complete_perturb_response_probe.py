"""Preserve an inadmissible sensitivity trial and finish the predeclared probe."""
from copy import deepcopy
from pathlib import Path

import torch

from .audit_stand_failures import digest, read, save
from .probe_perturb_response import OUT, VelocityProbe
from .run_perturb_distance import PARENT, MODEL
from .evaluate_perturb_distance import impact
from .runtime import activate


def run():
    from .rig import Rig
    from .stand_experiment_evidence import load_actor, seal
    from .evaluate_stand_ball_gpu import compare
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    protocol=read(OUT/'protocol.json')
    inputs={**protocol['inputs'],str(Path(__file__)):digest(Path(__file__))}
    save(OUT/'completion-protocol.json',dict(inputs=inputs,
        exclusion='gain 0.75 / onset 2 failed backend admission. Preserve all 16 raw cases but exclude this entire condition from quantitative conclusions.',
        unchanged_conditions='Finish only the already declared gain 1.25 at onsets 5 and 2; no replacement sweep.',
        raw_cases=96,excluded_cases=16,admitted_cases=80))
    rig=Rig.load(PARENT/'dummy.xml',policy_exclude=())
    ref,world=PARENT/'stationary_reference.npz',PARENT/'ball-5kg.xml'
    actor=load_actor(rig,ref,read(PARENT/'protocol.json'),MODEL)
    results={}
    for gain in (1.,.75):
        results[str(gain)]={}
        for onset in (5,2):
            label=f'gain-{gain:g}/hit-onset-{onset}'
            native,gpu=(read(OUT/f'{label}-{b}'/'report.json') for b in ('native','gpu'))
            parity=compare(native,gpu,rig.model.opt.timestep*rig.decimation)
            results[str(gain)][str(onset)]=dict(native=native,gpu=gpu,parity=parity,
                admitted=not(gain==.75 and onset==2),excluded=bool(gain==.75 and onset==2))
    folder=OUT/'gain-1.25'
    folder.mkdir()
    results['1.25']={}
    for schedule in ([5,9],[2,6]):
        row=impact(rig,ref,VelocityProbe(deepcopy(actor),1.25),world,folder,
            f'onset-{schedule[0]}',1.6,1,schedule=schedule,distance=.46)
        row['admitted']=bool(row['parity']['passed'] and row['chest_first'])
        row['excluded']=not row['admitted']
        results['1.25'][str(schedule[0])]=row
        save(OUT/'completion-progress.json',results)
    excluded=sum(16 for gain in results.values() for row in gain.values() if row['excluded'])
    save(OUT/'report.json',dict(results=results,physical_cases=96,excluded_cases=excluded,admitted_cases=96-excluded,
        training_updates=0,diagnostic_only=True,arena_change_eligible=False))
    seal(OUT,inputs,[],verified=True,physical_cases=96,excluded_cases=excluded,training_updates=0)


if __name__=='__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run()
