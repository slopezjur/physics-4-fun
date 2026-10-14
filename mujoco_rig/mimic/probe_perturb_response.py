"""Frozen sensitivity probe; altered velocity inputs are never an export candidate."""
from copy import deepcopy
from pathlib import Path

import torch

from .audit_stand_failures import digest, read, save
from .runtime import activate
from .run_perturb_distance import PARENT, MODEL, ASSESSMENT
from .evaluate_perturb_distance import impact

OUT = Path('logs/mimickit-training/perturb-response-probe-20261008-01')


class VelocityProbe(torch.nn.Module):
    def __init__(self, actor, gain):
        super().__init__()
        self.actor = actor
        self.register_buffer('gain', torch.tensor(float(gain)))

    def forward(self, observation):
        changed = observation.clone()
        changed[:,9:11] *= self.gain
        return self.actor(changed)


def run():
    from .rig import Rig
    from .stand_experiment_evidence import load_actor, seal
    from .run_perturb_preview import sealed
    from util import mp_util
    mp_util.init(0,1,'cuda:0',None)
    sealed(PARENT,[MODEL,MODEL.with_suffix('.amp.pt')])
    OUT.mkdir(exist_ok=False)
    paths = [Path(__file__), Path('mujoco_rig/mimic/PERTURB_RESPONSE_PLAN.md'), MODEL,
             PARENT/'protocol.json', ASSESSMENT/'report.json',
             Path('logs/mimickit-training/perturb-response-audit-20261008-01/report.json')]
    inputs = {str(p):digest(p) for p in paths}
    save(OUT/'protocol.json',dict(inputs=inputs,gains=[1.,.75,1.25],
        schedules=[[5,9],[2,6]],cases=96,training_updates=0,
        factor='Scale world horizontal root velocity input only; physical velocity, delay and action mapping unchanged',
        scope='Diagnostic sensitivity, not learned recovery or promotion. No sweep beyond these three gains.'))
    rig=Rig.load(PARENT/'dummy.xml',policy_exclude=())
    ref,world=PARENT/'stationary_reference.npz',PARENT/'ball-5kg.xml'
    actor=load_actor(rig,ref,read(PARENT/'protocol.json'),MODEL)
    result={}
    for gain in (1.,.75,1.25):
        folder=OUT/f'gain-{gain:g}'
        folder.mkdir()
        result[str(gain)]={}
        for schedule in ([5,9],[2,6]):
            row=impact(rig,ref,VelocityProbe(deepcopy(actor),gain),world,folder,
                f'onset-{schedule[0]}',1.6,1,schedule=schedule,distance=.46)
            assert row['parity']['passed'] and row['chest_first']
            result[str(gain)][str(schedule[0])]=row
        save(OUT/'progress.json',result)
    save(OUT/'report.json',dict(results=result,physical_cases=96,training_updates=0,
        diagnostic_only=True,arena_change_eligible=False))
    seal(OUT,inputs,[],verified=True,physical_cases=96,training_updates=0)


if __name__=='__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    torch.set_num_threads(1)
    run()
