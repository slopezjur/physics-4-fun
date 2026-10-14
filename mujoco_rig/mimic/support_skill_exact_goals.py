"""Use the requested positive foot lift, retaining the sealed initial prototype."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from .support_skill_experiments import metric_adapter
from . import run_support_skill_study as study
from .runtime import activate
from pathlib import Path


@contextmanager
def exact_goals():
    from . import support_skill_core as core
    original=core.SkillEnvironment

    class ExactLiftEnvironment(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            q=self.task.reference.qpos.cpu().numpy()
            qi=self.rig.model.jnt_qposadr[self.rig.model.actuator_trnid[self.rig.actuators,0]]
            cache={}
            for i,g in enumerate(self.goals):
                if not 0<g.lift<.02:continue
                key=(i%8,g.side,g.shift,g.lift,g.distance)
                if key not in cache:
                    poses,checks=[],[]
                    for lift,distance in ((0.,0.),(g.lift,0.),(g.lift,g.distance)):
                        pose,check=core.fit_goal(self.rig.model,q[i%8],shift=g.shift*(1 if g.side==0 else -1),
                            lift=lift,distance=distance,swing=g.side)
                        poses.append(pose);checks.append(check)
                    values=np.stack((poses[0][qi]-q[i%8,qi],poses[1][qi]-poses[0][qi],poses[2][qi]-poses[1][qi]))
                    cache[key]=(torch.tensor(values,device=self.device,dtype=torch.float32),checks)
                self.basis[i],self.fits[i]=cache[key]

        def contract(self):
            return dict(super().contract(),positive_lift_template='exact requested height; no 20mm minimum')

    with patch.object(core,'SkillEnvironment',ExactLiftEnvironment):yield


if __name__=='__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    with exact_goals(),metric_adapter():study.main()
