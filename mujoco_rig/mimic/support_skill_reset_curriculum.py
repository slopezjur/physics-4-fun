"""Compare original resets and admitted prepared poses under one lift objective."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate
from . import support_skill_learning as learning
from . import run_support_skill_study as study


def environment_type(base,prepared):
    class HoldEnvironment(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.prepared=prepared
            if prepared:
                from .support_skill_core import fit_goal
                from .support_skill_physical_teacher import clear_arms
                from .static_support import solve_support
                cache={};poses=[]
                references=self.task.reference.qpos.cpu().numpy()
                for i,g in enumerate(self.goals):
                    key=(i%8,g.side,g.shift,g.lift)
                    if key not in cache:
                        start=clear_arms(self.rig.model,references[i%8],.10)
                        pose,fit=fit_goal(self.rig.model,start,shift=g.shift*(1 if g.side==0 else -1),lift=g.lift,distance=0,swing=g.side)
                        support=solve_support(self.rig,pose,support_names=tuple(b+'_'+('R' if g.side==0 else 'L') for b in ('Foot','Toe')))
                        if not support['feasible'] or fit['penetration_m']>.0002:raise ValueError('Prepared hold pose failed admission')
                        cache[key]=pose
                    poses.append(cache[key])
                self.prepared_qpos=np.asarray(poses)

        def reset(self,env_ids=None):
            super().reset(env_ids)
            if self.prepared:
                ids=self.task.ids if env_ids is None else env_ids
                self.engine.reset_envs(ids,self.prepared_qpos[ids.cpu().numpy()],np.zeros((len(ids),self.rig.model.nv)))
            return self.observations(),{}

        def goal_features(self):
            value=super().goal_features();lift=self.coefficients()[:,1]
            sign=torch.where(self.side==0,-1.,1.)
            value[:,0]=(1-lift)*value[:,0]+lift*sign
            return value

        def contract(self):
            return dict(super().contract(),prepared_pose_reset=self.prepared,
                reset_velocity='zero',reset_pose_source='Admitted 11cm / 20mm IK pose with hands cleared by 10cm; not a dynamically settled state',
                lift_load_goal='100 percent on the support foot at full requested lift',
                external_support=False)

    return HoldEnvironment


if __name__=='__main__':
    protocol=Path(sys.argv[sys.argv.index('--protocol')+1])
    spec=json.loads(protocol.read_text())
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from . import support_skill_core as core
    base=core.SkillEnvironment;original=study.evaluate

    def evaluate(rig,anchor,arm,backend,out,policy=None):
        with patch.object(core,'SkillEnvironment',environment_type(base,arm.get('prepared_reset',False))):
            return original(rig,anchor,arm,backend,out,policy)

    with patch.object(core,'SkillEnvironment',environment_type(base,spec['prepared_reset'])),patch.object(study,'evaluate',evaluate):
        learning.main()
