"""Ablate equilibrium torque differences through bounded, delayed joint targets."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from . import run_support_skill_study as study
from .support_skill_experiments import metric_adapter


def evaluate(rig,anchor,arm,backend,out,policy=None):
    from . import support_skill_core as core
    from .static_support import solve_support
    base=core.SkillEnvironment

    class FeedforwardEnvironment(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.action_mode!='full':raise ValueError('Feedforward diagnostic requires full joint offsets')
            q=self.task.reference.qpos.cpu().numpy();values=[];self.equilibrium=[]
            for i,goal in enumerate(self.goals):
                pose=q[i%8]
                endpoint,fit=core.fit_goal(rig.model,pose,shift=goal.shift*(1 if goal.side==0 else -1),lift=0,distance=0,swing=goal.side)
                a,b=solve_support(rig,pose),solve_support(rig,endpoint)
                if not a['feasible'] or not b['feasible']:raise ValueError('Static support necessary condition failed')
                values.append((np.array(b['normalized_motor_torque'])-np.array(a['normalized_motor_torque']))/4.)
                self.equilibrium.append(dict(reference=a,target=b,fit=fit))
            self.feedforward=torch.tensor(np.asarray(values),device=self.device,dtype=torch.float32)

        def teacher_action(self):
            c=self.coefficients()
            offset=torch.einsum('ni,nij->nj',c,self.basis)
            offset+=arm.get('feedforward',0.)*c[:,:1]*self.feedforward
            return (offset/self.radius).clamp(-1,1)

        def contract(self):
            return dict(super().contract(),equilibrium_feedforward_weight=arm.get('feedforward',0.),
                equilibrium=self.equilibrium,feedforward_formula='coefficient*(target_equilibrium_torque-reference_equilibrium_torque)/(4*torque_limit)',
                no_inverse_dynamics_support_force=True,quasistatic_feasibility_is_not_dynamic_success=True)

    adjusted=dict(arm,action_mode='full')
    with patch.object(core,'SkillEnvironment',FeedforwardEnvironment):
        return original_evaluate(rig,anchor,adjusted,backend,out,policy)


original_evaluate=study.evaluate

if __name__=='__main__':
    with metric_adapter(),patch.object(study,'evaluate',evaluate):study.main()
