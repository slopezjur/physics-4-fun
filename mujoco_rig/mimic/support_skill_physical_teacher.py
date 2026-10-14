"""Joint-only equilibrium teacher and explicit pose-reset feasibility diagnostics."""
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from .runtime import activate
from . import run_support_skill_study as study
from .support_skill_experiments import metric_adapter
from .support_skill_exact_goals import exact_goals


def clear_arms(model,pose,distance):
    """Private IK opens the hands sideways; no live body or root is repositioned."""
    data=mujoco.MjData(model);data.qpos[:]=pose;mujoco.mj_forward(model,data)
    targets={side:data.xpos[model.body('Hand_'+side).id].copy()+[0,sign*distance,0] for side,sign in [('L',-1),('R',1)]}
    for side in ('L','R'):
        joints=[i for i in range(1,model.njnt) if model.joint(i).name.startswith(('UpperArm_'+side,'Forearm_'+side))]
        qi,vi=model.jnt_qposadr[joints],model.jnt_dofadr[joints]
        jac=np.zeros((3,model.nv));body=model.body('Hand_'+side).id
        for _ in range(60):
            mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            mujoco.mj_jacBody(model,data,jac,None,body)
            a=jac[:,vi];error=targets[side]-data.xpos[body]
            delta=np.linalg.solve(a.T@a+1e-4*np.eye(len(vi)),a.T@error)
            data.qpos[qi]=np.clip(data.qpos[qi]+np.clip(delta,-.05,.05),model.jnt_range[joints,0],model.jnt_range[joints,1])
    return data.qpos.copy()


def evaluate(rig,anchor,arm,backend,out,policy=None):
    from . import support_skill_core as core
    from .static_support import solve_support
    base=core.SkillEnvironment

    class PhysicalTeacher(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            q=self.task.reference.qpos.cpu().numpy();qi=rig.model.jnt_qposadr[rig.model.actuator_trnid[rig.actuators,0]]
            poses=[];offsets=[];torques=[];self.static_reports=[];cache={}
            for i,g in enumerate(self.goals):
                key=(i%8,g.side,g.shift,g.lift,g.distance)
                if key not in cache:
                    reference=q[i%8];start=clear_arms(rig.model,reference,arm.get('arm_clearance',0.))
                    targets=[];checks=[];supports=[]
                    for lift,distance in ((0.,0.),(g.lift,0.),(g.lift,g.distance)):
                        target,fit=core.fit_goal(rig.model,start,shift=g.shift*(1 if g.side==0 else -1),lift=lift,distance=distance,swing=g.side)
                        names=tuple(f'{body}_{side}' for body in ('Foot','Toe') for side in ('L','R') if not lift or side!=('L','R')[g.side])
                        support=solve_support(rig,target,support_names=names)
                        targets.append(target);checks.append(fit);supports.append(support)
                    values=np.stack([r.get('normalized_motor_torque',[0.]*33) for r in supports])/4.
                    offsets=np.stack((targets[0][qi]-reference[qi],targets[1][qi]-targets[0][qi],targets[2][qi]-targets[1][qi]))
                    cache[key]=(targets,offsets,values,dict(fits=checks,equilibrium=supports))
                targets,delta,feedforward,reports=cache[key]
                poses.append(targets[1]);torques.append(feedforward);self.static_reports.append(reports)
                self.basis[i]=torch.tensor(delta,device=self.device,dtype=torch.float32)
            self.hold_poses=np.asarray(poses)
            self.feedforward=torch.tensor(np.asarray(torques),device=self.device,dtype=torch.float32)

        def reset(self,env_ids=None):
            result=super().reset(env_ids)
            if arm.get('pose_reset',False):
                ids=self.task.ids if env_ids is None else env_ids
                selected=ids.cpu().numpy()
                self.engine.reset_envs(ids,self.hold_poses[selected],np.zeros((len(selected),rig.model.nv)))
                result=(self.observations(),{})
            return result

        def teacher_action(self):
            c=self.coefficients()
            offset=torch.einsum('ni,nij->nj',c,self.basis)
            if arm.get('absolute_equilibrium',True):
                q,_=self.task.sample_reference(self.task.offset+self.task.steps*self.dt)
                control=self.engine.policy_control
                source=self.anchor(self.task.observations())
                source_offset=(q[:,control.q_indices]+.25*source).clamp(control.lower,control.upper)-q[:,control.q_indices]
                offset+=c[:,:1]*(self.feedforward[:,0]-source_offset)
                offset+=c[:,1:2]*(self.feedforward[:,1]-self.feedforward[:,0])
                offset+=c[:,2:3]*(self.feedforward[:,2]-self.feedforward[:,1])
            return (offset/self.radius).clamp(-1,1)

        def contract(self):
            return dict(super().contract(),teacher='scripted joint-target equilibrium; not a learned actor',
                pose_reset=arm.get('pose_reset',False),reset_scope='Initial qpos only; no root writes or support during rollouts',
                static_diagnostics=self.static_reports,arm_clearance_m=arm.get('arm_clearance',0.),
                absolute_equilibrium=arm.get('absolute_equilibrium',True),
                source_actor_cancellation='At full command only, scripted offsets cancel source residual to test plant feasibility')

    with patch.object(core,'SkillEnvironment',PhysicalTeacher):
        return original(rig,anchor,dict(arm,action_mode='full'),backend,out,policy)


original=study.evaluate

if __name__=='__main__':
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    with exact_goals(),metric_adapter(),patch.object(study,'evaluate',evaluate):study.main()
