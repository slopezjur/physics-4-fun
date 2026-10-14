"""Native development teacher: choose a free foot and place it from current balance state."""
import mujoco
import numpy as np
import torch

from .recovery_motor import authority_gate


def bridge_source_action(task,action,observation):
    """Convert an old physical target to the new action units before any correction."""
    control=task.engine.policy_control
    gate=authority_gate(observation)
    radius=.25+(control.maximum_radians-.25)*gate[:,None]
    return torch.where(control.leg_mask,action*.25/radius,action)


class CaptureStepTeacher:
    """Offline expert candidate; transitions depend on measured clearance/contact, never a step clock."""
    def __init__(self,task,actor):
        if task.device!='cpu' or task.recovery_motor_radians is None:
            raise ValueError('Teacher requires the isolated native recovery motor contract')
        self.task,self.actor=task,actor
        count=task.get_num_envs()
        self.phase=np.zeros(count,dtype=int)
        self.foot=np.zeros(count,dtype=int)
        self.goal=np.zeros((count,3))
        self.start=np.zeros((count,3))
        self.rotation=np.zeros((count,3,3))
        self.triggers=np.zeros((count,2),dtype=int)
        self.kinematic_data=mujoco.MjData(task.rig.model)
        self.jac=np.zeros((3,task.rig.model.nv));self.jacr=np.zeros_like(self.jac)
        self.body=[task.rig.model.body(f'Foot_{s}').id for s in ('L','R')]
        self.joints=[];self.actions=[]
        names=task.observation_contract()['actuator_names']
        for side in ('L','R'):
            joint_names=[f'Thigh_{side}_{axis}' for axis in ('rx','ry','rz')]+[
                f'Shin_{side}_rx',f'Foot_{side}_rx',f'Foot_{side}_rz']
            self.joints.append(np.array([task.rig.model.joint(n).id for n in joint_names]))
            self.actions.append(np.array([names.index(n) for n in joint_names]))

    @torch.no_grad()
    def __call__(self,observation):
        task=self.task;control=task.engine.policy_control
        action=bridge_source_action(task,self.actor(observation),observation)
        self.phase[task.steps.numpy()==0]=0
        gate=authority_gate(observation).numpy()
        positions=task.engine.get_body_pos(0)[:,task.feet].numpy()
        loads=(observation[:,93:95]*1000).numpy()
        com=task.center_of_mass().numpy()
        velocity=task.engine.get_center_of_mass_velocity().numpy()
        reference,_=task.sample_reference(task.offset)
        reference=reference.numpy()
        qpos=task.engine.native_qpos().numpy()
        for world in range(len(action)):
            if self.phase[world]==0:
                foot=int(np.argmin(loads[world]));other=1-foot
                if gate[world]<=0 or loads[world,foot]>=5 or loads[world,other]<50:
                    continue
                omega=np.sqrt(max(.05,com[world,2]-positions[world,:,2].min())/9.81)
                target=com[world,:2]+velocity[world,:2]*omega
                delta=target-positions[world,foot,:2]
                distance=np.linalg.norm(delta)
                if distance<.04:continue
                self.foot[world]=foot;self.start[world]=positions[world,foot]
                self.goal[world]=positions[world,foot]
                self.goal[world,:2]+=delta*min(1.,.08/distance)
                self.rotation[world]=task.engine.datas[world].xmat[self.body[foot]].reshape(3,3)
                self.phase[world]=1;self.triggers[world,foot]+=1
            foot=self.foot[world]
            current=positions[world,foot]
            if self.phase[world]==1 and current[2]>=self.start[world,2]+.025:
                self.phase[world]=2
            if self.phase[world]==2 and np.linalg.norm(current[:2]-self.goal[world,:2])<.025:
                self.phase[world]=3
            if self.phase[world]==3 and loads[world,foot]>=30:
                self.phase[world]=0
                continue
            goal=self.goal[world].copy()
            if self.phase[world]==1:goal[:2]=self.start[world,:2]
            goal[2]=self.start[world,2]+(.035 if self.phase[world]<3 else 0.)
            joints=self.joints[foot];indices=task.rig.model.jnt_qposadr[joints]
            dofs=task.rig.model.jnt_dofadr[joints];motors=self.actions[foot]
            radius=.25+(control.maximum_radians-.25)*gate[world]
            ranges=np.where(control.leg_mask[motors].numpy(),radius,.25)
            lower=np.maximum(task.rig.model.jnt_range[joints,0],reference[world,indices]-ranges)
            upper=np.minimum(task.rig.model.jnt_range[joints,1],reference[world,indices]+ranges)
            data=self.kinematic_data;data.qpos[:]=qpos[world]
            for _ in range(5):
                mujoco.mj_kinematics(task.rig.model,data);mujoco.mj_comPos(task.rig.model,data)
                mujoco.mj_jacBody(task.rig.model,data,self.jac,self.jacr,self.body[foot])
                current_rotation=data.xmat[self.body[foot]].reshape(3,3)
                angular=.5*np.cross(current_rotation.T,self.rotation[world].T).sum(0)
                error=np.concatenate((goal-data.xpos[self.body[foot]],.10*angular))
                jac=np.vstack((self.jac[:,dofs],.10*self.jacr[:,dofs]))
                delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-4,error)
                data.qpos[indices]=np.clip(data.qpos[indices]+np.clip(delta,-.12,.12),lower,upper)
            action[world,motors]=torch.from_numpy((data.qpos[indices]-reference[world,indices])/ranges).float()
        return action.clamp(-1,1)
