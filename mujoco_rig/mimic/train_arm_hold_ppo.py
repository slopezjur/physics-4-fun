"""Finite-horizon on-policy arm-hold learning with exact deterministic adapter parity."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .runtime import activate
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_support import PARENT,evaluate
from .run_learned_handoff import PRIOR,GuardedLearner
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold import episode

INITIAL=Path('logs/mimickit-training/arm-hold-queue-ablation-20261008-01/model/final.pt')
BASELINE=INITIAL.parent.parent/'nominal/trace.npz'
OUT=Path('logs/mimickit-training/arm-hold-ppo-20261008-01')
TRAIN_IDS=torch.tensor([0,1,2,3,4,5,8,9,10,11,12,13])


class ArmRollout:
    """Physical command adapter shared by stochastic learning and mean-policy parity."""
    def __init__(self,env,anchor,poses):
        from .predictive_target_support import blend,blend_pose
        self.env,self.anchor,self.poses=env,anchor,poses
        self.control=env.engine.policy_control;self.qi=self.control.q_indices
        self.frames=int(np.ceil(9/env.dt));self.previous=torch.zeros((16,33))
        self.hold_ok=torch.ones(16,dtype=torch.bool);self.return_ok=self.hold_ok.clone()
        goals=[];amounts=[];coms=[];data=mujoco.MjData(env.rig.model)
        for frame in range(self.frames+1):
            now=frame*env.dt;future=now+2*env.dt;row=[];com=[]
            for i in range(16):
                ref,clear=poses[i,:2]
                if now<1:q=ref
                elif future<2.5:q=blend_pose(ref,clear,future-1,1.5)
                elif future<5.5:q=clear
                else:q=blend_pose(clear,ref,future-5.5,1.5)
                row.append(q)
                data.qpos[:]=q;mujoco.mj_forward(env.rig.model,data);com.append(data.subtree_com[1].copy())
            goals.append(row);coms.append(com);amounts.append(float(blend(0.,1.,now-1.,1.)))
        self.goals=torch.tensor(np.array(goals),dtype=torch.float32)
        self.target_com=torch.tensor(np.array(coms),dtype=torch.float32)
        self.authority=torch.tensor(amounts,dtype=torch.float32)
        self.reference=torch.tensor(poses[:,0],dtype=torch.float32)
        self.ids=torch.arange(16)
        self.reset()

    def reset(self,ids=None):
        self.env.reset(ids)
        where=slice(None) if ids is None else ids
        self.previous[where]=0;self.hold_ok[where]=True;self.return_ok[where]=True
        if ids is None:self.initial_xy=self.env.measurements()['foot_xy_m'].clone()

    @torch.no_grad()
    def observation(self):
        from .support_teacher_dataset import policy_input
        frame=self.env.task.steps.long().clamp_max(self.frames)
        goals=self.goals[frame,self.ids];amount=self.authority[frame]
        x=torch.tensor(policy_input(self.env.task.observations().numpy(),goals.numpy(),amount.numpy()))
        quiet=(goals-self.reference).abs().amax(-1)<=1e-6
        return x,goals,amount,quiet,frame

    def critic_input(self,x,frame,actor):
        return torch.cat((((x-actor.mean)/actor.scale).clamp(-20,20),self.previous/.1,(frame.float()/self.frames)[:,None]),-1)

    @torch.no_grad()
    def step(self,action,active):
        from .target_control33 import DelayedTargetControl33
        from .run_single_support_feedback import flags
        x,goals,amount,quiet,frame=self.observation()
        c=self.control
        c.set_reference(*self.env.task.sample_reference(self.env.task.offset+self.env.task.steps*self.env.dt))
        source=DelayedTargetControl33.target_positions(c,self.anchor(x[:,:374]))
        raw=goals[:,self.qi]+action
        displacement=(goals[:,self.qi]-self.reference[:,self.qi]).abs().amax(-1)
        desired=(raw-source).clamp(-1,1)*amount[:,None]*(displacement/.05).clamp(0,1)[:,None]
        addition=self.previous+(desired-self.previous).clamp(-.02,.02)
        addition[quiet]=0;addition[~active]=0;self.previous.copy_(addition)
        pre=self.env.measurements();prestand,presupport=flags(self.env,pre)
        self.env.engine.active=active.numpy().copy()
        _,_,done,_=self.env.step(addition)
        post=self.env.last;stand,support=flags(self.env,post)
        stand=torch.tensor(stand&prestand);support=torch.tensor(support&presupport)
        q=self.env.engine.native_qpos();issued=c.pending[(c.cursor-1)%2,:,:33]
        assert torch.isfinite(q[active]).all() and not self.env.engine.solver_warning[active.numpy()].any()
        assert (issued-(source+addition).clamp(c.lower,c.upper)).abs().max()<2e-6
        assert (c.ctrl[:,self.env.rig.actuators].abs()<=torch.tensor(self.env.rig.action_scale)+1e-4).all()
        target=goals[:,self.qi]
        error=(q[:,self.qi]-target).square().mean(-1)
        com_error=(post['com_m'][:,:2]-self.target_com[frame,self.ids,:2]).square().sum(-1)
        foot_error=(post['foot_xy_m']-self.initial_xy).square().sum(-1)
        reward=.45*torch.exp(-error/.05**2)+.20*torch.exp(-com_error/.02**2)+.10*torch.exp(-foot_error.mean(-1)/.02**2)+.25*support.float()
        end_time=(frame.double()+1)*self.env.dt
        holding=(end_time>3.5)&(end_time<=5.5)
        self.hold_ok &= ~holding | (support&(error<=.05**2)&(com_error<=.02**2)&(foot_error.amax(-1)<=.02**2))
        self.return_ok &= (end_time<=7.) | stand
        failed=done==1;horizon=frame+1>=self.frames
        terminal=failed|horizon
        success=horizon&~failed&self.hold_ok&self.return_ok
        reward[failed]=-2.;reward+=10.*success.float()
        return reward,terminal,dict(failed=failed,success=success,issued=issued.clone(),qpos=q.clone(),
            supported=support,stand=stand,actor_active=~quiet&(amount>0),tracking_rmse=error.sqrt())


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(455,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)


@torch.no_grad()
def parity(adapter,actor):
    baseline=np.load(BASELINE);adapter.reset();active=torch.zeros(16,dtype=torch.bool);active[:8]=True
    maxq=maxt=0.
    for frame in range(len(baseline['time'])):
        x,*_=adapter.observation();_,terminal,info=adapter.step(actor(x),active)
        assert np.array_equal(active[:8].numpy(),baseline['alive'][frame])
        maxq=max(maxq,float(abs(info['qpos'][:8].numpy()-baseline['qpos'][frame]).max()))
        maxt=max(maxt,float(abs(info['issued'][:8].numpy()-baseline['issued'][frame]).max()))
        active &= ~terminal
    assert maxq==0 and maxt==0,(maxq,maxt)
    return dict(frames=len(baseline['time']),maximum_qpos_error=maxq,maximum_issued_error=maxt,exact=True)


def learn(adapter,initial,out,seed):
    torch.manual_seed(seed)
    actor=deepcopy(initial).train();critic=Critic()
    ao=torch.optim.Adam(actor.parameters(),lr=1e-5);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    out.mkdir();torch.save(actor.state_dict(),out/'initial.pt')
    adapter.reset();active=torch.zeros(16,dtype=torch.bool);active[TRAIN_IDS]=True
    history=[];start=time.perf_counter();completed=failures=actor_steps=0
    for update in range(32):
        rows=[]
        for _ in range(64):
            with torch.no_grad():
                x,_,_,_,frame=adapter.observation();ci=adapter.critic_input(x,frame,actor)
                mean=actor(x);distribution=Normal(mean,.005)
                action=distribution.sample();lp=distribution.log_prob(action).sum(-1);value=critic(ci)
                reward,terminal,info=adapter.step(action,active)
                nx,_,_,_,nf=adapter.observation();next_value=critic(adapter.critic_input(nx,nf,actor))
                rows.append(dict(x=x[TRAIN_IDS],ci=ci[TRAIN_IDS],action=action[TRAIN_IDS],lp=lp[TRAIN_IDS],value=value[TRAIN_IDS],
                    reward=reward[TRAIN_IDS],terminal=terminal[TRAIN_IDS],next_value=next_value[TRAIN_IDS],mask=info['actor_active'][TRAIN_IDS]))
                failures+=int((info['failed']&active).sum());completed+=int((info['success']&active).sum())
                resets=torch.nonzero(terminal&active).flatten()
                if len(resets):adapter.reset(resets)
        batch={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        advantage=torch.zeros_like(batch['reward']);running=torch.zeros(len(TRAIN_IDS))
        for t in reversed(range(64)):
            keep=(~batch['terminal'][t]).float()
            delta=batch['reward'][t]+.995*batch['next_value'][t]*keep-batch['value'][t]
            running=delta+.995*.98*keep*running;advantage[t]=running
        returns=(advantage+batch['value']).flatten()
        flat={k:v.flatten(0,1) for k,v in batch.items()}
        av=advantage.flatten();mask=flat['mask'];av=(av-av[mask].mean())/av[mask].std().clamp_min(1e-6) if mask.any() else av*0
        actor_stopped=False;max_kl=0.;loss_value=0.;cycle_actor_steps=0
        for _ in range(3):
            order=torch.randperm(len(av))
            for ids in order.split(256):
                selected=ids[mask[ids]]
                if len(selected) and not actor_stopped:
                    logp=Normal(actor(flat['x'][selected]),.005).log_prob(flat['action'][selected]).sum(-1)
                    log_ratio=logp-flat['lp'][selected];ratio=log_ratio.exp()
                    kl=float(((ratio-1)-log_ratio).mean().detach());max_kl=max(max_kl,kl)
                    if not np.isfinite(kl) or kl>.01:
                        actor_stopped=True
                    else:
                        loss=-torch.minimum(ratio*av[selected],ratio.clamp(.8,1.2)*av[selected]).mean()
                        ao.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);ao.step()
                        actor_steps+=1;cycle_actor_steps+=1
                value_loss=(critic(flat['ci'][ids])-returns[ids]).square().mean()
                co.zero_grad(set_to_none=True);value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
                loss_value=float(value_loss.detach())
        assert torch.isfinite(av).all() and torch.isfinite(returns).all()
        history.append(dict(update=update+1,transitions=(update+1)*768,mean_reward=float(batch['reward'].mean()),
            actor_eligible=int(mask.sum()),actor_steps=cycle_actor_steps,max_checked_kl=max_kl,actor_early_stop=actor_stopped,
            critic_loss=loss_value,completed_episodes=completed,failed_episodes=failures))
        if (update+1)%4==0:
            print('PPO',seed,history[-1],flush=True);save(out/'history.json',history)
    actor.eval();torch.save(actor.state_dict(),out/'final.pt');torch.save(critic.state_dict(),out/'critic-final.pt')
    save(out/'architecture.json',dict(actor='QueueBlindLearner',masked_columns=[95,229],distribution='Gaussian std 0.005; mean only for evaluation'))
    result=dict(seed=seed,collection_cycles=32,transitions=24576,actor_gradient_steps=actor_steps,wall_seconds=time.perf_counter()-start,
        completed_training_episodes=completed,failed_training_episodes=failures,training_pose_ids=TRAIN_IDS.tolist())
    save(out/'report.json',result);return actor,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('ARM_HOLD_PPO_PLAN.md')]
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),INITIAL,BASELINE,PRIOR/'goals.npz',FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,seeds=[501028,511028],training_ids=TRAIN_IDS.tolist(),heldout_poses=[6,7],
        cycles_per_seed=32,frames_per_cycle=64,physical_training_worlds=12,distribution_std_rad=.005,
        actor_lr=1e-5,critic_lr=3e-4,epochs=3,minibatch=256,gamma=.995,gae_lambda=.98,clip=.2,target_kl=.01,arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.);poses=np.load(PRIOR/'goals.npz')['poses']
    state=torch.load(INITIAL,weights_only=True);initial=QueueBlindLearner(state['mean'],state['scale'],state['output_scale']);initial.load_state_dict(state);initial.eval()
    adapter=ArmRollout(env,anchor,poses);results=dict(parity=parity(adapter,initial))
    save(OUT/'parity.json',results['parity']);print('PARITY',results['parity'],flush=True)
    for seed in (501028,511028):
        net,training=learn(adapter,initial,OUT/str(seed),seed)
        result=episode(env,anchor,net,poses,OUT/f'{seed}-nominal')
        results[str(seed)]=dict(training=training,evaluation=result)
        bank={i:dict(poses=poses[i]) for i in range(16)}
        results[f'quiet-{seed}']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/f'quiet-{seed}')
        assert results[f'quiet-{seed}']['whole_interval_stand']==16
        if result['complete']>0 and result['stable_holds']>=4 and result['returned']>=4:
            results[f'varied-{seed}']=episode(env,anchor,net,poses,OUT/f'{seed}-varied',variant=True)
        save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
