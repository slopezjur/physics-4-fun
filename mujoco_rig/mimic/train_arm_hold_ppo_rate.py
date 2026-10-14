"""Matched learning-rate comparison with observation-only PPO diagnostics."""
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
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT, evaluate
from .run_learned_handoff import PRIOR, GuardedLearner
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold import episode
from .train_arm_hold_ppo import ArmRollout, Critic, INITIAL, BASELINE, TRAIN_IDS, parity

OUT = Path('logs/mimickit-training/arm-hold-ppo-rate-20261008-01')
COMPONENTS = ('tracking', 'com', 'feet', 'support', 'failure_adjustment', 'success')
PHASES = ('quiet', 'ramp', 'hold', 'return', 'settle')


class FixedHoldGate:
    def __init__(self, model, poses, qi, dt, initial_xy):
        self.goal = poses[:, 1, qi]
        self.xy = initial_xy.copy()
        data = mujoco.MjData(model)
        com = []
        for pose in poses:
            data.qpos[:] = pose[1]
            mujoco.mj_forward(model, data)
            com.append(data.subtree_com[1, :2].copy())
        self.com = np.array(com)
        self.dt = dt
        self.frames = int(np.ceil(9 / dt))
        self.return_frames = int(np.ceil(2 / dt))
        self.qi = qi
        self.hold = np.ones(len(poses), dtype=bool)
        self.returned = self.hold.copy()

    def reset(self, ids=None):
        where = slice(None) if ids is None else np.asarray(ids)
        self.hold[where] = True
        self.returned[where] = True

    def step(self, frame, qpos, com, xy, supported, stand, failed):
        # Match the frozen evaluator's time arithmetic and full-window targets.
        end_time = frame * self.dt + self.dt
        holding = (end_time > 3.5) & (end_time <= 5.5)
        error = np.sqrt(np.mean((qpos[:, self.qi] - self.goal) ** 2, axis=-1))
        tracked = supported & (error <= .05)
        tracked &= np.linalg.norm(com[:, :2] - self.com, axis=-1) <= .02
        tracked &= (np.linalg.norm(xy - self.xy, axis=-1) <= .02).all(-1)
        self.hold &= ~holding | tracked
        self.returned &= (frame < self.frames - self.return_frames) | stand
        return (frame + 1 >= self.frames) & ~failed & self.hold & self.returned


class DiagnosticRollout(ArmRollout):
    def __init__(self, env, anchor, poses):
        super().__init__(env, anchor, poses)
        self.gate = FixedHoldGate(env.rig.model, poses, self.qi.numpy(), env.dt, self.initial_xy.numpy())

    def reset(self, ids=None):
        super().reset(ids)
        if hasattr(self, 'gate'):
            self.gate.reset(None if ids is None else ids.numpy())

    @torch.no_grad()
    def step(self, action, active):
        _, goals, _, _, frame = self.observation()
        reward, terminal, info = super().step(action, active)
        post = self.env.last
        success = torch.tensor(self.gate.step(frame.numpy(), info['qpos'].numpy(), post['com_m'].numpy(),
            post['foot_xy_m'].numpy(), info['supported'].numpy(), info['stand'].numpy(), info['failed'].numpy()))
        reward = reward - 10 * info['success'].float() + 10 * success.float()
        info['success'] = success
        error = (info['qpos'][:, self.qi] - goals[:, self.qi]).square().mean(-1)
        com_error = (post['com_m'][:, :2] - self.target_com[frame, self.ids, :2]).square().sum(-1)
        foot_error = (post['foot_xy_m'] - self.initial_xy).square().sum(-1)
        parts = torch.stack((.45 * torch.exp(-error / .05**2), .20 * torch.exp(-com_error / .02**2),
            .10 * torch.exp(-foot_error.mean(-1) / .02**2), .25 * info['supported'].float()), -1)
        failure = torch.where(info['failed'], -2 - parts.sum(-1), 0)
        info['components'] = torch.cat((parts, failure[:, None], 10 * success.float()[:, None]), -1)
        assert torch.allclose(info['components'].sum(-1)[active], reward[active], atol=1e-6, rtol=0)
        now = frame.double() * self.env.dt
        info['phase'] = torch.bucketize(now, torch.tensor([1., 2.5, 5.5, 7.], dtype=torch.float64), right=True)
        return reward, terminal, info


def gate_replay(adapter):
    results = []
    for folder in (INITIAL.parent.parent / 'nominal',
                   Path('logs/mimickit-training/arm-hold-learning-20261008-01/teacher-nominal')):
        z = np.load(folder / 'trace.npz')
        report = json.loads((folder / 'report.json').read_text())
        gate = FixedHoldGate(adapter.env.rig.model, adapter.poses[:8], adapter.qi.numpy(),
            adapter.env.dt, adapter.initial_xy.numpy()[:8])
        completed = np.zeros(8, dtype=bool)
        for frame in range(len(z['time'])):
            success = gate.step(np.full(8, frame), z['qpos'][frame], z['com'][frame], z['xy'][frame],
                z['support'][frame], z['stand'][frame], z['done'][frame] == 1)
            completed |= success & z['alive'][frame]
        assert completed.tolist() == [c['complete'] for c in report['cases']]
        results.append(dict(source=str(folder), complete=int(completed.sum()), exact=True))
    return results


def reward_diagnostics(batch):
    result = {}
    for index, name in enumerate(PHASES):
        mask = batch['phase'] == index
        values = batch['components'][mask]
        if not len(values):
            continue
        result[name] = dict(count=len(values), sums=dict(zip(COMPONENTS, values.double().sum(0).tolist())),
            tracking_below_one_percent=int((values[:, 0] < .0045).sum()),
            com_below_one_percent=int((values[:, 1] < .002).sum()))
    return result


def learn(adapter,initial,out,seed,actor_lr):
    torch.manual_seed(seed)
    actor=deepcopy(initial).train();critic=Critic()
    ao=torch.optim.Adam(actor.parameters(),lr=actor_lr);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
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
                    old_mean=mean[TRAIN_IDS],components=info['components'][TRAIN_IDS],phase=info['phase'][TRAIN_IDS],reward=reward[TRAIN_IDS],terminal=terminal[TRAIN_IDS],next_value=next_value[TRAIN_IDS],mask=info['actor_active'][TRAIN_IDS]))
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
        actor_stopped=False;max_kl=0.;loss_value=0.;cycle_actor_steps=0;post_update_kl=[]
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
                        with torch.no_grad():
                            mean_delta=actor(flat['x'][mask])-flat['old_mean'][mask]
                            full_kl=float((mean_delta.square().sum(-1)/(2*.005**2)).mean())
                            assert np.isfinite(full_kl)
                            post_update_kl.append(full_kl)
                value_loss=(critic(flat['ci'][ids])-returns[ids]).square().mean()
                co.zero_grad(set_to_none=True);value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
                loss_value=float(value_loss.detach())
        assert torch.isfinite(av).all() and torch.isfinite(returns).all()
        history.append(dict(update=update+1,transitions=(update+1)*768,mean_reward=float(batch['reward'].mean()),
            actor_eligible=int(mask.sum()),actor_steps=cycle_actor_steps,max_checked_kl=max_kl,actor_early_stop=actor_stopped,
            post_update_full_batch_kl=post_update_kl,reward_parts=reward_diagnostics(batch),critic_loss=loss_value,completed_episodes=completed,failed_episodes=failures))
        if (update+1)%4==0:
            print('PPO',out.name,{k:v for k,v in history[-1].items() if k not in ('reward_parts','post_update_full_batch_kl')},'full_batch_kl',post_update_kl,flush=True);save(out/'history.json',history)
    actor.eval();torch.save(actor.state_dict(),out/'final.pt');torch.save(critic.state_dict(),out/'critic-final.pt')
    save(out/'architecture.json',dict(actor='QueueBlindLearner',masked_columns=[95,229],distribution='Gaussian std 0.005; mean only for evaluation'))
    result=dict(seed=seed,actor_lr=actor_lr,collection_cycles=32,transitions=24576,actor_gradient_steps=actor_steps,wall_seconds=time.perf_counter()-start,
        completed_training_episodes=completed,failed_training_episodes=failures,training_pose_ids=TRAIN_IDS.tolist())
    save(out/'report.json',result);return actor,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    (OUT / 'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('ARM_HOLD_PPO_RATE_PLAN.md')]
    inputs = {str(p): digest(p) for p in paths + [PARENT, PARENT.with_suffix('.amp.pt'), INITIAL, BASELINE,
        PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz']}
    for p in paths:
        shutil.copy2(p, OUT / 'sources' / p.name)
    save(OUT / 'protocol.json', dict(inputs=inputs, seeds=[501028, 511028], actor_rates=[1e-5, 3e-6],
        training_ids=TRAIN_IDS.tolist(), transitions_per_run=24576, full_batch_kl='Analytic Gaussian KL; observation only',
        success_gate='Fixed double-precision hold goal and evaluator return-frame count', arena_promotion=False))
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz', json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    env = make_environment(rig, anchor, 40.)
    poses = np.load(PRIOR / 'goals.npz')['poses']
    state = torch.load(INITIAL, weights_only=True)
    initial = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    initial.load_state_dict(state)
    initial.eval()
    adapter = DiagnosticRollout(env, anchor, poses)
    results = dict(parity=parity(adapter, initial), gate_replay=gate_replay(adapter))
    save(OUT / 'checks.json', results)
    print('CHECKS', results, flush=True)
    for seed in (501028, 511028):
        for label, rate in (('control', 1e-5), ('lower', 3e-6)):
            name = f'{label}-{seed}'
            net, training = learn(adapter, initial, OUT / name, seed, rate)
            nominal = episode(env, anchor, net, poses, OUT / f'{name}-nominal')
            bank = {i: dict(poses=poses[i]) for i in range(16)}
            quiet = evaluate(env, anchor, bank, GuardedLearner(net, anchor, env), 'goal', 1., True, OUT / f'quiet-{name}')
            assert quiet['whole_interval_stand'] == 16
            results[name] = dict(training=training, nominal=nominal, quiet=quiet)
            if nominal['complete'] > 0 and nominal['stable_holds'] >= 4 and nominal['returned'] >= 4:
                results[name]['varied'] = episode(env, anchor, net, poses, OUT / f'{name}-varied', variant=True)
            save(OUT / 'progress.json', results)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT / 'report.json', dict(results=results, inputs_unchanged=True, arena_changed=False,
        output_hashes={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__ == '__main__':
    main()

