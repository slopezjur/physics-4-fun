"""Bounded PPO on feasible corrections to a frozen physical proposal policy."""
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
from .run_learned_support import PARENT
from .run_learned_handoff import PRIOR
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold_ppo import INITIAL, BASELINE, TRAIN_IDS
from .train_arm_hold_ppo_rate import FixedHoldGate, reward_diagnostics
from .probe_arm_feasible_response import FeasibleCorrectionRollout
from .target_control33 import DelayedTargetControl33

OUT = Path('logs/mimickit-training/arm-correction-ppo-20261008-01')


class CorrectionNetwork(nn.Module):
    def __init__(self, outputs=33):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(488, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.outputs = outputs

    def forward(self, x):
        y = self.net(x)
        return y.squeeze(-1) if self.outputs == 1 else y


class TimedGate(FixedHoldGate):
    def step(self, frame, qpos, com, xy, supported, stand, failed):
        end_time = frame * self.dt + self.dt
        holding = (end_time > self.hold_end - 2) & (end_time <= self.hold_end)
        error = np.sqrt(np.mean((qpos[:, self.qi] - self.goal) ** 2, axis=-1))
        tracked = supported & (error <= .05)
        tracked &= np.linalg.norm(com[:, :2] - self.com, axis=-1) <= .02
        tracked &= (np.linalg.norm(xy - self.xy, axis=-1) <= .02).all(-1)
        self.hold &= ~holding | tracked
        self.returned &= (frame < self.frames - self.return_frames) | stand
        return (frame + 1 >= self.frames) & ~failed & self.hold & self.returned


class CorrectionEnvironment:
    def __init__(self, env, anchor, proposal, poses, *, quiet=False, variant=False):
        from .predictive_target_support import blend, blend_pose
        self.env, self.proposal = env, proposal
        self.physical = FeasibleCorrectionRollout(env, anchor, poses)
        self.quiet = quiet
        self.onset, self.ramp, self.hold = (1.2, 1.7, 3.5) if variant else (1., 1.5, 3.)
        self.hold_end = self.onset + self.ramp + self.hold
        self.seconds = 30. if quiet else self.hold_end + 3.5
        p = self.physical
        p.frames = int(np.ceil(self.seconds / env.dt))
        goals, amounts, coms = [], [], []
        data = mujoco.MjData(env.rig.model)
        for frame in range(p.frames + 1):
            now = frame * env.dt
            future = now + 2 * env.dt
            row, com = [], []
            for pose in poses:
                ref, clear = pose[:2]
                if quiet or now < self.onset:
                    goal = ref
                elif future < self.onset + self.ramp:
                    goal = blend_pose(ref, clear, future - self.onset, self.ramp)
                elif future < self.hold_end:
                    goal = clear
                else:
                    goal = blend_pose(clear, ref, future - self.hold_end, 1.5)
                row.append(goal)
                data.qpos[:] = goal
                mujoco.mj_forward(env.rig.model, data)
                com.append(data.subtree_com[1].copy())
            goals.append(row)
            coms.append(com)
            amounts.append(float(blend(0., 1., now - self.onset, 1.)))
        p.goals = torch.tensor(np.array(goals), dtype=torch.float32)
        p.target_com = torch.tensor(np.array(coms), dtype=torch.float32)
        p.authority = torch.tensor(amounts, dtype=torch.float32)
        p.gate = TimedGate(env.rig.model, poses, p.qi.numpy(), env.dt, p.initial_xy.numpy())
        p.gate.frames, p.gate.hold_end = p.frames, self.hold_end

    def reset(self, ids=None):
        self.physical.reset(ids)

    @torch.no_grad()
    def observation(self):
        p = self.physical
        x, goals, amount, quiet, frame = p.observation()
        c = p.control
        c.set_reference(*self.env.task.sample_reference(self.env.task.offset + self.env.task.steps * self.env.dt))
        source = DelayedTargetControl33.target_positions(c, p.anchor(x[:, :374]))
        proposal = self.proposal(x)
        fade = ((goals[:, c.q_indices] - p.reference[:, c.q_indices]).abs().amax(-1) / .05).clamp(0, 1)
        desired = (goals[:, c.q_indices] + proposal - source).clamp(-1, 1) * amount[:, None] * fade[:, None]
        nominal = p.previous + (desired - p.previous).clamp(-.02, .02)
        features = torch.cat((((x - self.proposal.mean) / self.proposal.scale).clamp(-20, 20),
            p.previous / .1, nominal / .1, (frame.float() / 540)[:, None]), -1)
        return features, goals, amount, quiet, frame

    def critic_input(self, x, frame, actor):
        return x

    @torch.no_grad()
    def step(self, action, active):
        p = self.physical
        x, goals, amount, quiet, frame = p.observation()
        before = p.previous.clone()
        p.correction.copy_(action)
        _, terminal, info = p.step(self.proposal(x), active)
        commanded = active & ~quiet
        assert (p.previous[commanded] - before[commanded]).abs().max() <= .0200001 if commanded.any() else True
        assert p.previous[active].abs().max() <= 1.0000001 if active.any() else True
        error = (info['qpos'][:, p.qi] - goals[:, p.qi]).square().mean(-1)
        com_error = (self.env.last['com_m'][:, :2] - p.target_com[frame, p.ids, :2]).square().sum(-1)
        parts = info['components'].clone()
        parts[:, 0] = .45 * torch.exp(-error / .10**2)
        parts[:, 1] = .20 * torch.exp(-com_error / .04**2)
        parts[:, 4] = torch.where(info['failed'], -2 - parts[:, :4].sum(-1), 0)
        info['components'] = parts
        now = frame.double() * self.env.dt
        info['phase'] = torch.bucketize(now, torch.tensor([1., 2.5, 5.5, 7.], dtype=torch.float64), right=True)
        reward = info['broad_reward']
        assert torch.allclose(parts.sum(-1)[active], reward[active], atol=1e-6, rtol=0)
        info['transmitted'] = (p.previous - info['nominal_addition']).abs() > 1e-7
        info['transmitted'][~commanded] = False
        info['addition'] = p.previous.clone()
        info['previous_addition'] = before
        return reward, terminal, info


@torch.no_grad()
def evaluate(adapter, actor, out, *, parity=False):
    p, env = adapter.physical, adapter.env
    adapter.reset()
    n = 16 if adapter.quiet else 8
    active = torch.zeros(16, dtype=torch.bool)
    active[:n] = True
    xy = env.measurements()['foot_xy_m'].numpy()[:n].copy()
    rows = []
    for frame in range(p.frames):
        x, *_ = adapter.observation()
        action = actor(x)
        before = active.clone()
        _, terminal, info = adapter.step(action, active)
        post = env.last
        rows.append(dict(time=frame * env.dt + env.dt, alive=before[:n].numpy(), qpos=info['qpos'][:n].numpy(),
            stand=info['stand'][:n].numpy(), support=info['supported'][:n].numpy(), com=post['com_m'][:n].numpy(),
            xy=post['foot_xy_m'][:n].numpy(), issued=info['issued'][:n].numpy(), addition=info['addition'][:n].numpy(),
            previous_addition=info['previous_addition'][:n].numpy(), nominal=info['nominal_addition'][:n].numpy(),
            correction=action[:n].numpy(), done=info['failed'][:n].numpy().astype(int),
            tracking_rmse=np.sqrt(np.mean((info['qpos'][:n, p.qi].numpy() - p.poses[:n, 1, p.qi.numpy()])**2, axis=-1))))
        active &= ~terminal
    z = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    if parity:
        baseline = np.load(BASELINE)
        assert all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
        assert (z['correction'] == 0).all()
    cases = []
    hold = (z['time'] > adapter.hold_end - 2) & (z['time'] <= adapter.hold_end)
    for i in range(n):
        survived = z['alive'][-1, i] and not z['done'][-1, i]
        returned = bool(survived and z['stand'][-int(np.ceil(2 / env.dt)):, i].all())
        stable = bool(z['alive'][hold, i].all() and z['support'][hold, i].all())
        tracked = bool(stable and (z['tracking_rmse'][hold, i] <= .05).all()
            and (np.linalg.norm(z['com'][hold, i, :2] - p.gate.com[i], axis=-1) <= .02).all()
            and (np.linalg.norm(z['xy'][hold, i] - xy[i], axis=-1) <= .02).all())
        cases.append(dict(pose=i % 8, heldout=i % 8 >= 6, stable_hold=stable, tracked_hold=tracked,
            returned_stand=returned, complete=tracked and returned,
            whole_interval_stand=bool(z['alive'][:, i].all() and z['stand'][:, i].all() and survived)))
    out.mkdir()
    np.savez_compressed(out / 'trace.npz', **z)
    result = dict(quiet=adapter.quiet, hold_end=adapter.hold_end, seconds=adapter.seconds, cases=cases,
        stable_holds=sum(c['stable_hold'] for c in cases), returned=sum(c['returned_stand'] for c in cases),
        complete=sum(c['complete'] for c in cases), whole_interval_stand=sum(c['whole_interval_stand'] for c in cases))
    save(out / 'report.json', result)
    print('EVALUATED', out.name, {k: v for k, v in result.items() if k != 'cases'}, flush=True)
    return result


def learn(adapter,initial,out,seed,actor_lr):
    torch.manual_seed(seed)
    actor=deepcopy(initial).train();critic=CorrectionNetwork(1)
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
                    transmitted=info['transmitted'][TRAIN_IDS],old_mean=mean[TRAIN_IDS],components=info['components'][TRAIN_IDS],phase=info['phase'][TRAIN_IDS],reward=reward[TRAIN_IDS],terminal=terminal[TRAIN_IDS],next_value=next_value[TRAIN_IDS],mask=info['actor_active'][TRAIN_IDS]))
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
            transmitted_coordinates=int(batch['transmitted'].sum()),eligible_coordinates=int(mask.sum())*33,post_update_full_batch_kl=post_update_kl,reward_parts=reward_diagnostics(batch),critic_loss=loss_value,completed_episodes=completed,failed_episodes=failures))
        if (update+1)%4==0:
            print('PPO',out.name,{k:v for k,v in history[-1].items() if k not in ('reward_parts','post_update_full_batch_kl')},'full_batch_kl',post_update_kl,flush=True);save(out/'history.json',history)
    actor.eval();torch.save(actor.state_dict(),out/'final.pt');torch.save(critic.state_dict(),out/'critic-final.pt')
    save(out/'architecture.json',dict(actor='CorrectionNetwork',inputs=488,outputs=33,proposal_frozen=True,distribution='Gaussian std 0.005; mean only for evaluation'))
    result=dict(seed=seed,actor_lr=actor_lr,collection_cycles=32,transitions=24576,actor_gradient_steps=actor_steps,wall_seconds=time.perf_counter()-start,
        completed_training_episodes=completed,failed_training_episodes=failures,training_pose_ids=TRAIN_IDS.tolist())
    save(out/'report.json',result);return actor,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    torch.manual_seed(521027)
    initial = CorrectionNetwork().eval()
    OUT.mkdir(exist_ok=False)
    (OUT / 'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('ARM_CORRECTION_PPO_PLAN.md')]
    inputs = {str(p): digest(p) for p in paths + [PARENT, PARENT.with_suffix('.amp.pt'), INITIAL, BASELINE,
        PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz']}
    for p in paths:
        shutil.copy2(p, OUT / 'sources' / p.name)
    torch.save(initial.state_dict(), OUT / 'initial-correction.pt')
    save(OUT / 'protocol.json', dict(inputs=inputs, seeds=[521028, 531028], initial_seed=521027,
        training_ids=TRAIN_IDS.tolist(), transitions_per_seed=24576, actor_lr=3e-6,
        correction_units='Unprojected Gaussian radians, std 0.005', width_multiplier=2,
        architecture='488-128ELU-128ELU-33, final layer zero', arena_promotion=False))
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz', json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    state = torch.load(INITIAL, weights_only=True)
    proposal = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    proposal.load_state_dict(state)
    proposal.eval().requires_grad_(False)
    env = make_environment(rig, anchor, 40.)
    poses = np.load(PRIOR / 'goals.npz')['poses']
    adapter = CorrectionEnvironment(env, anchor, proposal, poses)
    quiet_adapter = CorrectionEnvironment(env, anchor, proposal, poses, quiet=True)
    results = dict(zero_nominal=evaluate(adapter, initial, OUT / 'zero-nominal', parity=True),
        zero_quiet=evaluate(quiet_adapter, initial, OUT / 'zero-quiet'))
    assert results['zero_quiet']['whole_interval_stand'] == 16
    baseline_quiet = np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    zero_quiet = np.load(OUT / 'zero-quiet/trace.npz')
    assert all(np.array_equal(zero_quiet[k], baseline_quiet[k]) for k in ('qpos', 'issued'))
    save(OUT / 'checks.json', dict(nominal_bitwise_parity=True, quiet_bitwise_parity=True, quiet_passed=16))
    for seed in (521028, 531028):
        net, training = learn(adapter, initial, OUT / str(seed), seed, 3e-6)
        nominal = evaluate(adapter, net, OUT / f'{seed}-nominal')
        quiet = evaluate(quiet_adapter, net, OUT / f'quiet-{seed}')
        assert quiet['whole_interval_stand'] == 16
        results[str(seed)] = dict(training=training, nominal=nominal, quiet=quiet)
        if nominal['complete'] > 0 and nominal['stable_holds'] >= 4 and nominal['returned'] >= 4:
            varied = CorrectionEnvironment(env, anchor, proposal, poses, variant=True)
            results[str(seed)]['varied'] = evaluate(varied, net, OUT / f'{seed}-varied')
        save(OUT / 'progress.json', results)
    assert all(torch.equal(v, proposal.state_dict()[k]) for k, v in state.items())
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT / 'report.json', dict(results=results, inputs_unchanged=True, proposal_unchanged=True, arena_changed=False,
        output_hashes={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__ == '__main__':
    main()

