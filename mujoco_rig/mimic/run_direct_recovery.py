"""Bounded direct push recovery over an immutable deployed controller."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from envs.base_env import DoneFlags, EnvMode
from .capture_reward import capture_distance, rotate_xyzw
from .corrective_step_audit import foot_geometry, measure_steps
from .representation_task import RepresentationTask
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .step_reward import TransferRecoveryReward
from .support_skill_core import PhysicalSkillTask
from .target_control33 import DelayedTargetControl33

OUT = Path('logs/mimickit-training/direct-recovery-20261009-01')
TRAIN = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
DIRECTIONS = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
RADIUS, SLEW, STD = .1, .02, .1


class Head(nn.Module):
    def __init__(self, outputs=33):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(407, 128), nn.ELU(), nn.Linear(128, 128),
                                 nn.ELU(), nn.Linear(128, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.outputs = outputs

    def forward(self, x):
        value = self.net(x)
        return value.squeeze(-1) if self.outputs == 1 else value


class Recovery:
    def __init__(self, rig, anchor, *, bare=False, seconds=6.):
        self.rig, self.anchor = rig, anchor
        reference = FOUNDATION / 'stationary_reference.npz'
        self.task = (RepresentationTask(rig, reference, 16, 'cpu', representation='target_pd',
            reward_mode='reference_v9', episode_seconds=seconds) if bare else
            PhysicalSkillTask(rig, reference, 16, 'cpu', seconds))
        self.task.set_mode(EnvMode.TEST)
        self.engine, self.dt = self.task.engine, self.task.dt
        self.previous = torch.zeros(16, 33)
        self.force = torch.zeros(16, 3)
        self.onset = torch.zeros(16, dtype=torch.long)
        self.chest = rig.model.body('Chest').id
        self.bare = bare
        self.reward_fn = TransferRecoveryReward(imitation_weight=0.)
        gids = [rig.model.geom(f'g_{body}_{side}').id for side in ('L', 'R') for body in ('Foot', 'Toe')]
        self.bodies = [rig.model.body(f'{body}_{side}').id - 1 for side in ('L', 'R') for body in ('Foot', 'Toe')]
        self.centers = torch.tensor(rig.model.geom_pos[gids], dtype=torch.float32)
        self.sizes = torch.tensor(rig.model.geom_size[gids], dtype=torch.float32)
        self.reset()

    def reset(self, ids=None, rng=None):
        ids = self.task.ids if ids is None else ids
        self.task.reset(ids)
        self.previous[ids] = 0
        if rng is not None:
            n = len(ids)
            directions = torch.randint(4, (n,), generator=rng)
            magnitude = 150. * (1 + torch.randint(2, (n,), generator=rng))
            magnitude[torch.rand(n, generator=rng) < .25] = 0
            self.force[ids] = DIRECTIONS[directions] * magnitude[:, None]
            onset = .8 + .4 * torch.randint(2, (n,), generator=rng)
            self.onset[ids] = torch.ceil(onset / self.dt).long()

    @torch.no_grad()
    def features(self):
        raw = self.task.observations()
        return torch.cat((self.anchor.obs_norm.normalize(raw).clamp(-20, 20), self.previous / RADIUS), -1)

    def state(self):
        s = self.task.standing_measurements()
        positions, rotations = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
        rot = rotations[:, self.bodies]
        centers = positions[:, self.bodies] + rotate_xyzw(rot, self.centers)
        x, y, z, w = rot.unbind(-1)
        row = torch.stack((2 * (x*z-w*y), 2 * (y*z+w*x), 1-2 * (x*x+y*y)), -1)
        s['clearance'] = (centers[..., 2] - (row.abs()*self.sizes).sum(-1)).reshape(16, 2, 2).amin(-1)
        s['feet'] = positions[:, self.task.feet]
        s['com'] = self.task.center_of_mass(positions, rotations)
        s['velocity'] = self.engine.get_center_of_mass_velocity()
        s['angular'] = self.engine.get_root_ang_vel(0)
        flags = {**self.task.balance_criteria(s), **self.task.posture_criteria(s)}
        s['stand'] = torch.stack(list(flags.values())).all(0)
        s['settled'] = s['stand'] & (s['foot_loads_n'].amin(-1) >= 30.)
        return s

    @torch.no_grad()
    def step(self, latent, active):
        self.engine.active[:] = active.numpy()
        pre = self.state()
        source = self.anchor(self.task.observations())
        before = self.previous.clone()
        desired = RADIUS * latent.tanh()
        addition = before + (desired - before).clamp(-SLEW, SLEW)
        addition[~active] = before[~active]
        if self.bare:
            assert not addition.any()
        else:
            self.engine.policy_control.addition.copy_(addition)
        pushing = active & (self.task.steps >= self.onset) & (self.task.steps < self.onset + 6)
        applied = self.force * pushing[:, None]
        self.engine.set_external_force(self.chest, applied)
        _, _, done, _ = self.task.step(source)
        self.previous.copy_(addition)
        s = self.state()
        c = self.engine.policy_control
        nominal = DelayedTargetControl33.target_positions(c, source)
        issued = c.target_positions(source)
        expected = (nominal + addition).clamp(c.lower, c.upper)
        assert torch.equal(issued[active], expected[active])
        assert (addition.abs() <= RADIUS + 1e-7).all()
        assert ((addition - before).abs() <= SLEW + 1e-7).all()
        failed = (done == DoneFlags.FAIL.value) | torch.from_numpy(self.engine.solver_warning.copy())
        terminal = (done != DoneFlags.NULL.value) | failed
        capture = capture_distance(s['com'], s['velocity'], s['feet'], s['foot_loads_n'])
        cp = s['com'][:, :2] + s['velocity'][:, :2] * (s['com'][:, 2].clamp_min(.05) / 9.81).sqrt()[:, None]
        loads = s['foot_loads_n']
        transfer = (1 - 2*loads / loads.sum(-1, keepdim=True).clamp_min(1)).clamp(0, 1)
        lift = (s['clearance'] / .02).clamp(0, 1) * torch.exp(-((s['clearance']-.04).clamp_min(0)/.06).square())
        swing = (torch.exp(-((s['feet'][:, :, :2] - cp[:, None])/.15).square().sum(-1))
            * (.3*transfer + .7*lift*(1-loads/5).clamp(0, 1)) * (loads.flip(-1)/100).clamp(0, 1)).amax(-1)
        reward = self.reward_fn(torch.zeros(16), s['pelvis_height_m'], self.task.reference.qpos[self.task.ids % 8, 2],
            s['pelvis_up'], s['velocity'], s['angular'], capture, swing, loads)
        reward -= .005*(addition/RADIUS).square().mean(-1)
        reward[failed] = -2.
        assert torch.isfinite(reward).all()
        info = dict(qpos=self.engine.native_qpos().clone(), qvel=self.engine._array('qvel'),
            issued=issued.clone(), active_target=c.active.clone(), addition=addition, previous=before,
            nominal=nominal, force=applied, failed=failed, stand=s['stand'] & pre['stand'],
            settled=s['settled'] & pre['settled'], loads=loads, clearance=s['clearance'],
            xy=s['feet'][:, :, :2], com=s['com'], capture=capture, swing=swing,
            warning=torch.from_numpy(self.engine.solver_warning.copy()))
        return reward, terminal, info


def configure(env, group, quiet=False):
    env.reset()
    env.force.zero_()
    if not quiet:
        strength, axis = (150., 300.)[group // 2], group % 2
        env.force[:8] = strength * DIRECTIONS[2*axis]
        env.force[8:] = strength * DIRECTIONS[2*axis+1]
        env.onset[:] = int(np.ceil((1. if axis == 0 else 1.7) / env.dt))


@torch.no_grad()
def rollout(env, actor, folder, *, group=0, seconds=6., quiet=False, sampled=False):
    configure(env, group, quiet)
    active = torch.ones(16, dtype=torch.bool)
    rng = torch.Generator().manual_seed(700009)
    rows = []
    for frame in range(int(np.ceil(seconds/env.dt))):
        latent = torch.zeros(16, 33) if actor is None else actor(env.features())
        if sampled:
            latent = latent + STD * torch.randn(latent.shape, generator=rng)
        before = active.clone()
        _, terminal, info = env.step(latent, active)
        rows.append(dict(time=np.array((frame+1)*env.dt), alive=before.numpy(),
                         **{k: v.numpy().copy() for k, v in info.items()}))
        active &= ~terminal
    z = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    assert not z['warning'].any(), 'Native solver warning invalidates this experiment'
    xy, clearance = foot_geometry(env.rig.model, z['qpos'])
    assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
    assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
    cases = []
    tail = int(np.ceil(2/env.dt))
    for i in range(16):
        hit = None if quiet else float(env.onset[i]*env.dt)
        valid_alive = z['alive'][:, i] & ~z['failed'][:, i]
        survived = bool(valid_alive.all())
        settled = bool(survived and z['settled'][-tail:, i].all())
        events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], valid_alive,
                               z['settled'][:, i], hit_time=hit)
        post = valid_alive & (z['time'] >= (hit or 0))
        unilateral = (z['loads'][:, i].min(-1) < 5) & (z['loads'][:, i].max(-1) >= 30)
        lifted = ((clearance[:, i] >= .015) & (z['loads'][:, i] < 5)
                  & (z['loads'][:, i, ::-1] >= 30)).any(-1)
        cases.append(dict(case=i, pose=i % 8, heldout=i % 8 >= 6, survived=survived,
            settled=settled, whole_stand=bool(survived and z['stand'][:, i].all()),
            hit_seconds=hit, force_n=env.force[i].tolist(), impulse_ns=(z['force'][:, i].sum(0)*env.dt).tolist(),
            unilateral_frames=int((post & unilateral).sum()), lifted_frames=int((post & lifted).sum()),
            steps=events, recovered_with_step=settled and any(e['settled_window_start'] is not None for e in events)))
    folder.mkdir()
    np.savez_compressed(folder/'trace.npz', **z)
    report = dict(group=group, quiet=quiet, sampled=sampled, seconds=seconds, dt=env.dt, cases=cases,
        survived=sum(c['survived'] for c in cases), settled=sum(c['settled'] for c in cases),
        whole_stand=sum(c['whole_stand'] for c in cases), measured_steps=sum(len(c['steps']) for c in cases),
        recovered_with_step=sum(c['recovered_with_step'] for c in cases))
    save(folder/'report.json', report)
    print('EVAL', folder.name, {k: v for k, v in report.items() if k != 'cases'}, flush=True)
    return report


def learn(env, initial, folder, seed):
    folder.mkdir()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    actor, critic = deepcopy(initial), Head(1)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    torch.save(actor.state_dict(), folder/'initial.pt')
    env.reset(rng=rng)
    active = torch.zeros(16, dtype=torch.bool)
    active[TRAIN] = True
    history, start = [], time.perf_counter()
    for update in range(32):
        rows = []
        for _ in range(64):
            with torch.no_grad():
                x = env.features()
                mean = actor(x)
                dist = Normal(mean, STD)
                action = dist.sample()
                reward, terminal, info = env.step(action, active)
                rows.append(dict(x=x[TRAIN], action=action[TRAIN], mean=mean[TRAIN],
                    logp=dist.log_prob(action).sum(-1)[TRAIN], value=critic(x)[TRAIN], reward=reward[TRAIN],
                    terminal=terminal[TRAIN], failed=info['failed'][TRAIN], next_value=critic(env.features())[TRAIN]))
                ids = torch.nonzero(active & terminal).flatten()
                if len(ids):
                    env.reset(ids, rng)
        batch = {k: torch.stack([row[k] for row in rows]) for k in rows[0]}
        advantage = torch.zeros_like(batch['reward'])
        running = torch.zeros(len(TRAIN))
        for t in reversed(range(64)):
            delta = batch['reward'][t] + .995*batch['next_value'][t]*(~batch['failed'][t]) - batch['value'][t]
            running = delta + .995*.98*(~batch['terminal'][t])*running
            advantage[t] = running
        returns = (advantage + batch['value']).flatten()
        av = advantage.flatten()
        av = (av-av.mean())/av.std().clamp_min(1e-6)
        flat = {k: v.flatten(0, 1) for k, v in batch.items()}
        stopped, accepted, rejected, kls = False, 0, 0, []
        for _ in range(3):
            for ids in torch.randperm(len(av)).split(256):
                if not stopped:
                    old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(ao.state_dict())
                    logp = Normal(actor(flat['x'][ids]), STD).log_prob(flat['action'][ids]).sum(-1)
                    ratio = (logp-flat['logp'][ids]).exp()
                    loss = -torch.minimum(ratio*av[ids], ratio.clamp(.8, 1.2)*av[ids]).mean()
                    ao.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    ao.step()
                    with torch.no_grad():
                        kl = float(((actor(flat['x'])-flat['mean']).square().sum(-1)/(2*STD**2)).mean())
                    if not np.isfinite(kl) or kl > .01:
                        actor.load_state_dict(old_actor)
                        ao.load_state_dict(old_optimizer)
                        stopped, rejected = True, rejected+1
                    else:
                        accepted += 1
                        kls.append(kl)
                value_loss = (critic(flat['x'][ids])-returns[ids]).square().mean()
                co.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                co.step()
        assert torch.isfinite(returns).all() and torch.isfinite(value_loss)
        history.append(dict(update=update+1, transitions=(update+1)*768, reward=float(batch['reward'].mean()),
            failures=int(batch['failed'].sum()), completed=int(batch['terminal'].sum()),
            actor_updates=accepted, rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach())))
        save(folder/'history.json', history)
        if (update+1) % 4 == 0:
            print('TRAIN', seed, {k: v for k, v in history[-1].items() if k != 'accepted_kl'}, flush=True)
    torch.save(actor.state_dict(), folder/'final.pt')
    torch.save(critic.state_dict(), folder/'critic-final.pt')
    save(folder/'report.json', dict(seed=seed, transitions=24576, train_world_ids=TRAIN.tolist(),
        train_poses=list(range(6)), updates=sum(h['actor_updates'] for h in history), wall_seconds=time.perf_counter()-start))
    return actor.eval()


def setup(out):
    from .rig import Rig
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    inputs = {}
    for source in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(source, out/'sources'/source.name)
        inputs[str(source)] = digest(source)
    for source in (PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml',
                   FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
                   FOUNDATION/'protocol.json', Path(__file__).with_name('DIRECT_RECOVERY_PLAN.md')):
        inputs[str(source)] = digest(source)
    save(out/'inputs.json', inputs)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
                        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    assert not any(p.requires_grad for p in anchor.parameters())
    save(out/'contract.json', dict(control=PhysicalSkillTask(rig, FOUNDATION/'stationary_reference.npz',
        16, 'cpu', 6.).observation_contract(), radius=RADIUS, slew=SLEW, latent_std=STD,
        normalization='Frozen source normalization; previous addition divided by radius',
        actor_inputs=407, training_seeds=[901009, 911009], action_outputs=33))
    return rig, anchor


def evaluate_suite(rig, anchor, actor, out, name):
    reports = []
    for group in range(4):
        reports.append(rollout(Recovery(rig, anchor), actor, out/f'{name}-push-{group}', group=group))
    noise = rollout(Recovery(rig, anchor), actor, out/f'{name}-sampled', sampled=True)
    quiet = rollout(Recovery(rig, anchor, seconds=60.), actor, out/f'{name}-quiet', seconds=60., quiet=True)
    result = {k: sum(r[k] for r in reports) for k in ('survived', 'settled', 'measured_steps', 'recovered_with_step')}
    result.update(cases=64, heldout_settled=sum(c['heldout'] and c['settled'] for r in reports for c in r['cases']),
        heldout_cases=16, sampled_settled=noise['settled'], quiet=quiet['whole_stand'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    rig, anchor = setup(out)
    start = time.perf_counter()
    torch.manual_seed(900009)
    initial = Head().eval()
    # Same live observations, same source actor and same native trajectories.
    for quiet in (True, False):
        tag = 'quiet' if quiet else 'push'
        rollout(Recovery(rig, anchor, bare=True), None, out/f'parity-bare-{tag}', quiet=quiet)
        rollout(Recovery(rig, anchor), initial, out/f'parity-zero-{tag}', quiet=quiet)
        with np.load(out/f'parity-bare-{tag}'/'trace.npz') as a, np.load(out/f'parity-zero-{tag}'/'trace.npz') as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files), 'Zero-correction parity failed'
    save(out/'parity.json', dict(exact=True, cases=32, quiet_and_force_pulse=True))
    results = {'baseline': evaluate_suite(rig, anchor, initial, out, 'baseline')}
    for seed in (901009, 911009):
        actor = learn(Recovery(rig, anchor), initial, out/f'fit-{seed}', seed)
        results[str(seed)] = evaluate_suite(rig, anchor, actor, out, str(seed))
        save(out/'summary.json', results)
    inputs = json.loads((out/'inputs.json').read_text())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(out/'completed.json', dict(wall_seconds=time.perf_counter()-start, sources_unchanged=True,
        total_transitions=49152, results=results,
        outputs={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', results, flush=True)


if __name__ == '__main__':
    main()
