"""State-driven scalar ankle correction with measured history and conditional PPO."""
import argparse
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from torch import nn

from . import run_direct_recovery as prior
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor

OUT = Path('logs/mimickit-training/learned-ankle-20261009-01')
CROSSED = Path('logs/mimickit-training/direct-leg-response-20261009-01')
QUIET = Path('logs/mimickit-training/direct-recovery-20261009-01/baseline-quiet')
RHO = .9
SIGMA = .4 * (1-RHO**2)**.5
ONSET_FRAMES = torch.tensor([i for i in range(40, 91) if not 55 <= i <= 65])


class AnkleHead(nn.Module):
    def __init__(self, outputs=33):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(427, 128), nn.ELU(), nn.Linear(128, 128),
                                 nn.ELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.critic = outputs == 1

    def forward(self, x):
        value = self.net(x)
        return value.squeeze(-1) if self.critic else RHO*x[:, -1:] + (1-RHO)*value


class AnkleRecovery(prior.Recovery):
    def __init__(self, rig, anchor, *, seconds=6.):
        names = [rig.model.joint(int(j)).name for j in rig.model.actuator_trnid[rig.actuators, 0]]
        self.ankles = [names.index(f'Foot_{s}_rx') for s in ('L', 'R')]
        self.history = torch.zeros(16, 3, 17)
        self.last_latent = torch.zeros(16, 1)
        self.reset_records = []
        super().__init__(rig, anchor, seconds=seconds)

    @torch.no_grad()
    def balance_features(self):
        s = self.state()
        center = s['feet'][:, :, :2].mean(1)
        height = (s['com'][:, 2]-s['feet'][:, :, 2].amin(-1)).clamp_min(.05)
        capture = s['com'][:, :2] + s['velocity'][:, :2]*(height/9.81).sqrt()[:, None]
        return torch.cat(((s['com'][:, :2]-center)/.1, s['velocity'][:, :2]/.5,
            s['angular']/2, (capture-center)/.1, s['foot_loads_n']/(self.task.total_mass*9.81),
            ((s['pelvis_height_m']-self.task.reference.qpos[self.task.ids % 8, 2])/.15)[:, None],
            s['pelvis_up'][:, None], ((s['feet'][:, :, :2]-s['com'][:, None, :2])/.2).flatten(1)), -1)

    def reset(self, ids=None, rng=None):
        super().reset(ids)
        ids = self.task.ids if ids is None else ids
        self.history[ids] = self.balance_features()[ids, None, :].expand(-1, 3, -1)
        self.last_latent[ids] = 0
        if rng is not None:
            ids = ids[ids % 8 < 6]
            n = len(ids)
            sign = 2*torch.randint(2, (n,), generator=rng)-1
            quiet = torch.rand(n, generator=rng) < .25
            self.force[ids] = 0
            self.force[ids, 0] = 150.*sign*(~quiet)
            self.onset[ids] = ONSET_FRAMES[torch.randint(len(ONSET_FRAMES), (n,), generator=rng)]
            self.reset_records.extend(dict(world=int(i), pose=int(i % 8), force_n=self.force[i].tolist(),
                                           onset_frame=int(self.onset[i])) for i in ids)

    @torch.no_grad()
    def features(self):
        x = torch.cat((self.anchor.obs_norm.normalize(self.task.observations()).clamp(-20, 20),
            self.history.flatten(1).clamp(-20, 20), self.previous[:, self.ankles[:1]]/prior.RADIUS,
            self.last_latent), -1)
        assert x.shape == (16, 427) and torch.isfinite(x).all()
        return x

    @torch.no_grad()
    def step(self, latent, active):
        assert latent.shape == (16, 1) and torch.isfinite(latent).all()
        before_history = self.history.clone()
        before_latent = self.last_latent.clone()
        expanded = torch.zeros(16, 33)
        expanded[:, self.ankles] = latent.expand(-1, 2)
        reward, terminal, info = super().step(expanded, active)
        live = self.balance_features()
        self.history[active] = torch.cat((live[active, None], before_history[active, :2]), 1)
        self.last_latent[active] = latent[active]
        assert torch.equal(self.previous[:, self.ankles[0]], self.previous[:, self.ankles[1]])
        info.update(balance_features=live, history_before=before_history,
                    previous_latent=before_latent, latent=latent.clone())
        return reward, terminal, info


@torch.no_grad()
def checks(rig, anchor):
    env = AnkleRecovery(rig, anchor)
    x = env.features()
    assert torch.equal(x, env.features()) and x.shape[-1] == 427
    assert torch.equal(env.history[:, 0], env.history[:, 1])
    assert torch.equal(env.history[:, 1], env.history[:, 2])
    active = torch.ones(16, dtype=torch.bool)
    old_h = env.history.clone()
    old_x = env.features().clone()
    env.force[:, 0] = 150.
    env.onset[:] = 99
    assert torch.equal(env.features(), old_x), 'Privileged perturbation data leaked into observations'
    env.step(torch.zeros(16, 1), active)
    assert torch.equal(env.history[:, 1:], old_h[:, :2])
    assert torch.equal(env.history[:, 0], env.balance_features())
    # Native subtree velocity must equal the COM Jacobian times generalized velocity.
    errors = []
    for frame in range(12):
        env.step(torch.full((16, 1), .5 if frame < 6 else -.5), active)
        measured = env.state()['velocity'].numpy()
        for i, data in enumerate(env.engine.datas):
            jac = np.zeros((3, rig.model.nv))
            mujoco.mj_jacSubtreeCom(rig.model, data, jac, 1)
            errors.append(float(np.abs(jac@data.qvel-measured[i]).max()))
    assert max(errors) < 1e-6
    saved_history, saved_latent = env.history.clone(), env.last_latent.clone()
    env.reset(torch.tensor([0, 8]))
    other = torch.tensor([i for i in range(16) if i not in (0, 8)])
    assert torch.equal(env.history[other], saved_history[other])
    assert torch.equal(env.last_latent[other], saved_latent[other])
    assert not env.last_latent[[0, 8]].any()
    assert torch.equal(env.history[[0, 8], 0], env.history[[0, 8], 2])
    env.reset()
    initial = AnkleHead()
    assert not initial(env.features()).any()
    x = env.features()
    x[:, -1] = .5
    assert torch.allclose(initial(x), torch.full((16, 1), .45))
    return dict(passed=True, observation_width=427, history_frames=3, physical_check_intervals=13,
        read_idempotence=True, reset_isolation=True, force_and_schedule_excluded=True,
        max_com_jacobian_velocity_error=max(errors), zero_initial_mean=True,
        conditional_mean_memory_verified=True)


def evaluate(rig, anchor, actor, out, name):
    records = []
    for timing, onset in enumerate((1., 1.7)):
        def configure(env, group, quiet):
            env.reset()
            env.force.zero_()
            if not quiet:
                strength, axis = (150., 300.)[group//2], group % 2
                env.force[:8] = strength*prior.DIRECTIONS[2*axis]
                env.force[8:] = strength*prior.DIRECTIONS[2*axis+1]
            env.onset[:] = int(np.ceil(onset/env.dt))
        for group in range(4):
            folder = out/f'{name}-t{timing}-g{group}'
            with patch.object(prior, 'configure', configure):
                report = prior.rollout(AnkleRecovery(rig, anchor), actor, folder, group=group)
            records.append(dict(timing=timing, **report))
            if name == 'baseline':
                with np.load(folder/'trace.npz') as new, np.load(CROSSED/f'baseline-t{timing}-g{group}'/'trace.npz') as old:
                    assert all(np.array_equal(new[k], old[k]) for k in new.files if k in old.files)
        with patch.object(prior, 'configure', configure), patch.object(prior, 'STD', SIGMA):
            noise = prior.rollout(AnkleRecovery(rig, anchor), actor, out/f'{name}-sampled-t{timing}', sampled=True)
        records.append(dict(timing=timing, **noise))
    quiet = prior.rollout(AnkleRecovery(rig, anchor, seconds=60.), actor, out/f'{name}-quiet', quiet=True, seconds=60.)
    if name == 'baseline':
        with np.load(out/f'{name}-quiet'/'trace.npz') as new, np.load(QUIET/'trace.npz') as old:
            assert all(np.array_equal(new[k], old[k]) for k in old.files)
    result = dict(quiet=quiet['whole_stand'], quiet_cases=16, groups={})
    for group in range(4):
        cases = [c for r in records if not r['sampled'] and r['group'] == group for c in r['cases']]
        result['groups'][str(group)] = dict(cases=len(cases), settled=sum(c['settled'] for c in cases),
            heldout_settled=sum(c['heldout'] and c['settled'] for c in cases),
            by_sign=[sum(c['settled'] for c in cases if (c['case'] < 8) == (s == 0)) for s in (0, 1)],
            recovered_with_step=sum(c['recovered_with_step'] for c in cases))
    result['sampled_settled'] = sum(r['settled'] for r in records if r['sampled'])
    save(out/f'{name}-summary.json', result)
    return result


def main():
    from .rig import Rig
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    inputs = {}
    for p in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(p)] = digest(p)
        shutil.copy2(p, out/'sources'/p.name)
    for p in (PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
              FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
              Path(__file__).with_name('LEARNED_ANKLE_RECOVERY_PLAN.md'), CROSSED/'audit.json', QUIET/'trace.npz'):
        inputs[str(p)] = digest(p)
    for p in CROSSED.glob('baseline-*/trace.npz'):
        inputs[str(p)] = digest(p)
    save(out/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    save(out/'checks.json', checks(rig, anchor))
    env = AnkleRecovery(rig, anchor)
    save(out/'contract.json', dict(observation_width=427, history_width=17, history_frames=3,
        ankles=env.ankles, rho=RHO, innovation_std=SIGMA, action_radius=.1, slew=.02,
        training_onset_frames=ONSET_FRAMES.tolist(), dt=env.dt,
        control=env.task.observation_contract(), training_seeds=[921009, 931009]))
    torch.manual_seed(920009)
    initial = AnkleHead().eval()
    start = time.perf_counter()
    results = {'baseline': evaluate(rig, anchor, initial, out, 'baseline')}
    for seed in (921009, 931009):
        training = AnkleRecovery(rig, anchor)
        # Only the actor/critic architecture and conditional innovation change here.
        with patch.object(prior, 'Head', AnkleHead), patch.object(prior, 'STD', SIGMA):
            actor = prior.learn(training, initial, out/f'fit-{seed}', seed)
        save(out/f'fit-{seed}'/'resets.json', training.reset_records)
        results[str(seed)] = evaluate(rig, anchor, actor, out, str(seed))
        save(out/'summary.json', results)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(out/'completed.json', dict(results=results, total_transitions=49152,
        wall_seconds=time.perf_counter()-start, inputs_unchanged=True,
        outputs={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', results, flush=True)


if __name__ == '__main__':
    main()
