"""Frozen matched exploration trajectories and 64-interval credit evidence."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .recovery_start_bank import RecoveryStart
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead, AnkleRecovery, SIGMA

OUT = Path('logs/mimickit-training/recovery-exploration-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-start-learning-20261009-01')
TRAIN = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
NOISE_SEEDS = (951001, 951002, 951003, 951004)


def longest(mask):
    best = current = 0
    for value in mask:
        current = current+1 if value else 0
        best = max(best, current)
    return best


def credit(z):
    reward, value, next_value = (torch.from_numpy(z[key][:64]) for key in ('reward', 'value', 'next_value'))
    alive, failed, terminal = (torch.from_numpy(z[key][:64]) for key in ('alive', 'failed', 'terminal'))
    advantage = torch.zeros_like(value)
    running = torch.zeros_like(value[0])
    for f in reversed(range(64)):
        delta = reward[f, :, None]+.995*next_value[f]*(~failed[f, :, None])-value[f]
        running = delta+.995*.98*(~terminal[f, :, None])*running
        running[~alive[f]] = 0
        advantage[f] = running
    normalized = torch.zeros_like(advantage)
    for c in range(3):
        selected = advantage[:, :, c][alive]
        normalized[:, :, c][alive] = (selected-selected.mean())/selected.std().clamp_min(1e-6)
    return advantage.numpy(), normalized.numpy()


@torch.no_grad()
def rollout(env, start, actor, critics, folder, *, mode, noise_seed, onset, quiet, actor_name):
    start.restore(env)
    initial = env.features().clone()
    active = torch.zeros(16, dtype=torch.bool)
    active[TRAIN] = True
    rng = torch.Generator().manual_seed(noise_seed or 0)
    rows, features = [], []
    duration = int(np.ceil(6/env.dt))-onset-1
    for frame in range(duration):
        x = env.features()
        values = torch.stack([torch.zeros(16), *[c(x) for c in critics.values()]], -1)
        mean = torch.zeros(16, 1) if actor is None else actor(x)
        innovation = SIGMA*torch.randn((16, 1), generator=rng) if mode == 'sample' else torch.zeros(16, 1)
        if mode in ('negative', 'positive'):
            magnitude = (-.075 if mode == 'negative' else .075) if frame < 12 else 0.
            latent = torch.atanh(torch.full((16, 1), magnitude)/.1)
        elif mode == 'zero':
            latent = torch.zeros(16, 1)
        else:
            latent = mean+innovation
        if frame < 64:
            features.append(x[TRAIN].numpy().copy())
        reward, terminal, info = env.step(latent, active)
        next_x = env.features()
        next_values = torch.stack([torch.zeros(16), *[c(next_x) for c in critics.values()]], -1)
        rows.append(dict(time=np.array((onset+frame+2)*env.dt), alive=active[TRAIN].numpy().copy(),
            reward=reward[TRAIN].numpy().copy(), terminal=terminal[TRAIN].numpy().copy(),
            mean=mean[TRAIN].numpy().copy(), innovation=innovation[TRAIN].numpy().copy(),
            value=values[TRAIN].numpy().copy(), next_value=next_values[TRAIN].numpy().copy(),
            **{k: v[TRAIN].numpy().copy() for k, v in info.items()}))
        if frame == 63:
            endpoint = next_x[TRAIN].numpy().copy()
        active &= ~terminal
    z = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    assert not z['warning'].any()
    z['features64'], z['features64_endpoint'] = np.stack(features), endpoint
    assert np.array_equal(z['features64'][0], initial[TRAIN].numpy())
    z['advantage64'], z['normalized_advantage64'] = credit(z)
    xy, clearance = foot_geometry(env.rig.model, z['qpos'])
    assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
    assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
    cases = []
    for j, world in enumerate(TRAIN.tolist()):
        alive = z['alive'][:, j] & ~z['failed'][:, j]
        survived = bool(alive.all())
        settled = bool(survived and z['settled'][-int(np.ceil(2/env.dt)):, j].all())
        events = measure_steps(z['time'], xy[:, j], clearance[:, j], z['loads'][:, j], alive,
            z['settled'][:, j], hit_time=None if quiet else onset*env.dt)
        early = z['addition'][:30, j, env.ankles[0]]
        sign = -1 if world < 8 else 1
        coverage = None if quiet else {str(threshold): longest((sign*early >= threshold) & z['alive'][:30, j]) for threshold in (.025, .06)}
        cases.append(dict(world=world, pose=world % 8, survived=survived, settled=settled,
            whole_stand=bool(survived and z['stand'][:, j].all()), steps=events,
            reward_return=float(np.dot(.995**np.arange(duration), z['reward'][:, j]*z['alive'][:, j])),
            advantage0=z['advantage64'][0, j].tolist(), normalized_advantage0=z['normalized_advantage64'][0, j].tolist(),
            early_max_abs_rad=float(np.abs(early).max()), early_rms_rad=float(np.sqrt(np.mean(early**2))),
            early_helpful_run=coverage))
    folder.mkdir()
    np.savez_compressed(folder/'trace.npz', **z)
    report = dict(mode=mode, actor=actor_name, noise_seed=noise_seed, onset_frame=onset,
        quiet=quiet, dt=env.dt, cases=cases, settled=sum(c['settled'] for c in cases),
        survived=sum(c['survived'] for c in cases), steps=sum(len(c['steps']) for c in cases))
    save(folder/'report.json', report)
    print('EVAL', folder.name, report['settled'], report['survived'], report['steps'], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
        FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_EXPLORATION_PLAN.md'), LEARNING/'audit.json', LEARNING/'completed.json',
        Path('logs/mimickit-training/reward-alignment-20261009-01/audit.json')]
    paths += [LEARNING/f'fit-{s}'/n for s in (921009, 931009) for n in ('initial.pt', 'final.pt', 'critic-final.pt')]
    inputs = {str(p): digest(p) for p in paths}
    assert json.loads((LEARNING/'audit.json').read_text())['passed']
    old = json.loads((LEARNING/'completed.json').read_text())['outputs']
    for path in paths:
        if path.is_relative_to(LEARNING) and str(path.relative_to(LEARNING)) in old:
            assert digest(path) == old[str(path.relative_to(LEARNING))]
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    actors, critics = {}, {}
    for name, path in [('initial', LEARNING/'fit-921009'/'initial.pt'),
                       *[(str(s), LEARNING/f'fit-{s}'/'final.pt') for s in (921009, 931009)]]:
        actor = AnkleHead().eval().requires_grad_(False)
        actor.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        actors[name] = actor
    for seed in (921009, 931009):
        critic = AnkleHead(1).eval().requires_grad_(False)
        critic.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/'critic-final.pt', weights_only=True, map_location='cpu'))
        critics[str(seed)] = critic
    frozen = {name: {k: v.detach().clone() for k, v in model.state_dict().items()}
        for name, model in [('anchor', anchor), *actors.items(), *[(f'critic-{k}', v) for k, v in critics.items()]]}
    start_time = time.perf_counter()
    env = AnkleRecovery(rig, anchor)
    save(OUT/'contract.json', dict(worlds=TRAIN.tolist(), poses=list(range(6)), onsets=[44, 74],
        noise_seeds=list(NOISE_SEEDS), sigma=SIGMA, gamma=.995, gae_lambda=.98, gae_intervals=64,
        critics=['initial-zero', *critics], ankles=env.ankles, control=env.task.observation_contract(),
        cases=864, trajectories=72, optimizer_updates=0))
    for onset in (44, 74):
        env.reset()
        env.force.zero_()
        env.onset.fill_(onset)
        active = torch.zeros(16, dtype=torch.bool)
        active[TRAIN] = True
        for _ in range(onset):
            _, terminal, _ = env.step(torch.zeros(16, 1), active)
            assert not terminal[TRAIN].any()
        common = RecoveryStart.capture(env)
        for quiet in (True, False):
            common.restore(env)
            env.force.zero_()
            if not quiet:
                env.force[:8, 0], env.force[8:, 0] = 150., -150.
            _, terminal, _ = env.step(torch.zeros(16, 1), active)
            assert not terminal[TRAIN].any()
            start = RecoveryStart.capture(env)
            branch = 'quiet' if quiet else 'push'
            np.savez_compressed(OUT/f'{onset}-{branch}-start.npz', features=env.features()[TRAIN].numpy(),
                pending=start.pending[:, TRAIN].numpy(), history=env.history[TRAIN].numpy(),
                qpos=env.engine.native_qpos()[TRAIN].numpy(), qvel=env.engine._array('qvel')[TRAIN].numpy())
            for mode in ('zero', 'negative', 'positive'):
                rollout(env, start, None, critics, OUT/f'{onset}-{branch}-{mode}', mode=mode,
                    noise_seed=None, onset=onset, quiet=quiet, actor_name=None)
            for name, actor in actors.items():
                rollout(env, start, actor, critics, OUT/f'{onset}-{branch}-{name}-mean', mode='mean',
                    noise_seed=None, onset=onset, quiet=quiet, actor_name=name)
                for seed in NOISE_SEEDS:
                    rollout(env, start, actor, critics, OUT/f'{onset}-{branch}-{name}-{seed}', mode='sample',
                        noise_seed=seed, onset=onset, quiet=quiet, actor_name=name)
            del start
        del common
    for name, model in [('anchor', anchor), *actors.items(), *[(f'critic-{k}', v) for k, v in critics.items()]]:
        assert all(torch.equal(frozen[name][key], value) for key, value in model.state_dict().items())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'completed.json', dict(cases=864, trajectories=72, optimizer_updates=0, frozen_models_unchanged=True,
        wall_seconds=time.perf_counter()-start_time, inputs_unchanged=True,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
