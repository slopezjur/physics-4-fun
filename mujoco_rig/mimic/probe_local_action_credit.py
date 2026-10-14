"""Frozen first-action interventions from identical native recovery starts."""
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

OUT = Path('logs/mimickit-training/recovery-local-action-credit-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
BASE = Path('logs/mimickit-training/recovery-exploration-20261009-01')
ACTORS = (921009, 931009)
NOISE_SEEDS = (951011, 951012, 951013, 951014)
WORLDS = {'quiet': torch.arange(6), 'push': torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])}
HALF_SIGMA = SIGMA/2


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def credit(z):
    reward, value, next_value = (torch.from_numpy(z[key][:64]) for key in ('reward', 'value', 'next_value'))
    alive, failed, terminal = (torch.from_numpy(z[key][:64]) for key in ('alive', 'failed', 'terminal'))
    advantage = torch.zeros_like(value)
    running = torch.zeros_like(value[0])
    for frame in reversed(range(64)):
        delta = reward[frame]+.995*next_value[frame]*(~failed[frame])-value[frame]
        running = delta+.995*.98*(~terminal[frame])*running
        running[~alive[frame]] = 0
        advantage[frame] = running
    return advantage.numpy()


def case_metrics(z, j):
    reward = z['reward'][:, j].astype(float)*z['alive'][:, j]
    alive = z['alive'][:64, j]
    coefficient = (.995*.98)**np.arange(64)
    delta_value = .995*z['next_value'][:64, j].astype(float)*(~z['failed'][:64, j])-z['value'][:64, j].astype(float)
    last = int(np.flatnonzero(alive)[-1])
    observed64 = float(np.dot(.995**np.arange(64), reward[:64]))
    bootstrap = 0. if z['failed'][last, j] else .995**(last+1)*float(z['next_value'][last, j])
    return dict(reward_return=float(np.dot(.995**np.arange(len(reward)), reward)),
        gae0=float(z['advantage64'][0, j]),
        lambda_reward0=float(np.dot(coefficient, reward[:64])),
        value_contribution0=float(np.dot(coefficient, delta_value*alive)),
        return64=observed64, bootstrapped_return64=float(observed64+bootstrap))


@torch.no_grad()
def rollout(env, start, actor, critic, folder, *, actor_seed, noise_seed, onset, branch, first_sign):
    start.restore(env)
    restored = RecoveryStart.capture(env)
    before, after = start.arrays(), restored.arrays()
    assert all(np.array_equal(before[key], after[key]) for key in before)
    del restored
    worlds = WORLDS[branch]
    initial = env.features().clone()
    active = torch.zeros(16, dtype=torch.bool)
    active[worlds] = True
    rng = torch.Generator().manual_seed(noise_seed)
    rows, features = [], []
    duration = int(np.ceil(6/env.dt))-onset-1
    for frame in range(duration):
        x = env.features()
        value, mean = critic(x), actor(x)
        innovation = HALF_SIGMA*torch.randn((16, 1), generator=rng)
        if frame == 0:
            innovation.fill_(first_sign*HALF_SIGMA)
        latent = mean+innovation
        if frame < 64:
            features.append(x[worlds].numpy().copy())
        reward, terminal, info = env.step(latent, active)
        next_x = env.features()
        next_value = critic(next_x)
        rows.append(dict(time=np.array((onset+frame+2)*env.dt), alive=active[worlds].numpy().copy(),
            reward=reward[worlds].numpy().copy(), terminal=terminal[worlds].numpy().copy(),
            mean=mean[worlds].numpy().copy(), innovation=innovation[worlds].numpy().copy(),
            value=value[worlds].numpy().copy(), next_value=next_value[worlds].numpy().copy(),
            **{key: val[worlds].numpy().copy() for key, val in info.items()}))
        if frame == 63:
            endpoint = next_x[worlds].numpy().copy()
        active &= ~terminal
    z = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert not z['warning'].any()
    z['features64'], z['features64_endpoint'] = np.stack(features), endpoint
    assert np.array_equal(z['features64'][0], initial[worlds].numpy())
    z['advantage64'] = credit(z)
    xy, clearance = foot_geometry(env.rig.model, z['qpos'])
    assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
    assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
    cases = []
    for j, world in enumerate(worlds.tolist()):
        alive = z['alive'][:, j] & ~z['failed'][:, j]
        survived = bool(alive.all())
        settled = bool(survived and z['settled'][-int(np.ceil(2/env.dt)):, j].all())
        events = measure_steps(z['time'], xy[:, j], clearance[:, j], z['loads'][:, j], alive,
            z['settled'][:, j], hit_time=None if branch == 'quiet' else onset*env.dt)
        cases.append(dict(world=world, pose=world % 8, survived=survived, settled=settled,
            whole_stand=bool(survived and z['stand'][:, j].all()), steps=events, **case_metrics(z, j)))
    folder.mkdir()
    np.savez_compressed(folder/'trace.npz', **z)
    report = dict(actor=actor_seed, noise_seed=noise_seed, onset_frame=onset,
        quiet=branch == 'quiet', first_sign=first_sign, dt=env.dt, worlds=worlds.tolist(), cases=cases,
        settled=sum(c['settled'] for c in cases), survived=sum(c['survived'] for c in cases),
        steps=sum(len(c['steps']) for c in cases), exact_native_restore=True)
    save(folder/'report.json', report)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
        FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_LOCAL_ACTION_CREDIT_PLAN.md'),
        LEARNING/'audit.json', LEARNING/'completed.json', LEARNING/'inputs.json', LEARNING/'contract.json',
        BASE/'audit.json', BASE/'completed.json']
    paths += [LEARNING/f'fit-{seed}'/name for seed in ACTORS for name in ('final.pt', 'critic-final.pt')]
    paths += [BASE/f'{onset}-{branch}-start.npz' for onset in (44, 74) for branch in WORLDS]
    paths += [BASE/f'{onset}-{branch}-zero'/'trace.npz' for onset in (44, 74) for branch in WORLDS]
    inputs = {str(path): digest(path) for path in paths}
    for root in (LEARNING, BASE):
        assert json.loads((root/'audit.json').read_text())['passed']
        old = json.loads((root/'completed.json').read_text())['outputs']
        for path in paths:
            if path.is_relative_to(root) and str(path.relative_to(root)) in old:
                assert digest(path) == old[str(path.relative_to(root))], path
    historical = json.loads((LEARNING/'inputs.json').read_text())
    common = sorted(set(historical) & set(inputs))
    assert all(inputs[path] == historical[path] for path in common)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', dict(passed=True, shared_paths=common, count=len(common)))
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    actors, critics = {}, {}
    for seed in ACTORS:
        actors[seed], critics[seed] = AnkleHead().eval(), AnkleHead(1).eval()
        for model, name in ((actors[seed], 'final.pt'), (critics[seed], 'critic-final.pt')):
            model.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/name, weights_only=True, map_location='cpu'))
            model.requires_grad_(False)
    models = [anchor, *actors.values(), *critics.values()]
    assert all(not p.requires_grad for model in models for p in model.parameters())
    frozen = [{key: value.clone() for key, value in model.state_dict().items()} for model in models]
    env = AnkleRecovery(rig, anchor)
    save(OUT/'contract.json', dict(worlds={key: value.tolist() for key, value in WORLDS.items()},
        actors=list(ACTORS), noise_seeds=list(NOISE_SEEDS), onsets=[44, 74], poses=list(range(6)),
        sigma=HALF_SIGMA, gamma=.995, gae_lambda=.98, gae_intervals=64, dt=env.dt,
        ankles=env.ankles, control=env.task.observation_contract(),
        cases=576, pairs=288, trajectories=64, optimizer_updates=0,
        first_draw_consumed=True, first_signs=[-1, 1], episode_endpoint_seconds=6.,
        historical_input_hashes=len(common), quiet_worlds_unique=True))
    records, start_time = [], time.perf_counter()
    for onset in (44, 74):
        env.reset()
        env.force.zero_()
        env.onset.fill_(onset)
        active = torch.zeros(16, dtype=torch.bool)
        active[WORLDS['push']] = True
        for _ in range(onset):
            _, terminal, info = env.step(torch.zeros(16, 1), active)
            assert not terminal[active].any() and not info['warning'].any()
        common_start = RecoveryStart.capture(env)
        for branch, worlds in WORLDS.items():
            common_start.restore(env)
            env.force.zero_()
            if branch == 'push':
                env.force[:8, 0], env.force[8:, 0] = 150., -150.
            _, terminal, info = env.step(torch.zeros(16, 1), active)
            assert not terminal[active].any() and not info['warning'].any()
            start = RecoveryStart.capture(env)
            saved_start = dict(features=env.features()[worlds].numpy(), pending=start.pending[:, worlds].numpy(),
                history=env.history[worlds].numpy(), qpos=env.engine.native_qpos()[worlds].numpy(),
                qvel=env.engine._array('qvel')[worlds].numpy())
            baseline = arrays(BASE/f'{onset}-{branch}-start.npz')
            indices = np.arange(len(worlds))
            assert all(np.array_equal(value, np.take(baseline[key], indices, axis=1 if key == 'pending' else 0))
                       for key, value in saved_start.items())
            np.savez_compressed(OUT/f'{onset}-{branch}-start.npz', worlds=worlds.numpy(), **saved_start)
            np.savez_compressed(OUT/f'{onset}-{branch}-native-start.npz', **start.arrays())
            for actor_seed in ACTORS:
                for noise_seed in NOISE_SEEDS:
                    for first_sign, tag in ((-1, 'negative'), (1, 'positive')):
                        name = f'{onset}-{branch}-{actor_seed}-{noise_seed}-{tag}'
                        report = rollout(env, start, actors[actor_seed], critics[actor_seed], OUT/name,
                            actor_seed=actor_seed, noise_seed=noise_seed, onset=onset,
                            branch=branch, first_sign=first_sign)
                        records.append(dict(name=name, cases=len(worlds), settled=report['settled'],
                                            survived=report['survived'], steps=report['steps']))
                    print('PAIR', onset, branch, actor_seed, noise_seed,
                          [record['settled'] for record in records[-2:]], flush=True)
            del start
        del common_start
    assert len(records) == 64 and sum(row['cases'] for row in records) == 576
    save(OUT/'summary.json', records)
    assert all(torch.equal(value, model.state_dict()[key])
               for state, model in zip(frozen, models) for key, value in state.items())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'completed.json', dict(cases=576, pairs=288, trajectories=64, optimizer_updates=0,
        frozen_models_unchanged=True, inputs_unchanged=True, wall_seconds=time.perf_counter()-start_time,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
