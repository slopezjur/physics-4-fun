"""Independent frozen-model, Gaussian, GAE and physical-outcome verification."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-exploration-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-start-learning-20261009-01')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


@torch.no_grad()
def main():
    inputs, done, contract = (read(OUT/name) for name in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(OUT/p) == sha for p, sha in done['outputs'].items())
    assert done['frozen_models_unchanged'] and done['optimizer_updates'] == 0
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    worlds, ankles = contract['worlds'], contract['ankles']
    lower, upper = (np.array(contract['control'][f'target_{side}_radians']) for side in ('lower', 'upper'))
    actors, critics = {}, []
    for name in ('initial', '921009', '931009'):
        actor = AnkleHead().eval()
        path = LEARNING/'fit-921009'/'initial.pt' if name == 'initial' else LEARNING/f'fit-{name}'/'final.pt'
        actor.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        actors[name] = actor
    for seed in (921009, 931009):
        critic = AnkleHead(1).eval()
        critic.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/'critic-final.pt', weights_only=True, map_location='cpu'))
        critics.append(critic)
    reports, steps, gae_error = {}, 0, 0.
    for path in sorted(OUT.glob('*/trace.npz')):
        z, r = arrays(path), read(path.with_name('report.json'))
        reports[path.parent.name] = r
        onset, quiet, dt = r['onset_frame'], r['quiet'], r['dt']
        branch = 'quiet' if quiet else 'push'
        start = arrays(OUT/f'{onset}-{branch}-start.npz')
        assert np.array_equal(z['features64'][0], start['features'])
        assert np.array_equal(z['history_before'][0], start['history'])
        assert np.array_equal(z['active_target'][:2], start['pending'])
        assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
        assert z['qpos'].shape[1] == 12
        assert np.allclose(z['time'], (np.arange(len(z['time']))+onset+2)*dt, atol=1e-10, rtol=0)
        base = arrays(OUT/f'{onset}-{branch}-zero'/'trace.npz')
        assert all(np.array_equal(z[key][:2], base[key][:2]) for key in ('qpos', 'qvel'))
        if r['actor'] == 'initial' and r['mode'] == 'mean':
            assert all(np.array_equal(z[key], base[key]) for key in z)
        if r['mode'] == 'sample':
            rng = torch.Generator().manual_seed(r['noise_seed'])
            expected = torch.stack([contract['sigma']*torch.randn((16, 1), generator=rng)[worlds] for _ in z['time']]).numpy()
            assert np.array_equal(z['innovation'], expected)
        else:
            assert not z['innovation'].any()
        if r['actor']:
            prediction = actors[r['actor']](torch.from_numpy(z['features64'])).numpy()
            assert np.allclose(prediction, z['mean'][:64], atol=2e-6, rtol=0)
            assert np.array_equal(z['latent'], z['mean']+z['innovation'])
        else:
            desired = np.zeros_like(z['latent'])
            if r['mode'] in ('negative', 'positive'):
                desired[:12] = -.075 if r['mode'] == 'negative' else .075
            assert np.allclose(.1*np.tanh(z['latent']), desired, atol=1e-7, rtol=0)
        for k, critic in enumerate(critics, 1):
            assert np.allclose(critic(torch.from_numpy(z['features64'])).numpy(), z['value'][:64, :, k], atol=4e-5, rtol=0)
            assert np.allclose(critic(torch.from_numpy(z['features64_endpoint'])).numpy(), z['next_value'][63, :, k], atol=4e-5, rtol=0)
        assert not z['value'][:, :, 0].any() and not z['next_value'][:, :, 0].any()
        assert np.array_equal(z['next_value'][:63], z['value'][1:64])
        alive, failed, terminal = z['alive'], z['failed'], z['terminal']
        assert alive[0].all() and np.array_equal(alive[1:], (alive & ~terminal)[:-1])
        delta = z['reward'][:64, :, None].astype(float)+.995*z['next_value'][:64]*(~failed[:64, :, None])-z['value'][:64]
        independently = np.zeros_like(z['advantage64'], dtype=np.float64)
        for start_frame in range(64):
            valid = alive[start_frame].copy()
            weight = 1.
            for f in range(start_frame, 64):
                independently[start_frame, valid] += weight*delta[f, valid]
                valid &= ~terminal[f]
                weight *= .995*.98
        error = float(np.abs(independently-z['advantage64']).max())
        gae_error = max(gae_error, error)
        assert error < 5e-5, (path, error)
        for c in range(3):
            original = z['advantage64'][:, :, c][alive[:64]].astype(float)
            expected = (original-original.mean())/max(original.std(ddof=1), 1e-6)
            assert np.allclose(expected, z['normalized_advantage64'][:, :, c][alive[:64]], atol=2e-5, rtol=0)
        history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
        history[~alive[:-1]] = z['history_before'][:-1][~alive[:-1]]
        assert np.array_equal(z['history_before'][1:], history)
        memory = np.where(alive[:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
        assert np.array_equal(z['previous_latent'][1:], memory)
        expected = z['previous'][:, :, ankles[:1]]+np.clip(.1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
        expected[~alive] = z['previous'][:, :, ankles[:1]][~alive]
        assert np.allclose(z['addition'][:, :, ankles[0]], expected[:, :, 0], atol=1e-7, rtol=0)
        assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
        assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
        assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
        assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
        force = np.zeros_like(z['force'])
        if not quiet:
            force[:5, :6, 0], force[:5, 6:, 0] = 150., -150.
        force[~alive] = 0
        assert np.array_equal(z['force'], force)
        xy, clearance = foot_geometry(model, z['qpos'])
        assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
        assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
        for j, case in enumerate(r['cases']):
            assert case['world'] == worlds[j] and case['pose'] == worlds[j] % 8 < 6
            valid = alive[:, j] & ~failed[:, j]
            assert case['survived'] == bool(valid.all())
            assert case['settled'] == bool(valid.all() and z['settled'][-120:, j].all())
            assert case['whole_stand'] == bool(valid.all() and z['stand'][:, j].all())
            events = measure_steps(z['time'], xy[:, j], clearance[:, j], z['loads'][:, j], valid,
                z['settled'][:, j], hit_time=None if quiet else onset*dt)
            assert events == case['steps']
            steps += len(events)
            total = 0.
            end = int(alive[:, j].sum())
            for f in reversed(range(end)):
                total = float(z['reward'][f, j])+.995*total
            assert abs(total-case['reward_return']) < 1e-9
            assert np.array_equal(np.array(case['advantage0'], dtype=np.float32), z['advantage64'][0, j])
            if not quiet:
                signed = (-1 if worlds[j] < 8 else 1)*z['addition'][:30, j, ankles[0]]
                for threshold in (.025, .06):
                    mask = (signed >= threshold) & alive[:30, j]
                    edges = np.diff(np.r_[False, mask, False].astype(int))
                    run = max(np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1), default=0)
                    assert int(run) == case['early_helpful_run'][str(threshold)]
        assert r['settled'] == sum(c['settled'] for c in r['cases'])
    summary, pairs = {}, []
    for actor in ('initial', '921009', '931009'):
        critic_index = ('initial', '921009', '931009').index(actor)
        selected = [r for r in reports.values() if r['actor'] == actor and r['mode'] == 'sample']
        for r in selected:
            branch = 'quiet' if r['quiet'] else 'push'
            baseline = reports[f"{r['onset_frame']}-{branch}-zero"]
            for c, b in zip(r['cases'], baseline['cases']):
                outcome = 'rescue' if c['settled'] and not b['settled'] else 'loss' if b['settled'] and not c['settled'] else 'unchanged'
                pairs.append(dict(actor=actor, onset=r['onset_frame'], quiet=r['quiet'], noise_seed=r['noise_seed'],
                    world=c['world'], outcome=outcome, reward_delta=c['reward_return']-b['reward_return'],
                    gae_delta=[x-y for x, y in zip(c['advantage0'], b['advantage0'])], own_critic=critic_index,
                    own_advantage=c['advantage0'][critic_index], own_normalized_advantage=c['normalized_advantage0'][critic_index],
                    helpful_run=c['early_helpful_run'], max_abs=c['early_max_abs_rad']))
        pp = [p for p in pairs if p['actor'] == actor and not p['quiet']]
        summary[actor] = dict(sampled_push_settled=sum(r['settled'] for r in selected if not r['quiet']),
            sampled_quiet_settled=sum(r['settled'] for r in selected if r['quiet']), cases_each=96,
            mean_push_settled=sum(r['settled'] for r in reports.values() if r['actor'] == actor and r['mode'] == 'mean' and not r['quiet']),
            mean_quiet_settled=sum(r['settled'] for r in reports.values() if r['actor'] == actor and r['mode'] == 'mean' and r['quiet']),
            early_helpful_025_six_intervals=sum(p['helpful_run']['0.025'] >= 6 for p in pp),
            early_helpful_060_six_intervals=sum(p['helpful_run']['0.06'] >= 6 for p in pp),
            outcomes={outcome: dict(count=len(group), reward_positive=sum(p['reward_delta'] > 0 for p in group),
                own_gae_delta_positive=sum(p['gae_delta'][critic_index] > 0 for p in group),
                own_normalized_advantage_positive=sum(p['own_normalized_advantage'] > 0 for p in group))
                for outcome in ('rescue', 'loss', 'unchanged') if (group := [p for p in pp if p['outcome'] == outcome])})
    controls = {mode: dict(push_settled=sum(r['settled'] for r in reports.values() if r['actor'] is None and r['mode'] == mode and not r['quiet']),
        quiet_settled=sum(r['settled'] for r in reports.values() if r['actor'] is None and r['mode'] == mode and r['quiet']))
        for mode in ('zero', 'negative', 'positive')}
    assert len(reports) == done['trajectories'] == 72 and len(reports)*12 == done['cases'] == 864
    audit = dict(passed=True, trajectories=72, cases=864, measured_steps=steps,
        input_hashes=len(inputs), output_hashes=len(done['outputs']), max_independent_gae_error=gae_error,
        initial_mean_exact_zero=True, gaussian_noise_verified=True, controls=controls, summary=summary)
    save(OUT/'audit.json', audit)
    save(OUT/'paired-credit.json', pairs)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
