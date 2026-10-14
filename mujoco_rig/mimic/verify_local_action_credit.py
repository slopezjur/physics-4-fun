"""Independent frozen first-action pair, credit and physical-outcome audit."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-local-action-credit-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
BASE = Path('logs/mimickit-training/recovery-exploration-20261009-01')
TRAIN = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
NOISE_SEEDS = [951011, 951012, 951013, 951014]
GAMMA, LAMBDA, SIGMA = .995, .98, .08717797887081347


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assert_close(actual, expected, label, errors, atol=1e-4, rtol=2e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    error = float(np.abs(actual.astype(float)-expected.astype(float)).max(initial=0.))
    errors[label] = max(errors.get(label, 0.), error)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (label, error)


def canonical(mapping):
    result = {str(Path(path).resolve()).casefold(): sha for path, sha in mapping.items()}
    assert len(result) == len(mapping)
    return result


def provenance(inputs):
    previous = canonical(read(LEARNING/'inputs.json'))
    current = canonical(inputs)
    shared = previous.keys() & current.keys()
    assert shared
    assert all(previous[path] == current[path] for path in shared)
    saved = read(OUT/'historical-inputs.json')
    assert saved['passed'] is True
    recorded = {str(Path(path).resolve()).casefold() for path in saved['shared_paths']}
    assert recorded == shared and saved['count'] == len(shared)
    for root in (LEARNING, BASE):
        assert read(root/'audit.json')['passed'] is True
        outputs = read(root/'completed.json')['outputs']
        for path in map(Path, inputs):
            resolved = path.resolve()
            if resolved.is_relative_to(root.resolve()):
                relative = str(resolved.relative_to(root.resolve()))
                if relative in outputs:
                    assert digest(path) == outputs[relative], path
    return len(shared)


def starts(onset, branch):
    worlds = list(range(6)) if branch == 'quiet' else TRAIN
    selected = [TRAIN.index(world) for world in worlds]
    current = arrays(OUT/f'{onset}-{branch}-start.npz')
    original = arrays(BASE/f'{onset}-{branch}-start.npz')
    assert np.array_equal(current['worlds'], worlds)
    for key in ('features', 'history', 'qpos', 'qvel'):
        assert np.array_equal(current[key], original[key][selected]), (onset, branch, key)
    assert np.array_equal(current['pending'], original['pending'][:, selected])
    native = arrays(OUT/f'{onset}-{branch}-native-start.npz')
    assert all(np.isfinite(value).all() for value in native.values())
    assert native['native_integration'].shape[0] == 16
    assert np.array_equal(native['pending_oldest_first'][:, worlds], current['pending'])
    assert np.array_equal(native['env_history'][worlds], current['history'])
    assert not native['env_previous'][worlds].any() and not native['env_last_latent'][worlds].any()
    assert (native['task_steps'][worlds] == onset+1).all()
    assert (native['env_onset'][worlds] == onset).all()
    assert not current['features'][:, -2:].any()
    if branch == 'quiet':
        assert not native['env_force'][worlds].any()
    else:
        expected = np.zeros((12, 3))
        expected[:6, 0], expected[6:, 0] = 150., -150.
        assert np.array_equal(native['env_force'][worlds], expected)
    return worlds, selected, current


def credit(trace):
    reward = trace['reward'][:64].astype(float)
    value = trace['value'][:64].astype(float)
    next_value = trace['next_value'][:64].astype(float)
    value_delta = GAMMA*next_value*(~trace['failed'][:64])-value
    result = []
    for source in (reward+value_delta, reward, value_delta):
        advantage = np.zeros_like(source)
        for start in range(64):
            weight = trace['alive'][start].astype(float)
            for frame in range(start, 64):
                advantage[start] += weight*source[frame]
                weight *= GAMMA*LAMBDA*(~trace['terminal'][frame])
        result.append(advantage)
    return result


def physics(trace, report, start, zero, selected, model, contract, errors):
    z, worlds = trace, report['worlds']
    count, frames = len(worlds), len(z['time'])
    onset, dt, ankles = report['onset_frame'], report['dt'], contract['ankles']
    assert frames == int(np.ceil(6/dt))-onset-1
    assert np.allclose(z['time'], (np.arange(frames)+onset+2)*dt, atol=1e-10, rtol=0)
    assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
    assert z['qpos'].shape == (frames, count, model.nq)
    assert z['qvel'].shape == (frames, count, model.nv)
    assert np.array_equal(z['features64'][0], start['features'])
    assert np.array_equal(z['history_before'][0], start['history'])
    assert np.array_equal(z['active_target'][:2], start['pending'])
    assert np.array_equal(z['qpos'][:2], zero['qpos'][:2, selected])
    assert np.array_equal(z['qvel'][:2], zero['qvel'][:2, selected])
    alive, terminal, failed = z['alive'], z['terminal'], z['failed']
    assert alive.shape == terminal.shape == failed.shape == (frames, count)
    assert alive.dtype == terminal.dtype == failed.dtype == np.bool_
    assert alive[0].all() and np.array_equal(alive[1:], (alive & ~terminal)[:-1])
    assert not np.any(failed & ~terminal)
    assert not z['previous'][0].any() and not z['previous_latent'][0].any()
    assert np.array_equal(z['previous'][1:], z['addition'][:-1])
    history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
    history[~alive[:-1]] = z['history_before'][:-1][~alive[:-1]]
    assert np.array_equal(z['history_before'][1:], history)
    memory = np.where(alive[:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
    assert np.array_equal(z['previous_latent'][1:], memory)
    expected = z['previous'][:, :, ankles[:1]]+np.clip(
        .1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
    expected[~alive] = z['previous'][:, :, ankles[:1]][~alive]
    assert_close(z['addition'][:, :, ankles[0]], expected[:, :, 0], 'addition', errors, 1e-7, 0.)
    assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
    assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
    assert np.abs(z['addition']).max() <= .1+1e-7
    assert np.abs(z['addition']-z['previous']).max() <= .02+1e-7
    lower, upper = (np.asarray(contract['control'][f'target_{side}_radians']) for side in ('lower', 'upper'))
    assert_close(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), 'issued', errors, 1e-7, 0.)
    assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
    assert not z['active_target'][:, :, 33:66].any() and (z['active_target'][:, :, -1] == 1).all()
    force = np.zeros_like(z['force'])
    if not report['quiet']:
        force[:5, :6, 0], force[:5, 6:, 0] = 150., -150.
    force[~alive] = 0
    assert np.array_equal(z['force'], force)
    assert np.array_equal(z['features64'][:, :, -1:], z['previous_latent'][:64])
    assert_close(z['features64'][:, :, -2], z['previous'][:64, :, ankles[0]]/.1,
                 'feature_previous_addition', errors, 1e-7, 0.)
    assert np.array_equal(z['features64'][:, :, 374:425], z['history_before'][:64].reshape(64, count, 51).clip(-20, 20))
    xy, clearance = foot_geometry(model, z['qpos'])
    assert_close(xy, z['xy'], 'foot_xy', errors, 2e-6, 0.)
    assert_close(clearance, z['clearance'], 'foot_clearance', errors, 2e-6, 0.)
    assert len(report['cases']) == count
    events_count = 0
    for j, case in enumerate(report['cases']):
        assert case['world'] == worlds[j] and case['pose'] == worlds[j] % 8 < 6
        valid = alive[:, j] & ~failed[:, j]
        assert case['survived'] == bool(valid.all())
        assert case['settled'] == bool(valid.all() and z['settled'][-int(np.ceil(2/dt)):, j].all())
        assert case['whole_stand'] == bool(valid.all() and z['stand'][:, j].all())
        events = measure_steps(z['time'], xy[:, j], clearance[:, j], z['loads'][:, j], valid,
                               z['settled'][:, j], hit_time=None if report['quiet'] else onset*dt)
        assert events == case['steps']
        events_count += len(events)
    assert report['steps'] == events_count
    assert report['settled'] == sum(case['settled'] for case in report['cases'])
    assert report['survived'] == sum(case['survived'] for case in report['cases'])
    return events_count


@torch.no_grad()
def model_and_credit(trace, report, actor, critic, errors):
    z, count = trace, len(report['worlds'])
    assert z['features64'].shape == (64, count, 427) and z['features64_endpoint'].shape == (count, 427)
    x = torch.from_numpy(z['features64']).reshape(-1, 427)
    assert_close(actor(x).numpy().reshape(64, count, 1), z['mean'][:64], 'actor_mean', errors, 2e-6, 0.)
    assert_close(critic(x).numpy().reshape(64, count), z['value'][:64], 'critic_value', errors, 4e-5, 0.)
    endpoint = torch.from_numpy(z['features64_endpoint'])
    assert_close(critic(endpoint).numpy(), z['next_value'][63], 'critic_endpoint', errors, 4e-5, 0.)
    assert np.array_equal(z['next_value'][:63], z['value'][1:64])
    assert np.array_equal(z['latent'], z['mean']+z['innovation'])
    expected_gae, reward_part, value_part = credit(z)
    assert_close(z['advantage64'], expected_gae, 'independent_gae', errors)
    assert not z['advantage64'][~z['alive'][:64]].any()
    for j, case in enumerate(report['cases']):
        total, partial = 0., 0.
        alive_frames = np.flatnonzero(z['alive'][:, j])
        assert len(alive_frames) and np.array_equal(alive_frames, np.arange(len(alive_frames)))
        for frame in reversed(alive_frames):
            total = float(z['reward'][frame, j])+GAMMA*total
        first = min(64, len(alive_frames))
        for frame in reversed(range(first)):
            partial = float(z['reward'][frame, j])+GAMMA*partial
        last = first-1
        bootstrapped = partial+(0. if z['failed'][last, j] else GAMMA**first*float(z['next_value'][last, j]))
        assert abs(total-case['reward_return']) < 1e-9
        assert abs(partial-case['return64']) < 1e-9
        assert abs(bootstrapped-case['bootstrapped_return64']) < 1e-8
        assert float(z['advantage64'][0, j]) == case['gae0']
        assert_close(case['gae0'], expected_gae[0, j], 'reported_gae', errors)
        assert_close(case['lambda_reward0'], reward_part[0, j], 'lambda_reward', errors, 1e-9, 1e-12)
        assert_close(case['value_contribution0'], value_part[0, j], 'value_contribution', errors, 1e-9, 1e-12)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    inputs, done, contract = (read(OUT/name) for name in ('inputs.json', 'completed.json', 'contract.json'))
    for path, sha in inputs.items():
        assert digest(Path(path)) == sha, path
    for path, sha in done['outputs'].items():
        assert digest(OUT/path) == sha, path
    shared_inputs = provenance(inputs)
    assert done['frozen_models_unchanged'] is True and done['inputs_unchanged'] is True
    assert done['optimizer_updates'] == contract['optimizer_updates'] == 0
    assert (done['cases'], done['pairs'], done['trajectories']) == (576, 288, 64)
    assert (contract['cases'], contract['pairs'], contract['trajectories']) == (576, 288, 64)
    assert contract['sigma'] == SIGMA and contract['gamma'] == GAMMA and contract['gae_lambda'] == LAMBDA
    assert contract['gae_intervals'] == 64 and contract['episode_endpoint_seconds'] == 6.
    assert contract['first_draw_consumed'] is True and contract['first_signs'] == [-1, 1]
    assert contract['quiet_worlds_unique'] is True
    assert contract['worlds'] == dict(quiet=list(range(6)), push=TRAIN)
    assert contract['onsets'] == [44, 74] and contract['actors'] == [921009, 931009]
    assert contract['noise_seeds'] == NOISE_SEEDS and contract['poses'] == list(range(6))
    assert contract['historical_input_hashes'] == shared_inputs
    assert not list(OUT.rglob('*.pt'))
    assert len(list(OUT.glob('*/trace.npz'))) == 64
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    actors, critics = {}, {}
    for seed in (921009, 931009):
        actors[seed], critics[seed] = AnkleHead().eval(), AnkleHead(1).eval()
        for network, filename in ((actors[seed], 'final.pt'), (critics[seed], 'critic-final.pt')):
            path = LEARNING/f'fit-{seed}'/filename
            assert str(path.resolve()).casefold() in canonical(inputs)
            network.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            network.requires_grad_(False)
    summary = {row['name']: row for row in read(OUT/'summary.json')}
    assert len(summary) == 64
    errors, pairs, total_cases, step_count = {}, [], 0, 0
    for onset in (44, 74):
        for branch in ('quiet', 'push'):
            worlds, selected, start = starts(onset, branch)
            zero_path = BASE/f'{onset}-{branch}-zero'/'trace.npz'
            assert str(zero_path.resolve()).casefold() in canonical(inputs)
            zero = arrays(zero_path)
            duration = int(np.ceil(6/contract['dt']))-onset-1
            for actor_seed in (921009, 931009):
                initial = torch.from_numpy(start['features'])
                initial_mean, initial_value = actors[actor_seed](initial).numpy(), critics[actor_seed](initial).numpy()
                for noise_seed in NOISE_SEEDS:
                    generator = torch.Generator().manual_seed(noise_seed)
                    noise = torch.stack([SIGMA*torch.randn((16, 1), generator=generator) for _ in range(duration)]).numpy()
                    arms = []
                    for sign, tag in ((-1, 'negative'), (1, 'positive')):
                        name = f'{onset}-{branch}-{actor_seed}-{noise_seed}-{tag}'
                        folder = OUT/name
                        z, report = arrays(folder/'trace.npz'), read(folder/'report.json')
                        assert report['actor'] == actor_seed and report['noise_seed'] == noise_seed
                        assert report['first_sign'] == sign and report['onset_frame'] == onset
                        assert report['quiet'] == (branch == 'quiet') and report['worlds'] == worlds
                        assert report['exact_native_restore'] is True and report['dt'] == contract['dt']
                        assert z['innovation'].shape == z['mean'].shape == z['latent'].shape == (duration, len(worlds), 1)
                        assert z['value'].shape == z['next_value'].shape == z['reward'].shape == (duration, len(worlds))
                        expected = noise[:, worlds].copy()
                        expected[0].fill(np.float32(sign*SIGMA))
                        assert np.array_equal(z['innovation'], expected), (name, 'Gaussian sequence')
                        assert_close(z['mean'][0], initial_mean, 'initial_mean', errors, 2e-6, 0.)
                        assert_close(z['value'][0], initial_value, 'initial_value', errors, 4e-5, 0.)
                        step_count += physics(z, report, start, zero, selected, model, contract, errors)
                        model_and_credit(z, report, actors[actor_seed], critics[actor_seed], errors)
                        total_cases += len(worlds)
                        assert summary[name]['cases'] == len(worlds)
                        for key in ('settled', 'survived', 'steps'):
                            assert summary[name][key] == report[key]
                        arms.append((z, report))
                    (negative, negative_report), (positive, positive_report) = arms
                    for key in ('qpos', 'qvel', 'active_target', 'balance_features'):
                        assert np.array_equal(negative[key][:2], positive[key][:2]), (onset, branch, actor_seed, noise_seed, key)
                    for key in ('features64', 'mean', 'value', 'history_before', 'previous', 'previous_latent', 'nominal'):
                        assert np.array_equal(negative[key][0], positive[key][0]), key
                    assert np.array_equal(negative['innovation'][1:], positive['innovation'][1:])
                    assert np.array_equal(negative['innovation'][0], -positive['innovation'][0])
                    assert_close(positive['latent'][0]-negative['latent'][0],
                                 np.full((len(worlds), 1), 2*SIGMA), 'first_latent_separation', errors, 1e-7, 0.)
                    for left, right in zip(negative_report['cases'], positive_report['cases'], strict=True):
                        assert left['world'] == right['world']
                        pairs.append(dict(actor=actor_seed, onset_frame=onset, quiet=branch == 'quiet',
                                          noise_seed=noise_seed, world=left['world'],
                                          negative_settled=left['settled'], positive_settled=right['settled'],
                                          reward_delta=right['reward_return']-left['reward_return'],
                                          gae_delta=right['gae0']-left['gae0'],
                                          lambda_reward_delta=right['lambda_reward0']-left['lambda_reward0'],
                                          value_delta=right['value_contribution0']-left['value_contribution0'],
                                          return64_delta=right['return64']-left['return64'],
                                          bootstrapped_return64_delta=right['bootstrapped_return64']-left['bootstrapped_return64']))
    assert total_cases == 576 and len(pairs) == 288
    assert sum(not row['quiet'] for row in pairs) == 192 and sum(row['quiet'] for row in pairs) == 96
    assert all(row['world'] % 8 < 6 for row in pairs)
    audit = dict(passed=True, trajectories=64, cases=576, pairs=288, quiet_cases=192, push_cases=384,
                 measured_steps=step_count, optimizer_updates=0, new_policy_checkpoints=0,
                 input_hashes=len(inputs), output_hashes=len(done['outputs']), historical_input_hashes=shared_inputs,
                 exact_historical_selected_starts=True, exact_paired_initial_states=True,
                 first_two_physical_intervals_equal=True, first_draw_consumed=True,
                 common_future_noise_verified=True, own_actor_critic_first64_verified=True,
                 frozen_models_verified=True, geometry_and_outcomes_verified=True,
                 independent_full_and_truncated_returns=True, independent_gae_and_decomposition=True,
                 max_absolute_errors=errors,
                 scope='Frozen paired first-action interventions on training poses. Later latent commands '
                 'remain state-dependent under shared future innovations. These are finite-horizon sampled '
                 'continuation and critic comparisons, not historical PPO replay or a calibration proof.')
    save(OUT/'audit-pairs.json', pairs)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
