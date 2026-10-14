"""Independent saved-array audit of the instrumented historical PPO streams."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train_coordinated_ankle import AnkleHead

BASE = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-update-trace-20261009-01')
EXPLORATION = Path('logs/mimickit-training/recovery-exploration-20261009-01')
WORLDS = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
SIGMA = .08717797887081347
DISCOUNT, LAMBDA = .995, .98


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(actual, expected, label, errors, *, atol=1e-4, rtol=2e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    error = float(np.abs(actual.astype(float)-expected.astype(float)).max(initial=0.))
    errors[label] = max(errors.get(label, 0.), error)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (label, error)


def independent_credit(batch):
    reward = batch['reward'].astype(float)
    value = batch['value'].astype(float)
    following = batch['next_value'].astype(float)
    continuation = DISCOUNT*LAMBDA*(~batch['terminal'])
    value_delta = DISCOUNT*following*(~batch['failed'])-value
    result = []
    for source in (reward+value_delta, reward, value_delta):
        terms = np.zeros_like(source)
        # Explicit finite sums independently check the production backward recurrence.
        for start in range(64):
            weight = np.ones(12)
            for stop in range(start, 64):
                terms[start] += weight*source[stop]
                weight *= continuation[stop]
        result.append(terms)
    return result


def masks(batch):
    force, age = batch['force'].ravel(), batch['age'].ravel()
    return [force == 0, force > 0, force < 0, age < 30, age >= 30]


def check_gradients(batch, errors, initial=None, original=None):
    groups = masks(batch)
    counts = np.array([mask.sum() for mask in groups])
    assert np.array_equal(counts, batch['group_counts'])
    assert counts[:3].sum() == counts[3:].sum() == 768
    gradient = batch['gradient_last']
    assert gradient.shape == batch['gradient_last_analytic'].shape == (5, 129)
    close(gradient, batch['gradient_last_analytic'], 'last_layer_analytic', errors,
          atol=5e-6, rtol=5e-5)
    close(gradient[:3].sum(0), gradient[3:].sum(0), 'last_layer_partitions', errors,
          atol=5e-6, rtol=5e-5)
    score = ((batch['action'].ravel().astype(float)-batch['mean'].ravel())/SIGMA**2
             *batch['normalized_advantage'].ravel())
    bias = np.array([.1*score[group].sum()/768 for group in groups])
    close(gradient[:, -1], bias, 'independent_bias_gradient', errors, atol=5e-6, rtol=5e-5)
    if initial is not None:
        with torch.no_grad():
            hidden = initial.net[:-1](torch.from_numpy(original['x']).reshape(-1, 427)).numpy()
        design = np.c_[hidden.astype(float), np.ones(768)]
        independent = np.stack([.1*design[group].T@score[group]/768 for group in groups])
        close(gradient, independent, 'initial_independent_last_layer', errors,
              atol=5e-6, rtol=5e-5)
    gram, norm = batch['gradient_gram'], batch['gradient_norm']
    assert gram.shape == (5, 5) and norm.shape == (5,)
    close(gram, gram.T, 'gram_symmetry', errors, atol=1e-12, rtol=1e-12)
    close(np.diag(gram), norm**2, 'gradient_norm_identity', errors, atol=1e-12, rtol=1e-10)
    scale = max(1., float(np.trace(gram)))
    assert np.linalg.eigvalsh(gram).min() >= -1e-10*scale
    close(gram[:3].sum(0), gram[3:].sum(0), 'full_gradient_partitions', errors,
          atol=5e-6, rtol=5e-5)
    close(np.array(gram[:3, :3].sum()), np.array(gram[3:, 3:].sum()),
          'full_gradient_partition_norm', errors, atol=5e-6, rtol=5e-5)
    # Removing the final layer leaves another Gram matrix of the hidden-layer gradients.
    residual = gram-gradient@gradient.T
    assert np.linalg.eigvalsh(residual).min() >= -1e-9*scale


def check_labels(batches, records):
    force = np.concatenate([z['force'] for z in batches])
    age = np.concatenate([z['age'] for z in batches])
    terminal = np.concatenate([z['terminal'] for z in batches])
    previous = np.concatenate([z['previous_latent'] for z in batches])
    action = np.concatenate([z['action'] for z in batches])
    assert force.shape == age.shape == terminal.shape == previous.shape == (2048, 12)
    assert all(row['world'] in WORLDS for row in records)
    consumed = 0
    for j, world in enumerate(WORLDS):
        sequence = [row for row in records if row['world'] == world]
        assert sequence
        for row in sequence:
            assert row['pose'] == int(world % 8) < 6
            assert row['start_frame'] == row['onset_frame']+1
            assert 40 <= row['onset_frame'] <= 90 and not 55 <= row['onset_frame'] <= 65
            assert row['force_n'][0] in (-150., 0., 150.) and row['force_n'][1:] == [0., 0.]
        episode, expected_age = 0, 0
        for frame in range(2048):
            assert episode < len(sequence), (world, frame, episode)
            assert force[frame, j] == sequence[episode]['force_n'][0], (world, frame, 'branch')
            assert age[frame, j] == expected_age, (world, frame, 'age')
            assert previous[frame, j] == (0. if expected_age == 0 else action[frame-1, j, 0])
            expected_age += 1
            if terminal[frame, j]:
                episode += 1
                expected_age = 0
        # The training loop also performs a reset after a terminal on its final frame.
        assert len(sequence) == episode+1, (world, len(sequence), episode)
        consumed += len(sequence)
    assert consumed == len(records)
    return dict(reset_records=consumed, terminal_events=int(terminal.sum()),
                initial_assignments=len(WORLDS), frames=2048, samples=24576)


def check_first_last(batch, original):
    fields = ('reward', 'terminal', 'failed', 'value', 'next_value', 'mean', 'action', 'logp',
              'advantage', 'normalized_advantage', 'returns', 'post_mean', 'post_value')
    for key in fields:
        assert np.array_equal(batch[key].reshape(original[key].shape), original[key]), key
    assert np.array_equal(batch['previous_latent'], original['x'][:, :, -1])
    assert float(original['sigma']) == SIGMA
    return len(fields)+1


@torch.no_grad()
def predictions(actor, critic, features):
    flat = torch.from_numpy(features).reshape(-1, 427)
    shape = features.shape[:-1]
    return actor(flat).numpy().reshape(shape), critic(flat).numpy().reshape(shape)


def main():
    torch.set_num_threads(1)
    inputs, done = read(OUT/'inputs.json'), read(OUT/'completed.json')
    for path, sha in inputs.items():
        assert digest(Path(path)) == sha, path
    for path, sha in done['outputs'].items():
        assert digest(OUT/path) == sha, path
    assert done['exact_historical_parity'] is True and done['source_unchanged'] is True
    assert done['training_transitions_replayed'] == 49152
    assert not list(OUT.rglob('*.pt')), 'Replay must not produce new policy checkpoints'
    assert read(BASE/'audit.json')['passed'] is True
    old_manifest = read(BASE/'completed.json')['outputs']
    probe = arrays(OUT/'probes.npz')
    features = probe['features']
    assert features.shape == (2, 2, 30, 12, 427)
    assert np.array_equal(probe['worlds'], WORLDS)
    assert np.isfinite(features).all() and not features[..., -2:].any()
    for i, onset in enumerate((44, 74)):
        for j, branch in enumerate(('quiet', 'push')):
            source = arrays(EXPLORATION/f'{onset}-{branch}-zero'/'trace.npz')
            assert np.array_equal(features[i, j], source['features64'][:30])
    errors, reports = {}, {}
    for seed in (921009, 931009):
        folder, original = OUT/f'fit-{seed}', BASE/f'fit-{seed}'
        for filename in ('initial.pt', 'final.pt', 'critic-final.pt', 'collection-1.npz',
                         'collection-32.npz', 'resets.json', 'history.json'):
            path = original/filename
            assert digest(path) == old_manifest[str(path.relative_to(BASE))], path
        history, records = read(folder/'history.json'), read(folder/'resets.json')
        assert history == read(original/'history.json') and len(history) == 32
        assert records == read(original/'resets.json')
        parity = read(folder/'parity.json')
        required = ('exact_initial', 'exact_final_actor', 'exact_final_critic', 'exact_history',
                    'exact_resets', 'exact_accounting', 'exact_first_last_batches')
        assert parity['passed'] is True and all(parity[key] is True for key in required)
        report = read(folder/'report.json')
        assert report['seed'] == seed and report['transitions'] == 24576
        assert report['train_world_ids'] == WORLDS.tolist() and report['train_poses'] == list(range(6))
        assert report['sigma'] == SIGMA and report['updates'] == 288
        assert report['max_independent_kl_error'] < 2e-7
        initial, final, critic = AnkleHead().eval(), AnkleHead().eval(), AnkleHead(1).eval()
        for model, filename in ((initial, 'initial.pt'), (final, 'final.pt'), (critic, 'critic-final.pt')):
            model.load_state_dict(torch.load(original/filename, weights_only=True, map_location='cpu'))
            model.requires_grad_(False)
        assert not initial.net[-1].weight.any() and not initial.net[-1].bias.any()
        files = list(folder.glob('diagnostic-*.npz'))
        assert len(files) == 32
        batches = [arrays(folder/f'diagnostic-{update}.npz') for update in range(1, 33)]
        first_original = arrays(original/'collection-1.npz')
        matched_fields = check_first_last(batches[0], first_original)
        check_first_last(batches[-1], arrays(original/'collection-32.npz'))
        for i, batch in enumerate(batches):
            for key, value in batch.items():
                assert np.isfinite(value).all(), (seed, i+1, key)
            assert batch['terminal'].dtype == batch['failed'].dtype == np.bool_
            assert not np.any(batch['failed'] & ~batch['terminal'])
            logp = (-.5*((batch['action'][..., 0].astype(float)-batch['mean'][..., 0])/SIGMA)**2
                    -np.log(SIGMA)-.5*np.log(2*np.pi))
            close(batch['logp'], logp, 'normal_log_probability', errors, atol=3e-6, rtol=0.)
            advantage, reward_part, value_part = independent_credit(batch)
            close(batch['advantage'], advantage, 'independent_gae', errors)
            close(batch['reward_gae'], reward_part, 'reward_gae', errors)
            close(batch['value_gae'], value_part, 'value_gae', errors)
            close(batch['reward_gae'].astype(float)+batch['value_gae'], batch['advantage'],
                  'gae_decomposition', errors)
            assert np.array_equal(batch['returns'], batch['advantage']+batch['value'])
            # GAE recurrence was independently checked above. Isolate normalization
            # from its permitted float32 recurrence rounding error.
            recorded = batch['advantage'].astype(float)
            normalized = (recorded.ravel()-recorded.mean())/max(recorded.std(ddof=1), 1e-6)
            close(batch['normalized_advantage'].ravel(), normalized, 'normalized_advantage', errors,
                  atol=3e-6, rtol=2e-6)
            check_gradients(batch, errors, initial if i == 0 else None, first_original if i == 0 else None)
            assert history[i]['failures'] == int(batch['failed'].sum())
            assert history[i]['completed'] == int(batch['terminal'].sum())
            assert history[i]['update'] == i+1 and history[i]['transitions'] == (i+1)*768
            assert history[i]['actor_updates'] == 9 and history[i]['rolled_back'] == 0
            assert len(history[i]['accepted_kl']) == 9 and max(history[i]['accepted_kl']) < .01
            for key in ('probe_mean_before', 'probe_mean_after', 'probe_value_before', 'probe_value_after'):
                assert batch[key].shape == features.shape[:-1]
            if i:
                assert np.array_equal(batch['probe_mean_before'], batches[i-1]['probe_mean_after'])
                assert np.array_equal(batch['probe_value_before'], batches[i-1]['probe_value_after'])
        with torch.no_grad():
            initial_mean = initial(torch.from_numpy(features).reshape(-1, 427)).numpy().reshape(features.shape[:-1])
        assert np.array_equal(batches[0]['probe_mean_before'], initial_mean)
        assert not batches[0]['probe_value_before'].any()
        final_mean, final_value = predictions(final, critic, features)
        assert np.array_equal(batches[-1]['probe_mean_after'], final_mean)
        assert np.array_equal(batches[-1]['probe_value_after'], final_value)
        reports[str(seed)] = dict(**check_labels(batches, records), updates=32,
                                  exact_first_last_retained_fields=matched_fields,
                                  exact_probe_boundary_continuity=True,
                                  exact_final_probe_predictions=True)
    audit = dict(passed=True, seeds=reports, collections=64, transitions=49152,
                 input_hashes=len(inputs), output_hashes=len(done['outputs']),
                 exact_historical_records=True, exact_fixed_probe_sources=True,
                 independent_last_layer_bias_all_updates=True,
                 independent_full_last_layer_initial_updates=True,
                 stored_middle_last_layer_analytic_and_partition_checks=True,
                 max_absolute_errors=errors, new_policy_checkpoints=0,
                 limitations='Historical middle batches and intermediate actor weights were not retained. '
                 'Middle full-network gradients are checked through their saved analytic final-layer and '
                 'partition identities, not independently replayed from weights. Value targets are '
                 'bootstrapped quantities, not independent critic calibration.')
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
