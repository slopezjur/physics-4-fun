"""Independent saved-data and scratch-optimizer audit of local PPO objectives."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
SOURCE = Path('logs/mimickit-training/recovery-local-action-credit-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
SIGMA = .08717797887081347
NOISE_SEEDS = [951011, 951012, 951013, 951014]
TRAIN = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, label, errors, atol=2e-6, rtol=2e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    error = float(np.abs(actual.astype(float)-expected.astype(float)).max(initial=0.))
    errors[label] = max(errors.get(label, 0.), error)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (label, error)


def flattened_grad(objective, actor, *, retain_graph=False):
    gradients = torch.autograd.grad(objective, tuple(actor.parameters()), retain_graph=retain_graph)
    return torch.cat([value.reshape(-1) for value in gradients])


def parameter_vector(actor):
    return torch.cat([value.detach().reshape(-1) for value in actor.parameters()])


def nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        return isinstance(expected, torch.Tensor) and torch.equal(actual, expected)
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(nested_equal(actual[key], expected[key]) for key in actual)
    if isinstance(actual, (tuple, list)):
        return len(actual) == len(expected) and all(nested_equal(x, y) for x, y in zip(actual, expected))
    return actual == expected


def source_rows(seed):
    rows, metadata = [], []
    for onset in (44, 74):
        for branch in ('quiet', 'push'):
            worlds = list(range(6)) if branch == 'quiet' else TRAIN
            for noise in NOISE_SEEDS:
                sources = {}
                for sign, tag in ((-1, 'negative'), (1, 'positive')):
                    folder = SOURCE/f'{onset}-{branch}-{seed}-{noise}-{tag}'
                    sources[sign] = (folder, arrays(folder/'trace.npz'), read(folder/'report.json'))
                for index, world in enumerate(worlds):
                    for sign in (-1, 1):
                        folder, trace, report = sources[sign]
                        case = report['cases'][index]
                        assert case['world'] == world and case['pose'] == world % 8
                        rows.append(dict(x=trace['features64'][0, index], action=trace['latent'][0, index],
                                         old_mean=trace['mean'][0, index], gae=trace['advantage64'][0, index],
                                         value=trace['value'][0, index], return_full=case['reward_return'],
                                         branch=0 if branch == 'quiet' else 1 if world < 8 else 2,
                                         onset=onset, pose=world % 8, noise=noise, sign=sign))
                        metadata.append(dict(folder=folder.name, world=world, index=index))
    result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert result['x'].shape == (288, 427) and result['action'].shape == result['old_mean'].shape == (288, 1)
    return result, metadata


def check_dataset(seed, dataset, metadata, actor, critic, errors):
    original, source_metadata = source_rows(seed)
    assert metadata['rows'] == source_metadata
    for key, value in original.items():
        assert np.array_equal(dataset[key], value), (seed, key)
    assert np.array_equal(dataset['pair'], np.repeat(np.arange(144), 2))
    for key in ('x', 'old_mean', 'value', 'branch', 'onset', 'pose', 'noise', 'probe'):
        assert np.array_equal(dataset[key][::2], dataset[key][1::2]), key
    assert np.array_equal(dataset['sign'], np.tile([-1, 1], 144))
    close(dataset['action']-dataset['old_mean'], dataset['sign'][:, None]*SIGMA,
          'opposite_innovations', errors, atol=2e-8, rtol=0.)
    advantage = dataset['gae'].astype(float)
    denominator = max(advantage.std(ddof=1), 1e-6)
    close(dataset['advantage_std'], denominator, 'advantage_std', errors, atol=2e-6, rtol=2e-6)
    close(dataset['ordinary'], (advantage-advantage.mean())/denominator,
          'ordinary_normalization', errors, atol=2e-6, rtol=2e-6)
    centered = (advantage-advantage.reshape(144, 2).mean(1).repeat(2))/denominator
    close(dataset['centered'], centered, 'pair_centering', errors, atol=2e-6, rtol=2e-6)
    probes, lookup = [], {}
    for i in range(288):
        key = tuple(int(dataset[field][i]) for field in ('onset', 'branch', 'pose'))
        if key not in lookup:
            lookup[key] = len(probes)
            probes.append(dict(zip(('onset', 'branch', 'pose'), key)))
        assert dataset['probe'][i] == lookup[key]
    assert len(probes) == 36 and metadata['probes'] == probes
    for probe, description in enumerate(probes):
        indices = np.flatnonzero(dataset['probe'] == probe)
        assert len(indices) == 8
        assert np.array_equal(dataset['x'][indices], np.repeat(dataset['probe_x'][probe:probe+1], 8, axis=0))
        assert np.array_equal(dataset['probe_old_mean'][probe], dataset['old_mean'][indices[0]])
        for field in ('branch', 'onset', 'pose'):
            assert dataset['probe_'+field][probe] == description[field]
        quiet_index = lookup[(description['onset'], 0, description['pose'])]
        assert dataset['probe_quiet_index'][probe] == quiet_index
        positive = indices[dataset['sign'][indices] == 1]
        negative = indices[dataset['sign'][indices] == -1]
        difference = dataset['return_full'][positive].mean()-dataset['return_full'][negative].mean()
        assert dataset['probe_return_delta'][probe] == difference
        assert dataset['probe_preferred'][probe] == (np.sign(difference) if abs(difference) > 1e-3 else 0)
    assert (dataset['probe_preferred'][dataset['probe_branch'] == 1] == -1).all()
    assert (dataset['probe_preferred'][dataset['probe_branch'] == 2] == 1).all()
    generator = torch.Generator().manual_seed(961011)
    assert dataset['permutations'].shape == (3, 288)
    for permutation in dataset['permutations']:
        assert np.array_equal(permutation, torch.randperm(288, generator=generator).numpy())
    with torch.no_grad():
        x = torch.from_numpy(dataset['x'])
        close(actor(x).numpy(), dataset['old_mean'], 'frozen_actor_mean', errors, atol=2e-7, rtol=0.)
        close(critic(x).numpy(), dataset['value'], 'frozen_critic_value', errors, atol=4e-5, rtol=0.)


def objective(actor, x, actions, old_logp, advantage, ids):
    ratio = (Normal(actor(x[ids]), SIGMA).log_prob(actions[ids]).sum(-1)-old_logp[ids]).exp()
    return torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids]).mean(), ratio


def objective_gradient(actor, x, actions, old_logp, advantage, ids):
    score, _ = objective(actor, x, actions, old_logp, advantage, ids)
    return flattened_grad(score, actor).detach()


def comparison(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    norm, reference = np.linalg.norm(actual), np.linalg.norm(expected)
    return dict(norm=float(norm), reference_norm=float(reference), norm_ratio=float(norm/max(reference, 1e-30)),
                cosine=float(np.dot(actual, expected)/max(norm*reference, 1e-30)),
                relative_difference=float(np.linalg.norm(actual-expected)/max(reference, 1e-30)),
                max_absolute_difference=float(np.abs(actual-expected).max()))


def compare_report(recorded, expected, label):
    assert recorded.keys() == expected.keys(), label
    for key, value in expected.items():
        actual = recorded[key]
        if isinstance(value, dict):
            compare_report(actual, value, label+'.'+key)
        elif isinstance(value, (int, bool)):
            assert actual == value, (label, key, actual, value)
        else:
            assert np.isclose(actual, value, atol=2e-9, rtol=2e-7), (label, key, actual, value)


def split_counts(indices):
    _, counts = np.unique(np.asarray(indices)//2, return_counts=True)
    return dict(complete_pairs=int(np.count_nonzero(counts == 2)),
                unmatched_rows=int(np.count_nonzero(counts == 1)), samples=len(indices))


def movement(mean, dataset):
    delta = np.asarray(mean, dtype=float).ravel()-dataset['probe_old_mean'].astype(float).ravel()
    response = delta-delta[dataset['probe_quiet_index']]
    quiet, direction = dataset['probe_branch'] == 0, dataset['probe_preferred']
    rms = lambda value: float(np.sqrt(np.mean(value**2)))
    def ranks(value, mask):
        directed = value[mask]*direction[mask]
        return dict(helpful=int(np.count_nonzero(directed > 1e-7)), harmful=int(np.count_nonzero(directed < -1e-7)),
                    tied=int(np.count_nonzero(np.abs(directed) <= 1e-7)), signed_mean=float(directed.mean()), rms=rms(value[mask]))
    return dict(quiet_mean=float(delta[quiet].mean()), quiet_rms=rms(delta[quiet]), all_rms=rms(delta),
                constant_energy_fraction=float(delta.mean()**2/max(np.mean(delta**2), 1e-30)),
                raw_push=ranks(delta, ~quiet), response_push=ranks(response, ~quiet),
                by_branch={name: dict(mean_change=float(delta[dataset['probe_branch'] == branch].mean()),
                                     response_mean=float(response[dataset['probe_branch'] == branch].mean()),
                                     raw=ranks(delta, dataset['probe_branch'] == branch),
                                     response=ranks(response, dataset['probe_branch'] == branch))
                           for branch, name in ((1, '+X'), (2, '-X'))})


def check_initial(actor, dataset, saved, report, errors):
    x, action = torch.from_numpy(dataset['x']), torch.from_numpy(dataset['action'])
    old_logp = Normal(torch.from_numpy(dataset['old_mean']), SIGMA).log_prob(action).sum(-1)
    ids = torch.arange(288)
    gradients = []
    for name in ('ordinary', 'centered'):
        av = torch.from_numpy(dataset[name])
        groups = []
        for branch in (None, 0, 1, 2):
            masked = av if branch is None else av*torch.from_numpy((dataset['branch'] == branch).astype(np.float32))
            groups.append(objective_gradient(actor, x, action, old_logp, masked, ids).numpy())
        gradients.append(np.stack(groups))
    gradients = np.stack(gradients)
    close(saved['gradient'], gradients, 'initial_gradient', errors)
    close(gradients[0], gradients[1], 'pair_centering_gradient_invariant', errors)
    relative = np.linalg.norm((gradients[0, 0]-gradients[1, 0]).astype(float))/np.linalg.norm(gradients[1, 0].astype(float))
    assert relative < 2e-4
    close(gradients[:, 1:].sum(1), gradients[:, 0], 'branch_gradient_sum', errors)
    jacobian = []
    for row in torch.from_numpy(dataset['probe_x']):
        jacobian.append(flattened_grad(actor(row[None]).sum(), actor).detach().numpy())
    jacobian = np.asarray(jacobian, dtype=float)
    gram = jacobian@jacobian.T
    close(saved['jacobian_gram'], gram, 'jacobian_gram', errors, atol=1e-10, rtol=1e-8)
    response = np.stack([np.stack([jacobian@g.astype(float) for g in group]) for group in gradients])
    close(saved['response'], response, 'branch_state_response', errors, atol=1e-10, rtol=1e-8)
    with torch.no_grad():
        prediction = actor(x)
        ratio = (Normal(prediction, SIGMA).log_prob(action).sum(-1)-old_logp).exp().numpy()
        score = ((action-prediction)/SIGMA**2).numpy().ravel()
    assert ((ratio > .8) & (ratio < 1.2)).all()
    for number, name in enumerate(('ordinary', 'centered')):
        multiplier = ratio.astype(float)*score.astype(float)*dataset[name].astype(float)/288
        analytic = (jacobian[dataset['probe']]*multiplier[:, None]).sum(0)
        close(gradients[number, 0], analytic, 'independent_score_jacobian_gradient', errors)
    minibatches = [torch.from_numpy(part) for order in dataset['permutations'] for part in (order[:256], order[256:])]
    mini = np.stack([np.stack([objective_gradient(actor, x, action, old_logp,
                          torch.from_numpy(dataset[name]), batch).numpy() for batch in minibatches])
                     for name in ('ordinary', 'centered')])
    close(saved['initial_minibatch_gradient'], mini, 'initial_minibatch_gradient', errors)
    for j, batch in enumerate(minibatches):
        indices = batch.numpy()
        counts = np.bincount(indices//2, minlength=144)
        unmatched = counts[indices//2] == 1
        baseline = torch.from_numpy(dataset['ordinary']-dataset['centered'])
        mask = torch.zeros(288)
        mask[indices[unmatched]] = 1
        contribution = objective_gradient(actor, x, action, old_logp, baseline*mask, ids)*288/len(indices)
        error = float(np.abs((mini[0, j]-mini[1, j])-contribution.numpy()).max())
        assert error < 2e-6
        record = report['batches'][j]
        assert record['epoch'] == j//2 and record['batch'] == j % 2
        for key, value in split_counts(indices).items():
            assert record[key] == value
        compare_report(record['ordinary'], comparison(mini[0, j], gradients[0, 0]), 'initial ordinary')
        compare_report(record['centered'], comparison(mini[1, j], gradients[1, 0]), 'initial centered')
        assert abs(record['unmatched_baseline_max_error']-error) < 2e-6
    for number in range(2):
        for epoch in range(3):
            weighted = (256*mini[number, 2*epoch]+32*mini[number, 2*epoch+1])/288
            close(weighted, gradients[number, 0], 'sample_weighted_minibatch_identity', errors)
    compare_report(report['invariant'], comparison(gradients[0, 0], gradients[1, 0]), 'initial invariant')
    norms = np.linalg.norm(gradients[0].astype(float), axis=1)
    cosine = gradients[0].astype(float)@gradients[0].astype(float).T/np.maximum(norms[:, None]*norms, 1e-30)
    close(report['branch_gradient_norm'], norms, 'branch_norm_summary', errors, atol=1e-10, rtol=1e-8)
    close(report['branch_gradient_cosine'], cosine, 'branch_cosine_summary', errors, atol=1e-10, rtol=1e-8)
    assert np.isclose(report['branch_retention'], norms[0]/norms[1:].sum(), atol=1e-10, rtol=1e-8)
    jacobian_norm = np.sqrt(np.diag(gram))
    cosine = gram/np.maximum(jacobian_norm[:, None]*jacobian_norm, 1e-30)
    matched = [cosine[i, dataset['probe_quiet_index'][i]] for i in range(36) if dataset['probe_branch'][i] != 0]
    close(report['matched_push_quiet_jacobian_cosines'], matched, 'matched_state_jacobians', errors, atol=1e-10, rtol=1e-8)
    assert report['advantage_sign_changes'] == int(np.count_nonzero((dataset['ordinary'] > 0) != (dataset['centered'] > 0)))
    return float(relative)


def model_hash(actor):
    return hashlib.sha256(parameter_vector(actor).numpy().tobytes()).hexdigest()


def replay_arm(initial, dataset, name, saved, report):
    actor = deepcopy(initial)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8)
    assert not optimizer.state
    x, action, probes = (torch.from_numpy(dataset[key]) for key in ('x', 'action', 'probe_x'))
    old_mean, av = torch.from_numpy(dataset['old_mean']), torch.from_numpy(dataset[name])
    old_logp = Normal(old_mean, SIGMA).log_prob(action).sum(-1)
    all_ids = torch.arange(288)
    attempts, accepted, rejected = 0, 0, 0
    assert report['objective'] == name
    assert report['initial_parameter_sha256'] == model_hash(initial)
    with torch.no_grad():
        assert np.array_equal(saved['means'][0], actor(probes).numpy())
    for epoch, order in enumerate(dataset['permutations']):
        for batch, selection in enumerate((order[:256], order[256:])):
            ids = torch.from_numpy(selection)
            full = objective_gradient(actor, x, action, old_logp, av, all_ids)
            mini = objective_gradient(actor, x, action, old_logp, av, ids)
            before = parameter_vector(actor).clone()
            old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
            with torch.no_grad():
                _, ratios_all = objective(actor, x, action, old_logp, av, all_ids)
            score, ratios = objective(actor, x, action, old_logp, av, ids)
            optimizer.zero_grad(set_to_none=True)
            (-score).backward()
            clip_norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
            optimizer.step()
            with torch.no_grad():
                kl = float(((actor(x)-old_mean).square().sum(-1)/(2*SIGMA**2)).mean())
                candidate = actor(probes).numpy().copy()
            delta = parameter_vector(actor)-before
            accept = bool(np.isfinite(kl) and kl <= .01)
            if not accept:
                actor.load_state_dict(old_actor)
                optimizer.load_state_dict(old_optimizer)
                assert nested_equal(actor.state_dict(), old_actor)
                assert nested_equal(optimizer.state_dict(), old_optimizer)
                rejected += 1
            else:
                accepted += 1
            with torch.no_grad():
                mean = actor(probes).numpy().copy()
            step = report['records'][attempts]
            assert step['epoch'] == epoch and step['batch'] == batch
            assert step['ids'] == selection.tolist()
            assert step['accepted'] is accept
            assert step['loss'] == float(-score.detach())
            assert step['gradient_norm'] == float(mini.norm()) and step['clip_norm'] == clip_norm
            assert step['kl'] == (kl if np.isfinite(kl) else None)
            compare_report(step['gradient_compare'], comparison(mini.numpy(), full.numpy()), 'current gradient comparison')
            assert step['split_counts'] == split_counts(selection)
            clipped = float(((ratios < .8) | (ratios > 1.2)).float().mean())
            surrogate_clipped = float(((ratios*av[ids]) > ratios.clamp(.8, 1.2)*av[ids]).float().mean())
            assert step['clipped_ratio_fraction'] == clipped
            assert step['surrogate_clipped_fraction'] == surrogate_clipped
            for key, actual in (('candidate_means', candidate), ('ratios', ratios_all.numpy()),
                                ('minibatch_gradient', mini.numpy()), ('full_gradient', full.numpy()),
                                ('parameter_delta', delta.numpy())):
                assert np.array_equal(saved[key][attempts], actual, equal_nan=True), (name, attempts, key)
            assert np.array_equal(saved['means'][attempts+1], mean)
            if np.isfinite(candidate).all():
                compare_report(step['candidate_metrics'], movement(candidate, dataset), 'candidate movement')
            else:
                assert step['candidate_metrics'] is None
            compare_report(step['metrics'], movement(mean, dataset), 'accepted movement')
            if not accept:
                assert np.array_equal(saved['means'][attempts+1], saved['means'][attempts])
            attempts += 1
            if not accept:
                break
        if rejected:
            break
    assert 1 <= attempts <= 6 and len(report['records']) == attempts
    assert report['accepted'] == accepted and report['rejected'] == rejected and report['attempted'] == attempts
    for key in ('candidate_means', 'ratios', 'minibatch_gradient', 'full_gradient', 'parameter_delta'):
        assert len(saved[key]) == attempts
    assert len(saved['means']) == attempts+1
    assert report['final_parameter_sha256'] == model_hash(actor)
    compare_report(report['final_summary'], movement(saved['means'][-1], dataset), 'final movement')
    return dict(attempted=attempts, accepted=accepted, rejected=rejected,
                exact_saved_arrays=True, exact_optimizer_replay=True,
                rollback_replayed=bool(rejected))


def provenance(inputs, completed):
    for path, sha in inputs.items():
        assert digest(Path(path)) == sha, path
    for path, sha in completed['outputs'].items():
        assert digest(OUT/path) == sha, path
    assert read(SOURCE/'audit.json')['passed'] is True
    assert read(LEARNING/'audit.json')['passed'] is True
    historical = read(SOURCE/'inputs.json')
    shared = set(historical) & set(inputs)
    assert shared and all(historical[path] == inputs[path] for path in shared)
    for root in (SOURCE, LEARNING):
        manifest = read(root/'completed.json')['outputs']
        for text in inputs:
            path = Path(text)
            if path.is_relative_to(root) and str(path.relative_to(root)) in manifest:
                assert digest(path) == manifest[str(path.relative_to(root))], path
    return len(shared)


def main():
    torch.set_num_threads(1)
    inputs, completed, contract, results = (read(OUT/name) for name in
                                           ('inputs.json', 'completed.json', 'contract.json', 'results.json'))
    shared = provenance(inputs, completed)
    assert completed['inputs_unchanged'] is True
    for key in ('physical_rollouts', 'critic_updates', 'new_checkpoints'):
        assert completed[key] == contract[key] == 0
    assert completed['samples'] == 576 and completed['pairs'] == 288 and completed['actor_starts'] == 72
    assert not list(OUT.rglob('*.pt')) and not list(OUT.rglob('trace.npz'))
    assert contract['actors'] == [921009, 931009] and contract['objectives'] == ['ordinary', 'centered']
    assert contract['samples_per_actor'] == 288 and contract['pairs_per_actor'] == 144
    assert contract['unique_starts_per_actor'] == 36 and contract['noise_repeats'] == 4
    assert contract['sigma'] == SIGMA and contract['permutation_seed'] == 961011
    assert contract['epochs'] == 3 and contract['minibatch_sizes'] == [256, 32]
    assert contract['learning_rate'] == 1e-4 and contract['adam_betas'] == [.9, .999]
    assert contract['adam_epsilon'] == 1e-8 and contract['gradient_clip'] == .5
    assert contract['ppo_ratio_clip'] == [.8, 1.2] and contract['kl_limit'] == .01
    assert contract['standard_deviation_correction'] == 1
    assert contract['invariant_atol'] == 2e-6 and contract['invariant_rtol'] == 2e-4
    assert contract['invariant_relative_norm_tolerance'] == 2e-4
    assert contract['movement_sign_tolerance'] == 1e-7 and contract['max_attempted_updates'] == 24
    assert contract['shared_historical_inputs'] == shared
    errors, audited, attempted = {}, {}, 0
    for seed in (921009, 931009):
        folder = OUT/str(seed)
        dataset, metadata = arrays(folder/'dataset.npz'), read(folder/'metadata.json')
        actor, critic = AnkleHead().eval(), AnkleHead(1).eval()
        for model, filename in ((actor, 'final.pt'), (critic, 'critic-final.pt')):
            model.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/filename, weights_only=True, map_location='cpu'))
        critic.requires_grad_(False)
        initial_actor, initial_critic = deepcopy(actor.state_dict()), deepcopy(critic.state_dict())
        check_dataset(seed, dataset, metadata, actor, critic, errors)
        initial = read(folder/'initial.json')
        assert results[str(seed)]['initial'] == initial
        relative = check_initial(actor, dataset, arrays(folder/'initial.npz'), initial, errors)
        arms = {}
        for name in ('ordinary', 'centered'):
            evidence, report = arrays(folder/(name+'.npz')), read(folder/(name+'.json'))
            arms[name] = replay_arm(actor, dataset, name, evidence, report)
            attempted += arms[name]['attempted']
            expected = {key: report[key] for key in ('accepted', 'rejected', 'attempted', 'final_summary')}
            assert results[str(seed)]['arms'][name] == expected
        assert nested_equal(actor.state_dict(), initial_actor)
        assert nested_equal(critic.state_dict(), initial_critic)
        audited[str(seed)] = dict(samples=288, pairs=144, unique_starts=36,
                                 initial_gradient_relative_difference=relative, arms=arms)
    assert attempted == completed['attempted_updates'] <= 24
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    audit = dict(passed=True, actors=audited, samples=576, pairs=288, unique_actor_starts=72,
                 input_hashes=len(inputs), output_hashes=len(completed['outputs']), historical_input_hashes=shared,
                 exact_source_rows=True, fixed_permutations_verified=True, shared_normalization_denominator=True,
                 initial_gradient_invariant_verified=True, independent_score_jacobian_gradient=True,
                 initial_unmatched_pair_baseline_verified=True, full_and_minibatch_gradients_verified=True,
                 scratch_replay_attempted_updates=attempted, physical_rollouts=0, critic_updates=0,
                 new_checkpoints=0, max_absolute_errors=errors,
                 scope='Fresh scratch optimizers reproduce saved offline updates. This is not historical '
                 'optimizer replay or a physical recovery evaluation. Optimizer rollback is reproduced '
                 'from scratch and checked against restored state; no intermediate optimizer artifact is retained.')
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
