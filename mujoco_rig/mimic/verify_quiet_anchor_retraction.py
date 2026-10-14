"""Independent audit of two fixed rejected-update anchor corrections."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_constrained_recovery import actor_hash, optimizer_hash, projection_verify

OUT = Path('logs/mimickit-training/recovery-quiet-anchor-retraction-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
PROBES = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
ACTORS = (921009, 931009)
SIGMA = .08717797887081347
CUTOFF = 1e-6
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def linear_verify(jacobian, residual, correction, singular_values):
    """Verify the SVD correction using the independent quiet Gram eigensystem."""
    quiet, residual, supplied, singular_values = (
        np.asarray(value, dtype=np.float64)
        for value in (jacobian, residual, correction, singular_values))
    assert quiet.ndim == 2 and quiet.shape[0] == 12
    assert residual.shape == singular_values.shape == (12,) and supplied.shape == (quiet.shape[1],)
    assert all(np.isfinite(value).all() for value in (quiet, residual, supplied, singular_values))
    gram = quiet@quiet.T
    values, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, eigenvectors = values[order], eigenvectors[:, order]
    floor = 64*np.finfo(float).eps*max(1., float(values[0]))
    assert values[-1] >= -floor
    singular = np.sqrt(np.maximum(values, 0.))
    assert singular[0] > 0 and np.allclose(singular, singular_values, atol=1e-9, rtol=2e-5)
    squared_cutoff = (CUTOFF*singular[0])**2
    assert np.min(np.abs(values-squared_cutoff)) > floor, 'ambiguous Gram rank'
    keep = values > squared_cutoff
    assert int(keep.sum()) == int(np.count_nonzero(singular_values > CUTOFF*singular_values[0]))
    basis = eigenvectors[:, keep]
    coordinates = basis.T@residual
    independent = -quiet.T@(basis@(coordinates/values[keep]))
    maximum_error = float(np.abs(supplied-independent).max())
    relative_error = float(np.linalg.norm(supplied-independent)/max(np.linalg.norm(independent), 1e-30))
    assert maximum_error <= 2e-8 and relative_error <= 2e-5, (maximum_error, relative_error)
    expected_residual = residual-basis@coordinates
    actual_residual = residual+quiet@supplied
    residual_error = float(np.linalg.norm(actual_residual-expected_residual))
    residual_tolerance = 1e-10+2e-7*np.linalg.norm(residual)
    assert residual_error <= residual_tolerance
    rowspace_projection = quiet.T@(basis@((basis.T@(quiet@supplied))/values[keep]))
    rowspace_error = float(np.linalg.norm(supplied-rowspace_projection))
    assert rowspace_error <= 2e-8+2e-5*np.linalg.norm(supplied)
    expected_norm_squared = float(np.sum(coordinates**2/values[keep]))
    norm_error = abs(float(supplied@supplied)-expected_norm_squared)
    assert norm_error <= 2e-10+2e-7*expected_norm_squared
    return dict(passed=True, method='float64 quiet Gram eigendecomposition', threshold=CUTOFF,
        rank=int(keep.sum()), eigenvalues=values.tolist(), eigenvalue_numerical_floor=floor,
        spectrum_max_absolute_error=float(np.abs(singular-singular_values).max()),
        correction_max_absolute_error=maximum_error, correction_relative_l2_error=relative_error,
        linear_residual_l2=float(np.linalg.norm(actual_residual)),
        expected_truncated_residual_l2=float(np.linalg.norm(expected_residual)),
        linear_residual_error=residual_error, rowspace_error=rowspace_error,
        minimum_norm_squared_error=norm_error)


def quiet_jacobian(actor, features):
    rows = []
    for point in features:
        gradient = torch.autograd.grad(actor(point[None]).sum(), tuple(actor.parameters()))
        rows.append(torch.cat([part.flatten() for part in gradient]).detach().double().numpy())
    return np.stack(rows)


def tensors(batch):
    return dict(x=torch.from_numpy(batch['x']).flatten(0, 1),
        action=torch.from_numpy(batch['action']).flatten(0, 1),
        old_mean=torch.from_numpy(batch['mean']).flatten(0, 1),
        old_logp=torch.from_numpy(batch['logp']).flatten(),
        advantage=torch.from_numpy(batch['normalized_advantage']),
        weight=torch.from_numpy(batch['actor_weight']).flatten())


def surrogate(actor, batch, ids):
    ratio = (Normal(actor(batch['x'][ids]), SIGMA).log_prob(batch['action'][ids]).sum(-1)
             -batch['old_logp'][ids]).exp()
    advantage = batch['advantage'][ids]
    return (batch['weight'][ids]*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()


def replay_candidate(seed, quiet_features):
    """Recreate actor/Adam history; saved batch and indices require no critic or RNG draws."""
    folder = LEARNING/f'fit-{seed}'
    batch = tensors(arrays(folder/'collection-1.npz'))
    objective = arrays(folder/'objective-1.npz')
    records = read(folder/'projection-1.json')
    assert len(records) == 9 and all(row['accepted'] for row in records[:8])
    assert not records[8]['accepted'] and records[8]['reasons'] == ['anchor']
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    before_ninth = None
    for index, record in enumerate(records):
        assert record['minibatch_index'] == index
        assert actor_hash(actor) == record['actor_pre']
        assert optimizer_hash(optimizer) == record['optimizer_pre']
        ids = torch.from_numpy(objective['minibatch_indices'][index])
        loss = -surrogate(actor, batch, ids)
        assert float(loss.detach()) == record['actor_loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
        assert norm == record['gradient_norm']
        before = parameters_to_vector(actor.parameters()).detach().clone()
        if index == 8:
            before_ninth = deepcopy(actor)
        quiet = quiet_jacobian(actor, quiet_features)
        _, singular, right = np.linalg.svd(quiet, full_matrices=False)
        basis = right[singular > CUTOFF*singular[0]]
        optimizer.step()
        assert actor_hash(actor) == record['actor_tentative']
        assert optimizer_hash(optimizer) == record['optimizer_tentative']
        tentative = parameters_to_vector(actor.parameters()).detach().double().numpy()-before.double().numpy()
        projected = tentative-basis.T@(basis@tentative)
        projection_verify(quiet, tentative, projected, singular)
        vector_to_parameters(before+torch.from_numpy(projected.astype(np.float32)), actor.parameters())
        assert actor_hash(actor) == record['actor_candidate']
        with torch.no_grad():
            assert np.array_equal(actor(quiet_features).numpy().ravel(), record['anchor_attempted'])
        if index < 8:
            assert actor_hash(actor) == record['actor_post']
            assert optimizer_hash(optimizer) == record['optimizer_post']
    assert before_ninth is not None
    assert records[8]['actor_post'] == records[8]['actor_pre']
    assert records[8]['optimizer_post'] == records[8]['optimizer_pre']
    with torch.no_grad():
        assert np.array_equal(before_ninth(batch['x']).numpy(), arrays(folder/'collection-1.npz')['post_mean'])
    return before_ninth, actor, optimizer, batch, objective['minibatch_indices'][8], records[8]


def compare(actual, expected, label, atol=1e-10, rtol=2e-7):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key, atol, rtol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label+'.'+str(index), atol, rtol)
    elif isinstance(expected, (int, bool, str)) or expected is None:
        assert actual == expected, (label, actual, expected)
    else:
        assert actual is not None and np.isclose(actual, expected, atol=atol, rtol=rtol), (label, actual, expected)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def analytical_metrics(mean, anchor, batch, minibatch_ids, target, minibatch_mean=None):
    """Evaluate probability/objective arithmetic in float64 without distribution helpers."""
    values = {key: tensor.detach().numpy().astype(float) for key, tensor in batch.items()}
    mean, anchor = np.asarray(mean, dtype=float), np.asarray(anchor, dtype=float)
    old = values['old_mean']
    logp = (-.5*((values['action']-mean)/SIGMA)**2-np.log(SIGMA)-.5*np.log(2*np.pi)).sum(-1)
    ratio = np.exp(logp-values['old_logp'])
    advantage, weights = values['advantage'], values['weight']
    objective = weights*np.minimum(ratio*advantage, np.clip(ratio, .8, 1.2)*advantage)
    minibatch_objective = objective[minibatch_ids]
    if minibatch_mean is not None:
        minibatch_logp = (-.5*((values['action'][minibatch_ids]-minibatch_mean)/SIGMA)**2
                         -np.log(SIGMA)-.5*np.log(2*np.pi)).sum(-1)
        minibatch_ratio = np.exp(minibatch_logp-values['old_logp'][minibatch_ids])
        minibatch_advantage = advantage[minibatch_ids]
        minibatch_objective = weights[minibatch_ids]*np.minimum(minibatch_ratio*minibatch_advantage,
            np.clip(minibatch_ratio, .8, 1.2)*minibatch_advantage)
    kl = float(np.mean(np.sum((mean-old)**2, axis=-1)/(2*SIGMA**2)))
    return dict(kl=finite_number(kl), anchor_max=finite_number(np.abs(anchor-target).max()),
        full_surrogate=finite_number(objective.mean()),
        minibatch_surrogate=finite_number(minibatch_objective.mean()))


def verify_provenance():
    inputs, done = read(OUT/'inputs.json'), read(OUT/'completed.json')
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    resolved = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert digest(Path(__file__)) == resolved[Path(__file__).resolve()]
    for path, sha in inputs.items():
        path = Path(path)
        if path.suffix == '.py' and path.resolve().parent == Path(__file__).resolve().parent:
            assert digest(OUT/'sources'/path.name) == sha
    required = [LEARNING/'inputs.json', LEARNING/'completed.json', LEARNING/'audit.json',
        LEARNING/'quiet-anchors.npz', PROBES/'inputs.json', PROBES/'completed.json', PROBES/'audit.json']
    required += [LEARNING/f'fit-{seed}'/name for seed in ACTORS for name in
                 ('initial.pt', 'collection-1.npz', 'objective-1.npz', 'projection-1.json', 'history.json')]
    required += [PROBES/str(seed)/'dataset.npz' for seed in ACTORS]
    assert all(path.resolve() in resolved for path in required)
    historical = {}
    historical_record = read(OUT/'historical-inputs.json')
    for root in (LEARNING, PROBES):
        assert read(root/'audit.json')['passed']
        manifest = read(root/'completed.json')['outputs']
        count = 0
        for path, sha in resolved.items():
            if path.is_relative_to(root.resolve()):
                relative = str(path.relative_to(root.resolve()))
                if relative in manifest:
                    assert sha == manifest[relative], (root, relative)
                    count += 1
        prior_inputs = {Path(path).resolve(): sha for path, sha in read(root/'inputs.json').items()}
        shared = sorted(set(resolved) & set(prior_inputs))
        assert all(resolved[path] == prior_inputs[path] for path in shared)
        recorded = historical_record[str(root)]
        assert recorded['passed'] and recorded['count'] == len(shared)
        assert sorted(Path(path).resolve() for path in recorded['shared_paths']) == shared
        historical[str(root)] = dict(sealed_outputs=count, shared_inputs=len(shared))
    return inputs, done, historical


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float)**2)))


def probe_changes(delta, probes):
    values = np.asarray(delta, dtype=float).reshape(36)
    branch = probes['probe_branch']
    conditional = values-values[probes['probe_quiet_index']]
    result = dict(quiet_rms=rms(values[branch == 0]), quiet_max=float(np.abs(values[branch == 0]).max()))
    for label, mask in (('push', branch != 0), ('positive_x', branch == 1), ('negative_x', branch == 2)):
        result[label] = dict(raw_mean=float(values[mask].mean()), raw_rms=rms(values[mask]),
            response_mean=float(conditional[mask].mean()), response_rms=rms(conditional[mask]))
    return result


def stage(actor, name, batch, ids, features, target, probes):
    with torch.no_grad():
        mean = actor(batch['x'])
        mini_mean = actor(batch['x'][ids])
        anchor = actor(features).numpy().ravel()
        probe = actor(torch.from_numpy(probes['probe_x'])).numpy()
        ratio = (Normal(mean, SIGMA).log_prob(batch['action']).sum(-1)-batch['old_logp']).exp()
        mini_ratio = (Normal(mini_mean, SIGMA).log_prob(batch['action'][ids]).sum(-1)-batch['old_logp'][ids]).exp()
        av = batch['advantage']
        full_objective = (batch['weight']*torch.minimum(ratio*av, ratio.clamp(.8, 1.2)*av)).mean()
        mini_objective = (batch['weight'][ids]*torch.minimum(mini_ratio*av[ids], mini_ratio.clamp(.8, 1.2)*av[ids])).mean()
        kl = float(((mean-batch['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
    evidence = dict(collection_predictions=mean.numpy(), minibatch_predictions=mini_mean.numpy(),
        anchor_predictions=anchor, probe_predictions=probe, full_ratio=ratio.numpy(), minibatch_ratio=mini_ratio.numpy())
    finite = bool(all(np.isfinite(value).all() for value in evidence.values())
                  and torch.isfinite(parameters_to_vector(actor.parameters())).all()
                  and np.isfinite(kl) and torch.isfinite(full_objective) and torch.isfinite(mini_objective))
    anchor_max = float(np.abs(anchor-target).max())
    metrics = dict(name=name, actor_sha256=actor_hash(actor), finite=finite,
        full_surrogate=finite_number(float(full_objective)), minibatch_surrogate=finite_number(float(mini_objective)),
        kl=finite_number(kl), anchor_max=finite_number(anchor_max),
        passes_guards=finite and kl <= KL_LIMIT and anchor_max <= ANCHOR_LIMIT)
    independent = analytical_metrics(mean.numpy(), anchor, batch, ids.numpy(), target,
                                    minibatch_mean=mini_mean.numpy().astype(float))
    errors = {}
    for key, expected in independent.items():
        if expected is not None and metrics[key] is not None:
            error = abs(metrics[key]-expected)
            tolerance = 5e-7 if 'surrogate' in key else 1e-9
            assert np.isclose(metrics[key], expected, atol=tolerance, rtol=2e-6), (name, key, metrics[key], expected)
            errors[key] = error
        else:
            assert expected is metrics[key] is None
    return evidence, metrics, errors


def verify_actor(seed, features, target):
    folder = OUT/str(seed)
    evidence, report, replay = arrays(folder/'evidence.npz'), read(folder/'report.json'), read(folder/'replay.json')
    pre, rejected, optimizer, batch, indices, ninth = replay_candidate(seed, features)
    probes = arrays(PROBES/str(seed)/'dataset.npz')
    assert probes['probe_x'].shape == (36, 427)
    assert np.array_equal(probes['probe_x'][probes['probe_branch'] == 0], features.numpy())
    assert np.array_equal(evidence['minibatch_indices'], indices)
    assert report['seed'] == seed and report['collection'] == 1 and report['attempt'] == 9
    assert report['state_order'] == ['pre', 'rejected', 'corrected']
    historical = read(LEARNING/f'fit-{seed}'/'projection-1.json')
    assert len(replay) == 9
    for index, (record, previous) in enumerate(zip(replay, historical)):
        assert record['index'] == index and record['exact_historical_hashes']
        for key in ('accepted', 'actor_loss', 'gradient_norm', 'clipped_norm', 'kl', 'anchor_max',
                    'actor_pre', 'actor_tentative', 'actor_candidate', 'actor_post',
                    'optimizer_pre', 'optimizer_tentative', 'optimizer_post'):
            assert record[key] == previous[key], (seed, index, key)
    pre_parameters = parameters_to_vector(pre.parameters()).detach().clone()
    rejected_parameters = parameters_to_vector(rejected.parameters()).detach().clone()
    assert np.array_equal(evidence['parameters_pre'], pre_parameters.numpy())
    assert np.array_equal(evidence['parameters_rejected'], rejected_parameters.numpy())
    quiet = quiet_jacobian(rejected, features)
    assert np.array_equal(quiet, evidence['jacobian_candidate'])
    with torch.no_grad():
        residual = rejected(features).numpy().ravel().astype(float)-target.astype(float)
    assert np.array_equal(residual, evidence['residual'])
    left, singular, right = np.linalg.svd(quiet, full_matrices=False)
    keep = singular > CUTOFF*singular[0]
    correction = -right[keep].T@((left[:, keep].T@residual)/singular[keep])
    assert np.array_equal(singular, evidence['singular_values'])
    assert np.array_equal(correction, evidence['correction'])
    linear = linear_verify(quiet, residual, correction, singular)
    compare(report['independent'], linear, 'independent correction')
    corrected = deepcopy(rejected)
    corrected_optimizer = torch.optim.Adam(corrected.parameters(), lr=1e-4)
    corrected_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    optimizer_before = optimizer_hash(corrected_optimizer)
    corrected_parameters = rejected_parameters+torch.from_numpy(correction.astype(np.float32))
    assert np.array_equal(corrected_parameters.numpy(), evidence['parameters_corrected'])
    assert evidence['parameters_corrected'].dtype == np.float32
    vector_to_parameters(corrected_parameters, corrected.parameters())
    actual = corrected_parameters.double().numpy()-rejected_parameters.double().numpy()
    assert np.array_equal(actual, evidence['actual_correction'])
    assert np.array_equal(quiet@correction, evidence['linear_prediction'])
    assert np.array_equal(quiet@actual, evidence['rounded_prediction'])
    assert optimizer_hash(corrected_optimizer) == optimizer_before == ninth['optimizer_tentative']
    assert report['optimizer_tentative_before'] == report['optimizer_tentative_after'] == optimizer_before
    assert report['tentative_optimizer_unchanged'] is True
    ids = torch.from_numpy(indices)
    stages, arithmetic_errors = [], []
    for index, (name, actor) in enumerate(zip(('pre', 'rejected', 'corrected'), (pre, rejected, corrected))):
        expected, metrics, errors = stage(actor, name, batch, ids, features, target, probes)
        for key, value in expected.items():
            assert np.array_equal(evidence[key][index], value, equal_nan=True), (seed, name, key)
        compare(report['states'][index], metrics, 'states.'+name, atol=0, rtol=0)
        stages.append(metrics)
        arithmetic_errors.append(errors)
    assert stages[0]['passes_guards'] and not stages[1]['passes_guards']
    proposal = rejected_parameters.double().numpy()-pre_parameters.double().numpy()
    corrected_change = corrected_parameters.double().numpy()-pre_parameters.double().numpy()
    proposal_norm, corrected_norm = np.linalg.norm(proposal), np.linalg.norm(corrected_change)
    nonlinear = evidence['anchor_predictions'][2].astype(float)-evidence['anchor_predictions'][1].astype(float)-quiet@actual
    geometry = dict(rank=int(keep.sum()), threshold=CUTOFF, proposed_norm=float(np.linalg.norm(correction)),
        actual_norm=finite_number(np.linalg.norm(actual)), rounding_norm=finite_number(np.linalg.norm(actual-correction)),
        rejected_minus_pre_norm=float(proposal_norm),
        correction_to_proposal_norm=float(np.linalg.norm(correction)/proposal_norm),
        actual_correction_to_proposal_norm=finite_number(np.linalg.norm(actual)/proposal_norm),
        corrected_minus_pre_norm=finite_number(corrected_norm),
        corrected_step_cosine_to_rejected=finite_number(corrected_change@proposal/(corrected_norm*proposal_norm)),
        linear_residual_max=float(np.abs(residual+quiet@correction).max()),
        rounded_linear_residual_max=finite_number(np.abs(residual+quiet@actual).max()),
        nonlinear_rms=rms(nonlinear) if stages[2]['finite'] else None,
        nonlinear_max=float(np.abs(nonlinear).max()) if stages[2]['finite'] else None,
        tentative_adam_norm=ninth['tentative_norm'],
        correction_to_tentative_adam_norm=float(np.linalg.norm(correction)/ninth['tentative_norm']),
        actual_correction_to_tentative_adam_norm=finite_number(np.linalg.norm(actual)/ninth['tentative_norm']))
    compare(report['correction'], geometry, 'correction geometry')
    changes = {}
    for label, end, start in (('rejected_minus_pre', 1, 0), ('corrected_minus_pre', 2, 0),
                              ('corrected_minus_rejected', 2, 1)):
        changes[label] = None if not (stages[end]['finite'] and stages[start]['finite']) else dict(
            full_surrogate=stages[end]['full_surrogate']-stages[start]['full_surrogate'],
            minibatch_surrogate=stages[end]['minibatch_surrogate']-stages[start]['minibatch_surrogate'],
            probes=probe_changes(evidence['probe_predictions'][end].astype(float)
                                -evidence['probe_predictions'][start].astype(float), probes))
    compare(report['changes'], changes, 'changes')
    return dict(exact_historical_actor_attempts=9, accepted_historical_attempts=8, rejected_historical_attempts=1,
        actor_and_adam_hashes_exact=True, candidate_jacobian_exact=True,
        predictions_and_float32_application_exact=True, tentative_optimizer_unchanged=True,
        independent_minimum_norm=linear, float64_probability_objective_errors=arithmetic_errors,
        corrected_passes_guards=stages[2]['passes_guards'],
        corrected_full_surrogate_improvement=changes['corrected_minus_pre']['full_surrogate'] if changes['corrected_minus_pre'] else None,
        corrected_minibatch_surrogate_improvement=changes['corrected_minus_pre']['minibatch_surrogate'] if changes['corrected_minus_pre'] else None,
        correction_to_proposal_norm=geometry['correction_to_proposal_norm'],
        scope='Constraint feasibility and objective movement are separate; no physical recovery conclusion.')


def main():
    inputs, done, historical = verify_provenance()
    contract = read(OUT/'contract.json')
    assert contract['seeds'] == list(ACTORS) and contract['source'] == str(LEARNING)
    assert contract['fixed_collection'] == 1 and contract['fixed_attempt'] == 9
    assert contract['state_order'] == ['pre', 'rejected', 'corrected']
    assert contract['sigma'] == SIGMA and contract['singular_relative_cutoff'] == CUTOFF
    assert contract['anchor_limit'] == ANCHOR_LIMIT and contract['kl_limit'] == KL_LIMIT
    assert contract['corrections_per_actor'] == 1 and not contract['line_search']
    assert contract['retries'] == 0 and not contract['correction_rescaled'] and not contract['optimizer_moments_changed']
    assert contract['independent_linear_gate_before_application'] is True
    assert contract['anchor_count'] == 12 and contract['probe_count'] == 36
    assert contract['saved_collection_samples'] == 768 and contract['rejected_minibatch_samples'] == 256
    assert contract['historical_actor_attempts_replayed'] == done['historical_actor_attempts_replayed'] == 18
    for key in ('newly_collected_transitions', 'physical_rollouts', 'new_training_updates', 'critic_updates', 'new_checkpoints'):
        assert contract[key] == done[key] == 0
    assert done['actors'] == done['corrections'] == 2
    assert done['inputs_unchanged'] and done['exact_historical_hash_parity']
    assert done['independent_linear_checks_passed'] and done['tentative_optimizers_unchanged']
    assert not list(OUT.rglob('*.pt'))
    bank = arrays(LEARNING/'quiet-anchors.npz')
    assert bank['features'].shape == (12, 427) and not bank['features'][:, -2:].any()
    assert bank['target'].shape == (12, 1) and not bank['target'].any()
    assert np.isfinite(bank['features']).all()
    features, target = torch.from_numpy(bank['features']), bank['target'].ravel()
    torch.set_num_threads(1)
    actors = {str(seed): verify_actor(seed, features, target) for seed in ACTORS}
    results = read(OUT/'results.json')
    assert set(results) == {str(seed) for seed in ACTORS}
    for seed in ACTORS:
        assert results[str(seed)] == read(OUT/str(seed)/'report.json')
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    audit = dict(passed=True, input_hashes=len(inputs), output_hashes=len(done['outputs']),
        historical=historical, actors=actors, independent_actor_attempts_replayed=18,
        corrected_actor_copies=2, new_physical_rollouts=0, new_training_updates=0,
        scope='Audit success does not require either correction to pass guards or preserve surrogate improvement; no physical benefit inferred.')
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
