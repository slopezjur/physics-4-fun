"""Independent Gram-eigen audit of quiet-preserving local actor projections."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-quiet-actor-projection-20261009-01')
BASE = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
LEARNING = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
SIGMA = .08717797887081347
THRESHOLDS = (1e-5, 1e-6, 1e-7)


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def linear_verify(jacobian, g, projected, singular_values, thresholds, quiet_indices):
    """Check SVD-produced directions using the quiet Gram matrix's eigenvectors."""
    jacobian, g, projected, singular_values = (np.asarray(value, dtype=np.float64)
                                              for value in (jacobian, g, projected, singular_values))
    quiet_indices = np.asarray(quiet_indices, dtype=np.int64)
    thresholds = tuple(float(value) for value in thresholds)
    assert thresholds == THRESHOLDS
    assert jacobian.ndim == 2 and jacobian.shape[0] == 36
    assert g.shape == (jacobian.shape[1],) and projected.shape == (3, len(g))
    assert singular_values.shape == quiet_indices.shape == (12,)
    assert len(np.unique(quiet_indices)) == 12 and ((quiet_indices >= 0) & (quiet_indices < 36)).all()
    assert all(np.isfinite(value).all() for value in (jacobian, g, projected, singular_values))
    quiet = jacobian[quiet_indices]
    gram = quiet@quiet.T
    values, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, eigenvectors = values[order], eigenvectors[:, order]
    numerical_floor = 64*np.finfo(float).eps*max(1., float(values[0]))
    assert values[-1] >= -numerical_floor
    singular = np.sqrt(np.maximum(values, 0.))
    assert np.allclose(singular, singular_values, atol=1e-9, rtol=2e-5)
    gradient_norm = float(np.linalg.norm(g))
    assert gradient_norm > 0 and singular[0] > 0
    rows = []
    for i, threshold in enumerate(thresholds):
        cutoff_squared = (threshold*singular[0])**2
        # A cutoff indistinguishable from a Gram eigenvalue does not license selecting a rank.
        assert np.min(np.abs(values-cutoff_squared)) > numerical_floor, ('ambiguous rank', threshold)
        keep = values > cutoff_squared
        assert int(keep.sum()) == int(np.count_nonzero(singular_values > threshold*singular_values[0]))
        basis = eigenvectors[:, keep]
        removed = quiet.T@(basis@((basis.T@(quiet@g))/values[keep]))
        independent = g-removed
        supplied = projected[i]
        max_error = float(np.max(np.abs(supplied-independent)))
        relative_error = float(np.linalg.norm(supplied-independent)/max(np.linalg.norm(independent), 1e-30))
        assert max_error <= 2e-8 and relative_error <= 2e-5, ('projection disagreement', threshold, max_error, relative_error)
        root_removed = g-supplied
        orthogonality = float(np.dot(root_removed, supplied))
        projected_squared = float(np.dot(supplied, supplied))
        removed_squared = float(np.dot(root_removed, root_removed))
        norm_error = abs(gradient_norm**2-projected_squared-removed_squared)
        ascent_error = abs(float(np.dot(g, supplied))-projected_squared)
        identity_tolerance = 2e-10+2e-7*gradient_norm**2
        assert abs(orthogonality) <= identity_tolerance
        assert norm_error <= identity_tolerance and ascent_error <= identity_tolerance
        quiet_residual = float(np.linalg.norm(quiet@supplied))
        residual_bound = max(1e-10, 2*singular_values[0]*threshold*gradient_norm)
        assert quiet_residual <= residual_bound, ('quiet residual', threshold, quiet_residual, residual_bound)
        retained_residual = float(np.linalg.norm(basis.T@(quiet@supplied)))
        assert retained_residual <= 1e-10+2e-7*singular_values[0]*gradient_norm
        rows.append(dict(threshold=threshold, rank=int(keep.sum()),
                         projection_max_absolute_error=max_error, projection_relative_l2_error=relative_error,
                         quiet_residual_l2=quiet_residual, quiet_residual_bound=residual_bound,
                         retained_quiet_residual_l2=retained_residual,
                         removed_projected_inner_product=orthogonality,
                         pythagorean_error=norm_error, ascent_identity_error=ascent_error,
                         retained_gradient_norm_fraction=float(np.linalg.norm(supplied)/gradient_norm)))
    return dict(passed=True, method='float64 quiet Gram eigendecomposition',
                spectrum_max_absolute_error=float(np.max(np.abs(singular-singular_values))),
                eigenvalue_numerical_floor=numerical_floor,
                ranks=[row['rank'] for row in rows], thresholds=rows)


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float)**2)))


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def metrics(displacement, dataset):
    value = np.asarray(displacement, dtype=float).reshape(36)
    branch = dataset['probe_branch']
    quiet = branch == 0
    response = value-value[dataset['probe_quiet_index']]
    def ranks(vector, mask):
        directed = vector[mask]*dataset['probe_preferred'][mask]
        return dict(helpful=int(np.count_nonzero(directed > 1e-7)), harmful=int(np.count_nonzero(directed < -1e-7)),
                    tied=int(np.count_nonzero(np.abs(directed) <= 1e-7)),
                    signed_minimum=float(directed.min()), signed_mean=float(directed.mean()), rms=rms(vector[mask]))
    return dict(all_rms=rms(value), quiet_rms=rms(value[quiet]), quiet_max=float(np.abs(value[quiet]).max()),
                raw_push=ranks(value, ~quiet), response_push=ranks(response, ~quiet),
                by_branch={name: ranks(value, branch == i) for name, i in (('+X', 1), ('-X', 2))})


def compare(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label+'.'+str(i))
    elif isinstance(expected, (int, bool, str)) or expected is None:
        assert actual == expected, (label, actual, expected)
    else:
        assert np.isclose(actual, expected, atol=1e-10, rtol=1e-7), (label, actual, expected)


def surrogate(actor, dataset):
    x, actions, old_mean, advantage = (torch.from_numpy(dataset[key]) for key in ('x', 'action', 'old_mean', 'ordinary'))
    ids = torch.arange(len(advantage))
    old_logp = Normal(old_mean, SIGMA).log_prob(actions).sum(-1)
    ratio = (Normal(actor(x[ids]), SIGMA).log_prob(actions[ids]).sum(-1)-old_logp[ids]).exp()
    return torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids]).mean()


def reconstruct(actor, dataset):
    objective = surrogate(actor, dataset)
    gradient = torch.cat([value.reshape(-1) for value in torch.autograd.grad(objective, tuple(actor.parameters()))])
    jacobian = []
    for point in torch.from_numpy(dataset['probe_x']):
        jacobian.append(torch.cat([value.reshape(-1) for value in
                                  torch.autograd.grad(actor(point[None]).sum(), tuple(actor.parameters()))]).detach().double().numpy())
    return gradient.detach().double().numpy(), np.stack(jacobian)


def parameter_hash(parameters):
    return hashlib.sha256(parameters.detach().numpy().tobytes()).hexdigest()


def check_finite(actor, dataset, g, jacobian, projected, alpha_initial, evidence, report):
    assert np.isfinite(alpha_initial) and alpha_initial > 0
    original = parameters_to_vector(actor.parameters()).detach().clone()
    samples = torch.from_numpy(dataset['x'])
    probes = torch.from_numpy(dataset['probe_x'])
    old_mean = torch.from_numpy(dataset['old_mean'])
    with torch.no_grad():
        before = actor(probes).numpy().copy()
        assert report['initial_surrogate'] == float(surrogate(actor, dataset))
    assert np.array_equal(evidence['before_mean'], before)
    count = len(report['attempts'])
    assert 1 <= count <= 9
    assert evidence['means'].shape == (count, 2, 36, 1)
    assert evidence['predicted'].shape == evidence['rounded_predictions'].shape == (count, 2, 36)
    assert evidence['alphas'].shape == (count,) and evidence['kls'].shape == (count, 2)
    accepted = False
    for attempt, recorded in enumerate(report['attempts']):
        assert not accepted, 'A common accepted step must stop the reduction sequence'
        alpha = alpha_initial/2**attempt
        assert recorded['halvings'] == attempt and recorded['alpha'] == evidence['alphas'][attempt] == alpha
        assert len(recorded['arms']) == 2 and len(recorded['kls']) == 2
        decisions = []
        for arm_index, (name, direction) in enumerate((('original', g), ('projected', projected))):
            nominal = alpha*direction
            candidate = deepcopy(actor)
            parameters = original+torch.from_numpy(nominal.astype(np.float32))
            vector_to_parameters(parameters, candidate.parameters())
            actual = parameters.double().numpy()-original.double().numpy()
            prediction, rounded = jacobian@nominal, jacobian@actual
            with torch.no_grad():
                mean = candidate(probes).numpy().copy()
                sample_mean = candidate(samples)
                kl = float(((sample_mean-old_mean).square().sum(-1)/(2*SIGMA**2)).mean())
                finite = bool(np.isfinite(mean).all() and torch.isfinite(sample_mean).all()
                              and torch.isfinite(parameters).all() and np.isfinite(kl))
                objective = float(surrogate(candidate, dataset)) if finite else None
            delta = mean.astype(float).ravel()-before.astype(float).ravel()
            for key, value in (('means', mean), ('predicted', prediction), ('rounded_predictions', rounded)):
                assert np.array_equal(evidence[key][attempt, arm_index], value, equal_nan=True), (attempt, name, key)
            assert np.array_equal(np.asarray(evidence['kls'][attempt, arm_index]), np.asarray(kl), equal_nan=True)
            assert recorded['kls'][arm_index] == (kl if np.isfinite(kl) else None)
            expected = dict(name=name, finite=finite, parameter_sha256=parameter_hash(parameters),
                            parameter_delta_norm=finite_number(np.linalg.norm(actual)),
                            parameter_rounding_norm=finite_number(np.linalg.norm(actual-nominal)),
                            metrics=metrics(delta, dataset) if finite else None,
                            predicted_metrics=metrics(prediction, dataset),
                            rounded_prediction_metrics=metrics(rounded, dataset) if np.isfinite(rounded).all() else None,
                            linearization_rms=rms(delta-prediction) if finite else None,
                            max_linearization_error=float(np.abs(delta-prediction).max()) if finite else None,
                            nonlinear_rms=rms(delta-rounded) if finite else None,
                            max_nonlinear_error=float(np.abs(delta-rounded).max()) if finite else None,
                            surrogate=objective, expected_surrogate_change=float(g@nominal),
                            rounded_expected_surrogate_change=finite_number(g@actual))
            compare(recorded['arms'][arm_index], expected, 'finite arm')
            decisions.append(finite and kl <= .01)
        accepted = all(decisions)
        assert recorded['accepted'] is accepted
    assert report['accepted'] is accepted and report['final_halvings'] == count-1
    assert accepted or count == 9
    assert torch.equal(parameters_to_vector(actor.parameters()).detach(), original)
    return dict(attempts=count, accepted=accepted, halvings=count-1,
                exact_float32_parameter_addition=True, exact_means_and_predictions=True,
                common_step_and_kl_guard_verified=True)


def provenance(inputs, completed):
    for path, sha in inputs.items():
        assert digest(Path(path)) == sha, path
    for path, sha in completed['outputs'].items():
        assert digest(OUT/path) == sha, path
    assert read(BASE/'audit.json')['passed'] is True
    previous = read(BASE/'inputs.json')
    shared = previous.keys() & inputs.keys()
    assert shared and all(previous[key] == inputs[key] for key in shared)
    for root in (BASE, LEARNING):
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
    assert completed['inputs_unchanged'] is True and completed['independent_linear_checks_passed'] is True
    assert completed['actors'] == 2 and completed['unique_actor_starts'] == 72
    for key in ('physical_rollouts', 'optimizer_updates', 'critic_updates', 'new_checkpoints'):
        assert completed[key] == contract[key] == 0
    assert not list(OUT.rglob('*.pt')) and not list(OUT.rglob('trace.npz'))
    assert contract['actors'] == [921009, 931009] and contract['thresholds'] == list(THRESHOLDS)
    assert contract['primary_index'] == 1 and contract['projected_direction_renormalized'] is False
    assert contract['parameter_dtype'] == 'float32' and contract['algebra_dtype'] == 'float64'
    assert contract['probe_count_per_actor'] == 36 and contract['quiet_count_per_actor'] == 12
    assert contract['push_count_per_actor'] == 24 and contract['saved_samples_per_actor'] == 288
    assert contract['sigma'] == SIGMA and contract['target_original_predicted_rms'] == .01*SIGMA
    assert contract['target_original_predicted_kl'] == 5e-5
    assert contract['kl_limit'] == .01 and contract['max_halvings'] == 8
    assert contract['finite_direction_order'] == ['original', 'projected']
    assert contract['movement_sign_tolerance'] == 1e-7
    assert contract['independent_linear_check_before_finite'] is True
    assert contract['shared_historical_inputs'] == shared
    audited, displacements, accepted = {}, 0, 0
    for seed in (921009, 931009):
        folder = OUT/str(seed)
        dataset = arrays(BASE/str(seed)/'dataset.npz')
        prior = arrays(BASE/str(seed)/'initial.npz')
        linear, linear_report = arrays(folder/'linear.npz'), read(folder/'linear.json')
        finite, finite_report = arrays(folder/'finite.npz'), read(folder/'finite.json')
        assert results[str(seed)] == dict(linear=linear_report, finite=finite_report)
        actor = AnkleHead().eval()
        actor.load_state_dict(torch.load(LEARNING/f'fit-{seed}'/'final.pt', weights_only=True, map_location='cpu'))
        original = parameters_to_vector(actor.parameters()).detach().clone()
        g, jacobian = reconstruct(actor, dataset)
        assert np.array_equal(g, prior['gradient'][0, 0].astype(float))
        assert np.array_equal(g, linear['gradient'])
        assert np.allclose(jacobian@jacobian.T, prior['jacobian_gram'], atol=1e-10, rtol=1e-8)
        assert np.allclose(jacobian@g, prior['response'][0, 0], atol=1e-10, rtol=1e-8)
        assert np.array_equal(jacobian@jacobian.T, linear['jacobian_gram'])
        quiet = np.flatnonzero(dataset['probe_branch'] == 0)
        assert np.array_equal(quiet, linear['quiet_indices'])
        assert np.array_equal(quiet, np.r_[np.arange(6), np.arange(18, 24)])
        independent = linear_verify(jacobian, g, linear['projected'], linear['singular_values'], THRESHOLDS, quiet)
        compare(linear_report['independent'], independent, 'independent linear gate')
        assert linear_report['thresholds'] == list(THRESHOLDS) and linear_report['ranks'] == independent['ranks']
        responses = np.stack([jacobian@g, *[jacobian@direction for direction in linear['projected']]])
        assert np.array_equal(linear['responses'], responses)
        with torch.no_grad():
            before = actor(torch.from_numpy(dataset['probe_x'])).numpy()
        assert np.array_equal(before, linear['mean_before']) and np.array_equal(before, finite['before_mean'])
        alpha = .01*SIGMA/rms(responses[0])
        assert linear_report['alpha_initial'] == alpha
        assert np.isclose(np.mean((alpha*responses[0])**2)/(2*SIGMA**2), 5e-5, atol=1e-15, rtol=1e-12)
        compare(linear_report['original_metrics'], metrics(responses[0], dataset), 'linear original metrics')
        assert linear_report['original_gradient_norm'] == float(np.linalg.norm(g))
        assert len(linear_report['sensitivity']) == len(THRESHOLDS)
        for index, item in enumerate(linear_report['sensitivity']):
            expected = dict(threshold=THRESHOLDS[index], rank=independent['ranks'][index],
                            retained_norm_fraction=float(np.linalg.norm(linear['projected'][index])/np.linalg.norm(g)),
                            metrics=metrics(responses[index+1], dataset))
            compare(item, expected, 'projection sensitivity')
        finite_audit = check_finite(actor, dataset, g, jacobian, linear['projected'][1], alpha, finite, finite_report)
        assert torch.equal(parameters_to_vector(actor.parameters()).detach(), original)
        audited[str(seed)] = dict(linear=independent, finite=finite_audit)
        displacements += 2*finite_audit['attempts']
        accepted += int(finite_audit['accepted'])
    assert displacements == completed['finite_candidate_displacements']
    assert accepted == completed['successful_actor_comparisons']
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    audit = dict(passed=True, actors=audited, unique_actor_starts=72, quiet_actor_starts=24, push_actor_starts=48,
                 input_hashes=len(inputs), output_hashes=len(completed['outputs']), historical_input_hashes=shared,
                 independent_gram_eigen_projection=True, exact_historical_gradient=True,
                 historical_jacobian_gram_and_response_verified=True,
                 finite_candidate_displacements=displacements, successful_actor_comparisons=accepted,
                 physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
                 scope='Local fitted-state feasibility. Float32 rounding and nonlinear output changes are '
                 'separated from tangent predictions; no physical recovery or heldout performance is measured.')
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
