"""Quiet-output tangent projection and bounded finite actor displacements."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .probe_local_actor_updates import (
    ACTORS, HALF_SIGMA, LEARNING, arrays, gradient, load_actor, parameter_hash,
    parameter_vector, read, surrogate, tensors,
)
from .run_support_skill_study import digest, save
from .verify_quiet_actor_projection import linear_verify

BASE = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
OUT = Path('logs/mimickit-training/recovery-quiet-actor-projection-20261009-01')
THRESHOLDS = np.array([1e-5, 1e-6, 1e-7])
PRIMARY = 1
MOVEMENT_TOLERANCE = 1e-7


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float)**2)))


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def metrics(delta, z):
    delta = np.asarray(delta, dtype=float).reshape(36)
    branch = z['probe_branch']
    quiet, push = branch == 0, branch != 0
    response = delta-delta[z['probe_quiet_index']]
    def ranks(value, mask):
        signed = value[mask]*z['probe_preferred'][mask]
        return dict(helpful=int((signed > MOVEMENT_TOLERANCE).sum()),
            harmful=int((signed < -MOVEMENT_TOLERANCE).sum()),
            tied=int((np.abs(signed) <= MOVEMENT_TOLERANCE).sum()),
            signed_minimum=float(signed.min()), signed_mean=float(signed.mean()), rms=rms(value[mask]))
    return dict(all_rms=rms(delta), quiet_rms=rms(delta[quiet]),
        quiet_max=float(np.max(np.abs(delta[quiet]))), raw_push=ranks(delta, push),
        response_push=ranks(response, push),
        by_branch={label: ranks(delta, branch == number) for label, number in (('+X', 1), ('-X', 2))})


def reconstruct(actor, t):
    g = gradient(actor, t, t['ordinary']).double().numpy()
    rows = []
    for x in t['probe_x']:
        rows.append(torch.cat([part.flatten() for part in
            torch.autograd.grad(actor(x[None]).sum(), actor.parameters())]).detach().double().numpy())
    return g, np.stack(rows)


def project(g, jacobian, quiet_indices):
    q = jacobian[quiet_indices]
    _, singular, vt = np.linalg.svd(q, full_matrices=False)
    directions, ranks = [], []
    for threshold in THRESHOLDS:
        basis = vt[singular > threshold*singular[0]]
        directions.append(g-basis.T@(basis@g))
        ranks.append(len(basis))
    return np.stack(directions), singular, ranks


def finite_comparison(actor, t, z, g, jacobian, projected, alpha_initial):
    directions = (g, projected)
    original = parameter_vector(actor).clone()
    with torch.no_grad():
        before = actor(t['probe_x']).numpy().copy()
        initial_objective = float(surrogate(actor, t, t['ordinary'])[0])
    evidence = {key: [] for key in ('means', 'predicted', 'rounded_predictions', 'alphas', 'kls')}
    attempts = []
    for halvings in range(9):
        alpha = alpha_initial/(2**halvings)
        means, predictions, rounded_predictions, kls, arms = [], [], [], [], []
        for name, direction in zip(('original', 'projected'), directions):
            proposed = alpha*direction
            candidate = deepcopy(actor)
            parameters = original+torch.from_numpy(proposed.astype(np.float32))
            torch.nn.utils.vector_to_parameters(parameters, candidate.parameters())
            displacement = parameters.double().numpy()-original.double().numpy()
            prediction = jacobian@proposed
            rounded_prediction = jacobian@displacement
            with torch.no_grad():
                mean = candidate(t['probe_x']).numpy().copy()
                sample_mean = candidate(t['x'])
                kl = float(((sample_mean-t['old_mean']).square().sum(-1)/(2*HALF_SIGMA**2)).mean())
                finite = bool(np.isfinite(mean).all() and torch.isfinite(sample_mean).all()
                              and torch.isfinite(parameters).all() and np.isfinite(kl))
                objective = float(surrogate(candidate, t, t['ordinary'])[0]) if finite else None
            delta = mean.astype(float).ravel()-before.astype(float).ravel()
            means.append(mean)
            predictions.append(prediction)
            rounded_predictions.append(rounded_prediction)
            kls.append(kl)
            arms.append(dict(name=name, finite=finite, parameter_sha256=parameter_hash(candidate),
                parameter_delta_norm=finite_number(np.linalg.norm(displacement)),
                parameter_rounding_norm=finite_number(np.linalg.norm(displacement-proposed)),
                metrics=metrics(delta, z) if finite else None,
                predicted_metrics=metrics(prediction, z),
                rounded_prediction_metrics=metrics(rounded_prediction, z) if np.isfinite(rounded_prediction).all() else None,
                linearization_rms=rms(delta-prediction) if finite else None,
                max_linearization_error=float(np.max(np.abs(delta-prediction))) if finite else None,
                nonlinear_rms=rms(delta-rounded_prediction) if finite else None,
                max_nonlinear_error=float(np.max(np.abs(delta-rounded_prediction))) if finite else None,
                surrogate=objective, expected_surrogate_change=float(g@proposed),
                rounded_expected_surrogate_change=finite_number(g@displacement)))
        accepted = all(arm['finite'] and kl <= .01 for arm, kl in zip(arms, kls))
        attempts.append(dict(halvings=halvings, alpha=alpha,
            kls=[kl if np.isfinite(kl) else None for kl in kls], accepted=accepted, arms=arms))
        for key, value in (('means', means), ('predicted', predictions), ('rounded_predictions', rounded_predictions),
                           ('alphas', alpha), ('kls', kls)):
            evidence[key].append(value)
        if accepted:
            break
    assert torch.equal(parameter_vector(actor), original)
    return {**{key: np.asarray(value) for key, value in evidence.items()}, 'before_mean': before}, dict(
        initial_surrogate=initial_objective, attempts=attempts, accepted=attempts[-1]['accepted'],
        final_halvings=attempts[-1]['halvings'],
        scope='Finite latent-mean changes on fitted diagnostic observations, not recovery or heldout performance.')


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    sources = sorted(Path(__file__).parent.glob('*.py'))
    for path in sources:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths = sources+[Path(__file__).with_name('RECOVERY_QUIET_ACTOR_PROJECTION_PLAN.md'),
        BASE/'inputs.json', BASE/'completed.json', BASE/'audit.json', BASE/'contract.json', BASE/'results.json']
    assert read(BASE/'audit.json')['passed']
    old_outputs = read(BASE/'completed.json')['outputs']
    reused = [BASE/str(seed)/name for seed in ACTORS for name in
              ('dataset.npz', 'metadata.json', 'initial.npz', 'initial.json')]
    for path in reused:
        assert digest(path) == old_outputs[str(path.relative_to(BASE))]
    paths += reused+[LEARNING/f'fit-{seed}'/name for seed in ACTORS for name in ('final.pt', 'critic-final.pt')]
    inputs = {str(path): digest(path) for path in paths}
    historical = read(BASE/'inputs.json')
    common = sorted(set(historical) & set(inputs))
    assert all(historical[path] == inputs[path] for path in common)
    save(OUT/'inputs.json', inputs)
    save(OUT/'contract.json', dict(actors=list(ACTORS), thresholds=THRESHOLDS.tolist(), primary_index=PRIMARY,
        parameter_dtype='float32', algebra_dtype='float64', projected_direction_renormalized=False,
        probe_count_per_actor=36, quiet_count_per_actor=12, push_count_per_actor=24, saved_samples_per_actor=288,
        sigma=HALF_SIGMA, target_original_predicted_rms=.01*HALF_SIGMA, target_original_predicted_kl=5e-5,
        kl_limit=.01, max_halvings=8, finite_direction_order=['original', 'projected'],
        step_cast='cast alpha*direction once to float32, then add original float32 parameters',
        movement_sign_tolerance=MOVEMENT_TOLERANCE, independent_linear_check_before_finite=True,
        physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        shared_historical_inputs=len(common)))
    torch.set_num_threads(1)
    results = {}
    for seed in ACTORS:
        folder = OUT/str(seed)
        folder.mkdir()
        z = arrays(BASE/str(seed)/'dataset.npz')
        prior = arrays(BASE/str(seed)/'initial.npz')
        actor, t = load_actor(seed), tensors(z)
        original_hash = parameter_hash(actor)
        g, jacobian = reconstruct(actor, t)
        assert np.array_equal(g, prior['gradient'][0, 0].astype(float))
        assert np.allclose(jacobian@jacobian.T, prior['jacobian_gram'], atol=1e-10, rtol=1e-8)
        assert np.allclose(jacobian@g, prior['response'][0, 0], atol=1e-10, rtol=1e-8)
        quiet_indices = np.flatnonzero(z['probe_branch'] == 0)
        projected, singular, ranks = project(g, jacobian, quiet_indices)
        independent = linear_verify(jacobian, g, projected, singular, THRESHOLDS, quiet_indices)
        assert independent['passed']
        responses = np.stack([jacobian@g, *[jacobian@direction for direction in projected]])
        assert rms(responses[0]) > 0
        alpha_initial = .01*HALF_SIGMA/rms(responses[0])
        assert np.isfinite(alpha_initial)
        with torch.no_grad():
            before = actor(t['probe_x']).numpy().copy()
        evidence = dict(gradient=g, projected=projected, jacobian_gram=jacobian@jacobian.T,
            singular_values=singular, responses=responses, mean_before=before, quiet_indices=quiet_indices)
        np.savez_compressed(folder/'linear.npz', **evidence)
        linear = dict(thresholds=THRESHOLDS.tolist(), ranks=ranks, alpha_initial=alpha_initial,
            original_metrics=metrics(responses[0], z), original_gradient_norm=float(np.linalg.norm(g)),
            sensitivity=[dict(threshold=float(threshold), rank=rank,
                retained_norm_fraction=float(np.linalg.norm(direction)/np.linalg.norm(g)),
                metrics=metrics(response, z)) for threshold, rank, direction, response in
                zip(THRESHOLDS, ranks, projected, responses[1:])], independent=independent)
        save(folder/'linear.json', linear)
        # The independent algebra gate is complete before any finite actor copy is changed.
        finite_arrays, finite = finite_comparison(actor, t, z, g, jacobian, projected[PRIMARY], alpha_initial)
        np.savez_compressed(folder/'finite.npz', **finite_arrays)
        save(folder/'finite.json', finite)
        assert parameter_hash(actor) == original_hash
        results[str(seed)] = dict(linear=linear, finite=finite)
        print('ACTOR', seed, json.dumps(dict(ranks=ranks,
            retained_norm=linear['sensitivity'][PRIMARY]['retained_norm_fraction'],
            accepted=finite['accepted'], halvings=finite['final_halvings'],
            final=[dict(name=arm['name'], metrics=arm['metrics']) for arm in finite['attempts'][-1]['arms']])), flush=True)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-started, inputs_unchanged=True,
        independent_linear_checks_passed=True, actors=2, unique_actor_starts=72,
        finite_candidate_displacements=2*sum(len(r['finite']['attempts']) for r in results.values()),
        successful_actor_comparisons=sum(r['finite']['accepted'] for r in results.values()),
        physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
