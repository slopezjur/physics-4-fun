"""Independent projection, optimizer, collection and physical-outcome audit."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from . import verify_age_weighted_recovery_v2 as inherited
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead

OUT = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
FULL = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
PROBES = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
SIGMA = .08717797887081347
CUTOFF = 1e-6
ANCHOR_LIMIT = 1e-5


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def projection_verify(jacobian, tentative, projected, singular_values):
    """Independently verify an SVD projection through a quiet Gram eigensystem."""
    quiet, proposal, supplied, singular_values = (
        np.asarray(value, dtype=np.float64)
        for value in (jacobian, tentative, projected, singular_values))
    assert quiet.ndim == 2 and quiet.shape[0] == 12
    assert proposal.shape == supplied.shape == (quiet.shape[1],)
    assert singular_values.shape == (12,)
    assert all(np.isfinite(value).all() for value in (quiet, proposal, supplied, singular_values))
    gram = quiet@quiet.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    floor = 64*np.finfo(float).eps*max(1., float(eigenvalues[0]))
    assert eigenvalues[-1] >= -floor
    singular = np.sqrt(np.maximum(eigenvalues, 0.))
    assert np.allclose(singular, singular_values, atol=1e-9, rtol=2e-5)
    assert singular[0] > 0
    squared_cutoff = (CUTOFF*singular[0])**2
    assert np.min(np.abs(eigenvalues-squared_cutoff)) > floor, 'ambiguous Gram rank'
    keep = eigenvalues > squared_cutoff
    rank = int(keep.sum())
    assert rank == int(np.count_nonzero(singular_values > CUTOFF*singular_values[0]))
    basis = eigenvectors[:, keep]
    q_proposal = quiet@proposal
    removed = quiet.T@(basis@((basis.T@q_proposal)/eigenvalues[keep]))
    independent = proposal-removed
    difference = supplied-independent
    max_error = float(np.max(np.abs(difference)))
    relative_error = float(np.linalg.norm(difference)/max(np.linalg.norm(proposal), 1e-30))
    assert max_error <= 2e-8 and relative_error <= 2e-5, (max_error, relative_error)
    supplied_removed = proposal-supplied
    proposal_squared = float(proposal@proposal)
    supplied_squared = float(supplied@supplied)
    removed_squared = float(supplied_removed@supplied_removed)
    orthogonality = float(supplied_removed@supplied)
    norm_error = abs(proposal_squared-supplied_squared-removed_squared)
    identity_error = abs(float(proposal@supplied)-supplied_squared)
    identity_tolerance = 2e-10+2e-7*proposal_squared
    assert abs(orthogonality) <= identity_tolerance
    assert norm_error <= identity_tolerance and identity_error <= identity_tolerance
    q_projected = quiet@supplied
    residual = float(np.linalg.norm(q_projected))
    residual_bound = max(1e-10, 2*singular_values[0]*CUTOFF*np.sqrt(proposal_squared))
    assert residual <= residual_bound, (residual, residual_bound)
    retained_residual = float(np.linalg.norm(basis.T@q_projected))
    assert retained_residual <= 1e-10+2e-7*singular_values[0]*np.sqrt(proposal_squared)
    return dict(passed=True, method='float64 quiet Gram eigendecomposition',
        threshold=CUTOFF, rank=rank, quiet_gram=gram.tolist(),
        q_tentative=q_proposal.tolist(), q_projected=q_projected.tolist(),
        eigenvalues=eigenvalues.tolist(), eigenvalue_numerical_floor=floor,
        spectrum_max_absolute_error=float(np.abs(singular-singular_values).max()),
        projection_max_absolute_error=max_error, projection_relative_proposal_error=relative_error,
        quiet_residual_l2=residual, quiet_residual_bound=residual_bound,
        retained_quiet_residual_l2=retained_residual,
        removed_projected_inner_product=orthogonality,
        pythagorean_error=norm_error, projection_identity_error=identity_error)


def actor_hash(actor):
    raw = parameters_to_vector(actor.parameters()).detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def optimizer_hash(optimizer):
    def canonical(value):
        if isinstance(value, torch.Tensor):
            return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        return value
    encoded = json.dumps(canonical(optimizer.state_dict()), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def close(actual, expected, label, atol=1e-10, rtol=2e-7):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (label, actual, expected)


def quiet_jacobian(actor, features):
    parameters = tuple(actor.parameters())
    rows = []
    for feature in features:
        gradients = torch.autograd.grad(actor(feature[None]).sum(), parameters)
        rows.append(torch.cat([g.reshape(-1) for g in gradients]).detach().double().numpy())
    return np.stack(rows)


def check_compact_projection(record):
    """Recheck saved low-dimensional identities; full-vector agreement was gated live."""
    gram = np.asarray(record['quiet_gram'], dtype=float)
    singular = np.asarray(record['singular_values'], dtype=float)
    q_tentative = np.asarray(record['q_tentative'], dtype=float)
    q_projected = np.asarray(record['q_projected'], dtype=float)
    q_actual = np.asarray(record['q_actual'], dtype=float)
    assert gram.shape == (12, 12) and singular.shape == q_tentative.shape == q_projected.shape == q_actual.shape == (12,)
    assert all(np.isfinite(x).all() for x in (gram, singular, q_tentative, q_projected, q_actual))
    close(gram, gram.T, 'Gram symmetry', atol=1e-13)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > (CUTOFF*singular[0])**2
    assert record['rank'] == int(keep.sum())
    close(np.sqrt(np.maximum(eigenvalues[::-1], 0.)), singular, 'Gram spectrum', atol=1e-9, rtol=2e-5)
    basis = eigenvectors[:, keep]
    coordinates = basis.T@q_tentative
    expected_q = q_tentative-gram@(basis@(coordinates/eigenvalues[keep]))
    close(q_projected, expected_q, 'projected quiet product', atol=2e-10, rtol=2e-5)
    expected_removed_squared = float(np.sum(coordinates**2/eigenvalues[keep]))
    tentative = float(record['tentative_norm'])
    projected = float(record['projected_norm'])
    removed = float(record['removed_norm'])
    tolerance = 2e-10+2e-7*tentative**2
    assert all(np.isfinite(value) and value >= 0 for value in
               (tentative, projected, removed, record['actual_norm'], record['rounding_norm']))
    close(removed**2, expected_removed_squared, 'removed norm', atol=tolerance, rtol=2e-5)
    assert abs(tentative**2-projected**2-removed**2) <= tolerance
    assert abs(record['removed_dot_projected']) <= tolerance
    assert abs(record['actual_norm']-projected) <= record['rounding_norm']+1e-12
    assert np.linalg.norm(q_actual-q_projected) <= singular[0]*record['rounding_norm']+1e-10
    independent = record['independent']
    assert independent['passed'] and independent['threshold'] == CUTOFF
    assert independent['rank'] == record['rank']
    assert independent['method'] == 'float64 quiet Gram eigendecomposition'
    for key in ('quiet_gram', 'q_tentative', 'q_projected'):
        assert np.array_equal(np.asarray(record[key]), np.asarray(independent[key])), key
    assert independent['projection_max_absolute_error'] <= 2e-8
    assert independent['projection_relative_proposal_error'] <= 2e-5
    assert independent['quiet_residual_l2'] <= independent['quiet_residual_bound']
    assert abs(independent['removed_projected_inner_product']) <= tolerance
    assert independent['pythagorean_error'] <= tolerance
    assert independent['projection_identity_error'] <= tolerance


def inherited_audit():
    """Reuse sealed physics checks without publishing an incomplete audit result."""
    captured = {}
    def capture(path, content):
        assert path.name in ('audit.json', 'paired-outcomes.json')
        captured[path.name] = content
    with patch.object(inherited, 'OUT', OUT), patch.object(inherited, 'FULL', FULL), \
            patch.object(inherited, 'save', capture), redirect_stdout(io.StringIO()):
        inherited.main()
    assert set(captured) == {'audit.json', 'paired-outcomes.json'}
    for row in captured['paired-outcomes.json']:
        if row['comparator'] == 'unweighted-half-noise':
            row['comparator'] = 'age-weighted-unconstrained'
    return captured['audit.json'], captured['paired-outcomes.json']


def verify_anchor_bank():
    bank = arrays(OUT/'quiet-anchors.npz')
    assert bank['features'].shape == (12, 427) and bank['target'].shape == (12, 1)
    assert bank['features'].dtype == bank['target'].dtype == np.float32
    assert not bank['features'][:, -2:].any() and not bank['target'].any()
    assert np.array_equal(bank['onset'], np.repeat([44, 74], 6))
    assert np.array_equal(bank['pose'], np.tile(np.arange(6), 2))
    for seed in (921009, 931009):
        source = arrays(PROBES/str(seed)/'dataset.npz')
        quiet = source['probe_branch'] == 0
        for key, source_key in (('features', 'probe_x'), ('onset', 'probe_onset'), ('pose', 'probe_pose')):
            assert np.array_equal(bank[key], source[source_key][quiet]), (seed, key)
        fit = arrays(OUT/f'fit-{seed}'/'quiet-anchors.npz')
        assert np.array_equal(fit['features'], bank['features'])
        assert np.array_equal(fit['target'], bank['target'].ravel())
        actor = AnkleHead().eval()
        actor.load_state_dict(torch.load(OUT/f'fit-{seed}'/'initial.pt', weights_only=True, map_location='cpu'))
        with torch.no_grad():
            assert np.array_equal(actor(torch.from_numpy(bank['features'])).numpy(), bank['target'])
    return torch.from_numpy(bank['features']), bank['target'].ravel()


def verify_projection_history(folder, seed, features, target):
    history = read(folder/'history.json')
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    previous_actor, previous_optimizer = actor_hash(actor), optimizer_hash(optimizer)
    previous_anchor = target.copy()
    attempts, accepts, rejects = 0, 0, 0
    reasons = {'anchor': 0, 'kl': 0, 'nonfinite': 0}
    ranks, retained = [], []
    max_anchor, max_projection_error, max_kl_error = 0., 0., 0.
    assert len(list(folder.glob('projection-*.json'))) == 32
    for update, row in enumerate(history, 1):
        objective = arrays(folder/f'objective-{update}.npz')
        baseline = arrays(FULL/f'fit-{seed}'/f'objective-{update}.npz')
        assert np.array_equal(objective['minibatch_indices'], baseline['minibatch_indices']), (seed, update, 'RNG parity')
        records = read(folder/f'projection-{update}.json')
        assert len(records) == row['actor_updates']+row['rolled_back']
        assert 1 <= len(records) <= 9 and [r['minibatch_index'] for r in records] == list(range(len(records)))
        assert all(record['accepted'] for record in records[:-1])
        assert sum(not record['accepted'] for record in records) == row['rolled_back']
        assert row['accepted_kl'] == [record['kl'] for record in records if record['accepted']]
        for index, record in enumerate(records):
            assert (record['update'], record['epoch'], record['batch']) == (update, index//3, index % 3)
            assert record['actor_pre'] == previous_actor and record['optimizer_pre'] == previous_optimizer
            assert np.array_equal(record['anchor_before'], previous_anchor)
            assert record['actor_loss'] == float(objective['actor_loss'][index])
            assert np.isfinite(record['gradient_norm']) and record['gradient_norm'] >= 0
            assert np.isfinite(record['clipped_norm']) and 0 <= record['clipped_norm'] <= .500001
            before = np.asarray(record['anchor_before'])
            attempted = np.asarray(record['anchor_attempted'], dtype=float)
            after = np.asarray(record['anchor_after'])
            assert before.shape == attempted.shape == after.shape == (12,)
            assert np.isfinite(before).all() and np.isfinite(after).all()
            assert np.abs(before-target).max() <= ANCHOR_LIMIT and np.abs(after-target).max() <= ANCHOR_LIMIT
            # A nonfinite training attempt is reported as an audit failure, not silently certified.
            assert record['independent'] is not None and np.isfinite(attempted).all()
            check_compact_projection(record)
            assert record['kl'] is not None and np.isfinite(record['kl']) and record['kl'] >= 0
            close(record['anchor_max'], float(np.abs(attempted-target).max()), 'cumulative anchor guard', atol=0, rtol=0)
            expected_reasons = (['kl'] if record['kl'] > .01 else [])
            if record['anchor_max'] > ANCHOR_LIMIT:
                expected_reasons.append('anchor')
            assert record['reasons'] == expected_reasons
            assert record['accepted'] == (not expected_reasons)
            close(record['independent_kl_error'], abs(record['kl']-record['independent_kl']), 'independent KL', atol=0, rtol=0)
            assert record['independent_kl_error'] < 2e-7
            if record['accepted']:
                assert record['actor_post'] == record['actor_candidate']
                assert record['optimizer_post'] == record['optimizer_tentative']
                assert np.array_equal(after, attempted)
                accepts += 1
            else:
                assert index == len(records)-1
                assert record['actor_post'] == record['actor_pre']
                assert record['optimizer_post'] == record['optimizer_pre']
                assert np.array_equal(after, before)
                rejects += 1
                for reason in expected_reasons:
                    reasons[reason] += 1
            attempts += 1
            ranks.append(record['rank'])
            if record['tentative_norm'] > 0:
                retained.append(record['projected_norm']/record['tentative_norm'])
            max_anchor = max(max_anchor, float(np.abs(after-target).max()))
            max_projection_error = max(max_projection_error, record['independent']['projection_max_absolute_error'])
            max_kl_error = max(max_kl_error, record['independent_kl_error'])
            previous_actor, previous_optimizer = record['actor_post'], record['optimizer_post']
            previous_anchor = after
        assert row['actor_sha256'] == previous_actor and row['optimizer_sha256'] == previous_optimizer
        close(row['anchor_max'], float(np.abs(previous_anchor-target).max()), 'history anchor bound', atol=0, rtol=0)
    actor.load_state_dict(torch.load(folder/'final.pt', weights_only=True, map_location='cpu'))
    assert actor_hash(actor) == previous_actor
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), previous_anchor)
    report = read(folder/'report.json')
    assert report['updates'] == accepts
    close(report['max_independent_kl_error'], max_kl_error, 'reported independent KL', atol=0, rtol=0)
    return dict(attempts=attempts, accepted=accepts, rejected=rejects, rejection_reasons=reasons,
        rank_min=min(ranks), rank_max=max(ranks), retained_norm_min=min(retained),
        retained_norm_median=float(np.median(retained)), retained_norm_max=max(retained),
        max_accepted_cumulative_anchor_deviation=max_anchor,
        max_runtime_projection_absolute_error=max_projection_error,
        all_32_control_permutations_exact=True, actor_optimizer_hash_chain_verified=True,
        finite_projections_gated_before_parameter_application=True)


def replay_first_collection(folder, seed, features, target):
    """Reproduce the first collection's optimizer work without collecting any trajectories."""
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    torch.manual_seed(seed)
    critic = AnkleHead(1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    batch, objective = arrays(folder/'collection-1.npz'), arrays(folder/'objective-1.npz')
    records = read(folder/'projection-1.json')
    x = torch.from_numpy(batch['x']).flatten(0, 1)
    action = torch.from_numpy(batch['action']).flatten(0, 1)
    old_mean = torch.from_numpy(batch['mean']).flatten(0, 1)
    old_logp = torch.from_numpy(batch['logp']).flatten()
    advantage = torch.from_numpy(batch['normalized_advantage'])
    weights = torch.from_numpy(batch['actor_weight']).flatten()
    returns = torch.from_numpy(batch['returns'])
    with torch.no_grad():
        close(actor(x).numpy(), old_mean.numpy(), 'initial collection actor', atol=2e-7)
        close(critic(x).numpy(), batch['value'].ravel(), 'seeded initial critic', atol=2e-6)
    accepted, rejected = 0, 0
    for index, ids_array in enumerate(objective['minibatch_indices']):
        ids = torch.from_numpy(ids_array)
        if index < len(records):
            record = records[index]
            assert actor_hash(actor) == record['actor_pre']
            assert optimizer_hash(actor_optimizer) == record['optimizer_pre']
            ratio = (Normal(actor(x[ids]), SIGMA).log_prob(action[ids]).sum(-1)-old_logp[ids]).exp()
            assert np.array_equal(ratio.detach().numpy(), objective['policy_ratio'][index])
            loss = -(weights[ids]*torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids])).mean()
            assert float(loss.detach()) == record['actor_loss']
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
            close(gradient_norm, record['gradient_norm'], 'gradient norm', atol=0, rtol=0)
            clipped_norm = float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actor.parameters()])))
            close(clipped_norm, record['clipped_norm'], 'clipped gradient norm', atol=0, rtol=0)
            old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(actor_optimizer.state_dict())
            before = parameters_to_vector(actor.parameters()).detach().clone()
            quiet = quiet_jacobian(actor, features)
            _, singular, right = np.linalg.svd(quiet, full_matrices=False)
            retained = right[singular > CUTOFF*singular[0]]
            actor_optimizer.step()
            assert actor_hash(actor) == record['actor_tentative']
            assert optimizer_hash(actor_optimizer) == record['optimizer_tentative']
            displacement = parameters_to_vector(actor.parameters()).detach().double().numpy()-before.double().numpy()
            projected = displacement-retained.T@(retained@displacement)
            projection_verify(quiet, displacement, projected, singular)
            close(singular, record['singular_values'], 'replayed singular values', atol=0, rtol=0)
            close(quiet@displacement, record['q_tentative'], 'replayed actual Adam product', atol=1e-13)
            candidate = before+torch.from_numpy(projected.astype(np.float32))
            vector_to_parameters(candidate, actor.parameters())
            assert actor_hash(actor) == record['actor_candidate']
            with torch.no_grad():
                anchor_attempted = actor(features).numpy().ravel()
                mean = actor(x)
                kl = float(((mean-old_mean).square().sum(-1)/(2*SIGMA**2)).mean())
            assert np.array_equal(anchor_attempted, record['anchor_attempted'])
            close(kl, record['kl'], 'replayed candidate KL', atol=0, rtol=0)
            accepted_here = kl <= .01 and np.abs(anchor_attempted-target).max() <= ANCHOR_LIMIT
            assert accepted_here == record['accepted']
            if not accepted_here:
                actor.load_state_dict(old_actor)
                actor_optimizer.load_state_dict(old_optimizer)
                rejected += 1
            else:
                accepted += 1
            assert actor_hash(actor) == record['actor_post']
            assert optimizer_hash(actor_optimizer) == record['optimizer_post']
        residual = critic(x[ids])-returns[ids]
        assert np.array_equal(residual.detach().numpy(), objective['critic_residual'][index])
        value_loss = residual.square().mean()
        assert float(value_loss.detach()) == float(objective['critic_loss'][index])
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        critic_optimizer.step()
    with torch.no_grad():
        assert np.array_equal(actor(x).numpy(), batch['post_mean'])
        assert np.array_equal(critic(x).numpy(), batch['post_value'])
    history = read(folder/'history.json')[0]
    assert actor_hash(actor) == history['actor_sha256']
    assert optimizer_hash(actor_optimizer) == history['optimizer_sha256']
    return dict(actor_attempts=len(records), actor_accepted=accepted, actor_rejected=rejected,
        critic_minibatches=9, seeded_initial_critic_verified=True, candidate_and_optimizer_hashes_exact=True,
        post_actor_outputs_exact=True, post_critic_outputs_exact=True,
        scope='Independent optimizer replay of saved first collection; no simulated trajectories.')


def main():
    inputs, done, contract = (read(OUT/path) for path in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    resolved_inputs = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert digest(Path(__file__)) == resolved_inputs[Path(__file__).resolve()]
    historical = read(FULL/'inputs.json')
    shared = sorted(set(inputs) & set(historical))
    assert all(inputs[path] == historical[path] for path in shared)
    history_audit = read(OUT/'historical-inputs.json')
    assert history_audit['passed'] and history_audit['count'] == len(shared)
    assert history_audit['shared_paths'] == shared
    for path in Path(__file__).parent.glob('*.py'):
        assert digest(path) == digest(OUT/'sources'/path.name), path
    assert contract['training_seeds'] == [921009, 931009] and contract['initial_actor_seed'] == 920009
    assert contract['comparator'] == str(FULL) and contract['nominal_transitions_per_seed'] == 24576
    projection_contract = contract['quiet_projection']
    assert projection_contract['anchor_count'] == 12
    assert projection_contract['anchor_onsets'] == [44, 74] and projection_contract['anchor_poses'] == list(range(6))
    assert projection_contract['singular_relative_cutoff'] == CUTOFF
    assert projection_contract['anchor_max_absolute_deviation'] == ANCHOR_LIMIT
    assert projection_contract['kl_limit'] == .01 and not projection_contract['renormalized']
    assert projection_contract['rollback_actor_and_optimizer'] and projection_contract['stop_actor_for_collection_on_rejection']
    assert projection_contract['target'] == 'fresh initial head outputs'
    torch.set_num_threads(1)
    features, target = verify_anchor_bank()
    projections, replays = {}, {}
    for seed in contract['training_seeds']:
        folder = OUT/f'fit-{seed}'
        projections[str(seed)] = verify_projection_history(folder, seed, features, target)
        replays[str(seed)] = replay_first_collection(folder, seed, features, target)
    audit, pairs = inherited_audit()
    # Recheck paired identities rather than trusting positional correspondence alone.
    for row in pairs:
        name = f"{row['seed']}-t{row['timing']}-g{row['group']}"
        if row['mode'] == 'matched':
            name = 'matched-'+name
        current = read(OUT/name/'report.json')['cases'][row['case']]
        reference_root = FULL if row['comparator'] == 'age-weighted-unconstrained' else inherited.PILOT
        reference_name = name if reference_root == FULL else f"baseline-t{row['timing']}-g{row['group']}"
        previous = read(reference_root/reference_name/'report.json')['cases'][row['case']]
        for key in ('case', 'pose', 'heldout', 'force_n', 'hit_seconds'):
            assert current[key] == previous[key], (name, row['case'], key)
    audit.update(quiet_anchors=dict(count=12, source_features_exact_across_seeds=True,
        fresh_initial_targets_zero=True, action_memory_zero=True), projections=projections,
        first_collection_optimizer_replay=replays, shared_historical_inputs=len(shared),
        comparator='age-weighted-unconstrained', finite_projected_updates_independently_gated=True,
        intermediate_full_parameter_states_retained=False,
        audit_scope='All compact projection records plus exact first-collection replay and inherited full physical audit.')
    save(OUT/'paired-outcomes.json', pairs)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
