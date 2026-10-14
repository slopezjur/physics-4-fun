"""Independent audit of the fixed 1e-3 actor-rate contrast."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from . import verify_retracted_recovery as original
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_retracted_recovery import (
    read, arrays, close, actor_hash, optimizer_hash, quiet_jacobian,
    projection_verify, check_compact_projection, check_retraction, linear_verify, compare,
)

OUT = Path('logs/mimickit-training/recovery-retracted-actor-rate-20261009-01')
FULL = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
PROJECTED = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
WEIGHTED = original.WEIGHTED
PROBES = original.PROBES
PILOT = original.inherited.PILOT
SEEDS = original.SEEDS
SIGMA = original.SIGMA
CUTOFF = original.CUTOFF
ANCHOR_LIMIT = original.ANCHOR_LIMIT
KL_LIMIT = original.KL_LIMIT
ACTOR_LR = 1e-3
CRITIC_LR = 3e-4
DIAGNOSTIC = Path(__file__).with_name('probe_recovery_exploration.py')
ARCHIVED = PILOT/'sources'/'probe_recovery_exploration.py'
HISTORICAL_SHA256 = 'd3bd741568548f16a059a5f2b236161715c71f74c74563a5c48ea76370dd50d5'
CURRENT_SHA256 = '2d0441a09deeaaabcf5cb9096cc01a2be1c2a3234fa91f359a8b5e46adec1a8e'

# These two replay/accounting routines retain the sealed audit's checks, with an
# explicitly constructed 1e-3 actor optimizer and unchanged 3e-4 critic optimizer.


def verify_history(folder, seed, features, target):
    history = read(folder/'history.json')
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    previous_actor, previous_optimizer = actor_hash(actor), optimizer_hash(optimizer)
    previous_anchor = target.copy()
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), target)
    attempts, accepted, rejected, intermediate_violations = 0, 0, 0, 0
    reasons = {'anchor': 0, 'kl': 0, 'nonfinite': 0}
    ranks, correction_ratios = [], []
    maxima = dict(anchor=0., local_residual=0., projection_error=0., correction_error=0., independent_kl_error=0.)
    assert len(history) == 32 and len(list(folder.glob('projection-*.json'))) == 32
    for update, row in enumerate(history, 1):
        objective = arrays(folder/f'objective-{update}.npz')
        for control in (FULL, WEIGHTED):
            previous = arrays(control/f'fit-{seed}'/f'objective-{update}.npz')
            assert np.array_equal(objective['minibatch_indices'], previous['minibatch_indices']), (seed, update, control, 'RNG parity')
        records = read(folder/f'projection-{update}.json')
        assert len(records) == row['actor_updates']+row['rolled_back']
        assert 1 <= len(records) <= 9 and [r['minibatch_index'] for r in records] == list(range(len(records)))
        assert all(record['accepted'] for record in records[:-1])
        assert sum(not record['accepted'] for record in records) == row['rolled_back']
        assert row['accepted_kl'] == [record['kl'] for record in records if record['accepted']]
        for index, record in enumerate(records):
            assert record['actor_learning_rate'] == ACTOR_LR
            assert (record['update'], record['epoch'], record['batch']) == (update, index//3, index % 3)
            assert record['actor_pre'] == previous_actor and record['optimizer_pre'] == previous_optimizer
            assert np.array_equal(record['anchor_before'], previous_anchor)
            assert record['actor_loss'] == float(objective['actor_loss'][index])
            assert np.isfinite(record['gradient_norm']) and record['gradient_norm'] >= 0
            assert np.isfinite(record['clipped_norm']) and 0 <= record['clipped_norm'] <= .500001
            # Nonfinite numerical attempts are not certified through a missing linear check.
            assert record['projected_finite'] and record['retraction_applied'] and record['nonfinite_stage'] is None
            check_compact_projection(record)
            correction = check_retraction(record)
            assert correction['threshold'] == CUTOFF
            close(correction['correction_to_projected_norm'], correction['correction_norm']/max(record['actual_norm'], 1e-30), 'correction ratio')
            assert abs(correction['total_step_norm']-record['actual_norm']) <= correction['actual_norm']+1e-12
            before = np.asarray(record['anchor_before'], dtype=float)
            projected = np.asarray(record['anchor_projected'], dtype=float)
            final = np.asarray(record['anchor_attempted'], dtype=float)
            after = np.asarray(record['anchor_after'], dtype=float)
            assert all(value.shape == (12,) and np.isfinite(value).all() for value in (before, projected, final, after))
            assert np.abs(before-target).max() <= ANCHOR_LIMIT and np.abs(after-target).max() <= ANCHOR_LIMIT
            close(record['projected_anchor_max'], float(np.abs(projected-target).max()), 'projected zero-target bound', atol=0, rtol=0)
            close(record['anchor_max'], float(np.abs(final-target).max()), 'final zero-target bound', atol=0, rtol=0)
            assert np.isfinite(record['projected_kl']) and record['projected_kl'] >= 0
            assert np.isfinite(record['kl']) and record['kl'] >= 0
            expected_reasons = ['kl'] if record['kl'] > KL_LIMIT else []
            if record['anchor_max'] > ANCHOR_LIMIT:
                expected_reasons.append('anchor')
            assert record['reasons'] == expected_reasons and record['accepted'] == (not expected_reasons)
            close(record['independent_kl_error'], abs(record['kl']-record['independent_kl']), 'KL error', atol=0, rtol=0)
            assert record['independent_kl_error'] < 2e-7
            if record['accepted']:
                assert record['actor_post'] == record['actor_candidate']
                assert record['optimizer_post'] == record['optimizer_tentative']
                assert np.array_equal(after, final)
                accepted += 1
            else:
                assert index == len(records)-1
                assert record['actor_post'] == record['actor_pre']
                assert record['optimizer_post'] == record['optimizer_pre']
                assert np.array_equal(after, before)
                rejected += 1
                for reason in expected_reasons:
                    reasons[reason] += 1
            attempts += 1
            intermediate_violations += int(record['projected_anchor_max'] > ANCHOR_LIMIT or record['projected_kl'] > KL_LIMIT)
            ranks.append((record['rank'], correction['rank']))
            correction_ratios.append(correction['correction_to_projected_norm'])
            maxima['anchor'] = max(maxima['anchor'], float(np.abs(after-target).max()))
            maxima['local_residual'] = max(maxima['local_residual'], correction['local_residual_max'])
            maxima['projection_error'] = max(maxima['projection_error'], record['independent']['projection_max_absolute_error'])
            maxima['correction_error'] = max(maxima['correction_error'], correction['independent']['correction_max_absolute_error'])
            maxima['independent_kl_error'] = max(maxima['independent_kl_error'], record['independent_kl_error'])
            previous_actor, previous_optimizer, previous_anchor = record['actor_post'], record['optimizer_post'], after
        assert row['actor_sha256'] == previous_actor and row['optimizer_sha256'] == previous_optimizer
        close(row['anchor_max'], float(np.abs(previous_anchor-target).max()), 'history anchor bound', atol=0, rtol=0)
    actor.load_state_dict(torch.load(folder/'final.pt', weights_only=True, map_location='cpu'))
    assert actor_hash(actor) == previous_actor
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), previous_anchor)
    report = read(folder/'report.json')
    assert report['updates'] == accepted
    assert report['actor_learning_rate'] == ACTOR_LR and report['critic_learning_rate'] == CRITIC_LR
    assert report['nominal_actor_slots'] == 288 and report['effective_rate_set_before_first_update']
    assert report['initial_constructor_actor_rate'] == 1e-4
    assert report['actor_rate_attempts'] == attempts and report['actor_optimizer_initial_state_empty']
    assert report['actor_rate_constant_across_attempts'] and not report['global_adam_constructor_patched']
    close(report['max_independent_kl_error'], maxima['independent_kl_error'], 'reported independent KL', atol=0, rtol=0)
    protocol = report['retraction']
    assert protocol['target'] == 'pre-step quiet outputs' and protocol['acceptance_target'] == 'original zero outputs'
    assert protocol['corrections_per_finite_projected_candidate'] == 1 and protocol['singular_relative_cutoff'] == CUTOFF
    assert protocol['intermediate_anchor_violation_rejected'] is False
    assert protocol['retries'] == 0 and not protocol['line_search'] and not protocol['correction_rescaled']
    assert not protocol['surrogate_acceptance_gate'] and protocol['tentative_adam_retained_only_on_acceptance']
    return dict(attempts=attempts, accepted=accepted, rejected=rejected, rejection_reasons=reasons,
        corrections=attempts, intermediate_guard_violations=intermediate_violations,
        projection_ranks=sorted(set(rank[0] for rank in ranks)), correction_ranks=sorted(set(rank[1] for rank in ranks)),
        correction_to_projected_norm=dict(minimum=min(correction_ratios), median=float(np.median(correction_ratios)), maximum=max(correction_ratios)),
        maxima=maxima, all_32_permutations_match_both_controls=True,
        actor_optimizer_hash_chain_verified=True, final_original_zero_guard_verified=True,
        both_linear_operations_independently_gated=True)


def replay_first_collection(folder, seed, features, target):
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
    torch.manual_seed(seed)
    critic = AnkleHead(1)
    ao, co = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR), torch.optim.Adam(critic.parameters(), lr=3e-4)
    batch, objective = arrays(folder/'collection-1.npz'), arrays(folder/'objective-1.npz')
    for control in (FULL, WEIGHTED):
        reference = arrays(control/f'fit-{seed}'/'collection-1.npz')
        for key, value in reference.items():
            if key != 'post_mean':
                assert np.array_equal(batch[key], value), (seed, control, key)
    records = read(folder/'projection-1.json')
    x = torch.from_numpy(batch['x']).flatten(0, 1)
    action = torch.from_numpy(batch['action']).flatten(0, 1)
    old_mean = torch.from_numpy(batch['mean']).flatten(0, 1)
    old_logp = torch.from_numpy(batch['logp']).flatten()
    advantage = torch.from_numpy(batch['normalized_advantage'])
    weights = torch.from_numpy(batch['actor_weight']).flatten()
    returns = torch.from_numpy(batch['returns'])
    with torch.no_grad():
        close(actor(x).numpy(), old_mean.numpy(), 'initial actor means', atol=2e-7)
        close(critic(x).numpy(), batch['value'].ravel(), 'seeded critic', atol=2e-6)
    accepted, rejected = 0, 0
    for index, ids_array in enumerate(objective['minibatch_indices']):
        ids = torch.from_numpy(ids_array)
        if index < len(records):
            record = records[index]
            assert record['actor_learning_rate'] == ACTOR_LR
            assert actor_hash(actor) == record['actor_pre'] and optimizer_hash(ao) == record['optimizer_pre']
            ratio = (Normal(actor(x[ids]), SIGMA).log_prob(action[ids]).sum(-1)-old_logp[ids]).exp()
            assert np.array_equal(ratio.detach().numpy(), objective['policy_ratio'][index])
            loss = -(weights[ids]*torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids])).mean()
            assert float(loss.detach()) == record['actor_loss']
            ao.zero_grad(set_to_none=True)
            loss.backward()
            assert float(nn.utils.clip_grad_norm_(actor.parameters(), .5)) == record['gradient_norm']
            assert float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actor.parameters()]))) == record['clipped_norm']
            old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(ao.state_dict())
            before = parameters_to_vector(actor.parameters()).detach().clone()
            with torch.no_grad():
                anchor_before = actor(features).numpy().ravel().copy()
            assert np.array_equal(anchor_before, record['anchor_before'])
            quiet = quiet_jacobian(actor, features)
            _, singular, right = np.linalg.svd(quiet, full_matrices=False)
            basis = right[singular > CUTOFF*singular[0]]
            ao.step()
            assert actor_hash(actor) == record['actor_tentative'] and optimizer_hash(ao) == record['optimizer_tentative']
            tentative = parameters_to_vector(actor.parameters()).detach().double().numpy()-before.double().numpy()
            projected = tentative-basis.T@(basis@tentative)
            projection_verify(quiet, tentative, projected, singular)
            assert np.array_equal(singular, record['singular_values'])
            projected_parameters = before+torch.from_numpy(projected.astype(np.float32))
            vector_to_parameters(projected_parameters, actor.parameters())
            assert actor_hash(actor) == record['actor_projected']
            with torch.no_grad():
                projected_anchor = actor(features).numpy().ravel().copy()
                projected_kl = float(((actor(x)-old_mean).square().sum(-1)/(2*SIGMA**2)).mean())
            assert np.array_equal(projected_anchor, record['anchor_projected'])
            assert projected_kl == record['projected_kl']
            candidate_quiet = quiet_jacobian(actor, features)
            residual = projected_anchor.astype(float)-anchor_before.astype(float)
            left, correction_singular, correction_right = np.linalg.svd(candidate_quiet, full_matrices=False)
            keep = correction_singular > CUTOFF*correction_singular[0]
            correction = -correction_right[keep].T@((left[:, keep].T@residual)/correction_singular[keep])
            independent = linear_verify(candidate_quiet, residual, correction, correction_singular)
            item = record['retraction']
            compare(item['independent'], independent, 'replayed correction gate')
            assert np.array_equal(candidate_quiet@candidate_quiet.T, item['quiet_gram'])
            assert np.array_equal(correction_singular, item['singular_values'])
            assert np.array_equal(residual, item['residual'])
            close(candidate_quiet@correction, item['q_correction'], 'replayed correction product', atol=1e-13)
            corrected_parameters = projected_parameters+torch.from_numpy(correction.astype(np.float32))
            vector_to_parameters(corrected_parameters, actor.parameters())
            assert actor_hash(actor) == record['actor_candidate']
            assert optimizer_hash(ao) == record['optimizer_tentative']
            with torch.no_grad():
                anchor = actor(features).numpy().ravel()
                kl = float(((actor(x)-old_mean).square().sum(-1)/(2*SIGMA**2)).mean())
            assert np.array_equal(anchor, record['anchor_attempted']) and kl == record['kl']
            accepted_here = kl <= KL_LIMIT and np.abs(anchor-target).max() <= ANCHOR_LIMIT
            assert accepted_here == record['accepted']
            if not accepted_here:
                actor.load_state_dict(old_actor)
                ao.load_state_dict(old_optimizer)
                rejected += 1
            else:
                accepted += 1
            assert actor_hash(actor) == record['actor_post'] and optimizer_hash(ao) == record['optimizer_post']
        residual = critic(x[ids])-returns[ids]
        assert np.array_equal(residual.detach().numpy(), objective['critic_residual'][index])
        value_loss = residual.square().mean()
        assert float(value_loss.detach()) == float(objective['critic_loss'][index])
        co.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        co.step()
    with torch.no_grad():
        assert np.array_equal(actor(x).numpy(), batch['post_mean'])
        assert np.array_equal(critic(x).numpy(), batch['post_value'])
    history = read(folder/'history.json')[0]
    assert actor_hash(actor) == history['actor_sha256'] and optimizer_hash(ao) == history['optimizer_sha256']
    return dict(actor_attempts=len(records), accepted=accepted, rejected=rejected, critic_minibatches=9,
        initial_collection_and_post_critic_match_both_controls=True,
        intermediate_final_actor_and_optimizer_hashes_exact=True,
        candidate_jacobians_and_minimum_norm_corrections_verified=True,
        post_actor_outputs_exact=True, post_critic_outputs_exact=True)


def provenance():
    inputs, done = read(OUT/'inputs.json'), read(OUT/'completed.json')
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    resolved = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert digest(Path(__file__)) == resolved[Path(__file__).resolve()]
    assert resolved[DIAGNOSTIC.resolve()] == CURRENT_SHA256
    assert resolved[ARCHIVED.resolve()] == HISTORICAL_SHA256 and digest(ARCHIVED) == HISTORICAL_SHA256
    for path, sha in inputs.items():
        path = Path(path)
        if path.suffix == '.py' and path.resolve().parent == Path(__file__).resolve().parent:
            assert digest(OUT/'sources'/path.name) == sha
    historical, saved = {}, read(OUT/'historical-inputs.json')
    for root in (FULL, PROJECTED, WEIGHTED, PILOT):
        assert read(root/'audit.json')['passed']
        outputs = read(root/'completed.json')['outputs']
        for path, sha in outputs.items():
            assert resolved[(root/path).resolve()] == sha, (root, path)
        previous = read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared
        divergences = []
        for path in shared:
            if inputs[path] == previous[path]:
                continue
            assert root == PILOT and Path(path).resolve() == DIAGNOSTIC.resolve(), (root, path)
            assert previous[path] == HISTORICAL_SHA256 and inputs[path] == CURRENT_SHA256
            divergences.append(dict(path=path, historical_sha256=HISTORICAL_SHA256,
                current_sha256=CURRENT_SHA256, archived_source=str(ARCHIVED)))
        assert len(divergences) == (1 if root == PILOT else 0)
        assert saved[str(root)] == dict(passed=True, count=len(shared), shared_paths=shared, source_divergences=divergences)
        historical[str(root)] = dict(shared_inputs=len(shared), sealed_outputs=len(outputs), source_divergences=divergences)
    assert set(saved) == set(historical)
    for seed in SEEDS:
        path = PROBES/str(seed)/'dataset.npz'
        manifest = read(PROBES/'completed.json')['outputs']
        assert resolved[path.resolve()] == manifest[str(path.relative_to(PROBES))]
    return inputs, done, historical


def audit_run():
    inputs, done, historical = provenance()
    contract = read(OUT/'contract.json')
    assert contract['training_seeds'] == list(SEEDS) and contract['initial_actor_seed'] == 920009
    assert contract['nominal_transitions_per_seed'] == 24576 and contract['comparator'] == str(FULL)
    assert contract['additional_comparators'] == [str(PROJECTED), str(WEIGHTED), str(PILOT)]
    assert contract['actor_learning_rate'] == ACTOR_LR and contract['control_actor_learning_rate'] == 1e-4
    assert contract['critic_learning_rate'] == CRITIC_LR
    assert contract['sigma'] == SIGMA and contract['baseline_sigma'] == SIGMA
    projection = contract['quiet_projection']
    assert projection['anchor_count'] == 12 and projection['anchor_onsets'] == [44, 74]
    assert projection['anchor_poses'] == list(range(6)) and projection['target'] == 'fresh initial head outputs'
    assert projection['singular_relative_cutoff'] == CUTOFF
    assert projection['anchor_max_absolute_deviation'] == ANCHOR_LIMIT and projection['kl_limit'] == KL_LIMIT
    assert not projection['renormalized'] and projection['rollback_actor_and_optimizer']
    assert projection['stop_actor_for_collection_on_rejection']
    correction = contract['finite_correction']
    assert correction['target'] == 'actual pre-step quiet outputs' and correction['global_target'] == 'fresh initial zero outputs'
    assert correction['candidate_jacobian'] and correction['every_finite_projected_candidate']
    assert correction['corrections_per_attempt'] == 1 and correction['singular_relative_cutoff'] == CUTOFF
    assert correction['float32_displacement_addition'] and not correction['intermediate_anchor_rejection']
    assert correction['final_guard_only'] and correction['independent_linear_gate']
    assert correction['retries'] == 0 and not correction['line_search']
    assert not correction['renormalized'] and not correction['surrogate_acceptance_gate']
    assert contract['shared_historical_inputs'] == {root: row['shared_inputs'] for root, row in historical.items()}
    torch.set_num_threads(1)
    features, target = original.verify_anchor_bank()
    updates, replays = {}, {}
    for seed in SEEDS:
        folder = OUT/f'fit-{seed}'
        updates[str(seed)] = verify_history(folder, seed, features, target)
        replays[str(seed)] = replay_first_collection(folder, seed, features, target)
    audit, pairs = original.inherited_audit()
    for row in pairs:
        name = f"{row['seed']}-t{row['timing']}-g{row['group']}"
        if row['mode'] == 'matched':
            name = 'matched-'+name
        current = read(OUT/name/'report.json')['cases'][row['case']]
        reference_root = FULL if row['comparator'] == 'projected-only' else PILOT
        reference_name = name if reference_root == FULL else f"baseline-t{row['timing']}-g{row['group']}"
        previous = read(reference_root/reference_name/'report.json')['cases'][row['case']]
        for key in ('case', 'pose', 'heldout', 'force_n', 'hit_seconds'):
            assert current[key] == previous[key], (name, row['case'], key)
        if row['comparator'] == 'projected-only':
            row['comparator'] = 'retracted-actor-lr-1e-4'
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    audit.update(comparator='retracted-actor-lr-1e-4', historical=historical,
        actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR,
        actor_optimizer_independently_initialized_at_effective_rate=True,
        projected_retracted_updates=updates, first_collection_optimizer_replay=replays,
        quiet_anchors=dict(count=12, source_features_exact_across_seeds=True,
            fresh_initial_targets_zero=True, action_memory_zero=True),
        intermediate_full_parameter_states_retained=False,
        audit_scope='Effective actor-rate/hash audit, independently gated compact geometry, exact first actor/Adam/critic replay, and inherited physical/objective audit.')
    save(OUT/'paired-outcomes.json', pairs)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


def main():
    with patch.object(original, 'OUT', OUT), patch.object(original, 'FULL', FULL):
        audit_run()


if __name__ == '__main__':
    main()

