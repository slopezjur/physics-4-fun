"""Independent training, quiet-coverage and frozen physical audit for 48 constraints."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import verify_retracted_recovery as replay
from . import verify_age_weighted_recovery_v2 as physical
from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_constrained_recovery import actor_hash, optimizer_hash
from .verify_quiet_anchor_retraction import compare
from .verify_quiet_coverage import projection_verify, correction_verify
from .verify_command_generalization import verify_probe_identity, probe_group, classifications

OUT = Path('logs/mimickit-training/recovery-coverage-learning-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
COVERAGE = Path('logs/mimickit-training/recovery-quiet-coverage-20261009-01')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
PILOT = physical.PILOT
SEEDS = (921009, 931009)
ONSETS = (40, 44, 48, 54, 66, 74, 82, 90)
SIGMA, CUTOFF, ANCHOR_LIMIT, KL_LIMIT = .08717797887081347, 1e-6, 1e-5, .01
METHOD = 'float64 thin QR plus reduced-transpose SVD'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, label, atol=1e-10, rtol=2e-7):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), label


def spectrum_record(item):
    """Check compact spectra without taking square roots of noisy Gram eigenvalues."""
    gram, singular = np.asarray(item['quiet_gram'], float), np.asarray(item['singular_values'], float)
    assert gram.shape == (48, 48) and singular.shape == (48,)
    assert np.isfinite(gram).all() and np.isfinite(singular).all() and singular[0] > 0
    assert np.all(singular >= 0) and np.all(np.diff(singular) <= 0)
    close(gram, gram.T, 'Gram symmetry', 1e-13)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    close(eigenvalues, singular**2, 'Gram squared spectrum', 1e-12, 2e-7)
    assert eigenvalues[-1] >= -128*np.finfo(float).eps*max(1., eigenvalues[0])
    rank = int(np.count_nonzero(singular > CUTOFF*singular[0]))
    assert item['rank'] == rank
    independent = item['independent']
    assert independent['passed'] and independent['method'] == METHOD and independent['threshold'] == CUTOFF
    alternative = np.asarray(independent['singular_values'], float)
    assert alternative.shape == (48,) and np.isfinite(alternative).all()
    close(alternative, singular, 'independent QR spectrum', 1e-10, 2e-7)
    assert independent['rank'] == rank == int(np.count_nonzero(alternative > CUTOFF*alternative[0]))
    floor = 64*np.finfo(float).eps*max(1., float(alternative[0]))
    margin = float(np.min(np.abs(alternative-CUTOFF*alternative[0])))
    close(independent['rank_numerical_floor'], floor, 'rank floor', 1e-20)
    close(independent['rank_margin'], margin, 'rank margin', 1e-15)
    assert margin > floor and independent['spectrum_max_absolute_error'] <= 1e-10+2e-7*singular[0]
    return gram, singular, independent


def check_projection(record):
    gram, singular, independent = spectrum_record(record)
    vectors = {key: np.asarray(record[key], float) for key in ('q_tentative', 'q_projected', 'q_actual')}
    assert all(value.shape == (48,) and np.isfinite(value).all() for value in vectors.values())
    close(independent['quiet_gram'], gram, 'runtime Gram', 0, 0)
    for key in ('q_tentative', 'q_projected'):
        close(independent[key], vectors[key], key, 1e-13)
    names = ('tentative_norm', 'projected_norm', 'removed_norm', 'actual_norm', 'rounding_norm')
    assert all(np.isfinite(record[key]) and record[key] >= 0 for key in names)
    proposed, projected, removed, actual, rounding = (float(record[key]) for key in names)
    tolerance = 2e-10+2e-7*proposed**2
    assert abs(record['removed_dot_projected']) <= tolerance
    assert abs(proposed**2-projected**2-removed**2) <= tolerance
    assert projected <= proposed+1e-10 and abs(actual-projected) <= rounding+1e-12
    assert np.linalg.norm(vectors['q_actual']-vectors['q_projected']) <= singular[0]*rounding+1e-10
    assert np.linalg.norm(vectors['q_tentative']) <= singular[0]*proposed+1e-10
    bound = max(1e-10, 2*singular[0]*CUTOFF*proposed)
    assert np.linalg.norm(vectors['q_projected']) <= bound
    close(independent['quiet_residual_bound'], bound, 'projection residual bound')
    close(independent['quiet_residual_l2'], np.linalg.norm(vectors['q_projected']), 'projection residual')
    assert independent['projection_max_absolute_error'] <= 2e-8
    assert independent['projection_relative_proposal_error'] <= 2e-5
    assert independent['retained_quiet_residual_l2'] <= 1e-10+2e-7*singular[0]*proposed
    assert abs(independent['removed_projected_inner_product']) <= tolerance
    assert independent['pythagorean_error'] <= tolerance and independent['projection_identity_error'] <= tolerance


def check_correction(record):
    item = record['retraction']
    assert item is not None and item['threshold'] == CUTOFF
    _, singular, independent = spectrum_record(item)
    target, residual, q, actual_q = (np.asarray(item[key], float) for key in ('target', 'residual', 'q_correction', 'q_actual'))
    before, projected, final = (np.asarray(record[key], float) for key in ('anchor_before', 'anchor_projected', 'anchor_attempted'))
    assert all(value.shape == (48,) and np.isfinite(value).all() for value in (target, residual, q, actual_q, before, projected, final))
    assert np.array_equal(target, before) and np.array_equal(residual, projected-before)
    norm, actual_norm, rounding = (float(item[key]) for key in ('correction_norm', 'actual_norm', 'rounding_norm'))
    assert all(np.isfinite(value) and value >= 0 for value in (norm, actual_norm, rounding, item['total_step_norm']))
    assert abs(actual_norm-norm) <= rounding+1e-12
    assert np.linalg.norm(actual_q-q) <= singular[0]*rounding+1e-10
    assert np.linalg.norm(q) <= singular[0]*norm+1e-10
    local = final-before
    assert np.array_equal(item['local_residual'], local)
    close(item['local_residual_max'], np.abs(local).max(), 'local residual', 0, 0)
    close(item['linear_residual_max'], np.abs(residual+q).max(), 'linear residual', 1e-13)
    close(item['rounded_linear_residual_max'], np.abs(residual+actual_q).max(), 'rounded residual', 1e-13)
    nonlinear = final-projected-actual_q
    close(item['nonlinear_rms'], np.sqrt(np.mean(nonlinear**2)), 'nonlinear RMS', 1e-13)
    close(item['nonlinear_max'], np.abs(nonlinear).max(), 'nonlinear max', 1e-13)
    close(item['correction_to_projected_norm'], norm/max(record['actual_norm'], 1e-30), 'correction ratio')
    assert abs(item['total_step_norm']-record['actual_norm']) <= actual_norm+1e-12
    assert independent['correction_max_absolute_error'] <= 2e-8 and independent['correction_relative_l2_error'] <= 2e-5
    assert independent['rowspace_error'] <= 2e-8+2e-5*norm
    assert independent['minimum_norm_squared_error'] <= 2e-10+2e-7*norm**2
    close(independent['linear_residual_l2'], np.linalg.norm(residual+q), 'QR residual', 1e-13)
    residual_tolerance = 1e-10+2e-7*np.linalg.norm(residual)
    assert independent['linear_residual_error'] <= residual_tolerance
    assert abs(independent['linear_residual_l2']-independent['expected_truncated_residual_l2']) <= residual_tolerance
    return item


def verify_bank():
    bank, source = arrays(OUT/'quiet-anchors.npz'), arrays(COVERAGE/'bank.npz')
    assert bank.keys() == source.keys() and all(np.array_equal(bank[key], value) for key, value in source.items())
    assert bank['features'].shape == (48, 427) and bank['features'].dtype == np.float32
    assert bank['target'].shape == (48, 1) and bank['target'].dtype == np.float32 and not bank['target'].any()
    assert not bank['features'][:, -2:].any() and np.isfinite(bank['features']).all()
    assert np.array_equal(bank['onset'], np.repeat(ONSETS, 6))
    assert np.array_equal(bank['pose'], np.tile(np.arange(6), 8))
    assert np.array_equal(bank['world'], bank['pose']) and np.array_equal(bank['issue'], bank['onset']+1)
    assert np.array_equal(bank['original_indices'], np.r_[6:12, 30:36])
    old = arrays(LOW/'quiet-anchors.npz')
    for key in ('features', 'target', 'onset', 'pose'):
        assert np.array_equal(bank[key][bank['original_indices']], old[key])
    for seed in SEEDS:
        fit = arrays(OUT/f'fit-{seed}'/'quiet-anchors.npz')
        assert np.array_equal(fit['features'], bank['features']) and np.array_equal(fit['target'], bank['target'].ravel())
    return torch.from_numpy(bank['features']), bank['target'].ravel()


def verify_history(folder, seed, features, target):
    history = read(folder/'history.json')
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', map_location='cpu', weights_only=True))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    previous_actor, previous_optimizer, previous_anchor = actor_hash(actor), optimizer_hash(optimizer), target.copy()
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), target)
    assert len(history) == 32 and len(list(folder.glob('projection-*.json'))) == 32
    attempts = accepted = rejected = skipped = intermediate_violations = 0
    reasons = {'anchor': 0, 'kl': 0, 'nonfinite': 0}
    ranks, retained, corrections, kls = [], [], [], []
    maxima = dict(anchor=0., local_residual=0., projection_error=0., correction_error=0., independent_kl_error=0.)
    for update, row in enumerate(history, 1):
        objective = arrays(folder/f'objective-{update}.npz')
        control = arrays(LOW/f'fit-{seed}'/f'objective-{update}.npz')
        assert np.array_equal(objective['minibatch_indices'], control['minibatch_indices']), (seed, update, 'RNG parity')
        records = read(folder/f'projection-{update}.json')
        assert len(records) == row['actor_updates']+row['rolled_back']
        assert 1 <= len(records) <= 9 and [item['minibatch_index'] for item in records] == list(range(len(records)))
        assert all(item['accepted'] for item in records[:-1])
        assert sum(not item['accepted'] for item in records) == row['rolled_back']
        assert row['accepted_kl'] == [item['kl'] for item in records if item['accepted']]
        skipped += 9-len(records)
        for index, record in enumerate(records):
            assert (record['update'], record['epoch'], record['batch']) == (update, index//3, index % 3)
            assert record['actor_pre'] == previous_actor and record['optimizer_pre'] == previous_optimizer
            assert np.array_equal(record['anchor_before'], previous_anchor)
            assert record['actor_loss'] == float(objective['actor_loss'][index])
            assert np.isfinite(record['gradient_norm']) and record['gradient_norm'] >= 0
            assert np.isfinite(record['clipped_norm']) and 0 <= record['clipped_norm'] <= .500001
            assert record['projected_finite'] and record['retraction_applied'] and record['nonfinite_stage'] is None
            check_projection(record)
            correction = check_correction(record)
            before, projected, final, after = (np.asarray(record[key], float) for key in
                                               ('anchor_before', 'anchor_projected', 'anchor_attempted', 'anchor_after'))
            assert all(value.shape == (48,) and np.isfinite(value).all() for value in (before, projected, final, after))
            assert np.abs(before-target).max() <= ANCHOR_LIMIT and np.abs(after-target).max() <= ANCHOR_LIMIT
            close(record['projected_anchor_max'], np.abs(projected-target).max(), 'projected zero bound', 0, 0)
            close(record['anchor_max'], np.abs(final-target).max(), 'final zero bound', 0, 0)
            assert np.isfinite(record['projected_kl']) and record['projected_kl'] >= 0
            assert np.isfinite(record['kl']) and record['kl'] >= 0
            expected_reasons = ['kl'] if record['kl'] > KL_LIMIT else []
            if record['anchor_max'] > ANCHOR_LIMIT:
                expected_reasons.append('anchor')
            assert record['reasons'] == expected_reasons and record['accepted'] == (not expected_reasons)
            close(record['independent_kl_error'], abs(record['kl']-record['independent_kl']), 'KL error', 0, 0)
            assert record['independent_kl_error'] < 2e-7
            if record['accepted']:
                assert record['actor_post'] == record['actor_candidate'] and record['optimizer_post'] == record['optimizer_tentative']
                assert np.array_equal(after, final)
                accepted += 1
            else:
                assert index == len(records)-1
                assert record['actor_post'] == record['actor_pre'] and record['optimizer_post'] == record['optimizer_pre']
                assert np.array_equal(after, before)
                rejected += 1
                for reason in expected_reasons:
                    reasons[reason] += 1
            attempts += 1
            intermediate_violations += int(record['projected_anchor_max'] > ANCHOR_LIMIT or record['projected_kl'] > KL_LIMIT)
            ranks.append((record['rank'], correction['rank']))
            retained.append(record['projected_norm']/max(record['tentative_norm'], 1e-30))
            corrections.append(correction['correction_to_projected_norm'])
            kls.append(record['kl'])
            maxima['anchor'] = max(maxima['anchor'], float(np.abs(after-target).max()))
            maxima['local_residual'] = max(maxima['local_residual'], correction['local_residual_max'])
            maxima['projection_error'] = max(maxima['projection_error'], record['independent']['projection_max_absolute_error'])
            maxima['correction_error'] = max(maxima['correction_error'], correction['independent']['correction_max_absolute_error'])
            maxima['independent_kl_error'] = max(maxima['independent_kl_error'], record['independent_kl_error'])
            previous_actor, previous_optimizer, previous_anchor = record['actor_post'], record['optimizer_post'], after
        assert row['actor_sha256'] == previous_actor and row['optimizer_sha256'] == previous_optimizer
        close(row['anchor_max'], np.abs(previous_anchor-target).max(), 'history bound', 0, 0)
    assert attempts+skipped == 288 and attempts == accepted+rejected
    actor.load_state_dict(torch.load(folder/'final.pt', map_location='cpu', weights_only=True))
    assert actor_hash(actor) == previous_actor
    with torch.no_grad():
        assert np.array_equal(actor(features).numpy().ravel(), previous_anchor)
    report = read(folder/'report.json')
    assert report['updates'] == accepted and report['transitions'] == 24576
    assert report['actor_learning_rate'] == 1e-4 and report['critic_learning_rate'] == 3e-4
    assert report['nominal_actor_slots'] == 288 and report['quiet_anchor_count'] == 48
    assert report['quiet_onsets'] == list(ONSETS) and report['original_quiet_indices'] == np.r_[6:12, 30:36].tolist()
    assert report['independent_projection_and_correction'] == 'thin QR(J.T), then SVD(R.T)'
    assert report['comparison_root'] == str(LOW)
    assert report['initial_collection_and_critic_predictions_exact'] and report['first_post_critic_predictions_exact']
    assert report['all_collection_minibatch_permutations_exact']
    assert report['projection_rtol'] == CUTOFF and report['anchor_limit'] == ANCHOR_LIMIT and report['kl_limit'] == KL_LIMIT
    assert not report['projected_direction_renormalized'] and report['optimizer_restored_on_rejection']
    assert report['anchor_target_recomputed_from_initial']
    close(report['max_independent_kl_error'], maxima['independent_kl_error'], 'reported KL error', 0, 0)
    protocol = report['retraction']
    assert protocol['target'] == 'pre-step quiet outputs' and protocol['acceptance_target'] == 'original zero outputs'
    assert protocol['corrections_per_finite_projected_candidate'] == 1 and protocol['singular_relative_cutoff'] == CUTOFF
    assert not protocol['intermediate_anchor_violation_rejected'] and protocol['retries'] == 0 and not protocol['line_search']
    assert not protocol['correction_rescaled'] and not protocol['surrogate_acceptance_gate']
    assert protocol['tentative_adam_retained_only_on_acceptance'] and protocol['actor_step'] == 'retracted_ppo.retracted_step'
    summarize = lambda values: dict(minimum=min(values), median=float(np.median(values)), maximum=max(values))
    return dict(attempts=attempts, accepted=accepted, rejected=rejected, skipped_actor_minibatches=skipped,
        rejection_reasons=reasons, corrections=attempts, intermediate_guard_violations=intermediate_violations,
        projection_ranks=sorted(set(pair[0] for pair in ranks)), correction_ranks=sorted(set(pair[1] for pair in ranks)),
        retained_norm_fraction=summarize(retained), correction_to_projected_norm=summarize(corrections),
        collection_kl=summarize(kls), maxima=maxima, all_32_permutations_match_LOW=True,
        actor_optimizer_hash_chain_verified=True, final_original_zero_guard_verified=True,
        both_linear_operations_independently_gated=True)


def first_collection(folder, seed, features, target):
    with patch.object(replay, 'OUT', OUT), patch.object(replay, 'FULL', LOW), patch.object(replay, 'WEIGHTED', LOW), \
         patch.object(replay, 'projection_verify', projection_verify), patch.object(replay, 'linear_verify', correction_verify):
        result = replay.replay_first_collection(folder, seed, features, target)
    assert result.pop('initial_collection_and_post_critic_match_both_controls')
    result['initial_collection_and_post_critic_match_LOW'] = True
    first = read(folder/'projection-1.json')[0]
    previous = read(COVERAGE/str(seed)/'report.json')
    assert first['actor_pre'] == previous['actor_initial_sha256']
    assert first['actor_tentative'] == previous['actor_tentative_sha256']
    assert first['optimizer_pre'] == previous['optimizer_initial_sha256']
    assert first['optimizer_tentative'] == previous['optimizer_tentative_sha256']
    assert first['actor_candidate'] == previous['arms']['expanded48']['final_actor_sha256']
    assert first['actor_projected'] == previous['arms']['expanded48']['projected_actor_sha256']
    assert first['accepted'] and first['actor_post'] == first['actor_candidate']
    result['first_candidate_matches_frozen_expanded48_feasibility'] = True
    return result


def verify_fixed_probes(folder, source):
    branch, onset, pose, quiet_index = verify_probe_identity(source)
    evidence, records = arrays(folder/'fixed-probes.npz'), read(folder/'fixed-probes.json')
    history = read(folder/'history.json')
    assert len(records) == 33 and all(value.shape == (33, 36) for value in evidence.values())
    endpoint_means, endpoint_hashes = {}, {}
    for update, filename in ((0, 'initial.pt'), (32, 'final.pt')):
        actor = AnkleHead().eval()
        actor.load_state_dict(torch.load(folder/filename, map_location='cpu', weights_only=True))
        with torch.no_grad():
            endpoint_means[update] = actor(torch.from_numpy(source['probe_x'])).numpy().ravel()
        endpoint_hashes[update] = actor_hash(actor)
        if update == 0:
            initial_optimizer = optimizer_hash(torch.optim.Adam(actor.parameters(), lr=1e-4))
    reference = np.select((branch == 1, branch == 2), (-1, 1), default=0).astype(np.int64)
    masks = dict(all=np.ones(36, bool), quiet=branch == 0, push=branch != 0,
                 positive_x=branch == 1, negative_x=branch == 2)
    for update, row in enumerate(records):
        assert row['update'] == update
        if update == 0:
            assert row['actor_sha256'] == endpoint_hashes[0] and row['optimizer_sha256'] == initial_optimizer
        else:
            assert row['actor_sha256'] == history[update-1]['actor_sha256']
            assert row['optimizer_sha256'] == history[update-1]['optimizer_sha256']
        item = {key: value[update] for key, value in evidence.items()}
        mean = item['mean']
        assert mean.dtype == np.float32 and np.isfinite(mean).all()
        if update in endpoint_means:
            assert np.array_equal(mean, endpoint_means[update]), (update, 'endpoint predictions')
        previous_latent = source['probe_x'][:, -1]
        memory = np.float32(.9)*previous_latent
        previous = np.float32(.1)*source['probe_x'][:, -2]
        desired = np.float32(.1)*np.tanh(mean)
        requested = previous+np.clip(desired-previous, np.float32(-.02), np.float32(.02))
        expected = dict(branch=branch, onset=onset, pose=pose, quiet_index=quiet_index, reference_sign=reference,
            mean=mean, previous_latent=previous_latent, memory=memory, innovation=mean-memory,
            previous_addition=previous, desired_addition=desired, requested_addition=requested,
            slew_limited=np.abs(desired-previous) > np.float32(.02))
        for key in ('desired_addition', 'requested_addition'):
            assert item[key].dtype == np.float32
            close(item[key], expected[key], key, 1e-8)
            expected[key] = item[key]
        for key, paired in (('mean', 'paired_mean'), ('innovation', 'paired_innovation'),
                            ('requested_addition', 'paired_requested_addition')):
            values = expected[key].astype(float)
            expected[paired] = values-values[quiet_index]
        for key in ('mean', 'innovation', 'paired_mean'):
            expected[key+'_over_sigma'] = expected[key].astype(float)/SIGMA
        expected['raw_alignment'] = classifications(mean, reference)
        expected['paired_alignment'] = classifications(expected['paired_mean'], reference)
        assert set(item) == set(expected)
        for key, value in expected.items():
            assert item[key].dtype == value.dtype and np.array_equal(item[key], value), (update, key)
        summary = row['summary']
        assert isinstance(summary['scope'], str) and summary['scope']
        expected_summary = dict(probes=36, quiet_probes=12, pushed_probes=24, poses=list(range(6)), onsets=[44, 74],
            zero_action_memory_verified=True, pairing_verified=True, sigma=SIGMA, sign_band=1e-7,
            rho=.9, addition_radius=.1, addition_slew=.02,
            groups={name: probe_group(expected, mask) for name, mask in masks.items()},
            by_onset={str(value): probe_group(expected, onset == value) for value in (44, 74)}, scope=summary['scope'])
        compare(row, dict(update=update, actor_sha256=row['actor_sha256'], optimizer_sha256=row['optimizer_sha256'],
                          summary=expected_summary), 'fixed probe boundary '+str(update))
    assert records[-1]['actor_sha256'] == endpoint_hashes[32]
    compare(read(folder/'report.json')['fixed_probes'], dict(boundaries=33, initial_included=True,
        collection_interval=1, cases=36, quiet=12, pushed=24, optimizer_or_rng_mutated=False,
        intermediate_checkpoints=0), 'fixed-probe protocol')
    return dict(boundaries=33, predictions_per_boundary=36, stored_predictions=1188,
        all_numeric_summaries_recomputed=True, all_boundary_hashes_match_history=True,
        initial_and_final_forward_predictions_exact=True, intermediate_predictions_replayed=False,
        intermediate_checkpoint_series=False)


def inherited_physical_audit():
    captured = {}
    def capture(path, content):
        assert path.name in ('audit.json', 'paired-outcomes.json')
        captured[path.name] = content
    with patch.object(physical, 'OUT', OUT), patch.object(physical, 'FULL', LOW), \
         patch.object(physical, 'save', capture), redirect_stdout(io.StringIO()):
        physical.main()
    assert set(captured) == {'audit.json', 'paired-outcomes.json'}
    for row in captured['paired-outcomes.json']:
        name = f"{row['seed']}-t{row['timing']}-g{row['group']}"
        if row['mode'] == 'matched':
            name = 'matched-'+name
        if row['comparator'] == 'unweighted-half-noise':
            reference_root, reference_name = LOW, name
            row['comparator'] = 'original12'
        else:
            assert row['comparator'] == 'zero'
            reference_root, reference_name = PILOT, f"baseline-t{row['timing']}-g{row['group']}"
        current = read(OUT/name/'report.json')['cases'][row['case']]
        previous = read(reference_root/reference_name/'report.json')['cases'][row['case']]
        for key in ('case', 'pose', 'heldout', 'force_n', 'hit_seconds'):
            assert current[key] == previous[key], (name, row['case'], key)
    return captured['audit.json'], captured['paired-outcomes.json']


def provenance():
    inputs, completed = read(OUT/'inputs.json'), read(OUT/'completed.json')
    assert inputs and all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in completed['outputs'].items())
    resolved = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert resolved[Path(__file__).resolve()] == digest(Path(__file__))
    sources = list(Path(__file__).parent.glob('*.py'))
    plan = Path(__file__).with_name('RECOVERY_COVERAGE_LEARNING_PLAN.md')
    for path in [*sources, plan]:
        assert resolved[path.resolve()] == digest(path) == digest(OUT/'sources'/path.name)
    consumed = {}
    for root in (LOW, PILOT, COVERAGE, DATA):
        assert read(root/'audit.json')['passed'] is True
        for name in ('audit.json', 'completed.json'):
            assert resolved[(root/name).resolve()] == digest(root/name)
        outputs = read(root/'completed.json')['outputs']
        if root in (LOW, PILOT):
            selected = list(outputs)
        elif root == COVERAGE:
            selected = ['inputs.json', 'contract.json', 'bank.npz', 'bank-report.json']
            selected.extend(str(Path(str(seed))/'report.json') for seed in SEEDS)
        else:
            selected = ['inputs.json']+[str(Path(str(seed))/'dataset.npz') for seed in SEEDS]
        for relative in selected:
            assert resolved[(root/relative).resolve()] == outputs[relative]
        consumed[str(root)] = len(selected)
    from .run_support_skill_study import FOUNDATION
    from .run_learned_support import PARENT
    external = Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning')
    required = [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json',
                FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
                external/'normalizer.py', external/'base_agent.py']
    assert all(path.resolve() in resolved for path in required)
    for row in read(FOUNDATION/'protocol.json')['demonstrations']:
        assert resolved[Path(row['path']).resolve()] == row['sha256']
    historical = {}
    for root in (LOW, COVERAGE):
        previous = read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared and all(inputs[path] == previous[path] for path in shared)
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
    compare(read(OUT/'historical-inputs.json'), historical, 'modern source identity')
    assert digest(OUT/'quiet-anchors.npz') == digest(COVERAGE/'bank.npz')
    models = {str(path.relative_to(OUT)) for path in OUT.rglob('*.pt')}
    expected_models = {str(Path(f'fit-{seed}')/name) for seed in SEEDS
                       for name in ('initial.pt', 'final.pt', 'critic-final.pt')}
    assert models == expected_models, 'unexpected policy checkpoint series'
    return inputs, completed, historical, consumed


def verify_contract(historical):
    contract = read(OUT/'contract.json')
    assert contract['training_seeds'] == list(SEEDS) and contract['initial_actor_seed'] == 920009
    assert contract['nominal_transitions_per_seed'] == 24576 and contract['comparator'] == str(LOW)
    assert contract['additional_comparators'] == [str(PILOT)] and contract['coverage_source'] == str(COVERAGE)
    assert contract['actor_learning_rate'] == 1e-4 and contract['critic_learning_rate'] == 3e-4
    assert contract['sigma'] == contract['baseline_sigma'] == SIGMA
    assert contract['independent_linear_algebra'] == 'thin QR(J.T), then SVD(R.T)'
    projection = contract['quiet_projection']
    assert projection['anchor_count'] == 48 and projection['anchor_onsets'] == list(ONSETS)
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
    assert contract['shared_historical_inputs'] == {root: row['count'] for root, row in historical.items()}
    compare(contract['fixed_probe_tracking'], dict(boundaries=33, probes=36, quiet_probes=12,
        pushed_probes=24, anchor_probes=48, zero_action_memory=True, observation_only=True,
        intermediate_checkpoints=False, reference_label_training=False), 'probe tracking contract')
    compare(read(OUT/'checks.json')['quiet_anchors'], dict(count=48, width=427,
        identical_source_features=True, zero_action_memory=True, fresh_initial_targets_zero=True,
        sealed_bank_exact=True), 'bank checks')


def audit():
    torch.set_num_threads(1)
    inputs, completed, historical, consumed = provenance()
    verify_contract(historical)
    features, target = verify_bank()
    updates, replays, probes = {}, {}, {}
    for seed in SEEDS:
        folder = OUT/f'fit-{seed}'
        updates[str(seed)] = verify_history(folder, seed, features, target)
        replays[str(seed)] = first_collection(folder, seed, features, target)
        probes[str(seed)] = verify_fixed_probes(folder, arrays(DATA/str(seed)/'dataset.npz'))
    result, pairs = inherited_physical_audit()
    assert result['passed'] and result['traces'] == 45 and result['cases'] == 720
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    result.update(comparator='original12', input_files=len(inputs), output_files=len(completed['outputs']),
        modern_shared_inputs={root: row['count'] for root, row in historical.items()},
        consumed_historical_outputs=consumed, actor_learning_rate=1e-4, critic_learning_rate=3e-4,
        projected_retracted_updates=updates, first_collection_optimizer_replay=replays, fixed_probe_tracking=probes,
        quiet_anchors=dict(count=48, sealed_bank_exact=True, original12_subset_exact=True,
                          fresh_initial_targets_zero=True, action_memory_zero=True, reserved_states_in_bank=False),
        independent_linear_algebra='thin QR(J.T), then SVD(R.T)',
        intermediate_full_parameter_states_retained=False,
        audit_scope='Runtime full-vector QR gates, compact48 geometry, exact first actor/Adam/critic replay, boundary summary arithmetic, endpoint probe inference, and complete physical/objective audit.')
    return result, pairs


def main():
    destination = OUT/'audit.json'
    assert not destination.exists(), 'Preserve prior audit; use a successor for any repair.'
    try:
        result, pairs = audit()
    except Exception as error:
        save(destination, dict(passed=False, error_type=type(error).__name__, error=str(error)))
        raise
    save(OUT/'paired-outcomes.json', pairs)
    save(destination, result)
    print('COVERAGE RECOVERY AUDIT PASSED', flush=True)


if __name__ == '__main__':
    main()
