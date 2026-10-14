"""Independent audit of a training-only quiet bank and two fixed Adam proposals."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_constrained_recovery import actor_hash, optimizer_hash, quiet_jacobian
from .verify_quiet_anchor_retraction import analytical_metrics, compare, tensors
from .verify_command_generalization import probe_group, classifications, verify_probe_identity

OUT = Path('logs/mimickit-training/recovery-quiet-coverage-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
BASE = Path('logs/mimickit-training/recovery-command-generalization-20261009-01')
PILOT = Path('logs/mimickit-training/learned-ankle-20261009-01')
SEEDS = (921009, 931009)
ONSETS = (40, 44, 48, 54, 66, 74, 82, 90)
SIGMA = .08717797887081347
CUTOFF = 1e-6
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, label, atol=1e-10, rtol=2e-7):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), label


def independent_basis(jacobian, singular_values):
    quiet, supplied = (np.asarray(value, dtype=np.float64) for value in (jacobian, singular_values))
    assert quiet.ndim == 2 and quiet.shape[0] in (12, 48)
    assert supplied.shape == (quiet.shape[0],)
    assert np.isfinite(quiet).all() and np.isfinite(supplied).all()
    orthogonal, triangular = np.linalg.qr(quiet.T, mode='reduced')
    left, singular, right = np.linalg.svd(triangular.T, full_matrices=False)
    assert singular[0] > 0 and np.allclose(singular, supplied, atol=1e-10, rtol=2e-7)
    floor = 64*np.finfo(float).eps*max(1., float(singular[0]))
    margin = float(np.min(np.abs(singular-CUTOFF*singular[0])))
    assert margin > floor, 'ambiguous independent QR rank'
    keep = singular > CUTOFF*singular[0]
    assert int(keep.sum()) == int(np.count_nonzero(supplied > CUTOFF*supplied[0]))
    rows = right[keep]@orthogonal.T
    metadata = dict(method='float64 thin QR plus reduced-transpose SVD', threshold=CUTOFF,
                    rank=int(keep.sum()), singular_values=singular.tolist(),
                    spectrum_max_absolute_error=float(np.abs(singular-supplied).max()),
                    rank_margin=margin, rank_numerical_floor=floor)
    return quiet, left[:, keep], singular[keep], rows, metadata


def projection_verify(jacobian, tentative, projected, singular_values):
    """Verify direct-Jacobian SVD projection through QR and a small triangular SVD."""
    quiet, basis, retained, rows, metadata = independent_basis(jacobian, singular_values)
    proposal, supplied = (np.asarray(value, dtype=np.float64) for value in (tentative, projected))
    assert proposal.shape == supplied.shape == (quiet.shape[1],)
    assert np.isfinite(proposal).all() and np.isfinite(supplied).all()
    q_proposal = quiet@proposal
    independent = proposal-rows.T@(rows@proposal)
    difference = supplied-independent
    maximum = float(np.abs(difference).max())
    relative = float(np.linalg.norm(difference)/max(np.linalg.norm(proposal), 1e-30))
    assert maximum <= 2e-8 and relative <= 2e-5, (maximum, relative)
    removed = proposal-supplied
    proposal_squared, supplied_squared, removed_squared = (float(value@value) for value in (proposal, supplied, removed))
    orthogonality = float(removed@supplied)
    norm_error = abs(proposal_squared-supplied_squared-removed_squared)
    identity_error = abs(float(proposal@supplied)-supplied_squared)
    tolerance = 2e-10+2e-7*proposal_squared
    assert abs(orthogonality) <= tolerance and norm_error <= tolerance and identity_error <= tolerance
    q_projected = quiet@supplied
    residual = float(np.linalg.norm(q_projected))
    residual_bound = max(1e-10, 2*singular_values[0]*CUTOFF*np.sqrt(proposal_squared))
    assert residual <= residual_bound
    retained_residual = float(np.linalg.norm(basis.T@q_projected))
    assert retained_residual <= 1e-10+2e-7*singular_values[0]*np.sqrt(proposal_squared)
    return dict(passed=True, **metadata, quiet_gram=(quiet@quiet.T).tolist(),
        q_tentative=q_proposal.tolist(), q_projected=q_projected.tolist(),
        projection_max_absolute_error=maximum, projection_relative_proposal_error=relative,
        quiet_residual_l2=residual, quiet_residual_bound=residual_bound,
        retained_quiet_residual_l2=retained_residual, removed_projected_inner_product=orthogonality,
        pythagorean_error=norm_error, projection_identity_error=identity_error)


def correction_verify(jacobian, residual, correction, singular_values):
    """Verify minimum-norm correction through QR without squaring the condition number."""
    quiet, basis, retained, rows, metadata = independent_basis(jacobian, singular_values)
    residual, supplied = (np.asarray(value, dtype=np.float64) for value in (residual, correction))
    assert residual.shape == (quiet.shape[0],) and supplied.shape == (quiet.shape[1],)
    assert np.isfinite(residual).all() and np.isfinite(supplied).all()
    coordinates = basis.T@residual
    independent = -rows.T@(coordinates/retained)
    maximum = float(np.abs(supplied-independent).max())
    relative = float(np.linalg.norm(supplied-independent)/max(np.linalg.norm(independent), 1e-30))
    assert maximum <= 2e-8 and relative <= 2e-5, (maximum, relative)
    expected_residual = residual-basis@coordinates
    actual_residual = residual+quiet@supplied
    residual_error = float(np.linalg.norm(actual_residual-expected_residual))
    assert residual_error <= 1e-10+2e-7*np.linalg.norm(residual)
    rowspace = rows.T@(rows@supplied)
    rowspace_error = float(np.linalg.norm(supplied-rowspace))
    assert rowspace_error <= 2e-8+2e-5*np.linalg.norm(supplied)
    expected_norm_squared = float(np.sum((coordinates/retained)**2))
    norm_error = abs(float(supplied@supplied)-expected_norm_squared)
    assert norm_error <= 2e-10+2e-7*expected_norm_squared
    return dict(passed=True, **metadata,
        correction_max_absolute_error=maximum, correction_relative_l2_error=relative,
        linear_residual_l2=float(np.linalg.norm(actual_residual)),
        expected_truncated_residual_l2=float(np.linalg.norm(expected_residual)),
        linear_residual_error=residual_error, rowspace_error=rowspace_error,
        minimum_norm_squared_error=norm_error)


def verify_bank(bank, prefix, report, reference, original, parent_state, datasets):
    """Audit issue-phase extraction and observation arithmetic without integrating physics."""
    extra = {'state_'+key for key in ('features', 'observation', 'history', 'steps', 'previous',
                                     'latent', 'qpos', 'qvel', 'balance')}
    extra.update(('normalizer_mean', 'normalizer_std', 'normalizer_clip'))
    assert set(prefix) == set(reference) | extra
    for key, value in reference.items():
        assert np.array_equal(prefix[key], value[:91]), 'archived prefix '+key
    assert prefix['alive'].shape == (91, 16) and prefix['alive'].all()
    assert not prefix['failed'].any() and not prefix['warning'].any()
    assert all(not prefix[key].any() for key in ('force', 'addition', 'previous', 'latent', 'previous_latent'))
    dt = float(prefix['time'][0])
    assert dt == .016668
    assert np.array_equal(prefix['time'], (np.arange(91)+1)*dt)
    shapes = dict(features=(92, 16, 427), observation=(92, 16, 374), history=(92, 16, 3, 17),
                  steps=(92, 16), previous=(92, 16, 33), latent=(92, 16, 1),
                  qpos=(92, 16, 46), qvel=(92, 16, 45), balance=(92, 16, 17))
    for key, shape in shapes.items():
        value = prefix['state_'+key]
        assert value.shape == shape and np.isfinite(value).all(), key
        if key == 'steps':
            assert np.issubdtype(value.dtype, np.integer)
        elif key not in ('qpos', 'qvel'):
            assert value.dtype == np.float32, key
    assert np.array_equal(prefix['state_steps'], np.repeat(np.arange(92)[:, None], 16, axis=1))
    assert np.array_equal(prefix['state_history'][:-1], prefix['history_before'])
    assert np.array_equal(prefix['state_history'][0], np.repeat(prefix['state_balance'][0, :, None], 3, axis=1))
    assert np.array_equal(prefix['state_history'][1:, :, 0], prefix['balance_features'])
    assert np.array_equal(prefix['state_history'][1:, :, 1:], prefix['state_history'][:-1, :, :2])
    assert np.array_equal(prefix['state_balance'][1:], prefix['balance_features'])
    for state, traced in (('qpos', 'qpos'), ('qvel', 'qvel'), ('previous', 'addition'), ('latent', 'latent')):
        assert np.array_equal(prefix['state_'+state][1:], prefix[traced]), state
    assert not prefix['state_previous'].any() and not prefix['state_latent'].any()
    for name in ('mean', 'std'):
        value = prefix['normalizer_'+name]
        expected = parent_state['_obs_norm._'+name].detach().cpu().numpy()
        assert value.shape == (374,) and value.dtype == np.float32
        assert np.array_equal(value, expected), 'frozen normalizer '+name
    assert (prefix['normalizer_std'] > 0).all() and prefix['normalizer_clip'].shape == ()
    assert float(prefix['normalizer_clip']) == 10.
    observation = torch.from_numpy(prefix['state_observation'])
    normalized = ((observation-torch.from_numpy(prefix['normalizer_mean']))
                  /torch.from_numpy(prefix['normalizer_std'])).clamp(-10., 10.).clamp(-20., 20.).numpy()
    reconstructed = np.concatenate((normalized, np.clip(prefix['state_history'].reshape(92, 16, 51), -20., 20.),
        prefix['state_previous'][:, :, 21:22]/np.float32(.1), prefix['state_latent']), axis=-1)
    assert np.array_equal(reconstructed, prefix['state_features']), 'independent feature arithmetic'
    assert set(bank) == {'features', 'target', 'onset', 'pose', 'world', 'issue', 'original_indices'}
    onset, pose = np.repeat(np.array(ONSETS, dtype=np.int64), 6), np.tile(np.arange(6, dtype=np.int64), 8)
    original_indices = np.r_[6:12, 30:36].astype(np.int64)
    for key, value in dict(onset=onset, pose=pose, world=pose, issue=onset+1, original_indices=original_indices).items():
        assert bank[key].dtype == np.int64 and np.array_equal(bank[key], value), key
    assert bank['features'].shape == (48, 427) and bank['features'].dtype == np.float32
    assert bank['target'].shape == (48, 1) and bank['target'].dtype == np.float32 and not bank['target'].any()
    assert np.array_equal(bank['features'], reconstructed[onset+1, pose])
    assert not bank['features'][:, -2:].any()
    for key in ('features', 'target', 'onset', 'pose'):
        assert np.array_equal(bank[key][original_indices], original[key]), 'original bank '+key
    for source in datasets.values():
        branch, source_onset, source_pose, _ = verify_probe_identity(source)
        quiet = branch == 0
        assert np.array_equal(source['probe_x'][quiet], original['features'])
        assert np.array_equal(source_onset[quiet], original['onset'])
        assert np.array_equal(source_pose[quiet], original['pose'])
    for key in ('phase', 'normalization', 'scope'):
        assert isinstance(report[key], str) and report[key]
    expected_report = dict(passed=True, dt=dt, prefix_intervals=91, native_worlds=16,
        native_world_intervals=1456, captured_states=92, anchors=48, original_anchors=12,
        added_anchors=36, onsets=list(ONSETS), poses=list(range(6)), ankles=[21, 29],
        original_indices=original_indices.tolist(), exact_reference_fields=sorted(reference),
        archived_prefix_exact=True, original_features_exact=True, all_active=True, zero_forces=True,
        zero_correction=True, zero_action_memory=True, initial_targets_zero=True,
        frozen_source_unchanged=True, frozen_initial_unchanged=True,
        reserved_poses_in_bank=False, reserved_onsets_in_bank=False,
        optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        feature_layout=dict(normalized_observation=[0, 374], balance_history=[374, 425],
                            previous_ankle_addition=[425, 426], previous_latent=[426, 427]),
        **{key: report[key] for key in ('phase', 'normalization', 'scope')})
    compare(report, expected_report, 'bank report')
    return dict(anchors=48, original_anchors_exact=12, added_anchors=36, prefix_intervals=91,
                prefix_world_intervals=1456, phase_exact=True, archived_prefix_fields_exact=len(reference),
                raw_feature_reconstruction_exact=True, frozen_normalizer_exact=True,
                reserved_bank_rows=0)


def first_proposal(seed):
    """Replay exactly one archived actor Adam proposal; no trajectory or critic update."""
    folder = LOW/f'fit-{seed}'
    actor = AnkleHead().eval()
    actor.load_state_dict(torch.load(folder/'initial.pt', map_location='cpu', weights_only=True))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    source = arrays(folder/'collection-1.npz')
    batch, objective = tensors(source), arrays(folder/'objective-1.npz')
    record = read(folder/'projection-1.json')[0]
    assert (record['update'], record['epoch'], record['batch'], record['minibatch_index']) == (1, 0, 0, 0)
    assert record['accepted'] and not record['reasons']
    assert actor_hash(actor) == record['actor_pre'] and optimizer_hash(optimizer) == record['optimizer_pre']
    before = parameters_to_vector(actor.parameters()).detach().clone()
    initial = deepcopy(actor)
    ids = torch.from_numpy(objective['minibatch_indices'][0])
    assert ids.shape == (256,) and np.unique(ids.numpy()).size == 256
    ratio = (Normal(actor(batch['x'][ids]), SIGMA).log_prob(batch['action'][ids]).sum(-1)
             -batch['old_logp'][ids]).exp()
    assert np.array_equal(ratio.detach().numpy(), objective['policy_ratio'][0])
    advantage = batch['advantage'][ids]
    loss = -(batch['weight'][ids]*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()
    assert float(loss.detach()) == record['actor_loss'] == float(objective['actor_loss'][0])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
    clipped = float(torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in actor.parameters()])))
    assert norm == record['gradient_norm'] and clipped == record['clipped_norm']
    optimizer.step()
    assert actor_hash(actor) == record['actor_tentative'] and optimizer_hash(optimizer) == record['optimizer_tentative']
    tentative = parameters_to_vector(actor.parameters()).detach().clone()
    assert not initial.net[-1].weight.detach().any() and not initial.net[-1].bias.detach().any()
    return initial, actor, before, tentative, batch, ids, record, optimizer


def verify_probe_scores(actor, source, evidence, prefix, report):
    branch, onset, pose, paired = verify_probe_identity(source)
    with torch.no_grad():
        mean = actor(torch.from_numpy(source['probe_x'])).numpy().ravel()
    reference = np.select((branch == 1, branch == 2), (-1, 1), default=0).astype(np.int64)
    previous_latent = source['probe_x'][:, -1]
    memory = np.float32(.9)*previous_latent
    previous = np.float32(.1)*source['probe_x'][:, -2]
    desired = np.float32(.1)*np.tanh(mean)
    requested = previous+np.clip(desired-previous, np.float32(-.02), np.float32(.02))
    expected = dict(branch=branch, onset=onset, pose=pose, quiet_index=paired, reference_sign=reference,
        mean=mean, previous_latent=previous_latent, memory=memory, innovation=mean-memory,
        previous_addition=previous, desired_addition=desired, requested_addition=requested,
        slew_limited=np.abs(desired-previous) > np.float32(.02))
    for key in ('desired_addition', 'requested_addition'):
        recorded = evidence[prefix+key]
        assert recorded.shape == (36,) and recorded.dtype == np.float32
        close(recorded, expected[key], prefix+key, 1e-8)
        expected[key] = recorded
    for key, target in (('mean', 'paired_mean'), ('innovation', 'paired_innovation'),
                        ('requested_addition', 'paired_requested_addition')):
        values = expected[key].astype(np.float64)
        expected[target] = values-values[paired]
    for key in ('mean', 'innovation', 'paired_mean'):
        expected[key+'_over_sigma'] = expected[key].astype(np.float64)/SIGMA
    expected['raw_alignment'] = classifications(mean, reference)
    expected['paired_alignment'] = classifications(expected['paired_mean'], reference)
    for key, value in expected.items():
        recorded = evidence[prefix+key]
        assert recorded.dtype == value.dtype and np.array_equal(recorded, value), prefix+key
    rows, labels = [], {-1: 'opposed', 0: 'within_band', 1: 'aligned'}
    for index in range(36):
        pushed = branch[index] != 0
        row = dict(index=index, onset_frame=int(onset[index]), pose=int(pose[index]),
            branch=('quiet', 'positive_x', 'negative_x')[branch[index]], quiet_probe_index=int(paired[index]),
            reference_sign=int(reference[index]) if pushed else None,
            raw_reference=labels[int(expected['raw_alignment'][index])] if pushed else None,
            paired_reference=labels[int(expected['paired_alignment'][index])] if pushed else None,
            slew_limited=bool(expected['slew_limited'][index]))
        for key in ('mean', 'previous_latent', 'memory', 'innovation', 'paired_mean', 'paired_innovation',
                    'mean_over_sigma', 'innovation_over_sigma', 'paired_mean_over_sigma',
                    'previous_addition', 'desired_addition', 'requested_addition', 'paired_requested_addition'):
            row[key] = float(expected[key][index])
        rows.append(row)
    masks = dict(all=np.ones(36, dtype=bool), quiet=branch == 0, push=branch != 0,
                 positive_x=branch == 1, negative_x=branch == 2)
    assert isinstance(report['scope'], str) and report['scope']
    expected_report = dict(probes=36, quiet_probes=12, pushed_probes=24, poses=list(range(6)), onsets=[44, 74],
        zero_action_memory_verified=True, pairing_verified=True, sigma=SIGMA, sign_band=1e-7,
        rho=.9, addition_radius=.1, addition_slew=.02,
        groups={name: probe_group(expected, mask) for name, mask in masks.items()},
        by_onset={str(value): probe_group(expected, onset == value) for value in (44, 74)},
        per_probe=rows, scope=report['scope'])
    compare(report, expected_report, prefix+'report')
    return {prefix+key for key in expected}


def verify_score(actor, evidence, prefix, batch, ids, features):
    with torch.no_grad():
        predictions = dict(quiet_predictions=actor(features).numpy().ravel(),
            collection_predictions=actor(batch['x']).numpy(),
            minibatch_predictions=actor(batch['x'][ids]).numpy())
    for key, value in predictions.items():
        assert evidence[prefix+key].dtype == np.float32
        assert np.array_equal(evidence[prefix+key], value), prefix+key
    values = {}
    for kind, selected in (('full', torch.arange(768)), ('minibatch', ids)):
        mean = predictions['collection_predictions' if kind == 'full' else 'minibatch_predictions']
        distribution = Normal(torch.from_numpy(mean), SIGMA)
        ratio = (distribution.log_prob(batch['action'][selected]).sum(-1)-batch['old_logp'][selected]).exp()
        advantage, weight = batch['advantage'][selected], batch['weight'][selected]
        score = (weight*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()
        assert evidence[prefix+kind+'_ratio'].dtype == np.float32
        assert np.array_equal(evidence[prefix+kind+'_ratio'], ratio.numpy()), prefix+kind+' ratio'
        values[kind+'_surrogate'] = float(score)
    mean = torch.from_numpy(predictions['collection_predictions'])
    kl = float(((mean-batch['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
    quiet = predictions['quiet_predictions']
    values.update(kl=kl, quiet48_max=float(np.abs(quiet).max()),
                  quiet48_rms=float(np.sqrt(np.mean(quiet.astype(float)**2))),
                  passes_all48_guards=bool(kl <= KL_LIMIT and np.abs(quiet).max() <= ANCHOR_LIMIT))
    analytical = analytical_metrics(predictions['collection_predictions'], quiet, batch, ids.numpy(),
                                    np.zeros(48), predictions['minibatch_predictions'])
    for key in ('full_surrogate', 'minibatch_surrogate', 'kl'):
        close(values[key], analytical[key], prefix+'float64 '+key, 5e-7, 2e-6)
    keys = {prefix+key for key in (*predictions.keys(), 'full_ratio', 'minibatch_ratio')}
    return values, keys


def verify_candidate(seed, bank, probes, evidence, report):
    initial, tentative_actor, before, tentative, batch, ids, record, optimizer = first_proposal(seed)
    before_np, tentative_np = before.numpy(), tentative.numpy()
    proposal = tentative.double().numpy()-before.double().numpy()
    features = torch.from_numpy(bank['features'])
    jacobian = quiet_jacobian(initial, features)
    expected_shared = dict(initial_parameters=before_np, tentative_parameters=tentative_np,
        tentative_delta=proposal, initial_quiet_jacobian=jacobian, minibatch_indices=ids.numpy())
    for key, value in expected_shared.items():
        assert evidence[key].dtype == value.dtype and np.array_equal(evidence[key], value), key
    assert float(np.linalg.norm(proposal)) == record['tentative_norm']
    consumed = set(expected_shared)
    initial_metrics, keys = verify_score(initial, evidence, 'initial_', batch, ids, features)
    consumed.update(keys)
    compare(report['initial_metrics'], initial_metrics, 'initial metrics')
    consumed.update(verify_probe_scores(initial, probes, evidence, 'initial_probe_', report['initial_probes']))
    expected_arms, audit_arms = {}, {}
    for name, indices in (('original12', bank['original_indices']), ('expanded48', np.arange(48, dtype=np.int64))):
        prefix = name+'_'
        item = report['arms'][name]
        actor = deepcopy(initial)
        assert np.array_equal(evidence[prefix+'anchor_indices'], indices)
        own_features = features[torch.from_numpy(indices)]
        own_jacobian = jacobian[indices]
        singular = evidence[prefix+'projection_singular_values']
        projected = evidence[prefix+'projected_delta']
        projection = projection_verify(own_jacobian, proposal, projected, singular)
        compare(item['projection']['independent'], projection, name+' independent projection')
        # The supplied, independently certified direction is applied exactly once in float32.
        intermediate = before+torch.from_numpy(projected.astype(np.float32))
        assert np.array_equal(evidence[prefix+'projected_parameters'], intermediate.numpy())
        actual_projection = intermediate.double().numpy()-before.double().numpy()
        assert np.array_equal(evidence[prefix+'actual_projected_delta'], actual_projection)
        vector_to_parameters(intermediate, actor.parameters())
        projected_hash = actor_hash(actor)
        with torch.no_grad():
            target = initial(own_features).numpy().ravel()
            projected_anchor = actor(own_features).numpy().ravel()
            projected_kl = float(((actor(batch['x'])-batch['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
        assert not target.any()
        assert np.array_equal(evidence[prefix+'target'], target)
        assert np.array_equal(evidence[prefix+'projected_anchor_predictions'], projected_anchor)
        candidate_jacobian = quiet_jacobian(actor, own_features)
        assert np.array_equal(evidence[prefix+'projected_jacobian'], candidate_jacobian)
        residual = projected_anchor.astype(float)-target.astype(float)
        assert np.array_equal(evidence[prefix+'residual'], residual)
        correction = evidence[prefix+'correction']
        correction_singular = evidence[prefix+'correction_singular_values']
        correction_gate = correction_verify(candidate_jacobian, residual, correction, correction_singular)
        compare(item['correction']['independent'], correction_gate, name+' independent correction')
        final = intermediate+torch.from_numpy(correction.astype(np.float32))
        assert np.array_equal(evidence[prefix+'final_parameters'], final.numpy())
        actual_correction = final.double().numpy()-intermediate.double().numpy()
        assert np.array_equal(evidence[prefix+'actual_correction'], actual_correction)
        vector_to_parameters(final, actor.parameters())
        final_hash = actor_hash(actor)
        with torch.no_grad():
            own_anchor = actor(own_features).numpy().ravel()
        assert np.array_equal(evidence[prefix+'anchor_predictions'], own_anchor)
        metrics, keys = verify_score(actor, evidence, prefix, batch, ids, features)
        consumed.update(keys)
        own_max = float(np.abs(own_anchor).max())
        metrics.update(own_anchor_max=own_max, passes_own_guards=bool(metrics['kl'] <= KL_LIMIT and own_max <= ANCHOR_LIMIT),
            full_surrogate_change=metrics['full_surrogate']-initial_metrics['full_surrogate'],
            minibatch_surrogate_change=metrics['minibatch_surrogate']-initial_metrics['minibatch_surrogate'])
        consumed.update(verify_probe_scores(actor, probes, evidence, prefix+'probe_', item['probes']))
        if name == 'original12':
            assert projected_hash == record['actor_projected']
            assert final_hash == record['actor_candidate'] == record['actor_post']
            assert optimizer_hash(optimizer) == record['optimizer_post']
            assert np.array_equal(singular, record['singular_values'])
            assert np.array_equal(target, record['anchor_before'])
            assert np.array_equal(projected_anchor, record['anchor_projected'])
            assert projected_kl == record['projected_kl']
            assert np.array_equal(own_anchor, record['anchor_attempted'])
            assert metrics['kl'] == record['kl'] and own_max == record['anchor_max']
            assert metrics['passes_own_guards'] is record['accepted']
            assert np.array_equal(correction_singular, record['retraction']['singular_values'])
            assert np.array_equal(residual, record['retraction']['residual'])
        local_residual = own_anchor.astype(float)-target.astype(float)
        nonlinear = own_anchor.astype(float)-projected_anchor.astype(float)-candidate_jacobian@actual_correction
        total = final.double().numpy()-before.double().numpy()
        geometry_projection = dict(rank=projection['rank'], threshold=CUTOFF,
            tentative_norm=float(np.linalg.norm(proposal)), proposed_norm=float(np.linalg.norm(projected)),
            actual_norm=float(np.linalg.norm(actual_projection)),
            retained_norm_fraction=float(np.linalg.norm(projected)/np.linalg.norm(proposal)),
            rounding_norm=float(np.linalg.norm(actual_projection-projected)),
            projected_anchor_max=float(np.abs(projected_anchor).max()), projected_kl=projected_kl,
            independent=projection)
        geometry_correction = dict(rank=correction_gate['rank'], threshold=CUTOFF,
            proposed_norm=float(np.linalg.norm(correction)), actual_norm=float(np.linalg.norm(actual_correction)),
            rounding_norm=float(np.linalg.norm(actual_correction-correction)),
            correction_to_projected_norm=float(np.linalg.norm(correction)/max(np.linalg.norm(actual_projection), 1e-30)),
            local_residual_max=float(np.abs(local_residual).max()),
            linear_residual_max=float(np.abs(residual+candidate_jacobian@correction).max()),
            rounded_linear_residual_max=float(np.abs(residual+candidate_jacobian@actual_correction).max()),
            nonlinear_rms=float(np.sqrt(np.mean(nonlinear**2))), nonlinear_max=float(np.abs(nonlinear).max()),
            total_step_norm=float(np.linalg.norm(total)), independent=correction_gate)
        expected_arms[name] = dict(anchor_count=len(indices), projected_actor_sha256=projected_hash,
            final_actor_sha256=final_hash, optimizer_sha256=optimizer_hash(optimizer),
            historical_candidate_exact=name == 'original12', metrics=metrics, probes=item['probes'],
            projection=geometry_projection, correction=geometry_correction)
        consumed.update(prefix+key for key in ('anchor_indices', 'target', 'projection_singular_values',
            'projected_delta', 'actual_projected_delta', 'projected_parameters', 'projected_anchor_predictions',
            'projected_jacobian', 'residual', 'correction_singular_values', 'correction', 'actual_correction',
            'final_parameters', 'anchor_predictions'))
        for key in ('target', 'projected_parameters', 'projected_anchor_predictions', 'final_parameters', 'anchor_predictions'):
            assert evidence[prefix+key].dtype == np.float32, prefix+key
        for key in ('projection_singular_values', 'projected_delta', 'actual_projected_delta', 'projected_jacobian',
                    'residual', 'correction_singular_values', 'correction', 'actual_correction'):
            assert evidence[prefix+key].dtype == np.float64, prefix+key
        assert evidence[prefix+'anchor_indices'].dtype == np.int64
        audit_arms[name] = dict(projection_rank=projection['rank'], correction_rank=correction_gate['rank'],
            projected_and_corrected_float32_application_exact=True, predictions_and_objectives_verified=True,
            original12_historical_parity=name == 'original12', all48_guard=metrics['passes_all48_guards'])
    assert set(evidence) == consumed, (set(evidence)-consumed, consumed-set(evidence))
    assert all(np.isfinite(value).all() for value in evidence.values())
    assert actor_hash(tentative_actor) == record['actor_tentative'] and optimizer_hash(optimizer) == record['optimizer_tentative']
    assert actor_hash(initial) == record['actor_pre']
    assert isinstance(report['scope'], str) and report['scope']
    expected_report = dict(seed=seed, actor_learning_rate=1e-4, collection=1, minibatch_index=0,
        arm_order=['original12', 'expanded48'], actor_optimizer_steps_replayed=1, new_training_updates=0,
        critic_updates=0, physical_rollouts=0, new_checkpoints=0, proposal_shared=True,
        original12_historical_parity=True, tentative_optimizer_unchanged=True,
        actor_initial_sha256=record['actor_pre'], actor_tentative_sha256=record['actor_tentative'],
        optimizer_initial_sha256=record['optimizer_pre'], optimizer_tentative_sha256=record['optimizer_tentative'],
        gradient_norm=record['gradient_norm'], clipped_norm=record['clipped_norm'], initial_actor_loss=record['actor_loss'],
        initial_metrics=initial_metrics, initial_probes=report['initial_probes'], arms=expected_arms, scope=report['scope'])
    compare(report, expected_report, str(seed)+' report')
    return dict(historical_Adam_proposal_exact=True, actor_and_optimizer_hashes_exact=True,
                initial_jacobian_exact=True, shared_proposal_exact=True, arms=audit_arms)


def verify_provenance():
    inputs, completed = read(OUT/'inputs.json'), read(OUT/'completed.json')
    assert inputs and all(digest(Path(path)) == sha for path, sha in inputs.items())
    sources = list(Path(__file__).parent.glob('*.py'))
    plan = Path(__file__).with_name('RECOVERY_QUIET_COVERAGE_PLAN.md')
    external = Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning')
    required = {str(path) for path in [*sources, plan, PARENT, PARENT.with_suffix('.amp.pt'),
        FOUNDATION/'dummy.xml', FOUNDATION/'protocol.json', FOUNDATION/'stationary_reference.npz',
        FOUNDATION/'stationary_reference.json', external/'normalizer.py', external/'base_agent.py']}
    for demonstration in read(FOUNDATION/'protocol.json')['demonstrations']:
        path = Path(demonstration['path'])
        required.add(str(path))
        assert digest(path) == demonstration['sha256']
    consumed = {}
    for root in (LOW, DATA, BASE, PILOT):
        required.update(str(root/name) for name in ('completed.json', 'audit.json'))
        assert read(root/'audit.json')['passed'] is True
        outputs = read(root/'completed.json')['outputs']
        artifacts = {'inputs.json'}
        if root == LOW:
            artifacts.update(('quiet-anchors.npz', 'contract.json'))
            artifacts.update(str(Path(f'fit-{seed}')/name) for seed in SEEDS for name in
                             ('initial.pt', 'collection-1.npz', 'objective-1.npz', 'projection-1.json'))
        elif root == DATA:
            artifacts.update(str(Path(str(seed))/'dataset.npz') for seed in SEEDS)
        elif root == PILOT:
            artifacts.update(str(Path('baseline-quiet')/name) for name in ('trace.npz', 'report.json'))
        for relative in artifacts:
            path = root/relative
            required.add(str(path))
            assert digest(path) == outputs[relative], str(path)
        consumed[str(root)] = len(artifacts)
    assert set(inputs) == required, (set(inputs)-required, required-set(inputs))
    for path in [*sources, plan]:
        assert digest(OUT/'sources'/path.name) == inputs[str(path)]
    historical = {}
    for root in (LOW, BASE):
        previous = read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared and all(inputs[path] == previous[path] for path in shared)
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
    compare(read(OUT/'historical-inputs.json'), historical, 'modern shared provenance')
    expected_outputs = {'bank.npz', 'prefix.npz', 'bank-report.json', 'inputs.json',
                        'historical-inputs.json', 'contract.json', 'results.json'}
    expected_outputs.update(str(Path(str(seed))/name) for seed in SEEDS for name in ('evidence.npz', 'report.json'))
    assert set(completed['outputs']) == expected_outputs
    for relative, sha in completed['outputs'].items():
        path = OUT/relative
        assert path.resolve().is_relative_to(OUT.resolve()) and digest(path) == sha, relative
    actual_outputs = {str(path.relative_to(OUT)) for path in OUT.rglob('*') if path.is_file()
                      and 'sources' not in path.parts and path.name not in ('completed.json', 'audit.json')}
    assert actual_outputs == expected_outputs
    assert np.isfinite(completed['wall_seconds']) and completed['wall_seconds'] >= 0
    expected_completion = dict(wall_seconds=completed['wall_seconds'], inputs_unchanged=True,
        frozen_source_unchanged=True, original_features_exact=True, physical_rollouts=1,
        prefix_intervals=91, native_world_intervals=1456, historical_adam_reconstructions=2,
        candidate_comparisons=4, new_ppo_transitions=0, new_learning_updates=0,
        critic_updates=0, new_checkpoints=0,
        historical_shared_inputs={root: item['count'] for root, item in historical.items()},
        outputs=completed['outputs'])
    compare(completed, expected_completion, 'completion')
    return inputs, dict(input_files=len(inputs), output_files=len(expected_outputs),
                        consumed_historical_outputs=consumed,
                        modern_shared_inputs=expected_completion['historical_shared_inputs'])


def verify_contract():
    source = read(LOW/'contract.json')
    assert source['sigma'] == SIGMA and source['dt'] == .016668 and source['ankles'] == [21, 29]
    assert source['training_seeds'] == list(SEEDS) and source['initial_actor_seed'] == 920009
    expected = dict(seeds=list(SEEDS), source_root=str(LOW), dataset_root=str(DATA), baseline_root=str(PILOT),
        actor_learning_rate=1e-4, actor_gradient_clip=.5, initial_actor_seed=920009, sigma=SIGMA,
        dt=.016668, ankles=[21, 29], control=source['control'], original_anchors=12, expanded_anchors=48,
        poses=list(range(6)), onsets=list(ONSETS), original_onsets=[44, 74], excluded_poses=[6, 7],
        excluded_onsets=[60, 102], feature_width=427, zero_action_memory=True, zero_anchor_target=True,
        state_index='onset + 1', singular_relative_cutoff=CUTOFF, anchor_limit=ANCHOR_LIMIT, kl_limit=KL_LIMIT,
        finite_corrections_per_candidate=1,
        primary_algebra='float64 direct Jacobian SVD with float32 parameter application',
        independent_algebra='thin QR(J.T), then SVD(R.T)',
        spectrum_atol=1e-10, spectrum_rtol=2e-7, vector_atol=2e-8, vector_rtol=2e-5,
        full_samples=768, minibatch_samples=256, common_pushed_probes=24, latent_sign_band=1e-7,
        prefix_intervals=91, native_worlds=16, native_world_intervals=1456, physical_rollouts=1,
        historical_adam_reconstructions=2, candidate_comparisons=4, new_ppo_transitions=0,
        new_learning_updates=0, critic_updates=0, new_checkpoints=0,
        reserved_states_in_bank=False, arena_changed=False)
    compare(read(OUT/'contract.json'), expected, 'contract')


def audit():
    torch.set_num_threads(1)
    inputs, provenance = verify_provenance()
    verify_contract()
    states = {seed: torch.load(LOW/f'fit-{seed}'/'initial.pt', map_location='cpu', weights_only=True) for seed in SEEDS}
    assert states[SEEDS[0]].keys() == states[SEEDS[1]].keys()
    assert all(torch.equal(value, states[SEEDS[1]][key]) for key, value in states[SEEDS[0]].items())
    probes = {seed: arrays(DATA/str(seed)/'dataset.npz') for seed in SEEDS}
    for key in ('probe_x', 'probe_branch', 'probe_onset', 'probe_pose', 'probe_quiet_index'):
        assert np.array_equal(probes[SEEDS[0]][key], probes[SEEDS[1]][key]), key
    bank, prefix = arrays(OUT/'bank.npz'), arrays(OUT/'prefix.npz')
    reference_report = read(PILOT/'baseline-quiet'/'report.json')
    assert reference_report['quiet'] and not reference_report['sampled']
    assert reference_report['seconds'] == 60. and reference_report['dt'] == .016668
    parent = torch.load(PARENT.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
    bank_audit = verify_bank(bank, prefix, read(OUT/'bank-report.json'),
        arrays(PILOT/'baseline-quiet'/'trace.npz'), arrays(LOW/'quiet-anchors.npz'), parent, probes)
    results = read(OUT/'results.json')
    assert set(results) == {str(seed) for seed in SEEDS}
    candidates = {}
    for seed in SEEDS:
        evidence, report = arrays(OUT/str(seed)/'evidence.npz'), read(OUT/str(seed)/'report.json')
        compare(results[str(seed)], report, str(seed)+' aggregate parity')
        candidates[str(seed)] = verify_candidate(seed, bank, probes[seed], evidence, report)
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    return dict(passed=True, **provenance, bank=bank_audit, seeds=candidates,
        inputs_unchanged=True, runner_historical_adam_reconstructions=2,
        audit_optimizer_steps_replayed=2, audit_additional_physics_intervals=0, candidate_comparisons=4,
        own_and_all48_guards_separate=True, all48_quiet_and24_push_scores_verified=True,
        independent_algebra='thin QR(J.T), then SVD(R.T)', spectrum_atol=1e-10, spectrum_rtol=2e-7,
        vector_atol=2e-8, vector_rtol=2e-5, float64_objective_atol=5e-7, float64_objective_rtol=2e-6,
        sign_band=1e-7, new_ppo_transitions=0, new_learning_updates=0, critic_updates=0, new_checkpoints=0,
        scope='First-proposal feasibility only. No conclusion about accumulated training drift or physical recovery.')


def main():
    destination = OUT/'audit.json'
    assert not destination.exists(), 'Preserve prior audit; use a successor for any repair.'
    try:
        result = audit()
    except Exception as error:
        save(destination, dict(passed=False, error_type=type(error).__name__, error=str(error)))
        raise
    save(destination, result)
    print('QUIET COVERAGE AUDIT PASSED', flush=True)


if __name__ == '__main__':
    main()
