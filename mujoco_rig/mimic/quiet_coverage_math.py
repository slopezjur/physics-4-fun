"""Compare two quiet constraint banks on one exactly reconstructed Adam proposal."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .command_generalization_probes import compute as probe_compute
from .quiet_constrained_ppo import actor_hash, optimizer_hash, quiet_jacobian
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_coverage import projection_verify, correction_verify

SIGMA = .08717797887081347
CUTOFF = 1e-6
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01
ACTOR_RATE = 1e-4
ARM_ORDER = ('original12', 'expanded48')
ONSETS = (40, 44, 48, 54, 66, 74, 82, 90)


def vector(actor):
    return parameters_to_vector(actor.parameters()).detach().clone()


def tensors(batch):
    return dict(x=torch.from_numpy(batch['x']).flatten(0, 1),
        action=torch.from_numpy(batch['action']).flatten(0, 1),
        old_mean=torch.from_numpy(batch['mean']).flatten(0, 1),
        old_logp=torch.from_numpy(batch['logp']).flatten(),
        advantage=torch.from_numpy(batch['normalized_advantage']),
        weight=torch.from_numpy(batch['actor_weight']).flatten())


def surrogate(mean, batch, ids):
    ratio = (Normal(mean, SIGMA).log_prob(batch['action'][ids]).sum(-1)-batch['old_logp'][ids]).exp()
    advantage = batch['advantage'][ids]
    value = (batch['weight'][ids]*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()
    return value, ratio


def score(actor, batch, ids, quiet_features):
    with torch.no_grad():
        full_mean = actor(batch['x'])
        minibatch_mean = actor(batch['x'][ids])
        quiet = actor(quiet_features).numpy().ravel().copy()
        full_score, full_ratio = surrogate(full_mean, batch, torch.arange(len(batch['x'])))
        minibatch_score, minibatch_ratio = surrogate(minibatch_mean, batch, ids)
        kl = float(((full_mean-batch['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
    assert all(torch.isfinite(value).all() for value in
               (full_mean, minibatch_mean, full_ratio, minibatch_ratio, full_score, minibatch_score))
    assert np.isfinite(quiet).all() and np.isfinite(kl)
    arrays = dict(quiet_predictions=quiet, collection_predictions=full_mean.numpy().copy(),
                  minibatch_predictions=minibatch_mean.numpy().copy(),
                  full_ratio=full_ratio.numpy().copy(), minibatch_ratio=minibatch_ratio.numpy().copy())
    metrics = dict(full_surrogate=float(full_score), minibatch_surrogate=float(minibatch_score),
        kl=kl, quiet48_max=float(np.abs(quiet).max()), quiet48_rms=float(np.sqrt(np.mean(quiet.astype(float)**2))),
        passes_all48_guards=bool(kl <= KL_LIMIT and np.abs(quiet).max() <= ANCHOR_LIMIT))
    return arrays, metrics


def validate_bank(bank, probes):
    features, target = np.asarray(bank['features']), np.asarray(bank['target'])
    assert features.shape == (48, 427) and features.dtype == np.float32 and np.isfinite(features).all()
    assert target.shape == (48, 1) and target.dtype == np.float32 and not target.any()
    assert not features[:, -2:].any()
    assert np.array_equal(bank['onset'], np.repeat(ONSETS, 6))
    assert np.array_equal(bank['pose'], np.tile(np.arange(6), 8))
    assert np.array_equal(bank['world'], bank['pose'])
    assert np.array_equal(bank['issue'], np.asarray(bank['onset'])+1)
    original = np.asarray(bank['original_indices'])
    assert np.array_equal(original, np.r_[np.arange(6, 12), np.arange(30, 36)])
    assert np.issubdtype(original.dtype, np.integer)
    assert np.array_equal(features[original], probes['probe_x'][np.asarray(probes['probe_branch']) == 0])
    return torch.from_numpy(features.copy()), original.astype(np.int64, copy=True)


def compute(initial_state, batch_npz, objective_npz, first_record, bank_npz, probes_npz, seed):
    """Return bounded evidence; run one historical actor Adam step and no new training."""
    assert seed in (921009, 931009) and torch.get_num_threads() == 1
    assert (first_record['update'], first_record['epoch'], first_record['batch'], first_record['minibatch_index']) == (1, 0, 0, 0)
    assert first_record['accepted'] and first_record['reasons'] == []
    assert float(batch_npz['sigma']) == SIGMA
    quiet_features, original_indices = validate_bank(bank_npz, probes_npz)
    actor = AnkleHead().eval()
    actor.load_state_dict(initial_state)
    assert all(parameter.dtype == torch.float32 and parameter.device.type == 'cpu' for parameter in actor.parameters())
    initial = deepcopy(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_RATE)
    initial_parameters = vector(actor)
    assert actor_hash(actor) == first_record['actor_pre']
    assert optimizer_hash(optimizer) == first_record['optimizer_pre'] and not optimizer.state
    with torch.no_grad():
        initial_quiet = initial(quiet_features).numpy().ravel().copy()
    assert not initial_quiet.any()
    batch = tensors(batch_npz)
    assert batch['x'].shape == (768, 427) and batch['action'].shape == (768, 1)
    indices = np.asarray(objective_npz['minibatch_indices'])
    assert indices.shape == (9, 256)
    assert all(np.array_equal(np.sort(indices[i:i+3].ravel()), np.arange(768)) for i in (0, 3, 6))
    ids = torch.from_numpy(indices[0])
    value, ratio = surrogate(actor(batch['x'][ids]), batch, ids)
    loss = -value
    assert np.array_equal(ratio.detach().numpy(), objective_npz['policy_ratio'][0])
    assert float(loss.detach()) == first_record['actor_loss'] == float(objective_npz['actor_loss'][0])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
    clipped_norm = float(torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in actor.parameters()])))
    assert gradient_norm == first_record['gradient_norm'] and clipped_norm == first_record['clipped_norm']
    initial_jacobian = quiet_jacobian(initial, quiet_features)
    assert initial_jacobian.shape[0] == 48 and np.isfinite(initial_jacobian).all()
    optimizer.step()
    tentative_parameters = vector(actor)
    proposal = tentative_parameters.double().numpy()-initial_parameters.double().numpy()
    tentative_actor_hash, tentative_optimizer_hash = actor_hash(actor), optimizer_hash(optimizer)
    assert tentative_actor_hash == first_record['actor_tentative']
    assert tentative_optimizer_hash == first_record['optimizer_tentative']
    assert float(np.linalg.norm(proposal)) == first_record['tentative_norm']
    initial_arrays, initial_metrics = score(initial, batch, ids, quiet_features)
    initial_probe_arrays, initial_probe_report = probe_compute(initial, probes_npz)
    evidence = dict(initial_parameters=initial_parameters.numpy(), tentative_parameters=tentative_parameters.numpy(),
        tentative_delta=proposal, initial_quiet_jacobian=initial_jacobian, minibatch_indices=ids.numpy(),
        **{'initial_'+key: array for key, array in initial_arrays.items()},
        **{'initial_probe_'+key: array for key, array in initial_probe_arrays.items()})
    arms = {}
    for name, anchor_indices in (('original12', original_indices), ('expanded48', np.arange(48, dtype=np.int64))):
        features = quiet_features[torch.from_numpy(anchor_indices)]
        candidate = deepcopy(initial)
        with torch.no_grad():
            target = candidate(features).numpy().ravel().copy()
        assert not target.any()
        jacobian = initial_jacobian[anchor_indices]
        _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
        kept = right[singular > CUTOFF*singular[0]]
        projected = proposal-kept.T@(kept@proposal)
        projection_gate = projection_verify(jacobian, proposal, projected, singular)
        assert projection_gate['passed']
        projected_parameters = initial_parameters+torch.from_numpy(projected.astype(np.float32))
        vector_to_parameters(projected_parameters, candidate.parameters())
        projected_actual = projected_parameters.double().numpy()-initial_parameters.double().numpy()
        projected_hash = actor_hash(candidate)
        with torch.no_grad():
            projected_anchor = candidate(features).numpy().ravel().copy()
            projected_kl = float(((candidate(batch['x'])-batch['old_mean']).square().sum(-1)/(2*SIGMA**2)).mean())
        assert np.isfinite(projected_anchor).all() and np.isfinite(projected_kl)
        candidate_jacobian = quiet_jacobian(candidate, features)
        residual = projected_anchor.astype(float)-target.astype(float)
        left, correction_singular, correction_right = np.linalg.svd(candidate_jacobian, full_matrices=False)
        keep = correction_singular > CUTOFF*correction_singular[0]
        correction = -correction_right[keep].T@((left[:, keep].T@residual)/correction_singular[keep])
        correction_gate = correction_verify(candidate_jacobian, residual, correction, correction_singular)
        assert correction_gate['passed']
        final_parameters = projected_parameters+torch.from_numpy(correction.astype(np.float32))
        vector_to_parameters(final_parameters, candidate.parameters())
        correction_actual = final_parameters.double().numpy()-projected_parameters.double().numpy()
        with torch.no_grad():
            final_anchor = candidate(features).numpy().ravel().copy()
        assert np.isfinite(final_anchor).all()
        state_arrays, metrics = score(candidate, batch, ids, quiet_features)
        probe_arrays, probe_report = probe_compute(candidate, probes_npz)
        own_anchor_max = float(np.abs(final_anchor).max())
        metrics.update(own_anchor_max=own_anchor_max,
            passes_own_guards=bool(metrics['kl'] <= KL_LIMIT and own_anchor_max <= ANCHOR_LIMIT),
            full_surrogate_change=metrics['full_surrogate']-initial_metrics['full_surrogate'],
            minibatch_surrogate_change=metrics['minibatch_surrogate']-initial_metrics['minibatch_surrogate'])
        final_hash = actor_hash(candidate)
        if name == 'original12':
            assert projected_hash == first_record['actor_projected']
            assert final_hash == first_record['actor_candidate'] == first_record['actor_post']
            assert tentative_optimizer_hash == first_record['optimizer_post']
            assert np.array_equal(singular, first_record['singular_values'])
            assert np.array_equal(target, first_record['anchor_before'])
            assert np.array_equal(projected_anchor, first_record['anchor_projected'])
            assert projected_kl == first_record['projected_kl']
            assert np.array_equal(final_anchor, first_record['anchor_attempted'])
            assert metrics['kl'] == first_record['kl'] and own_anchor_max == first_record['anchor_max']
            assert metrics['passes_own_guards'] is first_record['accepted']
            assert np.array_equal(correction_singular, first_record['retraction']['singular_values'])
            assert np.array_equal(residual, first_record['retraction']['residual'])
        local_residual = final_anchor.astype(float)-target.astype(float)
        nonlinear = final_anchor.astype(float)-projected_anchor.astype(float)-candidate_jacobian@correction_actual
        total_delta = final_parameters.double().numpy()-initial_parameters.double().numpy()
        arm_arrays = dict(anchor_indices=anchor_indices, target=target,
            projection_singular_values=singular, projected_delta=projected, actual_projected_delta=projected_actual,
            projected_parameters=projected_parameters.numpy(), projected_anchor_predictions=projected_anchor,
            projected_jacobian=candidate_jacobian, residual=residual,
            correction_singular_values=correction_singular, correction=correction, actual_correction=correction_actual,
            final_parameters=final_parameters.numpy(), anchor_predictions=final_anchor,
            **state_arrays, **{'probe_'+key: array for key, array in probe_arrays.items()})
        evidence.update({name+'_'+key: array for key, array in arm_arrays.items()})
        arms[name] = dict(anchor_count=len(anchor_indices), projected_actor_sha256=projected_hash,
            final_actor_sha256=final_hash, optimizer_sha256=tentative_optimizer_hash,
            historical_candidate_exact=name == 'original12', metrics=metrics, probes=probe_report,
            projection=dict(rank=len(kept), threshold=CUTOFF, tentative_norm=float(np.linalg.norm(proposal)),
                proposed_norm=float(np.linalg.norm(projected)), actual_norm=float(np.linalg.norm(projected_actual)),
                retained_norm_fraction=float(np.linalg.norm(projected)/np.linalg.norm(proposal)),
                rounding_norm=float(np.linalg.norm(projected_actual-projected)),
                projected_anchor_max=float(np.abs(projected_anchor).max()), projected_kl=projected_kl,
                independent=projection_gate),
            correction=dict(rank=int(keep.sum()), threshold=CUTOFF, proposed_norm=float(np.linalg.norm(correction)),
                actual_norm=float(np.linalg.norm(correction_actual)),
                rounding_norm=float(np.linalg.norm(correction_actual-correction)),
                correction_to_projected_norm=float(np.linalg.norm(correction)/max(np.linalg.norm(projected_actual), 1e-30)),
                local_residual_max=float(np.abs(local_residual).max()),
                linear_residual_max=float(np.abs(residual+candidate_jacobian@correction).max()),
                rounded_linear_residual_max=float(np.abs(residual+candidate_jacobian@correction_actual).max()),
                nonlinear_rms=float(np.sqrt(np.mean(nonlinear**2))), nonlinear_max=float(np.abs(nonlinear).max()),
                total_step_norm=float(np.linalg.norm(total_delta)), independent=correction_gate))
    assert actor_hash(actor) == tentative_actor_hash and optimizer_hash(optimizer) == tentative_optimizer_hash
    assert actor_hash(initial) == first_record['actor_pre']
    report = dict(seed=seed, actor_learning_rate=ACTOR_RATE, collection=1, minibatch_index=0,
        arm_order=list(ARM_ORDER), actor_optimizer_steps_replayed=1, new_training_updates=0,
        critic_updates=0, physical_rollouts=0, new_checkpoints=0, proposal_shared=True,
        original12_historical_parity=True, tentative_optimizer_unchanged=True,
        actor_initial_sha256=first_record['actor_pre'], actor_tentative_sha256=tentative_actor_hash,
        optimizer_initial_sha256=first_record['optimizer_pre'], optimizer_tentative_sha256=tentative_optimizer_hash,
        gradient_norm=gradient_norm, clipped_norm=clipped_norm, initial_actor_loss=float(loss.detach()),
        initial_metrics=initial_metrics, initial_probes=initial_probe_report, arms=arms,
        scope='One fixed initial Adam proposal per seed, not a PPO run. Both final candidates are scored on '
              'all48 training-only quiet states and the same24 pushed probes. Reference polarity is inherited '
              'from older weighted-policy continuations; it is not demonstrated current optimality or recovery.')
    return evidence, report
