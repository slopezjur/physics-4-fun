"""Reuse the sealed PPO schedule with one local quiet-output retraction per attempt."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from . import quiet_constrained_ppo as base
from .run_support_skill_study import save
from .verify_quiet_anchor_retraction import linear_verify

HALF_SIGMA = base.HALF_SIGMA
PROJECTION_RTOL = base.PROJECTION_RTOL
ANCHOR_LIMIT = base.ANCHOR_LIMIT
KL_LIMIT = base.KL_LIMIT


def retracted_step(actor, optimizer, quiet_features, target, collection_x, collection_mean,
                   sigma, projection_verify):
    """Project Adam's step, retract once to its local anchor target, then apply the original guards."""
    before_actor, before_optimizer = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
    before = parameters_to_vector(actor.parameters()).detach().clone()
    hashes = dict(actor_pre=base.actor_hash(actor), optimizer_pre=base.optimizer_hash(optimizer))
    with torch.no_grad():
        anchor_before = actor(quiet_features).numpy().ravel().copy()
    assert np.isfinite(anchor_before).all() and np.max(np.abs(anchor_before-target)) <= ANCHOR_LIMIT
    jacobian = base.quiet_jacobian(actor, quiet_features)
    assert np.isfinite(jacobian).all()
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    retained = right[singular > PROJECTION_RTOL*singular[0]]

    optimizer.step()
    hashes.update(actor_tentative=base.actor_hash(actor), optimizer_tentative=base.optimizer_hash(optimizer))
    tentative_parameters = parameters_to_vector(actor.parameters()).detach().clone()
    tentative = tentative_parameters.double().numpy()-before.double().numpy()
    with torch.no_grad():
        anchor_tentative = actor(quiet_features).numpy().ravel().copy()
    projected = tentative-retained.T@(retained@tentative)
    independent = projection_verify(jacobian, tentative, projected, singular) if np.isfinite(tentative).all() else None
    if independent is not None:
        assert independent['passed']
    projected_parameters = before+torch.from_numpy(projected.astype(np.float32))
    vector_to_parameters(projected_parameters, actor.parameters())
    hashes['actor_projected'] = base.actor_hash(actor)
    projected_actual = projected_parameters.double().numpy()-before.double().numpy()
    removed = tentative-projected

    with torch.no_grad():
        anchor_projected = actor(quiet_features).numpy().ravel().copy()
        projected_mean = actor(collection_x)
        projected_kl = float(((projected_mean-collection_mean).square().sum(-1)/(2*sigma**2)).mean())
        projected_finite = bool(torch.isfinite(projected_parameters).all() and torch.isfinite(projected_mean).all()
                                and np.isfinite(anchor_projected).all() and np.isfinite(projected_kl))

    retraction = None
    retraction_applied = False
    nonfinite_stage = None if projected_finite else 'projected_candidate'
    if projected_finite:
        # Local outputs define the correction; original zero targets still define acceptance.
        correction_jacobian = base.quiet_jacobian(actor, quiet_features)
        if np.isfinite(correction_jacobian).all():
            residual = anchor_projected.astype(np.float64)-anchor_before.astype(np.float64)
            left, correction_singular, correction_right = np.linalg.svd(correction_jacobian, full_matrices=False)
            keep = correction_singular > PROJECTION_RTOL*correction_singular[0]
            correction = -correction_right[keep].T@((left[:, keep].T@residual)/correction_singular[keep])
            if np.isfinite(correction).all():
                independent_correction = linear_verify(correction_jacobian, residual, correction, correction_singular)
                assert independent_correction['passed']
                corrected_parameters = projected_parameters+torch.from_numpy(correction.astype(np.float32))
                vector_to_parameters(corrected_parameters, actor.parameters())
                correction_actual = corrected_parameters.double().numpy()-projected_parameters.double().numpy()
                total_step = corrected_parameters.double().numpy()-before.double().numpy()
                with torch.no_grad():
                    corrected_anchor = actor(quiet_features).numpy().ravel().copy()
                local_residual = corrected_anchor.astype(np.float64)-anchor_before.astype(np.float64)
                correction_q = correction_jacobian@correction
                correction_actual_q = correction_jacobian@correction_actual
                nonlinear = corrected_anchor.astype(np.float64)-anchor_projected.astype(np.float64)-correction_actual_q
                retraction = dict(target=anchor_before.tolist(), residual=residual.tolist(), rank=int(keep.sum()),
                    threshold=PROJECTION_RTOL, quiet_gram=(correction_jacobian@correction_jacobian.T).tolist(),
                    singular_values=correction_singular.tolist(), q_correction=base.finite_list(correction_q),
                    q_actual=base.finite_list(correction_actual_q), correction_norm=float(np.linalg.norm(correction)),
                    actual_norm=base.finite_number(np.linalg.norm(correction_actual)),
                    rounding_norm=base.finite_number(np.linalg.norm(correction_actual-correction)),
                    correction_to_projected_norm=float(np.linalg.norm(correction)/max(np.linalg.norm(projected_actual), 1e-30)),
                    local_residual=base.finite_list(local_residual),
                    local_residual_max=base.finite_number(np.abs(local_residual).max()),
                    linear_residual_max=base.finite_number(np.abs(residual+correction_q).max()),
                    rounded_linear_residual_max=base.finite_number(np.abs(residual+correction_actual_q).max()),
                    nonlinear_rms=base.finite_number(np.sqrt(np.mean(nonlinear**2))),
                    nonlinear_max=base.finite_number(np.abs(nonlinear).max()),
                    total_step_norm=base.finite_number(np.linalg.norm(total_step)),
                    independent=independent_correction)
                retraction_applied = True
            else:
                nonfinite_stage = 'correction'
        else:
            nonfinite_stage = 'candidate_jacobian'

    hashes['actor_candidate'] = base.actor_hash(actor)
    with torch.no_grad():
        anchor_attempted = actor(quiet_features).numpy().ravel().copy()
        mean = actor(collection_x)
        candidate = parameters_to_vector(actor.parameters()).detach()
        kl = float(((mean-collection_mean).square().sum(-1)/(2*sigma**2)).mean())
        anchor_max = float(np.max(np.abs(anchor_attempted-target)))
        finite = bool(torch.isfinite(candidate).all() and torch.isfinite(mean).all()
                      and np.isfinite(anchor_attempted).all() and np.isfinite(kl))
        independent_kl = float(kl_divergence(Normal(collection_mean, sigma), Normal(mean, sigma)).sum(-1).mean()) if finite else None
    if not finite and nonfinite_stage is None:
        nonfinite_stage = 'corrected_candidate'
    kl_error = abs(kl-independent_kl) if finite else None
    assert kl_error is None or kl_error < 2e-7
    reasons = []
    if not finite or not retraction_applied:
        reasons.append('nonfinite')
    if not np.isfinite(kl) or kl > KL_LIMIT:
        reasons.append('kl')
    if not np.isfinite(anchor_max) or anchor_max > ANCHOR_LIMIT:
        reasons.append('anchor')
    accepted = not reasons
    if not accepted:
        actor.load_state_dict(before_actor)
        optimizer.load_state_dict(before_optimizer)
    hashes.update(actor_post=base.actor_hash(actor), optimizer_post=base.optimizer_hash(optimizer))
    if accepted:
        assert hashes['actor_post'] == hashes['actor_candidate']
        assert hashes['optimizer_post'] == hashes['optimizer_tentative']
    else:
        assert hashes['actor_post'] == hashes['actor_pre']
        assert hashes['optimizer_post'] == hashes['optimizer_pre']
    with torch.no_grad():
        anchor_after = actor(quiet_features).numpy().ravel().copy()
    assert np.isfinite(anchor_after).all() and np.max(np.abs(anchor_after-target)) <= ANCHOR_LIMIT
    return dict(rank=len(retained), singular_values=singular.tolist(), quiet_gram=(jacobian@jacobian.T).tolist(),
        q_tentative=base.finite_list(jacobian@tentative), q_projected=base.finite_list(jacobian@projected),
        q_actual=base.finite_list(jacobian@projected_actual), tentative_norm=base.finite_number(np.linalg.norm(tentative)),
        projected_norm=base.finite_number(np.linalg.norm(projected)), removed_norm=base.finite_number(np.linalg.norm(removed)),
        removed_dot_projected=base.finite_number(removed@projected), actual_norm=base.finite_number(np.linalg.norm(projected_actual)),
        rounding_norm=base.finite_number(np.linalg.norm(projected_actual-projected)),
        anchor_before=base.finite_list(anchor_before), anchor_tentative=base.finite_list(anchor_tentative),
        anchor_projected=base.finite_list(anchor_projected), anchor_attempted=base.finite_list(anchor_attempted),
        anchor_after=base.finite_list(anchor_after), projected_kl=base.finite_number(projected_kl),
        projected_anchor_max=base.finite_number(np.abs(anchor_projected-target).max()),
        projected_finite=projected_finite, retraction_applied=retraction_applied, nonfinite_stage=nonfinite_stage,
        retraction=retraction, kl=base.finite_number(kl), anchor_max=base.finite_number(anchor_max),
        independent_kl=independent_kl, independent_kl_error=kl_error, accepted=accepted, reasons=reasons,
        independent=independent, **hashes)


def learn(env, initial, folder, seed, sigma, quiet_features):
    """Keep the sealed collection/critic schedule, replacing only its actor-step callable."""
    original_step = base.constrained_step
    with patch.object(base, 'constrained_step', retracted_step):
        actor = base.learn(env, initial, folder, seed, sigma, quiet_features)
    assert base.constrained_step is original_step
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    report['retraction'] = dict(target='pre-step quiet outputs', acceptance_target='original zero outputs',
        corrections_per_finite_projected_candidate=1, singular_relative_cutoff=PROJECTION_RTOL,
        intermediate_anchor_violation_rejected=False, retries=0, line_search=False,
        correction_rescaled=False, surrogate_acceptance_gate=False,
        tentative_adam_retained_only_on_acceptance=True,
        unchanged_rollout_and_critic_loop='quiet_constrained_ppo.learn')
    save(folder/'report.json', report)
    return actor
