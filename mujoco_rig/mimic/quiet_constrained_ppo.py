"""Age-weighted PPO with projected Adam displacements and fixed quiet anchors."""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .age_weighted_ppo import HALF_SIGMA, actor_weights, probability_checks, weighting_checks
from .run_direct_recovery import TRAIN
from .run_support_skill_study import save
from .train_coordinated_ankle import AnkleHead

PROJECTION_RTOL = 1e-6
ANCHOR_LIMIT = 1e-5
KL_LIMIT = .01


def actor_hash(actor):
    value = parameters_to_vector(actor.parameters()).detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def optimizer_hash(optimizer):
    def encode(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                        sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value
    encoded = json.dumps(encode(optimizer.state_dict()), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def quiet_jacobian(actor, quiet_features):
    rows = []
    for feature in quiet_features:
        parts = torch.autograd.grad(actor(feature[None]).sum(), tuple(actor.parameters()))
        rows.append(torch.cat([part.flatten() for part in parts]).detach().double().numpy())
    return np.stack(rows)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def finite_list(value):
    return [finite_number(item) for item in np.asarray(value).reshape(-1)]


def constrained_step(actor, optimizer, quiet_features, target, collection_x, collection_mean,
                     sigma, projection_verify):
    """Tentatively advance Adam, constrain its actual displacement, and guard the result."""
    before_actor, before_optimizer = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
    before = parameters_to_vector(actor.parameters()).detach().clone()
    hashes = dict(actor_pre=actor_hash(actor), optimizer_pre=optimizer_hash(optimizer))
    with torch.no_grad():
        anchor_before = actor(quiet_features).numpy().ravel().copy()
    jacobian = quiet_jacobian(actor, quiet_features)
    assert np.isfinite(jacobian).all()
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    retained = right[singular > PROJECTION_RTOL*singular[0]]
    optimizer.step()
    hashes.update(actor_tentative=actor_hash(actor), optimizer_tentative=optimizer_hash(optimizer))
    tentative_parameters = parameters_to_vector(actor.parameters()).detach().clone()
    tentative = tentative_parameters.double().numpy()-before.double().numpy()
    with torch.no_grad():
        anchor_tentative = actor(quiet_features).numpy().ravel().copy()
    projected = tentative-retained.T@(retained@tentative)
    # The independent gate sees the actual Adam displacement before any constrained parameter write.
    independent = projection_verify(jacobian, tentative, projected, singular) if np.isfinite(tentative).all() else None
    if independent is not None:
        assert independent['passed']
    candidate = before+torch.from_numpy(projected.astype(np.float32))
    vector_to_parameters(candidate, actor.parameters())
    hashes['actor_candidate'] = actor_hash(actor)
    actual = candidate.double().numpy()-before.double().numpy()
    removed = tentative-projected
    with torch.no_grad():
        anchor_attempted = actor(quiet_features).numpy().ravel().copy()
        mean = actor(collection_x)
        kl = float(((mean-collection_mean).square().sum(-1)/(2*sigma**2)).mean())
        anchor_max = float(np.max(np.abs(anchor_attempted-target)))
        finite = bool(torch.isfinite(candidate).all() and torch.isfinite(mean).all()
                      and np.isfinite(anchor_attempted).all() and np.isfinite(kl))
        independent_kl = float(kl_divergence(Normal(collection_mean, sigma), Normal(mean, sigma)).sum(-1).mean()) if finite else None
    kl_error = abs(kl-independent_kl) if finite else None
    assert kl_error is None or kl_error < 2e-7
    reasons = []
    if not finite:
        reasons.append('nonfinite')
    if not np.isfinite(kl) or kl > KL_LIMIT:
        reasons.append('kl')
    if not np.isfinite(anchor_max) or anchor_max > ANCHOR_LIMIT:
        reasons.append('anchor')
    accepted = not reasons
    if not accepted:
        actor.load_state_dict(before_actor)
        optimizer.load_state_dict(before_optimizer)
    hashes.update(actor_post=actor_hash(actor), optimizer_post=optimizer_hash(optimizer))
    if accepted:
        assert hashes['actor_post'] == hashes['actor_candidate']
        assert hashes['optimizer_post'] == hashes['optimizer_tentative']
    else:
        assert hashes['actor_post'] == hashes['actor_pre']
        assert hashes['optimizer_post'] == hashes['optimizer_pre']
    with torch.no_grad():
        anchor_after = actor(quiet_features).numpy().ravel().copy()
    assert np.max(np.abs(anchor_after-target)) <= ANCHOR_LIMIT
    record = dict(rank=len(retained), singular_values=singular.tolist(), quiet_gram=(jacobian@jacobian.T).tolist(),
        q_tentative=finite_list(jacobian@tentative), q_projected=finite_list(jacobian@projected),
        q_actual=finite_list(jacobian@actual), tentative_norm=finite_number(np.linalg.norm(tentative)),
        projected_norm=finite_number(np.linalg.norm(projected)), removed_norm=finite_number(np.linalg.norm(removed)),
        removed_dot_projected=finite_number(removed@projected), actual_norm=finite_number(np.linalg.norm(actual)),
        rounding_norm=finite_number(np.linalg.norm(actual-projected)),
        anchor_before=finite_list(anchor_before), anchor_tentative=finite_list(anchor_tentative),
        anchor_attempted=finite_list(anchor_attempted), anchor_after=finite_list(anchor_after),
        kl=finite_number(kl), anchor_max=finite_number(anchor_max), independent_kl=independent_kl,
        independent_kl_error=kl_error, accepted=accepted, reasons=reasons, independent=independent, **hashes)
    return record


def learn(env, initial, folder, seed, sigma, quiet_features):
    from .verify_quiet_constrained_recovery import projection_verify

    assert sigma == HALF_SIGMA
    quiet_features = torch.as_tensor(quiet_features, dtype=torch.float32).detach().clone()
    assert quiet_features.shape == (12, 427) and torch.isfinite(quiet_features).all()
    assert torch.count_nonzero(quiet_features[:, -2:]) == 0
    max_kl_error = 0.
    folder.mkdir()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed+1)
    actor, critic = deepcopy(initial), AnkleHead(1)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    with torch.no_grad():
        target = actor(quiet_features).numpy().ravel().copy()
    assert not np.any(target)
    np.savez_compressed(folder/'quiet-anchors.npz', features=quiet_features.numpy(), target=target)
    torch.save(actor.state_dict(), folder/'initial.pt')
    env.reset(rng=rng)
    active = torch.zeros(16, dtype=torch.bool)
    active[TRAIN] = True
    history, start = [], time.perf_counter()
    for update in range(32):
        rows = []
        for _ in range(64):
            with torch.no_grad():
                x = env.features()
                mean = actor(x)
                dist = Normal(mean, sigma)
                assert torch.all(dist.scale == sigma)
                action = dist.sample()
                age = (env.task.steps[TRAIN]-env.onset[TRAIN]-1).clone()
                force = env.force[TRAIN, 0].clone()
                assert (age >= 0).all()
                reward, terminal, info = env.step(action, active)
                rows.append(dict(age=age, force=force, x=x[TRAIN], action=action[TRAIN], mean=mean[TRAIN],
                    logp=dist.log_prob(action).sum(-1)[TRAIN], value=critic(x)[TRAIN], reward=reward[TRAIN],
                    terminal=terminal[TRAIN], failed=info['failed'][TRAIN], next_value=critic(env.features())[TRAIN]))
                ids = torch.nonzero(active & terminal).flatten()
                if len(ids):
                    env.reset(ids, rng)
        batch = {k: torch.stack([row[k] for row in rows]) for k in rows[0]}
        advantage = torch.zeros_like(batch['reward'])
        running = torch.zeros(len(TRAIN))
        for t in reversed(range(64)):
            delta = batch['reward'][t]+.995*batch['next_value'][t]*(~batch['failed'][t])-batch['value'][t]
            running = delta+.995*.98*(~batch['terminal'][t])*running
            advantage[t] = running
        returns = (advantage+batch['value']).flatten()
        av = advantage.flatten()
        av = (av-av.mean())/av.std().clamp_min(1e-6)
        batch['actor_weight'] = actor_weights(batch['age'])
        flat = {k: v.flatten(0, 1) for k, v in batch.items()}
        evidence = {k: v.detach().numpy().copy() for k, v in batch.items()} if update in (0, 31) else None
        stopped, accepted, rejected, kls = False, 0, 0, []
        projections = []
        objective = {key: batch[key].numpy().copy() for key in
                     ('age', 'force', 'terminal', 'failed', 'actor_weight')}
        objective['normalized_advantage'] = av.numpy().copy()
        indices, ratios, actor_losses, residuals, critic_losses = [], [], [], [], []
        for epoch in range(3):
            for minibatch, ids in enumerate(torch.randperm(len(av)).split(256)):
                indices.append(ids.numpy().copy())
                ratios.append(np.full(256, np.nan, dtype=np.float32))
                actor_losses.append(np.nan)
                if not stopped:
                    logp = Normal(actor(flat['x'][ids]), sigma).log_prob(flat['action'][ids]).sum(-1)
                    ratio = (logp-flat['logp'][ids]).exp()
                    loss = -(flat['actor_weight'][ids]*
                             torch.minimum(ratio*av[ids], ratio.clamp(.8, 1.2)*av[ids])).mean()
                    ratios[-1] = ratio.detach().numpy().copy()
                    actor_losses[-1] = float(loss.detach())
                    ao.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = float(nn.utils.clip_grad_norm_(actor.parameters(), .5))
                    clipped_norm = float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actor.parameters()])))
                    projection = constrained_step(actor, ao, quiet_features, target, flat['x'], flat['mean'], sigma, projection_verify)
                    projection.update(update=update+1, epoch=epoch, batch=minibatch, minibatch_index=len(indices)-1,
                                      actor_loss=actor_losses[-1], gradient_norm=gradient_norm, clipped_norm=clipped_norm)
                    projections.append(projection)
                    if projection['independent_kl_error'] is not None:
                        max_kl_error = max(max_kl_error, projection['independent_kl_error'])
                    if not projection['accepted']:
                        stopped, rejected = True, rejected+1
                    else:
                        accepted += 1
                        kls.append(projection['kl'])
                residual = critic(flat['x'][ids])-returns[ids]
                value_loss = residual.square().mean()
                residuals.append(residual.detach().numpy().copy())
                critic_losses.append(float(value_loss.detach()))
                co.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                co.step()
        objective.update(minibatch_indices=np.stack(indices), policy_ratio=np.stack(ratios),
            actor_loss=np.array(actor_losses), critic_residual=np.stack(residuals), critic_loss=np.array(critic_losses))
        np.savez_compressed(folder/f'objective-{update+1}.npz', **objective)
        save(folder/f'projection-{update+1}.json', projections)
        early = batch['age'] < 30
        exposure = dict(early_samples=int(early.sum()), late_samples=int((~early).sum()),
            early_weight_mass=float(batch['actor_weight'][early].sum()), late_weight_mass=float(batch['actor_weight'][~early].sum()),
            mean_weight=float(batch['actor_weight'].mean()), max_weight=float(batch['actor_weight'].max()),
            branches={label: dict(samples=int(mask.sum()), weight_mass=float(batch['actor_weight'][mask].sum()))
                for label, mask in [('quiet', batch['force'] == 0), ('positive_x', batch['force'] > 0),
                                   ('negative_x', batch['force'] < 0)]})
        if evidence is not None:
            with torch.no_grad():
                evidence.update(advantage=advantage.numpy(), normalized_advantage=av.numpy(),
                    returns=returns.numpy(), post_mean=actor(flat['x']).numpy(), post_value=critic(flat['x']).numpy(),
                    sigma=np.array(sigma))
            np.savez_compressed(folder/f'collection-{update+1}.npz', **evidence)
        assert torch.isfinite(returns).all() and torch.isfinite(value_loss)
        with torch.no_grad():
            anchor_max = float(np.max(np.abs(actor(quiet_features).numpy().ravel()-target)))
        assert anchor_max <= ANCHOR_LIMIT
        history.append(dict(update=update+1, transitions=(update+1)*768, reward=float(batch['reward'].mean()),
            failures=int(batch['failed'].sum()), completed=int(batch['terminal'].sum()), actor_updates=accepted,
            rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach()), weighting=exposure,
            anchor_max=anchor_max, actor_sha256=actor_hash(actor), optimizer_sha256=optimizer_hash(ao)))
        save(folder/'history.json', history)
        if (update+1) % 4 == 0:
            print('TRAIN', seed, {k: v for k, v in history[-1].items()
                  if k not in ('accepted_kl', 'weighting', 'actor_sha256', 'optimizer_sha256')}, flush=True)
    torch.save(actor.state_dict(), folder/'final.pt')
    torch.save(critic.state_dict(), folder/'critic-final.pt')
    save(folder/'report.json', dict(seed=seed, transitions=24576, train_world_ids=TRAIN.tolist(),
        weighting=weighting_checks(), sigma=sigma, max_independent_kl_error=max_kl_error,
        train_poses=list(range(6)), updates=sum(h['actor_updates'] for h in history),
        projection_rtol=PROJECTION_RTOL, anchor_limit=ANCHOR_LIMIT, kl_limit=KL_LIMIT,
        projected_direction_renormalized=False, optimizer_restored_on_rejection=True,
        anchor_target_recomputed_from_initial=True, wall_seconds=time.perf_counter()-start))
    return actor.eval()
