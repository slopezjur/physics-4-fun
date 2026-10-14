"""Fixed-budget retracted PPO with the sealed 48-state quiet constraint bank."""
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from . import quiet_constrained_ppo as base
from . import retracted_ppo as retracted
from .command_generalization_probes import compute as probe_compute, validate_probes
from .run_direct_recovery import TRAIN
from .run_support_skill_study import save
from .train_coordinated_ankle import AnkleHead
from .verify_quiet_coverage import projection_verify, correction_verify

HALF_SIGMA = base.HALF_SIGMA
PROJECTION_RTOL = base.PROJECTION_RTOL
ANCHOR_LIMIT = base.ANCHOR_LIMIT
KL_LIMIT = base.KL_LIMIT
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 3e-4
NOMINAL_ACTOR_SLOTS = 288
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class _FixedProbes:
    """Observe the same inputs without changing RNG, parameters or optimizer state."""

    def __init__(self, probes):
        self.probes = probes
        self.arrays = []
        self.records = []

    def record(self, update, actor, optimizer):
        assert update == len(self.records)
        actor_sha = base.actor_hash(actor)
        optimizer_sha = base.optimizer_hash(optimizer)
        rng = torch.get_rng_state().clone()
        arrays, report = probe_compute(actor, self.probes)
        assert base.actor_hash(actor) == actor_sha
        assert base.optimizer_hash(optimizer) == optimizer_sha
        assert torch.equal(torch.get_rng_state(), rng)
        self.arrays.append(arrays)
        self.records.append(dict(update=update, actor_sha256=actor_sha, optimizer_sha256=optimizer_sha,
            summary={key: value for key, value in report.items() if key != 'per_probe'}))

    def save(self, folder):
        assert len(self.records) == len(self.arrays) == 33
        np.savez_compressed(folder/'fixed-probes.npz',
            **{key: np.stack([row[key] for row in self.arrays]) for key in self.arrays[0]})
        save(folder/'fixed-probes.json', self.records)


def learn(env, initial, folder, seed, sigma, quiet_features, probe_data):
    """Retain the sealed learning schedule, changing only the frozen quiet bank."""
    assert not initial.training
    quiet_features = torch.as_tensor(quiet_features, dtype=torch.float32).detach().clone()
    assert quiet_features.shape == (48, 427) and torch.isfinite(quiet_features).all()
    assert torch.count_nonzero(quiet_features[:, -2:]) == 0
    features, metadata = validate_probes(probe_data)
    original_indices = np.r_[6:12, 30:36]
    assert np.array_equal(quiet_features.numpy()[original_indices], features[metadata['branch'] == 0])
    probes = _FixedProbes(probe_data)
    original_gate = retracted.linear_verify
    with patch.object(retracted, 'linear_verify', correction_verify):
        actor = _learn(env, initial, folder, seed, sigma, quiet_features, probes)
    assert retracted.linear_verify is original_gate
    return actor


def _learn(env, initial, folder, seed, sigma, quiet_features, probes):
    # Keep RNG ordering, collection/critic math and output evidence aligned with the sealed loop.
    assert sigma == HALF_SIGMA
    assert seed in (921009, 931009)
    comparison_folder = LOW/f'fit-{seed}'
    initial_collection = _arrays(comparison_folder/'collection-1.npz')
    max_kl_error = 0.
    folder.mkdir()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed+1)
    actor, critic = deepcopy(initial), AnkleHead(1)
    ao = torch.optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
    co = torch.optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)
    with torch.no_grad():
        target = actor(quiet_features).numpy().ravel().copy()
    assert not np.any(target)
    np.savez_compressed(folder/'quiet-anchors.npz', features=quiet_features.numpy(), target=target)
    torch.save(actor.state_dict(), folder/'initial.pt')
    probes.record(0, actor, ao)
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
        batch['actor_weight'] = base.actor_weights(batch['age'])
        if update == 0:
            expected = dict(**{key: value.numpy() for key, value in batch.items()},
                advantage=advantage.numpy(), normalized_advantage=av.numpy(), returns=returns.numpy(),
                sigma=np.array(sigma))
            assert set(expected) == set(initial_collection)-{'post_mean', 'post_value'}
            assert all(np.array_equal(value, initial_collection[key]) for key, value in expected.items())
        expected_indices = _arrays(comparison_folder/f'objective-{update+1}.npz')['minibatch_indices']
        assert expected_indices.shape == (9, 256)
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
                assert np.array_equal(ids.numpy(), expected_indices[len(indices)])
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
                    projection = retracted.retracted_step(actor, ao, quiet_features, target,
                        flat['x'], flat['mean'], sigma, projection_verify)
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
            if update == 0:
                assert np.array_equal(evidence['post_value'], initial_collection['post_value'])
            np.savez_compressed(folder/f'collection-{update+1}.npz', **evidence)
        assert torch.isfinite(returns).all() and torch.isfinite(value_loss)
        with torch.no_grad():
            anchor_max = float(np.max(np.abs(actor(quiet_features).numpy().ravel()-target)))
        assert anchor_max <= ANCHOR_LIMIT
        history.append(dict(update=update+1, transitions=(update+1)*768, reward=float(batch['reward'].mean()),
            failures=int(batch['failed'].sum()), completed=int(batch['terminal'].sum()), actor_updates=accepted,
            rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach()), weighting=exposure,
            anchor_max=anchor_max, actor_sha256=base.actor_hash(actor), optimizer_sha256=base.optimizer_hash(ao)))
        probes.record(update+1, actor, ao)
        save(folder/'history.json', history)
        if (update+1) % 4 == 0:
            print('TRAIN', seed, {k: v for k, v in history[-1].items()
                  if k not in ('accepted_kl', 'weighting', 'actor_sha256', 'optimizer_sha256')}, flush=True)
    torch.save(actor.state_dict(), folder/'final.pt')
    torch.save(critic.state_dict(), folder/'critic-final.pt')
    probes.save(folder)
    save(folder/'report.json', dict(seed=seed, transitions=24576, train_world_ids=TRAIN.tolist(),
        weighting=base.weighting_checks(), sigma=sigma, max_independent_kl_error=max_kl_error,
        train_poses=list(range(6)), updates=sum(h['actor_updates'] for h in history),
        projection_rtol=PROJECTION_RTOL, anchor_limit=ANCHOR_LIMIT, kl_limit=KL_LIMIT,
        projected_direction_renormalized=False, optimizer_restored_on_rejection=True,
        anchor_target_recomputed_from_initial=True, wall_seconds=time.perf_counter()-start,
        actor_learning_rate=ACTOR_LEARNING_RATE, critic_learning_rate=CRITIC_LEARNING_RATE,
        nominal_actor_slots=NOMINAL_ACTOR_SLOTS, quiet_anchor_count=48,
        quiet_onsets=[40, 44, 48, 54, 66, 74, 82, 90], original_quiet_indices=np.r_[6:12, 30:36].tolist(),
        independent_projection_and_correction='thin QR(J.T), then SVD(R.T)',
        initial_collection_and_critic_predictions_exact=True, first_post_critic_predictions_exact=True,
        all_collection_minibatch_permutations_exact=True, comparison_root=str(LOW),
        fixed_probes=dict(boundaries=33, initial_included=True, collection_interval=1, cases=36,
            quiet=12, pushed=24, optimizer_or_rng_mutated=False, intermediate_checkpoints=0),
        retraction=dict(target='pre-step quiet outputs', acceptance_target='original zero outputs',
            corrections_per_finite_projected_candidate=1, singular_relative_cutoff=PROJECTION_RTOL,
            intermediate_anchor_violation_rejected=False, retries=0, line_search=False,
            correction_rescaled=False, surrogate_acceptance_gate=False,
            tentative_adam_retained_only_on_acceptance=True,
            actor_step='retracted_ppo.retracted_step',
            learning_schedule='quiet_constrained_ppo.learn copied with 48 anchors and fixed-probe observation')))
    return actor.eval()
