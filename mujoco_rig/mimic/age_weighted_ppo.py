"""Bounded PPO with collection-normalized early-recovery actor weighting."""
from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from .run_direct_recovery import TRAIN
from .run_support_skill_study import save
from .train_coordinated_ankle import AnkleHead, SIGMA

HALF_SIGMA = SIGMA/2


def probability_checks():
    mean = torch.tensor([[.1], [-.2]], dtype=torch.float64)
    action = torch.tensor([[.25], [.05]], dtype=torch.float64)
    other = mean+.01
    def manual(mu):
        return -.5*((action-mu)/HALF_SIGMA)**2-np.log(HALF_SIGMA)-.5*np.log(2*np.pi)
    old, new = Normal(mean, HALF_SIGMA), Normal(other, HALF_SIGMA)
    assert torch.allclose(old.log_prob(action), manual(mean), atol=1e-12, rtol=0)
    assert torch.allclose((new.log_prob(action)-old.log_prob(action)).exp(),
                          (manual(other)-manual(mean)).exp(), atol=1e-12, rtol=0)
    kl = ((other-mean)**2/(2*HALF_SIGMA**2)).sum(-1).mean()
    assert torch.allclose(kl, kl_divergence(old, new).sum(-1).mean(), atol=1e-12, rtol=0)
    return dict(passed=True, sigma=HALF_SIGMA, analytical_logp_ratio_kl=True)


def actor_weights(age):
    raw = torch.where(age < 30, 5., 1.)
    return raw / raw.mean()


def weighting_checks():
    age = torch.tensor([[0, 29, 30, 100], [30, 0, 31, 200]])
    weights = actor_weights(age)
    assert torch.allclose(weights.mean(), torch.tensor(1.))
    assert torch.allclose(weights[age < 30], 5*weights[age >= 30][0])
    assert torch.equal(actor_weights(torch.zeros(64, 12)), torch.ones(64, 12))
    assert torch.equal(actor_weights(torch.full((64, 12), 30)), torch.ones(64, 12))
    return dict(passed=True, early_age_intervals=30, raw_early_weight=5., raw_late_weight=1.,
                normalization='whole collection mean one', metadata_only=True)


def learn(env, initial, folder, seed, sigma):
    assert sigma == HALF_SIGMA
    max_kl_error = 0.
    folder.mkdir()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    actor, critic = deepcopy(initial), AnkleHead(1)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
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
            delta = batch['reward'][t] + .995*batch['next_value'][t]*(~batch['failed'][t]) - batch['value'][t]
            running = delta + .995*.98*(~batch['terminal'][t])*running
            advantage[t] = running
        returns = (advantage + batch['value']).flatten()
        av = advantage.flatten()
        av = (av-av.mean())/av.std().clamp_min(1e-6)
        batch['actor_weight'] = actor_weights(batch['age'])
        flat = {k: v.flatten(0, 1) for k, v in batch.items()}
        evidence = {k: v.detach().numpy().copy() for k, v in batch.items()} if update in (0, 31) else None
        stopped, accepted, rejected, kls = False, 0, 0, []
        objective = {key: batch[key].numpy().copy() for key in
                     ('age', 'force', 'terminal', 'failed', 'actor_weight')}
        objective['normalized_advantage'] = av.numpy().copy()
        indices, ratios, actor_losses, residuals, critic_losses = [], [], [], [], []
        for _ in range(3):
            for ids in torch.randperm(len(av)).split(256):
                indices.append(ids.numpy().copy())
                ratios.append(np.full(256, np.nan, dtype=np.float32))
                actor_losses.append(np.nan)
                if not stopped:
                    old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(ao.state_dict())
                    logp = Normal(actor(flat['x'][ids]), sigma).log_prob(flat['action'][ids]).sum(-1)
                    ratio = (logp-flat['logp'][ids]).exp()
                    loss = -(flat['actor_weight'][ids] *
                             torch.minimum(ratio*av[ids], ratio.clamp(.8, 1.2)*av[ids])).mean()
                    ratios[-1] = ratio.detach().numpy().copy()
                    actor_losses[-1] = float(loss.detach())
                    ao.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    ao.step()
                    with torch.no_grad():
                        kl = float(((actor(flat['x'])-flat['mean']).square().sum(-1)/(2*sigma**2)).mean())
                        independent_kl = float(kl_divergence(Normal(flat['mean'], sigma), Normal(actor(flat['x']), sigma)).sum(-1).mean())
                        max_kl_error = max(max_kl_error, abs(kl-independent_kl))
                        assert abs(kl-independent_kl) < 2e-7
                    if not np.isfinite(kl) or kl > .01:
                        actor.load_state_dict(old_actor)
                        ao.load_state_dict(old_optimizer)
                        stopped, rejected = True, rejected+1
                    else:
                        accepted += 1
                        kls.append(kl)
                residual = critic(flat['x'][ids])-returns[ids]
                value_loss = residual.square().mean()
                residuals.append(residual.detach().numpy().copy())
                critic_losses.append(float(value_loss.detach()))
                co.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                co.step()
        objective.update(minibatch_indices=np.stack(indices), policy_ratio=np.stack(ratios),
            actor_loss=np.array(actor_losses), critic_residual=np.stack(residuals),
            critic_loss=np.array(critic_losses))
        np.savez_compressed(folder/f'objective-{update+1}.npz', **objective)
        early = batch['age'] < 30
        exposure = dict(early_samples=int(early.sum()), late_samples=int((~early).sum()),
            early_weight_mass=float(batch['actor_weight'][early].sum()),
            late_weight_mass=float(batch['actor_weight'][~early].sum()),
            mean_weight=float(batch['actor_weight'].mean()), max_weight=float(batch['actor_weight'].max()),
            branches={label: dict(samples=int(mask.sum()),
                weight_mass=float(batch['actor_weight'][mask].sum())) for label, mask in
                [('quiet', batch['force'] == 0), ('positive_x', batch['force'] > 0),
                 ('negative_x', batch['force'] < 0)]})
        if evidence is not None:
            with torch.no_grad():
                evidence.update(advantage=advantage.numpy(), normalized_advantage=av.numpy(),
                    returns=returns.numpy(), post_mean=actor(flat['x']).numpy(), post_value=critic(flat['x']).numpy(),
                    sigma=np.array(sigma))
            np.savez_compressed(folder/f'collection-{update+1}.npz', **evidence)
        assert torch.isfinite(returns).all() and torch.isfinite(value_loss)
        history.append(dict(update=update+1, transitions=(update+1)*768, reward=float(batch['reward'].mean()),
            failures=int(batch['failed'].sum()), completed=int(batch['terminal'].sum()),
            actor_updates=accepted, rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach()),
            weighting=exposure))
        save(folder/'history.json', history)
        if (update+1) % 4 == 0:
            print('TRAIN', seed, {k: v for k, v in history[-1].items() if k not in ('accepted_kl', 'weighting')}, flush=True)
    torch.save(actor.state_dict(), folder/'final.pt')
    torch.save(critic.state_dict(), folder/'critic-final.pt')
    save(folder/'report.json', dict(seed=seed, transitions=24576, train_world_ids=TRAIN.tolist(),
        weighting=weighting_checks(), sigma=sigma, max_independent_kl_error=max_kl_error, train_poses=list(range(6)), updates=sum(h['actor_updates'] for h in history), wall_seconds=time.perf_counter()-start))
    return actor.eval()

