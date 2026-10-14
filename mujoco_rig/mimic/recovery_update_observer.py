"""RNG-neutral per-collection diagnostics for an exact historical PPO replay."""
import numpy as np
import torch
from torch.distributions import Normal

GROUPS = ('quiet', '+X', '-X', 'early', 'late')


def masks(force, age):
    force, age = force.flatten(), age.flatten()
    return [force == 0, force > 0, force < 0, age < 30, age >= 30]


def state_guard(actor):
    return torch.get_rng_state().clone(), [p.grad.clone() if p.grad is not None else None for p in actor.parameters()]


def assert_guard(actor, guard):
    rng, grads = guard
    assert torch.equal(torch.get_rng_state(), rng)
    for param, previous in zip(actor.parameters(), grads):
        assert (param.grad is None) if previous is None else (param.grad is not None and torch.equal(param.grad, previous))


@torch.no_grad()
def probes(actor, critic, features):
    x = features.reshape(-1, 427)
    shape = features.shape[:-1]
    return actor(x).reshape(shape).numpy().copy(), critic(x).reshape(shape).numpy().copy()


def before(actor, critic, batch, advantage, normalized, force, age, probe_features, sigma):
    guard = state_guard(actor)
    x = batch['x'].flatten(0, 1)
    actions = batch['action'].flatten(0, 1)
    group_masks = masks(force, age)
    gradients = []
    for mask in group_masks:
        logp = Normal(actor(x), sigma).log_prob(actions).sum(-1)
        objective = (normalized*logp*torch.from_numpy(mask)).sum()/len(normalized)
        values = torch.autograd.grad(objective, tuple(actor.parameters()))
        gradients.append(torch.cat([v.flatten() for v in values]).detach().numpy())
    gradients = np.stack(gradients).astype(float)
    with torch.no_grad():
        hidden = actor.net[:-1](x).numpy().astype(float)
        score = ((actions-batch['mean'].flatten(0, 1))/sigma**2).numpy().flatten().astype(float)
        score *= normalized.numpy()
        design = np.c_[hidden, np.ones(len(hidden))]
        analytic = np.stack([.1*(design[mask].T@score[mask])/len(score) for mask in group_masks])
        # Earlier layers may now be nonzero; this independent identity is only for the final layer.
        assert np.allclose(analytic, gradients[:, -129:], atol=5e-6, rtol=5e-5)
        reward_gae = torch.zeros_like(advantage)
        running = torch.zeros_like(advantage[0])
        for t in reversed(range(64)):
            running = batch['reward'][t]+.995*.98*(~batch['terminal'][t])*running
            reward_gae[t] = running
        probe_mean, probe_value = probes(actor, critic, probe_features)
    norm = np.linalg.norm(gradients, axis=1)
    gram = gradients@gradients.T
    assert np.allclose(gradients[:3].sum(0), gradients[3:].sum(0), atol=5e-6, rtol=5e-5)
    data = {k: batch[k].numpy().copy() for k in ('reward', 'terminal', 'failed', 'value', 'next_value', 'mean', 'action', 'logp')}
    data.update(force=force, age=age, previous_latent=batch['x'][:, :, -1].numpy().copy(),
        advantage=advantage.numpy().copy(), normalized_advantage=normalized.numpy().copy(),
        reward_gae=reward_gae.numpy(), value_gae=(advantage-reward_gae).numpy(),
        gradient_gram=gram, gradient_norm=norm, gradient_last=gradients[:, -129:],
        gradient_last_analytic=analytic, group_counts=np.array([m.sum() for m in group_masks]),
        probe_mean_before=probe_mean, probe_value_before=probe_value)
    assert_guard(actor, guard)
    return data


def after(actor, critic, batch, returns, data, probe_features):
    guard = state_guard(actor)
    with torch.no_grad():
        x = batch['x'].flatten(0, 1)
        data['post_mean'] = actor(x).reshape(64, 12, 1).numpy().copy()
        data['post_value'] = critic(x).reshape(64, 12).numpy().copy()
        data['returns'] = returns.reshape(64, 12).numpy().copy()
        data['probe_mean_after'], data['probe_value_after'] = probes(actor, critic, probe_features)
    assert_guard(actor, guard)
    for value in data.values():
        assert np.isfinite(value).all()
    return data
