"""A small, self-contained PPO.

**Why not rsl_rl.** The project's stack is the natural choice and was tried first, but this build's
config schema and TensorDict handling cost several rounds of guess-and-fail: `actor`/`critic` blocks
instead of `policy`, resolved defaults that are not constructor arguments, a NaN guard that cannot
read a TensorDict, and a normaliser calling `.var(unbiased=...)` on one. None of that is physics, and
the failures were in plumbing rather than in the thing under test.

This implements clipped PPO with GAE, so the training loop is fully inspectable and
a surprising result can be attributed to the environment rather than to an unfamiliar framework.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian torque actor and independent value critic.

    Zero output means no active torque on the current plant, not a stable PD hold.
    Perturb fine-tuning must start from a validated standing controller.
    """

    def __init__(self, num_obs, num_actions, hidden=(256, 128, 64), init_std=0.03, min_std=0.01):
        super().__init__()
        self.actor = mlp([num_obs, *hidden, num_actions])
        self.critic = mlp([num_obs, *hidden, 1])
        # A fresh actor starts at zero torque. It still has to learn to stand.
        last = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # State-independent std, as rsl_rl uses: simple, and enough for a locomotion task.
        self.log_std = nn.Parameter(torch.full((num_actions,), float(np.log(init_std))))
        # A floor, because nothing else stops std collapsing to zero once the entropy bonus is this
        # small - and a policy with no exploration left cannot recover if the task changes under it.
        self.min_std = min_std

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp().clamp(min=self.min_std))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class PPO:
    def __init__(self, num_obs, num_actions, device="cpu", lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=5, minibatches=4, entropy_coef=0.0005, value_coef=1.0,
                 max_grad_norm=1.0, desired_kl=0.01, init_std=0.03, lr_max=1e-2,
                 temporal_smoothness_coef=0.0, critic_lr=3e-4, value_clip=0.2,
                 kl_stop_multiplier=1.5):
        self.net = ActorCritic(num_obs, num_actions, init_std=init_std).to(device)
        self.actor_parameters = list(self.net.actor.parameters()) + [self.net.log_std]
        self.lr = min(lr, lr_max)
        self.opt = torch.optim.Adam(self.actor_parameters, lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.net.critic.parameters(), lr=critic_lr)
        self.value_clip = value_clip
        self.kl_stop_multiplier = kl_stop_multiplier
        self.critic_warmup_remaining = 0
        self.device = device
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.max_grad_norm, self.desired_kl = max_grad_norm, desired_kl
        # Ceiling on the adaptive learning rate. Every collapse measured on 2026-09-09 was preceded
        # by a KL spike, and the surviving run actually reached this ceiling - so it is a suspect in
        # the small-batch instability rather than an inert safety bound.
        self.lr_max = lr_max
        self.temporal_smoothness_coef = temporal_smoothness_coef
        self.reference = None

    @torch.no_grad()
    def act(self, obs):
        d = self.dist_of(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.net.value(obs)

    def dist_of(self, obs):
        return self.net.dist(obs)

    def temporal_loss(self, obs, next_obs, valid):
        """L1 changes in deterministic means; never smooth through episode resets.

        Regularising means keeps exploration noise out of this objective. L1 allows
        an urgent correction without the quadratic cost of a large squared spike.
        """
        delta = (self.net.actor(obs) - self.net.actor(next_obs)).abs().mean(-1)
        return (delta * valid).sum() / valid.sum().clamp_min(1.0)

    def update(self, batch, next_obs=None, transition_valid=None, critic_only=False):
        obs, act, old_logp, adv, ret, old_val = batch
        if self.temporal_smoothness_coef and not critic_only and (next_obs is None or transition_valid is None):
            raise ValueError("Temporal smoothing requires adjacent observations and reset masks")
        # Report the values that generated the advantages before fitting these
        # same targets. Post-fit EV alone can hide a poor policy-gradient signal.
        with torch.no_grad():
            target_variance = ret.var(unbiased=False).clamp_min(1e-8)
            residual = old_val - ret
            value_ev_before = (1 - residual.var(unbiased=False) / target_variance).item()
            value_bias_before = residual.mean().item()
            value_rmse_before = residual.square().mean().sqrt().item()
            advantage_std = adv.std(unbiased=False).item()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = obs.shape[0]
        size = max(1, n // self.minibatches)
        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0,
                     temporal_loss=0.0, reference_loss=0.0, actor_updates=0, kl_stopped=0,
                     value_ev_before=value_ev_before, value_bias_before=value_bias_before,
                     value_rmse_before=value_rmse_before, advantage_std=advantage_std)
        critic_count = actor_count = 0
        actor_stopped = critic_only
        with torch.no_grad():
            old = self.dist_of(obs)
            old_mean, old_std = old.loc.detach().clone(), old.scale.detach().clone()

        for _ in range(self.epochs):
            idx = torch.randperm(n, device=obs.device)
            for start in range(0, n, size):
                sel = idx[start:start + size]
                # Separate optimisers and gradient clipping keep a large value error
                # from suppressing the actor gradient. Warm-up touches only the critic.
                value = self.net.value(obs[sel])
                value_loss = (value - ret[sel]).square().mean()
                if self.value_clip is not None and not critic_only:
                    clipped = old_val[sel] + (value - old_val[sel]).clamp(-self.value_clip, self.value_clip)
                    value_loss = torch.maximum((value - ret[sel]).square(),
                                               (clipped - ret[sel]).square()).mean()
                self.critic_opt.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()
                stats['value_loss'] += value_loss.item()
                critic_count += 1
                if actor_stopped:
                    continue

                d = self.dist_of(obs[sel])
                with torch.no_grad():
                    old_dist = torch.distributions.Normal(old_mean[sel], old_std[sel])
                    kl = torch.distributions.kl_divergence(old_dist, d).sum(-1).mean().item()
                stats['kl'] = max(stats['kl'], kl)
                if not np.isfinite(kl) or kl > self.kl_stop_multiplier * self.desired_kl:
                    actor_stopped = True
                    stats['kl_stopped'] = 1
                    continue
                logp = d.log_prob(act[sel]).sum(-1)
                ratio = (logp - old_logp[sel]).exp()
                policy_loss = -torch.minimum(ratio * adv[sel],
                    ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * adv[sel]).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = policy_loss - self.entropy_coef * entropy
                if self.temporal_smoothness_coef:
                    smooth = self.temporal_loss(obs[sel], next_obs[sel], transition_valid[sel])
                    loss = loss + self.temporal_smoothness_coef * smooth
                    stats['temporal_loss'] += smooth.item()
                if self.reference is not None and self.reference.coefficient:
                    retention = self.reference.loss(self.net.actor)
                    loss = loss + self.reference.coefficient * retention
                    stats['reference_loss'] += retention.item()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                self.opt.step()
                stats['policy_loss'] += policy_loss.item()
                stats['entropy'] += entropy.item()
                actor_count += 1

        # Include the final update, even when it was the last minibatch.
        with torch.no_grad():
            current = self.dist_of(obs)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mean, old_std), current).sum(-1).mean().item()
            stats['kl'] = max(stats['kl'], kl)
            variance = ret.var(unbiased=False)
            stats['explained_variance'] = (1 - (ret - self.net.value(obs)).var(unbiased=False)
                                          / variance.clamp_min(1e-8)).item()
        if not critic_only:
            if stats['kl_stopped'] or stats['kl'] > self.desired_kl * 2:
                # A conservative ceiling must still leave room to reduce a bad step.
                self.lr = max(min(1e-5, self.lr_max / 10), self.lr / 1.5)
            elif stats['kl'] < self.desired_kl / 2:
                self.lr = min(self.lr_max, self.lr * 1.5)
            for group in self.opt.param_groups:
                group['lr'] = self.lr
        for key in ('policy_loss', 'entropy', 'temporal_loss', 'reference_loss'):
            stats[key] /= max(actor_count, 1)
        stats['value_loss'] /= max(critic_count, 1)
        stats['actor_updates'] = actor_count
        if self.reference is not None:
            stats.update(self.reference.measure(self.net.actor))
        return stats

    def compute_returns(self, rewards, dones, values, last_value, timeout_values=None):
        """GAE-lambda over a (steps, envs) rollout."""
        steps = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(last_value)
        for t in reversed(range(steps)):
            next_value = last_value if t == steps - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            # Bootstrapping and GAE continuation are distinct: truncation uses the
            # final state's value but cannot propagate advantage across the reset.
            bootstrap = 0.0 if timeout_values is None else timeout_values[t]
            delta = rewards[t] + self.gamma * (next_value * not_done + bootstrap) - values[t]
            last_gae = delta + self.gamma * self.lam * not_done * last_gae
            adv[t] = last_gae
        return adv, adv + values
