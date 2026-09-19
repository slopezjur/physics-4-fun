"""A small, self-contained PPO.

**Why not rsl_rl.** The project's stack is the natural choice and was tried first, but this build's
config schema and TensorDict handling cost several rounds of guess-and-fail: `actor`/`critic` blocks
instead of `policy`, resolved defaults that are not constructor arguments, a NaN guard that cannot
read a TensorDict, and a normaliser calling `.var(unbiased=...)` on one. None of that is physics, and
the failures were in plumbing rather than in the thing under test.

This is ~150 lines doing standard clipped PPO with GAE, so the training loop is fully inspectable and
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
    """Gaussian actor + value critic, deliberately born as the do-nothing controller.

    **The starting policy must BE the baseline, not a random one.** Measured 2026-09-09 on this
    plant: holding every joint at its rest pose - which is what an all-zero action does, through
    position actuators at the authored kp of up to 1800 - stands for the full 40 s at 100% upright.
    A randomly initialised actor throws that away on step one, and the first run did exactly that:
    at iteration 475 the policy stood 27.9% of a QUIET 40 s where doing nothing stands 100%, having
    spent its whole capacity learning to survive its own exploration noise.

    So the last actor layer is zeroed and the initial std is small. Training starts from a controller
    that already works and can only be judged against it.
    """

    def __init__(self, num_obs, num_actions, hidden=(256, 128, 64), init_std=0.03, min_std=0.01):
        super().__init__()
        self.actor = mlp([num_obs, *hidden, num_actions])
        self.critic = mlp([num_obs, *hidden, 1])
        # Zero the output layer: the initial mean action is exactly zero for every observation, so
        # the untrained policy IS the 100%-upright rest-pose hold.
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
                 max_grad_norm=1.0, desired_kl=0.01, init_std=0.03, lr_max=1e-2):
        # **Exploration noise is measured against the plant, not inherited from a tutorial.**
        # `probe_noise.py`, 2026-09-09: zero-mean actions in a quiet room, no ball, 15 s. What
        # governs survival is the PRODUCT of std and action authority - the size of the random jump
        # in the joint target, which a position actuator at kp up to 1800 answers immediately:
        #
        #     std x authority   0.0375  0.0250  0.0125  0.0100  0.0075  0.0050
        #     time to fall       1.57s   2.33s   6.96s  10.81s   never   never
        #
        # Authority stays at 0.25 because a protective step needs the range. So std comes down to
        # 0.03 (product 0.0075, the largest value with a 100% survival rate). The usual 0.5 put the
        # body on the floor in 1.6 s - before the first ball ever fired at 2-4 s - so the first two
        # runs were teaching the policy to survive PPO rather than to survive an impact.
        #
        # `entropy_coef` drops in step. Summed over 36 dims the usual 0.005 outweighed the advantage
        # signal and drove std UP (0.500 -> 0.512 over 475 iterations) while returns rose.
        self.net = ActorCritic(num_obs, num_actions, init_std=init_std).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.max_grad_norm, self.desired_kl = max_grad_norm, desired_kl
        # Ceiling on the adaptive learning rate. Every collapse measured on 2026-09-09 was preceded
        # by a KL spike, and the surviving run actually reached this ceiling - so it is a suspect in
        # the small-batch instability rather than an inert safety bound.
        self.lr_max = lr_max
        self.lr = lr

    @torch.no_grad()
    def act(self, obs):
        d = self.dist_of(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.net.value(obs)

    def dist_of(self, obs):
        return self.net.dist(obs)

    def update(self, batch):
        obs, act, old_logp, adv, ret, old_val = batch
        # Normalising the advantage stabilises the update across wildly different return scales.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        idx = torch.randperm(n, device=obs.device)
        size = max(1, n // self.minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        count = 0

        for _ in range(self.epochs):
            for start in range(0, n, size):
                sel = idx[start:start + size]
                d = self.dist_of(obs[sel])
                logp = d.log_prob(act[sel]).sum(-1)
                ratio = (logp - old_logp[sel]).exp()
                surr1 = ratio * adv[sel]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[sel]
                policy_loss = -torch.min(surr1, surr2).mean()

                value = self.net.value(obs[sel])
                clipped = old_val[sel] + (value - old_val[sel]).clamp(-self.clip, self.clip)
                value_loss = torch.max((value - ret[sel]) ** 2,
                                       (clipped - ret[sel]) ** 2).mean()
                entropy = d.entropy().sum(-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (old_logp[sel] - logp).mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl
                count += 1

        # Adaptive learning rate on the KL, as rsl_rl does: too large a step shrinks it, too small
        # a step grows it. Without this PPO either crawls or diverges depending on the reward scale.
        mean_kl = abs(stats["kl"] / max(count, 1))
        if mean_kl > self.desired_kl * 2.0:
            self.lr = max(1e-5, self.lr / 1.5)
        elif mean_kl < self.desired_kl / 2.0:
            self.lr = min(self.lr_max, self.lr * 1.5)
        for g in self.opt.param_groups:
            g["lr"] = self.lr

        return {k: v / max(count, 1) for k, v in stats.items()}

    def compute_returns(self, rewards, dones, values, last_value):
        """GAE-lambda over a (steps, envs) rollout."""
        steps = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(last_value)
        for t in reversed(range(steps)):
            next_value = last_value if t == steps - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            last_gae = delta + self.gamma * self.lam * not_done * last_gae
            adv[t] = last_gae
        return adv, adv + values
