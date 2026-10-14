"""Exact historical PPO replay with per-update measurements and no new checkpoints."""
from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from pathlib import Path
import json
from . import recovery_update_observer as observer
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


def learn(env, initial, folder, seed, sigma, original, probe_features):
    assert sigma == HALF_SIGMA
    max_kl_error = 0.
    folder.mkdir()
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    actor, critic = deepcopy(initial), AnkleHead(1)
    ao = torch.optim.Adam(actor.parameters(), lr=1e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    saved_initial = torch.load(original/'initial.pt', weights_only=True, map_location='cpu')
    assert all(torch.equal(v, saved_initial[k]) for k, v in actor.state_dict().items())
    env.reset(rng=rng)
    active = torch.zeros(16, dtype=torch.bool)
    active[TRAIN] = True
    history, start = [], time.perf_counter()
    for update in range(32):
        rows, metadata = [], []
        for _ in range(64):
            with torch.no_grad():
                x = env.features()
                mean = actor(x)
                dist = Normal(mean, sigma)
                assert torch.all(dist.scale == sigma)
                action = dist.sample()
                metadata.append(dict(force=env.force[TRAIN, 0].numpy().copy(),
                    age=(env.task.steps[TRAIN]-env.onset[TRAIN]-1).numpy().copy()))
                reward, terminal, info = env.step(action, active)
                rows.append(dict(x=x[TRAIN], action=action[TRAIN], mean=mean[TRAIN],
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
        flat = {k: v.flatten(0, 1) for k, v in batch.items()}
        evidence = {k: v.detach().numpy().copy() for k, v in batch.items()} if update in (0, 31) else None
        force, age = (np.stack([r[k] for r in metadata]) for k in ('force', 'age'))
        diagnostic = observer.before(actor, critic, batch, advantage, av, force, age, probe_features, sigma)
        stopped, accepted, rejected, kls = False, 0, 0, []
        for _ in range(3):
            for ids in torch.randperm(len(av)).split(256):
                if not stopped:
                    old_actor, old_optimizer = deepcopy(actor.state_dict()), deepcopy(ao.state_dict())
                    logp = Normal(actor(flat['x'][ids]), sigma).log_prob(flat['action'][ids]).sum(-1)
                    ratio = (logp-flat['logp'][ids]).exp()
                    loss = -torch.minimum(ratio*av[ids], ratio.clamp(.8, 1.2)*av[ids]).mean()
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
                value_loss = (critic(flat['x'][ids])-returns[ids]).square().mean()
                co.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                co.step()
        if evidence is not None:
            with torch.no_grad():
                evidence.update(advantage=advantage.numpy(), normalized_advantage=av.numpy(),
                    returns=returns.numpy(), post_mean=actor(flat['x']).numpy(), post_value=critic(flat['x']).numpy(),
                    sigma=np.array(sigma))
            with np.load(original/f'collection-{update+1}.npz', allow_pickle=False) as expected:
                assert set(evidence) == set(expected.files)
                assert all(np.array_equal(evidence[k], expected[k]) for k in evidence), update+1
        diagnostic = observer.after(actor, critic, batch, returns, diagnostic, probe_features)
        np.savez_compressed(folder/f'diagnostic-{update+1}.npz', **diagnostic)
        assert torch.isfinite(returns).all() and torch.isfinite(value_loss)
        history.append(dict(update=update+1, transitions=(update+1)*768, reward=float(batch['reward'].mean()),
            failures=int(batch['failed'].sum()), completed=int(batch['terminal'].sum()),
            actor_updates=accepted, rolled_back=rejected, accepted_kl=kls, critic_loss=float(value_loss.detach())))
        assert history[-1] == json.loads((original/'history.json').read_text())[update], (seed, update+1)
        save(folder/'history.json', history)
        if (update+1) % 4 == 0:
            print('TRAIN', seed, {k: v for k, v in history[-1].items() if k != 'accepted_kl'}, flush=True)
    for model, filename in ((actor, 'final.pt'), (critic, 'critic-final.pt')):
        expected = torch.load(original/filename, weights_only=True, map_location='cpu')
        assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items()), filename
    assert env.reset_records == json.loads((original/'resets.json').read_text())
    save(folder/'resets.json', env.reset_records)
    account = json.loads((original/'start-accounting.json').read_text())
    assert env.collected == account['collected'] == 24576
    assert env.prefix_intervals == account['source_prefix_intervals']
    assert env.exact_transfers == account['exact_transfers']
    save(folder/'parity.json', dict(passed=True, exact_initial=True, exact_final_actor=True, exact_final_critic=True,
        exact_history=True, exact_resets=True, exact_accounting=True, exact_first_last_batches=True))
    save(folder/'report.json', dict(seed=seed, transitions=24576, train_world_ids=TRAIN.tolist(),
        sigma=sigma, max_independent_kl_error=max_kl_error, train_poses=list(range(6)), updates=sum(h['actor_updates'] for h in history), wall_seconds=time.perf_counter()-start))
    return actor.eval()

