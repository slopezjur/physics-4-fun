"""Exact first-collection optimizer replay and branch gradient decomposition."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead, ONSET_FRAMES

BASE = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-ppo-gradients-20261009-01')
WORLDS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
BRANCHES = ('quiet', '+X', '-X')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def labels(seed, batch, records):
    rng = torch.Generator().manual_seed(seed+1)
    force, age = np.zeros(12), np.zeros(12, dtype=int)
    cursor, calls = 0, []

    def reset(worlds, frame):
        nonlocal cursor
        n = len(worlds)
        sign = 2*torch.randint(2, (n,), generator=rng)-1
        quiet = torch.rand(n, generator=rng) < .25
        onset = ONSET_FRAMES[torch.randint(len(ONSET_FRAMES), (n,), generator=rng)]
        expected = [dict(world=w, pose=w % 8, force_n=[float(150*sign[j]*(~quiet[j])), 0., 0.],
                         onset_frame=int(onset[j]), start_frame=int(onset[j])+1) for j, w in enumerate(worlds)]
        expected.sort(key=lambda r: (r['onset_frame'], r['world']))
        assert records[cursor:cursor+n] == expected, (seed, frame, cursor)
        calls.append(dict(before_frame=frame, worlds=worlds, record_start=cursor, count=n))
        cursor += n
        for row in expected:
            j = WORLDS.index(row['world'])
            force[j], age[j] = row['force_n'][0], 0

    reset(WORLDS, 0)
    forces, ages = [], []
    for t in range(64):
        forces.append(force.copy())
        ages.append(age.copy())
        fresh = age == 0
        assert not batch['x'][t, fresh, -1].any()
        assert not batch['x'][t, fresh, -2].any()
        if t:
            assert np.array_equal(batch['x'][t, ~fresh, -1], batch['action'][t-1, ~fresh, 0])
        age += 1
        ended = [w for j, w in enumerate(WORLDS) if batch['terminal'][t, j]]
        if ended:
            reset(ended, t+1)
    force = np.stack(forces)
    branch = np.where(force == 0, 0, np.where(force > 0, 1, 2))
    return branch, np.stack(ages), dict(records_consumed=cursor, reset_calls=calls)


def replay(seed, initial, batch, history, sigma):
    actor = deepcopy(initial)
    torch.manual_seed(seed)
    critic = AnkleHead(1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for t in range(64):
        mean = torch.zeros(16, 1)
        mean[WORLDS] = torch.from_numpy(batch['mean'][t])
        action = Normal(mean, sigma).sample()
        assert np.array_equal(action[WORLDS].numpy(), batch['action'][t]), (seed, t)
    permutation_state = torch.get_rng_state().clone()
    x = torch.from_numpy(batch['x']).flatten(0, 1)
    actions = torch.from_numpy(batch['action']).flatten(0, 1)
    old_mean = torch.from_numpy(batch['mean']).flatten(0, 1)
    old_logp = torch.from_numpy(batch['logp']).flatten()
    advantage = torch.from_numpy(batch['normalized_advantage'])
    returns = torch.from_numpy(batch['returns'])
    with torch.no_grad():
        assert torch.equal(actor(x), old_mean)
        assert not critic(x).any() and not batch['value'].any() and not batch['next_value'].any()
    orders, kls = [], []
    for _ in range(3):
        order = torch.randperm(768)
        orders.append(order.numpy().copy())
        for ids in order.split(256):
            ratio = (Normal(actor(x[ids]), sigma).log_prob(actions[ids]).sum(-1)-old_logp[ids]).exp()
            loss = -torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids]).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), .5)
            actor_optimizer.step()
            with torch.no_grad():
                kl = float(((actor(x)-old_mean).square().sum(-1)/(2*sigma**2)).mean())
            assert np.isfinite(kl) and kl <= .01
            kls.append(kl)
            value_loss = (critic(x[ids])-returns[ids]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            critic_optimizer.step()
    with torch.no_grad():
        mean, value = actor(x).numpy(), critic(x).numpy()
    assert np.array_equal(mean, batch['post_mean']), (seed, 'actor', np.abs(mean-batch['post_mean']).max())
    assert np.array_equal(value, batch['post_value']), (seed, 'critic')
    assert kls == history[0]['accepted_kl'] and history[0]['actor_updates'] == 9 and history[0]['rolled_back'] == 0
    return dict(replay_mean=mean, replay_value=value, permutations=np.stack(orders), kl=np.array(kls),
                permutation_rng=permutation_state.numpy()), dict(exact_actions=True, exact_actor=True,
                exact_critic=True, exact_kl=True, actor_steps=9, critic_steps=9)


def gradient_analysis(initial, batch, branch, sigma):
    x = torch.from_numpy(batch['x']).flatten(0, 1)
    action = torch.from_numpy(batch['action']).flatten(0, 1)
    advantage = torch.from_numpy(batch['normalized_advantage'])
    group = branch.flatten()
    with torch.no_grad():
        hidden = initial.net[:-1](x).numpy()
    gradients = []
    for mask in [np.ones(768, dtype=bool), *[group == i for i in range(3)]]:
        initial.zero_grad(set_to_none=True)
        logp = Normal(initial(x), sigma).log_prob(action).sum(-1)
        objective = (advantage*logp*torch.from_numpy(mask)).sum()/768
        objective.backward()
        # At zero output weights, only the final layer has a first-order gradient.
        assert all(not p.grad.any() for p in list(initial.parameters())[:-2])
        gradients.append(np.r_[initial.net[-1].weight.grad.numpy().ravel(), initial.net[-1].bias.grad.numpy()])
    gradients = np.stack(gradients)
    score = (batch['action'].flatten().astype(float)-batch['mean'].flatten())/sigma**2*advantage.numpy()
    design = np.c_[hidden, np.ones(768)]
    analytic = np.stack([.1*(design*score[:, None]*mask[:, None]).sum(0)/768
                         for mask in [np.ones(768, bool), *[group == i for i in range(3)]]])
    assert np.allclose(gradients, analytic, atol=3e-6, rtol=3e-5)
    assert np.allclose(gradients[0], gradients[1:].sum(0), atol=3e-6, rtol=3e-5)
    linear = .1*gradients@design.T
    norms = np.linalg.norm(gradients.astype(float), axis=1)
    cosine = gradients.astype(float)@gradients.astype(float).T/np.maximum(norms[:, None]*norms, 1e-30)
    change = batch['post_mean'].flatten().astype(float)-batch['mean'].flatten()
    rows = {}
    for i, name in enumerate(BRANCHES):
        mask = group == i
        g = gradients[i+1]
        gb = float(g[-1])
        common_weights = hidden.mean(0)*gb
        rows[name] = dict(samples=int(mask.sum()), weight=float(mask.mean()),
            advantage_mean=float(advantage.numpy()[mask].mean()),
            positive_advantage_fraction=float((advantage.numpy()[mask] > 0).mean()),
            reward_mean=float(batch['reward'].flatten()[mask].mean()),
            gradient_norm=float(norms[i+1]), output_bias_ascent=gb,
            centered_hidden_gradient_norm=float(np.linalg.norm(g[:-1]-common_weights)),
            shared_hidden_gradient_norm=float(np.linalg.norm(common_weights)),
            predicted_mean_ascent_by_branch={n: float(linear[i+1, group == j].mean()) for j, n in enumerate(BRANCHES)},
            actual_mean_change=float(change[mask].mean()), actual_change_std=float(change[mask].std()))
    summary = dict(branches=rows, gradient_cosines=cosine.tolist(),
        net_norm_over_sum_branch_norms=float(norms[0]/norms[1:].sum()),
        actual_mean_change=float(change.mean()), actual_change_rms=float(np.sqrt(np.mean(change**2))),
        actual_change_std=float(change.std()), constant_change_energy_fraction=float(change.mean()**2/np.mean(change**2)),
        analytic_gradient_max_error=float(np.abs(gradients-analytic).max()))
    return dict(hidden=hidden, gradients=gradients, linear_response=linear, change=change), summary


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    assert read(BASE/'audit.json')['passed']
    manifest = read(BASE/'completed.json')['outputs']
    inputs = {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('verify_recovery_ppo_gradients.py'),
        Path(__file__).with_name('RECOVERY_PPO_GRADIENT_PLAN.md'), BASE/'completed.json', BASE/'audit.json',
        Path(__file__).with_name('half_noise_ppo.py'), Path(__file__).with_name('train_coordinated_ankle.py'),
        Path(__file__).with_name('recovery_start_training.py')]}
    OUT.mkdir(parents=True, exist_ok=False)
    results = {}
    for seed in (921009, 931009):
        folder = BASE/f'fit-{seed}'
        for name in ('initial.pt', 'collection-1.npz', 'resets.json', 'history.json'):
            path = folder/name
            assert digest(path) == manifest[str(path.relative_to(BASE))]
            inputs[str(path)] = digest(path)
        batch, records, history = arrays(folder/'collection-1.npz'), read(folder/'resets.json'), read(folder/'history.json')
        sigma = float(batch['sigma'])
        initial = AnkleHead().eval()
        initial.load_state_dict(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
        branch, age, assignment = labels(seed, batch, records)
        replay_arrays, replay_report = replay(seed, initial, batch, history, sigma)
        gradients, summary = gradient_analysis(initial, batch, branch, sigma)
        np.savez_compressed(OUT/f'{seed}.npz', branch=branch, age=age, **replay_arrays, **gradients)
        results[str(seed)] = dict(assignment=assignment, replay=replay_report, gradient=summary)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(OUT/'inputs.json', inputs)
    save(OUT/'results.json', results)
    save(OUT/'completed.json', dict(physical_rollouts=0, new_training_transitions=0, saved_samples=1536,
        replay_actor_steps=18, replay_critic_steps=18, wall_seconds=time.perf_counter()-start,
        outputs={p.name: digest(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
