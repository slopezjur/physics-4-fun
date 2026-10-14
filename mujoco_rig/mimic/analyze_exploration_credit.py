"""Separate same-window reward evidence from frozen-critic GAE contributions."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-exploration-20261009-01')


def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    manifest = json.loads((OUT/'completed.json').read_text())['outputs']
    inputs = {str(Path(__file__)): digest(Path(__file__)), str(OUT/'audit.json'): digest(OUT/'audit.json'),
              str(OUT/'completed.json'): digest(OUT/'completed.json'), str(OUT/'paired-credit.json'): digest(OUT/'paired-credit.json')}
    cache = {}

    def load(name):
        if name not in cache:
            path = OUT/name/'trace.npz'
            inputs[str(path)] = digest(path)
            assert inputs[str(path)] == manifest[str(path.relative_to(OUT))]
            with np.load(path, allow_pickle=False) as z:
                # Only the diagnostic's first collection window is needed here.
                cache[name] = {k: z[k][:64].copy() for k in ('reward', 'alive', 'failed', 'value', 'next_value', 'advantage64')}
        return cache[name]

    pairs = json.loads((OUT/'paired-credit.json').read_text())
    cases, error = [], 0.
    for pair in pairs:
        branch = 'quiet' if pair['quiet'] else 'push'
        base = load(f"{pair['onset']}-{branch}-zero")
        sample = load(f"{pair['onset']}-{branch}-{pair['actor']}-{pair['noise_seed']}")
        j = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13].index(pair['world'])
        coefficient = (.995*.98)**np.arange(64)
        reward_parts, critic_parts, observed = [], [], []
        for z in (sample, base):
            reward = z['reward'][:, j].astype(float)*z['alive'][:, j]
            reward_parts.append(float(np.dot(coefficient, reward)))
            observed.append(float(np.dot(.995**np.arange(64), reward)))
            value_delta = .995*z['next_value'][:, j].astype(float)*(~z['failed'][:, j, None])-z['value'][:, j]
            contribution = (coefficient[:, None]*value_delta*z['alive'][:, j, None]).sum(0)
            critic_parts.append(contribution)
            reconstructed = contribution+reward_parts[-1]
            error = max(error, float(np.abs(reconstructed-z['advantage64'][0, j]).max()))
        dr = reward_parts[0]-reward_parts[1]
        dc = critic_parts[0]-critic_parts[1]
        index = pair['own_critic']
        own = pair['gae_delta'][index]
        # Each stored float32 GAE may differ from the float64 sum by up to 1e-4.
        assert abs(dr+dc[index]-own) < 2e-4
        cases.append(dict(**pair, observed_64_delta=observed[0]-observed[1],
            lambda_reward_delta=dr, critic_contribution_delta=dc.tolist(),
            own_critic_reverses_reward=(dr*own < 0),
            flips_by_critic=[bool(dr*g < 0) for g in pair['gae_delta']]))
    assert error < 1e-4
    summary = {}
    for actor in ('initial', '921009', '931009'):
        branches = {}
        for quiet in (False, True):
            selected = [c for c in cases if c['actor'] == actor and c['quiet'] == quiet]
            branches['quiet' if quiet else 'push'] = dict(cases=len(selected),
                full_reward_improvements=sum(c['reward_delta'] > 0 for c in selected),
                reward_64_improvements=sum(c['observed_64_delta'] > 0 for c in selected),
                lambda_reward_improvements=sum(c['lambda_reward_delta'] > 0 for c in selected),
                own_critic_reward_sign_reversals=sum(c['own_critic_reverses_reward'] for c in selected),
                own_critic_reversals_on_positive_reward=sum(c['lambda_reward_delta'] > 0 and c['own_critic_reverses_reward'] for c in selected),
                rescues=[dict(onset=c['onset'], noise_seed=c['noise_seed'], world=c['world'],
                    lambda_reward_delta=c['lambda_reward_delta'], own_gae_delta=c['gae_delta'][c['own_critic']],
                    full_reward_delta=c['reward_delta'], helpful_run=c['helpful_run']) for c in selected if c['outcome'] == 'rescue'])
        summary[actor] = branches
    save(OUT/'credit-decomposition.json', dict(inputs=inputs, summary=summary, cases=cases,
        max_gae_decomposition_error=error, scope='Frozen same-window reward/critic decomposition; no optimizer or new rollouts.'))
    print(json.dumps(dict(summary=summary, max_error=error)), flush=True)


if __name__ == '__main__':
    main()
