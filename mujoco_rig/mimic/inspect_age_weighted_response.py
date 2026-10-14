"""Case-level outcomes and early ankle response from already saved evaluations."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
PILOT = Path('logs/mimickit-training/learned-ankle-20261009-01')


def main():
    inputs = {str(Path(__file__)): digest(Path(__file__))}
    manifests = {root: json.loads((root/'completed.json').read_text())['outputs'] for root in (OUT, PILOT)}

    def load(path):
        sha = digest(path)
        root = OUT if path.is_relative_to(OUT) else PILOT
        assert manifests[root][str(path.relative_to(root))] == sha
        inputs[str(path)] = sha
        if path.suffix == '.json':
            return json.loads(path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    assert json.loads((OUT/'audit.json').read_text())['passed']
    inputs[str(OUT/'audit.json')] = digest(OUT/'audit.json')
    results = {}
    for seed in (921009, 931009):
        modes = {}
        for mode in ('continuous', 'matched'):
            gains, losses = [], []
            for timing in (0, 1):
                for group in range(4):
                    folder = f'{seed}-t{timing}-g{group}'
                    if mode == 'matched':
                        folder = 'matched-'+folder
                    current = load(OUT/folder/'report.json')['cases']
                    old = load(PILOT/f'baseline-t{timing}-g{group}'/'report.json')['cases']
                    for c, baseline in zip(current, old, strict=True):
                        row = dict(timing=timing, group=group, case=c['case'], heldout=c['heldout'])
                        if c['settled'] and not baseline['settled']:
                            gains.append(row)
                        if baseline['settled'] and not c['settled']:
                            losses.append(row)
            modes[mode] = dict(gains=gains, losses=losses)
        amplitude, response = [], []
        for timing in (0, 1):
            z = load(OUT/f'matched-{seed}-t{timing}-g0'/'trace.npz')
            quiet = load(OUT/f'matched-{seed}-t{timing}-quiet'/'trace.npz')
            start = (60, 102)[timing]+1
            frames = slice(start, start+30)
            assert z['alive'][frames].all()
            addition = z['addition'][frames, :, 21]
            amplitude.append(addition)
            response.append(addition-quiet['addition'][frames, :, 21])
        amplitude, response = np.concatenate(amplitude), np.concatenate(response)
        collections = {}
        for update in (1, 32):
            z = load(OUT/f'fit-{seed}'/f'collection-{update}.npz')
            sigma = float(z['sigma'])
            action, mean = z['action'].reshape(-1, 1).astype(float), z['mean'].reshape(-1, 1).astype(float)
            memory = .9*z['x'][..., -1:].reshape(-1, 1)
            post = z['post_mean'].astype(float)
            av = z['normalized_advantage'].astype(float)
            ratio = np.exp((-.5*((action-post)/sigma)**2+.5*((action-mean)/sigma)**2).sum(-1))
            weights = z['actor_weight'].flatten().astype(float)
            clipped = (weights*np.minimum(ratio*av, np.clip(ratio, .8, 1.2)*av)).mean()
            collections[str(update)] = dict(samples=len(av),
                pre_learned_mean_rms=float(np.sqrt(np.mean((mean-memory)**2))),
                post_learned_mean_rms=float(np.sqrt(np.mean((post-memory)**2))),
                conditional_noise_sigma=sigma,
                post_update_kl=float(np.mean((post-mean)**2/(2*sigma**2))),
                weighted_clipped_surrogate_gain=float(clipped-(weights*av).mean()),
                critic_target_mse_before=float(np.mean((z['value'].flatten()-z['returns'])**2)),
                critic_target_mse_after=float(np.mean((z['post_value']-z['returns'])**2)),
                target_std=float(z['returns'].std()), advantage_std=float(z['advantage'].std()))
        results[str(seed)] = dict(outcomes=modes, collections=collections, first_500ms=dict(
            addition_rms_rad=float(np.sqrt(np.mean(amplitude**2))),
            addition_max_abs_rad=float(np.abs(amplitude).max()),
            difference_from_quiet_rms_rad=float(np.sqrt(np.mean(response**2))),
            difference_from_quiet_max_abs_rad=float(np.abs(response).max())))
    paired_path = OUT/'paired-outcomes.json'
    inputs[str(paired_path)] = digest(paired_path)
    pairs = json.loads(paired_path.read_text())
    for seed in (921009, 931009):
        results[str(seed)]['paired'] = {}
        for mode in ('continuous', 'matched'):
            results[str(seed)]['paired'][mode] = {}
            for comparator in ('unweighted-half-noise', 'zero'):
                selected = [p for p in pairs if p['seed'] == seed and p['mode'] == mode and p['comparator'] == comparator]
                results[str(seed)]['paired'][mode][comparator] = {
                    outcome: [p for p in selected if p['outcome'] == outcome] for outcome in ('gain', 'loss')}
    for root in manifests:
        inputs[str(root/'completed.json')] = digest(root/'completed.json')
    save(OUT/'response_diagnostics.json', dict(inputs=inputs, results=results,
        scope='Saved evaluation traces and first/last PPO collections; no new training or physical rollouts.'))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
