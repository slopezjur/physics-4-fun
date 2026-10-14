"""Case-level outcomes and early ankle response from already saved evaluations."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-start-learning-20261009-01')
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
        results[str(seed)] = dict(outcomes=modes, first_500ms=dict(
            addition_rms_rad=float(np.sqrt(np.mean(amplitude**2))),
            addition_max_abs_rad=float(np.abs(amplitude).max()),
            difference_from_quiet_rms_rad=float(np.sqrt(np.mean(response**2))),
            difference_from_quiet_max_abs_rad=float(np.abs(response).max())))
    for root in manifests:
        inputs[str(root/'completed.json')] = digest(root/'completed.json')
    save(OUT/'response_diagnostics.json', dict(inputs=inputs, results=results,
        scope='Saved deterministic traces only; no new training or physical rollouts.'))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
