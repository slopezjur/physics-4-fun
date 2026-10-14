"""Summarize compact historical update measurements without new simulations."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-update-trace-20261009-01')
GROUPS = ('quiet', '+X', '-X', 'early', 'late')


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))


def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    manifest = json.loads((OUT/'completed.json').read_text())['outputs']
    inputs = {str(p): digest(p) for p in (Path(__file__), OUT/'audit.json', OUT/'completed.json')}
    results = {}
    for seed in (921009, 931009):
        rows = []
        for update in range(1, 33):
            path = OUT/f'fit-{seed}'/f'diagnostic-{update}.npz'
            assert digest(path) == manifest[str(path.relative_to(OUT))]
            inputs[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                z = {k: archive[k] for k in archive.files}
            force, age = z['force'], z['age']
            masks = [force == 0, force > 0, force < 0, age < 30, age >= 30]
            norm, gram = z['gradient_norm'], z['gradient_gram']
            cosine = gram/np.maximum(norm[:, None]*norm[None, :], 1e-30)
            total_norm = float(np.sqrt(max(0., gram[:3, :3].sum())))
            probe = z['probe_mean_after'].astype(float)
            quiet, push = probe[:, 0], probe[:, 1]
            response = push-quiet
            change = z['post_mean'][..., 0].astype(float)-z['mean'][..., 0]
            groups = {}
            for i, name in enumerate(GROUPS):
                mask = masks[i]
                n = int(mask.sum())
                groups[name] = dict(samples=n, gradient_norm=float(norm[i]),
                    advantage_mean=float(z['advantage'][mask].mean()) if n else None,
                    normalized_advantage_mean=float(z['normalized_advantage'].reshape(64, 12)[mask].mean()) if n else None,
                    reward_gae_mean=float(z['reward_gae'][mask].mean()) if n else None,
                    value_gae_mean=float(z['value_gae'][mask].mean()) if n else None,
                    critic_target_mse_before=float(np.mean((z['value'][mask]-z['returns'][mask])**2)) if n else None,
                    critic_target_mse_after=float(np.mean((z['post_value'][mask]-z['returns'][mask])**2)) if n else None,
                    actual_mean_change=float(change[mask].mean()) if n else None)
            rows.append(dict(update=update, groups=groups,
                branch_gradient_retention=total_norm/max(1e-30, norm[:3].sum()),
                age_gradient_retention=total_norm/max(1e-30, norm[3:].sum()),
                gradient_cosines=cosine.tolist(),
                quiet_probe_mean=float(quiet.mean()), quiet_probe_rms=rms(quiet),
                push_probe_rms=rms(push), paired_response_rms=rms(response),
                response_rms_by_sign={'+X': rms(response[..., :6]), '-X': rms(response[..., 6:])},
                paired_response_mean_by_sign={'+X': float(response[..., :6].mean()), '-X': float(response[..., 6:].mean())},
                probe_response_to_quiet_rms=rms(response)/max(1e-30, rms(quiet)),
                actual_change_rms=rms(change),
                actual_constant_change_energy_fraction=float(change.mean()**2/max(1e-30, np.mean(change**2))),
                learned_drive_rms=rms(z['mean'][..., 0]-.9*z['previous_latent']),
                critic_value_mean=float(z['value'].mean()),
                failures=int(z['failed'].sum())))
        counts = {name: sum(r['groups'][name]['samples'] for r in rows) for name in GROUPS}
        peak = max(rows, key=lambda r: r['paired_response_rms'])
        strongest = {name: sum(r['groups'][name]['gradient_norm'] == max(r['groups'][b]['gradient_norm'] for b in GROUPS[:3]) for r in rows) for name in GROUPS[:3]}
        summary = dict(total_samples=24576, counts=counts, strongest_branch_gradient_updates=strongest,
            late_stronger_gradient_updates=sum(r['groups']['late']['gradient_norm'] > r['groups']['early']['gradient_norm'] for r in rows),
            peak_paired_response_update=peak['update'], peak_paired_response_rms=peak['paired_response_rms'],
            final_paired_response_rms=rows[-1]['paired_response_rms'], final_quiet_probe_rms=rows[-1]['quiet_probe_rms'],
            response_to_quiet_rms_at_end=rows[-1]['probe_response_to_quiet_rms'],
            selected_updates=[{k: r[k] for k in ('update', 'quiet_probe_mean', 'quiet_probe_rms', 'paired_response_rms',
                'paired_response_mean_by_sign', 'branch_gradient_retention', 'age_gradient_retention')} for r in rows if r['update'] in (1, 4, 8, 16, 24, 32)])
        results[str(seed)] = dict(summary=summary, updates=rows)
    save(OUT/'interpretation.json', dict(inputs=inputs, results=results,
        scope='Historical update diagnostics and fixed observation probes; no new closed-loop recovery claims.'))
    print(json.dumps({seed: r['summary'] for seed, r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
