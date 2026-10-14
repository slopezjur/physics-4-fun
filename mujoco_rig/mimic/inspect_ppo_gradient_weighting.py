"""Saved-data sensitivity of initial gradients to branch weighting and centering."""
import json
from pathlib import Path
import numpy as np
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-ppo-gradients-20261009-01')
BASE = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')


def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    original_manifest = json.loads((BASE/'completed.json').read_text())['outputs']
    manifest = json.loads((OUT/'completed.json').read_text())['outputs']
    inputs = {str(p): digest(p) for p in (Path(__file__), OUT/'audit.json', OUT/'completed.json', BASE/'completed.json')}
    results = {}
    for seed in (921009, 931009):
        zp, bp = OUT/f'{seed}.npz', BASE/f'fit-{seed}'/'collection-1.npz'
        assert digest(zp) == manifest[zp.name]
        assert digest(bp) == original_manifest[str(bp.relative_to(BASE))]
        inputs.update({str(zp): digest(zp), str(bp): digest(bp)})
        with np.load(zp) as z, np.load(bp) as b:
            group = z['branch'].flatten()
            design = np.c_[z['hidden'].astype(float), np.ones(768)]
            advantage = b['normalized_advantage'].astype(float)
            score = (b['action'].flatten().astype(float)-b['mean'].flatten())/float(b['sigma'])**2
            original = .1*design.T@(score*advantage)/768
            assert np.allclose(original, z['gradients'][0], atol=3e-6, rtol=3e-5)
            weight = np.bincount(group, minlength=3)/768
            centered = advantage.copy()
            for i in range(3):
                centered[group == i] -= centered[group == i].mean()
            variants = dict(equal_branch_weight=advantage/(3*weight[group]), branch_centered=centered)
            results[str(seed)] = {}
            for name, adjusted in variants.items():
                gradient = .1*design.T@(score*adjusted)/768
                response = .1*design@gradient
                results[str(seed)][name] = dict(cosine_to_original=float(original@gradient/np.linalg.norm(original)/np.linalg.norm(gradient)),
                    norm_relative_to_original=float(np.linalg.norm(gradient)/np.linalg.norm(original)),
                    linear_mean_ascent_by_branch={n: float(response[group == i].mean()) for i, n in enumerate(('quiet', '+X', '-X'))})
    save(OUT/'weighting-sensitivity.json', dict(inputs=inputs, results=results,
        scope='Initial infinitesimal gradient only; no optimizer updates or physical outcome claim.'))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
