"""Audit independent physical outcomes and matched corrective fitting budgets."""
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_corrective_comparison import OUT, TAKEOVER, TEACHER, PRIOR_ROOT, equal_weights
from .challenge_arm_proposals import HEADS
from .analyze_arm_proposal_generalization import physical_results
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR


def main():
    from .rig import Rig
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    for path, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(path)) == h, path
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    poses = np.load(PRIOR/'goals.npz')['poses']
    original = np.load(TEACHER/'training-samples.npz')
    correction = np.load(TAKEOVER/'training-samples.npz')
    assert not correction['validation'].any() and (correction['world'] < 6).all()
    assert set(original['world'][~original['validation']]) == set(range(6))
    assert set(original['world'][original['validation']]) == {6, 7}
    prior = physical_results()
    for seed in (711028, 721028):
        replay, augmented = OUT/f'replay-{seed}', OUT/f'augmented-{seed}'
        assert equal_weights(replay/'initial.pt', augmented/'initial.pt')
        assert equal_weights(replay/'final.pt', HEADS[f'all-{seed}']/'final.pt')
        for recipe, folder in (('replay', replay), ('augmented', augmented)):
            r = json.loads((folder/'report.json').read_text())
            assert r['updates'] == 4000 and r['batch_size'] == 512 and r['lr'] == 1e-4
            assert r['seed'] == seed and r['heldout_examples'] == int(original['validation'].sum())
            assert r['train_examples'] == int((~original['validation']).sum()) + (len(correction['world']) if recipe == 'augmented' else 0)
            initial = torch.load(folder/'initial.pt', weights_only=True)
            final = torch.load(folder/'final.pt', weights_only=True)
            assert (initial['net.4.weight'] == 0).all() and (initial['net.4.bias'] == 0).all()
            assert all(torch.isfinite(v).all() for v in final.values())
            assert any(not torch.equal(v, final[k]) for k, v in initial.items())
    baseline_report = json.loads((PRIOR_ROOT/'step-29/report.json').read_text())
    baseline = np.load(PRIOR_ROOT/'step-29'/baseline_report['results'][0]['folder']/'trace.npz')
    outcomes, quiet = [], 0
    for r in report['results']:
        folder = OUT/r['folder']
        outcomes.append(dict(folder=r['folder'], **gates(rig.model, poses, folder, r)))
        z = np.load(folder/'trace.npz')
        assert not z['correction'].any() and not r['config']['teacher']
        if r['config'].get('quiet'):
            assert all(np.array_equal(z[k], baseline[k]) for k in ('qpos', 'issued', 'alive'))
            assert r['quiet_passed'] == 16
            quiet += 16
        elif r['config']['recipe'] == 'replay':
            previous = next(path for path, row in prior[f'all-{r["config"]["seed"]}']
                            if row['config'].get('variant', False) == r['config']['variant'])
            old = np.load(previous/'trace.npz')
            assert all(np.array_equal(z[k], old[k]) for k in ('qpos', 'issued', 'alive', 'done'))
    scores = {}
    for recipe in ('replay', 'augmented'):
        rows = [r for r in report['results'] if r['config']['recipe'] == recipe and not r['config'].get('quiet')]
        scores[recipe] = [min(r['complete'] for r in rows), sum(r['complete'] for r in rows)]
    assert scores == report['scores']
    assert report['selected'] == ('augmented' if scores['augmented'] > scores['replay'] else 'replay')
    assert quiet == 32
    save(OUT/'audit.json', dict(passed=True, outcomes=outcomes, quiet60_exact=quiet, scores=scores,
        selected=report['selected'], updates=16000, sampled_presentations=8192000,
        independent_fitting_seeds=2, independent_collection_datasets=1, physical_cases=96,
        original_quiet_training_rows=int((original['quiet'] & ~original['validation']).sum()),
        heldout_correction_rows=0, arena_changed=False))
    print('AUDIT PASSED', scores, 'selected', report['selected'], flush=True)


if __name__ == '__main__':
    main()
