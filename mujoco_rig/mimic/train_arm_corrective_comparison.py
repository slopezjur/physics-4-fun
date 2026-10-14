"""Fresh matched replay versus admitted learner-state teacher corrections."""
import json
from pathlib import Path

import numpy as np
import torch

from .collect_arm_corrective_takeovers import OUT as TAKEOVER, ARENA
from .probe_arm_proposal_foundations import TEACHER, ROOT as PRIOR_ROOT
from .probe_arm_execution_distribution import setup, seal, episode
from .train_arm_relative_proposal import fit, load_head
from .train_arm_correction_ppo import CorrectionEnvironment
from .challenge_arm_proposals import HEADS, adapter_for
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-corrective-fit-20261008-01')
PLAN = Path(__file__).with_name('ARM_CORRECTIVE_FIT_PLAN.md')


def equal_weights(left, right):
    a, b = (torch.load(p, weights_only=True) for p in (left, right))
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main():
    audit = json.loads((TAKEOVER/'audit.json').read_text())
    assert audit['passed'] and audit['admitted_samples'] > 0
    baseline_report = json.loads((PRIOR_ROOT/'step-29/report.json').read_text())
    baseline_folder = baseline_report['results'][0]['folder']
    inputs = seal(OUT, [PLAN, ARENA, TAKEOVER/'audit.json', TAKEOVER/'training-samples.npz',
                       TEACHER/'training-samples.npz', PRIOR_ROOT/'step-29/report.json',
                       PRIOR_ROOT/'step-29'/baseline_folder/'trace.npz']
                       + [HEADS[f'all-{seed}']/'final.pt' for seed in (711028, 721028)])
    original, correction = (dict(np.load(p/'training-samples.npz')) for p in (TEACHER, TAKEOVER))
    assert not correction['validation'].any() and (correction['world'] < 6).all()
    augmented = {k: np.concatenate((original[k], correction[k])) for k in original}
    assert np.array_equal(augmented['x'][augmented['validation']], original['x'][original['validation']])
    save(OUT/'protocol.json', dict(inputs=inputs, seeds=[711028, 721028], updates=4000, batch_size=512,
        original_training_rows=int((~original['validation']).sum()), correction_rows=len(correction['world']),
        original_quiet_training_rows=int((original['quiet'] & ~original['validation']).sum()),
        heldout_rows=int(original['validation'].sum()), collection_seeds_coupled=True))
    env, anchor, base, poses, actors = setup()
    training, results = [], []
    for seed in (711028, 721028):
        pair = {}
        for recipe, data in (('replay', original), ('augmented', augmented)):
            folder = OUT/f'{recipe}-{seed}'
            net, tr = fit(base, env.engine.policy_control, data, 'inherited', seed, folder)
            training.append(dict(recipe=recipe, **tr))
            pair[recipe] = folder
            if recipe == 'replay':
                assert equal_weights(folder/'final.pt', HEADS[f'all-{seed}']/'final.pt')
            else:
                assert equal_weights(folder/'initial.pt', pair['replay']/'initial.pt')
            for variant in (False, True):
                name = folder.name + ('-varied' if variant else '-nominal')
                adapter = CorrectionEnvironment(env, anchor, net, poses, variant=variant)
                r = episode(adapter, actors['zero'], dict(mode='mean', recipe=recipe, seed=seed,
                    variant=variant, teacher=False), OUT/name)
                results.append(dict(folder=name, **r))
                save(OUT/'progress.json', dict(training=training, results=results))
    from .run_single_support_feedback import make_environment
    env = make_environment(env.rig, anchor, 80.)
    quiet_base = np.load(PRIOR_ROOT/'step-29'/baseline_folder/'trace.npz')
    for seed in (711028, 721028):
        name = f'augmented-{seed}-quiet60'
        net = load_head(OUT/f'augmented-{seed}')
        adapter = adapter_for(env, anchor, net, poses, quiet_seconds=60.)
        r = episode(adapter, actors['zero'], dict(mode='mean', recipe='augmented', seed=seed,
                    quiet=True, teacher=False), OUT/name)
        z = np.load(OUT/name/'trace.npz')
        assert r['quiet_passed'] == 16 and all(np.array_equal(z[k], quiet_base[k]) for k in ('qpos', 'issued', 'alive'))
        results.append(dict(folder=name, quiet_exact=True, **r))
        save(OUT/'progress.json', dict(training=training, results=results))
    scores = {}
    for recipe in ('replay', 'augmented'):
        group = [r for r in results if r['config']['recipe'] == recipe and not r['config'].get('quiet')]
        scores[recipe] = [min(r['complete'] for r in group), sum(r['complete'] for r in group)]
    selected = 'augmented' if scores['augmented'] > scores['replay'] else 'replay'
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(training=training, results=results, scores=scores, selected=selected,
        inputs_unchanged=True, arena_changed=False,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', scores, 'selected', selected, flush=True)


if __name__ == '__main__':
    main()
