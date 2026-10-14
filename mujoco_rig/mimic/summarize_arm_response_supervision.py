"""Reconcile fitted objectives with frozen physical success and retained references."""
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_response_supervision import OUT, COLLECTION, PREVIOUS, TEACHER, SEEDS, RECIPES
from .train_arm_pending_effect import load_head
from .run_support_skill_study import digest, save


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    audit = json.loads((OUT/'audit.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    assert audit['passed']
    pairs = np.load(COLLECTION/'response-pairs.npz')
    teacher = np.load(TEACHER/'training-samples.npz')
    inputs = [OUT/'audit.json', OUT/'report.json', COLLECTION/'response-pairs.npz',
              TEACHER/'training-samples.npz', PREVIOUS/'audit.json', Path(__file__)]
    rows = []
    for seed in SEEDS:
        for recipe in RECIPES:
            folder = OUT/f'{recipe}-{seed}'
            inputs += [folder/'final.pt', folder/'architecture.json']
            net = load_head(folder)
            p0, p1 = net(torch.tensor(pairs['x0'])), net(torch.tensor(pairs['x1']))
            y0, y1 = (torch.tensor(pairs[k]-pairs['goal']) for k in ('raw_target0', 'raw_target1'))
            pair_mse = .5*(((p0-y0)/net.output_scale).square().mean()+((p1-y1)/net.output_scale).square().mean())
            secant = (p1-p0-torch.tensor(pairs['raw_target1']-pairs['raw_target0']))/net.output_scale/torch.tensor(abs(pairs['issued_delta']).astype(np.float32)[:, None])
            huber = torch.where(secant.abs() < 1, .5*secant.square(), secant.abs()-.5).mean()
            train = ~teacher['validation']
            prediction = net(torch.tensor(teacher['x'][train]))
            original = ((prediction-torch.tensor((teacher['raw_target']-teacher['goal'])[train]))/net.output_scale).square().mean()
            core = [r for r in report['results'] if r['config']['recipe'] == recipe and r['config']['seed'] == seed and r['config']['challenge'] is False]
            challenge = [r for r in report['results'] if r['config']['recipe'] == recipe and r['config']['seed'] == seed and r['config']['challenge'] is True]
            rows.append(dict(recipe=recipe, seed=seed, original_normalized_mse=float(original),
                paired_absolute_normalized_mse=float(pair_mse), full_paired_response_huber=float(huber),
                core_full=sum(r['complete'] for r in core), challenge_full=sum(r['complete'] for r in challenge),
                core_training_full=sum(c['complete'] for r in core for c in r['cases'] if not c['heldout']),
                core_heldout_full=sum(c['complete'] for r in core for c in r['cases'] if c['heldout'])))
    changes = []
    for seed in SEEDS:
        a = next(r for r in rows if r['seed'] == seed and r['recipe'] == 'absolute')
        b = next(r for r in rows if r['seed'] == seed and r['recipe'] == 'response')
        changes.append(dict(seed=seed,
            response_huber_reduction=1-b['full_paired_response_huber']/a['full_paired_response_huber'],
            original_mse_ratio=b['original_normalized_mse']/a['original_normalized_mse'],
            core_pass_change=b['core_full']-a['core_full'], challenge_pass_change=b['challenge_full']-a['challenge_full']))
    result = dict(passed=True, rows=rows, changes=changes, inputs={str(p): digest(p) for p in inputs},
                  training_updates=0, physical_cases=0, teacher_queries=0)
    save(OUT/'objective-summary.json', result)
    print(json.dumps(dict(rows=rows, changes=changes), indent=2), flush=True)


if __name__ == '__main__':
    main()
