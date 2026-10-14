"""Common-data regression diagnostics without additional training or physics."""
from pathlib import Path

import numpy as np
import torch

from .train_arm_corrective_comparison import OUT, TAKEOVER, TEACHER
from .train_arm_relative_proposal import load_head
from .run_support_skill_study import digest, save


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    original = dict(np.load(TEACHER/'training-samples.npz'))
    correction = dict(np.load(TAKEOVER/'training-samples.npz'))
    parts = {}
    for heldout in (False, True):
        for quiet in (False, True):
            mask = (original['validation'] == heldout) & (original['quiet'] == quiet)
            name = ('heldout' if heldout else 'train') + ('-quiet' if quiet else '-active')
            parts[name] = {k: v[mask] for k, v in original.items()}
    parts['admitted-corrections'] = correction
    rows, paths = [], [TEACHER/'training-samples.npz', TAKEOVER/'training-samples.npz', Path(__file__)]
    for seed in (711028, 721028):
        for recipe in ('replay', 'augmented'):
            folder = OUT/f'{recipe}-{seed}'
            paths += [folder/'final.pt', folder/'architecture.json']
            net = load_head(folder)
            for name, data in parts.items():
                target = torch.tensor(data['raw_target']-data['goal'])
                error = net(torch.tensor(data['x']))-target
                rows.append(dict(seed=seed, recipe=recipe, subset=name, rows=len(target),
                                 rmse_rad=float(error.square().mean().sqrt()),
                                 normalized_mse=float((error/net.output_scale).square().mean())))
    save(OUT/'common-regression.json', dict(rows=rows, inputs={str(p): digest(p) for p in paths},
        original_quiet_sampling_fraction=float((original['quiet'] & ~original['validation']).sum()/(~original['validation']).sum()),
        augmented_quiet_sampling_fraction=float((original['quiet'] & ~original['validation']).sum()/((~original['validation']).sum()+len(correction['world']))),
        interpretation='Observed regression errors do not establish the cause of closed-loop degradation.'))
    print(rows, flush=True)


if __name__ == '__main__':
    main()
