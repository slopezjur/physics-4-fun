"""Frozen learned responses on admitted, reachable training-state pairs."""
import json
from pathlib import Path

import numpy as np
import torch

from .collect_arm_reachable_responses import OUT, FOCUS
from .train_arm_pending_effect import OUT as FITS, RECIPES, SEEDS, load_head
from .run_support_skill_study import digest, save


def quantiles(values):
    return dict(zip(('minimum', 'median', 'p95', 'maximum'), [float(v) for v in np.quantile(values, [0, .5, .95, 1])]))


def response_metrics(predicted, target, mask):
    p, t = predicted[mask].astype(np.float64), target[mask].astype(np.float64)
    pn, tn = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    nonzero = (pn > 1e-10) & (tn > 1e-10)
    cosine = (p[nonzero]*t[nonzero]).sum(-1)/(pn[nonzero]*tn[nonzero])
    return dict(pairs=len(p), response_rmse=float(np.sqrt(np.mean((p-t)**2))),
        zero_response_rmse=float(np.sqrt(np.mean(t*t))),
        error_energy_ratio=float(np.sum((p-t)**2)/max(np.sum(t*t), 1e-30)),
        predicted_norm_median=float(np.median(pn)), target_norm_median=float(np.median(tn)),
        nonzero_cosine_pairs=int(nonzero.sum()), median_cosine=float(np.median(cosine)) if len(cosine) else None,
        opposite_responses=int((cosine < 0).sum()))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    audit = json.loads((OUT/'audit.json').read_text())
    assert audit['passed'] and audit['paired_examples'] > 0
    data = np.load(OUT/'response-pairs.npz')
    assert not data['validation'].any() and set(data['world']) <= set(FOCUS)
    assert np.array_equal(data['x0'][:, 374:], data['x1'][:, 374:])
    target = data['raw_target1']-data['raw_target0']
    issued = abs(data['issued_delta'])
    rows, predictions, diagnostics = [], {}, []
    for seed in SEEDS:
        for recipe in RECIPES:
            net = load_head(FITS/f'{recipe}-{seed}')
            x0, x1 = torch.tensor(data['x0']), torch.tensor(data['x1'])
            raw0 = (x0[:, 374:420][:, net.qi]+net(x0)).numpy()
            raw1 = (x1[:, 374:420][:, net.qi]+net(x1)).numpy()
            prediction = raw1-raw0
            predictions[f'{recipe}_{seed}'] = prediction
            own = data['seed'] == seed
            absolute_rmse = float(np.sqrt(np.mean(np.concatenate((raw0[own]-data['raw_target0'][own],
                                                                 raw1[own]-data['raw_target1'][own]))**2)))
            row = dict(recipe=recipe, seed=seed, absolute_rmse=absolute_rmse,
                       **response_metrics(prediction, target, own))
            rows.append(row)
            for variant in (False, True):
                for world in FOCUS:
                    for offset in (0, 1, 3, 6, 12):
                        mask = own & (data['variant'] == variant) & (data['world'] == world) & (data['offset'] == offset)
                        if mask.any():
                            diagnostics.append(dict(recipe=recipe, seed=seed, variant=variant, world=world, offset=offset,
                                                    **response_metrics(prediction, target, mask)))
    teacher_norm = np.linalg.norm(target.astype(np.float64), axis=-1)
    source_delta = data['source1']-data['source0']
    inputs = [OUT/'audit.json', OUT/'response-pairs.npz', Path(__file__)]
    for seed in SEEDS:
        for recipe in RECIPES:
            inputs += [FITS/f'{recipe}-{seed}'/name for name in ('final.pt', 'architecture.json')]
    np.savez_compressed(OUT/'frozen-responses.npz', teacher_delta=target, **predictions)
    report = dict(passed=True, paired_examples=len(target), summaries=rows, strata=diagnostics,
        teacher_response_norm_rad=quantiles(teacher_norm), issued_pulse_rad=quantiles(issued),
        teacher_response_norm_per_issued_rad=quantiles(teacher_norm/issued),
        teacher_signal_above_1e_6=int((teacher_norm > 1e-6).sum()),
        source_response_norm_rad=quantiles(np.linalg.norm(source_delta, axis=-1)),
        teacher_queries=0, training_updates=0, heldout_queries=0,
        interpretation='Finite paired responses across physically different states after an executed command pulse; not isolated input derivatives or a learned recovery result.',
        inputs={str(p): digest(p) for p in inputs}, output_sha256=digest(OUT/'frozen-responses.npz'))
    save(OUT/'response-diagnostic.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('strata', 'inputs')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
