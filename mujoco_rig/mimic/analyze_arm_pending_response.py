"""Compare frozen queue responses with previously archived teacher alternatives."""
import json
from pathlib import Path

import numpy as np
import torch

from .train_arm_pending_effect import OUT, SEEDS, RECIPES, load_head
from .probe_arm_private_queue import OUT as QUEUE
from .probe_arm_matched_feedback import OUT as BRANCHES
from .run_support_skill_study import digest, save


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    assert json.loads((OUT/'audit.json').read_text())['passed']
    assert json.loads((QUEUE/'audit.json').read_text())['passed']
    validation = json.loads((OUT/'feature-validation.json').read_text())
    teacher = json.loads((QUEUE/'report.json').read_text())['records']
    protocol = json.loads((BRANCHES/'protocol.json').read_text())
    probes = np.load(OUT/'feature-probes.npz')['x']
    assert len(teacher) == len(probes) == len(validation['probes']) == 120
    for t, p in zip(teacher, validation['probes']):
        assert all(t[k] == p[k] for k in ('folder', 'world', 'frame', 'joint', 'sign'))
    records, summaries = [], []
    for seed in SEEDS:
        for recipe in RECIPES:
            net = load_head(OUT/f'{recipe}-{seed}')
            x = torch.tensor(probes)
            raw = x[:, 374:420][:, net.qi]+net(x)
            raw = raw.numpy()
            group = []
            for start in range(0, 120, 5):
                if f'-{seed}-' not in teacher[start]['folder']:
                    continue
                for i in range(start+1, start+5):
                    t = teacher[i]
                    motor = protocol['motors'][t['joint']]
                    target = np.asarray(t['teacher_raw'])-np.asarray(teacher[start]['teacher_raw'])
                    learned = raw[i]-raw[start]
                    delta = t['actual_queue_delta']
                    assert abs(delta) > 0
                    tn, ln = np.linalg.norm(target), np.linalg.norm(learned)
                    cosine = float(np.dot(target, learned)/(tn*ln)) if tn*ln > 0 else None
                    row = dict(seed=seed, recipe=recipe, folder=t['folder'], world=t['world'],
                        joint=t['joint'], sign=t['sign'], teacher_response_norm=float(tn),
                        learned_response_norm=float(ln), response_error_norm=float(np.linalg.norm(learned-target)),
                        cosine=cosine, teacher_own_derivative=float(target[motor]/delta),
                        learned_own_derivative=float(learned[motor]/delta))
                    group.append(row);records.append(row)
            assert len(group) == 48
            cosines = [r['cosine'] for r in group if r['cosine'] is not None]
            summaries.append(dict(seed=seed, recipe=recipe, alternatives=48,
                nonzero_learned=sum(r['learned_response_norm'] > 0 for r in group),
                median_cosine=float(np.median(cosines)) if cosines else None,
                opposite_vector_responses=sum(r['cosine'] is not None and r['cosine'] < 0 for r in group),
                median_teacher_own_derivative=float(np.median([r['teacher_own_derivative'] for r in group])),
                median_learned_own_derivative=float(np.median([r['learned_own_derivative'] for r in group])),
                own_derivative_sign_disagreements=sum(r['teacher_own_derivative']*r['learned_own_derivative'] < 0 for r in group),
                median_response_error_norm=float(np.median([r['response_error_norm'] for r in group]))))
    inputs = [OUT/'audit.json', OUT/'feature-validation.json', OUT/'feature-probes.npz',
              QUEUE/'audit.json', QUEUE/'report.json', BRANCHES/'protocol.json', Path(__file__)]
    inputs += [OUT/f'{recipe}-{seed}'/'final.pt' for seed in SEEDS for recipe in RECIPES]
    save(OUT/'queue-response-diagnostic.json', dict(passed=True, inputs={str(p): digest(p) for p in inputs},
        summaries=summaries, records=records, teacher_queries=0, training_updates=0, physical_cases=0,
        limitation='Local synthetic queue alternatives on archived masked-policy states; not a physical gain or stability certificate.'))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
