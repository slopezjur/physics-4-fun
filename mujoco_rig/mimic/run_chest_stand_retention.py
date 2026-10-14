"""Retained full-interval Stand for a declared pair of sealed frozen checkpoints."""
import argparse
import gc
from pathlib import Path
import traceback
from unittest.mock import patch

import torch

from .audit_stand_failures import digest, read, save
from . import ball_study_support as support
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import load_actor, seal


def run(args):
    s = read(args.protocol)
    assert __debug__ and s['schema'] == 'chest_stand_retention_v1'
    assert s['seeds'] == [301027, 301028] and len(s['actors']) == 1
    assert s['seconds'] in (60, 120) and s['maximum_cases'] == 32 and s['training_updates'] == 0
    for item in s['checkpoints'].values():
        root, model = Path(item['root']), Path(item['checkpoint'])
        proof = read(root / 'final-verification.json')
        assert proof['verified']
        hashes = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        assert str(root / 'final-verification.json').replace('\\', '/') in [p.replace('\\', '/') for p in s['evidence']]
        for p in (model, model.with_suffix('.amp.pt')):
            assert hashes.get(str(p.resolve())) == digest(p)
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    with patch.object(support, 'checkpoint', lambda seed, _: Path(s['checkpoints'][str(seed)]['checkpoint'])):
        parent, inputs, sources = support.protect_used(args.out, [Path(__file__), Path(__file__).with_name('evaluate_stand_sustained.py')], args.protocol, s)
    save(args.out / 'protocol.json', dict(settings=s, inputs=inputs, sources=sources,
         config=parent['config'], demonstrations=parent['demonstrations']))
    rig, reference = Rig.load(args.out / 'dummy.xml', policy_exclude=()), args.out / 'stationary_reference.npz'
    results = {}
    for seed in s['seeds']:
        actor = load_actor(rig, reference, parent, Path(s['checkpoints'][str(seed)]['checkpoint']))
        results[str(seed)] = {}
        for backend in ('native', 'gpu'):
            folder = args.out / f'{seed}-{backend}'
            results[str(seed)][backend] = evaluate_duration(rig, reference, 'target_pd', backend, actor, folder, seconds=s['seconds'])
            verify_duration(folder, s['seconds'])
            save(args.out / 'progress.json', results)
        del actor
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(results=results, evaluation_cases=32, training_updates=0,
        full_interval_stand=all(r['standing_passes'] == 8 for a in results.values() for r in a.values()))
    save(args.out / 'report.json', report)
    seal(args.out, inputs, sources, verified=True, evaluation_cases=32, training_updates=0,
         full_interval_stand=report['full_interval_stand'])
    print(dict(full_interval_stand=report['full_interval_stand'], passes={s:{b:r['standing_passes'] for b,r in a.items()} for s,a in results.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = p.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
