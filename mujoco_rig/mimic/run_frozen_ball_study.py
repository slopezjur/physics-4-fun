"""Prospectively configured frozen ball comparisons, with native contact attribution."""
import argparse
from contextlib import nullcontext
import gc
from pathlib import Path
import traceback

import numpy as np
import torch

from .audit_stand_failures import read, save
from .ball_study_support import FOUNDATION, body_observer, checkpoint, protect_used, save_bodies
from .evaluate_stand_ball_boundary import evaluate as native_evaluate
from .evaluate_stand_ball_gpu import evaluate as gpu_evaluate, compare
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import load_actor, seal


def run(args):
    if not __debug__:
        raise RuntimeError('Assertions required')
    settings = read(args.protocol)
    if settings['schema'] != 'frozen_ball_study_v1' or settings['training_updates'] != 0:
        raise ValueError('Unsupported protocol')
    expected = len(settings['seeds'])*len(settings['actors'])*len(settings['conditions'])*len(settings['signs'])*16
    if expected != settings['maximum_cases'] or expected > 640:
        raise ValueError('Unmatched or excessive declared case budget')
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0,1,'cuda:0',None)
    torch.set_num_threads(1)
    parent, inputs, sources = protect_used(args.out, [Path(__file__)], args.protocol, settings)
    save(args.out/'protocol.json', dict(inputs=inputs,sources=sources,settings=settings,
        config=parent['config'],demonstrations=parent['demonstrations'],
        evidence_scope='All used assets and inherited live source files checked; unused historical outputs are referenced through ancestor seals, not reread',
        automatic_promotion=False))
    rig, reference = Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    dt = rig.model.opt.timestep * rig.decimation
    worlds = {}
    for condition in settings['conditions']:
        mass = float(condition['mass_kg'])
        if mass not in worlds:
            worlds[mass] = args.out/f'ball-{mass:g}kg.xml'
            write_world(args.out/'dummy.xml',FOUNDATION/'ball-5kg.xml',worlds[mass],mass)
    results, cases = [],0
    for seed in settings['seeds']:
        for label in settings['actors']:
            actor = load_actor(rig,reference,parent,checkpoint(seed,label))
            for condition in settings['conditions']:
                cfg = dict(seconds=float(condition.get('seconds',12.)),
                    launch_seconds=float(condition.get('launch_seconds',5.)),
                    launch_distance_m=float(condition['distance_m']))
                if not (0 < cfg['launch_distance_m'] <= 2 and 0 < condition['speed_m_s'] <= 8
                        and 0 < cfg['launch_seconds'] < cfg['seconds'] <= 120):
                    raise ValueError('Out-of-scope physical condition')
                for sign in settings['signs']:
                    assessments, bodies = {},None
                    for backend,evaluate in (('native',native_evaluate),('gpu',gpu_evaluate)):
                        folder = args.out/f"{seed}-{label}-{condition['name']}-{sign:+d}-{backend}"
                        journal = {}
                        context = body_observer(journal,int(np.ceil(cfg['seconds']/dt))) if backend=='native' else nullcontext()
                        with context:
                            assessments[backend] = evaluate(rig,reference,actor,worlds[float(condition['mass_kg'])],
                                folder,cfg,float(condition['speed_m_s']),sign)
                        if backend=='native':
                            bodies = save_bodies(folder,journal)
                        cases += 8
                        save(args.out/'case-progress.json',dict(evaluation_cases=cases,maximum_cases=expected,
                            seed=seed,actor=label,condition=condition['name'],sign=sign,backend=backend))
                        gc.collect()
                        torch.cuda.empty_cache()
                    parity = compare(assessments['native'],assessments['gpu'],dt)
                    result = dict(seed=seed,actor=label,condition=condition,sign=sign,
                        assessments=assessments,bodies=bodies,parity=parity)
                    results.append(result)
                    save(args.out/'progress.json',dict(evaluation_cases=cases,results=results))
                    if not parity['passed']:
                        raise RuntimeError('Backend disagreement exceeds existing admitted tolerance')
            del actor
    assert cases == expected
    save(args.out/'report.json',dict(results=results,evaluation_cases=cases,training_updates=0,promoted=False))
    seal(args.out,inputs,sources,verified=True,evaluation_cases=cases,training_updates=0,promoted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=parser.parse_args()
    existed=args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
