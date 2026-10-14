"""Quiet retention and independent noisy admission for the matched PPO candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_execution_distribution import ROOT,setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment,CorrectionNetwork
from .run_support_skill_study import digest,save

QUIET=Path('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(26,30),required=True);args=parser.parse_args()
    steps=(24,25) if args.step==26 else (24,25,28,29)
    paths={str(step):ROOT/f'step-{step:02d}/fit/final.pt' for step in steps}
    out=ROOT/f'step-{args.step:02d}';inputs=seal(out,list(paths.values())+[QUIET])
    configs=[]
    if args.step==26:
        configs=[dict(actor=step,mode='mean',quiet=True,seed=612001) for step in paths]
    else:
        configs=[dict(actor=step,mode='sample',quiet=False,seed=seed) for step in ['zero',*paths] for seed in (612001,612002)]
        configs += [dict(actor=str(step),mode='mean',quiet=True,seed=612001) for step in (28,29)]
    save(out/'protocol.json',dict(step=args.step,inputs=inputs,configs=configs,training_updates=0))
    env,anchor,proposal,poses,actors=setup()
    for key,path in paths.items():
        actor=CorrectionNetwork();actor.load_state_dict(torch.load(path,weights_only=True));actors[key]=actor.eval().requires_grad_(False)
    adapters={quiet:CorrectionEnvironment(env,anchor,proposal,poses,quiet=quiet) for quiet in {c['quiet'] for c in configs}}
    baseline=np.load(QUIET);results=[]
    for index,c in enumerate(configs):
        folder=f'{index:02d}-{c["actor"]}-{c["seed"]}'
        result=episode(adapters[c['quiet']],actors[c['actor']],c,out/folder)
        if c['quiet']:
            z=np.load(out/folder/'trace.npz')
            assert result['quiet_passed']==16
            assert all(np.array_equal(z[k],baseline[k]) for k in ('qpos','issued'))
            result['quiet_exact']=True
        results.append(dict(folder=folder,**result));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=args.step,results=results,inputs_unchanged=True,arena_changed=False,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
