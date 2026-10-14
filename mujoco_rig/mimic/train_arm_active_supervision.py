"""Matched fresh-head comparison excluding bypassed quiet supervision."""
import argparse
from pathlib import Path

import numpy as np

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_relative_proposal import fit
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(21,22,23),required=True);args=parser.parse_args()
    out=ROOT/f'step-{args.step:02d}';active_only=args.step!=21;seed=711028 if args.step==22 else 721028
    inputs=seal(out,[PLAN,Path(__file__).with_name('ARM_PROPOSAL_30_FINAL_BLOCK.md'),TEACHER/'training-samples.npz'])
    env,anchor,base,poses,actors=setup();data=dict(np.load(TEACHER/'training-samples.npz'))
    if active_only:data={k:v[~data['quiet']] for k,v in data.items()}
    save(out/'protocol.json',dict(step=args.step,inputs=inputs,seed=seed,active_only=active_only,updates=4000,
        training_poses=list(range(6)),heldout_poses=[6,7],sole_matched_change='Exclude bypassed quiet samples' if active_only else 'Fresh seed replication'))
    net,training=fit(base,env.engine.policy_control,data,'inherited',seed,out/'model');results=[]
    for variant in (False,True):
        label='varied' if variant else 'nominal'
        adapter=CorrectionEnvironment(env,anchor,net,poses,variant=variant)
        r=episode(adapter,actors['zero'],dict(actor=f'{seed}-'+('active' if active_only else 'all'),mode='mean',variant=variant),out/label,keep_features=True)
        results.append(dict(folder=label,**r));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=args.step,results=results,training=[training],inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
