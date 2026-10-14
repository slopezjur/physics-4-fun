"""Matched PPO comparison changing only the initial critic weights."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from .probe_arm_execution_distribution import ROOT,setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment,CorrectionNetwork,learn
from .run_support_skill_study import digest,save

SPECS={24:(611028,True),25:(621028,True),28:(611028,False),29:(621028,False)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
    seed,warm=SPECS[args.step];out=ROOT/f'step-{args.step:02d}'
    admission=ROOT/'step-23/report.json'
    assert json.loads(admission.read_text())['admitted']
    initial_critic=ROOT/f'step-22/{seed}/initial.pt';warm_critic=ROOT/f'step-22/{seed}/final.pt'
    inputs=seal(out,[admission,initial_critic,warm_critic,Path(__file__).with_name('ARM_RECOVERY_30_ADAPTIVE_PLAN.md')])
    save(out/'protocol.json',dict(step=args.step,seed=seed,warm=warm,inputs=inputs,transitions=24576,
        change='Only critic initial weights; same actor, sampling seed and PPO hyperparameters',
        extra_resource='Shared Monte Carlo warm-up data and 188 value updates per warm critic'))
    env,anchor,proposal,poses,actors=setup();adapter=CorrectionEnvironment(env,anchor,proposal,poses)
    def critic_factory(outputs=33):
        assert outputs==1
        # Preserve exactly the original constructor's random-number consumption.
        critic=CorrectionNetwork(outputs)
        expected=torch.load(initial_critic,weights_only=True)
        assert all(torch.equal(v,expected[k]) for k,v in critic.state_dict().items())
        if warm:critic.load_state_dict(torch.load(warm_critic,weights_only=True))
        torch.save(critic.state_dict(),out/'critic-initial.pt')
        return critic
    actors['zero'].requires_grad_(True)
    with patch('mujoco_rig.mimic.train_arm_correction_ppo.CorrectionNetwork',critic_factory):
        actor,training=learn(adapter,actors['zero'],out/'fit',seed,3e-6)
    results=[]
    for variant in (False,True):
        current=adapter if not variant else CorrectionEnvironment(env,anchor,proposal,poses,variant=True)
        label='varied' if variant else 'nominal'
        config=dict(actor=f'{seed}-'+('warm' if warm else 'fresh'),mode='mean',seed=seed,variant=variant)
        result=episode(current,actor,config,out/label)
        results.append(dict(folder=label,**result))
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=args.step,training=training,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
