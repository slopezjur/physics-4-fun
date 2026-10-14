"""Full-episode frozen-policy data and on-policy return calibration diagnostic."""
from pathlib import Path

import numpy as np
import torch

from .probe_arm_execution_distribution import ROOT,setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment,CorrectionNetwork,OUT as PPO
from .run_support_skill_study import digest,save


def returns(z):
    reward=z['reward']*z['alive'];result=np.zeros_like(reward);running=np.zeros(reward.shape[1],dtype=np.float32)
    for t in reversed(range(len(reward))):
        running=reward[t]+.995*running*(~z['done'][t])*z['alive'][t]
        result[t]=running
    return result


@torch.no_grad()
def main():
    out=ROOT/'step-21';inputs=seal(out,[Path(__file__).with_name('ARM_RECOVERY_30_ADAPTIVE_PLAN.md')])
    spec=[dict(actor=actor,mode='sample',seed=seed) for actor in ('zero','521028','531028')
          for seed in (range(611001,611007) if actor=='zero' else (611001,611002))]
    save(out/'protocol.json',dict(step=21,configs=spec,inputs=inputs,training_updates=0))
    env,anchor,proposal,poses,actors=setup();adapter=CorrectionEnvironment(env,anchor,proposal,poses)
    results=[];calibration=[]
    for index,c in enumerate(spec):
        folder=f'{index:02d}-{c["actor"]}-{c["seed"]}'
        result=episode(adapter,actors[c['actor']],c,out/folder,keep_features=True)
        critic=CorrectionNetwork(1)
        if c['actor']!='zero':critic.load_state_dict(torch.load(PPO/c['actor']/'critic-final.pt',weights_only=True))
        z=dict(np.load(out/folder/'trace.npz'));target=returns(z);live=z['alive']
        value=critic(torch.tensor(z['x'])).numpy()
        for heldout in (False,True):
            pose_mask=np.arange(8)[None,:]>=6 if heldout else np.arange(8)[None,:]<6
            mask=live&pose_mask;error=value[mask]-target[mask]
            calibration.append(dict(actor=c['actor'],seed=c['seed'],heldout=heldout,count=int(mask.sum()),
                mse=float(np.mean(error**2)),bias=float(error.mean()),mean_value=float(value[mask].mean()),
                mean_realized_return=float(target[mask].mean())))
        results.append(dict(folder=folder,**result));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(results=results,calibration=calibration,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
