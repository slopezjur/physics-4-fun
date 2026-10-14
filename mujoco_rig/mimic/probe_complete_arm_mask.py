"""Complete the upper-arm ablation with both actuated elbow coordinates."""
from pathlib import Path

import torch

from .probe_arm_proposal_foundations import ROOT,PLAN,ProposalVariant
from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest,save


def main():
    out=ROOT/'step-14';inputs=seal(out,[PLAN,Path(__file__).with_name('ARM_PROPOSAL_30_ADAPTIVE_PLAN.md')])
    env,anchor,base,poses,actors=setup();results=[]
    names=[env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators,0]]
    mask=torch.tensor([name.startswith(('UpperArm_','Forearm_')) for name in names]);assert int(mask.sum())==8
    save(out/'protocol.json',dict(step=14,inputs=inputs,motors=[name for name,use in zip(names,mask) if use],
        correction='Original six-motor upper-arm comparisons retained; this adds both elbow motors',training_updates=0))
    for kind in ('arm_goal','arm_static','drop_arms'):
        proposal=ProposalVariant(base,env,anchor,poses,kind);proposal.arm_mask.copy_(mask)
        for variant in (False,True):
            folder=kind+('-varied' if variant else '-nominal')
            adapter=CorrectionEnvironment(env,anchor,proposal,poses,variant=variant)
            result=episode(adapter,actors['zero'],dict(actor=kind+'-eight-motors',mode='mean',variant=variant),out/folder)
            results.append(dict(folder=folder,**result));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=14,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
