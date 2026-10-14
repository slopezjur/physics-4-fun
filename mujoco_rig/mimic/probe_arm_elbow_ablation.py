"""Localize elbow contribution while preserving the full command contract."""
from pathlib import Path

import torch

from .probe_arm_proposal_foundations import ROOT,PLAN,ProposalVariant
from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest,save


class ElbowVariant(ProposalVariant):
    def __init__(self,base,env,anchor,poses,mode):
        super().__init__(base,env,anchor,poses,'static' if mode=='static_arms' else 'inherited')
        self.mode=mode
        names=[env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators,0]]
        selected={'left_elbow':['Forearm_L_rx'],'right_elbow':['Forearm_R_rx'],
                  'elbows':['Forearm_L_rx','Forearm_R_rx'],
                  'static_arms':[n for n in names if n.startswith(('UpperArm_','Forearm_'))]}[mode]
        self.register_buffer('selected',torch.tensor([n in selected for n in names]))
        assert int(self.selected.sum())==len(selected)

    @torch.no_grad()
    def forward(self,x):
        original=self.base(x)
        replacement=super().forward(x) if self.mode=='static_arms' else torch.zeros_like(original)
        return torch.where(self.selected,replacement,original)


def main():
    out=ROOT/'step-20';inputs=seal(out,[PLAN,Path(__file__).with_name('ARM_PROPOSAL_30_FINAL_BLOCK.md')])
    save(out/'protocol.json',dict(step=20,inputs=inputs,training_updates=0,role='Scripted output ablation of frozen learned proposal'))
    env,anchor,base,poses,actors=setup();results=[]
    for mode in ('left_elbow','right_elbow','elbows','static_arms'):
        proposal=ElbowVariant(base,env,anchor,poses,mode)
        for variant in (False,True):
            label=mode+('-varied' if variant else '-nominal')
            adapter=CorrectionEnvironment(env,anchor,proposal,poses,variant=variant)
            r=episode(adapter,actors['zero'],dict(actor=mode,mode='mean',variant=variant),out/label)
            results.append(dict(folder=label,**r));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=20,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
