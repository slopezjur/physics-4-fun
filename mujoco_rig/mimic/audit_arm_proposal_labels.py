"""Teacher-label semantics and inherited model error on exact teacher states."""
import numpy as np
import torch

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal
from .run_support_skill_study import digest,save


@torch.no_grad()
def main():
    path=TEACHER/'training-samples.npz';out=ROOT/'step-01';inputs=seal(out,[path,PLAN])
    env,anchor,proposal,poses,_=setup();c=env.engine.policy_control
    data=dict(np.load(path));x=torch.tensor(data['x']);world=data['world'];goals=x[:,374:420][:,c.q_indices]
    assert np.array_equal(goals.numpy(),data['goal'])
    y=data['raw_target']-data['goal'];pred=proposal(x).numpy()
    raw=goals+torch.tensor(pred)
    source=(torch.tensor(poses[world,0][:,c.q_indices],dtype=torch.float32)+.25*anchor(x[:,:374]).clamp(-1,1)).clamp(c.lower,c.upper)
    source_error=float(abs(source.numpy()-data['source']).max());assert source_error<2e-6
    displacement=(goals-torch.tensor(poses[world,0][:,c.q_indices],dtype=torch.float32)).abs().amax(-1)
    desired=(torch.tensor(data['raw_target'])-source).clamp(-1,1)*x[:,-1,None]*(displacement/.05).clamp(0,1)[:,None]
    previous=torch.tensor(data['previous']);addition=previous+(desired-previous).clamp(-.02,.02)
    addition[torch.tensor(data['quiet'])]=0
    issued=(source+addition).clamp(c.lower,c.upper)
    issue_error=float(abs(issued.numpy()-data['issued']).max());assert issue_error<2e-6
    norm=((x-proposal.mean)/proposal.scale).numpy();norm[:,95:229]=0
    names=[env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators,0]]
    left=names.index('UpperArm_L_ry');rows=[]
    for heldout in (False,True):
        for quiet in (False,True):
            use=(data['validation']==heldout)&(data['quiet']==quiet)
            rows.append(dict(heldout=heldout,quiet=quiet,samples=int(use.sum()),
                residual_rmse=float(np.sqrt(np.mean((pred[use]-y[use])**2))),
                left_residual_rmse=float(np.sqrt(np.mean((pred[use,left]-y[use,left])**2))),
                left_label_mean=float(y[use,left].mean()),left_prediction_mean=float(pred[use,left].mean()),
                normalized_clipped_fraction=float((abs(norm[use])>=20).mean())))
    result=dict(step=1,inputs=inputs,source_max_error=source_error,issued_max_error=issue_error,
        goal_columns_match=True,target='raw_target minus goal, before authority/fade/slew',results=rows,
        total_samples=len(x),quiet_fraction=float(data['quiet'].mean()),
        scale_min=float(proposal.scale.min()),scale_max=float(proposal.scale.max()),
        output_scale=proposal.output_scale.tolist(),arena_changed=False)
    save(out/'report.json',result);print(rows,flush=True)


if __name__=='__main__':main()
