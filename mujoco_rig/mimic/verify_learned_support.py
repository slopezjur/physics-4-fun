"""Reconcile the twenty experiments and check the unchanged anchor in their harness."""
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT, Learner, evaluate


class ZeroAddition(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((len(x),33),dtype=x.dtype,device=x.device)


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    root=Path('logs/mimickit-training/learned-support-20261008-01')
    out=Path('logs/mimickit-training/learned-support-review-20261008-01')
    out.mkdir(exist_ok=False)
    sealed=json.loads((root/'final-verification.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert all(digest(Path(p))==h for p,h in sealed['output_hashes'].items())
    assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
    progress=json.loads((root/'progress.json').read_text())
    assert [r['step'] for r in progress]==list(range(1,21))
    physical=[r for r in progress if r['kind']=='physical_evaluation']
    assert len(physical)==16 and sum(r['result']['updates'] for r in progress if r['kind']=='training')==16000
    for row in physical:
        z=np.load(root/row['experiment']/'trace.npz')
        for i,c in enumerate(row['result']['cases']):
            alive=z['alive'][:,i];end=np.flatnonzero(alive)[-1]
            assert alive[:end+1].all() and not alive[end+1:].any()
            assert not c['completed'] and c['reason']=='physical_failure' and z['done'][end,i]==1
            assert not c['final_stand'] and not c['whole_interval_stand']
            assert np.isfinite(z['qpos'][alive,i]).all()
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    poses=np.load(root/'goals.npz')['poses']
    bank={i:dict(poses=poses[i]) for i in range(16)}
    baseline=evaluate(env,anchor,bank,ZeroAddition(),'source',1.,True,out/'zero-addition-quiet30')
    assert baseline['whole_interval_stand']==16
    state=torch.load(root/'goal-421028/final.pt',weights_only=True)
    net=Learner(state['mean'],state['scale'],state['output_scale']);net.load_state_dict(state);net.eval()
    captured=[]
    handle=net.register_forward_hook(lambda module,args,result:captured.append(args[0].detach().numpy().copy()))
    diagnostic=evaluate(env,anchor,bank,net,'goal',1.,False,out/'goal-handoff')
    handle.remove()
    x=np.stack(captured)
    data=dict(np.load(root/'dataset.npz'))
    onset=61
    comparisons=[]
    for world in (0,6,9,15):
        matching=np.flatnonzero((data['world']==world)&(data['frame']==onset))
        if not len(matching):continue
        teacher=data['x'][matching[0]]
        delta=(x[onset,world]-teacher)/state['scale'].numpy()
        comparisons.append(dict(world=world,activation_frame=onset,standardized_rms_difference=float(np.sqrt(np.mean(delta**2))),
            max_standardized_difference=float(abs(delta).max()),largest_feature=int(abs(delta).argmax()),
            raw_root_position_difference_m=float(np.linalg.norm(x[onset,world,:3]-teacher[:3]))))
    np.savez_compressed(out/'handoff-observations.npz',observation=x)
    report=dict(completed_steps=20,training_updates=16000,frozen_cases=256,successful_entry_cases=0,
        quiet_candidate_passes=0,zero_addition_anchor_quiet_passes=16,extra_diagnostic_cases=32,
        protected_inputs_and_output_hashes_unchanged=True,activation_comparisons=comparisons,
        interpretation='Stable anchor; all four neural adapters destabilize closed-loop control. Offline error alone is not an admission criterion. Full saved-target replay validation failed later; early reconstruction matched closely.',
        next_priority='Preserve exact zero residual for quiet goals, establish a bounded handoff, and obtain corrective labels on learner-visited states before another broad timing block.',
        arena_parent_sha256=digest(PARENT),arena_changed=False,
        source_sha256=digest(Path(__file__)))
    save(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
