"""Matched supervised ablation of direct queued-command feedback to the learner."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_support import Learner,PARENT,evaluate
from .run_learned_handoff import PRIOR,GuardedLearner
from .train_arm_hold import episode,train

SOURCE=Path('logs/mimickit-training/arm-hold-learning-20261008-01')
OUT=Path('logs/mimickit-training/arm-hold-queue-ablation-20261008-01')


class QueueBlindLearner(Learner):
    """Retain physical delay while withholding command-history inputs from this head."""
    def forward(self,x):
        x=x.clone()
        x[:,95:229]=self.mean[95:229]
        return super().forward(x)


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'goal-421028/final.pt',
        PRIOR/'goals.npz',SOURCE/'training-samples.npz',SOURCE/'hold/report.json',SOURCE/'hold-nominal/report.json',
        FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,hypothesis='Direct sensitivity to the learners own delayed commands may destabilize imitation feedback',
        difference='Learner input columns [95,229) replaced with frozen training normalization means during both fitting and evaluation',
        unchanged='Physical delay, frozen Stand anchor observations, remaining learner inputs, initial weights, data, seed, optimizer and 4000 updates',
        checkpoint_loader='QueueBlindLearner from ablate_arm_hold_queue.py is required; ordinary Learner loading is incompatible',
        arena_promotion=False))
    state=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    initial=QueueBlindLearner(state['mean'],state['scale'],state['output_scale']);initial.load_state_dict(state);initial.eval()
    data=dict(np.load(SOURCE/'training-samples.npz'));net,training=train(initial,data,OUT/'model')
    save(OUT/'model/architecture.json',dict(class_name='QueueBlindLearner',mask=[95,229],fill='frozen mean',source_sha256=digest(Path(__file__))))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.);poses=np.load(PRIOR/'goals.npz')['poses']
    result=episode(env,anchor,net,poses,OUT/'nominal')
    bank={i:dict(poses=poses[i]) for i in range(16)}
    quiet=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/'quiet30')
    assert quiet['whole_interval_stand']==16
    results=dict(training=training,evaluation=result,quiet=quiet)
    if result['complete']==8:
        results['varied']=episode(env,anchor,net,poses,OUT/'varied',variant=True)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
