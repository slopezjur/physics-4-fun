"""Second matched training seed for the observed partial queue-ablation result."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_support import Learner,PARENT
from .run_learned_handoff import PRIOR
from .train_arm_hold import episode,train
from .ablate_arm_hold_queue import QueueBlindLearner,SOURCE

OUT=Path('logs/mimickit-training/arm-hold-queue-replication-20261008-01')


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'goal-421028/final.pt',
        PRIOR/'goals.npz',SOURCE/'training-samples.npz',FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,seed=461028,updates_per_arm=4000,
        reason='Replicate the partial supported-hold gain without lowering pose-tracking gates',
        comparison='Same initialization, exact data and training seed; queue inputs retained versus replaced with frozen means',
        arena_promotion=False))
    data=dict(np.load(SOURCE/'training-samples.npz'));state=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.);poses=np.load(PRIOR/'goals.npz')['poses'];results={}
    for name,cls in (('full',Learner),('queue_blind',QueueBlindLearner)):
        initial=cls(state['mean'],state['scale'],state['output_scale']);initial.load_state_dict(state);initial.eval()
        net,tr=train(initial,data,OUT/name,seed=461028)
        save(OUT/name/'architecture.json',dict(class_name=cls.__name__,mask=[95,229] if name=='queue_blind' else None))
        results[name]=dict(training=tr,evaluation=episode(env,anchor,net,poses,OUT/f'{name}-nominal'))
        save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
