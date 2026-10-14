"""Target the transfer states missed by the first bounded correction batch."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import Learner, PARENT, evaluate
from .run_learned_handoff import GuardedLearner, PRIOR, fit

EARLY=Path('logs/mimickit-training/learned-handoff-20261008-01')
OUT=Path('logs/mimickit-training/learned-handoff-late-20261008-01')


class LateCollector(GuardedLearner):
    def forward(self,x):
        self.collect=self.frame*self.env.dt>=3.5
        return super().forward(x)


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'dataset.npz',
        PRIOR/'goals.npz',EARLY/'correction/final.pt',EARLY/'corrections.npz',EARLY/'report.json',
        FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for path in paths:shutil.copy2(path,OUT/'sources'/path.name)
    save(OUT/'protocol.json',dict(inputs=inputs,reason='First 192 labels covered early handoff but filled before later transfer failures',
        query_start_seconds=3.5,maximum_queries=192,training_updates_per_arm=1500,
        control='Replay original examples plus the first correction batch',
        treatment='Same replay pool, replace 25 percent of each minibatch with new late-state corrections',
        split='No labels from whole-pose holdouts 6 and 7',arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    poses=np.load(PRIOR/'goals.npz')['poses'];bank={i:dict(poses=poses[i]) for i in range(16)}
    weights=torch.load(EARLY/'correction/final.pt',weights_only=True)
    initial=Learner(weights['mean'],weights['scale'],weights['output_scale']);initial.load_state_dict(weights);initial.eval()
    guard=LateCollector(initial,anchor,env,collect=True)
    results=dict(collection=evaluate(env,anchor,bank,guard,'goal',1.,False,OUT/'collection'))
    assert guard.samples
    samples={k:np.stack([r[k] for r in guard.samples]) for k in guard.samples[0]}
    assert (samples['frame']*env.dt>=3.5).all() and not (samples['world']%8>=6).any()
    np.savez_compressed(OUT/'corrections.npz',**samples)
    data=dict(np.load(PRIOR/'dataset.npz'));early=dict(np.load(EARLY/'corrections.npz'))
    pool={k:np.concatenate((data[k],early[k])) for k in ('x','issued','goal')}
    pool['validation']=np.r_[data['validation'],np.zeros(len(early['x']),dtype=bool)]
    results['labels']=dict(count=len(samples['x']),first_seconds=float(samples['frame'].min()*env.dt),
        last_seconds=float(samples['frame'].max()*env.dt),query_seconds=guard.query_seconds,worlds=np.unique(samples['world']).tolist())
    save(OUT/'progress.json',results)
    for corrected,key in ((False,'replay'),(True,'correction')):
        net,training=fit(initial,pool,samples,corrected,OUT/key)
        guarded=GuardedLearner(net,anchor,env)
        result=evaluate(env,anchor,bank,guarded,'goal',1.,False,OUT/f'entry-{key}')
        results[key]=dict(training=training,evaluation=result)
        if corrected:
            results['quiet_final']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/'quiet-final')
            assert results['quiet_final']['whole_interval_stand']==16
            if result['completed']==16:
                results['onset3']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',3.,False,OUT/'onset3-correction')
        save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    results.update(protected_inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts})
    save(OUT/'report.json',results)
    print('LATE_COMPLETE',json.dumps({k:results[k]['evaluation']['completed'] for k in ('replay','correction')}),flush=True)


if __name__=='__main__':main()
