"""Bound disturbance collection by demonstrated whole-pose recoverability."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_support import PARENT,evaluate
from .run_learned_handoff import PRIOR,GuardedLearner
from .train_arm_hold import episode
from .ablate_arm_hold_queue import QueueBlindLearner,SOURCE
from .train_arm_deviations import INITIAL,fit

OUT=Path('logs/mimickit-training/arm-hold-deviations-calibration-20261008-01')


def collect(env,anchor,net,poses,amplitude,out):
    from .interval_target_support import IntervalTargetSupport
    original=IntervalTargetSupport.target;noise=np.zeros((16,33))
    rng=[np.random.default_rng(471028+i) for i in range(16)];labels={}
    def disturbed(controller,world,goal,source,radius=1.):
        clean,info=original(controller,world,goal,source,radius)
        frame=int(env.task.steps[world]);now=frame*env.dt
        noise[world]=.85*noise[world]+.15*rng[world].uniform(-amplitude,amplitude,33) if 2.5<=now<4.5 else 0.
        control=env.engine.policy_control
        applied=np.clip(clean+noise[world],np.maximum(control.lower.numpy(),source-radius),np.minimum(control.upper.numpy(),source+radius))
        labels[(frame,world)]=(clean.astype(np.float32).copy(),applied.astype(np.float32).copy())
        return applied,info
    with patch.object(IntervalTargetSupport,'target',disturbed):
        result=episode(env,anchor,net,poses,out,teacher=True)
    # Keep attempted clean/applied targets even when the whole-trajectory data gate fails.
    keys=sorted(labels)
    np.savez_compressed(out/'target-queries.npz',frame=np.array([k[0] for k in keys]),world=np.array([k[1] for k in keys]),
        clean=np.array([labels[k][0] for k in keys]),applied=np.array([labels[k][1] for k in keys]))
    result['innovation_bound_rad']=amplitude
    result['maximum_applied_raw_disturbance_rad']=float(max(np.max(abs(labels[k][1]-labels[k][0])) for k in keys))
    if result['complete']!=8:return None,result
    d=dict(np.load(out/'samples.npz'));d['applied_raw_target']=d['raw_target'].copy()
    for row,(frame,world) in enumerate(zip(d['frame'],d['world'])):
        key=(int(frame),int(world))
        if key in labels:d['raw_target'][row]=labels[key][0]
    d['target_noise']=d['applied_raw_target']-d['raw_target']
    use=(d['frame']*env.dt>=2.5)&(d['frame']*env.dt<5.5)
    correction={k:v[use] for k,v in d.items()}
    assert (~correction['validation']).any() and correction['validation'].any()
    np.savez_compressed(out/'corrective-samples.npz',**correction)
    return correction,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),INITIAL,PRIOR/'goals.npz',SOURCE/'training-samples.npz',
        Path('logs/mimickit-training/arm-hold-deviations-20261008-01/teacher-disturbed/report.json'),
        FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,reason='Original 0.012-radian innovation bound passed only 1/8 and lacked successful heldout data',
        amplitude_candidates=[.003,.00075],seed=471028,collection_gate='All eight complete unchanged hold/return gates',
        maximum_additional_teacher_runs=2,training_gate='Train only after collection gate; otherwise stop',
        training='Same paired seed481028 4000-update replay/correction comparison as prior plan; preserve heldouts',arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.);poses=np.load(PRIOR/'goals.npz')['poses']
    weights=torch.load(INITIAL,weights_only=True);initial=QueueBlindLearner(weights['mean'],weights['scale'],weights['output_scale']);initial.load_state_dict(weights);initial.eval()
    results={};new=None
    for amplitude in (.003,.00075):
        key=f'teacher-{amplitude:g}';new,results[key]=collect(env,anchor,initial,poses,amplitude,OUT/key)
        save(OUT/'progress.json',results)
        if new is not None:
            results['selected_innovation_bound_rad']=amplitude;break
    if new is not None:
        old=dict(np.load(SOURCE/'training-samples.npz'))
        for corrected,key in ((False,'replay'),(True,'correction')):
            net,tr=fit(initial,old,new,corrected,OUT/key,481028)
            results[key]=dict(training=tr,evaluation=episode(env,anchor,net,poses,OUT/f'{key}-nominal'))
            if corrected:
                bank={i:dict(poses=poses[i]) for i in range(16)}
                results['quiet']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/'quiet30')
                assert results['quiet']['whole_interval_stand']==16
            save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,training_ran=new is not None,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
