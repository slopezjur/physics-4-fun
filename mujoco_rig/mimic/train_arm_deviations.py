"""Teach clean corrective feedback from physically disturbed successful holds."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_support import PARENT,evaluate
from .run_learned_handoff import PRIOR,GuardedLearner
from .train_arm_hold import episode
from .ablate_arm_hold_queue import QueueBlindLearner,SOURCE

INITIAL=Path('logs/mimickit-training/arm-hold-queue-ablation-20261008-01/model/final.pt')
OUT=Path('logs/mimickit-training/arm-hold-deviations-20261008-01')


def collect(env,anchor,net,poses):
    from .interval_target_support import IntervalTargetSupport
    original=IntervalTargetSupport.target
    noise=np.zeros((16,33));rng=[np.random.default_rng(471028+i) for i in range(16)]
    labels={}

    def disturbed(controller,world,goal,source,radius=1.):
        clean,info=original(controller,world,goal,source,radius)
        frame=int(env.task.steps[world]);now=frame*env.dt
        if 2.5<=now<4.5:
            noise[world]=.85*noise[world]+.15*rng[world].uniform(-.012,.012,33)
        else:
            noise[world]=0
        control=env.engine.policy_control
        applied=np.clip(clean+noise[world],np.maximum(control.lower.numpy(),source-radius),np.minimum(control.upper.numpy(),source+radius))
        labels[(frame,world)]=clean.astype(np.float32).copy()
        return applied,info

    with patch.object(IntervalTargetSupport,'target',disturbed):
        report=episode(env,anchor,net,poses,OUT/'teacher-disturbed',teacher=True)
    data=dict(np.load(OUT/'teacher-disturbed/samples.npz'))
    data['applied_raw_target']=data['raw_target'].copy()
    for row,(frame,world) in enumerate(zip(data['frame'],data['world'])):
        if (int(frame),int(world)) in labels:
            data['raw_target'][row]=labels[(int(frame),int(world))]
    data['target_noise']=data['applied_raw_target']-data['raw_target']
    assert abs(data['target_noise']).max()<=.012001
    mask=(data['frame']*env.dt>=2.5)&(data['frame']*env.dt<5.5)
    correction={k:v[mask] for k,v in data.items()}
    assert (~correction['validation']).any() and correction['validation'].any()
    np.savez_compressed(OUT/'corrective-samples.npz',**correction)
    save(OUT/'data-contract.json',dict(teacher_complete=report['complete'],examples=len(correction['x']),
        train_examples=int((~correction['validation']).sum()),heldout_examples=int(correction['validation'].sum()),
        maximum_raw_target_disturbance_rad=float(abs(data['target_noise']).max()),
        label='Clean raw corrective target at the actual disturbed state, before handoff',
        applied='Noisy raw target stored separately; physical authority/fade/slew unchanged',
        accepted_poses=np.unique(correction['world']).tolist(),excluded_poses=[i for i in range(8) if i not in correction['world']]))
    return correction,report


def fit(initial,old,new,corrected,folder,seed):
    torch.manual_seed(seed);rng=torch.Generator(device='cuda').manual_seed(seed)
    net=deepcopy(initial).cuda().train();optimizer=torch.optim.Adam(net.parameters(),lr=.0001)
    def tensors(d,train):
        mask=~d['validation'] if train else d['validation']
        return torch.tensor(d['x'][mask],device='cuda'),torch.tensor((d['raw_target']-d['goal'])[mask],device='cuda')
    x,y=tensors(old,True);cx,cy=tensors(new,True)
    folder.mkdir();torch.save(net.cpu().state_dict(),folder/'initial.pt');net.cuda()
    start=time.perf_counter()
    for _ in range(4000):
        ids=torch.randint(len(x),(512,),device='cuda',generator=rng)
        ci=torch.randint(len(cx),(256,),device='cuda',generator=rng)
        bx,by=x[ids],y[ids]
        if corrected:bx,by=torch.cat((bx[:256],cx[ci])),torch.cat((by[:256],cy[ci]))
        loss=((net(bx)-by)/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    net.eval();result=dict(seed=seed,updates=4000,corrected=corrected,wall_seconds=time.perf_counter()-start)
    with torch.no_grad():
        for key,d in (('old',old),('new',new)):
            vx,vy=tensors(d,False);result[key+'_heldout_rmse_rad']=float((net(vx)-vy).square().mean().sqrt())
    torch.save(net.cpu().state_dict(),folder/'final.pt')
    save(folder/'architecture.json',dict(class_name='QueueBlindLearner',masked_columns=[95,229],fill='frozen mean'))
    save(folder/'report.json',result);print('FIT',folder.name,result,flush=True);return net,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('ARM_HOLD_DEVIATIONS_PLAN.md')]
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),INITIAL,PRIOR/'goals.npz',SOURCE/'training-samples.npz',
        FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,noise_seed=471028,train_seed=481028,replication_seed=491028,
        maximum_noise_rad=.012,noise_recurrence=[.85,.15],noise_window_s=[2.5,4.5],correction_window_s=[2.5,5.5],
        new_data_fraction=.5,updates_per_arm=4000,arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.);poses=np.load(PRIOR/'goals.npz')['poses']
    state=torch.load(INITIAL,weights_only=True);initial=QueueBlindLearner(state['mean'],state['scale'],state['output_scale']);initial.load_state_dict(state);initial.eval()
    new,teacher=collect(env,anchor,initial,poses);old=dict(np.load(SOURCE/'training-samples.npz'))
    results=dict(teacher=teacher);save(OUT/'progress.json',results)
    for seed in (481028,491028):
        if seed==491028 and results['correction-481028']['evaluation']['complete']<=results['replay-481028']['evaluation']['complete']:break
        for corrected,label in ((False,'replay'),(True,'correction')):
            key=f'{label}-{seed}';net,tr=fit(initial,old,new,corrected,OUT/key,seed)
            result=episode(env,anchor,net,poses,OUT/f'{key}-nominal')
            results[key]=dict(training=tr,evaluation=result)
            if corrected:
                bank={i:dict(poses=poses[i]) for i in range(16)}
                results[f'quiet-{seed}']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/f'quiet-{seed}')
                assert results[f'quiet-{seed}']['whole_interval_stand']==16
                if result['complete']==8:results[f'varied-{seed}']=episode(env,anchor,net,poses,OUT/f'varied-{seed}',variant=True)
            save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
