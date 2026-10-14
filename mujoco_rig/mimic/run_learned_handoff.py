"""Bounded residual handoff and matched learner-state corrective supervision."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import Learner, PARENT, evaluate

PRIOR = Path('logs/mimickit-training/learned-support-20261008-01')
OUT = Path('logs/mimickit-training/learned-handoff-20261008-01')


class GuardedLearner(torch.nn.Module):
    """Apply a bounded correction only while the requested pose departs from Stand."""
    def __init__(self, model, anchor, env, collect=False):
        super().__init__()
        self.model, self.anchor, self.env = model, anchor, env
        self.register_buffer('previous', torch.zeros((16,33)))
        self.qi = env.engine.policy_control.q_indices
        self.reference = env.task.reference.qpos[torch.arange(16)%8].clone()
        self.collect = collect
        self.frame = 0
        self.samples = []
        self.maximum_addition = 0.
        self.maximum_slew = 0.
        self.query_seconds = 0.
        if collect:
            from .interval_target_support import IntervalTargetSupport
            self.teacher = IntervalTargetSupport(env.task, objective='interval_com')

    def forward(self, x):
        from .target_control33 import DelayedTargetControl33
        control = self.env.engine.policy_control
        source = DelayedTargetControl33.target_positions(control,self.anchor(x[:,:374]))
        goals = x[:,374:420]
        displacement = (goals[:,self.qi]-self.reference[:,self.qi]).abs().amax(-1)
        quiet = (goals-self.reference).abs().amax(-1)<=1e-6
        amount = (displacement/.05).clamp(0,1)*x[:,420]
        target = goals[:,self.qi]+self.model(x)
        desired = (target-source).clamp(-1,1)*amount[:,None]
        addition = self.previous+(desired-self.previous).clamp(-.02,.02)
        addition[quiet]=0
        self.maximum_addition=max(self.maximum_addition,float(addition.abs().max()))
        self.maximum_slew=max(self.maximum_slew,float((addition-self.previous).abs().max()))
        self.previous.copy_(addition)
        if self.collect and self.frame%6==0 and len(self.samples)<192:
            for world in (0,1,4,5,8,9,12,13):
                if len(self.samples)>=192:break
                if quiet[world] or not self.env.engine.active[world] or x[world,420]<=0:continue
                q=self.env.engine.datas[world].qpos.copy()
                v=self.env.engine.datas[world].qvel.copy()
                pending=control.pending.clone()
                start=time.perf_counter()
                label,info=self.teacher.target(world,goals[world].numpy(),source[world].numpy())
                self.query_seconds+=time.perf_counter()-start
                assert np.array_equal(q,self.env.engine.datas[world].qpos)
                assert np.array_equal(v,self.env.engine.datas[world].qvel)
                assert torch.equal(pending,control.pending)
                assert np.isfinite(label).all()
                self.samples.append(dict(x=x[world].numpy().copy(),issued=label.astype(np.float32),
                    goal=goals[world,self.qi].numpy().copy(),world=world,frame=self.frame,
                    cost_ratio=info['interval_final_cost']/max(info['interval_initial_cost'],1e-12)))
            if self.samples and len(self.samples)%32==0:
                print('corrective_queries',len(self.samples),round(self.query_seconds,1),flush=True)
        self.frame+=1
        # The shared evaluator expects a goal-relative prediction, not an addition.
        return source+addition-goals[:,self.qi]


def fit(initial, data, samples, corrected, folder):
    torch.manual_seed(441028)
    rng=torch.Generator(device='cuda').manual_seed(441028)
    net=deepcopy(initial).cuda().train()
    optimizer=torch.optim.Adam(net.parameters(),lr=.0001)
    train=~data['validation']
    x=torch.tensor(data['x'][train],device='cuda')
    y=torch.tensor((data['issued']-data['goal'])[train],device='cuda')
    cx=torch.tensor(samples['x'],device='cuda')
    cy=torch.tensor(samples['issued']-samples['goal'],device='cuda')
    start=time.perf_counter()
    for _ in range(1500):
        ids=torch.randint(len(x),(512,),device='cuda',generator=rng)
        correction_ids=torch.randint(len(cx),(128,),device='cuda',generator=rng)
        bx,by=x[ids],y[ids]
        if corrected:
            bx=torch.cat((bx[:384],cx[correction_ids]))
            by=torch.cat((by[:384],cy[correction_ids]))
        loss=((net(bx)-by)/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    net.eval()
    with torch.no_grad():
        vx=torch.tensor(data['x'][~train],device='cuda')
        vy=torch.tensor((data['issued']-data['goal'])[~train],device='cuda')
        result=dict(updates=1500,corrected=corrected,wall_seconds=time.perf_counter()-start,
            heldout_rmse_rad=float((net(vx)-vy).square().mean().sqrt()),
            correction_rmse_rad=float((net(cx)-cy).square().mean().sqrt()))
    folder.mkdir();torch.save(net.cpu().state_dict(),folder/'final.pt');save(folder/'report.json',result)
    print('FIT_COMPLETE',folder.name,result,flush=True)
    return net,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('LEARNED_HANDOFF_PLAN.md')]
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'dataset.npz',
        PRIOR/'goal-421028/final.pt',PRIOR/'goals.npz',FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for path in paths:shutil.copy2(path,OUT/'sources'/path.name)
    save(OUT/'protocol.json',dict(inputs=inputs,maximum_teacher_queries=192,seed=441028,
        action_slew_rad_per_frame=.02,quiet_goal_zero=True,training_updates_per_arm=1500))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    poses=np.load(PRIOR/'goals.npz')['poses'];bank={i:dict(poses=poses[i]) for i in range(16)}
    weights=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    initial=Learner(weights['mean'],weights['scale'],weights['output_scale']);initial.load_state_dict(weights);initial.eval()
    results={}
    quiet=GuardedLearner(initial,anchor,env)
    results['quiet_initial']=evaluate(env,anchor,bank,quiet,'goal',1.,True,OUT/'quiet-initial')
    baseline=np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    z=np.load(OUT/'quiet-initial/trace.npz')
    error=float(abs(z['qpos']-baseline['qpos']).max())
    assert error<1e-6 and quiet.maximum_addition==0 and results['quiet_initial']['whole_interval_stand']==16
    results['quiet_maximum_qpos_difference']=error
    guard=GuardedLearner(initial,anchor,env,collect=True)
    results['initial']=evaluate(env,anchor,bank,guard,'goal',1.,False,OUT/'entry-initial')
    samples={k:np.stack([r[k] for r in guard.samples]) for k in guard.samples[0]}
    assert len(samples['x'])<=192 and not (samples['world']%8>=6).any()
    np.savez_compressed(OUT/'corrections.npz',**samples)
    results['corrections']=dict(count=len(guard.samples),query_seconds=guard.query_seconds,
        maximum_addition_rad=guard.maximum_addition,maximum_change_rad=guard.maximum_slew,
        live_state_and_queue_unchanged=True,source='Private physical predictor labels; not successful demonstrations')
    save(OUT/'progress.json',results)
    data=dict(np.load(PRIOR/'dataset.npz'))
    for corrected,key in ((False,'replay'),(True,'correction')):
        net,training=fit(initial,data,samples,corrected,OUT/key)
        guard=GuardedLearner(net,anchor,env)
        results[key]=dict(training=training,evaluation=evaluate(env,anchor,bank,guard,'goal',1.,False,OUT/f'entry-{key}'),
            maximum_addition_rad=guard.maximum_addition,maximum_change_rad=guard.maximum_slew)
        if corrected:
            results['quiet_final']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/'quiet-final')
            assert results['quiet_final']['whole_interval_stand']==16
            if results[key]['evaluation']['completed']==16:
                results['onset3']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',3.,False,OUT/'onset3-correction')
        save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    results.update(protected_inputs_unchanged=True,arena_changed=False,
        next_gate='Stable learned support before displacement and hit recovery',
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts})
    save(OUT/'report.json',results)
    print('COMPLETE',json.dumps({k:results[k]['evaluation']['completed'] for k in ('replay','correction')}),flush=True)


if __name__=='__main__':main()
