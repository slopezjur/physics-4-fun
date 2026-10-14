"""Fresh, matched-capacity teacher distillation with explicit task-relative inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest,save


class ProposalHead(nn.Module):
    def __init__(self,mean,scale,output_scale,qi,vi,kind):
        super().__init__();self.kind=kind
        for key,value in (('mean',mean),('scale',scale),('output_scale',output_scale),('qi',qi),('vi',vi)):
            self.register_buffer(key,torch.as_tensor(value).clone())
        self.register_buffer('root_velocity_scale',torch.tensor([.5,.5,.5,2.,2.,2.]))
        self.net=nn.Sequential(nn.Linear(421,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,33))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def features(self,x):
        if self.kind=='inherited':
            z=((x-self.mean)/self.scale).clamp(-20,20);z=z.clone();z[:,95:229]=0
            return z
        if self.kind!='relative':raise ValueError(self.kind)
        from util.torch_util import quat_to_tan_norm
        goal=x[:,374:420]
        orientation=quat_to_tan_norm(goal[:,[4,5,6,3]])
        z=torch.cat(((x[:,:3]-goal[:,:3])/.05,(x[:,3:9]-orientation)/.1,
            x[:,9:15]/self.root_velocity_scale,(x[:,self.qi+8]-goal[:,self.qi])/.1,
            x[:,self.vi+48]*2,goal[:,self.qi],x[:,93:95]),-1)
        assert z.shape[-1]==116
        return torch.nn.functional.pad(z.clamp(-20,20),(0,421-116))

    def forward(self,x):return self.net(self.features(x))*self.output_scale


def load_head(folder):
    metadata=json.loads((folder/'architecture.json').read_text());s=torch.load(folder/'final.pt',weights_only=True)
    net=ProposalHead(s['mean'],s['scale'],s['output_scale'],s['qi'],s['vi'],metadata['kind'])
    net.load_state_dict(s);return net.eval().requires_grad_(False)


def fit(base,control,data,kind,seed,out,updates=4000):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed);net=ProposalHead(base.mean,base.scale,base.output_scale,control.q_indices,control.v_indices,kind)
    out.mkdir();torch.save(net.state_dict(),out/'initial.pt');net.to(device).train()
    mask=~data['validation'];assert set(data['world'][mask])==set(range(6))
    x=torch.tensor(data['x'][mask],device=device);y=torch.tensor((data['raw_target']-data['goal'])[mask],device=device)
    vx=torch.tensor(data['x'][~mask],device=device);vy=torch.tensor((data['raw_target']-data['goal'])[~mask],device=device)
    generator=torch.Generator(device=device).manual_seed(seed);optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
    start=time.perf_counter();history=[]
    for update in range(updates):
        ids=torch.randint(len(x),(512,),generator=generator,device=device)
        loss=((net(x[ids])-y[ids])/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (update+1)%500==0:
            history.append(dict(update=update+1,loss=float(loss.detach())));print('FIT',kind,seed,history[-1],flush=True)
    net.eval()
    with torch.no_grad():
        train_error=float((net(x)-y).square().mean().sqrt());heldout_error=float((net(vx)-vy).square().mean().sqrt())
    net.cpu();torch.save(net.state_dict(),out/'final.pt')
    metadata=dict(kind=kind,seed=seed,inputs=421,active_features=116 if kind=='relative' else 287,
        architecture='421-128ELU-128ELU-33, zero final layer, inherited per-motor output scale')
    save(out/'architecture.json',metadata);save(out/'history.json',history)
    result=dict(kind=kind,seed=seed,updates=updates,batch_size=512,lr=.0001,train_examples=len(x),heldout_examples=len(vx),
        train_rmse=train_error,heldout_rmse=heldout_error,device=device,wall_seconds=time.perf_counter()-start)
    save(out/'report.json',result);return net,result


def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(15,16,17,18,19),required=True);args=p.parse_args()
    step=args.step;out=ROOT/f'step-{step:02d}';extra=[PLAN,Path(__file__).with_name('ARM_PROPOSAL_30_ADAPTIVE_PLAN.md'),TEACHER/'training-samples.npz']
    if step in (16,17,19):extra += [ROOT/'step-15'/kind/'final.pt' for kind in ('inherited','relative')]
    if step==19:extra.append(ROOT/'step-18/relative/final.pt')
    inputs=seal(out,extra);save(out/'protocol.json',dict(step=step,inputs=inputs,training_poses=list(range(6)),heldout_poses=[6,7]))
    env,anchor,base,poses,actors=setup();results=[];training=[]
    if step in (15,18):
        data=dict(np.load(TEACHER/'training-samples.npz'))
        for kind in (('inherited','relative') if step==15 else ('relative',)):
            _,r=fit(base,env.engine.policy_control,data,kind,711028 if step==15 else 721028,out/kind);training.append(r)
    if step!=15:
        heads=[('inherited',ROOT/'step-15/inherited'),('relative',ROOT/'step-15/relative')]
        if step==18:heads=[('relative-replica',out/'relative')]
        elif step==19:heads.append(('relative-replica',ROOT/'step-18/relative'))
        for label,folder in heads:
            net=load_head(folder)
            for variant in ((False,True) if step==18 else (step==17,)):
                quiet=step==19;adapter=CorrectionEnvironment(env,anchor,net,poses,variant=variant,quiet=quiet)
                name=label+('-quiet' if quiet else '-varied' if variant else '-nominal')
                r=episode(adapter,actors['zero'],dict(actor=label,mode='mean',variant=variant,quiet=quiet),out/name,keep_features=not quiet)
                if quiet:
                    z=np.load(out/name/'trace.npz');baseline=np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
                    assert r['quiet_passed']==16 and all(np.array_equal(z[k],baseline[k]) for k in ('qpos','issued'))
                results.append(dict(folder=name,**r));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=step,results=results,training=training,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
