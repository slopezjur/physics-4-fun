"""Bounded value-only Monte Carlo warm-up, with untouched evaluation episodes."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .probe_arm_execution_distribution import ROOT,seal
from .probe_arm_onpolicy_value import returns
from .train_arm_correction_ppo import CorrectionNetwork
from .run_support_skill_study import digest,save


def dataset(root,rows,training):
    x=[];target=[];metadata=[]
    for r in rows:
        c=r['config']
        if c['actor']!='zero' or (c['seed']<=611004)!=training:continue
        z=dict(np.load(root/r['folder']/'trace.npz'));y=returns(z)
        pose=np.broadcast_to(np.arange(8),z['alive'].shape)
        mask=z['alive']&(pose<6 if training else True)
        x.append(z['x'][mask]);target.append(y[mask]);metadata.append(pose[mask])
    return torch.tensor(np.concatenate(x)),torch.tensor(np.concatenate(target)),np.concatenate(metadata)


def main():
    torch.set_num_threads(2)
    source=ROOT/'step-21';r=json.loads((source/'report.json').read_text())
    assert all(digest(Path(p))==h for p,h in r['outputs'].items())
    extra=[source/'report.json',Path(__file__).with_name('ARM_RECOVERY_30_ADAPTIVE_PLAN.md')]
    extra += [source/e['folder']/'trace.npz' for e in r['results'] if e['config']['actor']=='zero']
    out=ROOT/'step-22';inputs=seal(out,extra)
    x,y,ids=dataset(source,r['results'],True);vx,vy,vids=dataset(source,r['results'],False)
    assert set(ids)==set(range(6)) and set(vids)==set(range(8))
    save(out/'protocol.json',dict(step=22,inputs=inputs,seeds=[611028,621028],epochs=4,critic_lr=.0003,
        training_samples=len(y),training_seeds=list(range(611001,611005)),training_poses=list(range(6)),
        validation_seeds=[611005,611006],actor_updates=0))
    results=[];validation=[]
    for seed in (611028,621028):
        start=time.perf_counter();torch.manual_seed(seed);critic=CorrectionNetwork(1)
        folder=out/str(seed);folder.mkdir();torch.save(critic.state_dict(),folder/'initial.pt')
        with torch.no_grad():initial=critic(vx)
        optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4);history=[]
        for epoch in range(4):
            for batch in torch.randperm(len(y)).split(256):
                loss=(critic(x[batch])-y[batch]).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);optimizer.step()
                history.append(float(loss.detach()))
        critic.eval();torch.save(critic.state_dict(),folder/'final.pt')
        with torch.no_grad():prediction=critic(vx);train_mse=float((critic(x)-y).square().mean())
        for heldout in (False,True):
            mask=torch.tensor(vids>=6 if heldout else vids<6)
            mse=float((prediction[mask]-vy[mask]).square().mean())
            before=float((initial[mask]-vy[mask]).square().mean())
            validation.append(dict(seed=seed,heldout=heldout,count=int(mask.sum()),mse=mse,initial_mse=before,
                ratio=mse/before,admitted=mse<=.8*before,
                mean_prediction=float(prediction[mask].mean()),mean_return=float(vy[mask].mean()),
                bias=float((prediction[mask]-vy[mask]).mean())))
        results.append(dict(seed=seed,updates=len(history),wall_seconds=time.perf_counter()-start,train_mse=train_mse))
        save(folder/'history.json',history)
        print('WARMUP',results[-1],validation[-2:],flush=True)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=22,results=results,inputs_unchanged=True,actor_updates=0,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    review=ROOT/'step-23';review.mkdir(exist_ok=False)
    save(review/'report.json',dict(step=23,results=validation,admitted=all(v['admitted'] for v in validation),
        validation_seeds=[611005,611006],training_poses=list(range(6)),heldout_poses=[6,7],
        source_hash=digest(out/'report.json'),interpretation='Realized-return MSE only; does not certify action ordering.'))


if __name__=='__main__':main()
