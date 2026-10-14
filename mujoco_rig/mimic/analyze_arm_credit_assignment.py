"""Compare paired physical returns with frozen value-model credit signals."""
import json
from pathlib import Path

import numpy as np

from .probe_arm_execution_distribution import ROOT
from .run_support_skill_study import digest,save

GAMMA=.995


def signals(z,h):
    live=z['alive'][:h];rewards=z['reward'][:h]*live
    discount=GAMMA**np.arange(h)
    empirical=(discount[:,None]*rewards).sum(0)
    bootstrap=GAMMA**h*z['next_value'][h-1]*(live[h-1]&~z['terminal'][h-1])[:,None]
    td=z['reward'][:h,:,None]+GAMMA*z['next_value'][:h]*(~z['terminal'][:h,:,None])-z['value'][:h]
    gae=((GAMMA*.98)**np.arange(h)[:,None,None]*td*live[:,:,None]).sum(0)
    return dict(physical=empirical,bootstrap=bootstrap,gae=gae,
        one_step=td[0],total=empirical[:,None]+bootstrap-z['value'][0])


def agreement(rows,pred='one_step',h=120,heldout=None):
    chosen=[r for r in rows if r['horizon']==h and (heldout is None or r['heldout']==heldout)
            and abs(r['physical_delta'])>1e-5]
    values=[(r[pred][i],r['physical_delta']) for r in chosen for i in range(2) if abs(r[pred][i])>1e-5]
    return dict(comparisons=len(values),same_sign=sum(a*b>0 for a,b in values),
        fraction=sum(a*b>0 for a,b in values)/len(values) if values else None)


def main():
    rows=[];inputs={};checks=[]
    for step in (11,12,13):
        root=ROOT/f'step-{step:02d}';protocol=json.loads((root/'protocol.json').read_text());report=json.loads((root/'report.json').read_text())
        assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
        assert all(digest(Path(p))==h for p,h in report['outputs'].items())
        assert len({r['start_sha256'] for r in report['results']})==1
        traces={r['name']:dict(np.load(root/(r['name']+'.npz'))) for r in report['results']}
        for name,z in traces.items():
            inputs[str(root/(name+'.npz'))]=digest(root/(name+'.npz'))
            assert all(np.isfinite(z[k]).all() for k in ('qpos','value','next_value','reward'))
            assert np.allclose(z['components'].sum(-1),z['reward'],atol=1e-6,rtol=0)
            assert np.array_equal(z['value'][0],traces['zero']['value'][0])
            if name=='zero':continue
            for h in (3,12,60,120):
                result,base=signals(z,h),signals(traces['zero'],h)
                for pose in range(8):
                    row=dict(phase=protocol['phase'],branch=name,pose=pose,heldout=pose>=6,horizon=h,
                        physical_delta=float(result['physical'][pose]-base['physical'][pose]),
                        survived=bool(z['alive'][h-1,pose] and not z['failed'][h-1,pose]))
                    for key in ('bootstrap','gae','one_step','total'):
                        row[key]=(result[key][pose]-base[key][pose]).tolist()
                    rows.append(row)
        checks.append(dict(step=step,matched_prefixes=True,reward_components_match=True,branches=9,cases=72))
    for step,phase in ((14,'ramp'),(15,'hold'),(16,'return')):
        selected=[r for r in rows if r['phase']==phase];out=ROOT/f'step-{step:02d}';out.mkdir(exist_ok=False)
        save(out/'report.json',dict(step=step,phase=phase,rows=selected,
            one_step_ranking=agreement(selected),gae_ranking=agreement(selected,'gae'),inputs=inputs,
            limitation='Critics trained on noisy correction-policy data; common zero-policy prefixes and continuations are diagnostic off-policy states, not an on-policy critic certification.'))
    out=ROOT/'step-17';out.mkdir(exist_ok=False)
    key=lambda r:(r['phase'],r['branch'],r['pose'])
    short={key(r):r for r in rows if r['horizon']==60};long={key(r):r for r in rows if r['horizon']==120}
    pairs=[(short[k]['physical_delta'],long[k]['physical_delta']) for k in short if abs(short[k]['physical_delta'])>1e-5 and abs(long[k]['physical_delta'])>1e-5]
    save(out/'report.json',dict(step=17,nontrivial_comparisons=len(pairs),sign_reversals=sum(a*b<0 for a,b in pairs),
        by_horizon={str(h):dict(one_step=agreement(rows,h=h),gae=agreement(rows,'gae',h)) for h in (3,12,60,120)}))
    out=ROOT/'step-18';out.mkdir(exist_ok=False)
    groups=[]
    for phase in ('ramp','hold','return'):
        for h in (3,12,60,120):
            group=[r for r in rows if r['phase']==phase and r['horizon']==h]
            groups.append(dict(phase=phase,horizon=h,mean_abs_physical_delta=float(np.mean([abs(r['physical_delta']) for r in group])),
                mean_abs_bootstrap_delta=float(np.mean([abs(v) for r in group for v in r['bootstrap']])),
                mean_abs_one_step_delta=float(np.mean([abs(v) for r in group for v in r['one_step']]))))
    save(out/'report.json',dict(step=18,components=groups))
    out=ROOT/'step-19';out.mkdir(exist_ok=False)
    learned=[]
    for phase in ('ramp','hold','return'):
        for seed in (521028,531028):
            for h in (60,120):
                positive={r['pose']:r for r in rows if r['phase']==phase and r['horizon']==h and r['branch']==f'learned-{seed}'}
                negative={r['pose']:r for r in rows if r['phase']==phase and r['horizon']==h and r['branch']==f'opposite-{seed}'}
                learned.append(dict(phase=phase,actor=seed,horizon=h,better_than_zero=sum(r['physical_delta']>1e-5 for r in positive.values()),
                    better_than_opposite=sum(positive[i]['physical_delta']>negative[i]['physical_delta']+1e-5 for i in positive),
                    mean_delta=float(np.mean([r['physical_delta'] for r in positive.values()]))))
    save(out/'report.json',dict(step=19,results=learned))
    out=ROOT/'step-20';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(step=20,training_pose_rankings={str(h):agreement(rows,h=h,heldout=False) for h in (60,120)},
        heldout_pose_rankings={str(h):agreement(rows,h=h,heldout=True) for h in (60,120)},
        heldout_rows=[r for r in rows if r['heldout'] and r['horizon'] in (60,120)]))
    save(ROOT/'credit-audit.json',dict(checks=checks,steps=list(range(11,21)),inputs=inputs))
    print(json.dumps(dict(horizon=json.loads((ROOT/'step-17/report.json').read_text()),learned=learned),indent=2))


if __name__=='__main__':main()
