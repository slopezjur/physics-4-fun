"""Final provenance, physical admission and matched-learning audit."""
import json
from pathlib import Path

import numpy as np
import torch

from .audit_arm_execution_distribution import audit
from .probe_arm_execution_distribution import ROOT
from .train_arm_correction_ppo import OUT as PPO,CorrectionNetwork
from .fit_arm_value_warmup import dataset
from .run_support_skill_study import digest,save


def read(path):return json.loads(path.read_text())


def main():
    torch.set_num_threads(2)
    reports={step:read(ROOT/f'step-{step:02d}/report.json') for step in range(1,31)}
    evaluations=[]
    physical=(1,2,3,4,5,7,8,9,10,21,24,25,26,28,29,30)
    for step in physical:evaluations.append(audit(ROOT/f'step-{step:02d}'))
    for step in (11,12,13,22):
        root=ROOT/f'step-{step:02d}';protocol=read(root/'protocol.json');report=reports[step]
        assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
        assert all(digest(Path(p))==h for p,h in report['outputs'].items())
    for step in (6,14,15,16,27):
        assert all(digest(Path(p))==h for p,h in reports[step]['inputs'].items())
    assert reports[23]['source_hash']==digest(ROOT/'step-22/report.json')
    initial=torch.load(PPO/'initial-correction.pt',weights_only=True)
    training=[]
    for step,seed,warm in ((24,611028,True),(25,621028,True),(28,611028,False),(29,621028,False)):
        root=ROOT/f'step-{step:02d}';start=torch.load(root/'fit/initial.pt',weights_only=True)
        final=torch.load(root/'fit/final.pt',weights_only=True)
        assert all(torch.equal(v,start[k]) for k,v in initial.items())
        assert any(not torch.equal(v,final[k]) for k,v in initial.items())
        assert all(torch.isfinite(v).all() for v in final.values())
        critic=torch.load(root/'critic-initial.pt',weights_only=True)
        expected=torch.load(ROOT/f'step-22/{seed}'/('final.pt' if warm else 'initial.pt'),weights_only=True)
        assert all(torch.equal(v,expected[k]) for k,v in critic.items())
        history=read(root/'fit/history.json');run=reports[step]['training']
        assert len(history)==32 and run['transitions']==24576
        assert all(h['transitions']==(i+1)*768 for i,h in enumerate(history))
        assert sum(h['actor_steps'] for h in history)==run['actor_gradient_steps']
        for h in history:
            assert sum(p['count'] for p in h['reward_parts'].values())==768
            mean=sum(sum(p['sums'].values()) for p in h['reward_parts'].values())/768
            assert abs(mean-h['mean_reward'])<1e-6
            assert len(h['post_update_full_batch_kl'])==h['actor_steps']
        kls=[v for h in history for v in h['post_update_full_batch_kl']]
        training.append(dict(step=step,warm=warm,**run,early_stops=sum(h['actor_early_stop'] for h in history),
            max_full_kl=max(kls),mean_full_kl=float(np.mean(kls))))
    for warm,fresh in ((24,28),(25,29)):
        a=read(ROOT/f'step-{warm:02d}/fit/history.json')[0];b=read(ROOT/f'step-{fresh:02d}/fit/history.json')[0]
        for key in ('mean_reward','actor_eligible','reward_parts','transmitted_coordinates'):
            assert a[key]==b[key],(warm,key)
    # Independently re-evaluate held-out warm-up calibration from sealed traces.
    vx,vy,vids=dataset(ROOT/'step-21',reports[21]['results'],False)
    with torch.no_grad():
        for row in reports[23]['results']:
            c=CorrectionNetwork(1);c.load_state_dict(torch.load(ROOT/f'step-22/{row["seed"]}/final.pt',weights_only=True))
            mask=torch.tensor(vids>=6 if row['heldout'] else vids<6)
            mse=float((c(vx[mask])-vy[mask]).square().mean())
            assert abs(mse-row['mse'])<.002
    quiet_path=Path('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    base=np.load(quiet_path);quiet_cases=0
    for step in (26,30):
        for r in reports[step]['results']:
            if not r['config'].get('quiet'):continue
            z=np.load(ROOT/f'step-{step:02d}'/r['folder']/'trace.npz')
            assert all(np.array_equal(z[k],base[k]) for k in ('qpos','issued'))
            assert r['quiet_passed']==16;quiet_cases+=16
    outcomes=[]
    for step in physical:
        for r in reports[step]['results']:
            outcomes.append(dict(step=step,actor=r['config']['actor'],mode=r['config']['mode'],
                quiet=r['config'].get('quiet',False),cases=len(r['cases']),holds=r['stable_holds'],
                returned=r['returned'],complete=r['complete']))
    failed=ROOT/'step-24-failed-frozen-actor';failure_protocol=read(failed/'protocol.json')
    for path,h in failure_protocol['inputs'].items():
        archived=failed/'sources'/Path(path).name
        assert digest(archived if archived.exists() else Path(path))==h
    total_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    result=dict(completed_steps=list(range(1,31)),evaluation_audits=evaluations,
        exact_initial_actors=True,matched_first_collections=True,correct_critic_initializations=True,
        quiet_cases=quiet_cases,quiet_exact=True,training=training,outcomes=outcomes,
        ppo_transitions=sum(r['transitions'] for r in training),
        actor_gradient_updates=sum(r['actor_gradient_steps'] for r in training),
        ppo_wall_seconds=sum(r['wall_seconds'] for r in training),
        warmup_value_updates=sum(r['updates'] for r in reports[22]['results']),
        warmup_unique_samples=read(ROOT/'step-22/protocol.json')['training_samples'],
        end_to_end_cases=sum(r['cases'] for r in outcomes),matched_continuations=216,
        nonquiet_complete=sum(r['complete'] for r in outcomes if not r['quiet']),
        preserved_failed_attempt=dict(reason='Trainable actor left frozen by evaluation setup',
            collection_transitions=768,actor_updates=0,rerun_from_identical_initialization=True),
        bytes=total_bytes,arena_changed=False,
        report_hashes={str(ROOT/f'step-{step:02d}/report.json'):digest(ROOT/f'step-{step:02d}/report.json') for step in reports})
    save(ROOT/'final-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('outcomes','report_hashes')},indent=2))


if __name__=='__main__':main()
