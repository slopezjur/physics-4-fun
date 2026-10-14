"""Complete one fresh-seed matched curriculum comparison, including evidence verification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .audit_stand_failures import read, save
from .stand_experiment_evidence import protect, seal


def run(args):
    ledger=read(args.ledger)
    milestone=next(m for m in ledger['milestones'] if m['number']==args.number)
    seed=milestone['seed']
    assert args.number in (14,15) and seed in (301027,301028)
    milestone['status']='running'
    save(args.ledger,ledger)
    prior=args.prior
    reports,roots={},{}
    for arm in milestone['arm_order']:
        root=Path(milestone['arm_roots'][arm])
        assert not root.exists()
        command=[sys.executable,'-u','-m','mujoco_rig.mimic.train_stand_push_curriculum',
            '--prior',str(prior),'--out',str(root),'--seed',str(seed),'--arm',arm]
        with Path(str(root)+'.console.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(root/'final-verification.json')['verified']
        reports[arm]=read(root/'report.json')
        roots[arm]=root
        prior=root
        save(args.ledger.parent/f'milestone-{args.number}-progress.json',dict(completed_arms=list(reports)))
    quiet,curriculum=reports['quiet'],reports['curriculum']
    assert quiet['initial_hash']==curriculum['initial_hash']
    assert quiet['updates']==curriculum['updates']==1200
    assert quiet['forced_world_control_intervals']==0 and curriculum['forced_world_control_intervals']>0
    assert read(roots['quiet']/'protocol.json')['config']==read(roots['curriculum']/'protocol.json')['config']
    comparison={}
    for backend in ('native','gpu'):
        row={}
        for arm,report in reports.items():
            row[arm]=dict(quiet_stand=report['sustained_assessment']['results'][backend]['standing_passes'],
                recovery_90=sum(pair[backend]['recovery_passes'] for name,pair in report['push_results'].items() if name.startswith('90N')),
                recovery_180=sum(pair[backend]['recovery_passes'] for name,pair in report['push_results'].items() if name.startswith('180N')))
        row['curriculum_improves']=bool(row['curriculum']['quiet_stand']==8
            and row['curriculum']['recovery_90']>=row['quiet']['recovery_90']
            and row['curriculum']['recovery_180']>row['quiet']['recovery_180'])
        comparison[backend]=row
    out=Path(milestone['comparison_root'])
    inputs,sources=protect(prior,out,[Path(__file__)])
    last=read(prior/'protocol.json')
    save(out/'protocol.json',dict(inputs=inputs,sources=sources,config=last['config'],demonstrations=last['demonstrations'],
        model_sha256=last['model_sha256'],reference_sha256=last['reference_sha256'],seed=seed,arm_roots={k:str(v) for k,v in roots.items()},
        comparison_rule='Require 8/8 quiet 60-second Stand, no decrease in 90 N recovery, and higher 180 N recovery on each backend',
        initialize_from=None,maximum_training_updates=2400,maximum_cases=384))
    result=dict(seed=seed,comparison=comparison,curriculum_improves=all(x['curriculum_improves'] for x in comparison.values()),
        matched_initial_hash=quiet['initial_hash'],new_training_updates=2400,new_transitions=9830400,
        training_seconds=sum(r['training_seconds'] for r in reports.values()),cases=384,
        warmup_600_full_hashes={arm:r['assessments']['updates-600']['full_tensor_hash'] for arm,r in reports.items()},
        arena_changed=False)
    save(out/'report.json',result)
    seal(out,inputs,sources,verified=True,matched_fresh_initialization=True,matched_config_and_update_budget=True,
        cases=384,training_updates=2400)
    ledger=read(args.ledger)
    milestone=next(m for m in ledger['milestones'] if m['number']==args.number)
    milestone.update(status='complete',new_physical_cases=384,new_training_updates=2400,new_transitions=9830400,
        synchronized_training_seconds=result['training_seconds'],comparison=comparison,curriculum_improves=result['curriculum_improves'],
        outcome='Fresh matched comparison verified; curriculum improvement rule '+('passed.' if result['curriculum_improves'] else 'not passed.'))
    ledger['completed_experiments']=sum(m['status']=='complete' for m in ledger['milestones'])
    ledger['latest_sealed_root']=str(out)
    save(args.ledger,ledger)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--number',type=int,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=Path('logs/mimickit-training/stand-milestones-20261005-01/ledger.json'))
    run(parser.parse_args())
