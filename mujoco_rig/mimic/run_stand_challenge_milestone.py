"""Run one predeclared frozen-policy milestone sequentially and record verified results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def run(args):
    ledger = read(args.ledger)
    milestone = next(m for m in ledger['milestones'] if m['number'] == args.number)
    specifications = [Path(p) for p in milestone['specifications']]
    roots = [Path(p) for p in milestone['roots']]
    runner = milestone.get('runner', 'evaluate_stand_challenges')
    assert runner in ('evaluate_stand_challenges', 'evaluate_stand_push_feedback', 'evaluate_stand_push_feedback_v2')
    assert len(specifications) == len(roots) and roots
    assert all(read(p)['milestone'] == args.number for p in specifications)
    milestone['status'] = 'running'
    write(args.ledger, ledger)
    prior = args.prior
    results = []
    for root, specification in zip(roots, specifications):
        command = [sys.executable, '-u', '-m', f'mujoco_rig.mimic.{runner}',
                   '--prior', str(prior), '--out', str(root), '--specification', str(specification)]
        assert not root.exists(), f'Preserve existing evidence: {root}'
        with Path(str(root)+'.console.log').open('w', encoding='utf-8') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        verification, report = read(root/'final-verification.json'), read(root/'report.json')
        assert verification['verified'] and verification['inputs_and_sources_unchanged']
        assert verification['cases'] == report['cases'] == 16*len(read(specification)['scenarios'])
        result = dict(root=str(root), cases=report['cases'],
            stand=sum(r['standing_passes'] for pair in report['results'].values() for r in pair.values()),
            recovery=sum(r['recovery_passes'] for pair in report['results'].values() for r in pair.values()),
            scenarios={name: {backend: dict(stand=r['standing_passes'], recovery=r['recovery_passes'],
                survival=r['mean_survival_seconds']) for backend,r in pair.items()}
                for name,pair in report['results'].items()})
        results.append(result)
        write(args.ledger.parent/f'milestone-{args.number}-progress.json', dict(results=results))
        print(result, flush=True)
        prior = root
    ledger = read(args.ledger)
    milestone = next(m for m in ledger['milestones'] if m['number'] == args.number)
    cases = sum(r['cases'] for r in results)
    assert cases <= milestone['maximum_physical_cases']
    milestone.update(status='complete', new_physical_cases=cases, new_training_updates=0,
        continuous_stand_passes=sum(r['stand'] for r in results), recovery_passes=sum(r['recovery'] for r in results),
        outcome=f"{sum(r['stand'] for r in results)}/{cases} continuous Stand; {sum(r['recovery'] for r in results)}/{cases} recovery. Evidence verified.")
    ledger['completed_experiments'] = sum(m['status']=='complete' for m in ledger['milestones'])
    ledger['latest_sealed_root'] = str(prior)
    write(args.ledger, ledger)
    print(dict(completed_milestone=args.number, completed_experiments=ledger['completed_experiments']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, default=Path('logs/mimickit-training/stand-milestones-20261005-01/ledger.json'))
    run(parser.parse_args())
