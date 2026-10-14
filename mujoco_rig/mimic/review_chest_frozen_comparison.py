"""Verify direct saved evidence and compare declared frozen-policy assessments."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


def run(args):
    spec = read(args.protocol)
    inputs = {str(args.protocol): digest(args.protocol), str(Path(__file__)): digest(Path(__file__))}
    summaries, records, assets = {}, {}, {}
    for label, root_name in spec['roots'].items():
        root = Path(root_name)
        proof = read(root / 'final-verification.json')
        assert proof['verified'] and proof['inputs_and_sources_unchanged']
        inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
        hashes = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}

        def checked(path):
            h = digest(path)
            assert hashes.get(str(path.resolve())) == h, path
            inputs[str(path)] = h
            return path

        report, protocol = read(checked(root / 'report.json')), read(checked(root / 'protocol.json'))
        assert report['training_updates'] == 0
        assets[label] = [digest(checked(root / n)) for n in ('dummy.xml', 'stationary_reference.npz')]
        rows = []
        for row in report['results']:
            native = row['assessments']['native']
            bodies = row['bodies']['cases']
            chest = all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies'] and 'Head' not in c['clusters'][0]['bodies'] for c in bodies)
            assert native['actor_hash'] == row['assessments']['gpu']['actor_hash']
            for backend, assessment in row['assessments'].items():
                folder = root / f"{row['seed']}-{row['actor']}-{row['condition']['name']}-{row['sign']:+d}-{backend}"
                saved = read(checked(folder / 'report.json'))
                assert saved == assessment
                with np.load(checked(folder / 'trace.npz'), allow_pickle=False) as data:
                    initial = {k: data[k][0].tolist() for k in ('qpos_before', 'qvel_before', 'pending_before')}
                if backend == 'native':
                    assert read(checked(folder / 'body-report.json')) == row['bodies']
                key = (row['seed'], row['condition']['launch_seconds'], row['condition']['seconds'],
                       row['condition']['mass_kg'], row['condition']['speed_m_s'], row['condition']['distance_m'],
                       protocol['settings'].get('direction_degrees', 0), row['sign'], backend)
                assert key not in records.setdefault(label, {})
                records[label][key] = dict(initial=initial, assessment=assessment,
                    admitted=row['parity']['passed'] and chest,
                    settings=protocol['settings'], root=str(root))
            rows.append(dict(seed=row['seed'], condition=row['condition'], sign=row['sign'],
                backend_admitted=row['parity']['passed'], chest_first=chest,
                outcomes={b: dict(passes=a['recovery_passes'], survivors=sum(c['survived'] for c in a['cases']),
                    passed_cases=[c['case'] for c in a['cases'] if c['recovery_pass']],
                    failures=[c for c in a['cases'] if not c['recovery_pass']]) for b, a in row['assessments'].items()}))
        summaries[label] = dict(cases=report['evaluation_cases'],
            recoveries=sum(a['recovery_passes'] for r in report['results'] for a in r['assessments'].values()),
            admitted=all(r['backend_admitted'] and r['chest_first'] for r in rows), rows=rows)
    comparisons = []
    for left, right in spec.get('comparisons', []):
        assert assets[left] == assets[right]
        common = records[left].keys() & records[right].keys()
        assert common
        for key in sorted(common):
            a, b = records[left][key], records[right][key]
            initial = a['initial'] == b['initial']
            assert initial
            pa = {c['case'] for c in a['assessment']['cases'] if c['recovery_pass']}
            pb = {c['case'] for c in b['assessment']['cases'] if c['recovery_pass']}
            comparisons.append(dict(left=left, right=right, condition_key=key,
                identical_start=initial, admitted=a['admitted'] and b['admitted'],
                left_passes=sorted(pa), right_passes=sorted(pb),
                gained=sorted(pb-pa), lost=sorted(pa-pb)))
    args.out.mkdir(exist_ok=False)
    shutil.copy2(args.protocol, args.out / 'protocol.json')
    shutil.copy2(Path(__file__), args.out / 'review-source.py')
    save(args.out / 'report.json', dict(summaries=summaries, comparisons=comparisons,
        additional_simulations=0, additional_training_updates=0,
        all_admitted=all(s['admitted'] for s in summaries.values())))
    seal(args.out, inputs, [], verified=True, additional_simulations=0, additional_training_updates=0)
    print({k: {n: v[n] for n in ('cases', 'recoveries', 'admitted')} for k, v in summaries.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
