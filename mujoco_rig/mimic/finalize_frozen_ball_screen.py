"""Seal a fully collected screen while retaining rejected condition admissions."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(root, out):
    if out.exists():
        raise ValueError('Output exists')
    p = json.loads((root/'protocol.json').read_text())
    progress = json.loads((root/'progress.json').read_text())
    expected = p['settings']['maximum_cases']
    if progress['evaluation_cases'] != expected or len(progress['results'])*16 != expected:
        raise ValueError('Incomplete screen cannot be finalized')
    inputs = dict(p['inputs'])
    for s in p['sources']:
        inputs[s['snapshot']] = s['sha256']
    for name,h in inputs.items():
        if digest(Path(name)) != h:
            raise ValueError(f'Input/source mismatch: {name}')
    for f in root.rglob('*'):
        if f.is_file():
            inputs[str(f)] = digest(f)
    inputs[str(Path(__file__))] = digest(Path(__file__))
    conditions = {}
    for name in (c['name'] for c in p['settings']['conditions']):
        rows = [r for r in progress['results'] if r['condition']['name']==name]
        parity = all(r['parity']['passed'] for r in rows)
        hits = all(a['hit_cases']==8 for r in rows for a in r['assessments'].values())
        chest = all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies'] and
            'Head' not in c['clusters'][0]['bodies'] for r in rows for c in r['bodies']['cases'])
        conditions[name] = dict(cases=16*len(rows), backend_parity=parity, all_shots_hit=hits,
            chest_first_without_head=bool(chest), admitted=bool(parity and hits and chest),
            recoveries=sum(a['recovery_passes'] for r in rows for a in r['assessments'].values()))
    out.mkdir(parents=True)
    report = dict(source_root=str(root),evaluation_cases=expected,conditions=conditions,
        original_failure_preserved=(root/'failure.json').exists(),new_physical_cases=0,
        all_conditions_admitted=all(c['admitted'] for c in conditions.values()))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(verified=True,inputs=inputs,
        scope='Integrity of fully collected screen; rejected admissions remain rejected',
        output_sha256=digest(out/'report.json')),indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.out)
