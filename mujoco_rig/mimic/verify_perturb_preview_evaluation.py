"""Check final frozen evidence and report each repeated encounter's body contact."""
import argparse
from pathlib import Path

import numpy as np

from .audit_stand_failures import digest, read, save
from .run_chest_repeated_screen import repeated_summary


def main(args):
    proof = read(args.run/'final-verification.json')
    assert proof['verified']
    for path,expected in proof['output_hashes'].items():
        assert digest(Path(path))==expected,path
    report = read(args.run/'report.json')
    counts, bodies = {},{}
    for name,result in report['results'].items():
        counts[name] = dict(quiet={b:r['standing_passes'] for b,r in result['quiet'].items()},
            impacts={label:{b:rows[b]['recovery_passes'] for b in ('native','gpu')} for label,rows in result['impacts'].items()},
            admitted=result['admitted'])
        rows = []
        for backend in ('native','gpu'):
            label = result['impacts']['repeat'].get('trace_label','repeat')
            folder = args.run/name/f'hit-{label}-{backend}'
            trial,contract = read(folder/'report.json'),read(folder/'contract.json')
            with np.load(folder/'trace.npz') as z:
                reviewed = repeated_summary(z,[5,9],contract['control_timestep'])
                assert sum(c['passed'] for c in reviewed)==trial['recovery_passes']
                assert all(a['encounters']==b['encounters'] for a,b in zip(reviewed,trial['cases']))
            if backend=='native':
                attribution = read(folder/'body-report.json')
                for case,body in zip(trial['cases'],attribution['cases']):
                    for encounter in case['encounters']:
                        start = encounter['launch_frame']
                        end = int(np.ceil(9/contract['control_timestep'])) if encounter['index']==0 else 10**9
                        clusters = [c for c in body['clusters'] if start<=c['first_frame']<end]
                        chest = bool(clusters and 'Chest' in clusters[0]['bodies'] and 'Head' not in clusters[0]['bodies'])
                        rows.append(dict(case=case['case'],encounter=encounter['index'],hit=encounter['hit'],
                                         chest_first=chest,first_bodies=list(clusters[0]['bodies']) if clusters else []))
                        if encounter['passed']:
                            assert chest
        bodies[name] = rows
    selected = report['selected']
    assert report['physical_cases']==464 and report['excluded_original_repeat_cases']==16
    assert all(r['standing_passes']==8 for r in report['retention'].values())
    assert all(row['chest_first'] for row in bodies[selected] if row['hit'])
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'report.json',dict(verified=True,physical_cases=464,excluded_cases=16,admitted_cases=448,counts=counts,
         repeated_body_attribution=bodies,selected=selected,retention_seconds=120,
         checkpoint=report['checkpoint'],evaluation_seal_sha256=digest(args.run/'final-verification.json'),
         reviewer_sha256=digest(Path(__file__))))
    print(dict(selected=selected,counts=counts),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
