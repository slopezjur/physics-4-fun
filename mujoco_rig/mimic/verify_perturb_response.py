"""Independent outcome and training-accounting audit for the response block."""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit_stand_failures import digest, read, save
from .evaluate_stand_ball_boundary import summarize
from .evaluate_stand_sustained import duration_summary
from .run_chest_repeated_screen import repeated_summary
from .verify_stand_learning_curve import verify_gates


def verify_seal(root):
    proof=read(root/'final-verification.json')
    assert proof['verified']
    for path, expected in proof['output_hashes'].items():
        assert digest(Path(path))==expected, path
    for path, expected in read(root/'protocol.json')['inputs'].items():
        assert digest(Path(path))==expected, path
    return proof


def evaluation(root):
    verify_seal(root)
    cases=0
    counts={}
    for path in sorted(root.rglob('trace-retention.json')):
        folder=path.parent
        retention=read(path)
        assert digest(folder/'trace.npz')==retention['compact_trace_sha256']
        assert digest(folder/'report.json')==retention['report_sha256']
        report,contract=read(folder/'report.json'),read(folder/'contract.json')
        dt=contract['control_timestep']
        with np.load(folder/'trace.npz') as z:
            trace={k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in trace.values())
        verify_gates(trace,''); verify_gates(trace,'pre_')
        assert report['max_control_formula_error_nm']<=2e-4
        if 'launch_times_seconds' in report:
            summary=repeated_summary(trace,report['launch_times_seconds'],dt)
            assert all(a['encounters']==b['encounters'] for a,b in zip(summary,report['cases']))
            count=sum(c['passed'] for c in summary)
            assert count==report['recovery_passes']
        elif 'speed_m_s' in report:
            summary=summarize(trace,dt,report['requested_seconds'],5,False)
            count=summary['recovery_passes']
            assert count==report['recovery_passes']
            assert all(a['recovery_pass']==b['recovery_pass'] for a,b in zip(summary['cases'],report['cases']))
        else:
            summary=duration_summary(trace,dt,report['requested_seconds'])
            count=summary['standing_passes']
            assert count==report['standing_passes']
        counts[str(folder.relative_to(root))]=count
        cases+=8
    assert cases==read(root/'report.json')['physical_cases']
    return dict(verified=True,physical_cases=cases,counts=counts,
        limitation='Outcomes independently recomputed. Full substep controller verification occurred before documented compaction.')


def training(root):
    verify_seal(root)
    protocol=read(root/'protocol.json')
    result={}
    for label,expected in read(root/'report.json')['results'].items():
        folder=root/label
        dt=read(folder/'contract.json')['control_timestep']
        rows=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
        assert len(rows)==100 and rows[-1]['transitions']==409600
        assert rows[-1]['actor_optimizer_steps']==4800
        assert not list(folder.glob('updates-*'))
        starts=root/f'{label}-starts'
        assert read(starts/'report.json')['passed']
        assert read(folder/'start-admission.json')['common']['passed']
        assert read(folder/'start-admission.json')['extras_equal']
        if (starts/'cross-arm-admission.json').exists():
            assert read(starts/'cross-arm-admission.json')['passed']
        first=second=contacts=0
        coverage=[set(),set()]
        rollouts=sorted((folder/'rollouts').glob('*.npz'))
        assert len(rollouts)==100
        for p in rollouts:
            with np.load(p) as z:
                assert all(np.isfinite(z[k]).all() for k in z.files)
                a=(z['steps']==z['first_launch']) & z['launched']
                b=(z['steps']==z['first_launch']+int(np.ceil(4/dt))) & z['launched']
                assert np.array_equal(a|b,z['launched'])
                assert (z['position_error']<=2e-6).all() and (z['velocity_error']<=2e-6).all()
                first+=int(a.sum()); second+=int(b.sum())
                contacts+=int(np.count_nonzero(z['normal_impulse']))
                for mask,dest in zip((a,b),coverage):
                    t,w=np.nonzero(mask)
                    dest.update(zip((w%8).tolist(),z['first_launch'][t,w].tolist(),np.rint(z['launch_distance_m'][t,w]*100).astype(int).tolist()))
        assert first+second==expected['launches'] and contacts==expected['contact_world_intervals']
        assert all(len(c)==64 for c in coverage)
        assert expected['rehearsal']['examples']==1228800
        result[label]=dict(first_throws=first,second_throws=second,coverage=[len(c) for c in coverage])
    return dict(verified=True,training_updates=len(result)*100,transitions=len(result)*409600,arms=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training',action='store_true')
    args=parser.parse_args()
    result=training(args.run) if args.training else evaluation(args.run)
    result.update(seal_sha256=digest(args.run/'final-verification.json'),reviewer_sha256=digest(Path(__file__)))
    args.out.mkdir(exist_ok=False)
    save(args.out/'report.json',result)
    print(result)
