"""Reconcile compact frozen outcomes and the predeclared candidate decision."""
import argparse
from pathlib import Path
import numpy as np
from .audit_stand_failures import digest,read,save
from .evaluate_stand_sustained import duration_summary
from .evaluate_stand_ball_boundary import summarize
from .run_chest_repeated_screen import repeated_summary
from .verify_stand_learning_curve import verify_gates


def main(args):
    proof=read(args.run/'final-verification.json')
    assert proof['verified']
    for path,expected in proof['output_hashes'].items():
        assert digest(Path(path))==expected,path
    report=read(args.run/'report.json')
    checked=0
    for retention_path in args.run.rglob('trace-retention.json'):
        folder=retention_path.parent
        retained=read(retention_path)
        assert digest(folder/'trace.npz')==retained['compact_trace_sha256']
        assert digest(folder/'report.json')==retained['report_sha256']
        trial=read(folder/'report.json')
        contract=read(folder/'contract.json')
        dt=contract['control_timestep']
        with np.load(folder/'trace.npz') as z:
            trace={k:z[k] for k in z.files}
        assert not any(k.startswith('substep_') for k in trace)
        assert all(np.isfinite(v).all() for v in trace.values())
        verify_gates(trace,'');verify_gates(trace,'pre_')
        assert trial['max_control_formula_error_nm']<=1e-4
        if 'launch_times_seconds' in trial:
            rows=repeated_summary(trace,trial['launch_times_seconds'],dt)
            assert sum(r['passed'] for r in rows)==trial['recovery_passes']
            assert all(a['encounters']==b['encounters'] for a,b in zip(rows,trial['cases']))
        elif 'speed_m_s' in trial:
            result=summarize(trace,dt,trial['requested_seconds'],5,False)
            assert result['recovery_passes']==trial['recovery_passes']
        else:
            result=duration_summary(trace,dt,trial['requested_seconds'])
            assert result['standing_passes']==trial['standing_passes']
        checked+=8
    assert checked==report['physical_cases']
    counts={name:dict(quiet={b:r['standing_passes'] for b,r in value['quiet'].items()},
        impacts={key:{b:r[b]['recovery_passes'] for b in ('native','gpu')} for key,r in value['impacts'].items()},
        admitted=value['admitted'],qualifies=value.get('qualifies')) for name,value in report['results'].items()}
    if report['selected']:
        assert report['results'][report['selected']]['qualifies']
        assert all(r['standing_passes']==8 for r in report['retention'].values())
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'report.json',dict(verified=True,physical_cases=checked,counts=counts,selected=report['selected'],
        evaluation_seal_sha256=digest(args.run/'final-verification.json'),reviewer_sha256=digest(Path(__file__)),
        limitation='Outcome and gate evidence independently recomputed. Full control equations passed in evaluators before explicitly recorded substep-data compaction.'))
    print(dict(cases=checked,counts=counts,selected=report['selected']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
