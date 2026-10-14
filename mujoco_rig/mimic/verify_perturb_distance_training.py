"""Reconcile saved matched training and coverage without further simulation."""
import argparse
from pathlib import Path
import numpy as np
from .audit_stand_failures import digest,read,save


def main(args):
    proof=read(args.run/'final-verification.json')
    assert proof['verified'] and proof['training_updates']==200
    for path,expected in proof['output_hashes'].items():
        assert digest(Path(path))==expected,path
    protocol=read(args.run/'protocol.json')
    for path,expected in protocol['inputs'].items():
        assert digest(Path(path))==expected,path
    report=read(args.run/'report.json')
    assert read(args.run/'mixed-starts/cross-arm-admission.json')['passed']
    results={}
    for arm in ('fixed','mixed'):
        folder=args.run/arm
        dt=read(folder/'contract.json')['control_timestep']
        paths=sorted((folder/'rollouts').glob('*.npz'))
        assert len(paths)==100 and not list(folder.glob('updates-*'))
        first=second=contacts=0
        coverage=set()
        second_coverage=set()
        for path in paths:
            with np.load(path) as z:
                assert z['steps'].shape==(64,64)
                assert all(np.isfinite(z[k]).all() for k in z.files)
                a=(z['steps']==z['first_launch']) & z['launched']
                b=(z['steps']==z['first_launch']+int(np.ceil(4/dt))) & z['launched']
                assert np.array_equal(a|b,z['launched'])
                first+=int(a.sum());second+=int(b.sum())
                contacts+=int(np.count_nonzero(z['normal_impulse']))
                for mask,target in ((a,coverage),(b,second_coverage)):
                    frame,world=np.nonzero(mask)
                    target.update(zip((world%8).tolist(),z['first_launch'][frame,world].tolist(),
                                      np.rint(z['launch_distance_m'][frame,world]*100).astype(int).tolist()))
        expected=report['results'][arm]
        assert first+second==expected['launches'] and contacts==expected['contact_world_intervals']
        assert len(coverage)==(64 if arm=='mixed' else 32)
        assert len(second_coverage)==(64 if arm=='mixed' else 32)
        assert expected['updates']==100 and expected['transitions']==409600
        results[arm]=dict(first_throws=first,second_throws=second,first_pose_onset_distance_combinations=len(coverage),
                          second_pose_onset_distance_combinations=len(second_coverage),training_seconds=expected['training_seconds'])
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'report.json',dict(verified=True,results=results,training_updates=200,transitions=819200,
         rehearsal_presentations=2457600,training_seal_sha256=digest(args.run/'final-verification.json'),
         reviewer_sha256=digest(Path(__file__))))
    print(results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
