"""Reconcile bounded preview training from saved schedules and endpoint seals."""
import argparse
from pathlib import Path

import numpy as np

from .audit_stand_failures import digest, read, save


def main(args):
    proof = read(args.run/'final-verification.json')
    assert proof['verified'] and proof['training_updates']==200
    for path, expected in proof['output_hashes'].items():
        assert digest(Path(path))==expected,path
    protocol = read(args.run/'protocol.json')
    for path, expected in protocol['inputs'].items():
        assert digest(Path(path))==expected,path
    report = read(args.run/'report.json')
    dt = read(args.run/'single_ball/contract.json')['control_timestep']
    gap = int(np.ceil(4/dt))
    review = {}
    for arm in ('single_ball','quiet'):
        folder = args.run/arm
        rollouts = sorted((folder/'rollouts').glob('*.npz'))
        assert len(rollouts)==100 and not list(folder.glob('updates-*'))
        first = second = contacts = 0
        first_coverage, second_coverage = set(),set()
        for path in rollouts:
            with np.load(path) as z:
                assert z['steps'].shape==(64,64)
                assert all(np.isfinite(z[k]).all() for k in z.files)
                a = (z['steps']==z['first_launch']) & z['launched']
                b = (z['steps']==z['first_launch']+gap) & z['launched']
                assert np.array_equal(a|b,z['launched'])
                first += int(a.sum())
                second += int(b.sum())
                contacts += int(np.count_nonzero(z['normal_impulse']))
                assert (z['position_error']<=2e-6).all() and (z['velocity_error']<=2e-6).all()
                for mask,coverage in ((a,first_coverage),(b,second_coverage)):
                    frame,world = np.nonzero(mask)
                    coverage.update(zip((world%8).tolist(),z['first_launch'][frame,world].tolist()))
        expected = report['results'][arm]
        assert expected['updates']==100 and expected['transitions']==409600
        assert first+second==expected['launches'] and contacts==expected['contact_world_intervals']
        if arm=='single_ball':
            assert first>0 and second>0 and len(first_coverage)==32
        else:
            assert first==second==contacts==0
        review[arm] = dict(updates=100,transitions=409600,first_throws=first,second_throws=second,
                           first_pose_onset_pairs=len(first_coverage),second_pose_onset_pairs=len(second_coverage),
                           contact_world_intervals=contacts,training_seconds=expected['training_seconds'])
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'report.json',dict(verified=True,arms=review,training_updates=200,transitions=819200,
         rehearsal_presentations=sum(r['examples'] for r in report['rehearsal'].values()),
         physical_evaluation_cases=0,training_seal_sha256=digest(args.run/'final-verification.json'),
         reviewer_sha256=digest(Path(__file__)),no_intermediate_checkpoints=True))
    print(review,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
