"""Verify every saved rollout in the focused +X matched onset experiment."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


def run(args):
    proof = read(args.prior / 'final-verification.json')
    assert proof['verified'] and proof['inputs_and_sources_unchanged']
    known = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
    inputs = {str(args.prior / 'final-verification.json'): digest(args.prior / 'final-verification.json'), str(Path(__file__)): digest(Path(__file__))}

    def checked(p):
        h = digest(p)
        assert known.get(str(p.resolve())) == h, p
        inputs[str(p)] = h
        return p

    cfg = read(checked(args.prior / 'protocol.json'))['settings']
    assert cfg['quiet_fraction'] == 0 and cfg['impact_worlds'] == 64
    assert cfg['training_direction'] == [1, 0, 0]
    report = read(checked(args.prior / 'report.json'))
    records = []
    for seed in cfg['seeds']:
        root = args.prior / str(seed)
        for variant in ('fixed', 'varied'):
            assert read(checked(root / f'{variant}-starts' / 'report.json'))['passed']
        assert read(checked(root / 'varied-starts' / 'cross_schedule-admission.json'))['passed']
        for arm in cfg['arms']:
            schedule = cfg['varied_onsets'] if arm == 'varied' else cfg['fixed_onsets']
            contract = read(checked(root / arm / 'contract.json'))
            assert contract['training_impact_direction'] == [1, 0, 0]
            dt = contract['control_timestep']
            counts = {(case, sign, onset): 0 for case in range(8) for sign in (1,) for onset in schedule}
            n, contacts, consumed = 0, 0, 0
            logs = sorted((root / arm / 'rollouts').glob('*.npz'))
            assert len(logs) == cfg['updates_per_arm']
            for p in logs:
                with np.load(checked(p), allow_pickle=False) as data:
                    assert data['training_direction'].shape == (64, 64, 3)
                    assert (data['training_direction'][..., 0] == 1).all()
                    assert (data['training_direction'][..., 1:] == 0).all()
                    steps, frame, phase = data['steps'], data['scheduled_launch_step'], data['schedule_phase']
                    assert steps.shape == frame.shape == phase.shape == (64, 64)
                    expected = (steps == frame)
                    if arm == 'quiet':
                        expected[:] = False
                    assert np.array_equal(data['launched'], expected)
                    times = np.array([int(np.ceil(t / dt)) for t in schedule])
                    index = ((np.arange(64)[None] // 8) % 2 + phase) % len(schedule)
                    assert np.array_equal(frame, times[index])
                    for tick, world in np.argwhere(data['launched']):
                        counts[(int(world % 8), 1, schedule[index[tick, world]])] += 1
                    n += int(data['launched'].sum())
                    contacts += int(np.count_nonzero(data['normal_impulse'] > 0))
                    consumed += int(steps.size)
            expected = report['results'][str(seed)][arm]
            assert n == expected['launches'] and contacts == expected['contact_world_intervals']
            assert consumed == expected['transitions'] == cfg['updates_per_arm'] * 4096
            complete = all(v > 0 for v in counts.values()) if arm != 'quiet' else n == contacts == 0
            records.append(dict(seed=seed, arm=arm, launches=n, contact_world_intervals=contacts,
                transitions=consumed, declared_pose_sign_onset_coverage=complete,
                counts=[dict(pose=c, sign=d, onset_seconds=t, launches=v) for (c,d,t),v in counts.items()]))
    args.out.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), args.out / 'review-source.py')
    save(args.out / 'report.json', dict(records=records, all_declared_coverage=all(r['declared_pose_sign_onset_coverage'] for r in records),
        additional_training_updates=0, additional_simulations=0))
    seal(args.out, inputs, [], verified=True, additional_simulations=0, additional_training_updates=0)
    print([{k:r[k] for k in ('seed','arm','launches','declared_pose_sign_onset_coverage')} for r in records], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
