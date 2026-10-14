"""Review sealed repeated encounters, body attribution, and matched late controls."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


def run(args):
    spec = read(args.protocol)
    inputs = {str(args.protocol): digest(args.protocol), str(Path(__file__)): digest(Path(__file__))}
    roots = {}

    def load_root(name):
        root = Path(name)
        proof = read(root / 'final-verification.json')
        assert proof['verified'] and proof['inputs_and_sources_unchanged']
        inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
        known = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}

        def checked(path):
            assert known[str(path.resolve())] == digest(path), path
            inputs[str(path)] = digest(path)
            return path

        report = read(checked(root / 'report.json'))
        assert report['training_updates'] == 0
        return root, report, checked

    records = []
    for label, name in spec['roots'].items():
        control_name = spec.get('geometry_controls', {}).get(label, spec.get('geometry_control'))
        geometry, geometry_report, geometry_checked = load_root(control_name)
        root, report, checked = load_root(name)
        assert [digest(checked(root / n)) for n in ('dummy.xml', 'stationary_reference.npz')] == [
            digest(geometry_checked(geometry / n)) for n in ('dummy.xml', 'stationary_reference.npz')]
        read(checked(root / 'protocol.json'))
        rows = []
        for row in report['results']:
            seed, sign, condition = row['seed'], row['sign'], row['condition']
            repeated = 'launch_times_seconds' in condition
            native_folder = root / f"{seed}-{row['actor']}-{condition['name']}-{sign:+d}-native"
            native = row['assessments']['native']
            dt = read(checked(native_folder / 'contract.json'))['control_timestep']
            with np.load(checked(native_folder / 'trace.npz'), allow_pickle=False) as z:
                trace = {k: z[k] for k in ('qpos_before', 'qvel_before', 'pending_before', 'action',
                                          'ball_qpos_before', 'alive_before')}
            control = next(v for v in geometry_report['results'] if v['seed'] == seed
                and v['sign'] == sign and v['condition']['speed_m_s'] == 1.6)
            assert control['assessments']['native']['actor_hash'] == native['actor_hash']
            folder = geometry / f"{seed}-{control['actor']}-{control['condition']['name']}-{sign:+d}-native"
            frame = int(np.ceil(5 / dt))
            with np.load(geometry_checked(folder / 'trace.npz'), allow_pickle=False) as z:
                assert all(np.array_equal(trace[k][:frame], z[k][:frame])
                           for k in ('qpos_before', 'qvel_before', 'pending_before', 'action'))
            first_clear = native.get('launch_geometry', {}).get('all_clear', False)
            if repeated:
                assert condition['launch_seconds'] == control['condition']['launch_seconds'] == 5
                assert condition['distance_m'] == control['condition']['distance_m'] == .25
                with np.load(geometry_checked(folder / 'trace.npz'), allow_pickle=False) as z:
                    prefix = all(np.array_equal(trace[k][:frame + 1], z[k][:frame + 1])
                                 for k in ('qpos_before', 'qvel_before', 'pending_before', 'action'))
                    position = np.array_equal(trace['ball_qpos_before'][frame], z['ball_qpos_before'][frame])
                first_clear = bool(prefix and position and control['assessments']['native']['launch_geometry']['all_clear'])
                events = read(checked(native_folder / 'repeat-events.json'))['events']
                assert all(e['character_preserved'] and e['observation_error'] <= 2e-6
                           and e['position_error'] <= 2e-6 and e['velocity_error'] <= 2e-6 for e in events)
                assert all(c['distance_m'] >= -1e-9 for e in events for c in e['native_insertion_contacts'])
                checked(native_folder / 'repeat-launches.npz')
            with np.load(checked(native_folder / 'body-impulses.npz'), allow_pickle=False) as z:
                impulses, names = z['normal'], z['body_names']
            bodies, body_admitted = [], True
            for case in native['cases']:
                encounters = case.get('encounters', [dict(index=0, launch_frame=int(np.ceil(condition['launch_seconds'] / dt)), hit=case['hit'])])
                for i, event in enumerate(encounters):
                    begin = event['launch_frame']
                    end = encounters[i + 1]['launch_frame'] if i + 1 < len(encounters) else len(impulses)
                    hits = np.flatnonzero(impulses[begin:end, case['case']].sum(-1) > 0) + begin
                    assert bool(len(hits)) == event['hit']
                    first_bodies = []
                    if len(hits):
                        cluster = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)[0]
                        first_bodies = names[impulses[cluster, case['case']].sum(0) > 0].tolist()
                        body_admitted &= 'Chest' in first_bodies and 'Head' not in first_bodies
                    elif begin < len(trace['alive_before']) and trace['alive_before'][begin, case['case']]:
                        body_admitted = False
                    bodies.append(dict(case=case['case'], encounter=i, bodies=first_bodies,
                                       reached=bool(len(hits))))
            assert native['actor_hash'] == row['assessments']['gpu']['actor_hash']
            for backend, assessment in row['assessments'].items():
                folder = root / f"{seed}-{row['actor']}-{condition['name']}-{sign:+d}-{backend}"
                assert read(checked(folder / 'report.json')) == assessment
                checked(folder / 'trace.npz')
                if repeated:
                    checked(folder / 'repeat-events.json')
                    checked(folder / 'repeat-launches.npz')
            item = dict(label=label, seed=seed, sign=sign, condition=condition,
                first_launch_clear=first_clear, body_admitted=bool(body_admitted), first_bodies=bodies,
                admitted=bool(first_clear and body_admitted and row['parity']['passed']),
                recoveries={b: a['recovery_passes'] for b, a in row['assessments'].items()},
                cases={b: a['cases'] for b, a in row['assessments'].items()})
            rows.append(item)
        roots[label] = dict(cases=report['evaluation_cases'], recoveries=sum(sum(r['recoveries'].values()) for r in rows),
                            admitted=all(r['admitted'] for r in rows))
        records.extend(rows)
    args.out.mkdir(exist_ok=False)
    shutil.copy2(args.protocol, args.out / 'protocol.json')
    shutil.copy2(__file__, args.out / 'review-source.py')
    save(args.out / 'report.json', dict(summaries=roots, records=records,
        all_admitted=all(r['admitted'] for r in records), additional_simulations=0,
        additional_training_updates=0, persistent_multi_ball_arena=False))
    seal(args.out, inputs, [], verified=True, additional_simulations=0, additional_training_updates=0)
    print(roots, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
