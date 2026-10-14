"""Saved-evidence review of target-dose repeated and delayed-only comparisons."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


def run(base, out):
    if out.exists():
        raise FileExistsError(out)
    inputs, manifests, roots = {}, {}, {}

    def checked(path, root):
        value = digest(path)
        if manifests[root].get(str(path.resolve())) != value:
            raise ValueError(f'Output no longer matches its seal: {path}')
        inputs[str(path)] = value
        return path

    for kind in ('repeated', 'delayed'):
        for actor in ('source', 'candidate'):
            root = Path(f'{base}-{kind}-{actor}-20261007-01')
            proof = read(root / 'final-verification.json')
            assert proof['verified'] and proof['inputs_and_sources_unchanged']
            inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
            manifests[root] = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
            report = read(checked(root / 'report.json', root))
            assert report['evaluation_cases'] == 128 and report['training_updates'] == 0
            protocol = read(checked(root / 'protocol.json', root))['settings']
            assert protocol['seeds'] == [301027, 301028] and protocol['signs'] == [1, -1]
            assert all(c['mass_kg'] == 5 and c['speed_m_s'] == 1.4 and c['distance_m'] == .25
                       and c['seconds'] == 32 for c in protocol['conditions'])
            roots[kind, actor] = root, report

    rows, counts = [], []
    for actor in ('source', 'candidate'):
        rr, repeat = roots['repeated', actor]
        dr, delayed = roots['delayed', actor]
        for kind, report in (('repeated', repeat), ('delayed', delayed)):
            for row in report['results']:
                counts.append(dict(kind=kind, actor=actor, seed=row['seed'], sign=row['sign'],
                    condition=row['condition']['name'], admitted=row['parity']['passed'],
                    recoveries={b: a['recovery_passes'] for b, a in row['assessments'].items()},
                    survivors={b: sum(c['survived'] for c in a['cases']) for b, a in row['assessments'].items()}))
        for late in delayed['results']:
            onset, seed, sign = late['condition']['launch_seconds'], late['seed'], late['sign']
            name = {15: 'two', 25: 'three'}[onset]
            rep = next(a for a in repeat['results'] if a['seed'] == seed and a['sign'] == sign
                       and a['condition']['name'] == name)
            native_folder = rr / f'{seed}-{actor}-{name}-{sign:+d}-native'
            first_bodies = []
            with np.load(checked(native_folder / 'body-impulses.npz', rr), allow_pickle=False) as data:
                impulses, names = data['normal'], data['body_names']
            for case in rep['assessments']['native']['cases']:
                events = case['encounters']
                for i, event in enumerate(events):
                    begin = event['launch_frame']
                    end = events[i + 1]['launch_frame'] if i + 1 < len(events) else len(impulses)
                    hits = np.flatnonzero(impulses[begin:end, case['case']].sum(-1) > 0) + begin
                    assert bool(len(hits)) == event['hit']
                    if len(hits):
                        cluster = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)[0]
                        bodies = names[impulses[cluster, case['case']].sum(0) > 0].tolist()
                        first_bodies.append(dict(case=case['case'], encounter=i, bodies=bodies))
            for backend in ('native', 'gpu'):
                a, b = rep['assessments'][backend], late['assessments'][backend]
                pf = rr / f'{seed}-{actor}-{name}-{sign:+d}-{backend}' / 'trace.npz'
                qf = dr / f'{seed}-{actor}-late{onset}-{sign:+d}-{backend}' / 'trace.npz'
                with np.load(checked(pf, rr), allow_pickle=False) as p, np.load(checked(qf, dr), allow_pickle=False) as q:
                    initial = all(np.array_equal(p[k][0], q[k][0])
                                  for k in ('qpos_before', 'qvel_before', 'pending_before'))
                    prefix = {k: float(np.abs(p[k][:300] - q[k][:300]).max())
                              for k in ('qpos_before', 'qvel_before', 'pending_before', 'action')}
                native_prefix = max(prefix.values()) <= 1e-10 if backend == 'native' else None
                rp = [c['case'] for c in a['cases'] if c['recovery_pass']]
                dp = [c['case'] for c in b['cases'] if c['recovery_pass']]
                row = dict(actor=actor, seed=seed, sign=sign, onset=onset, backend=backend,
                    same_actor=a['actor_hash'] == b['actor_hash'], same_initial_state=initial,
                    native_prefix_verified=native_prefix, prefix_errors=prefix,
                    repeated_passes=rp, delayed_passes=dp,
                    final_encounter_passes=[c['case'] for c in a['cases'] if c['encounters'][-1]['passed']],
                    did_not_reach_last_hit=[c['case'] for c in a['cases'] if not c['encounters'][-1]['hit']],
                    lost_with_prior_hits=sorted(set(dp) - set(rp)), gained_with_prior_hits=sorted(set(rp) - set(dp)),
                    repeated_first_bodies=first_bodies,
                    delayed_first_bodies=[dict(case=c['case'], bodies=list(c['clusters'][0]['bodies'])
                        if c['clusters'] else []) for c in late['bodies']['cases']],
                    repeated_cases=a['cases'], delayed_cases=b['cases'],
                    physical_comparison_admitted=rep['parity']['passed'] and late['parity']['passed'])
                row['admitted'] = bool(row['physical_comparison_admitted'] and row['same_actor']
                                       and initial and native_prefix is not False)
                rows.append(row)
    # A causal comparison requires the admitted native prefix; GPU agrees only at the declared outcome level.
    for row in rows:
        native = next(a for a in rows if all(a[k] == row[k] for k in ('actor', 'seed', 'sign', 'onset'))
                      and a['backend'] == 'native')
        row['admitted'] &= native['admitted']
    totals = {f'{kind}_{actor}': dict(cases=128,
        recoveries=sum(sum(r['recoveries'].values()) for r in counts if r['kind'] == kind and r['actor'] == actor),
        all_conditions_admitted=all(r['admitted'] for r in counts if r['kind'] == kind and r['actor'] == actor))
        for kind in ('repeated', 'delayed') for actor in ('source', 'candidate')}
    inputs[str(Path(__file__))] = digest(Path(__file__))
    out.mkdir()
    shutil.copy2(__file__, out / 'review-source.py')
    result = dict(evaluation_cases=512, new_review_simulations=0, training_updates=0,
        totals=totals, conditions=counts, comparisons=rows, all_comparisons_admitted=all(r['admitted'] for r in rows),
        scope='Fixed parents and starts. Full-sequence failure is distinguished from not reaching a later hit. GPU prefixes are descriptive. One projectile is reused; not persistent multi-ball Arena.',
        promoted=False, arena_changed=False)
    save(out / 'report.json', result)
    save(out / 'inputs.json', inputs)
    seal(out, inputs, [], verified=True, evaluation_cases=512, training_updates=0,
         all_comparisons_admitted=result['all_comparisons_admitted'])
    print(dict(totals=totals, all_comparisons_admitted=result['all_comparisons_admitted']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('logs/mimickit-training/stand-ball-followup21'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.out)
