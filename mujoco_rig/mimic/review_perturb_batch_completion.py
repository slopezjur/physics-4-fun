"""Review sealed repeated/delayed trials and the selected Perturb checkpoints."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


BASE = Path('logs/mimickit-training')


def run(out):
    if out.exists():
        raise FileExistsError(out)
    inputs, manifests = {}, {}

    def checked(path, root):
        value = digest(path)
        if manifests[root].get(str(path.resolve())) != value:
            raise ValueError(f'Output differs from its seal: {path}')
        inputs[str(path)] = value
        return path

    def report(root):
        proof = read(root / 'final-verification.json')
        assert proof['verified'] and proof['inputs_and_sources_unchanged']
        inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
        manifests[root] = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        return read(checked(root / 'report.json', root))

    r15 = report(BASE / 'stand-ball-stage15-learning-curve-20261007-01')
    r16 = report(BASE / 'stand-ball-stage16-seed-replication-20261007-01')
    assert r15['qualified_impact_checkpoints'] == [200]
    assert r16['replicated_recipe_gain']['single_ball']
    assert not r16['replicated_impact_practice_gain']
    directional, sustained, repeated, timing = [], [], [], []
    for actor in ('source', 'candidate'):
        for angle in (0, 30, -30, 90):
            r = report(BASE / f'stand-ball-stage17-clear-{actor}-{angle}-20261007-01')
            assert r['all_conditions_admitted'] and r['evaluation_cases'] == 64
            for row in r['results']:
                assert row['parity']['native_launch_clear']
                directional.append(dict(actor=actor, angle=angle, seed=row['seed'], sign=row['sign'],
                    recoveries={b: a['recovery_passes'] for b, a in row['assessments'].items()}))
        r = report(BASE / f'stand-ball-stage18-sustained-{actor}-20261007-01')
        assert r['all_conditions_admitted'] and r['evaluation_cases'] == 64
        for row in r['results']:
            prior = r15['results']['200']['results'][str(row['seed'])]
            prior = prior['initial'] if actor == 'source' else prior['final']['single_ball']
            for backend, after in row['assessments'].items():
                before = prior['impacts'][f"5kg-1.4mps-{row['sign']:+d}"][backend]
                assert before['actor_hash'] == after['actor_hash']
                assert [c['recovery_pass'] for c in before['cases']] == [c['recovery_pass'] for c in after['cases']]
                sustained.append(dict(actor=actor, seed=row['seed'], sign=row['sign'], backend=backend,
                    recoveries=after['recovery_passes'], full_interval_stand=sum(c['standing_pass'] for c in after['cases'])))

        rr = BASE / f'stand-ball-stage19-repeated-{actor}-20261007-01'
        dr = BASE / f'stand-ball-stage20-delayed-{actor}-20261007-01'
        repeat, delayed = report(rr), report(dr)
        assert repeat['all_conditions_admitted'] and repeat['evaluation_cases'] == 192
        assert delayed['all_conditions_admitted'] and delayed['evaluation_cases'] == 128
        for row in repeat['results']:
            folder = rr / f"{row['seed']}-{actor}-{row['condition']['name']}-{row['sign']:+d}-native"
            with np.load(checked(folder / 'body-impulses.npz', rr), allow_pickle=False) as data:
                impulses, names = data['normal'], data['body_names']
            bodies = []
            for case in row['assessments']['native']['cases']:
                encounters = case['encounters']
                for i, event in enumerate(encounters):
                    begin = event['launch_frame']
                    end = encounters[i + 1]['launch_frame'] if i + 1 < len(encounters) else len(impulses)
                    hits = np.flatnonzero(impulses[begin:end, case['case']].sum(-1) > 0) + begin
                    assert bool(len(hits)) == event['hit']
                    if not len(hits):
                        continue
                    cluster = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)[0]
                    first = names[impulses[cluster, case['case']].sum(0) > 0].tolist()
                    assert 'Chest' in first and 'Head' not in first
                    bodies.append(dict(case=case['case'], encounter=i, first_bodies=first))
            assert row['parity']['native_repeat_launch_clear']
            repeated.append(dict(actor=actor, seed=row['seed'], sign=row['sign'], condition=row['condition']['name'],
                recoveries={b: a['recovery_passes'] for b, a in row['assessments'].items()}, native_contacts=bodies))

        for row in delayed['results']:
            onset = row['condition']['launch_seconds']
            name = 'two' if onset == 15 else 'three'
            partner = next(a for a in repeat['results'] if a['seed'] == row['seed']
                and a['sign'] == row['sign'] and a['condition']['name'] == name)
            assert row['parity']['native_launch_clear']
            assert all('Chest' in c['clusters'][0]['bodies'] and 'Head' not in c['clusters'][0]['bodies']
                       for c in row['bodies']['cases'])
            for backend in ('native', 'gpu'):
                a, b = partner['assessments'][backend], row['assessments'][backend]
                assert a['actor_hash'] == b['actor_hash']
                pf = rr / f"{row['seed']}-{actor}-{name}-{row['sign']:+d}-{backend}" / 'trace.npz'
                qf = dr / f"{row['seed']}-{actor}-late{onset}-{row['sign']:+d}-{backend}" / 'trace.npz'
                with np.load(checked(pf, rr), allow_pickle=False) as p, np.load(checked(qf, dr), allow_pickle=False) as q:
                    for key in ('qpos_before', 'qvel_before', 'pending_before'):
                        assert np.array_equal(p[key][0], q[key][0])
                    prefix = {k: float(np.abs(p[k][:300] - q[k][:300]).max())
                              for k in ('qpos_before', 'qvel_before', 'pending_before', 'action')}
                    if backend == 'native':
                        assert max(prefix.values()) <= 1e-10
                rp = [c['case'] for c in a['cases'] if c['recovery_pass']]
                dp = [c['case'] for c in b['cases'] if c['recovery_pass']]
                timing.append(dict(actor=actor, seed=row['seed'], sign=row['sign'], onset=onset, backend=backend,
                    repeated_passes=rp, delayed_passes=dp,
                    final_encounter_passes=[c['case'] for c in a['cases'] if c['encounters'][-1]['passed']],
                    did_not_reach_last_hit=[c['case'] for c in a['cases'] if not c['encounters'][-1]['hit']],
                    lost_with_prior_hits=sorted(set(dp) - set(rp)), gained_with_prior_hits=sorted(set(rp) - set(dp)),
                    same_initial_state=True, native_prefix_verified=backend == 'native', prefix_errors=prefix))

    # The completed pair that triggered Stage 20 must also match Stage 19's final seal.
    decision_path = BASE / 'stand-ball-stage20-decision-20261007-01/decision.json'
    decision = read(decision_path)
    inputs[str(decision_path)] = digest(decision_path)
    for name, expected in decision['inputs'].items():
        p = Path(name)
        assert digest(p) == expected
        checked(p, BASE / 'stand-ball-stage19-repeated-source-20261007-01')

    models = {}
    root = BASE / 'stand-ball-stage14-completion-20261007-01'
    report(root)
    for seed in (301027, 301028):
        p = root / str(seed) / 'single_ball/updates-200/model.pt'
        checked(p, root)
        checked(p.with_suffix('.amp.pt'), root)
        models[str(seed)] = dict(checkpoint=str(p), sha256=digest(p))

    inputs[str(Path(__file__))] = digest(Path(__file__))
    out.mkdir()
    shutil.copy2(__file__, out / 'review-source.py')
    result = dict(completed_stages=20, new_simulation_cases=0, new_training_updates=0,
        selected_checkpoints=models, directional=directional, sustained=sustained,
        repeated=repeated, timing_history=timing, independent_recipe_gain=True,
        independent_impact_practice_gain=False, arena_changed=False,
        scope='Fixed eight starts per parent/backend. Native causal prefixes match exactly; GPU prefix differences are descriptive, not an exact trajectory claim. Repeated trials reuse one ball and do not reproduce Arena persistent multi-ball behavior.')
    save(out / 'report.json', result)
    save(out / 'inputs.json', inputs)
    seal(out, inputs, [], verified=True, completed_stages=20, new_simulation_cases=0, new_training_updates=0)
    print(dict(completed_stages=20, selected_checkpoints=models,
        repeated={actor: sum(sum(r['recoveries'].values()) for r in repeated if r['actor'] == actor)
                  for actor in ('source', 'candidate')},
        delayed={actor: sum(len(r['delayed_passes']) for r in timing if r['actor'] == actor)
                 for actor in ('source', 'candidate')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
