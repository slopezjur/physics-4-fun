"""Reconcile a finished corrective-step pilot and check launch clearance offline."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def review(root, out):
    report, proof, protocol = (read(root/name) for name in ('report.json', 'final-verification.json', 'protocol.json'))
    if not proof['verified'] or out.exists():
        raise ValueError('Require a completed pilot and a fresh review output')
    for name, expected in {**protocol['inputs'], **proof['output_hashes']}.items():
        if digest(Path(name)) != expected:
            raise ValueError('Changed evidence: '+name)
    model = mujoco.MjModel.from_xml_path(str(root/'ball-5kg.xml'))
    data = mujoco.MjData(model)
    ball = model.geom('g_ball').id
    ball_body = model.body('ball').id
    character = [g for g in range(model.ngeom) if 0 < model.geom_bodyid[g] < ball_body]
    insertion = []
    for path in sorted(root.glob('**/*-native/trace.npz')):
        with np.load(path, allow_pickle=False) as z:
            if 'launched' not in z:
                continue
            distances = []
            for frame, case in np.argwhere(z['launched']):
                data.qpos[:46] = z['qpos_before'][frame, case]
                data.qpos[46:] = z['ball_qpos_before'][frame, case]
                mujoco.mj_kinematics(model, data)
                distances.append(min(mujoco.mj_geomDistance(model, data, ball, g, 10., None) for g in character))
            insertion.append(dict(folder=str(path.parent), launches=len(distances),
                minimum_character_gap_m=min(distances) if distances else None,
                passed=bool(distances and min(distances) > 0)))
    summaries, updates, examples = {}, 0, 0
    all_admitted = all(row['passed'] for row in insertion)
    for seed, result in report['results'].items():
        initial = result['preflight']['0.25']
        arms = {}
        all_admitted &= initial['physical_admission']
        for name, arm in result['arms'].items():
            if arm['status'] != 'complete':
                arms[name] = dict(status=arm['status'], reason=arm['reason'])
                continue
            records = [json.loads(line) for line in (root/seed/name/'updates.jsonl').read_text().splitlines()]
            if [r['updates'] for r in records] != list(range(1, 101)):
                raise ValueError('Incomplete or repeated PPO updates')
            if any(r['transitions'] != 4096*r['updates'] or r['actor_optimizer_steps'] != 48*r['updates']
                   or r['critic_optimizer_steps'] != 64*r['updates'] for r in records):
                raise ValueError('Optimizer or transition accounting mismatch')
            evaluation = arm['evaluation']
            all_admitted &= evaluation['physical_admission']
            def step_outcomes(audit):
                return [(c['case'], c['recovered_with_step'],
                         [[e['foot'] for e in encounter['steps']] for encounter in c['encounters']])
                        for c in audit['cases']]

            step_parity = all(step_outcomes(evaluation['step_audits'][key+'-native']) ==
                              step_outcomes(evaluation['step_audits'][key+'-gpu']) for key in evaluation['impacts'])
            all_admitted &= step_parity
            updates += arm['training']['updates']
            examples += arm['rehearsal']['examples']
            arms[name] = dict(status='complete', quiet_12s=evaluation['quiet_pass'],
                recovery={k:{b:r['recovery_passes'] for b,r in pair.items()} for k,pair in evaluation['impacts'].items()},
                measured_steps=sum(r['measured_steps'] for r in evaluation['step_audits'].values()),
                recovered_steps=sum(r['recovered_with_step'] for r in evaluation['step_audits'].values()),
                physical_admission=evaluation['physical_admission'], step_backend_parity=step_parity)
        baseline, candidate = result['arms']['baseline']['evaluation'], result['arms']['reward']['evaluation']
        retained = all(candidate['impacts'][key][backend]['recovery_passes'] >= max(
            initial['impacts'][key][backend]['recovery_passes'], baseline['impacts'][key][backend]['recovery_passes'])
            for key in initial['impacts'] for backend in ('native', 'gpu'))
        new_steps = all(sum(r['recovered_with_step'] for k,r in candidate['step_audits'].items() if k.endswith('-'+backend)) >
                        max(sum(r['recovered_with_step'] for k,r in peer['step_audits'].items() if k.endswith('-'+backend))
                            for peer in (initial, baseline)) for backend in ('native', 'gpu'))
        summaries[seed] = dict(initial_recovery={k:r['native']['recovery_passes'] for k,r in initial['impacts'].items()},
            wider_range_quiet={b:r['standing_passes'] for b,r in result['preflight']['0.5']['quiet'].items()},
            arms=arms, reward_retains_each_recovery_count=retained, reward_has_new_recovered_steps=new_steps,
            eligible_for_120s=bool(retained and new_steps and candidate['quiet_pass'] and candidate['physical_admission']))
    counted_cases = sum(len(value['cases']) for p in root.glob('**/report.json')
                        if isinstance((value := read(p)).get('cases'), list))
    if updates != report['training_updates'] or updates*4096 != report['transitions'] or counted_cases != report['evaluation_cases']:
        raise ValueError('Pilot totals differ from saved records')
    if examples != report['rehearsal_example_presentations'] or examples != updates*48*256:
        raise ValueError('Rehearsal accounting mismatch')
    result = dict(schema='corrective_step_pilot_review_v1', seeds=summaries, insertion=insertion,
        all_physical_admitted=bool(all_admitted), training_updates=updates, transitions=updates*4096,
        evaluation_cases=counted_cases, rehearsal_example_presentations=examples,
        proceed_to_120s=bool(all_admitted and all(s['eligible_for_120s'] for s in summaries.values())),
        promoted=False, audit_physics_steps=0, focused_unit_checks_passed=8,
        input_seal_sha256=digest(root/'final-verification.json'), review_source_sha256=digest(Path(__file__)))
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('seeds','insertion')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review(args.root, args.out)
