"""Verify saved learning-curve evidence without physics or network inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .audit_stand_failures import digest, read, save
from .measure_stand_learning_curve import BACKENDS, PRIOR, SEED, continuation, tree_hash
from .replicate_stand_representations import check_protected, summarize_trace, verify_control_trace
from .train_stand_imitation import tensor_hash


def verify_gates(trace, prefix):
    values = {name: trace[prefix+name] for name in ('pelvis_height_m', 'pelvis_up',
        'horizontal_speed_m_s', 'angular_speed_rad_s', 'foot_loads_n',
        'non_foot_ground_contacts_n', 'finite_state')}
    balance = dict(pelvis_height=values['pelvis_height_m'] >= .75,
        pelvis_upright=values['pelvis_up'] >= np.float32(np.cos(np.deg2rad(15.))),
        horizontal_speed=values['horizontal_speed_m_s'] <= .2,
        angular_speed=values['angular_speed_rad_s'] <= 1.,
        two_foot_support=values['foot_loads_n'].min(-1) >= 5.,
        no_non_foot_ground_contact=values['non_foot_ground_contacts_n'].max(-1) <= 1.,
        finite_state=values['finite_state'])
    posture = {}
    for index, (body, tilt, clearance) in enumerate(zip(('spine', 'chest', 'head'),
            (20., 20., 30.), (.05, .06, .10))):
        posture[f'{body}_tilt'] = trace[prefix+'upper_body_tilt_degrees'][..., index] <= tilt
        posture[f'{body}_clearance'] = trace[prefix+'upper_body_clearance_deficit_m'][..., index] <= clearance
    for group, flags in (('balance', balance), ('posture', posture)):
        assert all(np.array_equal(trace[prefix+group+'_gate_'+name], value) for name, value in flags.items())
        assert np.array_equal(trace[prefix+group], np.stack(list(flags.values())).all(0))


def verify(root):
    protocol, complete, report = (read(root/name) for name in
        ('protocol.json', 'completion-verification.json', 'report.json'))
    check_protected({**protocol['inputs'], **complete['output_hashes']}, protocol['sources'])
    prior = read(PRIOR/'protocol.json')
    assert protocol['seed'] == SEED == 301017 and protocol['sampling_seed'] == 301718
    assert protocol['initialize_from'] is None and protocol['maximum_training_seconds'] == 600.
    assert protocol['checkpoint_seconds'] == [180., 360., 600.]
    assert protocol['config'] == prior['config'] == yaml.safe_load((root/'agent.yaml').read_text())
    assert protocol['demonstrations'] == prior['demonstrations']
    for name, key in (('dummy.xml', 'model_sha256'), ('stationary_reference.npz', 'reference_sha256')):
        assert digest(root/name) == protocol[key] == prior[key]
    curve = read(root/'learning-curve.json')
    assert curve == report['assessments'] and curve
    rows = [json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()] if (root/'updates.jsonl').exists() else []
    assert all(row['iteration'] == index+1 and row['transitions'] == 4096*(index+1) for index, row in enumerate(rows))
    assert all(np.isfinite(value) for row in rows for value in row.values())
    assert all(b['training_seconds'] > a['training_seconds'] for a, b in zip(rows, rows[1:]))
    intervals = passes = 0
    stages = []
    for index, stage in enumerate(curve):
        threshold = [0., 180., 360., 600.][index]
        assert stage['requested_training_seconds'] == threshold
        directory = root/('initial' if index == 0 else f'seconds-{int(threshold)}')
        assert read(directory/'report.json') == stage
        assert Path(stage['checkpoint']).resolve() == (directory/'model.pt').resolve()
        portable = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        full = torch.load(directory/'model.amp.pt', map_location='cpu', weights_only=True)
        expected = {key: value for key, value in full.items()
            if not key.startswith(('_model._disc', '_disc_obs_norm'))}
        assert portable.keys() == expected.keys()
        assert all(torch.equal(value, expected[key]) for key, value in portable.items())
        assert tensor_hash(full) == stage['full_tensor_hash']
        assert full['_obs_norm._mean'].numel() == 374 and full['_a_norm._mean'].numel() == 33
        assert all(torch.isfinite(v).all() for v in full.values())
        if index == 0:
            assert stage['training_seconds'] == stage['ppo_updates'] == stage['transitions'] == 0
            assert int(full['_obs_norm._count']) == int(full['_a_norm._count']) == 0
            assert stage['full_tensor_hash'] != read(PRIOR/'report.json')['arms']['target_pd']['initial_full_tensor_hash']
        else:
            count = stage['ppo_updates']
            assert stage['transitions'] == count*4096
            assert rows[count-1]['training_seconds'] == stage['training_seconds'] >= threshold
            assert count == 1 or rows[count-2]['training_seconds'] < threshold
        assert stage['overshoot_seconds'] == stage['training_seconds']-threshold
        guard = read(directory/'isolation.json')
        assert guard['before'] == guard['after'] and guard['before']['learner'] == stage['full_tensor_hash']
        rng = torch.load(directory/'rng.pt', map_location='cpu', weights_only=True)
        assert tree_hash(rng['before']) == tree_hash(rng['restored']) == guard['rng_before'] == guard['rng_after']
        mapping = {'_obs_norm.': 'obs_norm.', '_a_norm.': 'action_norm.',
            '_model._actor_layers.': 'actor.', '_model._action_dist._mean_net.': 'mean.'}
        actor = {dest+key[len(source):]: value for source, dest in mapping.items()
            for key, value in full.items() if key.startswith(source)}
        for backend in BACKENDS:
            result = stage['results'][backend]
            contract = read(directory/backend/'contract.json')
            dt = contract['control_timestep']
            assert contract['num_obs'] == 374 and contract['passive_joint_stiffness'] == 0
            assert contract['standing_pose_servo'] and contract['representation'] == 'target_pd'
            for key in ('model_sha256', 'reference_sha256'):
                assert contract[key] == protocol[key]
            assert result == read(directory/backend/'report.json')
            assert result['actor_hash'] == tensor_hash(actor)
            assert not result['zero_action_baseline'] and not result['external_support'] and not result['inverse_contact_teacher']
            with np.load(directory/backend/'trace.npz', allow_pickle=False) as data:
                trace = {key: data[key] for key in data.files}
            live = trace['alive_before']
            assert live.shape[1] == 8 and len(live) <= 720
            assert np.array_equal(trace['time_s'], (np.arange(len(live))+1)*dt)
            assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
            assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
            assert all(np.isfinite(trace[key][live]).all() for key in
                ('observation', 'action', 'qpos_before', 'qpos_after', 'qvel_before', 'qvel_after'))
            assert not trace['solver_warning'][live].any()
            verify_gates(trace, '')
            verify_gates(trace, 'pre_')
            assert verify_control_trace(trace, contract) == result['max_control_formula_error_nm']
            assert all(result[key] == value for key, value in summarize_trace(trace, dt).items())
            for case in range(8):
                count = int(live[:, case].sum())
                assert live[:count, case].all() and not live[count:, case].any()
                assert (trace['done'][:count-1, case] == 0).all() and trace['done'][count-1, case] in (1, 3)
            assert int(live.sum()) == result['live_control_intervals']
            intervals += int(live.sum())
            passes += result['standing_passes']
        stages.append(dict(requested_training_seconds=threshold, training_seconds=stage['training_seconds'],
            ppo_updates=stage['ppo_updates'], transitions=stage['transitions'],
            results={b: {k: stage['results'][b][k] for k in ('standing_passes', 'minimum_survival_seconds',
                'mean_survival_seconds', 'mean_first_departure_seconds', 'mean_longest_valid_seconds')} for b in BACKENDS}))
    decision = None
    expected_reason = None
    for index, stage in enumerate(curve):
        if all(stage['results'][b]['standing_passes'] == 8 for b in BACKENDS):
            expected_reason = 'initial_16_of_16' if index == 0 else 'stand_16_of_16'
        elif stage['requested_training_seconds'] == 360.:
            decision = continuation(stage, curve[1], dt)
            if not decision['continue_training']:
                expected_reason = 'plateau_or_mixed_at_360'
        elif stage['requested_training_seconds'] == 600.:
            expected_reason = 'maximum_budget_reached'
        if expected_reason:
            assert index == len(curve)-1
            break
    assert expected_reason is not None and report['stop_reason'] == expected_reason
    assert report['continuation_decision'] == decision
    if decision is not None:
        assert read(root/'continuation-decision.json') == decision
    assert report['training_seconds'] == curve[-1]['training_seconds']
    assert report['ppo_updates'] == len(rows) and report['transitions'] == len(rows)*4096
    assert report['final_checkpoint'] == curve[-1]['checkpoint']
    assert report['full_learned_stand'] == (expected_reason == 'stand_16_of_16')
    assert not report['perturbations_ready'] and not report['arena_changed'] and not report['sustained_stand_verified']
    assert report['total_assessment_cases'] == len(curve)*16 <= 64
    save(root/'comparison.json', dict(stages=stages, stop_reason=expected_reason, continuation_decision=decision))
    save(root/'final-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_inputs=len(protocol['inputs']), sources=len(protocol['sources']),
        completion_hashes=len(complete['output_hashes']), hashes_verified=True,
        fresh_initialization_verified=True, same_recipe_verified=True,
        checkpoint_sidecars_verified=True, evaluation_isolation_verified=True,
        gates_and_control_traces_verified=True, budgets_and_stop_decision_verified=True,
        assessment_cases=len(curve)*16, assessment_stand_passes=passes,
        live_assessment_intervals=intervals, new_physics_steps=0, new_forward_passes=0, new_training_updates=0,
        output_hashes={str(p): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p != root/'final-verification.json'}))
    print(dict(verified=True, cases=len(curve)*16, passes=passes, stop_reason=expected_reason), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
