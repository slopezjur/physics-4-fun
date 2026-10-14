"""Saved-result checks for the matched-update actor-rate intervention."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .audit_stand_failures import digest, read, save
from .measure_stand_learning_curve import tree_hash
from .replicate_stand_representations import check_protected, summarize_trace, verify_control_trace
from .train_stand_imitation import tensor_hash
from .verify_stand_learning_curve import verify_gates


def verify_pair(root, seed, pair, decision):
    from .compare_stand_actor_rate import compare, REWARD_COLUMNS
    protocol = read(root/'protocol.json')
    intervals = passes = 0
    hashes = []
    for mode, arm in pair.items():
        folder = root/f'seed-{seed}'/mode
        assert read(folder/'report.json') == arm
        assert arm['completed_budget'] and arm['ppo_updates'] == protocol['updates_per_arm'] == 100
        assert arm['transitions'] == 409600 and arm['initialize_from'] is None
        init = read(folder/'initialization.json')
        assert init['seed'] == seed and init['sampling_seed'] == seed+701 and init['initialize_from'] is None
        reward_contract = read(folder/'reward-contract.json')
        assert reward_contract['mode'] == mode
        rows = [json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
        assert len(rows) == 100 and rows[-1]['training_seconds'] == arm['training_seconds']
        assert rows[-2]['training_seconds'] < protocol['maximum_seconds_per_arm']
        assert all(row['iteration'] == i+1 and row['transitions'] == 4096*(i+1) for i, row in enumerate(rows))
        assert all(row['actor_optimizer_steps'] == 48*(i+1) and row['critic_optimizer_steps'] == 64*(i+1)
            for i, row in enumerate(rows))
        assert all(np.isfinite(v) for row in rows for v in row.values())
        with np.load(folder/'reward-records.npz', allow_pickle=False) as data:
            values = data['values']
            assert data['columns'].tolist() == list(REWARD_COLUMNS)
        assert values.shape == (100, 64, 6) and np.isfinite(values).all()
        assert values[..., -1].max() < 1e-6
        assert (values[..., 1] <= values[..., 0]+1e-7).all()
        assert (values[..., 2] >= 0).all() and (values[..., 2] <= 1).all()
        assert np.array_equal(values[..., 0], values[..., 1]) and (values[..., 2] == 1).all()
        config = yaml.safe_load((folder/'agent.yaml').read_text())
        expected_rate = 3e-4 if mode == 'actor_lr_3e4' else 1e-4
        assert config['actor_optimizer']['learning_rate'] == reward_contract['actor_learning_rate'] == expected_rate
        config['actor_optimizer']['learning_rate'] = 1e-4
        assert config == protocol['config']
        assert reward_contract['formula'] == 'reference_v9'
        for row, array in zip(rows, values):
            assert all(row['reward_'+key] == float(value) for key, value in zip(REWARD_COLUMNS, array.mean(0)))
        for stage in ('initial', 'final'):
            directory, assessment = folder/stage, arm[stage]
            assert read(directory/'report.json') == assessment
            assert Path(assessment['checkpoint']).resolve() == (directory/'model.pt').resolve()
            portable = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
            full = torch.load(directory/'model.amp.pt', map_location='cpu', weights_only=True)
            expected = {k: v for k, v in full.items() if not k.startswith(('_model._disc', '_disc_obs_norm'))}
            assert portable.keys() == expected.keys() and all(torch.equal(v, expected[k]) for k, v in portable.items())
            assert tensor_hash(full) == assessment['full_tensor_hash']
            assert all(torch.isfinite(v).all() for v in full.values())
            assert full['_obs_norm._mean'].numel() == 374 and full['_a_norm._mean'].numel() == 33
            if stage == 'initial':
                assert int(full['_obs_norm._count']) == int(full['_a_norm._count']) == 0
                assert assessment['transitions'] == assessment['ppo_updates'] == assessment['training_seconds'] == 0
                hashes.append(assessment['full_tensor_hash'])
            else:
                assert assessment['transitions'] == 409600 and assessment['ppo_updates'] == 100
                assert assessment['training_seconds'] == arm['training_seconds']
            guard = read(directory/'isolation.json')
            assert guard['before'] == guard['after'] and guard['before']['learner'] == assessment['full_tensor_hash']
            rng = torch.load(directory/'rng.pt', weights_only=True, map_location='cpu')
            assert tree_hash(rng['before']) == tree_hash(rng['restored']) == guard['rng_before'] == guard['rng_after']
            mapping = {'_obs_norm.': 'obs_norm.', '_a_norm.': 'action_norm.',
                '_model._actor_layers.': 'actor.', '_model._action_dist._mean_net.': 'mean.'}
            actor = {dest+k[len(source):]: v for source, dest in mapping.items() for k, v in full.items() if k.startswith(source)}
            for backend in ('native', 'gpu'):
                result = assessment['results'][backend]
                contract = read(directory/backend/'contract.json')
                assert result == read(directory/backend/'report.json')
                assert tensor_hash(actor) == result['actor_hash']
                assert contract['num_obs'] == 374 and contract['passive_joint_stiffness'] == 0
                assert contract['representation'] == 'target_pd' and contract['standing_pose_servo']
                for key in ('model_sha256', 'reference_sha256'):
                    assert contract[key] == protocol[key]
                with np.load(directory/backend/'trace.npz', allow_pickle=False) as data:
                    trace = {k: data[k] for k in data.files}
                live, dt = trace['alive_before'], contract['control_timestep']
                assert live.shape[1] == 8 and len(live) <= 720
                assert np.array_equal(trace['time_s'], (np.arange(len(live))+1)*dt)
                assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
                assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
                assert all(np.isfinite(trace[k][live]).all() for k in ('observation', 'action', 'qpos_after', 'qvel_after'))
                assert not trace['solver_warning'][live].any()
                verify_gates(trace, '')
                verify_gates(trace, 'pre_')
                assert verify_control_trace(trace, contract) == result['max_control_formula_error_nm']
                assert all(result[k] == v for k, v in summarize_trace(trace, dt).items())
                for case in range(8):
                    n = int(live[:, case].sum())
                    assert live[:n, case].all() and not live[n:, case].any()
                    assert (trace['done'][:n-1, case] == 0).all() and trace['done'][n-1, case] in (1, 3)
                intervals += int(live.sum())
                passes += result['standing_passes']
    assert len(set(hashes)) == 1
    assert decision == compare(pair['actor_lr_3e4']['final'], pair['reference']['final'], dt)
    result = dict(seed=seed, verified=True, initial_full_tensor_hash=hashes[0], matched_updates=100,
        cases=64, passes=passes, live_intervals=intervals, candidate_wins=decision['candidate_wins'],
        new_physics_steps=0, new_forward_passes=0, new_training_updates=0)
    return result


def verify(root):
    protocol, report, completion = (read(root/name) for name in ('protocol.json', 'report.json', 'completion-verification.json'))
    check_protected({**protocol['inputs'], **completion['output_hashes']}, protocol['sources'])
    assert protocol['config'] == yaml.safe_load((root/'agent.yaml').read_text())
    assert protocol['seeds'] == [301020, 301021]
    assert protocol['orders'] == [['reference', 'actor_lr_3e4'], ['actor_lr_3e4', 'reference']]
    pairs = [verify_pair(root, int(seed), pair, report['decisions'][seed]) for seed, pair in report['arms'].items()]
    assert len(pairs) == 2 and len({p['initial_full_tensor_hash'] for p in pairs}) == 2
    assert report['replicated_gain'] == all(p['candidate_wins'] for p in pairs)
    assert report['full_learned_stand'] == all(pair['actor_lr_3e4']['final']['results'][b]['standing_passes'] == 8
        for pair in report['arms'].values() for b in ('native', 'gpu'))
    assert not report['arena_changed'] and not report['perturbations_ready']
    save(root/'comparison.json', dict(decisions=report['decisions'], replicated_gain=report['replicated_gain'],
        final={seed: {mode: arm['final']['results'] for mode, arm in pair.items()} for seed, pair in report['arms'].items()},
        training={seed: {mode: {k: arm[k] for k in ('ppo_updates', 'transitions', 'training_seconds')}
            for mode, arm in pair.items()} for seed, pair in report['arms'].items()}))
    save(root/'final-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_inputs=len(protocol['inputs']), sources=len(protocol['sources']), completion_hashes=len(completion['output_hashes']),
        pairs=pairs, all_verified=True, cases=sum(p['cases'] for p in pairs), passes=sum(p['passes'] for p in pairs),
        live_intervals=sum(p['live_intervals'] for p in pairs), new_physics_steps=0, new_forward_passes=0, new_training_updates=0,
        output_hashes={str(p): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p != root/'final-verification.json'}))
    print(dict(verified=True, replicated_gain=report['replicated_gain'], cases=128, passes=sum(p['passes'] for p in pairs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
