"""Saved-result checks for fresh duration replication and conditional extension."""
import json
import copy
from pathlib import Path
import numpy as np
import torch
import yaml
from .audit_stand_failures import read
from .measure_stand_learning_curve import tree_hash, continuation
from .stand_experiment_evidence import verify_rollout
from .train_stand_imitation import tensor_hash


def verify(root):
    protocol, report = read(root/'protocol.json'), read(root/'report.json')
    extending = protocol['mode'] == 'extension'
    seed, targets, cap = (301024, [0, 100, 300, 600], 1500.) if extending else (301023, [0, 100, 300], 900.)
    assert protocol['seed'] == seed and protocol['sampling_seed'] == seed+701
    assert protocol['checkpoint_updates'] == targets and protocol['maximum_training_seconds'] == cap
    prior_root = Path(protocol['prior_root'])
    assert read(root/'report.json')['assessments']['initial']['full_tensor_hash'] != read(prior_root/'report.json')['assessments']['initial']['full_tensor_hash']
    assert protocol['initialize_from'] is None and protocol['config']['actor_optimizer']['learning_rate'] == 3e-4
    expected_config = copy.deepcopy(read(Path('logs/mimickit-training/stand-actor-rate-20261005-01/protocol.json'))['config'])
    expected_config['actor_optimizer']['learning_rate'] = 3e-4
    assert protocol['config'] == expected_config == yaml.safe_load((root/'agent.yaml').read_text())
    rows = [json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    assert len(rows) == report['ppo_updates'] in targets[1:]
    assert report['transitions'] == len(rows)*4096
    assert rows[-1]['training_seconds'] == report['training_seconds']
    assert rows[-2]['training_seconds'] < cap
    assert all(r['iteration'] == i+1 and r['transitions'] == (i+1)*4096 and
        r['actor_optimizer_steps'] == (i+1)*48 and r['critic_optimizer_steps'] == (i+1)*64
        for i, r in enumerate(rows))
    assert all(np.isfinite(v) for row in rows for v in row.values())
    with np.load(root/'reward-records.npz') as data:
        reward = data['values']
    assert reward.shape == (len(rows), 64, 6) and np.isfinite(reward).all()
    assert np.array_equal(reward[..., 0], reward[..., 1]) and (reward[..., 2] == 1).all()
    assert report['assessments'] == read(root/'assessments.json')
    intervals = passes = 0
    for stage, result in report['assessments'].items():
        folder = root/stage
        assert result == read(folder/'report.json')
        count = 0 if stage == 'initial' else int(stage.split('-')[1])
        assert result['ppo_updates'] == count and result['transitions'] == count*4096
        assert result['training_seconds'] == (rows[count-1]['training_seconds'] if count else 0.)
        portable = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
        full = torch.load(folder/'model.amp.pt', map_location='cpu', weights_only=True)
        expected = {k: v for k, v in full.items() if not k.startswith(('_model._disc', '_disc_obs_norm'))}
        assert portable.keys() == expected.keys() and all(torch.equal(v, expected[k]) for k, v in portable.items())
        assert tensor_hash(full) == result['full_tensor_hash']
        if count == 0:
            assert int(full['_obs_norm._count']) == int(full['_a_norm._count']) == 0
        guard = read(folder/'isolation.json')
        assert guard['before'] == guard['after'] and guard['before']['learner'] == result['full_tensor_hash']
        rng = torch.load(folder/'rng.pt', map_location='cpu', weights_only=True)
        assert tree_hash(rng['before']) == tree_hash(rng['restored']) == guard['rng_before'] == guard['rng_after']
        mapping = {'_obs_norm.': 'obs_norm.', '_a_norm.': 'action_norm.',
            '_model._actor_layers.': 'actor.', '_model._action_dist._mean_net.': 'mean.'}
        actor = {dest+k[len(source):]: v for source, dest in mapping.items() for k, v in full.items() if k.startswith(source)}
        for backend in ('native', 'gpu'):
            trace, verified = verify_rollout(folder/backend)
            assert verified == result['results'][backend] and verified['actor_hash'] == tensor_hash(actor)
            intervals += int(trace['alive_before'].sum())
            passes += verified['standing_passes']
    last = list(report['assessments'].values())[-1]
    success = all(last['results'][b]['standing_passes'] == 8 for b in ('native', 'gpu'))
    assert success == report['full_learned_stand']
    stages = report['assessments']
    expected_stages = ['initial']+[f'updates-{n}' for n in targets[1:] if n <= len(rows)]
    assert list(stages) == expected_stages
    duration = continuation(stages['updates-300'], stages['updates-100'], read(root/'contract.json')['control_timestep']) if 'updates-300' in stages else None
    expected_reason = 'stand_16_of_16' if success else ('duration_gain_not_repeated' if extending and len(rows) == 300 else 'maximum_updates_reached')
    assert report['stop_reason'] == expected_reason
    if extending and len(rows) >= 300 and not (success and len(rows) == 300):
        assert (len(rows) == 600) == duration['continue_training']
    if extending and len(rows) == 600:
        assert not all(stages['updates-300']['results'][b]['standing_passes'] == 8 for b in ('native', 'gpu'))
    if len(rows) == 100:
        assert success
    else:
        assert not all(report['assessments']['updates-100']['results'][b]['standing_passes'] == 8 for b in ('native', 'gpu'))
    assert report['final_checkpoint'] == last['checkpoint']
    assert not report['perturbations_ready'] and not report['sustained_stand_verified'] and not report['arena_changed']
    return dict(verified=True, cases=16*len(report['assessments']), assessment_passes=passes,
        live_intervals=intervals, fresh_initialization=True, reward_and_budgets_verified=True,
        saved_gates_and_controls_verified=True, checkpoint_pairs_and_isolation_verified=True,
        additional_verification_physics_steps=0, additional_verification_forward_passes=0, additional_training_updates=0)
