"""Saved-state verification of independently trained sustained Stand actors."""
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .audit_stand_failures import read
from .measure_stand_learning_curve import tree_hash
from .stand_experiment_evidence import verify_rollout
from .evaluate_stand_sustained import verify_duration
from .train_stand_imitation import tensor_hash


def verify(root):
    protocol, report = read(root/'protocol.json'), read(root/'report.json')
    assert protocol['seed'] in (301025, 301026)
    assert protocol['sampling_seed'] == protocol['seed']+701
    assert protocol['initialize_from'] is None and protocol['checkpoint_updates'] == [0,100,300,600,1200]
    cap=2400.
    assert protocol['maximum_training_seconds'] == cap
    prior=read(Path(protocol['prior_root'])/'protocol.json')
    assert protocol['config'] == prior['config'] == yaml.safe_load((root/'agent.yaml').read_text())
    assert protocol['config']['actor_optimizer']['learning_rate'] == 3e-4
    assert report['assessments']['initial']['full_tensor_hash'] != read(Path(protocol['prior_root'])/'report.json')['assessments']['initial']['full_tensor_hash']
    rows = [json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    assert len(rows) == report['ppo_updates'] in (100, 300, 600, 1200)
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

    expected_stages=['initial']+[f'updates-{n}' for n in (100,300,600,1200) if n <= len(rows)]
    assert list(report['assessments']) == expected_stages
    sustained=report['sustained_assessments']
    for stage, result in sustained.items():
        folder=root/stage
        count=int(stage.split('-')[1])
        assert count in (100,300,600,1200) and count <= len(rows)
        assert result == read(folder/'report.json')
        guard=read(folder/'isolation.json')
        assert guard['before'] == guard['after']
        assert guard['before']['learner'] == report['assessments'][f'updates-{count}']['full_tensor_hash']
        rng=torch.load(folder/'rng.pt', map_location='cpu', weights_only=True)
        assert tree_hash(rng['before']) == tree_hash(rng['restored']) == guard['rng_before'] == guard['rng_after']
        for backend in ('native','gpu'):
            trace, checked=verify_duration(folder/backend,60.)
            assert checked == result['results'][backend]
            assert checked['actor_hash'] == report['assessments'][f'updates-{count}']['results'][backend]['actor_hash']
            intervals+=int(trace['alive_before'].sum())
            passes+=checked['standing_passes']
    for stage, result in report['assessments'].items():
        if stage=='initial': continue
        count=int(stage.split('-')[1])
        required=count>=600 or all(result['results'][b]['standing_passes']==8 for b in ('native','gpu'))
        assert (f'sustained-{count}' in sustained) == required
    successes=[stage for stage,result in sustained.items() if all(result['results'][b]['standing_passes']==8 for b in ('native','gpu'))]
    assert report['sustained_stand_verified'] == bool(successes)
    assert not successes or successes == [f'sustained-{len(rows)}']
    assert report['stop_reason'] == ('sustained_stand_16_of_16' if successes else 'maximum_updates_reached')
    assert successes or len(rows)==1200
    assert report['final_checkpoint'] == report['assessments'][expected_stages[-1]]['checkpoint']
    assert not report['perturbations_ready'] and not report['arena_changed']
    return dict(verified=True,cases=16*(len(report['assessments'])+len(sustained)),
        assessment_passes=passes,live_intervals=intervals,fresh_initialization=True,
        checkpoint_pairs_and_isolation_verified=True,reward_and_budgets_verified=True,
        sustained_stand_verified=bool(successes),additional_training_updates=0,
        additional_verification_physics_steps=0,additional_verification_forward_passes=0)
