"""Retain verified quiet behavior through an auxiliary loss during impact PPO."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .prepare_chest_normalizer_swap import policy_state
from .replicate_stand_representations import tensor_hash
from . import run_chest_stronger_onset_learning as study


def run(args):
    s = read(args.protocol)
    assert s['rehearsal_weight'] in (10, 100) and s['rehearsal_batch_size'] == 256
    assert s['rehearsal_seconds'] == 16 and s['training_speed_m_s'] == 1.6
    datasets, evidence = {}, []
    for seed in s['seeds']:
        item = s['rehearsal_sources'][str(seed)]
        root, folder = Path(item['root']), Path(item['folder'])
        proof = read(root / 'final-verification.json')
        assert proof['verified']
        known = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        for p in (folder / 'trace.npz', folder / 'report.json', folder / 'contract.json'):
            assert known[str(p.resolve())] == digest(p)
            evidence.append(str(p))
        evidence.append(str(root / 'final-verification.json'))
        report = read(folder / 'report.json')
        assert report['standing_passes'] == 8
        full = torch.load(Path(s['checkpoints'][str(seed)]['checkpoint']).with_suffix('.amp.pt'),
                          map_location='cpu', weights_only=True)
        assert tensor_hash(policy_state(full)) == report['actor_hash']
        with np.load(folder / 'trace.npz', allow_pickle=False) as z:
            keep = z['time_s'] <= s['rehearsal_seconds']
            assert z['alive_before'][keep].all()
            assert (z['pre_balance'][keep] & z['pre_posture'][keep] & z['balance'][keep] & z['posture'][keep]).all()
            observations = z['observation'][keep].reshape(-1, 374).copy()
            actions = z['action'][keep].reshape(-1, 33).copy()
        assert len(observations) >= 7600 and len(observations) % 8 == 0
        datasets[seed] = observations, actions
    assert all(p.replace('\\', '/') in [q.replace('\\', '/') for q in s['evidence']] for p in evidence)
    original_prepare, original_train = study.prepare_pair, study.pilot.train_arm
    original_protect, original_save = study.support.protect_used, study.save
    reports = []

    def attach(agent, observations, actions):
        obs = torch.as_tensor(observations, device=agent._device)
        target = torch.as_tensor(actions, device=agent._device)
        original_loss = agent._compute_actor_loss
        counter = dict(calls=0, examples=0)
        initial_hash = tensor_hash(agent.state_dict())

        def mean_action(x):
            distribution = agent._model.eval_actor(agent._obs_norm.normalize(x))
            return agent._a_norm.unnormalize(distribution.mode)

        with torch.no_grad():
            before = (mean_action(obs[:256]).clamp(-1, 1) - target[:256]).abs().max().item()
        assert before <= 2e-6 and initial_hash == tensor_hash(agent.state_dict())
        index = torch.arange(s['rehearsal_batch_size'], device=agent._device)

        def loss(batch):
            result = original_loss(batch)
            # Coprime strides cover all eight poses without consuming learner RNG.
            ids = (index * 31 + counter['calls'] * 257) % len(obs)
            mse = (mean_action(obs[ids]) - target[ids]).square().mean()
            result['ppo_loss_without_rehearsal'] = result['actor_loss'].detach()
            result['rehearsal_mse'] = mse.detach()
            result['actor_loss'] = result['actor_loss'] + s['rehearsal_weight'] * mse
            counter['calls'] += 1
            counter['examples'] += len(ids)
            return result

        agent._compute_actor_loss = loss
        agent._quiet_rehearsal = dict(counter=counter, observations=obs, actions=target,
                                     initial_max_action_error=before, predict=mean_action)

    def prepare(rig, reference, world, source, checkpoint, seed, out):
        pair = original_prepare(rig, reference, world, source, checkpoint, seed, out)
        parent_seed = seed - s['optimization_seed_offset']
        assert parent_seed == int(out.parent.name)
        obs, actions = datasets[parent_seed]
        np.savez_compressed(out / 'rehearsal-data.npz', observation=obs, action=actions)
        for agent, _, _ in pair.values():
            attach(agent, obs, actions)
        save(out / 'rehearsal-contract.json', dict(parent_seed=parent_seed, samples=len(obs),
            source=s['rehearsal_sources'][str(parent_seed)], weight=s['rehearsal_weight'],
            examples_per_actor_optimizer_step=s['rehearsal_batch_size'],
            source_seconds=s['rehearsal_seconds'], extra_environment_steps=0,
            extra_optimizer_steps=0, learner_controls_every_rollout=True))
        return pair

    def train(pair, arm, seed, starts, out, settings):
        result = original_train(pair, arm, seed, starts, out, settings)
        agent = pair[arm][0]
        state = agent._quiet_rehearsal
        assert state['counter']['calls'] == agent._actor_optimizer.get_steps() == 48 * s['updates_per_arm']
        assert state['counter']['examples'] == 256 * state['counter']['calls']
        before = tensor_hash(agent.state_dict())
        with torch.no_grad():
            total, maximum = 0., 0.
            for begin in range(0, len(state['observations']), 256):
                delta = state['predict'](state['observations'][begin:begin+256]) - state['actions'][begin:begin+256]
                total += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()))
        assert before == tensor_hash(agent.state_dict())
        record = dict(seed=seed, arm=out.name, **state['counter'],
            initial_max_action_error=state['initial_max_action_error'],
            final_saved_quiet_action_mse=total / state['actions'].numel(),
            final_saved_quiet_max_action_error=maximum, extra_physics_steps=0,
            extra_optimizer_steps=0, online_teacher_takeover=False)
        save(out / 'rehearsal-report.json', record)
        reports.append(record)
        return result

    def protect(out, paths, protocol, settings):
        return original_protect(out, [*paths, Path(__file__),
            args.mimickit / 'mimickit/learning/ppo_agent.py'], protocol, settings)

    def record(path, value):
        if path == args.out / 'report.json':
            value = dict(value, rehearsal=reports,
                rehearsal_example_presentations=sum(r['examples'] for r in reports),
                extra_physics_steps_for_rehearsal=0, extra_optimizer_steps=0,
                online_teacher_takeover=False)
        original_save(path, value)

    with ExitStack() as stack:
        stack.enter_context(patch.object(study, 'prepare_pair', prepare))
        stack.enter_context(patch.object(study.pilot, 'train_arm', train))
        stack.enter_context(patch.object(study.support, 'protect_used', protect))
        stack.enter_context(patch.object(study, 'save', record))
        study.run(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = p.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
