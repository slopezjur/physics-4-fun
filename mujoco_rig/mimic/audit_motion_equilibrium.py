"""Native fixed-pose holds and reward attribution; no optimizer or promotion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def load_tracker(checkpoint, source):
    from .motion_tracker import MotionTracker
    actor = MotionTracker(source).eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    prefix = '_model._actor_layers.'
    state = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
    added = {'queue_mean', 'queue_std', 'pose_error_scale', 'feedback.0.weight',
             'feedback.0.bias', 'feedback.2.weight', 'feedback.2.bias'}
    missing = actor.state_dict().keys() - state.keys()
    if missing not in (set(), added) or state.keys() - actor.state_dict().keys():
        raise ValueError('Unsupported tracker actor checkpoint')
    actor.load_state_dict({**actor.state_dict(), **state}, strict=True)
    # Direct actor evaluation is valid only for an identity physical readout.
    checks = {'_model._action_dist._mean_net.weight': torch.eye(30),
              '_model._action_dist._mean_net.bias': torch.zeros(30),
              '_a_norm._mean': torch.zeros(30), '_a_norm._std': torch.ones(30)}
    if any(not torch.equal(saved[k], v) for k, v in checks.items()):
        raise ValueError('Tracker requires its physical action readout')
    if any(not torch.equal(v, actor.source.state_dict()[k]) for k, v in source.state_dict().items()):
        raise ValueError('Tracker source provenance mismatch')
    return lambda obs: actor(obs).clamp(-1., 1.)


def reward_attribution(task, trace):
    """Use upstream reward with one-hot weights to attribute its five terms."""
    from envs.deepmimic_env import compute_reward
    from util.torch_util import quat_rotate
    count = trace['qpos'].shape[1]
    q = torch.from_numpy(trace['qpos']).flatten(0, 1)
    v = torch.from_numpy(trace['qvel']).flatten(0, 1)
    target = torch.from_numpy(trace['target_qpos']).flatten(0, 1)
    times = task.offset[None] + (torch.arange(len(trace['qpos']))[:, None] + 1) * task.dt
    _, velocity = task.sample_reference(times.T)
    velocity = velocity.transpose(0, 1).flatten(0, 1)
    positions, rotation = task.mapper.from_native_qpos(q)
    desired_pos, desired_rot = task.mapper.from_native_qpos(target)
    args = (q[:, :3], q[:, [4, 5, 6, 3]], v[:, :3],
            quat_rotate(q[:, [4, 5, 6, 3]], v[:, 3:6]), task.joint_rotations(rotation),
            v[:, 6:], positions[:, task.key_bodies], target[:, :3], target[:, [4, 5, 6, 3]],
            velocity[:, :3], quat_rotate(target[:, [4, 5, 6, 3]], velocity[:, 3:6]),
            task.joint_rotations(desired_rot), velocity[:, 6:], desired_pos[:, task.key_bodies],
            task.joint_weights, task.dof_weights, True, True)
    names = ('pose', 'joint_velocity', 'root_pose', 'root_velocity', 'key_position')
    terms = {}
    for i, name in enumerate(names):
        weights = [float(i == j) for j in range(5)]
        terms[name] = compute_reward(*args, *weights, .25, .01, 5., 1., 10.)
    base = compute_reward(*args, .5, .1, .15, .1, .15, .25, .01, 5., 1., 10.)
    reconstructed = sum(w * terms[n] for w, n in zip((.5, .1, .15, .1, .15), names))
    if not torch.allclose(base, reconstructed, atol=1e-7, rtol=0.):
        raise RuntimeError('Reward attribution failed upstream sum parity')
    # Independently compare FK reconstruction with the actual task/backend reward.
    from .native_engine import NativeDummyEngine
    from .target_control import DelayedTargetControl
    probe = NativeDummyEngine(task.rig, count, control_factory=DelayedTargetControl)
    original_engine, original_steps = task.engine, task.steps.clone()
    parity = 0.
    try:
        task.engine = probe
        for frame in (0, len(trace['qpos']) - 1):
            probe.reset_envs(task.ids, torch.from_numpy(trace['qpos'][frame]),
                             torch.from_numpy(trace['qvel'][frame]))
            task.steps.fill_(frame + 1)
            parity = max(parity, float((task.reference_reward() - base.reshape(-1, count)[frame]).abs().max()))
    finally:
        task.engine = original_engine
        task.steps.copy_(original_steps)
    if parity > 1e-5:
        raise RuntimeError(f'Reconstructed reward disagrees with native task: {parity}')
    terms['deepmimic'] = base
    from .motion_tracker_task import root_accuracy_reward
    terms['root_tracking'] = root_accuracy_reward(q[:, :3] - target[:, :3], task.root_reward_mode,
                                                task.root_tracking_sigma_m, task.root_accuracy_scale_m)
    terms['foot_tracking'] = torch.exp(-((positions[:, task.feet] - desired_pos[:, task.feet]) / .04).square().sum((-1, -2)))
    terms['total'] = ((.8 - task.root_tracking_weight) * base
                      + task.root_tracking_weight * terms['root_tracking'] + .2 * terms['foot_tracking'])
    active = torch.from_numpy(trace['active']).bool()
    masks = {'all': active, 'moving_reference': active & (times < task.bank.duration),
             'terminal_reference': active & (times >= task.bank.duration),
             'last_two_seconds': active & (torch.arange(len(active))[:, None] * task.dt >= 28.)}
    result = {'native_task_base_max_difference': parity}
    for label, mask in masks.items():
        result[label] = {'samples': int(mask.sum()), 'means': {
            n: float(values.reshape(-1, count)[mask].mean()) if mask.any() else None
            for n, values in terms.items()}}
    return result


def run(args):
    if args.out.exists():
        raise ValueError('Require a fresh audit output directory')
    activate(args.mimickit)
    from envs.base_env import EnvMode
    from learning.ppo_agent import PPOAgent
    from .balance_observation import task_classes
    from .evaluate_motion_tracker import evaluate
    from .motion_tracker_task import MotionTrackingTask
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .rig import Rig

    torch.set_num_threads(1)
    torch.manual_seed(300950)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    contract = json.loads(args.source.with_name('contract.json').read_text())
    QuietTask, _ = task_classes(contract)
    quiet = QuietTask(rig, args.stationary, 8, 'cpu', engine_factory=NativeDummyEngine)
    agent = PPOAgent(yaml.safe_load(args.source.with_name('agent.yaml').read_text()), quiet, 'cpu')
    agent.load(str(args.source))
    source = StandPolicy(agent).eval()
    options = dict(rig=rig, stationary=args.stationary, motions=[args.motion], num_envs=8,
                   device='cpu', quiet_fraction=0., engine_factory=NativeDummyEngine)
    moving = MotionTrackingTask(**options)
    moving.set_mode(EnvMode.TEST)
    moving.reset()
    args.out.mkdir(parents=True)
    provenance = {'source_sha256': sha256(args.source), 'retained_sha256': sha256(args.retained),
                  'feedback_sha256': sha256(args.feedback), 'rig_sha256': sha256(rig.path),
                  'motion_sha256': sha256(args.motion), 'stationary_sha256': sha256(args.stationary),
                  'script_sha256': sha256(Path(__file__)), 'optimizer_updates': 0,
                  'scope': 'Native diagnostic; no checkpoint modification or Arena promotion'}
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = {'schema': 'motion_equilibrium_audit_v1', 'provenance': provenance,
              'reward_attribution': {}, 'fixed_holds': {}}
    with torch.no_grad():
        for name, checkpoint in (('retained', args.retained), ('feedback', args.feedback)):
            trace_path = checkpoint.with_name('final-motion.npz')
            with np.load(trace_path) as arrays:
                report['reward_attribution'][name] = {
                    'trace_sha256': sha256(trace_path), **reward_attribution(moving, dict(arrays))}
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'reward_attribution': report['reward_attribution']}), flush=True)

        class FixedPoseTask(MotionTrackingTask):
            def sample_reference(self, time, env_ids=None):
                ids = self.ids if env_ids is None else env_ids
                shape = (len(ids),) + (1,) * (time.ndim - 1)
                # Eight phases, including both endpoints; phase is never an actor input.
                phase = ids.float() / 7. * self.bank.references[0].duration
                held = phase.reshape(shape).expand_as(time)
                q, v = self.bank.references[0].sample(held)
                return q, torch.zeros_like(v)

        actors = {'source_moving_goals': lambda obs: source(obs[:, :363]),
                  'retained': load_tracker(args.retained, source),
                  'feedback': load_tracker(args.feedback, source),
                  'zero_residual': lambda obs: torch.zeros((len(obs), 30))}
        for name, actor in actors.items():
            began = time.monotonic()
            task = FixedPoseTask(**options)
            metrics, trace = evaluate(task, actor)
            rows = []
            for i, case in enumerate(metrics['cases']):
                mask = trace['active'][:, i].astype(bool)
                late = mask & (np.arange(len(mask)) * task.dt >= 28.)
                error = trace['qpos'][:, i, :3] - trace['target_qpos'][:, i, :3]
                rows.append({**case, 'held_reference_time': i / 7. * task.bank.duration,
                             'mean_bias_mm': (error[mask].mean(0) * 1000).tolist(),
                             'final_bias_mm': (error[late].mean(0) * 1000).tolist() if late.any() else None})
            trace_path = args.out / f'{name}-holds.npz'
            np.savez_compressed(trace_path, **trace)
            report['fixed_holds'][name] = {'cases': rows, 'survived': metrics['successes'],
                'passed': metrics['passed'], 'native_seconds': time.monotonic() - began,
                'trace_sha256': sha256(trace_path), 'rewards': reward_attribution(task, trace)}
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'actor': name, 'survived': metrics['successes'],
                  'final_forward_bias_mm': [r['final_bias_mm'][0] if r['final_bias_mm'] else None for r in rows]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit', 'source', 'stationary', 'motion', 'retained', 'feedback', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
