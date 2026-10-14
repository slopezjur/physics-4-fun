"""Bounded MimicKit PPO foundation for commanded bilateral support motion."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .baseline import sha256
from .motion_tracker import ACTOR_NAME, MotionTracker, RawNormalizer, SCHEMA, quiet_input, migrate_feedback_checkpoint
from .runtime import activate, MIMICKIT_REVISION


def validate_continuation_contract(previous, current):
    """Permit physically revalidated curriculum data, retaining actor/control meaning."""
    current = json.loads(json.dumps(current))
    data_fields = {'reference_sha256', 'reference_duration', 'motion_reference_hashes'}
    if ({k:v for k,v in previous.items() if k not in data_fields} !=
            {k:v for k,v in current.items() if k not in data_fields}):
        raise ValueError('Tracker continuation policy/control contract mismatch')
    return {k:dict(previous=previous.get(k), current=current.get(k)) for k in data_fields
        if previous.get(k) != current.get(k)}


def native_selection_score(metrics):
    """Compare tracking only after quiet, survival and final support are retained."""
    if not metrics['quiet']['passed'] or metrics['quiet_max_qpos_difference'] != 0.:
        return None
    cases = metrics['motion']['cases'] + metrics['fixed']['cases']
    if any(not c['survived'] or c['final_stable_seconds'] < 2. for c in cases):
        return None
    score = float(np.mean([c['displacement_rmse_m'] / c['displacement_limit_m'] for c in cases]))
    return score if np.isfinite(score) else None


def run(args):
    if args.out.exists() or args.seconds <= 0 or args.episode_seconds <= 0 or args.envs < 32 or args.envs % 8:
        raise ValueError('Require a new output directory and bounded positive training settings')
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from learning.nets import net_builder
    from util import mp_util
    from .balance_observation import task_classes
    from .continuous_stand import evaluate_retention
    from .evaluate_motion_tracker import evaluate
    from .guarded_ppo import GuardedPPO
    from .motion_tracker_task import MotionTrackingTask
    from .native_engine import NativeDummyEngine
    from .newton_adapter import DummyNewtonEngine
    from .policy import StandPolicy
    from .rig import Rig
    from .training_quiet_admission import validate_quiet_initialization

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, 'cuda:0', None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    validate_quiet_initialization(json.loads(args.quiet_report.read_text()), args.checkpoint, args.stationary, rig.path)
    contract = json.loads(args.checkpoint.with_name('contract.json').read_text())
    QuietTask, _ = task_classes(contract)
    quiet = QuietTask(rig, args.stationary, 8, 'cpu', engine_factory=NativeDummyEngine)
    source_config = yaml.safe_load(args.checkpoint.with_name('agent.yaml').read_text())
    source_agent = PPOAgent(source_config, quiet, 'cpu')
    source_agent.load(str(args.checkpoint))
    source = StandPolicy(source_agent).eval()
    options = dict(stationary=args.stationary, motions=args.motion, seed=args.seed,
        root_tracking_sigma_m=args.root_tracking_sigma_m, root_tracking_weight=args.root_tracking_weight,
        root_reward_mode=args.root_reward_mode, root_accuracy_scale_m=args.root_accuracy_scale_m)
    native = MotionTrackingTask(rig, num_envs=8, device='cpu', quiet_fraction=0.,
        engine_factory=NativeDummyEngine, **options)
    fixed = MotionTrackingTask(rig, num_envs=8, device='cpu', quiet_fraction=0.,
        fixed_pose_eval=True, engine_factory=NativeDummyEngine, **options)
    train_task = MotionTrackingTask(rig, num_envs=args.envs, device='cuda:0',
        engine_factory=DummyNewtonEngine, training_seconds=args.episode_seconds,
        fixed_fraction=args.fixed_fraction, **options)
    setattr(net_builder, ACTOR_NAME, SimpleNamespace(build_net=lambda *_: (MotionTracker(source), {})))
    config = copy.deepcopy(source_config)
    config.update(agent_name='PPO', actor_max_kl=.02, actor_backtracks=2, actor_project_anchor=False,
        normalizer_samples=0, steps_per_iter=64, iters_per_output=args.evaluation_interval,
        test_episodes=0, action_bound_weight=.01)
    config['model'].update(actor_net=ACTOR_NAME, critic_net='fc_2layers_256units', action_std=.1)
    config['actor_optimizer'] = dict(type='Adam', learning_rate=3e-4)
    config['critic_optimizer'] = dict(type='Adam', learning_rate=3e-4, grad_clip=1.)
    args.out.mkdir(parents=True)
    timing = dict(training_seconds=0., native_seconds=0.)
    curve = []

    class Complete(Exception):
        pass

    class TrackerPPO(GuardedPPO):
        def _build_normalizers(self):
            super()._build_normalizers()
            self._obs_norm = RawNormalizer()

        def _build_model(self, cfg):
            super()._build_model(cfg)
            readout = self._model._action_dist._mean_net
            with torch.no_grad():
                readout.weight.copy_(torch.eye(30))
                readout.bias.zero_()
            readout.requires_grad_(False)

        def test_model(self, episodes):
            return dict(mean_return=0., mean_ep_len=0., num_eps=0)

        def _decide_action(self, observation, info):
            action, result = super()._decide_action(observation, info)
            quiet_mask = observation[:, -1] == 0
            if quiet_mask.any():
                with torch.no_grad():
                    distribution = self._model.eval_actor(observation)
                    action[quiet_mask] = self._a_norm.unnormalize(distribution.mode)[quiet_mask]
                    result['a_logp'][quiet_mask] = distribution.log_prob(distribution.mode)[quiet_mask]
                    result['rand_action_mask'][quiet_mask] = 0.
            return action, result

        def _train_iter(self):
            if timing['training_seconds'] >= args.seconds or (args.out / 'STOP').exists():
                raise Complete
            torch.cuda.synchronize()
            began = time.monotonic()
            result = super()._train_iter()
            torch.cuda.synchronize()
            timing['training_seconds'] += time.monotonic() - began
            if any(not np.isfinite(float(v)) for v in result.values()) or any(
                    not torch.isfinite(p).all() for p in self.parameters()):
                raise FloatingPointError('Nonfinite tracker update')
            completed = self._iter + 1
            if completed % 8 == 0:
                self.save(str(args.out / f'checkpoint_{completed:06d}.pt'))
            curve.append(dict(iteration=completed, training_seconds=timing['training_seconds'],
                samples=self._update_sample_count(), **{k:float(v) for k,v in result.items()}))
            (args.out / 'learning_curve.json').write_text(json.dumps(curve, indent=2)+'\n')
            if completed % args.evaluation_interval == 0:
                assess(f'iteration-{completed}')
            return result

    agent = TrackerPPO(config, train_task, 'cuda:0')
    curriculum_changes = {}
    migrated = False
    initial_trace = None
    if args.resume:
        previous = json.loads(args.resume.with_name('protocol.json').read_text())
        if previous['source_checkpoint_sha256'] != sha256(args.checkpoint) or previous['schema'] not in (SCHEMA, ACTOR_NAME):
            raise ValueError('Tracker continuation provenance mismatch')
        curriculum_changes = validate_continuation_contract(previous['contract'], train_task.observation_contract())
        saved = torch.load(args.resume, map_location='cuda:0', weights_only=False)
        state, migrated = migrate_feedback_checkpoint(saved, agent.state_dict())
        load_path = args.resume
        if migrated:
            load_path = args.out / 'migrated.pt'
            torch.save(state, load_path)
        agent.load(str(load_path))
        if not curriculum_changes:
            initial_trace = args.resume.with_name('final-motion.npz')
            if not initial_trace.is_file():
                raise ValueError('Same-curriculum continuation requires a retained native trace')
    prior_correction = {k:v.detach().cpu().clone()
        for k,v in agent._model._actor_layers.correction.state_dict().items()}
    protocol = dict(schema=ACTOR_NAME, source_checkpoint_sha256=sha256(args.checkpoint),
        source_quiet_report_sha256=sha256(args.quiet_report), model_sha256=sha256(rig.path),
        contract=train_task.observation_contract(), config=config, seed=args.seed, envs=args.envs,
        evaluation_interval=args.evaluation_interval, seconds=args.seconds,
        training_episode_seconds=args.episode_seconds,
        tracking_reward=dict(root_sigma_m=args.root_tracking_sigma_m, root_weight=args.root_tracking_weight,
            root_mode=args.root_reward_mode, accuracy_scale_m=args.root_accuracy_scale_m,
            foot_sigma_m=.04, foot_weight=.2, deepmimic_weight=.8-args.root_tracking_weight),
        training_curriculum=dict(quiet_fraction=.25, fixed_fraction=args.fixed_fraction,
            moving_fraction=.75-args.fixed_fraction, held_pose='Random reference phase, zero velocity',
            selected_foot=False, actor_clock=False),
        checkpoint_selection='Native initial/final assessment: quiet exact, survive and settle all16 fixed/moving trials, then minimize mean displacement/limit',
        curriculum_changes=curriculum_changes,
        feedback=dict(width=170, nearest_pose_goal_seconds=native.dt,
            root_error_scale_m=.03, orientation_error_scale=1., hinge_error_scale_rad=.25,
            queue_slots=2, queue_invalid_targets='masked', old_correction_frozen=True,
            migrated_from_v1=migrated, initial_trace_sha256=sha256(initial_trace) if initial_trace else None),
        seconds_semantics='Synchronized PPO update time; native checks excluded',
        quiet_fraction=.25, quiet_path='Frozen source; no exploration on quiet worlds',
        mimickit_revision=MIMICKIT_REVISION, resume_sha256=sha256(args.resume) if args.resume else None,
        resume_semantics='Model and normalizer weights restored; optimizer and iteration counters restart',
        scope='Motion tracking foundation; no ball training, recovery objective, Arena export or promotion')
    sources=('motion_tracker.py','motion_tracker_task.py','evaluate_motion_tracker.py','train_motion_tracker.py')
    protocol['source_hashes'] = {n:sha256(Path(__file__).with_name(n)) for n in sources}
    for name in sources:
        (args.out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.out / 'agent.yaml').write_text(yaml.safe_dump(config))
    (args.out / 'contract.json').write_text(json.dumps(train_task.observation_contract(), indent=2)+'\n')
    quiet_baseline = None

    def assess(label):
        nonlocal quiet_baseline
        began = time.monotonic()
        actor = StandPolicy(agent).cpu().eval()
        quiet_metrics, quiet_trace = evaluate_retention(quiet, lambda obs: actor(quiet_input(obs)))
        if quiet_baseline is None:
            quiet_baseline = quiet_trace['qpos_after'].copy()
        quiet_error = float(np.max(abs(quiet_trace['qpos_after'] - quiet_baseline)))
        if not quiet_metrics['passed'] or quiet_error != 0.:
            raise RuntimeError('Immutable quiet path lost native retention')
        motion, trace = evaluate(native, actor)
        fixed_metrics, fixed_trace = evaluate(fixed, actor)
        retained_trace_error = None
        if label == 'initial' and initial_trace:
            with np.load(initial_trace) as retained:
                if set(retained.files) != set(trace) or any(retained[k].shape != trace[k].shape for k in trace):
                    raise RuntimeError('Migrated tracker native trace layout changed')
                retained_trace_error = {k:float(np.max(np.abs(retained[k].astype(float)-trace[k]))) for k in trace}
                if any(retained_trace_error.values()):
                    raise RuntimeError('Migrated tracker changed retained native behavior')
        np.savez_compressed(args.out / f'{label}-motion.npz', **trace)
        np.savez_compressed(args.out / f'{label}-fixed.npz', **fixed_trace)
        metrics = dict(quiet=quiet_metrics, quiet_max_qpos_difference=quiet_error, motion=motion, fixed=fixed_metrics,
            retained_trace_max_difference=retained_trace_error)
        (args.out / f'{label}.json').write_text(json.dumps(metrics, indent=2)+'\n')
        timing['native_seconds'] += time.monotonic() - began
        print(json.dumps(dict(label=label, quiet_passed=True, motion_passed=motion['passed'],
            motion_survived=motion['successes'], fixed_survived=fixed_metrics['successes'],
            selection_score=native_selection_score(metrics))), flush=True)
        return metrics

    started = time.monotonic()
    initial = assess('initial')
    agent.save(str(args.out / 'initial.pt'))
    try:
        agent.train_model(args.envs * 64 * 100000, str(args.out), False, 'txt')
    except Complete:
        pass
    agent.save(str(args.out / 'model.pt'))
    final = assess('final')
    initial_score, final_score = native_selection_score(initial), native_selection_score(final)
    selected = 'final' if final_score is not None and (initial_score is None or final_score < initial_score) else 'initial'
    if initial_score is None and final_score is None:
        selected = None
    if selected:
        (args.out / 'best.pt').write_bytes((args.out / ('model.pt' if selected == 'final' else 'initial.pt')).read_bytes())
    frozen = agent._model._actor_layers.source.cpu().state_dict()
    if any(not torch.equal(v, frozen[k]) for k,v in source.state_dict().items()):
        raise RuntimeError('Frozen source tensors changed')
    correction = agent._model._actor_layers.correction.cpu().state_dict()
    if any(not torch.equal(v, correction[k]) for k,v in prior_correction.items()):
        raise RuntimeError('Frozen prior correction tensors changed')
    report = dict(schema=ACTOR_NAME, initial=initial, final=final, timing=timing,
        total_wall_seconds=time.monotonic()-started, iterations=agent._iter, samples=agent._sample_count,
        checkpoint_sha256=sha256(args.out / 'model.pt'), source_frozen=True, prior_correction_frozen=True, arena_changed=False,
        foundation_passed=final['quiet']['passed'] and final['motion']['passed'] and final['fixed']['passed'],
        selection=dict(selected=selected, initial_score=initial_score, final_score=final_score,
            checkpoint_sha256=sha256(args.out / 'best.pt') if selected else None),
        stepping_gate_achieved=False, unseen_hit_gate_tested=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('iterations','samples','foundation_passed','checkpoint_sha256')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','checkpoint','quiet-report','stationary','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--motion', type=Path, action='append', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--episode-seconds', type=float, default=30.)
    parser.add_argument('--root-tracking-sigma-m', type=float, default=.03)
    parser.add_argument('--root-tracking-weight', type=float, default=.25)
    parser.add_argument('--root-reward-mode', choices=('exponential','inverse_huber'), default='exponential')
    parser.add_argument('--root-accuracy-scale-m', type=float, default=.005)
    parser.add_argument('--fixed-fraction', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=300945)
    parser.add_argument('--envs', type=int, default=2048)
    parser.add_argument('--evaluation-interval', type=int, default=64)
    run(parser.parse_args())
