"""Bounded stationary-Stand PPO with explicit migration and native case retention."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import BestCheckpoint, validate_contract
from .continuous_stand import CONTRACT, GATES, evaluate_retention, passing_cases
from .critic_calibration import CalibrationProtocol, calibrate
from .evaluation_cache import EvaluationCache, evaluation_key, actor_fingerprint
from .motion_features import install, validate_lineage
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION
from .standing_readiness import (objective, initialize_value_scale, prepare_episodes,
                                 age_counts, require_coverage)


def validate_initialization(source, target, allow_migration):
    if source.get('diagnostic_only') or source.get('normalization_preparation'):
        raise ValueError('Require a trained, admitted source checkpoint')
    if source.get('observation_contract') == CONTRACT:
        validate_contract(source, target)
        return 'same_stationary_contract'
    if not allow_migration or source.get('observation_contract') != 'mimic_stand_target_pd_v2':
        raise ValueError('Finite-clip warm starts require explicit --allow-reference-migration from support_v2')
    if (source.get('observation_layout', [])[:-1] != target['observation_layout'][:-1]
            or source['observation_layout'][-1] != dict(name='reference_phase', offset=361, width=1)):
        raise ValueError('Only the phase meaning may change in the observation layout')
    changed = ('observation_contract', 'observation_layout', 'reference_sha256', 'reference_duration', 'reference_semantics')
    validate_contract(source, {k:v for k,v in target.items() if k not in changed})
    return 'explicit_stationary_reference_and_zero_phase; weights_and_normalization_unchanged'


def calibration_protocol(task, temporal=False):
    # Splitting ids modulo four would entirely withhold postures 3 and 7.
    return CalibrationProtocol(task.ids % 8, tuple(f'posture_{i}' for i in range(8)),
        (task.ids // 8) % 4 == 3, GATES['seconds'],
        dict(schema='deepmimic_stationary_tracking_v1', reference_sha256=task.reference.sha256,
             reward_scale=task.reward_scale if hasattr(task, 'reward_scale') else 1.),
        'replica index (world index // 8) modulo four equals three; all postures in both partitions',
        temporal_validation=temporal)


def run_iterations(agent, iterations, initial_state=None):
    """Keep episodes alive across output/evaluation; upstream's outer loop resets them."""
    agent._curr_obs, agent._curr_info = agent._reset_envs() if initial_state is None else initial_state
    agent._init_train()
    while agent._iter < iterations:
        agent._train_iter()
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1


def train(args):
    if (args.envs < 32 or args.envs % 32 or args.iterations < 1
            or not np.isfinite(args.seconds) or args.seconds <= 0
            or args.evaluation_interval < 8 or args.evaluation_interval % 8
            or args.calibration_steps < 1 or not np.isfinite(args.actor_max_kl) or args.actor_max_kl <= 0):
        raise ValueError('Require worlds in multiples of 32, positive budgets/KL, and evaluation intervals in multiples of eight')
    if args.objective != 'tracking_v1' and not args.temporal_ready:
        raise ValueError('A changed standing objective requires --temporal-ready validation')
    if args.temporal_ready and args.envs < 64:
        raise ValueError('Temporal preparation requires at least 64 worlds')
    activate(args.mimickit)
    from .continuous_stand_task import ContinuousStandTask
    from .guarded_ppo import GuardedPPO
    from util import mp_util
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, 'cuda:0', None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    native = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    contract = native.observation_contract()
    source = json.loads(args.source_contract.read_text())
    source_best = json.loads((args.initialize_from.parent / 'best.json').read_text())
    if source_best['checkpoint_sha256'] != sha256(args.initialize_from):
        raise ValueError('Initialization must match its selected best.json checkpoint hash')
    validate_contract(source, source_best['contract'])
    migration = validate_initialization(source, contract, args.allow_reference_migration)
    config_path = args.initialize_from.parent / 'agent.yaml'
    config = yaml.safe_load(config_path.read_text())
    source_discount = float(config['discount'])
    source_scale = float(config.get('standing_objective', {}).get('reward_scale', 1.))
    standing_objective = (objective(native.dt, args.objective) if args.objective != 'tracking_v1'
                          else dict(schema='tracking_v1', discount=source_discount, reward_scale=1.))
    config['discount'] = standing_objective['discount']
    config['standing_objective'] = standing_objective
    if source.get('normalization_lineage'):
        validate_lineage(source['normalization_lineage'], contract)
        contract['normalization_lineage'] = copy.deepcopy(source['normalization_lineage'])
        if config.get('motion_feature_indices') != contract['normalization_lineage']['motion_feature_indices']:
            raise ValueError('Agent feature architecture does not match source lineage')
    elif config.get('motion_feature_indices'):
        raise ValueError('Expanded features require their versioned normalization lineage')
    config.update(actor_max_kl=args.actor_max_kl, actor_gradient_clip=1., actor_backtracks=0,
                  actor_project_anchor=False, observation_normalization='frozen_loaded_statistics',
                  test_episodes=0, iters_per_output=args.evaluation_interval)
    config['critic_optimizer'] = {**config['critic_optimizer'], 'grad_clip':1.}
    args.out.mkdir(parents=True, exist_ok=False)
    contract['deployment_status'] = 'Experimental stationary training checkpoint; no matching Godot viewer/export adapter yet'
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config))
    (args.out/'contract.json').write_text(json.dumps(contract, indent=2))
    protocol = dict(schema='mimic_stationary_training_v2', task='stationary_stand', contract=contract,
        seed=args.seed, envs=args.envs, seconds=args.seconds, max_iterations=args.iterations,
        evaluation_interval=args.evaluation_interval, calibration_steps=args.calibration_steps,
        gates=GATES, agent_config=config, standing_objective=standing_objective,
        temporal_ready=args.temporal_ready, mimickit_revision=MIMICKIT_REVISION,
        initialization=dict(checkpoint=str(args.initialize_from), checkpoint_sha256=sha256(args.initialize_from),
                            source_contract_sha256=sha256(args.source_contract), config_sha256=sha256(config_path),
                            migration=migration, optimizer='fresh', normalization='frozen; no statistics rewritten'),
        safeguard='Rollout KL only; no fixed teacher anchor. Preserve every incumbent passing native case; stop at first measured regression.',
        rollout_lifecycle='Initial reset, then per-world failure/30-second timeout only. Logging and native evaluation do not reset training worlds.',
        evaluation_scope='Fixed training-posture validation, not held-out generalization; no legacy Godot promotion',
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
                      ('train_continuous_stand.py', 'continuous_stand.py', 'continuous_stand_task.py',
                       'critic_calibration.py', 'checkpoints.py', 'guarded_ppo.py', 'standing_readiness.py')})
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    viewers = Path(__file__).resolve().parents[2] / 'logs/mimickit-viewer'
    viewer_hashes = {str(p):sha256(p) for p in (viewers/'stand.json', viewers/'ball.json') if p.exists()}
    task = ContinuousStandTask(rig, args.reference, args.envs, 'cuda:0', seed=args.seed)
    task.reward_scale = standing_objective['reward_scale']
    cache = EvaluationCache()
    best = BestCheckpoint(args.out, contract)
    curve, updates = [], []
    timing = dict(calibration_seconds=0., native_seconds=0., rollout_seconds=0., optimization_seconds=0.)
    started = time.monotonic()

    def native_evaluation(actor):
        actor = copy.deepcopy(actor).cpu().eval()
        key = evaluation_key(actor, dict(contract=contract, gates=GATES, backend='native'))
        began = time.monotonic()
        metrics, traces = cache.get_or_compute(key, lambda:evaluate_retention(native, actor))
        timing['native_seconds'] += time.monotonic()-began
        return metrics, traces, key

    def assess(agent, iteration, samples):
        metrics, traces, key = native_evaluation(StandPolicy(agent))
        if best.consider(agent.save, metrics, iteration=iteration, samples=samples):
            cache.pin(key)
        row = dict(iteration=iteration, samples=samples, elapsed_seconds=time.monotonic()-started,
                   metrics=metrics, selection=best.decisions[-1])
        curve.append(row)
        (args.out/'learning_curve.json').write_text(json.dumps(curve, indent=2, allow_nan=False))
        print(json.dumps(dict(iteration=iteration, successes=metrics['successes'],
              retained_cases=sorted(passing_cases(metrics)), selection=row['selection'])), flush=True)
        return metrics, traces

    class PilotComplete(Exception):
        pass

    class Pilot(GuardedPPO):
        stop_reason = None
        def _build_model(self, config):
            super()._build_model(config)
            if 'motion_feature_indices' in config:
                install(self, config['motion_feature_indices'])

        def test_model(self, episodes):
            if episodes == 0:
                return dict(mean_return=0., mean_ep_len=0., num_eps=0)
            return super().test_model(episodes)

        def _init_iter(self):
            super()._init_iter()
            self._ages = torch.zeros((8,4), device=task.device, dtype=torch.long)

        def _record_data_pre_step(self, obs, info, action, action_info):
            super()._record_data_pre_step(obs, info, action, action_info)
            self._ages += age_counts(task.get_env_time(), task.ids)

        def _rollout_train(self, steps):
            torch.cuda.synchronize()
            began = time.monotonic()
            result = super()._rollout_train(steps)
            torch.cuda.synchronize()
            timing['rollout_seconds'] += time.monotonic()-began
            return result

        def _update_model(self):
            if args.temporal_ready and self._iter == 0:
                require_coverage(self._ages, passing_cases(initial))
                (args.out/'first_rollout_coverage.json').write_text(json.dumps(dict(
                    counts_by_posture_age=self._ages.cpu().tolist(), passed=True,
                    scope='Actual first PPO batch, before critic or actor updates; preparation samples excluded.'), indent=2))
            torch.cuda.synchronize()
            began = time.monotonic()
            result = super()._update_model()
            torch.cuda.synchronize()
            timing['optimization_seconds'] += time.monotonic()-began
            return result

        def _train_iter(self):
            if self.stop_reason or time.monotonic() >= self.deadline:
                self.stop_reason = self.stop_reason or 'time_budget'
                raise PilotComplete
            result = super()._train_iter()
            if (any(not torch.isfinite(p).all() for p in self.parameters())
                    or any(not np.isfinite(float(v)) for v in result.values())):
                raise FloatingPointError('Nonfinite stationary PPO update')
            completed = self._iter+1
            row = dict(iteration=completed, **{k:float(v) for k,v in result.items()})
            row['samples_age_15_plus'] = int(self._ages[:,2:].sum())
            updates.append(row)
            with (args.out/'updates.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            if completed % 8 == 0:
                print(json.dumps(dict(iteration=completed, actor_kl=row['actor_kl'],
                    accepted=row['actor_steps_accepted'], max_episode_seconds=float(task.get_env_time().max()))), flush=True)
            if completed % args.evaluation_interval == 0:
                self.save(str(args.out/f'evaluation_{completed:06d}.pt'))
                assess(self, completed, self._update_sample_count())
                if best.decisions[-1]['regressions']:
                    self.stop_reason = 'native_retention_regression'
            return result

    agent = Pilot(config, task, 'cuda:0')
    agent.load(str(args.initialize_from))
    if agent._obs_norm.get_count().item() <= 0:
        raise ValueError('Require trained normalization statistics')
    units = initialize_value_scale(agent, source_discount, source_scale, standing_objective)
    (args.out/'value_initialization.json').write_text(json.dumps(units, indent=2))
    agent.save(str(args.out/'initial.pt'))
    initial, _ = assess(agent, 0, 0)
    fingerprint = actor_fingerprint(StandPolicy(agent))
    began = time.monotonic()
    calibration = calibrate(agent, task, args.calibration_steps, args.out,
                            protocol=calibration_protocol(task, args.temporal_ready))
    timing['calibration_seconds'] = time.monotonic()-began
    if not calibration['passed']:
        assert actor_fingerprint(StandPolicy(agent)) == fingerprint
        assert all(sha256(Path(p)) == digest for p,digest in viewer_hashes.items())
        report = dict(schema='mimic_stationary_training_report_v2', stop_reason='critic_calibration_rejected',
            protocol_sha256=sha256(args.out/'protocol.json'), calibration_sha256=sha256(args.out/'critic_calibration.json'),
            initial_native=initial, iterations=0, samples=0, actor_unchanged=True,
            full_gate_passed=False, viewers_unchanged=True, viewer_hashes=viewer_hashes, exported=False,
            scope='Calibration rejected; no preparation or PPO. Initial actor retained.')
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stop_reason=report['stop_reason'], iterations=0)), flush=True)
        return report
    assert actor_fingerprint(StandPolicy(agent)) == fingerprint
    agent.save(str(args.out/'calibrated.pt'))
    state = None
    if args.temporal_ready:
        began = time.monotonic()
        state, preparation = prepare_episodes(agent, task, passing_cases(initial))
        timing['preparation_seconds'] = time.monotonic()-began
        (args.out/'preparation.json').write_text(json.dumps(preparation, indent=2))
    start = time.monotonic()
    agent.deadline = start+args.seconds
    try:
        run_iterations(agent, args.iterations, initial_state=state)
    except PilotComplete:
        pass
    training_seconds = time.monotonic()-start
    agent.save(str(args.out/'model.pt'))
    if not agent._sample_count:
        raise RuntimeError('No stationary PPO iterations completed')
    trained, traces = assess(agent, agent._iter, agent._sample_count)
    np.savez_compressed(args.out/'final-native.npz', **traces)
    if sha256(best.path) != best.metadata['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    agent.load(str(best.path))
    selected, traces, _ = native_evaluation(StandPolicy(agent))
    np.savez_compressed(args.out/'selected-native.npz', **traces)
    gpu_check = ContinuousStandTask(rig, args.reference, 8, 'cuda:0', seed=args.seed)
    selected_gpu, gpu_traces = evaluate_retention(gpu_check, StandPolicy(agent).eval())
    np.savez_compressed(args.out/'selected-gpu.npz', **gpu_traces)
    assert all(sha256(Path(p)) == digest for p,digest in viewer_hashes.items())
    report = dict(schema='mimic_stationary_training_report_v1', protocol_sha256=sha256(args.out/'protocol.json'),
        initial_native=initial, trained_native=trained, selected_native=selected, selected_gpu=selected_gpu,
        iterations=agent._iter, samples=agent._sample_count, training_seconds=training_seconds,
        stop_reason=agent.stop_reason or 'iteration_budget', timing=timing,
        selected_checkpoint=best.metadata, checkpoint_selection=best.decisions,
        calibration_sha256=sha256(args.out/'critic_calibration.json'),
        guard_summary=dict(max_accepted_rollout_kl=max(r['actor_kl'] for r in updates),
            accepted_actor_steps=sum(r['actor_steps_accepted'] for r in updates),
            rejected_actor_steps=sum(r['actor_steps_rejected'] for r in updates)),
        full_gate_passed=selected['passed'] and selected_gpu['passed'],
        evaluation_cache=cache.statistics(), viewer_hashes=viewer_hashes, viewers_unchanged=True,
        exported=False, scope='Stationary contract pilot; saved checkpoints only. No Godot adapter or policy promotion.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(iterations=report['iterations'], stop_reason=report['stop_reason'],
                         selected_cases=sorted(passing_cases(selected)), full_gate_passed=report['full_gate_passed'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--initialize-from', type=Path, required=True)
    parser.add_argument('--source-contract', type=Path, required=True)
    parser.add_argument('--allow-reference-migration', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--evaluation-interval', type=int, default=16)
    parser.add_argument('--calibration-steps', type=int, default=200)
    parser.add_argument('--actor-max-kl', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=210921)
    parser.add_argument('--objective', choices=('tracking_v1', 'retention_30s_v1'), default='tracking_v1')
    parser.add_argument('--temporal-ready', action='store_true', help='Require temporal critic validation and simulated age coverage before PPO')
    train(parser.parse_args())
