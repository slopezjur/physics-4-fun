"""Frozen native comparison of PPO sampling and deterministic Stand assessment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
import yaml

from .audit_stand_rewards import source_audit, trial
from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent

SEEDS = (301010, 301011, 301012)


class SamplingNativeEngine(DiagnosticNativeEngine):
    """Record the held motor torque without changing integration or controls."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held_torques = []

    def reset_envs(self, ids, qpos, qvel):
        super().reset_envs(ids, qpos, qvel)
        self.held_torques.clear()

    def step(self):
        super().step()
        self.held_torques.append(self.native_ctrl[:, self.rig.actuators].numpy().copy())


class SampledActor:
    """Call upstream PPO sampling and capture its draws without altering global RNG."""
    def __init__(self, agent):
        self.agent = agent
        self.records = []

    @torch.no_grad()
    def __call__(self, observation):
        dist = self.agent._model.eval_actor(self.agent._obs_norm.normalize(observation))
        generator = torch.Generator().set_state(torch.get_rng_state())
        # Clone the draw for provenance; the upstream method still performs sampling.
        noise = torch.normal(torch.zeros_like(dist.mean), torch.ones_like(dist.stddev), generator=generator)
        expected = self.agent._a_norm.unnormalize(dist.mean+dist.stddev*noise)
        action, info = self.agent._decide_action(observation, {})
        if not torch.equal(action, expected) or not (info['rand_action_mask'] == 1.).all():
            raise RuntimeError('Upstream Gaussian sampling/provenance mismatch')
        if not torch.allclose(dist.stddev, torch.full_like(dist.stddev, .1)):
            raise RuntimeError('Expected unchanged fixed sigma 0.1')
        self.records.append(dict(raw_sampled_action=action.numpy().copy(),
            mode_action=self.agent._a_norm.unnormalize(dist.mode).numpy().copy(),
            standard_normal_draw=noise.numpy().copy(), a_logp=info['a_logp'].numpy().copy(),
            rand_action_mask=info['rand_action_mask'].numpy().copy()))
        return action


def sampled_trial(task, scorer, agent, label, seed, seconds, out):
    """Assess one frozen sampled policy with the shared Phase 15 trace contract."""
    from learning.base_agent import AgentMode
    agent.set_mode(AgentMode.TRAIN)
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = SampledActor(agent)
    report = trial(task, scorer, actor, label, seconds, out)
    with np.load(out/'trace.npz') as trace:
        arrays = {key: trace[key] for key in trace.files}
    arrays.update({key: np.stack([record[key] for record in actor.records])
                   for key in actor.records[0]})
    arrays['bounded_command'] = arrays['action'].clip(-1., 1.)
    arrays['held_torque_nm'] = np.stack(task.engine.held_torques)
    expected = np.zeros_like(arrays['held_torque_nm'])
    expected[2:] = arrays['bounded_command'][:-2]*np.asarray(task.rig.action_scale, dtype=np.float32)
    if not np.array_equal(expected, arrays['held_torque_nm']):
        raise RuntimeError('Two-slot held torque mapping mismatch')
    np.savez_compressed(out/'trace.npz', **arrays)
    live = arrays['alive_before']
    applied_delta = (arrays['bounded_command']-arrays['mode_action'])[live]
    report.update(seed=seed, sampling_path='PPOAgent._decide_action/DistributionGaussianDiag.sample',
        critic_scope='Final frozen critic. Trained Gaussian matches its sampled policy; fresh Gaussian is off-policy. Native backend and final AMP snapshot remain diagnostic differences.',
        applied_noise_rms=float(np.sqrt(np.mean(applied_delta**2))),
        raw_command_clipping_fraction=float((np.abs(arrays['raw_sampled_action'][live]) > 1.).mean()),
        random_action_fraction=float(arrays['rand_action_mask'][live].mean()),
        delayed_torque_mapping_verified=True)
    write_json(out/'report.json', report)
    return report, arrays


def summarize(reports):
    cases = [case for report in reports for case in report['cases']]
    returns = np.array([c['discounted_observed_returns']['combined_reward'] for c in cases])
    predictions = np.array([c['initial_critic'] for c in cases])
    per_seed = {str(report['seed']): dict(mean_survival_seconds=report['mean_survival_seconds'],
        mean_observed_return=report['mean_discounted_observed_returns']['combined_reward'],
        mean_td_lambda_residual=report['mean_initial_td_lambda_residual']) for report in reports}
    return dict(assessed_cases=len(cases), standing_passes=sum(c['standing_pass'] for c in cases),
        survivors=sum(c['survived'] for c in cases),
        mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in cases])),
        min_survival_seconds=float(np.min([c['survival_seconds'] for c in cases])),
        max_survival_seconds=float(np.max([c['survival_seconds'] for c in cases])),
        mean_observed_return=float(returns.mean()), mean_initial_critic=float(predictions.mean()),
        mean_observed_return_minus_initial_critic=float((returns-predictions).mean()),
        mean_absolute_observed_return_error=float(np.abs(returns-predictions).mean()),
        mean_td_lambda_residual=float(np.mean([c['initial_td_lambda_residual'] for c in cases])),
        per_seed=per_seed)


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from learning.ppo_agent import PPOAgent
    from .amp_recovery import validate_sidecar
    from .stable_normalizer import install
    from .torque_stand_task import AMPTorqueStandTask
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(1)
    deterministic = json.loads((args.deterministic/'report.json').read_text())
    protocol = json.loads((args.deterministic/'protocol.json').read_text())
    if args.seconds != protocol['seconds'] or not deterministic['frozen_tensors_unchanged']:
        raise ValueError('Deterministic comparison protocol differs')
    protected = dict(protocol['inputs'])
    for label in ('fresh_actor', 'trained_actor'):
        for filename in ('trace.npz', 'report.json'):
            path = args.deterministic/label/filename
            protected[str(path)] = sha256(path)
    for filename in ('report.json', 'protocol.json'):
        path = args.deterministic/filename
        protected[str(path)] = sha256(path)
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Phase 14 protected inputs changed')
    teacher = json.loads((args.dataset/'report.json').read_text())
    corrections = json.loads((args.corrections/'report.json').read_text())
    records = teacher['reports']['native']['clips'] + corrections['reports']['native']['amp_clips']
    if records != protocol['demonstrations'] or not teacher['passed'] or not corrections['passed']:
        raise ValueError('Admitted demonstrations differ')
    config = yaml.safe_load((args.baseline/'agent.yaml').read_text())
    if config != protocol['config'] or float(config['model']['action_std']) != .1:
        raise ValueError('Fixed Phase 10 config required')
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(args.dataset/'dummy.xml', policy_exclude=())
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Spring-free torque plant required')
    task = AMPTorqueStandTask(rig, args.dataset/'stationary_reference.npz', 8, 'cpu', seed=301010,
        reward_mode='reference_v9', episode_seconds=args.seconds, engine_factory=SamplingNativeEngine,
        demonstrations=[Path(record['path']) for record in records], demonstration_rig=rig)
    task.mirror_demos = False
    scorer = build_agent(config, task, 'cpu')
    full = torch.load(args.baseline/'model.amp.pt', map_location='cpu', weights_only=True)
    portable = torch.load(args.baseline/'model.pt', map_location='cpu', weights_only=True)
    validate_sidecar(portable, full)
    scorer.load_state_dict(full)
    scorer.eval().requires_grad_(False)
    fresh = PPOAgent(config, task, 'cpu')
    install(fresh)
    initial = torch.load(args.baseline/'initial.pt', map_location='cpu', weights_only=True)
    initial_full = torch.load(args.baseline/'initial.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(initial, initial_full)
    fresh.load_state_dict(initial)
    fresh.eval().requires_grad_(False)
    agents = dict(fresh_actor=fresh, trained_actor=scorer)
    hashes = {key: tensor_hash(agent.state_dict()) for key, agent in agents.items()}
    if hashes['trained_actor'] != protocol['frozen_tensor_hashes']['scorer']:
        raise ValueError('Frozen scorer differs from Phase 14')
    sources = source_audit(args.out, args.mimickit)['sources']
    extra_sources = [Path(__file__), Path('mujoco_rig/mimic/rig.py'), Path('mujoco_rig/mimic/native_engine.py'),
                     args.mimickit/'mimickit/learning/distribution_gaussian_diag.py']
    for path in extra_sources:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=sha256(path)))
    write_json(args.out/'protocol.json', dict(inputs=protected, config=config, demonstrations=records,
        seeds=list(SEEDS), seconds=args.seconds, native_cases=8, action_std=.1,
        frozen_tensor_hashes=hashes, sources=sources, mimickit_revision=MIMICKIT_REVISION,
        sampling='Upstream PPOAgent._decide_action in AgentMode.TRAIN; exploration probability 1; CPU RNG reseeded per policy/seed.',
        resets='Identical eight reference resets; EnvMode.TEST reset assignment; no automatic reset within a trial.',
        pairing='Full eight-world draws at each frame, including inactive worlds. Common frame prefixes pair exactly across policies.',
        scoring='Frozen final Phase 10 critic/discriminator/normalizers for all trials; actor uses its own checkpoint normalizers.',
        scope='Native fixed-policy sampling assessment, not a replay of historical GPU rollout RNG or discriminator states.',
        training_updates=0, arena_changed=False))
    reports, traces = [], {}
    for label, agent in agents.items():
        for seed in SEEDS:
            out = args.out/f'{label}-{seed}'
            out.mkdir()
            report, arrays = sampled_trial(task, scorer, agent, label, seed, args.seconds, out)
            reports.append(report)
            traces[label, seed] = arrays
            write_json(args.out/'progress.json', dict(completed_trials=len(reports), total_trials=6))
    comparisons = {}
    for label in agents:
        with np.load(args.deterministic/label/'trace.npz') as trace:
            reset = {key: trace[key] for key in ('initial_qpos', 'initial_qvel')}
        if any(not np.array_equal(reset[key], traces[label, seed][key]) for seed in SEEDS for key in reset):
            raise RuntimeError('Sampled and deterministic reset states differ')
        summary = summarize([report for report in reports if report['controller']==label])
        previous = next(report for report in deterministic['controllers'] if report['controller']==label)
        summary['deterministic'] = dict(mean_survival_seconds=previous['mean_survival_seconds'],
            mean_observed_return=previous['mean_discounted_observed_returns']['combined_reward'],
            mean_initial_critic=previous['mean_initial_critic'],
            mean_td_lambda_residual=previous['mean_initial_td_lambda_residual'],
            standing_passes=previous['standing_passes'])
        comparisons[label] = summary
    for seed in SEEDS:
        left, right = traces['fresh_actor', seed], traces['trained_actor', seed]
        frames = min(len(left['done']), len(right['done']))
        if not np.array_equal(left['standard_normal_draw'][:frames], right['standard_normal_draw'][:frames]):
            raise RuntimeError('Paired Gaussian draws differ')
    after = {key: tensor_hash(agent.state_dict()) for key, agent in agents.items()}
    if after != hashes or any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError('Frozen tensors or inputs changed')
    if any(agent._obs_norm._pending_count for agent in agents.values()) or scorer._disc_obs_norm._pending_count:
        raise RuntimeError('Normalizer received samples')
    if any(sha256(Path(s['path'])) != s['sha256'] or sha256(Path(s['snapshot'])) != s['sha256'] for s in sources):
        raise RuntimeError('Source snapshot changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        trials=reports, comparison=comparisons, reset_states_identical=True, paired_gaussian_draws_identical=True,
        frozen_tensor_hashes_before=hashes, frozen_tensor_hashes_after=after,
        frozen_tensors_unchanged=True, normalizer_pending_samples=0, inputs_unchanged=True,
        source_snapshots_verified=True, training_updates=0, arena_changed=False))
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--corrections', type=Path, default=Path('logs/mimickit-training/stand-corrections-20261001-01'))
    parser.add_argument('--baseline', type=Path, default=Path('logs/mimickit-training/stand-ppo-noise-0p1-20261001-01'))
    parser.add_argument('--deterministic', type=Path, default=Path('logs/mimickit-training/stand-reward-audit-20261002-01'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
