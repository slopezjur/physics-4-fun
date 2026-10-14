"""Compare the isolated critic-rate pilot and its frozen sampled-policy calibration."""
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
from .audit_stand_sampling import SamplingNativeEngine, SEEDS, sampled_trial, summarize
from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


def read_json(path):
    return json.loads(path.read_text())


def effective_config(path):
    config = yaml.safe_load(path.read_text())
    for name in ('actor_optimizer', 'critic_optimizer', 'disc_optimizer'):
        config[name]['learning_rate'] = float(config[name]['learning_rate'])
    return config


def components(reports):
    cases = [case for report in reports for case in report['cases']]
    return {key: float(np.mean([case['discounted_observed_returns'][key] for case in cases]))
            for key in ('task_reward', 'amp_reward', 'combined_reward')}


def training_summary(path):
    report = read_json(path/'report.json')
    updates = [json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()]
    assessments = read_json(path/'assessments.json')
    at64 = next((entry['metrics'] for entry in assessments if entry['label']=='evaluation-000064-native'), None)
    gpu = [json.loads(line) for line in (path/'gpu-samples.jsonl').read_text().splitlines()]
    config = effective_config(path/'agent.yaml')
    result = dict(run=str(path), critic_learning_rate=config['critic_optimizer']['learning_rate'],
        training_seconds=report['training_seconds'], updates=report['ppo_updates'], transitions=report['transitions'],
        transitions_per_training_second=report['transitions']/report['training_seconds'],
        model_sha256=report['model_sha256'],
        mean_critic_loss=float(np.mean([row['critic_loss'] for row in updates])),
        last16_critic_loss=float(np.mean([row['critic_loss'] for row in updates[-16:]])),
        mean_ppo_clip_fraction=float(np.mean([row['clip_frac'] for row in updates])),
        mean_gpu_percent=float(np.mean([row['gpu_percent'] for row in gpu])) if gpu else None,
        mean_power_w=float(np.mean([row['power_w'] for row in gpu])) if gpu else None)
    for name in ('final_native', 'final_gpu'):
        assessment = report[name]
        result[name] = dict(standing_passes=assessment['standing_passes'], survivors=assessment['survivors'],
            mean_survival_seconds=float(np.mean([case['survival_seconds'] for case in assessment['cases']])),
            min_survival_seconds=float(np.min([case['survival_seconds'] for case in assessment['cases']])),
            max_survival_seconds=float(np.max([case['survival_seconds'] for case in assessment['cases']])))
    result['native_at64'] = None if at64 is None else dict(standing_passes=at64['standing_passes'],
        survivors=at64['survivors'], mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in at64['cases']])))
    return result


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    from .torque_stand_task import AMPTorqueStandTask
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(1)
    baseline_protocol = read_json(args.baseline/'protocol.json')
    candidate_protocol = read_json(args.candidate/'protocol.json')
    baseline_initial = read_json(args.baseline/'initialization.json')
    candidate_initial = read_json(args.candidate/'initialization.json')
    previous_sampling = read_json(args.sampling/'report.json')
    baseline_config = effective_config(args.baseline/'agent.yaml')
    config = effective_config(args.candidate/'agent.yaml')
    if baseline_config['critic_optimizer']['learning_rate'] != 1e-4 or config['critic_optimizer']['learning_rate'] != 1e-3:
        raise ValueError('Expected critic-only 1e-4 versus 1e-3')
    baseline_config['critic_optimizer']['learning_rate'] = 1e-3
    if baseline_config != config or candidate_initial['full_tensor_sha256'] != baseline_initial['full_tensor_sha256']:
        raise ValueError('Non-critic config or fresh initialization mismatch')
    if candidate_protocol['inputs'] != baseline_protocol['inputs'] or candidate_protocol['demonstrations'] != baseline_protocol['demonstrations']:
        raise ValueError('Controlled physical/training inputs differ')
    protected = dict(candidate_protocol['inputs'])
    for folder in (args.baseline, args.candidate):
        for filename in ('agent.yaml', 'initialization.json', 'initial.pt', 'initial.amp.pt',
                         'model.pt', 'model.amp.pt', 'report.json', 'protocol.json', 'updates.jsonl',
                         'assessments.json', 'runner.py', 'gpu-samples.jsonl'):
            path = folder/filename
            protected[str(path)] = sha256(path)
        if sha256(folder/'runner.py') != read_json(folder/'protocol.json')['source_sha256']:
            raise ValueError('Saved training runner hash mismatch')
    for filename in ('report.json', 'protocol.json'):
        path = args.sampling/filename
        protected[str(path)] = sha256(path)
    for seed in SEEDS:
        path = args.sampling/f'trained_actor-{seed}'/'trace.npz'
        protected[str(path)] = sha256(path)
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Protected input mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(args.candidate/'dummy.xml', policy_exclude=())
    reference = args.candidate/'stationary_reference.npz'
    if sha256(rig.path) != sha256(args.baseline/'dummy.xml') or sha256(reference) != sha256(args.baseline/'stationary_reference.npz'):
        raise ValueError('Copied plant/reference mismatch')
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Spring-free torque plant required')
    for path in (rig.path, reference, reference.with_suffix('.json')):
        protected[str(path)] = sha256(path)
    task = AMPTorqueStandTask(rig, reference, 8, 'cpu', seed=301010, reward_mode='reference_v9',
        episode_seconds=12., engine_factory=SamplingNativeEngine,
        demonstrations=[Path(record['path']) for record in candidate_protocol['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    scorer = build_agent(config, task, 'cpu')
    state = torch.load(args.candidate/'model.amp.pt', map_location='cpu', weights_only=True)
    portable = torch.load(args.candidate/'model.pt', map_location='cpu', weights_only=True)
    validate_sidecar(portable, state)
    scorer.load_state_dict(state)
    scorer.eval().requires_grad_(False)
    actor = StandPolicy(scorer).eval().requires_grad_(False)
    hashes = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    sources = source_audit(args.out, args.mimickit)['sources']
    for path in (Path(__file__), Path('mujoco_rig/mimic/audit_stand_sampling.py'), Path('mujoco_rig/mimic/rig.py')):
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=sha256(path)))
    write_json(args.out/'protocol.json', dict(inputs=protected, sources=sources, config=config,
        seeds=list(SEEDS), seconds=12., native_cases=8, frozen_tensor_hashes=hashes,
        baseline=str(args.baseline), candidate=str(args.candidate), baseline_sampling=str(args.sampling),
        config_diff='Only critic SGD learning rate, 1e-4 -> 1e-3; all fresh initialization tensors match.',
        scoring='Each run uses its own final frozen critic/AMP snapshot. Task return is reported separately; combined cross-run rewards are not identical objectives.',
        training_updates=0, arena_changed=False))
    out = args.out/'deterministic-native'
    out.mkdir()
    deterministic = trial(task, scorer, actor, 'trained_actor', 12., out)
    with np.load(out/'trace.npz') as trace:
        reset = {key: trace[key] for key in ('initial_qpos', 'initial_qvel')}
    reports = []
    for seed in SEEDS:
        out = args.out/f'trained_actor-{seed}'
        out.mkdir()
        report, arrays = sampled_trial(task, scorer, scorer, 'trained_actor', seed, 12., out)
        with np.load(args.sampling/f'trained_actor-{seed}'/'trace.npz') as previous:
            frames = min(len(arrays['done']), len(previous['done']))
            if not np.array_equal(arrays['standard_normal_draw'][:frames], previous['standard_normal_draw'][:frames]):
                raise ValueError('Cross-run paired Gaussian draws differ')
            if any(not np.array_equal(reset[key], arrays[key]) or not np.array_equal(reset[key], previous[key]) for key in reset):
                raise ValueError('Cross-run reset states differ')
        reports.append(report)
        write_json(args.out/'progress.json', dict(completed_sampled_trials=len(reports), total_sampled_trials=3))
    candidate_summary = summarize(reports)
    candidate_summary['discounted_components'] = components(reports)
    baseline_reports = [r for r in previous_sampling['trials'] if r['controller']=='trained_actor']
    baseline_summary = summarize(baseline_reports)
    baseline_summary['discounted_components'] = components(baseline_reports)
    after = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    if after != hashes or any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError('Frozen tensors or protected inputs changed')
    if scorer._obs_norm._pending_count or scorer._disc_obs_norm._pending_count:
        raise RuntimeError('Normalizer received samples')
    if any(sha256(Path(s['path'])) != s['sha256'] or sha256(Path(s['snapshot'])) != s['sha256'] for s in sources):
        raise RuntimeError('Source snapshot changed')
    comparison = dict(baseline=training_summary(args.baseline), candidate=training_summary(args.candidate),
        sampled_baseline=baseline_summary, sampled_candidate=candidate_summary)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), comparison=comparison,
        deterministic=deterministic, sampled_trials=reports, fresh_initialization_matches=True,
        only_critic_rate_differs=True, paired_gaussian_draws_identical=True, reset_states_identical=True,
        frozen_tensor_hashes_before=hashes, frozen_tensor_hashes_after=after, frozen_tensors_unchanged=True,
        normalizer_pending_samples=0, inputs_unchanged=True, source_snapshots_verified=True,
        training_updates=0, arena_changed=False)
    write_json(args.out/'report.json', report)
    write_json(args.out/'comparison.json', comparison)
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=Path('logs/mimickit-training/stand-ppo-noise-0p1-20261001-01'))
    parser.add_argument('--candidate', type=Path, default=Path('logs/mimickit-training/stand-ppo-critic-lr1em3-20261002-01'))
    parser.add_argument('--sampling', type=Path, default=Path('logs/mimickit-training/stand-sampling-audit-20261002-01'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
