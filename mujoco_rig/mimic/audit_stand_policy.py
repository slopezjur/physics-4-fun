"""Frozen final-policy calibration for the paired Stand replication pilots."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_critic import components, effective_config, read_json, training_summary
from .audit_stand_rewards import source_audit, trial
from .audit_stand_sampling import SamplingNativeEngine, SEEDS, sampled_trial, summarize
from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


def run(args, *, task_factory=None, source_paths=(), reward_variant=None, trace_augmenter=None):
    activate(args.mimickit)
    from util import mp_util
    from .amp_recovery import validate_sidecar
    from .torque_stand_task import AMPTorqueStandTask
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(1)
    protocol = read_json(args.run/'protocol.json')
    initialization = read_json(args.run/'initialization.json')
    training = read_json(args.run/'report.json')
    if not training['protected_inputs_unchanged'] or not initialization['fresh']:
        raise ValueError('Fresh completed controlled training required')
    protected = dict(protocol['inputs'])
    filenames = ('agent.yaml', 'initialization.json', 'initial.pt', 'initial.amp.pt',
        'model.pt', 'model.amp.pt', 'report.json', 'protocol.json', 'runner.py',
        'updates.jsonl', 'assessments.json', 'gpu-samples.jsonl', 'dummy.xml',
        'stationary_reference.npz', 'stationary_reference.json')
    for filename in filenames:
        path = args.run/filename
        protected[str(path)] = sha256(path)
    for filename in ('report.json', 'protocol.json'):
        path = args.paired_audit/filename
        protected[str(path)] = sha256(path)
    for seed in SEEDS:
        path = args.paired_audit/f'trained_actor-{seed}'/'trace.npz'
        protected[str(path)] = sha256(path)
    if sha256(args.run/'runner.py') != protocol['source_sha256']:
        raise ValueError('Saved training source hash mismatch')
    if sha256(args.run/'model.pt') != training['model_sha256']:
        raise ValueError('Final model hash mismatch')
    if any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Protected input hash mismatch')
    initial = torch.load(args.run/'initial.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(args.run/'initial.pt', map_location='cpu', weights_only=True), initial)
    if tensor_hash(initial) != initialization['full_tensor_sha256']:
        raise ValueError('Saved initial tensors differ from fresh initialization record')
    state = torch.load(args.run/'model.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(args.run/'model.pt', map_location='cpu', weights_only=True), state)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(args.run/'dummy.xml', policy_exclude=())
    reference = args.run/'stationary_reference.npz'
    if np.any(rig.model.jnt_stiffness) or np.any(rig.model.actuator_biasprm):
        raise ValueError('Spring-free torque plant required')
    original = Path('logs/mimickit-training/stand-teacher-20261001-01')
    if sha256(rig.path) != sha256(original/'dummy.xml') or sha256(reference) != sha256(original/'stationary_reference.npz'):
        raise ValueError('Physical plant/reference differ from admitted teacher')
    config = effective_config(args.run/'agent.yaml')
    factory = task_factory or AMPTorqueStandTask
    task = factory(rig, reference, 8, 'cpu', seed=protocol['seed'],
        reward_mode='reference_v9', episode_seconds=12., engine_factory=SamplingNativeEngine,
        demonstrations=[Path(record['path']) for record in protocol['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    scorer = build_agent(config, task, 'cpu')
    scorer.load_state_dict(state)
    scorer.eval().requires_grad_(False)
    actor = StandPolicy(scorer).eval().requires_grad_(False)
    before = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    sources = source_audit(args.out, args.mimickit)['sources']
    extra = [Path(__file__), Path('mujoco_rig/mimic/audit_stand_sampling.py'),
             Path('mujoco_rig/mimic/audit_stand_critic.py'), Path('mujoco_rig/mimic/rig.py'),
             Path('mujoco_rig/mimic/native_engine.py'),
             args.mimickit/'mimickit/learning/distribution_gaussian_diag.py']
    extra.extend(source_paths)
    for path in extra:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=sha256(path)))
    write_json(args.out/'protocol.json', dict(run=str(args.run), paired_audit=str(args.paired_audit),
        inputs=protected, sources=sources, config=config, reward_variant=reward_variant,
        training_seed=protocol['seed'],
        inference_seeds=list(SEEDS), seconds=12., native_cases=8,
        initial_tensor_sha256=tensor_hash(initial), frozen_tensor_hashes=before,
        scoring='Own final frozen critic/AMP snapshot; task component separately comparable across policies.',
        training_updates=0, arena_changed=False))
    out = args.out/'deterministic-native'
    out.mkdir()
    deterministic = trial(task, scorer, actor, 'trained_actor', 12., out)
    if trace_augmenter is not None:
        trace_augmenter(task, deterministic, out)
    with np.load(out/'trace.npz') as trace:
        reset = {key: trace[key] for key in ('initial_qpos', 'initial_qvel')}
    reports = []
    for seed in SEEDS:
        out = args.out/f'trained_actor-{seed}'
        out.mkdir()
        report, arrays = sampled_trial(task, scorer, scorer, 'trained_actor', seed, 12., out)
        if trace_augmenter is not None:
            trace_augmenter(task, report, out)
        with np.load(args.paired_audit/f'trained_actor-{seed}'/'trace.npz') as paired:
            frames = min(len(arrays['done']), len(paired['done']))
            if not np.array_equal(arrays['standard_normal_draw'][:frames], paired['standard_normal_draw'][:frames]):
                raise ValueError('Paired Gaussian draw prefixes differ')
            if any(not np.array_equal(reset[key], arrays[key]) or not np.array_equal(reset[key], paired[key]) for key in reset):
                raise ValueError('Paired native reset states differ')
        reports.append(report)
        write_json(args.out/'progress.json', dict(completed_sampled_trials=len(reports), total_sampled_trials=3))
    after = dict(scorer=tensor_hash(scorer.state_dict()), actor=tensor_hash(actor.state_dict()))
    if after != before or any(sha256(Path(path)) != digest for path, digest in protected.items()):
        raise RuntimeError('Frozen tensors or protected input changed')
    if scorer._obs_norm._pending_count or scorer._disc_obs_norm._pending_count:
        raise RuntimeError('Normalizer received audit samples')
    if any(sha256(Path(source['path'])) != source['sha256'] or
           sha256(Path(source['snapshot'])) != source['sha256'] for source in sources):
        raise RuntimeError('Audit source snapshots changed')
    summary = summarize(reports)
    summary['discounted_components'] = components(reports)
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        training=training_summary(args.run), deterministic=deterministic, sampled_trials=reports,
        sampled_summary=summary, training_seed=protocol['seed'],
        initial_checkpoint_verified=True, initial_final_sidecars_valid=True,
        paired_gaussian_draws_identical=True, reset_states_identical=True,
        frozen_tensor_hashes_before=before, frozen_tensor_hashes_after=after,
        frozen_tensors_unchanged=True, normalizer_pending_samples=0,
        inputs_unchanged=True, source_snapshots_verified=True, training_updates=0, arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--paired-audit', type=Path, default=Path('logs/mimickit-training/stand-sampling-audit-20261002-01'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
