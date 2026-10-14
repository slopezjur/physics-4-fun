"""Paired fresh, fixed-time supervised pilots with frozen shared normalization."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from .audit_stand_failures import digest, longest, read, save
from .audit_stand_student_errors import GROUPS, metrics, normalizers, predictions, subsets
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import evaluate_actor, tensor_hash


TEACHER = Path('logs/mimickit-training/stand-teacher-20261001-01')
CORRECTIONS = Path('logs/mimickit-training/stand-corrections-20261001-01')
MIXTURES = Path('logs/mimickit-training/stand-whole-body-blend-20261002-01')
PRIOR = Path('logs/mimickit-training/stand-student-errors-20261002-01')
CONFIG = Path('logs/mimickit-training/stand-correction-pilot-20261001-01/agent.yaml')
ADDED_GROUPS = ('actor-0-fraction-0.01', 'actor-0.25-fraction-0.01')
SEED = 301010
SECONDS = 180.
BATCH = 512
RATE = 1e-4


def check_files(protected, sources):
    assert all(digest(Path(path)) == value for path, value in protected.items()), 'Protected input changed'
    assert all(digest(Path(s[key])) == s['sha256'] for s in sources for key in ('path', 'snapshot'))


def protect_inputs(out):
    prior, completion = read(PRIOR/'protocol.json'), read(PRIOR/'completion-verification.json')
    protected = dict(prior['inputs'])
    protected.update(completion['output_hashes'])
    protected[str(PRIOR/'completion-verification.json')] = digest(PRIOR/'completion-verification.json')
    paths = [Path(__file__), Path('mujoco_rig/mimic/train_stand_imitation.py')]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    for root, names in ((TEACHER, ('report.json', 'native/dataset.npz', 'gpu/dataset.npz',
                                 'dummy.xml', 'stationary_reference.npz', 'stationary_reference.json')),
                        (CORRECTIONS, ('report.json', 'native/dataset.npz', 'gpu/dataset.npz')),
                        (CONFIG.parent, ('agent.yaml',))):
        for name in names:
            path = root/name
            value = digest(path)
            assert str(path) not in protected or protected[str(path)] == value
            protected[str(path)] = value
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        value = digest(path)
        assert str(path) not in protected or protected[str(path)] == value
        protected[str(path)] = value
    check_files(protected, [])
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources, prior['contract']


def load_pools():
    with np.load(TEACHER/'native/dataset.npz', allow_pickle=False) as data:
        teacher = (data['observation'].reshape(-1, 306), data['action'].reshape(-1, 33))
    with np.load(CORRECTIONS/'native/dataset.npz', allow_pickle=False) as data:
        corrections = (data['observation'], data['action'])
    traces = {}
    for group in GROUPS:
        with np.load(MIXTURES/group/'trace.npz', allow_pickle=False) as data:
            traces[group] = {key: data[key] for key in
                ('obs', 'teacher_only_command', 'alive_before', 'balance', 'posture', 'time_s')}
            if group in ADDED_GROUPS:
                assert data['obs'].shape == (720, 8, 306)
                assert data['alive_before'].all() and data['balance'].all() and data['posture'].all()
                assert np.array_equal(data['obs'][..., 95:161],
                                      data['pending_commands_oldest_first'].reshape(720, 8, 66))
    added = tuple(np.concatenate([traces[g][key].reshape(-1, width) for g in ADDED_GROUPS])
                  for key, width in (('obs', 306), ('teacher_only_command', 33)))
    pools = dict(teacher=teacher, corrections=corrections, added=added)
    assert [len(pool[0]) for pool in pools.values()] == [28800, 18720, 11520]
    for obs, action in pools.values():
        assert np.isfinite(obs).all() and np.isfinite(action).all()
        assert (np.abs(action) <= 1.).all()
    return pools, traces


def actor_normalizer_hash(actor):
    return tensor_hash({key: value for key, value in actor.state_dict().items()
                        if key.startswith(('obs_norm.', 'action_norm.'))})


def train_arm(agent, common_state, pools, early_frames, out, augmented, fixed_names, fixed_hash):
    from .amp_recovery import validate_sidecar
    agent.load_state_dict(common_state)
    initial_hash = tensor_hash(agent.state_dict())
    agent.save(str(out/'initial.pt'))
    actor = StandPolicy(agent).to('cuda:0').train()
    actor.obs_norm.requires_grad_(False)
    actor.action_norm.requires_grad_(False)
    actor_initial_hash = tensor_hash(actor.state_dict())
    normalizer_hash = actor_normalizer_hash(actor)
    counts = normalizers(actor)
    assert counts['observation_count'] == 59040 and not counts['observation_pending']
    assert counts['action_count'] == 0 and not counts['action_pending']
    parameters = list(actor.actor.parameters()) + list(actor.mean.parameters())
    optimizer = torch.optim.Adam(parameters, lr=RATE)
    assert not optimizer.state
    normalized = {name: (actor.obs_norm.normalize(obs).detach(),
                         actor.action_norm.normalize(action).detach())
                  for name, (obs, action) in pools.items()}
    seen = {name: torch.zeros(len(obs), dtype=torch.bool, device='cuda:0')
            for name, (obs, _) in pools.items()}
    rng = torch.Generator(device='cuda:0').manual_seed(SEED + 701)
    # Eight samples per generating actor and pose in each 128-row mixture stratum.
    group_case = torch.arange(16, device='cuda:0').repeat(8)

    def sample_indices():
        old_n = 64 if augmented else 128
        indices = dict(teacher=torch.cat((
            torch.randint(early_frames*8, (old_n,), generator=rng, device='cuda:0'),
            torch.randint(len(normalized['teacher'][0]), (old_n,), generator=rng, device='cuda:0'))),
            corrections=torch.randint(len(normalized['corrections'][0]),
                (128 if augmented else 256,), generator=rng, device='cuda:0'))
        if augmented:
            frames = torch.cat((torch.randint(early_frames, (128,), generator=rng, device='cuda:0'),
                                torch.randint(720, (128,), generator=rng, device='cuda:0')))
            pose = group_case.repeat(2)
            indices['added'] = (pose//8)*5760 + frames*8 + pose%8
        return indices

    save(out/'initialization.json', dict(seed=SEED, initial_full_tensor_hash=initial_hash,
        initial_actor_hash=actor_initial_hash, common_normalizer_hash=normalizer_hash,
        normalizer_counts=counts, independent_adam_state_empty=True, initialize_from=None,
        sampling_rng_seed=SEED+701))
    torch.cuda.synchronize()
    started = time.perf_counter()
    updates, next_output, curve = 0, 15., []
    while time.perf_counter() - started < SECONDS:
        indices = sample_indices()
        batch_x = torch.cat([normalized[name][0][picks] for name, picks in indices.items()])
        batch_y = torch.cat([normalized[name][1][picks] for name, picks in indices.items()])
        assert len(batch_x) == BATCH
        prediction = actor.mean(actor.actor(batch_x))
        loss = torch.nn.functional.mse_loss(prediction, batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite supervised loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        for name, picks in indices.items():
            seen[name][picks] = True
        updates += 1
        elapsed = time.perf_counter() - started
        if elapsed >= next_output:
            row = dict(seconds=elapsed, optimizer_updates=updates,
                supervised_sample_presentations=updates*BATCH, minibatch_mse=float(loss.detach()))
            curve.append(row)
            save(out/'progress.json', row)
            print(dict(arm=out.name, **row), flush=True)
            next_output += 15.
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    torch.save(optimizer.state_dict(), out/'imitation-optimizer.pt')
    actor.cpu().eval()
    assert actor_normalizer_hash(actor) == normalizer_hash and normalizers(actor) == counts
    agent._model._actor_layers.load_state_dict(actor.actor.state_dict())
    agent._model._action_dist._mean_net.load_state_dict(actor.mean.state_dict())
    assert tensor_hash({key: agent.state_dict()[key] for key in fixed_names}) == fixed_hash
    agent.save(str(out/'model.pt'))
    validate_sidecar(torch.load(out/'model.pt', weights_only=True),
                     torch.load(out/'model.amp.pt', weights_only=True))
    per_update = dict(teacher_early=64 if augmented else 128, teacher_uniform=64 if augmented else 128,
                     corrections=128 if augmented else 256, added_early=128 if augmented else 0,
                     added_uniform=128 if augmented else 0)
    report = dict(requested_training_seconds=SECONDS, training_seconds=training_seconds,
        optimizer_updates=updates, supervised_sample_presentations=updates*BATCH,
        sample_presentations_by_stratum={key: n*updates for key, n in per_update.items()},
        unique_available_labelled_pairs=59040 if augmented else 47520,
        unique_presented_pairs={name: int(mask.sum()) for name, mask in seen.items()},
        normalization_training_observations=59040, initial_full_tensor_hash=initial_hash,
        initial_actor_hash=actor_initial_hash, common_normalizer_hash=normalizer_hash,
        normalizer_counts=counts, final_actor_hash=tensor_hash(actor.state_dict()),
        model_sha256=digest(out/'model.pt'), critic_and_discriminator_unchanged=True,
        portable_sidecar_match=True, ppo_updates=0, discriminator_updates=0,
        repeated_supervised_samples_are_physics_transitions=False)
    save(out/'curve.json', curve)
    save(out/'training-report.json', report)
    return actor, report


def assess(rig, reference, actor, traces, contract, out):
    before, norm_before = tensor_hash(actor.state_dict()), normalizers(actor)
    evaluations = {}
    dt = contract['control_timestep']
    for backend in ('native', 'gpu'):
        destination = out/backend
        destination.mkdir()
        print(dict(arm=out.name, evaluation=backend, autonomous=True), flush=True)
        report = evaluate_actor(rig, reference, actor, backend, 12., destination)
        with np.load(destination/'trace.npz', allow_pickle=False) as trace:
            for row in report['cases']:
                case = row['case']
                valid = trace['alive_before'][:, case] & trace['balance'][:, case] & trace['posture'][:, case]
                row['longest_balanced_valid_posture_seconds'] = longest(valid)*dt
                if backend == 'gpu':
                    row['solver_warning'] = None
            report['live_control_intervals'] = int(trace['alive_before'].sum())
        report['mean_survival_seconds'] = float(np.mean([r['survival_seconds'] for r in report['cases']]))
        report['mean_longest_balanced_valid_posture_seconds'] = float(np.mean([
            r['longest_balanced_valid_posture_seconds'] for r in report['cases']]))
        report['solver_warning_measurement'] = ('MuJoCo BADQPOS/BADQVEL/BADQACC on live worlds'
            if backend == 'native' else 'Unavailable; inherited GPU trace field is an unmeasured false placeholder')
        report['solver_warnings'] = sum(r['solver_warning'] for r in report['cases']) if backend == 'native' else None
        save(destination/'report.json', report)
        evaluations[backend] = report
    inference = {}
    scale = np.asarray(contract['torque_scale_nm'])
    for group, trace in traces.items():
        destination = out/'frozen-inference'/group
        destination.mkdir(parents=True)
        arrays = predictions(actor, trace)
        masks = subsets(trace, dt)
        arrays.update({f'subset_{key}': value for key, value in masks.items()})
        np.savez_compressed(destination/'predictions.npz', **arrays)
        report = dict(trace=group, in_augmented_supervised_pool=group in ADDED_GROUPS,
            interpretation='Recorded-state command regression; augmented groups are fitting measurements',
            subsets={key: metrics(arrays['error_from_live_teacher'], arrays['clipped_features'], mask, scale)
                     for key, mask in masks.items()}, simulation_steps=0, training_updates=0)
        save(destination/'report.json', report)
        inference[group] = report
    assert tensor_hash(actor.state_dict()) == before and normalizers(actor) == norm_before
    return evaluations, inference


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .native_engine import NativeDummyEngine
    from .torque_stand_task import AMPTorqueStandTask
    from .train_torque_stand import build_agent
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(2)
    protected, sources, inherited_contract = protect_inputs(args.out)
    teacher_report, correction_report = read(TEACHER/'report.json'), read(CORRECTIONS/'report.json')
    assert teacher_report['passed'] and correction_report['passed']
    assert correction_report['model_sha256'] == digest(TEACHER/'dummy.xml')
    assert correction_report['reference_sha256'] == digest(TEACHER/'stationary_reference.npz')
    clips = teacher_report['reports']['native']['clips'] + correction_report['reports']['native']['amp_clips']
    for clip in clips:
        path = Path(clip['path'])
        assert digest(path) == clip['sha256']
        protected[str(path)] = clip['sha256']
        protected[str(path.with_suffix('.json'))] = digest(path.with_suffix('.json'))
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(TEACHER/name, args.out/name)
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    task = AMPTorqueStandTask(rig, reference, 8, 'cpu', engine_factory=NativeDummyEngine,
        reward_mode='reference_v9', episode_seconds=60., demonstrations=[Path(c['path']) for c in clips],
        demonstration_rig=rig)
    task.mirror_demos = False
    contract = task.observation_contract()
    for key in ('model_sha256', 'reference_sha256', 'actuator_names', 'torque_scale_nm',
                'observation_layout', 'num_obs', 'latency_steps', 'control_timestep', 'passive_joint_stiffness'):
        assert contract[key] == inherited_contract[key], key
    config = yaml.safe_load(CONFIG.read_text())
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = build_agent(config, task, 'cpu')
    fresh_hash = tensor_hash(agent.state_dict())
    agent.save(str(args.out/'fresh.pt'))
    fixed_names = [key for key in agent.state_dict()
                   if key.startswith(('_model._critic', '_model._disc', '_disc_obs_norm'))]
    fixed_hash = tensor_hash({key: agent.state_dict()[key] for key in fixed_names})
    pools, traces = load_pools()
    training_obs = torch.from_numpy(np.concatenate([obs for obs, _ in pools.values()]))
    assert len(training_obs) == 59040
    agent._obs_norm.record(training_obs)
    agent._obs_norm.update()
    common_state = copy.deepcopy(agent.state_dict())
    agent.save(str(args.out/'common-normalized-initial.pt'))
    gpu_pools = {name: tuple(torch.from_numpy(array).to('cuda:0') for array in pool)
                 for name, pool in pools.items()}
    protocol = dict(inputs=protected, sources=sources, contract=contract, seed=SEED,
        fresh_full_tensor_hash=fresh_hash, common_normalized_full_tensor_hash=tensor_hash(common_state),
        actor_architecture=config['model']['actor_net'], optimizer='Independent fresh Adam per arm',
        learning_rate=RATE, batch_size=BATCH, gradient_clip=1., loss='Normalized full-33-coordinate MSE',
        seconds_per_arm=SECONDS, normalization='Shared frozen moments fitted once on 59040 training observations; clip=10',
        training_pool_pairs={name: len(obs) for name, (obs, _) in pools.items()}, added_groups=list(ADDED_GROUPS),
        added_target='Recorded live full-teacher proposal, never the executed mixture command',
        sampling=dict(control=dict(corrections=256, teacher_early=128, teacher_uniform=128),
            augmented=dict(corrections=128, teacher_early=64, teacher_uniform=64, added_early=128, added_uniform=128)),
        mixture_sampling='Exactly eight rows per generating actor and pose in each 128-row stratum; random times',
        evaluations='Final only; 8 native and 8 GPU autonomous 12-second-or-FAIL cases per arm, 32 cases total',
        frozen_inference_groups=list(GROUPS), demonstrations=clips, new_amp_motion_clips=0,
        initialize_from=None, checkpoint_selection='Final elapsed-time checkpoint only',
        ppo_updates=0, discriminator_updates=0, arena_changed=False)
    save(args.out/'protocol.json', protocol)
    save(args.out/'contract.json', contract)
    save(args.out/'initialization.json', dict(seed=SEED, fresh_full_tensor_hash=fresh_hash,
        common_normalized_full_tensor_hash=tensor_hash(common_state), fixed_critic_discriminator_hash=fixed_hash))
    results = {}
    for label in ('control', 'augmented'):
        out = args.out/label
        out.mkdir()
        (out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        save(out/'contract.json', contract)
        actor, training = train_arm(agent, common_state, gpu_pools, int(np.ceil(2./task.dt)),
                                    out, label == 'augmented', fixed_names, fixed_hash)
        evaluations, inference = assess(rig, reference, actor, traces, contract, out)
        report = dict(**training, evaluations=evaluations, frozen_inference=inference,
            arena_changed=False, completed_utc=datetime.now(timezone.utc).isoformat())
        save(out/'report.json', report)
        results[label] = report
        print(dict(arm=label, survivors={b: r['survivors'] for b, r in evaluations.items()},
                   stand_passes={b: r['standing_passes'] for b, r in evaluations.items()}), flush=True)
    a, b = results.values()
    for key in ('initial_full_tensor_hash', 'initial_actor_hash', 'common_normalizer_hash', 'normalizer_counts'):
        assert a[key] == b[key], key
    check_files(protected, sources)
    intervals = sum(e['live_control_intervals'] for r in results.values() for e in r['evaluations'].values())
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(), arms=results,
        common_initialization_verified=True, common_frozen_normalization_verified=True,
        critic_and_discriminator_unchanged=True, portable_sidecar_match=True,
        final_autonomous_cases=32, live_evaluation_control_intervals=intervals,
        supervised_physics_transitions=0, frozen_inference_live_predictions=32061*2,
        ppo_updates=0, discriminator_updates=0, protected_inputs_unchanged=True,
        sources_and_snapshots_unchanged=True, arena_changed=False)
    save(args.out/'report.json', summary)
    output_hashes = {str(path): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}
    save(args.out/'completion-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_input_count=len(protected), source_count=len(sources), inputs_unchanged=True,
        sources_and_snapshots_unchanged=True, common_initialization_verified=True,
        common_frozen_normalization_verified=True, final_autonomous_cases=32,
        portable_sidecar_match=True, output_hashes=output_hashes))
    print(dict(completed=True, output=str(args.out)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
