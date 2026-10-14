"""Fixed-budget fresh actor imitation of admitted physical Stand demonstrations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def action_metrics(actor, observations, actions, dt):
    chunks = [actor(x) for x in observations.flatten(0, 1).split(2048)]
    prediction = torch.cat(chunks).reshape_as(actions)
    error = prediction-actions
    result = dict(rmse=float(error.square().mean().sqrt()),
        max_absolute_error=float(error.abs().max()),
        per_motor_rmse=error.square().mean((0, 1)).sqrt().cpu().tolist(),
        scope='Teacher-state action regression; not autonomous physical standing')
    if dt is not None:
        early = max(1, int(np.ceil(2./dt)))
        result['first_two_seconds_rmse'] = float(error[:early].square().mean().sqrt())
    return result


@torch.no_grad()
def evaluate_actor(rig, reference, actor, backend, seconds, out, task=None):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    if task is None:
        task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
                             episode_seconds=seconds, **options)
    task.episode_seconds = seconds
    task.set_mode(EnvMode.TEST)
    task.reset()
    actor.to(device).eval()
    active = np.ones(8, dtype=bool)
    records = {key: [] for key in ('alive_before', 'done', 'balance', 'posture', 'pelvis_height_m',
        'pelvis_tilt_degrees', 'foot_loads_n', 'command', 'obs_clip_fraction',
        'outside_clip10_fraction', 'solver_warning')}
    started = time.perf_counter()
    for step in range(int(np.ceil(seconds/task.dt))):
        if backend == 'native':
            task.engine.active = active.copy()
        obs = task.observations()
        unclipped = (obs-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
        action = actor(obs)
        action[torch.tensor(~active, device=device)] = 0.
        _, _, done, _ = task.step(action)
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings, device=device)] = DoneFlags.FAIL.value
        state = task.standing_measurements()
        values = dict(alive_before=active, done=done.cpu().numpy(),
            balance=torch.stack(list(task.balance_criteria(state).values())).all(0).cpu().numpy(),
            posture=torch.stack(list(task.posture_criteria(state).values())).all(0).cpu().numpy(),
            pelvis_height_m=state['pelvis_height_m'].cpu().numpy(),
            pelvis_tilt_degrees=state['pelvis_tilt_degrees'].cpu().numpy(),
            foot_loads_n=state['foot_loads_n'].cpu().numpy(), command=action.cpu().numpy(),
            obs_clip_fraction=(unclipped.abs()>actor.obs_norm._clip).float().mean(-1).cpu().numpy(),
            outside_clip10_fraction=(unclipped.abs()>10.).float().mean(-1).cpu().numpy(), solver_warning=warnings)
        for key, value in values.items():
            records[key].append(value.copy())
        active &= done.cpu().numpy() == DoneFlags.NULL.value
        if (step+1)%300 == 0:
            print(dict(evaluation=backend, simulated_seconds=(step+1)*task.dt,
                       live_cases=int(active.sum())), flush=True)
        if not active.any():
            break
    arrays = {key: np.stack(value) for key, value in records.items()}
    arrays['time_s'] = (np.arange(len(arrays['done']))+1)*task.dt
    np.savez_compressed(out/'trace.npz', **arrays)
    cases = []
    for case in range(8):
        live = arrays['alive_before'][:, case]
        end = np.flatnonzero(live)[-1]
        survived = bool(arrays['done'][end, case] == DoneFlags.TIME.value)
        final = live & (arrays['time_s'] > arrays['time_s'][end]-2.)
        full_window = arrays['time_s'][end] >= 2.
        balance = bool(survived and full_window and arrays['balance'][final, case].all())
        standing = bool(balance and arrays['posture'][final, case].all())
        cases.append(dict(case=case, survived=survived, balance_pass=balance, standing_pass=standing,
            survival_seconds=float(arrays['time_s'][end]), pelvis_height_end_m=float(arrays['pelvis_height_m'][end, case]),
            balance_violation_seconds=float((~arrays['balance'][live, case]).sum()*task.dt),
            posture_violation_seconds=float((~arrays['posture'][live, case]).sum()*task.dt),
            mean_observation_clip_fraction=float(arrays['obs_clip_fraction'][live, case].mean()),
            mean_outside_clip10_fraction=float(arrays['outside_clip10_fraction'][live, case].mean()),
            solver_warning=bool(arrays['solver_warning'][end, case])))
    report = dict(backend=backend, seconds=seconds, cases=cases,
        observation_clip=float(actor.obs_norm._clip) if np.isfinite(actor.obs_norm._clip) else None,
        survivors=sum(c['survived'] for c in cases), balance_passes=sum(c['balance_pass'] for c in cases),
        standing_passes=sum(c['standing_pass'] for c in cases), wall_seconds=time.perf_counter()-started,
        controller_at_inference=False, external_support=False, passive_springs=False,
        scope='Autonomous bounded neural actions through the unchanged two-step queue')
    write_json(out/'report.json', report)
    actor.cpu()
    return report


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .torque_stand_task import AMPTorqueStandTask
    from .native_engine import NativeDummyEngine
    from .train_torque_stand import build_agent, state_sha256
    from .amp_recovery import validate_sidecar
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(2)
    dataset_report = json.loads((args.dataset/'report.json').read_text())
    if not dataset_report['passed']:
        raise ValueError('Teacher dataset is not admitted')
    args.out.mkdir(parents=True, exist_ok=False)
    protected_paths = [args.dataset/'report.json', args.config,
                       args.dataset/'native/dataset.npz', args.dataset/'gpu/dataset.npz',
                       args.dataset/'dummy.xml', args.dataset/'stationary_reference.npz']
    protected = {p: sha256(p) for p in protected_paths}
    corrections = getattr(args, 'corrections', None)
    if corrections:
        correction_report = json.loads((corrections/'report.json').read_text())
        if not correction_report['passed']:
            raise ValueError('Correction dataset has no admitted native/GPU samples')
        for name in ('report.json', 'native/dataset.npz', 'gpu/dataset.npz'):
            protected[corrections/name] = sha256(corrections/name)
        if correction_report['model_sha256'] != sha256(args.dataset/'dummy.xml') or \
                correction_report['reference_sha256'] != sha256(args.dataset/'stationary_reference.npz'):
            raise ValueError('Correction plant/reference mismatch')
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(args.dataset/name, args.out/name)
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    clip_records = list(dataset_report['reports']['native']['clips'])
    if corrections:
        clip_records += correction_report['reports']['native']['amp_clips']
    clips = [Path(c['path']) for c in clip_records]
    for clip, record in zip(clips, clip_records):
        expected = record['sha256']
        if sha256(clip) != expected:
            raise ValueError('Teacher clip hash mismatch')
        protected[clip] = expected
        protected[clip.with_suffix('.json')] = sha256(clip.with_suffix('.json'))
    task = AMPTorqueStandTask(rig, reference, 8, 'cpu', engine_factory=NativeDummyEngine,
        reward_mode='reference_v9', episode_seconds=60., demonstrations=clips, demonstration_rig=rig)
    task.mirror_demos = False
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent = build_agent(config, task, 'cpu')
    initial_hash = state_sha256(agent)
    fixed_names = [key for key in agent.state_dict() if key.startswith(('_model._critic', '_model._disc', '_disc_obs_norm'))]
    fixed_hash = tensor_hash({key: agent.state_dict()[key] for key in fixed_names})
    agent.save(str(args.out/'initial.pt'))
    write_json(args.out/'initialization.json', dict(seed=args.seed, state_sha256=initial_hash,
        initialize_from=None, actor='fresh', critic='fresh', discriminator='fresh', normalizers='fresh',
        optimizer='Fresh supervised Adam; PPO optimizers receive zero updates'))
    contract = task.observation_contract()
    write_json(args.out/'contract.json', contract)
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    demo = task.fetch_disc_obs_demo(256)
    if demo.shape != (256, 364) or not torch.isfinite(demo).all():
        raise ValueError('AMP teacher feature contract failed')
    write_json(args.out/'amp-contract.json', dict(width=364, frames=4, hinge_count=39,
        passive_wrist_hinges=6, mirror_demos=False, finite_demo_sample=True,
        demonstrations=[dict(path=str(p), sha256=sha256(p)) for p in clips],
        discriminator_updates=0, scope='Physically admitted native clips; full original feature layout'))
    datasets = {}
    for backend in ('native', 'gpu'):
        with np.load(args.dataset/backend/'dataset.npz', allow_pickle=False) as arrays:
            datasets[backend] = (torch.tensor(arrays['observation'], device='cuda:0'),
                                 torch.tensor(arrays['action'], device='cuda:0'))
        if corrections:
            with np.load(corrections/backend/'dataset.npz', allow_pickle=False) as arrays:
                datasets[backend+'_corrections'] = (torch.tensor(arrays['observation'][:, None], device='cuda:0'),
                    torch.tensor(arrays['action'][:, None], device='cuda:0'))
    native_obs, native_action = datasets['native']
    # Fit observation moments on native training data only, never validation data.
    agent._obs_norm.record(native_obs.cpu().flatten(0, 1))
    if corrections:
        agent._obs_norm.record(datasets['native_corrections'][0].cpu().flatten(0, 1))
    agent._obs_norm.update()
    actor = StandPolicy(agent).to('cuda:0')
    actor.actor.requires_grad_(True)
    actor.mean.requires_grad_(True)
    for normalizer in (actor.obs_norm, actor.action_norm):
        normalizer.requires_grad_(False)
    parameters = list(actor.actor.parameters())+list(actor.mean.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    x = actor.obs_norm.normalize(native_obs.flatten(0, 1)).detach()
    y = actor.action_norm.normalize(native_action.flatten(0, 1)).detach()
    if corrections:
        correction_x = actor.obs_norm.normalize(datasets['native_corrections'][0].flatten(0, 1)).detach()
        correction_y = actor.action_norm.normalize(datasets['native_corrections'][1].flatten(0, 1)).detach()
    early_count = min(len(x), int(np.ceil(2./task.dt))*8)
    rng = torch.Generator(device='cuda:0').manual_seed(args.seed+701)
    protocol = dict(seconds=args.seconds, seed=args.seed, learning_rate=args.learning_rate,
        batch_size=args.batch_size, optimizer='Adam', gradient_clip=1., objective='Normalized action MSE',
        sampling=('Half native corrections; quarter original first two seconds; quarter original uniform'
                  if corrections else 'Half samples from first two seconds; half uniform over the full native dataset'),
        validation='Independent GPU teacher on the same eight poses; no unseen-pose generalization claim',
        normalization='Native training observations only; existing StableNormalizer and clip=10',
        actor_architecture=config['model']['actor_net'], initialize_from=None,
        checkpoint_selection='Final fixed-budget actor, no best-checkpoint selection',
        inputs={str(p): h for p, h in protected.items()}, source_sha256=sha256(Path(__file__)),
        ppo_updates=0, discriminator_updates=0, arena_changed=False)
    protocol['corrections'] = str(corrections) if corrections else None
    write_json(args.out/'protocol.json', protocol)
    before = {b: action_metrics(actor, *data, None if b.endswith('_corrections') else task.dt)
              for b, data in datasets.items()}
    write_json(args.out/'initial-action-metrics.json', before)
    started = time.perf_counter()
    iterations, samples, next_output = 0, 0, 15.
    curve = []
    while time.perf_counter()-started < args.seconds:
        n = args.batch_size//4 if corrections else args.batch_size//2
        correction_count = args.batch_size//2 if corrections else 0
        indices = torch.cat((torch.randint(early_count, (n,), generator=rng, device='cuda:0'),
            torch.randint(len(x), (args.batch_size-n-correction_count,), generator=rng, device='cuda:0')))
        batch_x, batch_y = x[indices], y[indices]
        if corrections:
            picks = torch.randint(len(correction_x), (correction_count,), generator=rng, device='cuda:0')
            batch_x, batch_y = torch.cat((batch_x, correction_x[picks])), torch.cat((batch_y, correction_y[picks]))
        prediction = actor.mean(actor.actor(batch_x))
        loss = torch.nn.functional.mse_loss(prediction, batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite supervised loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        iterations += 1
        samples += args.batch_size
        elapsed = time.perf_counter()-started
        if elapsed >= next_output:
            row = dict(seconds=elapsed, optimizer_updates=iterations, supervised_sample_presentations=samples,
                       minibatch_mse=float(loss.detach()))
            curve.append(row)
            print(row, flush=True)
            write_json(args.out/'progress.json', row)
            next_output += 15.
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-started
    after = {b: action_metrics(actor, *data, None if b.endswith('_corrections') else task.dt)
             for b, data in datasets.items()}
    write_json(args.out/'action-metrics.json', dict(initial=before, final=after))
    write_json(args.out/'curve.json', curve)
    torch.save(optimizer.state_dict(), args.out/'imitation-optimizer.pt')
    actor.cpu().eval()
    agent._model._actor_layers.load_state_dict(actor.actor.state_dict())
    agent._model._action_dist._mean_net.load_state_dict(actor.mean.state_dict())
    agent.save(str(args.out/'model.pt'))
    validate_sidecar(torch.load(args.out/'model.pt', weights_only=True),
                     torch.load(args.out/'model.amp.pt', weights_only=True))
    if tensor_hash({key: agent.state_dict()[key] for key in fixed_names}) != fixed_hash:
        raise RuntimeError('Critic/discriminator state changed during actor-only imitation')
    evaluations = {}
    for backend in ('native', 'gpu'):
        out = args.out/backend
        out.mkdir()
        print('Evaluating final autonomous actor:', backend, flush=True)
        evaluations[backend] = evaluate_actor(rig, reference, actor, backend, args.assessment_seconds, out)
        write_json(args.out/'evaluations.json', evaluations)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected input changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        requested_training_seconds=args.seconds, training_seconds=training_seconds,
        optimizer_updates=iterations, supervised_sample_presentations=samples,
        unique_native_training_transitions=len(x), independent_gpu_validation_transitions=int(datasets['gpu'][0].shape[0]*8),
        native_correction_samples=len(correction_x) if corrections else 0,
        initial_state_sha256=initial_hash, model_sha256=sha256(args.out/'model.pt'),
        final_action_metrics=after, evaluations=evaluations, critic_and_discriminator_unchanged=True,
        ppo_updates=0, discriminator_updates=0, arena_changed=False, protected_inputs_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--corrections', type=Path)
    parser.add_argument('--config', type=Path, default=Path('logs/mimickit-training/stand-fresh-winner-60min-20261001-01/agent.yaml'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--assessment-seconds', type=float, default=60.)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=301010)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    if args.seconds <= 0 or args.assessment_seconds < 2 or args.batch_size < 2 or args.learning_rate <= 0:
        parser.error('Require positive budget/rate, batch >=2 and evaluation >=2 seconds')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
