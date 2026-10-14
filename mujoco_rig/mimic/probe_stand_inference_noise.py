"""Fixed-weight clipping ablation and paired Gaussian exploration tolerance tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from .admit_stand_controller import gpu_native_state
from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController

CLIPS = (10., 20., 50., float('inf'))
SIGMAS = (0., .0025, .005, .01, .025, .1)
SEEDS = (301010, 301011, 301012)


def make_task(rig, reference, backend, seconds):
    from envs.base_env import EnvMode
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
                         episode_seconds=seconds, **options)
    task.set_mode(EnvMode.TEST)
    return task


@torch.no_grad()
def teacher_trial(task, backend, sigma, seed, noise, out):
    from envs.base_env import DoneFlags
    task.reset()
    controller = PredictiveStandController(task, state_reader=None if backend == 'native'
                                           else lambda: gpu_native_state(task))
    active = np.ones(8, dtype=bool)
    records = {key: [] for key in ('alive_before', 'done', 'balance', 'posture', 'teacher_action',
        'executed_action', 'pelvis_height_m', 'pelvis_tilt_degrees', 'foot_loads_n', 'solver_warning')}
    started = time.perf_counter()
    for step in range(len(noise)):
        if backend == 'native':
            task.engine.active = active.copy()
        command = controller.command(active).to(task.device)
        executed = (command+sigma*noise[step]).clamp(-1, 1)
        executed[torch.tensor(~active, device=task.device)] = 0.
        _, _, done, _ = task.step(executed)
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.tensor(warnings, device=task.device)] = DoneFlags.FAIL.value
        state = task.standing_measurements()
        values = dict(alive_before=active.copy(), done=done.cpu().numpy(),
            balance=torch.stack(list(task.balance_criteria(state).values())).all(0).cpu().numpy(),
            posture=torch.stack(list(task.posture_criteria(state).values())).all(0).cpu().numpy(),
            teacher_action=command.cpu().numpy(), executed_action=executed.cpu().numpy(),
            pelvis_height_m=state['pelvis_height_m'].cpu().numpy(),
            pelvis_tilt_degrees=state['pelvis_tilt_degrees'].cpu().numpy(),
            foot_loads_n=state['foot_loads_n'].cpu().numpy(), solver_warning=warnings)
        for key, value in values.items():
            records[key].append(value.copy())
        active &= done.cpu().numpy() == DoneFlags.NULL.value
        if (step+1)%300 == 0:
            print(dict(test='teacher_noise', backend=backend, sigma=sigma, seed=seed,
                       simulated_seconds=(step+1)*task.dt, live_cases=int(active.sum())), flush=True)
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
        balance = bool(survived and arrays['balance'][final, case].all())
        standing = bool(balance and arrays['posture'][final, case].all())
        delta = arrays['executed_action'][live, case]-arrays['teacher_action'][live, case]
        cases.append(dict(case=case, survived=survived, balance_pass=balance, standing_pass=standing,
            survival_seconds=float(arrays['time_s'][end]),
            final_pelvis_height_m=float(arrays['pelvis_height_m'][end, case]),
            balance_violation_seconds=float((~arrays['balance'][live, case]).sum()*task.dt),
            posture_violation_seconds=float((~arrays['posture'][live, case]).sum()*task.dt),
            applied_noise_rms=float(np.sqrt(np.mean(delta**2))),
            applied_noise_max=float(np.abs(delta).max()),
            action_saturation_fraction=float((np.abs(arrays['executed_action'][live, case])>=.98).mean()),
            solver_warning=bool(arrays['solver_warning'][end, case])))
    report = dict(backend=backend, sigma=sigma, seed=seed, seconds=task.episode_seconds, cases=cases,
        survivors=sum(c['survived'] for c in cases), standing_passes=sum(c['standing_pass'] for c in cases),
        balance_passes=sum(c['balance_pass'] for c in cases), wall_seconds=time.perf_counter()-started,
        noise_schedule='Gaussian noise at every control interval through the entire episode',
        teacher='Unchanged predictive controller reading its own live backend state',
        external_support=False, passive_springs=False, training_updates=0)
    write_json(out/'report.json', report)
    print(dict(test='teacher_noise_complete', backend=backend, sigma=sigma, seed=seed,
               standing_passes=report['standing_passes'], survivors=report['survivors']), flush=True)
    return report


def summary_reports(clipping, exploration):
    result = dict(clipping={}, exploration={})
    for backend in ('native', 'gpu'):
        result['clipping'][backend] = {}
        for label in ('10', '20', '50', 'off'):
            trials = [r for r in clipping if r['backend']==backend and r['clip_label']==label]
            durations = [c['survival_seconds'] for r in trials for c in r['cases']]
            result['clipping'][backend][label] = dict(repeats=len(trials),
                standing_passes=sum(r['standing_passes'] for r in trials),
                assessed_cases=len(durations), mean_survival_seconds=float(np.mean(durations)),
                min_survival_seconds=float(np.min(durations)), max_survival_seconds=float(np.max(durations)))
        result['exploration'][backend] = {}
        for sigma in SIGMAS:
            trials = [r for r in exploration if r['backend']==backend and r['sigma']==sigma]
            cases = [c for r in trials for c in r['cases']]
            result['exploration'][backend][str(sigma)] = dict(repeats=len(trials),
                standing_passes=sum(r['standing_passes'] for r in trials),
                survivors=sum(r['survivors'] for r in trials), assessed_cases=len(cases),
                mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in cases])),
                mean_applied_noise_rms=float(np.mean([c['applied_noise_rms'] for c in cases])))
    return result


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from learning.ppo_agent import PPOAgent
    from .native_engine import NativeDummyEngine
    from .torque_stand_task import TorqueStandTask
    from .stable_normalizer import install
    from .policy import StandPolicy
    from .train_stand_imitation import evaluate_actor, tensor_hash
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    model, reference = args.dataset/'dummy.xml', args.dataset/'stationary_reference.npz'
    protected = {p: sha256(p) for p in (model, reference, args.actor/'model.pt', args.actor/'agent.yaml')}
    rig = Rig.load(model, policy_exclude=())
    env = TorqueStandTask(rig, reference, 8, 'cpu', engine_factory=NativeDummyEngine, reward_mode='reference_v9')
    agent = PPOAgent(yaml.safe_load((args.actor/'agent.yaml').read_text()), env, 'cpu')
    install(agent)
    agent.load(str(args.actor/'model.pt'))
    actor = StandPolicy(agent).eval()
    frozen_hash = tensor_hash(actor.state_dict())
    shutil.copy2(Path(__file__), args.out/'runner.py')
    shutil.copy2(Path('mujoco_rig/mimic/train_stand_imitation.py'), args.out/'actor_evaluation_source.py')
    write_json(args.out/'protocol.json', dict(inputs={str(p): h for p, h in protected.items()},
        source_sha256=sha256(Path(__file__)), actor_evaluator_sha256=sha256(args.out/'actor_evaluation_source.py'),
        frozen_actor_tensor_sha256=frozen_hash,
        clips=[None if not np.isfinite(c) else c for c in CLIPS], clip_repeats={'native':1, 'gpu':2},
        clipping_seconds=args.clipping_seconds, sigmas=list(SIGMAS), seeds=list(SEEDS),
        noise_seconds=args.noise_seconds,
        noise='Float32 standard-normal CPU draws paired across backends and sigma settings; continuous per-control-frame noise',
        distribution_scope='Matches PPO Gaussian marginal and actuator/queue mapping; not its original RNG sequence or learned mean',
        warm_history='Full eight-world reset clears queue and solver history before every trial',
        training_updates=0, arena_changed=False))
    clipping = []
    for backend in ('native', 'gpu'):
        task = make_task(rig, reference, backend, args.clipping_seconds)
        for clip in CLIPS:
            label = 'off' if not np.isfinite(clip) else str(int(clip))
            for repeat in range(1 if backend=='native' else 2):
                out = args.out/f'clip-{backend}-{label}-{repeat+1}'
                out.mkdir()
                actor.obs_norm._clip = clip
                report = evaluate_actor(rig, reference, actor, backend, args.clipping_seconds, out, task=task)
                report.update(clip_label=label, repeat=repeat+1)
                clipping.append(report)
                write_json(out/'report.json', report)
                print(dict(test='clipping', backend=backend, clip=label, repeat=repeat+1,
                    standing_passes=report['standing_passes'],
                    mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in report['cases']]))), flush=True)
        if tensor_hash(actor.state_dict()) != frozen_hash:
            raise RuntimeError('Clipping test changed actor tensors')
    actor.obs_norm._clip = 10.
    write_json(args.out/'clipping.json', clipping)
    count = int(np.ceil(args.noise_seconds/(rig.model.opt.timestep*rig.decimation)))
    noises = {seed: torch.randn((count, 8, 33), generator=torch.Generator().manual_seed(seed)) for seed in SEEDS}
    np.savez_compressed(args.out/'standard-normal-noise.npz', **{str(seed): n.numpy() for seed, n in noises.items()})
    exploration = []
    for backend in ('native', 'gpu'):
        task = make_task(rig, reference, backend, args.noise_seconds)
        noise_by_seed = {seed: noise.to(task.device) for seed, noise in noises.items()}
        for sigma in SIGMAS:
            for seed in SEEDS:
                out = args.out/f'noise-{backend}-{sigma:g}-{seed}'
                out.mkdir()
                exploration.append(teacher_trial(task, backend, sigma, seed, noise_by_seed[seed], out))
                write_json(args.out/'exploration-progress.json', dict(completed_trials=len(exploration), total_trials=36))
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected input changed')
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        clipping=clipping, exploration=exploration, summary=summary_reports(clipping, exploration),
        noise_sha256=sha256(args.out/'standard-normal-noise.npz'), frozen_actor_unchanged=True,
        inputs_unchanged=True, training_updates=0, arena_changed=False)
    write_json(args.out/'report.json', report)
    write_json(args.out/'summary.json', report['summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--actor', type=Path, default=Path('logs/mimickit-training/stand-correction-pilot-20261001-01'))
    parser.add_argument('--clipping-seconds', type=float, default=60.)
    parser.add_argument('--noise-seconds', type=float, default=12.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    if args.clipping_seconds < 2 or args.noise_seconds < 2:
        parser.error('Require at least two seconds for the standing window')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
