"""Teacher labels on learner departures, filtered by actual teacher recovery."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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

SCHEDULES = [dict(name=f'learner-{seconds:g}s', learner_seconds=seconds, noise_amplitude=0.)
             for seconds in (.1, .2, .4, .6)] + [
    dict(name=f'torque-noise-{amplitude:g}', learner_seconds=0., noise_amplitude=amplitude)
    for amplitude in (.005, .02)]


@torch.no_grad()
def collect_backend(rig, reference, actor, backend, args, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
                         episode_seconds=args.episode_seconds, **options)
    task.set_mode(EnvMode.TEST)
    actor.to(device).eval()
    pooled = {key: [] for key in ('observation', 'action', 'executed_action', 'qpos_before', 'qvel_before',
        'pending_actions', 'case', 'episode', 'step', 'teacher_executed', 'departure')}
    reports, amp_clips = [], []
    count = int(np.ceil(args.episode_seconds/task.dt))
    source = json.loads(reference.with_suffix('.json').read_text())['source']
    for episode, schedule in enumerate(SCHEDULES):
        task.reset()
        controller = PredictiveStandController(task, state_reader=None if backend == 'native'
                                               else lambda: gpu_native_state(task))
        active = np.ones(8, dtype=bool)
        rows = {key: [] for key in ('observation', 'action', 'executed_action', 'qpos_before', 'qvel_before',
            'qpos_after', 'qvel_after', 'pending_actions', 'alive_before', 'done', 'balance', 'posture',
            'solver_warning', 'departure')}
        learner_steps = int(np.ceil(schedule['learner_seconds']/task.dt))
        noise_steps = int(np.ceil(2./task.dt)) if schedule['noise_amplitude'] else 0
        rng = np.random.default_rng(args.seed+episode*107)
        started = time.perf_counter()
        error = None
        for step in range(count):
            if backend == 'native':
                task.engine.active = active.copy()
            observation = task.observations()
            q, v = controller.state_reader()
            queue = task.engine.policy_control
            pending = torch.cat([queue.pending[(queue.cursor+i)%len(queue.pending)]
                                 for i in range(len(queue.pending))], dim=-1)
            try:
                label = controller.command(active).to(device)
            except RuntimeError as exc:
                error = str(exc)
                break
            departure = step < max(learner_steps, noise_steps)
            executed = actor(observation) if step < learner_steps else label.clone()
            if step < noise_steps:
                amplitude = schedule['noise_amplitude']
                executed += torch.tensor(rng.uniform(-amplitude, amplitude, (8, 33)),
                                         device=device, dtype=torch.float32)
            executed = executed.clamp(-1, 1)
            executed[torch.tensor(~active, device=device)] = 0.
            _, _, done, _ = task.step(executed)
            warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
            done[torch.tensor(warnings, device=device)] = DoneFlags.FAIL.value
            state = task.standing_measurements()
            after_q, after_v = controller.state_reader()
            values = dict(observation=observation.cpu().numpy(), action=label.cpu().numpy(),
                executed_action=executed.cpu().numpy(), qpos_before=q, qvel_before=v,
                qpos_after=after_q, qvel_after=after_v, pending_actions=pending.cpu().numpy(),
                alive_before=active.copy(), done=done.cpu().numpy(),
                balance=torch.stack(list(task.balance_criteria(state).values())).all(0).cpu().numpy(),
                posture=torch.stack(list(task.posture_criteria(state).values())).all(0).cpu().numpy(),
                solver_warning=warnings, departure=np.full(8, departure))
            for key, value in values.items():
                rows[key].append(value.copy())
            active &= done.cpu().numpy() == DoneFlags.NULL.value
            if (step+1)%300 == 0:
                print(dict(backend=backend, schedule=schedule['name'], simulated_seconds=(step+1)*task.dt,
                           live_cases=int(active.sum())), flush=True)
            if not active.any():
                break
        episode_out = out/schedule['name']
        episode_out.mkdir()
        if not rows['done']:
            reports.append(dict(schedule=schedule, accepted_cases=[], error=error))
            continue
        arrays = {key: np.stack(value) for key, value in rows.items()}
        arrays['time_s'] = (np.arange(len(arrays['done']))+1)*task.dt
        np.savez_compressed(episode_out/'trace.npz', **arrays)
        final = arrays['time_s'] > arrays['time_s'][-1]-2.
        selected, cases = [], []
        for case in range(8):
            finite = all(np.isfinite(a[:, case]).all() for key, a in arrays.items() if key != 'time_s')
            recovered = bool(error is None and len(arrays['done']) == count and
                arrays['done'][-1, case] == DoneFlags.TIME.value and arrays['balance'][final, case].all() and
                arrays['posture'][final, case].all() and not arrays['solver_warning'][:, case].any() and finite)
            if recovered:
                selected.append(case)
            cases.append(dict(case=case, accepted=recovered, finite=finite,
                survived=bool(arrays['done'][-1, case] == DoneFlags.TIME.value),
                balance_violation_seconds=float((~arrays['balance'][:, case] & arrays['alive_before'][:, case]).sum()*task.dt),
                posture_violation_seconds=float((~arrays['posture'][:, case] & arrays['alive_before'][:, case]).sum()*task.dt)))
        if selected:
            keep = np.zeros(arrays['done'].shape, dtype=bool)
            keep[:, selected] = True
            for key in pooled:
                if key in arrays:
                    pooled[key].append(arrays[key][keep])
            shape = arrays['done'].shape
            pooled['case'].append(np.broadcast_to(np.arange(8), shape)[keep])
            pooled['episode'].append(np.full(keep.sum(), episode, dtype=int))
            pooled['step'].append(np.broadcast_to(np.arange(len(keep))[:, None], shape)[keep])
            pooled['teacher_executed'].append(~arrays['departure'][keep])
            # Only a physically quiet teacher-controlled tail is an AMP style clip.
            tail = arrays['time_s'] > arrays['time_s'][-1]-4.
            for case in selected:
                if not (arrays['balance'][tail, case].all() and arrays['posture'][tail, case].all()):
                    continue
                path = episode_out/f'settled-case-{case}.npz'
                np.savez_compressed(path, qpos=arrays['qpos_after'][tail, case],
                                    qvel=arrays['qvel_after'][tail, case], dt=task.dt)
                write_json(path.with_suffix('.json'), dict(schema='mimic_physical_stand_teacher_v1',
                    passed=True, model_sha256=sha256(rig.path), reference_sha256=sha256(path), source=source,
                    backend=backend, case=case, schedule=schedule,
                    physical_admission='Last four seconds are teacher-controlled; every sampled frame passes standing gates',
                    modifications='Physically admitted settled tail after a learner or torque-noise departure',
                    qvel_semantics='Native local free-joint angular axes', external_support=False, passive_springs=False))
                amp_clips.append(dict(path=str(path), sha256=sha256(path)))
        report = dict(schedule=schedule, cases=cases, accepted_cases=selected,
            recorded_seconds=float(arrays['time_s'][-1]), wall_seconds=time.perf_counter()-started,
            trace_sha256=sha256(episode_out/'trace.npz'), error=error)
        reports.append(report)
        write_json(episode_out/'report.json', report)
        print(dict(backend=backend, schedule=schedule['name'], accepted_cases=selected), flush=True)
        write_json(out/'progress.json', dict(completed_episodes=len(reports), total_episodes=len(SCHEDULES),
                                           accepted_cases=sum(len(r['accepted_cases']) for r in reports)))
    arrays = {key: np.concatenate(value) for key, value in pooled.items()} if pooled['observation'] else None
    if arrays is not None:
        contract = task.observation_contract()
        group = next(g for g in contract['observation_layout'] if g['name']=='pending_actions_oldest_first')
        start, width = group['offset'], group['width']
        if not np.array_equal(arrays['observation'][:, start:start+width], arrays['pending_actions']):
            raise RuntimeError('Correction observations lost delayed-control history')
        np.savez_compressed(out/'dataset.npz', **arrays)
    report = dict(backend=backend, passed=arrays is not None, episodes=reports, amp_clips=amp_clips,
        admitted_samples=0 if arrays is None else len(arrays['observation']),
        learner_or_noise_state_labels=0 if arrays is None else int(arrays['departure'].sum()),
        counterfactual_label_scope='Teacher actions on departure states are queried labels; actual executed actions are recorded separately',
        amp_scope='Only settled teacher-controlled tails; no failed or learner-controlled motion is promoted to AMP')
    write_json(out/'report.json', report)
    actor.cpu()
    return report


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from learning.ppo_agent import PPOAgent
    from .native_engine import NativeDummyEngine
    from .torque_stand_task import TorqueStandTask
    from .stable_normalizer import install
    from .policy import StandPolicy
    mp_util.init(0, 1, 'cpu', 0)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    model, reference = args.dataset/'dummy.xml', args.dataset/'stationary_reference.npz'
    protected = {p: sha256(p) for p in (model, reference, args.actor/'model.pt', args.actor/'agent.yaml')}
    rig = Rig.load(model, policy_exclude=())
    task = TorqueStandTask(rig, reference, 8, 'cpu', engine_factory=NativeDummyEngine, reward_mode='reference_v9')
    agent = PPOAgent(yaml.safe_load((args.actor/'agent.yaml').read_text()), task, 'cpu')
    install(agent)
    agent.load(str(args.actor/'model.pt'))
    actor = StandPolicy(agent).eval()
    write_json(args.out/'protocol.json', dict(seed=args.seed, episode_seconds=args.episode_seconds,
        schedules=SCHEDULES, noise_seconds=2., source_sha256=sha256(Path(__file__)),
        controller_source_sha256=sha256(Path('mujoco_rig/mimic/stand_foundation_followup.py')),
        inputs={str(p): h for p, h in protected.items()}, training='Admitted native cases only',
        validation='Independent GPU cases; no validation samples train the actor',
        resets='Original eight candidate poses and empty queue for every episode',
        admission='No physical failure; final two seconds satisfy balance and posture',
        training_updates=0, arena_changed=False))
    reports = {}
    for backend in ('native', 'gpu'):
        out = args.out/backend
        out.mkdir()
        reports[backend] = collect_backend(rig, reference, actor, backend, args, out)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected input changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(r['passed'] for r in reports.values()), model_sha256=sha256(model),
        reference_sha256=sha256(reference), reports=reports, inputs_unchanged=True, training_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--actor', type=Path, default=Path('logs/mimickit-training/stand-imitation-pilot-20261001-01'))
    parser.add_argument('--episode-seconds', type=float, default=12.)
    parser.add_argument('--seed', type=int, default=301010)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    if args.episode_seconds < 8:
        parser.error('Require at least eight seconds for recovery and settled-tail admission')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
