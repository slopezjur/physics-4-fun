"""Record admitted native/GPU Stand teacher demonstrations in an isolated bundle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .admit_stand_controller import gpu_native_state
from .baseline import sha256
from .diagnose_stand_foundations import DiagnosticNativeEngine, write_json
from .rig import Rig
from .runtime import activate
from .stand_foundation_followup import PredictiveStandController


@torch.no_grad()
def collect(rig, reference, backend, seconds, out):
    from envs.base_env import EnvMode, DoneFlags
    from .torque_stand_task import TorqueStandTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    options = dict(engine_factory=DiagnosticNativeEngine) if backend == 'native' else {}
    task = TorqueStandTask(rig, reference, 8, device, reward_mode='reference_v9',
                         episode_seconds=seconds, **options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    controller = PredictiveStandController(task, state_reader=None if backend == 'native'
                                           else lambda: gpu_native_state(task))
    q0, v0 = controller.state_reader()
    records = {key: [] for key in ('observation', 'action', 'pending_actions', 'qpos_before',
        'qvel_before', 'qpos_after', 'qvel_after', 'balance', 'posture', 'done', 'solver_warning')}
    started = time.perf_counter()
    count = int(np.ceil(seconds/task.dt))
    progress_interval = int(np.ceil(5./task.dt))
    for step in range(count):
        q, v = controller.state_reader()
        queue = task.engine.policy_control
        pending = torch.cat([queue.pending[(queue.cursor+i)%len(queue.pending)]
                             for i in range(len(queue.pending))], dim=-1)
        obs = task.observations()
        action = controller.command(np.ones(8, dtype=bool)).to(device)
        _, _, done, _ = task.step(action)
        state = task.standing_measurements()
        after_q, after_v = controller.state_reader()
        warnings = task.engine.solver_warning if backend == 'native' else np.zeros(8, dtype=bool)
        values = dict(observation=obs.cpu().numpy(), action=action.cpu().numpy(),
            pending_actions=pending.cpu().numpy(), qpos_before=q, qvel_before=v,
            qpos_after=after_q, qvel_after=after_v,
            balance=torch.stack(list(task.balance_criteria(state).values())).all(0).cpu().numpy(),
            posture=torch.stack(list(task.posture_criteria(state).values())).all(0).cpu().numpy(),
            done=done.cpu().numpy(), solver_warning=warnings)
        for key, value in values.items():
            records[key].append(value.copy())
        if (step+1)%progress_interval == 0 or step+1 == count:
            progress = dict(backend=backend, simulated_seconds=(step+1)*task.dt,
                            wall_seconds=time.perf_counter()-started)
            write_json(out/'progress.json', progress)
            print(progress, flush=True)
        if (done == DoneFlags.FAIL.value).any() or np.any(warnings):
            break
    arrays = {key: np.stack(value) for key, value in records.items()}
    arrays['time_s'] = (np.arange(len(arrays['done']))+1)*task.dt
    np.savez_compressed(out/'dataset.npz', **arrays)
    contract = task.observation_contract()
    queue_group = next(g for g in contract['observation_layout'] if g['name']=='pending_actions_oldest_first')
    start, width = queue_group['offset'], queue_group['width']
    queue_matches = bool(np.array_equal(arrays['observation'][:, :, start:start+width], arrays['pending_actions']))
    finite = all(np.isfinite(a).all() for a in arrays.values())
    passed = bool(len(arrays['done']) == count and
        np.all(arrays['done'][-1] == DoneFlags.TIME.value) and arrays['balance'].all() and
        arrays['posture'].all() and not arrays['solver_warning'].any() and finite and queue_matches)
    metadata = json.loads(reference.with_suffix('.json').read_text())
    clips = []
    if passed:
        for case in range(8):
            path = out/f'case-{case}.npz'
            q = np.concatenate((q0[case:case+1], arrays['qpos_after'][:, case]))
            v = np.concatenate((v0[case:case+1], arrays['qvel_after'][:, case]))
            np.savez_compressed(path, qpos=q, qvel=v, dt=task.dt)
            manifest = dict(schema='mimic_physical_stand_teacher_v1', passed=True,
                model_sha256=sha256(rig.path), reference_sha256=sha256(path), source=metadata['source'],
                derived_from=dict(path=str(reference), sha256=sha256(reference)), backend=backend,
                case=case, seconds=float(arrays['time_s'][-1]), frames=len(q),
                physical_admission='Every recorded frame passes balance and posture; complete 60-second hold',
                qvel_semantics='Native MuJoCo velocity; free-joint angular velocity uses local axes',
                modifications='Physical predictive-controller rollout from the passive-wrist candidate',
                external_support=False, passive_springs=False)
            write_json(path.with_suffix('.json'), manifest)
            clips.append(dict(path=str(path), sha256=sha256(path)))
    report = dict(backend=backend, passed=passed, frames=len(arrays['done']),
        transitions=int(arrays['action'].shape[0]*arrays['action'].shape[1]),
        seconds=float(arrays['time_s'][-1]), finite=finite, queue_observation_matches=queue_matches,
        balance_violations=int((~arrays['balance']).sum()), posture_violations=int((~arrays['posture']).sum()),
        dataset_sha256=sha256(out/'dataset.npz'), clips=clips, wall_seconds=time.perf_counter()-started,
        gpu_warning_scope='Finite-state and physical gates; no native warning counters on GPU')
    write_json(out/'contract.json', contract)
    write_json(out/'report.json', report)
    if not passed:
        raise RuntimeError(f'{backend} teacher failed dataset admission')
    return report


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    protected = {p: sha256(p) for p in (args.model, args.reference)}
    shutil.copy2(args.model, args.out/'dummy.xml')
    for suffix in ('.npz', '.json'):
        shutil.copy2(args.reference.with_suffix(suffix), (args.out/'stationary_reference.npz').with_suffix(suffix))
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    write_json(args.out/'protocol.json', dict(seed=args.seed, seconds=args.seconds,
        source_sha256=sha256(Path(__file__)), controller_source_sha256=sha256(Path('mujoco_rig/mimic/stand_foundation_followup.py')),
        inputs={str(p): h for p, h in protected.items()}, training_split='Native teacher only',
        validation_split='Independent GPU teacher; same eight starting poses',
        mirror_demos=False, training_updates=0, arena_changed=False))
    reports = {}
    for backend in ('native', 'gpu'):
        out = args.out/backend
        out.mkdir()
        reports[backend] = collect(rig, args.out/'stationary_reference.npz', backend, args.seconds, out)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected input changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, reports=reports, inputs_unchanged=True, training_updates=0, arena_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('logs/mimickit-training/stand-fresh-winner-60min-20261001-01/dummy.xml'))
    parser.add_argument('--reference', type=Path, default=Path('logs/mimickit-training/stand-foundation-followup-20261001-01/passive-wrist-reference.npz'))
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--seed', type=int, default=301010)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    if args.seconds < 60:
        parser.error('Teacher admission requires at least 60 seconds')
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
