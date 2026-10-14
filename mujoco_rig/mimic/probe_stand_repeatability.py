"""Locate noiseless Stand divergence without changing the production solver."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .admit_stand_controller import gpu_native_state
from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .probe_stand_inference_noise import make_task, teacher_trial
from .rig import Rig
from .runtime import activate


def live_state(task, backend):
    if backend == 'gpu':
        return gpu_native_state(task)
    return (np.stack([d.qpos.copy() for d in task.engine.datas]),
            np.stack([d.qvel.copy() for d in task.engine.datas]))


def snapshot(task, backend):
    q, v = live_state(task, backend)
    control = task.engine.policy_control
    result = dict(qpos=q, qvel=v, pending=control.pending.cpu().numpy().copy(),
                  cursor=np.array(control.cursor),
                  ground_forces=task.engine.get_ground_contact_forces(0).cpu().numpy().copy())
    if backend == 'gpu':
        data = task.engine._solver.mjw_data
        for name in ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'ctrl', 'time', 'nefc', 'solver_niter'):
            result[name] = getattr(data, name).numpy().copy()
        n = int(data.nacon.numpy()[0])
        contact = data.contact
        rows = np.concatenate((contact.worldid.numpy()[:n, None], contact.geom.numpy()[:n],
            contact.dist.numpy()[:n, None], contact.pos.numpy()[:n],
            contact.frame.numpy()[:n].reshape(n, 9)), axis=1)
        result['contacts_raw'] = rows
        order = np.lexsort(tuple(rows[:, i] for i in reversed(range(7))))
        result['contacts_sorted'] = rows[order]
    else:
        for name in ('qacc', 'qacc_warmstart', 'qfrc_constraint', 'ctrl'):
            result[name] = np.stack([getattr(d, name).copy() for d in task.engine.datas])
        result['time'] = np.array([d.time for d in task.engine.datas])
    return result


def differences(first, second):
    result = {}
    for name, a in first.items():
        b = second[name]
        result[name] = dict(shape_matches=a.shape == b.shape, exactly_equal=np.array_equal(a, b),
            max_absolute_difference=float(np.abs(a.astype(float)-b).max())
                if a.shape == b.shape and a.size else None)
    return result


def trace_difference(first, second, dt):
    result = {}
    for name in ('qpos', 'qvel', 'teacher_action', 'foot_loads_n'):
        length = min(len(first[name]), len(second[name]))
        delta = np.abs(first[name][:length]-second[name][:length])
        frame_delta = delta.reshape(length, -1).max(1)
        differing = np.flatnonzero(frame_delta > 0)
        result[name] = dict(first_different_control_frame=int(differing[0]+1) if len(differing) else None,
            first_different_time_seconds=float((differing[0]+1)*dt) if len(differing) else None,
            first_difference_max=float(frame_delta[differing[0]]) if len(differing) else 0.,
            max_absolute_difference=float(delta.max()), common_frames=length)
    return result


def instrumented_hold(task, backend, seconds, out):
    task.episode_seconds = seconds
    out.mkdir()
    original_reset, original_step = task.reset, task.step
    states = {'qpos': [], 'qvel': []}
    reset_state = {}

    def reset(*args, **kwargs):
        value = original_reset(*args, **kwargs)
        reset_state.update(snapshot(task, backend))
        return value

    def step(action):
        value = original_step(action)
        q, v = live_state(task, backend)
        states['qpos'].append(q.copy())
        states['qvel'].append(v.copy())
        return value

    task.reset, task.step = reset, step
    try:
        count = int(np.ceil(seconds/task.dt))
        report = teacher_trial(task, backend, 0., 301010, torch.zeros(count, 8, 33, device=task.device), out)
    finally:
        task.reset, task.step = original_reset, original_step
    np.savez_compressed(out/'reset-state.npz', **reset_state)
    with np.load(out/'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k].copy() for k in saved.files}
    trace.update({k: np.stack(v) for k, v in states.items()})
    np.savez_compressed(out/'trace.npz', **trace)
    return report, reset_state, trace


@torch.no_grad()
def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    torch.manual_seed(301010)
    np.random.seed(301010)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(args.dataset/'dummy.xml', policy_exclude=())
    reference = args.dataset/'stationary_reference.npz'
    protected = {p: sha256(p) for p in (rig.path, reference)}
    shutil.copy2(Path(__file__), args.out/'runner.py')
    write_json(args.out/'protocol.json', dict(inputs={str(p): h for p, h in protected.items()},
        source_sha256=sha256(Path(__file__)), production_solver_unchanged=True, training_updates=0,
        arena_changed=False, reused_trials={'native': 2, 'gpu': 3}, reused_seconds=12.,
        fresh_gpu_seconds=60., replay_seconds=1., replay_repetitions=3,
        replay_scope='Identical commands from first GPU hold; no closed-loop controller during replay'))
    reports, comparisons = [], []
    for backend, repeats in (('native', 2), ('gpu', 3)):
        task = make_task(rig, reference, backend, 12.)
        first_reset, first_trace = None, None
        for repeat in range(repeats):
            report, reset, trace = instrumented_hold(task, backend, 12., args.out/f'{backend}-hold-{repeat+1}')
            reports.append(report)
            if first_reset is None:
                first_reset, first_trace = reset, trace
            else:
                comparisons.append(dict(backend=backend, repeat=repeat+1,
                    reset=differences(first_reset, reset), trajectory=trace_difference(first_trace, trace, task.dt)))
            write_json(args.out/'progress.json', dict(hold_reports=reports, comparisons=comparisons))
        if backend == 'gpu':
            commands = first_trace['teacher_action'][:int(np.ceil(1./task.dt))]
            replay_first = None
            for repeat in range(3):
                task.reset()
                reset = snapshot(task, backend)
                traces = dict(qpos=[], qvel=[], teacher_action=[], foot_loads_n=[])
                for command in commands:
                    task.step(torch.tensor(command, device=task.device))
                    q, v = live_state(task, backend)
                    for name, value in dict(qpos=q, qvel=v, teacher_action=command,
                            foot_loads_n=task.foot_loads().cpu().numpy()).items():
                        traces[name].append(value.copy())
                traces = {k: np.stack(v) for k, v in traces.items()}
                out = args.out/f'gpu-replay-{repeat+1}'
                out.mkdir()
                np.savez_compressed(out/'reset-state.npz', **reset)
                np.savez_compressed(out/'trace.npz', **traces)
                if replay_first is None:
                    replay_first = reset, traces
                else:
                    comparisons.append(dict(backend='gpu-open-loop', repeat=repeat+1,
                        reset=differences(replay_first[0], reset),
                        trajectory=trace_difference(replay_first[1], traces, task.dt)))
            fresh = make_task(rig, reference, 'gpu', 60.)
            report, _, _ = instrumented_hold(fresh, 'gpu', 60., args.out/'gpu-fresh-60s')
            reports.append(report)
    if any(sha256(p) != h for p, h in protected.items()):
        raise RuntimeError('Protected physics/reference changed')
    write_json(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        holds=reports, comparisons=comparisons, inputs_unchanged=True, training_updates=0,
        arena_changed=False, gpu_warning_scope='Native warning counters unavailable'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
