"""Frozen repeated-ball encounters with explicit per-encounter recovery evidence."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import read, save
from .ball import launch_state
from .replicate_stand_representations import cpu, pending_commands, read_state
from . import run_chest_frozen_followup as followup
from . import run_frozen_ball_screen as screen


def repeated_summary(trace, schedule, dt):
    """Require recovery after every hit, and two settled seconds before later hits."""
    required = int(np.ceil(2. / dt))
    launch_frames = [int(np.ceil(t / dt)) for t in schedule]
    rows = []
    for case in range(8):
        live = trace['alive_before'][:, case]
        valid = (trace['balance'][:, case] & trace['posture'][:, case]
                 & trace['pre_balance'][:, case] & trace['pre_posture'][:, case])
        encounters = []
        for index, begin in enumerate(launch_frames):
            end = launch_frames[index + 1] if index + 1 < len(launch_frames) else len(live)
            end = min(end, len(live))
            contacts = np.flatnonzero(trace['normal_impulse_ns'][begin:end, case] > 0.) + begin
            first, last = (int(contacts[0]), int(contacts[-1])) if len(contacts) else (None, None)
            pre_begin = 0 if index == 0 else max(0, begin - required)
            pre_end = first if first is not None else begin
            prequalified = bool(begin < len(live) and live[pre_begin:pre_end + 1].all()
                                and valid[pre_begin:pre_end].all())
            start = None
            if last is not None:
                for candidate in range(last + 1, min(last + required + 2, end)):
                    if (candidate + required <= end and live[candidate:candidate + required].all()
                            and valid[candidate:candidate + required].all()):
                        start = candidate
                        break
            final_valid = end - begin >= required and bool(live[end - required:end].all()
                                                           and valid[end - required:end].all())
            survived = begin < len(live) and end > begin and bool(live[begin:end].all())
            encounters.append(dict(index=index, launch_frame=begin, hit=first is not None,
                prequalified=prequalified, survived=bool(survived), final_valid=final_valid,
                first_contact_seconds=None if first is None else float(trace['time_s'][first]),
                last_contact_seconds=None if last is None else float(trace['time_s'][last]),
                recovery_window_start=None if start is None else float(trace['time_s'][start]),
                total_normal_impulse_ns=float(trace['normal_impulse_ns'][begin:end, case].sum()),
                passed=bool(first is not None and prequalified and survived and final_valid and start is not None)))
        rows.append(dict(case=case, encounters=encounters, passed=all(e['passed'] for e in encounters)))
    return rows


def wrap_evaluator(original, backend, settings):
    def evaluate(rig, reference, actor, world, out, cfg, speed, sign):
        from . import stand_ball_native, stand_ball_task
        matches = [c for c in settings['conditions'] if f"-{c['name']}-" in out.name]
        if len(matches) != 1:
            raise ValueError('Ambiguous repeated condition')
        condition = matches[0]
        schedule = condition['launch_times_seconds']
        if (schedule[0] != cfg['launch_seconds'] or sorted(set(schedule)) != schedule
                or schedule[-1] + 5 > cfg['seconds'] or len(schedule) > 3):
            raise ValueError('Invalid bounded repeated schedule')
        if condition.get('alternating', False):
            signs = [sign * (-1)**i for i in range(len(schedule))]
        else:
            signs = [sign] * len(schedule)
        module, name = (stand_ball_native, 'StandBallTask') if backend == 'native' else (stand_ball_task, 'StandBallGPUTask')
        base = getattr(module, name)
        journal = {}

        class RepeatedTask(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.repeat_frame = 0
                self.repeat_inside_step = False
                self.repeat_active = np.ones(8, dtype=bool)
                self.repeat_events = []
                self.repeat_frames = {int(np.ceil(t / self.dt)): i for i, t in enumerate(schedule) if i}
                journal['task'] = self

            def observations(self):
                frame = getattr(self, 'repeat_frame', -1)
                if not getattr(self, 'repeat_inside_step', True) and frame in self.repeat_frames:
                    index = self.repeat_frames.pop(frame)
                    ids = torch.as_tensor(np.flatnonzero(self.repeat_active), device=self.device)
                    q, v = read_state(self, backend)
                    pending = cpu(pending_commands(self.engine.policy_control))
                    active_targets = cpu(self.engine.policy_control.active)
                    before_obs = cpu(super().observations())
                    target = cpu(self.engine.get_body_pos(0))[self.repeat_active, rig.model.body('Chest').id - 1]
                    p, velocity = launch_state(target, [signs[index], 0., 0.], speed,
                                               rig.model.opt.gravity, cfg['launch_distance_m'])
                    if len(ids):
                        self.engine.launch_ball(ids, p, velocity)
                    insertion = []
                    if backend == 'native':
                        for case in ids.cpu().tolist():
                            data = self.engine.datas[case]
                            for contact in data.contact[:data.ncon]:
                                a, b = self.engine.rig.model.geom_bodyid[contact.geom]
                                body = b if a == self.engine.ball_body else a if b == self.engine.ball_body else 0
                                if 0 < body < self.engine.ball_body:
                                    insertion.append(dict(case=case, body=self.engine.rig.model.body(body).name,
                                                          distance_m=float(contact.dist)))
                    if backend == 'native':
                        ball_q = np.stack([d.qpos[46:].copy() for d in self.engine.datas])
                        ball_v = np.stack([d.qvel[45:].copy() for d in self.engine.datas])
                    else:
                        ball_q, ball_v = (cpu(t) for t in self.engine.ball_state())
                    position_error = float(np.abs(ball_q[self.repeat_active, :3] - p).max(initial=0.))
                    velocity_error = float(np.abs(ball_v[self.repeat_active, :3] - velocity).max(initial=0.))
                    if position_error > 2e-6 or velocity_error > 2e-6:
                        raise RuntimeError('Repeated projectile state differs from requested launch')
                    after_q, after_v = read_state(self, backend)
                    preserved = all(np.array_equal(a, b) for a, b in (
                        (q[:, :46], after_q[:, :46]), (v[:, :45], after_v[:, :45]),
                        (pending, cpu(pending_commands(self.engine.policy_control))),
                        (active_targets, cpu(self.engine.policy_control.active))))
                    if not preserved:
                        raise RuntimeError('Repeated launch changed character state or pending commands')
                    after_obs = cpu(super().observations())
                    observation_error = float(np.abs(before_obs[self.repeat_active] - after_obs[self.repeat_active]).max(initial=0.))
                    if observation_error > 2e-6:
                        raise RuntimeError('Repeated launch changed character observations')
                    self.repeat_events.append(dict(frame=frame, index=index, sign=signs[index],
                        active_cases=ids.cpu().tolist(), character_preserved=preserved,
                        observation_error=observation_error, position_error=position_error,
                        velocity_error=velocity_error, position=p.tolist(), velocity=velocity.tolist()))
                    self.repeat_events[-1]['native_insertion_contacts'] = insertion
                return super().observations()

            def step(self, action):
                self.repeat_inside_step = True
                try:
                    result = super().step(action)
                finally:
                    self.repeat_inside_step = False
                self.repeat_active &= cpu(result[2]) == 0
                self.repeat_frame += 1
                return result

        with patch.object(module, name, RepeatedTask):
            result = original(rig, reference, actor, world, out, cfg, speed, sign)
        with np.load(out / 'trace.npz', allow_pickle=False) as data:
            trace = {k: data[k] for k in data.files}
        task = journal['task']
        # The inherited trace marks the initial launch; later launches are explicit here.
        launched = trace['launched'].copy()
        for event in task.repeat_events:
            launched[event['frame'], event['active_cases']] = True
        expected = np.zeros_like(launched)
        for t in schedule:
            frame = int(np.ceil(t / task.dt))
            if frame < len(expected):
                expected[frame] = trace['alive_before'][frame]
        if not np.array_equal(launched, expected):
            raise RuntimeError('Repeated launch schedule differs from live-case plan')
        np.savez_compressed(out / 'repeat-launches.npz', launched=launched)
        rows = repeated_summary(trace, schedule, task.dt)
        for row, repeated in zip(result['cases'], rows, strict=True):
            row['single_projectile_summary_recovery_pass'] = row['recovery_pass']
            row['recovery_pass'] = repeated['passed'] and row['survived']
            row['encounters'] = repeated['encounters']
        result['recovery_passes'] = sum(r['recovery_pass'] for r in result['cases'])
        result['recovery_scope'] = 'Every scheduled encounter must hit, retain its prequalification, recover for two seconds within two seconds after last contact, and end valid; final trial must survive'
        result['launch_times_seconds'] = schedule
        result['launch_signs'] = signs
        result['native_repeat_launch_clear'] = (all(c['distance_m'] >= -1e-9
            for event in task.repeat_events for c in event['native_insertion_contacts'])
            if backend == 'native' else None)
        save(out / 'repeat-events.json', dict(events=task.repeat_events,
            scope='Reuse one projectile: its old state is replaced at each subsequent launch. Not a multi-projectile Arena simulation.'))
        save(out / 'report.json', result)
        print(dict(repeated_case=str(out), recoveries=result['recovery_passes']), flush=True)
        return result
    return evaluate


def run(args):
    settings = read(args.protocol)
    if settings.get('direction_degrees', 0.) != 0. or settings.get('feedback', 'live') != 'live':
        raise ValueError('Repeated protocol supports live feedback and X-axis throws only')
    protect, compare = screen.protect_used, screen.compare

    def protected(out, paths, protocol, cfg):
        return protect(out, [*paths, Path(__file__)], protocol, cfg)

    def compare_encounters(native, gpu, dt):
        result = compare(native, gpu, dt)
        rows = []
        for a, b in zip(native['cases'], gpu['cases'], strict=True):
            for x, y in zip(a['encounters'], b['encounters'], strict=True):
                exact = all(x[k] == y[k] for k in ('hit', 'prequalified', 'survived', 'final_valid', 'passed'))
                timing = all((x[k] is None and y[k] is None) or (x[k] is not None and y[k] is not None
                    and abs(x[k] - y[k]) <= 2 * dt) for k in ('first_contact_seconds', 'last_contact_seconds'))
                impulse = abs(x['total_normal_impulse_ns'] - y['total_normal_impulse_ns']) <= .25 + .1 * x['total_normal_impulse_ns']
                rows.append(dict(case=a['case'], encounter=x['index'], passed=bool(exact and timing and impulse)))
        result['encounters'] = rows
        result['native_repeat_launch_clear'] = native['native_repeat_launch_clear']
        result['passed'] &= all(r['passed'] for r in rows) and result['native_repeat_launch_clear']
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(screen, 'protect_used', protected))
        stack.enter_context(patch.object(screen, 'compare', compare_encounters))
        stack.enter_context(patch.object(screen, 'native_evaluate', wrap_evaluator(screen.native_evaluate, 'native', settings)))
        stack.enter_context(patch.object(screen, 'gpu_evaluate', wrap_evaluator(screen.gpu_evaluate, 'gpu', settings)))
        followup.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
