"""Bounded native single-ball screen; no learning or automatic promotion."""
import argparse
from pathlib import Path
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .ball import launch_state
from .evaluate_stand_sustained import duration_summary
from .replicate_stand_representations import cpu, gates, pending_commands, tensor_hash, verify_control_trace
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor
from .verify_stand_learning_curve import verify_gates
from .rig import Rig


def summarize(trace, dt, seconds, onset, quiet):
    result = duration_summary(trace, dt, seconds)
    required = int(np.ceil(2./dt))
    launch_frame = int(np.ceil(onset/dt))
    for case, row in enumerate(result['cases']):
        live = trace['alive_before'][:, case]
        valid = (trace['balance'][:, case] & trace['posture'][:, case]
                 & trace['pre_balance'][:, case] & trace['pre_posture'][:, case])
        contacts = np.flatnonzero(trace['normal_impulse_ns'][:, case] > 0.)
        first, last = (int(contacts[0]), int(contacts[-1])) if len(contacts) else (None, None)
        # All contacts from this single projectile must end before recovery begins.
        start = None
        if last is not None:
            for index in range(last+1, min(last+required+2, len(live))):
                end = index+required
                if end <= len(live) and live[index:end].all() and valid[index:end].all():
                    start = index
                    break
        pre_end = first if first is not None else launch_frame
        prequalified = bool(live[:pre_end+1].all() and valid[:pre_end].all())
        count = int(live.sum())
        final_valid = count >= required and bool(valid[count-required:count].all())
        passed = bool(row['survived'] and prequalified and final_valid and start is not None)
        row.update(hit=first is not None, prequalified=prequalified,
            first_contact_seconds=None if first is None else float(trace['time_s'][first]),
            last_contact_seconds=None if last is None else float(trace['time_s'][last]),
            total_normal_impulse_ns=float(trace['normal_impulse_ns'][:, case].sum()),
            impulse_vector_ns=trace['contact_impulse_ns'][:, case].sum(axis=0).tolist(),
            recovery_window_start=None if start is None else float(trace['time_s'][start]),
            recovery_pass=bool(row['standing_pass']) if quiet else passed)
    result.update(hit_cases=sum(c['hit'] for c in result['cases']),
        recovery_passes=sum(c['recovery_pass'] for c in result['cases']),
        recovery_scope='Physical contact required; survive; valid before first contact; start two valid seconds within two seconds after final ball contact, plus one control-frame tolerance; valid final two seconds',
        quiet_control=quiet)
    return result


@torch.no_grad()
def evaluate(rig, reference, actor, ball_path, out, settings, speed, sign):
    from envs.base_env import EnvMode
    from .stand_ball_native import StandBallTask
    seconds, onset = settings['seconds'], settings['launch_seconds']
    task = StandBallTask(rig, reference, ball_path=ball_path, seconds=seconds)
    task.set_mode(EnvMode.TEST)
    task.reset()
    out.mkdir()
    contract = task.observation_contract()
    contract['ball_world_sha256'] = digest(ball_path)
    save(out/'contract.json', contract)
    actor_hash = tensor_hash(actor.state_dict())
    rows, active = [], np.ones(8, dtype=bool)
    launch_frame = int(np.ceil(onset/task.dt))
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    qi = rig.model.jnt_qposadr[joints]
    for frame in range(int(np.ceil(seconds/task.dt))):
        engine, c = task.engine, task.engine.policy_control
        engine.active = active.copy()
        launch = np.zeros(8, dtype=bool)
        if speed > 0 and frame == launch_frame:
            launch = active.copy()
            ids = torch.as_tensor(np.flatnonzero(launch))
            target = cpu(engine.get_body_pos(0))[launch, rig.model.body('Chest').id-1]
            position, velocity = launch_state(target, [sign, 0., 0.], speed,
                rig.model.opt.gravity, settings['launch_distance_m'])
            engine.launch_ball(ids, position, velocity)
        obs = task.observations()
        q = np.stack([d.qpos.copy() for d in engine.datas])
        v = np.stack([d.qvel.copy() for d in engine.datas])
        pending = cpu(pending_commands(c))
        before, before_measurements = gates(task)
        action = actor(obs)
        action[torch.as_tensor(~active)] = 0.
        if not torch.isfinite(action).all() or not torch.isfinite(obs[torch.as_tensor(active)]).all():
            raise RuntimeError('Nonfinite actor input/output')
        rq, _ = task.sample_reference(task.offset+task.steps*task.dt)
        _, _, done, _ = task.step(action)
        after, measurements = gates(task)
        row = dict(alive_before=active.copy(), observation=cpu(obs), action=cpu(action),
            qpos_before=q[:, :46], qvel_before=v[:, :45],
            qpos_after=np.stack([d.qpos[:46].copy() for d in engine.datas]),
            qvel_after=np.stack([d.qvel[:45].copy() for d in engine.datas]),
            ball_qpos_before=q[:, 46:], ball_qvel_before=v[:, 45:], launched=launch,
            ball_qpos_after=np.stack([d.qpos[46:].copy() for d in engine.datas]),
            ball_qvel_after=np.stack([d.qvel[45:].copy() for d in engine.datas]),
            pending_before=pending, issued_entry=cpu(c.pending[(c.cursor-1)%2]),
            reference_motor_q=cpu(rq[:, qi]), active_targets=cpu(c.active),
            substep_q=cpu(engine.substep_q), substep_v=cpu(engine.substep_v),
            substep_torque=cpu(engine.substep_torque), done=cpu(done),
            solver_warning=engine.solver_warning.copy(),
            balance=cpu(after['balance']), posture=cpu(after['posture']),
            pre_balance=cpu(before['balance']), pre_posture=cpu(before['posture']),
            normal_impulse_ns=engine.normal_impulse.copy(), contact_impulse_ns=engine.contact_impulse.copy())
        row.update({key: cpu(value) for key, value in measurements.items()})
        row.update({f'pre_{key}': cpu(value) for key, value in before_measurements.items()})
        rows.append(row)
        if engine.solver_warning[active].any() or not np.isfinite(row['qpos_after'][active]).all():
            np.savez_compressed(out/'failed-trace.npz', **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            raise RuntimeError('Native integration warning/nonfinite state')
        active &= cpu(done) == 0
        if not active.any():
            break
    trace = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    trace['time_s'] = (np.arange(len(rows))+1)*task.dt
    np.savez_compressed(out/'trace.npz', **trace)
    # Verify persisted evidence, not merely live arrays.
    with np.load(out/'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k] for k in saved.files}
    if tensor_hash(actor.state_dict()) != actor_hash:
        raise RuntimeError('Frozen actor changed')
    assert contract['num_obs'] == 374 and contract['passive_joint_stiffness'] == 0
    assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
    assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
    assert all(np.isfinite(trace[k]).all() for k in trace)
    assert (trace['normal_impulse_ns'] >= 0.).all()
    assert not trace['solver_warning'].any()
    assert not trace['launched'][:launch_frame].any()
    assert not trace['launched'][launch_frame+1:].any()
    if speed == 0:
        assert not trace['launched'].any() and not trace['normal_impulse_ns'].any()
    elif len(rows) > launch_frame:
        assert np.array_equal(trace['launched'][launch_frame], trace['alive_before'][launch_frame])
    for case in range(8):
        count = int(trace['alive_before'][:, case].sum())
        assert trace['alive_before'][:count, case].all()
        assert not trace['alive_before'][count:, case].any()
        assert (trace['done'][:count-1, case] == 0).all()
        assert trace['done'][count-1, case] in (1, 3)
    verify_gates(trace, '')
    verify_gates(trace, 'pre_')
    error = verify_control_trace(trace, contract)
    report = dict(**summarize(trace, task.dt, seconds, onset, speed == 0),
        actor_hash=actor_hash, max_control_formula_error_nm=error,
        backend='native', speed_m_s=speed, sign=sign,
        ball_mass_kg=float(task.engine.rig.model.body_mass[task.engine.ball_body]))
    save(out/'report.json', report)
    print(dict(case=str(out), hits=report['hit_cases'], recoveries=report['recovery_passes']), flush=True)
    return report


def run(args):
    settings = read(args.protocol)
    if settings['schema'] != 'stand_ball_boundary_v1' or settings['seconds'] != 12. \
            or settings['launch_seconds'] != 5. or settings['seeds'] != [301027, 301028] \
            or settings['mass_kg'] != [1., 2., 5., 10.] or settings['speed_m_s'] != [1., 2., 4.] \
            or settings['launch_distance_m'] != .65:
        raise ValueError('Changed experiment requires a successor protocol')
    activate(args.mimickit)
    from .stand_ball_native import write_world
    torch.set_num_threads(1)
    roots = [Path(settings['sources'][str(seed)]) for seed in settings['seeds']]
    extra = [args.protocol, Path(settings['ball_template'])]
    for root in roots:
        hashes = read(root/'final-verification.json')['output_hashes']
        protected = {str(Path(path).resolve()): value for path, value in hashes.items()}
        for path in (root/'final/model.pt', root/'final/model.amp.pt'):
            if protected.get(str(path.resolve())) != digest(path):
                raise ValueError(f'Checkpoint differs from completed pilot evidence: {path}')
        extra.extend([root/'protocol.json', root/'final-verification.json', root/'final/model.pt', root/'final/model.amp.pt'])
    source_paths = [Path(__file__), Path(__file__).with_name('stand_ball_native.py'),
                    Path(__file__).with_name('ball.py'),
                    Path(__file__).parent/'scripts/evaluate-stand-ball-boundary.ps1']
    inputs, sources = protect(Path(settings['prior']), args.out, source_paths, extra_inputs=extra)
    parent = read(roots[0]/'protocol.json')
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=parent['config'], demonstrations=parent['demonstrations'],
        model_sha256=parent['model_sha256'], reference_sha256=parent['reference_sha256'],
        training_updates=0, maximum_cases=400, automatic_promotion=False))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    worlds = {}
    for mass in settings['mass_kg']:
        path = args.out/f'ball-{mass:g}kg.xml'
        write_world(rig.path, settings['ball_template'], path, mass)
        worlds[mass] = path
    results = {}
    for seed, root in zip(settings['seeds'], roots):
        actor = load_actor(rig, reference, read(root/'protocol.json'), root/'final/model.pt')
        quiet = evaluate(rig, reference, actor, worlds[1.], args.out/f'{seed}-quiet', settings, 0., 1)
        if quiet['standing_passes'] != 8 or quiet['hit_cases']:
            raise RuntimeError('Parked-ball quiet admission failed; stop before impacts')
        levels = {}
        for mass in settings['mass_kg']:
            for speed in settings['speed_m_s']:
                label = f'{mass:g}kg-{speed:g}mps'
                levels[label] = [evaluate(rig, reference, actor, worlds[mass],
                    args.out/f'{seed}-{label}-{sign:+d}', settings, speed, sign) for sign in (1, -1)]
        results[str(seed)] = dict(quiet=quiet, levels=levels)
    # A candidate must contain successes AND failures for both independent actors.
    # Tie-break by nominal horizontal kinetic energy; ballistic vertical speed differs.
    candidates = []
    for mass in settings['mass_kg']:
        for speed in settings['speed_m_s']:
            label = f'{mass:g}kg-{speed:g}mps'
            rows = [results[str(seed)]['levels'][label] for seed in settings['seeds']]
            if all(sum(r['hit_cases'] for r in pair) == 16
                   and 0 < sum(r['recovery_passes'] for r in pair) < 16 for pair in rows):
                candidates.append((mass*speed, .5*mass*speed**2, mass, speed, label))
    candidate = min(candidates)[-1] if candidates else None
    save(args.out/'report.json', dict(results=results, native_boundary_candidate=candidate,
        training_updates=0, promoted=False, gpu_parity_verified=False,
        next_gate='Verify the ball adapter on GPU and actual matched starts before a separately authorized training pilot'))
    seal(args.out, inputs, sources, verified=True, training_updates=0, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.exists() and not (args.out/'final-verification.json').exists():
            failure = args.out/'failure.json'
            if not failure.exists():
                save(failure, dict(error=traceback.format_exc(), training_updates=0))
        raise
