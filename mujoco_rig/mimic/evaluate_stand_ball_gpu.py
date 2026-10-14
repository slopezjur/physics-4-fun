"""Frozen GPU evidence using the sealed native screen's recovery definition."""
import numpy as np
import torch

from .audit_stand_failures import digest, save
from .ball import launch_state
from .evaluate_stand_ball_boundary import summarize
from .replicate_stand_representations import cpu, gates, pending_commands, read_state, tensor_hash, verify_control_trace
from .verify_stand_learning_curve import verify_gates


@torch.no_grad()
def evaluate(rig, reference, actor, ball_path, out, settings, speed, sign):
    from envs.base_env import EnvMode
    from .stand_ball_task import StandBallGPUTask
    task = StandBallGPUTask(rig, reference, ball_path=ball_path, seconds=settings['seconds'])
    task.set_mode(EnvMode.TEST)
    task.reset()
    out.mkdir()
    contract = task.observation_contract()
    contract.update(ball_world_sha256=digest(ball_path), ball_angular_velocity_frame='world')
    save(out/'contract.json', contract)
    actor.to('cuda:0').eval()
    actor_hash = tensor_hash(actor.state_dict())
    rows, active = [], np.ones(8, dtype=bool)
    launch_frame = int(np.ceil(settings['launch_seconds']/task.dt))
    qi = rig.model.jnt_qposadr[rig.model.actuator_trnid[rig.actuators, 0]]
    for frame in range(int(np.ceil(settings['seconds']/task.dt))):
        engine, c = task.engine, task.engine.policy_control
        launch = np.zeros(8, dtype=bool)
        if speed > 0 and frame == launch_frame:
            launch = active.copy()
            ids = torch.as_tensor(np.flatnonzero(launch), device=task.device)
            target = cpu(engine.get_body_pos(0))[launch, rig.model.body('Chest').id-1]
            position, velocity = launch_state(target, [sign, 0., 0.], speed,
                rig.model.opt.gravity, settings['launch_distance_m'])
            engine.launch_ball(ids, position, velocity)
        obs = task.observations()
        q, v = read_state(task, 'gpu')
        bq, bv = (cpu(t) for t in engine.ball_state())
        pending = cpu(pending_commands(c))
        before, before_measurements = gates(task)
        action = actor(obs)
        action[torch.as_tensor(~active, device=task.device)] = 0.
        if not torch.isfinite(action).all() or not torch.isfinite(obs).all():
            raise RuntimeError('Nonfinite GPU actor input/output')
        rq, _ = task.sample_reference(task.offset+task.steps*task.dt)
        _, _, done, _ = task.step(action)
        after, measurements = gates(task)
        aq, av = read_state(task, 'gpu')
        abq, abv = (cpu(t) for t in engine.ball_state())
        row = dict(alive_before=active.copy(), observation=cpu(obs), action=cpu(action),
            qpos_before=q, qvel_before=v, qpos_after=aq, qvel_after=av,
            ball_qpos_before=bq, ball_qvel_before=bv, launched=launch,
            ball_qpos_after=abq, ball_qvel_after=abv,
            pending_before=pending, issued_entry=cpu(c.pending[(c.cursor-1)%2]),
            reference_motor_q=cpu(rq[:, qi]), active_targets=cpu(c.active),
            substep_q=cpu(engine.substep_q), substep_v=cpu(engine.substep_v),
            substep_torque=cpu(engine.substep_torque), done=cpu(done),
            balance=cpu(after['balance']), posture=cpu(after['posture']),
            pre_balance=cpu(before['balance']), pre_posture=cpu(before['posture']),
            normal_impulse_ns=cpu(engine.normal_impulse), contact_impulse_ns=cpu(engine.contact_impulse))
        row.update({key: cpu(value) for key, value in measurements.items()})
        row.update({f'pre_{key}': cpu(value) for key, value in before_measurements.items()})
        # GPU keeps integrating terminated peers; they are outside the trial thereafter.
        row['normal_impulse_ns'][~active] = 0.
        row['contact_impulse_ns'][~active] = 0.
        rows.append(row)
        if not all(np.isfinite(value).all() for value in row.values()):
            np.savez_compressed(out/'failed-trace.npz', **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            raise RuntimeError('Nonfinite GPU trajectory')
        active &= cpu(done) == 0
        if not active.any():
            break
    trace = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    trace['time_s'] = (np.arange(len(rows))+1)*task.dt
    np.savez_compressed(out/'trace.npz', **trace)
    with np.load(out/'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k] for k in saved.files}
    assert tensor_hash(actor.state_dict()) == actor_hash
    assert contract['num_obs'] == 374 and contract['passive_joint_stiffness'] == 0
    assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
    assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
    assert (trace['normal_impulse_ns'] >= 0.).all()
    if speed == 0:
        assert not trace['launched'].any() and not trace['normal_impulse_ns'].any()
    else:
        assert not trace['launched'][:launch_frame].any() and not trace['launched'][launch_frame+1:].any()
        if len(rows) > launch_frame:
            assert np.array_equal(trace['launched'][launch_frame], trace['alive_before'][launch_frame])
    for case in range(8):
        count = int(trace['alive_before'][:, case].sum())
        assert trace['alive_before'][:count, case].all() and not trace['alive_before'][count:, case].any()
        assert (trace['done'][:count-1, case] == 0).all() and trace['done'][count-1, case] in (1, 3)
    verify_gates(trace, '')
    verify_gates(trace, 'pre_')
    error = verify_control_trace(trace, contract)
    report = dict(**summarize(trace, task.dt, settings['seconds'], settings['launch_seconds'], speed == 0),
        actor_hash=actor_hash, max_control_formula_error_nm=error, backend='gpu',
        speed_m_s=speed, sign=sign, solver_warnings_available=False,
        ball_mass_kg=float(task.engine.rig.model.body_mass[task.engine.ball_body]))
    save(out/'report.json', report)
    actor.cpu()
    print(dict(case=str(out), hits=report['hit_cases'], recoveries=report['recovery_passes']), flush=True)
    return report


def compare(native, gpu, dt):
    """Predeclared admission: same outcomes; contact timing/impulse tolerances only."""
    cases = []
    for a, b in zip(native['cases'], gpu['cases'], strict=True):
        exact = all(a[key] == b[key] for key in ('case', 'hit', 'survived', 'prequalified', 'recovery_pass'))
        timing = all((a[key] is None and b[key] is None) or
                     (a[key] is not None and b[key] is not None and abs(a[key]-b[key]) <= 2*dt)
                     for key in ('first_contact_seconds', 'last_contact_seconds'))
        impulse = abs(a['total_normal_impulse_ns']-b['total_normal_impulse_ns'])
        limit = .25 + .10*abs(a['total_normal_impulse_ns'])
        cases.append(dict(case=a['case'], passed=bool(exact and timing and impulse <= limit),
            outcomes_identical=exact, contact_timing_pass=timing, normal_impulse_difference_ns=impulse,
            normal_impulse_tolerance_ns=limit))
    return dict(passed=all(c['passed'] for c in cases), cases=cases,
        contact_time_tolerance_seconds=2*dt, normal_impulse_atol_ns=.25, normal_impulse_rtol=.10)
