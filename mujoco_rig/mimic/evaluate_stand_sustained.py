"""Frozen-policy duration assessments with full-interval Stand gates."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .replicate_stand_representations import (read_state, cpu, pending_commands, gates,
    tensor_hash, verify_control_trace, verify_partial_reset)
from .audit_stand_failures import read, save, longest
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, seal, load_actor
from .verify_stand_learning_curve import verify_gates


def duration_summary(trace, dt, seconds):
    cases = []
    for case in range(8):
        live = trace['alive_before'][:, case]
        end = int(np.flatnonzero(live)[-1])
        survived = bool(trace['done'][end, case] == 3 and trace['time_s'][end] >= seconds)
        balanced = bool(survived and trace['balance'][live, case].all() and trace['pre_balance'][live, case].all())
        valid = live & trace['balance'][:, case] & trace['posture'][:, case]
        both_valid = valid & trace['pre_balance'][:, case] & trace['pre_posture'][:, case]
        departed = np.flatnonzero(live & ~both_valid)
        standing = bool(balanced and trace['posture'][live, case].all() and trace['pre_posture'][live, case].all())
        cases.append(dict(case=case, survived=survived, balance_pass=balanced, standing_pass=standing,
            survival_seconds=float(trace['time_s'][end]),
            first_departure_seconds=float(trace['time_s'][departed[0]]) if len(departed) else None,
            longest_valid_seconds=longest(valid)*dt, terminal_done=int(trace['done'][end, case])))
    return dict(cases=cases, standing_passes=sum(c['standing_pass'] for c in cases),
        survivors=sum(c['survived'] for c in cases), minimum_survival_seconds=min(c['survival_seconds'] for c in cases),
        mean_survival_seconds=float(np.mean([c['survival_seconds'] for c in cases])),
        mean_longest_valid_seconds=float(np.mean([c['longest_valid_seconds'] for c in cases])),
        requested_seconds=seconds, pass_scope='Every pre/post control-frame balance and posture gate')


@torch.no_grad()
def evaluate_duration(rig, reference, mode, backend, actor, out, *, seconds, reset_probe=False):
    from envs.base_env import EnvMode
    from .representation_task import RepresentationTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    task = RepresentationTask(rig, reference, 8, device, representation=mode, trace=True,
        reward_mode='reference_v9', episode_seconds=seconds)
    task.set_mode(EnvMode.TEST)
    task.reset()
    out.mkdir()
    contract = task.observation_contract()
    save(out/'contract.json', contract)
    before = None
    if actor is not None:
        actor.to(device).eval()
        before = tensor_hash(actor.state_dict())
    active, rows = np.ones(8, dtype=bool), []
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    qindices = rig.model.jnt_qposadr[joints]
    for frame in range(int(np.ceil(seconds/task.dt))):
        if backend == 'native':
            task.engine.active = active.copy()
        obs = task.observations()
        q, v = read_state(task, backend)
        c = task.engine.policy_control
        pending = cpu(pending_commands(c))
        flags_before, measurements_before = gates(task)
        assert torch.isfinite(obs[torch.as_tensor(active, device=device)]).all()
        action = torch.zeros((8, 33), device=device) if actor is None else actor(obs)
        action[torch.as_tensor(~active, device=device)] = 0
        assert torch.isfinite(action).all() and (action.abs() <= 1).all()
        reference_q, _ = task.sample_reference(task.offset+task.steps*task.dt)
        _, _, done, _ = task.step(action)
        flags, measurements = gates(task)
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.as_tensor(warnings, device=device)] = 1
        after_q, after_v = read_state(task, backend)
        row = dict(alive_before=active.copy(), observation=cpu(obs), action=cpu(action),
            qpos_before=q, qvel_before=v, qpos_after=after_q, qvel_after=after_v,
            pending_before=pending, issued_entry=cpu(c.pending[(c.cursor-1)%2]),
            reference_motor_q=cpu(reference_q[:, qindices]),
            substep_q=cpu(task.engine.substep_q), substep_v=cpu(task.engine.substep_v),
            substep_torque=cpu(task.engine.substep_torque), done=cpu(done), solver_warning=warnings,
            balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            pre_balance=cpu(flags_before['balance']), pre_posture=cpu(flags_before['posture']))
        if mode == 'target_pd':
            row['active_targets'] = cpu(c.active)
        row.update({key: cpu(value) for key, value in measurements.items()})
        row.update({f'pre_{key}': cpu(value) for key, value in measurements_before.items()})
        rows.append(row)
        if warnings[active].any() or not np.isfinite(after_q[active]).all() or not np.isfinite(after_v[active]).all():
            np.savez_compressed(out/'failed-trace.npz', **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
            raise RuntimeError('Native warning/nonfinite admission failure')
        active &= cpu(done) == 0
        if not active.any():
            break
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays['time_s'] = (np.arange(len(rows))+1)*task.dt
    np.savez_compressed(out/'trace.npz', **arrays)
    formula_error = verify_control_trace(arrays, contract)
    reset_result = verify_partial_reset(task, backend, out) if reset_probe else None
    if actor is not None:
        assert tensor_hash(actor.state_dict()) == before
        actor.cpu()
    report = dict(**duration_summary(arrays, task.dt, seconds), representation=mode, backend=backend,
        zero_action_baseline=actor is None, actor_hash=before, passive_springs=False,
        active_pose_feedback=mode == 'target_pd', inverse_contact_teacher=False, external_support=False,
        live_control_intervals=int(arrays['alive_before'].sum()), max_control_formula_error_nm=formula_error,
        solver_warnings=0 if backend == 'native' else None, partial_reset=reset_result)
    save(out/'report.json', report)
    print(dict(evaluation=str(out), passes=report['standing_passes'], mean_survival=report['mean_survival_seconds']), flush=True)
    return report


def verify_duration(folder, seconds):
    contract, report = read(folder/'contract.json'), read(folder/'report.json')
    with np.load(folder/'trace.npz', allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    live, dt = trace['alive_before'], contract['control_timestep']
    assert live.shape[1] == 8 and len(live) <= int(np.ceil(seconds/dt))
    assert contract['representation'] == 'target_pd' and contract['num_obs'] == 374
    assert contract['passive_joint_stiffness'] == 0 and contract['standing_pose_servo']
    assert np.array_equal(trace['time_s'], (np.arange(len(live))+1)*dt)
    assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
    assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
    assert all(np.isfinite(trace[key][live]).all() for key in
        ('observation', 'action', 'qpos_before', 'qvel_before', 'qpos_after', 'qvel_after'))
    assert not trace['solver_warning'][live].any()
    verify_gates(trace, '')
    verify_gates(trace, 'pre_')
    assert verify_control_trace(trace, contract) == report['max_control_formula_error_nm']
    assert all(report[key] == value for key, value in duration_summary(trace, dt, seconds).items())
    for case in range(8):
        count = int(live[:, case].sum())
        assert live[:count, case].all() and not live[count:, case].any()
        assert (trace['done'][:count-1, case] == 0).all()
        assert trace['done'][count-1, case] in (1, 3)
        if trace['done'][count-1, case] == 3:
            assert count == int(np.ceil(seconds/dt))
    return trace, report


def run(args):
    assert 12 < args.seconds <= 180
    activate(args.mimickit)
    torch.set_num_threads(1)
    inputs, sources = protect(args.prior, args.out, [Path(__file__)])
    assert str(args.checkpoint.resolve()) in {str(Path(p).resolve()) for p in inputs}
    prior = read(args.prior/'protocol.json')
    protocol = dict(inputs=inputs, sources=sources, config=prior['config'],
        demonstrations=prior['demonstrations'], model_sha256=prior['model_sha256'],
        reference_sha256=prior['reference_sha256'], checkpoint=str(args.checkpoint),
        seconds=args.seconds, maximum_cases=16, training_updates=0,
        initial_state_action_absolute_tolerance=1e-6,
        scope='Frozen original-start assessment; every sampled pre/post gate must pass')
    save(args.out/'protocol.json', protocol)
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    actor = load_actor(rig, reference, protocol, args.checkpoint)
    results, comparisons = {}, {}
    for backend in ('native', 'gpu'):
        results[backend] = evaluate_duration(rig, reference, 'target_pd', backend,
            actor, args.out/backend, seconds=args.seconds)
        trace, _ = verify_duration(args.out/backend, args.seconds)
        with np.load(args.checkpoint.parent/backend/'trace.npz', allow_pickle=False) as original:
            comparisons[backend] = {key: float(np.abs(trace[key][0]-original[key][0]).max())
                for key in ('qpos_before', 'qvel_before', 'action')}
        assert all(value <= 1e-6 for value in comparisons[backend].values())
        assert results[backend]['actor_hash'] == read(args.checkpoint.parent/backend/'report.json')['actor_hash']
    report = dict(results=results, first_frame_differences=comparisons,
        full_interval_stand=all(r['standing_passes'] == 8 for r in results.values()),
        training_updates=0, new_cases=16, teacher_takeover=False, arena_changed=False)
    save(args.out/'report.json', report)
    seal(args.out, inputs, sources, verified=True, cases=16,
        full_interval_stand=report['full_interval_stand'], physics_gates_and_control_verified=True,
        actor_and_normalizers_unchanged=True, training_updates=0,
        additional_verification_physics_steps=0, additional_verification_forward_passes=0)
    print(dict(completed=True, full_interval_stand=report['full_interval_stand'],
        passes={b: r['standing_passes'] for b, r in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())

