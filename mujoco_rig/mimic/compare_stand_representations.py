"""Bounded controller admission and matched fresh torque/target PPO comparison."""
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
from .audit_stand_failures import digest, longest, read, save
from .capture_stand_student_states import cpu, gates, pending_commands
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


PRIOR = Path('logs/mimickit-training/stand-current-recovery-replication-20261003-01')
TEACHER = Path('logs/mimickit-training/stand-teacher-20261001-01')
CORRECTIONS = Path('logs/mimickit-training/stand-corrections-20261001-01')
CONFIG = Path('logs/mimickit-training/stand-ppo-critic-lr1em3-30min-20261002-01/agent.yaml')
SEED = 301015
SECONDS = 180.
ASSESS_SECONDS = 12.
TORQUE_TOLERANCE = 2e-4


def protect(out):
    prior, complete = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = {**prior['inputs'], **complete['output_hashes']}
    protected[str(PRIOR/'final-verification.json')] = digest(PRIOR/'final-verification.json')
    paths = [Path(__file__), *(Path(__file__).parent/name for name in
        ('target_control33.py', 'representation_engine.py', 'representation_task.py'))]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    protected[str(CONFIG)] = digest(CONFIG)
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    check_protected(protected, [])
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources


def check_protected(protected, sources):
    for path, expected in protected.items():
        assert digest(Path(path)) == expected, path
    for source in sources:
        for key in ('path', 'snapshot'):
            assert digest(Path(source[key])) == source['sha256']


def summarize_trace(trace, dt):
    cases = []
    for case in range(8):
        live = trace['alive_before'][:, case]
        end = int(np.flatnonzero(live)[-1])
        survived = bool(trace['done'][end, case] == 3)
        final = live & (trace['time_s'] > trace['time_s'][end]-2.)
        balanced = bool(survived and trace['time_s'][end] >= ASSESS_SECONDS and trace['balance'][final, case].all())
        standing = bool(balanced and trace['posture'][final, case].all())
        valid = live & trace['balance'][:, case] & trace['posture'][:, case]
        departed = np.flatnonzero(live & ~valid)
        cases.append(dict(case=case, survived=survived, balance_pass=balanced, standing_pass=standing,
            survival_seconds=float(trace['time_s'][end]),
            first_departure_seconds=float(trace['time_s'][departed[0]]) if len(departed) else None,
            longest_valid_seconds=longest(valid)*dt, terminal_done=int(trace['done'][end, case])))
    departures = [row['first_departure_seconds'] for row in cases if row['first_departure_seconds'] is not None]
    return dict(cases=cases, standing_passes=sum(row['standing_pass'] for row in cases),
        survivors=sum(row['survived'] for row in cases),
        minimum_survival_seconds=min(row['survival_seconds'] for row in cases),
        mean_survival_seconds=float(np.mean([row['survival_seconds'] for row in cases])),
        mean_first_departure_seconds=float(np.mean(departures)) if departures else None,
        mean_longest_valid_seconds=float(np.mean([row['longest_valid_seconds'] for row in cases])))


def verify_control_trace(trace, contract):
    """Bind every recorded substep torque to its delayed issued command/target."""
    live = trace['alive_before']
    scale = np.asarray(contract['torque_scale_nm'], dtype=np.float32)
    torques = trace['substep_torque']
    assert torques.shape[1:] == (4, 8, 33)
    mask = np.broadcast_to(live[:, None, :, None], torques.shape)
    assert np.isfinite(torques[mask]).all()
    assert (np.abs(torques) <= scale[None, None, None, :]+TORQUE_TOLERANCE)[mask].all()
    if contract['representation'] == 'target_pd':
        active = trace['active_targets']
        expected_active = np.zeros_like(active)
        expected_active[2:] = trace['issued_entry'][:-2]
        assert np.array_equal(active, expected_active)
        assert not active[:2].any()
        expected = (4.*(active[:, None, :, :33]-trace['substep_q']) +
            .08*(active[:, None, :, 33:66]-trace['substep_v']))
        expected = np.clip(expected, -1, 1)*scale[None, None, None, :]*active[:, None, :, 66:67]
        lower, upper = np.asarray(contract['target_lower_radians']), np.asarray(contract['target_upper_radians'])
        target = np.clip(trace['reference_motor_q']+.25*np.clip(trace['action'], -1, 1), lower, upper)
        assert np.allclose(trace['issued_entry'][..., :33], target, rtol=0, atol=2e-7)
        assert not trace['issued_entry'][..., 33:66].any()
        assert (trace['issued_entry'][..., 66] == 1).all()
        assert np.array_equal(trace['observation'][..., 95:229], trace['pending_before'])
    else:
        held = np.zeros_like(trace['action'])
        held[2:] = trace['action'][:-2]*scale
        expected = np.broadcast_to(held[:, None], torques.shape)
        assert np.array_equal(trace['observation'][..., 95:161], trace['pending_before'])
        assert not trace['observation'][..., 161:229].any()
        assert np.array_equal(trace['issued_entry'], trace['action'])
    difference = float(np.max(np.abs(torques-expected)[mask]))
    assert difference <= TORQUE_TOLERANCE, difference
    # Reconstruct pending queues before each action, including initial invalid/zero slots.
    width = trace['issued_entry'].shape[-1]
    queue = np.zeros((2, 8, width), dtype=np.float32)
    for frame, command in enumerate(trace['issued_entry']):
        assert np.array_equal(trace['pending_before'][frame], np.concatenate(queue, axis=-1))
        queue[0], queue[1] = queue[1].copy(), command
    return difference


def verify_partial_reset(task, backend, out):
    c = task.engine.policy_control
    before_q, before_v = read_state(task, backend)
    before_queue, before_ctrl, cursor = cpu(c.pending), cpu(task.engine.native_ctrl), c.cursor
    before_active = cpu(c.active) if hasattr(c, 'active') else None
    before_contacts = cpu(task.engine.get_ground_contact_forces(0))
    reset = np.array([1, 6])
    keep = np.array([0, 2, 3, 4, 5, 7])
    task.reset(torch.as_tensor(reset, device=task.device))
    after_q, after_v = read_state(task, backend)
    assert np.array_equal(before_q[keep], after_q[keep]) and np.array_equal(before_v[keep], after_v[keep])
    assert np.array_equal(before_queue[:, keep], cpu(c.pending)[:, keep]) and c.cursor == cursor
    assert np.array_equal(before_ctrl[keep], cpu(task.engine.native_ctrl)[keep])
    assert np.array_equal(before_contacts[keep], cpu(task.engine.get_ground_contact_forces(0))[keep])
    assert not cpu(c.pending)[:, reset].any() and not cpu(task.engine.native_ctrl)[reset].any()
    if before_active is not None:
        assert np.array_equal(before_active[keep], cpu(c.active)[keep]) and not cpu(c.active)[reset].any()
    arrays = dict(before_q=before_q, after_q=after_q, before_v=before_v, after_v=after_v,
        before_queue=before_queue, after_queue=cpu(c.pending), before_ctrl=before_ctrl,
        after_ctrl=cpu(task.engine.native_ctrl), before_contacts=before_contacts,
        after_contacts=cpu(task.engine.get_ground_contact_forces(0)), reset=reset, keep=keep)
    if before_active is not None:
        arrays.update(before_active=before_active, after_active=cpu(c.active))
    np.savez_compressed(out/'partial-reset.npz', **arrays)
    return dict(passed=True, reset_cases=reset.tolist(), unchanged_peer_cases=keep.tolist(),
        cursor_before=cursor, cursor_after=c.cursor, extra_physics_steps=0)


def read_state(task, backend):
    if backend == 'native':
        return (np.stack([data.qpos.copy() for data in task.engine.datas]),
                np.stack([data.qvel.copy() for data in task.engine.datas]))
    return gpu_native_state(task)


@torch.no_grad()
def evaluate(rig, reference, mode, backend, actor, out, *, reset_probe=False):
    from envs.base_env import EnvMode
    from .representation_task import RepresentationTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    task = RepresentationTask(rig, reference, 8, device, representation=mode, trace=True,
        reward_mode='reference_v9', episode_seconds=ASSESS_SECONDS)
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
    for frame in range(int(np.ceil(ASSESS_SECONDS/task.dt))):
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
    report = dict(**summarize_trace(arrays, task.dt), representation=mode, backend=backend,
        zero_action_baseline=actor is None, actor_hash=before, passive_springs=False,
        active_pose_feedback=mode == 'target_pd', inverse_contact_teacher=False, external_support=False,
        live_control_intervals=int(arrays['alive_before'].sum()), max_control_formula_error_nm=formula_error,
        solver_warnings=0 if backend == 'native' else None, partial_reset=reset_result)
    save(out/'report.json', report)
    print(dict(evaluation=str(out), passes=report['standing_passes'], mean_survival=report['mean_survival_seconds']), flush=True)
    return report


def formula_probe(rig, out):
    """No physics: shared input parity, saturation, issue sampling and queue resets."""
    import warp as wp
    wp.init()
    from .representation_engine import target_feedback
    from .target_control33 import DelayedTargetControl33
    records = []
    c = DelayedTargetControl33(rig, 8, 'cpu')
    q = torch.as_tensor(np.tile(rig.rest, (8, 1)), dtype=torch.float32)
    v = torch.linspace(-3., 3., 8*rig.model.nv).reshape(8, rig.model.nv)
    actions = torch.linspace(-2., 2., 8*33).reshape(8, 33)
    c.set_reference(q.clone(), torch.zeros_like(v))
    saved_queue = []
    for frame in range(5):
        action = actions if frame % 2 == 0 else -actions
        c.apply(action)
        saved_queue.append(cpu(c.pending[(c.cursor-1)%2]))
        if frame >= 2:
            assert np.array_equal(cpu(c.active), saved_queue[frame-2])
        else:
            assert not c.active.any()
        # Changing the reference after issue must not change its queued target.
        before = c.pending.clone()
        c.reference_q.add_(1e-3)
        assert torch.equal(before, c.pending)
        expected = cpu(c.physics_control(q, v))
        tensors = [q.flatten().cuda(), v.flatten().cuda(), c.active.cuda(), c.indices.to(torch.int32).cuda(),
            c.q_indices.to(torch.int32).cuda(), c.v_indices.to(torch.int32).cuda(), c.scale.cuda(),
            torch.zeros(8*rig.model.nu, device='cuda:0')]
        wp.launch(target_feedback, dim=(8, 33), inputs=[*[wp.from_torch(t) for t in tensors],
            rig.model.nq, rig.model.nv, rig.model.nu, 33, c.gain, c.damping_time])
        actual = cpu(tensors[-1]).reshape(8, rig.model.nu)
        error = float(np.max(np.abs(actual-expected)))
        assert error <= TORQUE_TOLERANCE
        records.append(dict(q=cpu(q), v=cpu(v), targets=cpu(c.active), native=expected, gpu=actual))
    peer = c.pending[:, [0, 2, 3, 4, 5, 7]].clone()
    cursor = c.cursor
    c.reset(torch.tensor([1, 6]))
    assert torch.equal(peer, c.pending[:, [0, 2, 3, 4, 5, 7]]) and c.cursor == cursor
    assert not c.pending[:, [1, 6]].any() and not c.active[[1, 6]].any() and not c.ctrl[[1, 6]].any()
    np.savez_compressed(out/'formula-probe.npz', **{key: np.stack([r[key] for r in records]) for key in records[0]})
    result = dict(passed=True, max_cpu_gpu_torque_difference_nm=float(max(np.max(np.abs(r['native']-r['gpu'])) for r in records)),
        tolerance_nm=TORQUE_TOLERANCE, issue_time_sampling=True, two_interval_delay=True, partial_reset_isolation=True,
        cases=8, command_frames=5, physics_steps=0)
    save(out/'formula-probe.json', result)
    return result


def comparison(candidate, baseline, dt):
    delta = np.asarray([a['survival_seconds']-b['survival_seconds'] for a, b in zip(candidate['cases'], baseline['cases'])])
    mean_gain = candidate['mean_survival_seconds']-baseline['mean_survival_seconds']
    min_gain = candidate['minimum_survival_seconds']-baseline['minimum_survival_seconds']
    survival_gain = bool(mean_gain >= dt-1e-12 and min_gain >= dt-1e-12 and (delta>1e-12).sum()>(delta< -1e-12).sum())
    return dict(paired_survival_change_seconds=delta.tolist(), mean_survival_gain_seconds=mean_gain,
        minimum_survival_gain_seconds=min_gain, improved_cases=int((delta>1e-12).sum()),
        worsened_cases=int((delta< -1e-12).sum()), survival_rule_passed=survival_gain,
        stand_pass_gain=candidate['standing_passes']-baseline['standing_passes'])


def train_arm(rig, reference, clips, config, mode, out, common_hash):
    from .representation_task import AMPRepresentationTask
    from .amp_recovery import validate_sidecar
    task = AMPRepresentationTask(rig, reference, 64, 'cuda:0', representation=mode,
        reward_mode='reference_v9', episode_seconds=60., seed=SEED, demonstrations=clips, demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = build_agent(config, task, 'cuda:0')
    initial_hash = tensor_hash(agent.state_dict())
    assert common_hash is None or initial_hash == common_hash
    assert task.get_obs_space().shape == (374,) and task.get_disc_obs_space().shape == (364,)
    out.mkdir()
    agent.save(str(out/'initial.pt'))
    save(out/'contract.json', task.observation_contract())
    save(out/'initialization.json', dict(seed=SEED, full_tensor_hash=initial_hash, initialize_from=None,
        fresh=True, sampling_seed=SEED+701))
    (out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    initial = {backend: evaluate(rig, reference, mode, backend, StandPolicy(agent), out/f'initial-{backend}')
        for backend in ('native', 'gpu')}
    agent._init_train()
    torch.manual_seed(SEED+701)
    np.random.seed(SEED+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    elapsed, next_output = 0., 30.
    while elapsed < SECONDS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        info = agent._train_iter()
        torch.cuda.synchronize()
        elapsed += time.perf_counter()-started
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
            **{key: float(value) for key, value in info.items()})
        assert all(np.isfinite(v) for v in row.values()) and all(torch.isfinite(p).all() for p in agent.parameters())
        with (out/'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        save(out/'progress.json', row)
        if elapsed >= next_output:
            print(dict(training=mode, seconds=elapsed, updates=agent._iter, transitions=agent._sample_count), flush=True)
            next_output += 30.
    agent.save(str(out/'model.pt'))
    validate_sidecar(torch.load(out/'model.pt', weights_only=True), torch.load(out/'model.amp.pt', weights_only=True))
    final = {backend: evaluate(rig, reference, mode, backend, StandPolicy(agent), out/f'final-{backend}')
        for backend in ('native', 'gpu')}
    report = dict(representation=mode, initial=initial, final=final, initial_full_tensor_hash=initial_hash,
        final_full_tensor_hash=tensor_hash(agent.state_dict()), training_seconds=elapsed,
        ppo_updates=agent._iter, transitions=agent._sample_count, initialize_from=None,
        selection='Final budget checkpoint', periodic_assessments=0)
    save(out/'report.json', report)
    return report


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    protected, sources = protect(args.out)
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(TEACHER/name, args.out/name)
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    assert len(rig.actuators) == 33 and not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    original = read(PRIOR/'contract.json')
    assert digest(rig.path) == original['model_sha256'] and digest(reference) == original['reference_sha256']
    records = read(TEACHER/'report.json')['reports']['native']['clips'] + read(CORRECTIONS/'report.json')['reports']['native']['amp_clips']
    for record in records:
        assert digest(Path(record['path'])) == record['sha256']
    config = yaml.safe_load(CONFIG.read_text())
    assert config['steps_per_iter'] == 64 and config['actor_epochs'] == 3
    assert config['actor_optimizer']['learning_rate'] == 1e-4 and config['critic_optimizer']['learning_rate'] == 1e-3
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, seed=SEED, config=config,
        model_sha256=digest(rig.path), reference_sha256=digest(reference), demonstrations=records,
        seconds_per_arm=SECONDS, evaluation_seconds=ASSESS_SECONDS, training_episode_seconds=60.,
        maximum_assessment_cases=96, environments=64, rollout_steps=64, transitions_per_update=4096,
        periodic_evaluation_interval=None, periodic_evaluations_disabled=True, observation_width=374,
        zero_action_baseline=True, baseline_16_passes_stop_before_training=True,
        torque_formula_tolerance_nm=TORQUE_TOLERANCE, springs=0., arena_changed=False,
        scope='Fresh PPO/AMP torque versus delayed target PD, same plant and reward; no imitation updates'))
    probe = formula_probe(rig, args.out)
    baseline = {}
    for mode in ('torque', 'target_pd'):
        baseline[mode] = {backend: evaluate(rig, reference, mode, backend, None,
            args.out/f'zero-{mode}-{backend}', reset_probe=True) for backend in ('native', 'gpu')}
    save(args.out/'baseline.json', baseline)
    arms, decision = {}, {}
    solved_by_controller = all(baseline['target_pd'][b]['standing_passes'] == 8 for b in ('native', 'gpu'))
    if not solved_by_controller:
        initial_hash = None
        for mode in ('torque', 'target_pd'):
            print(dict(starting_training=mode), flush=True)
            arms[mode] = train_arm(rig, reference, [Path(r['path']) for r in records], config, mode, args.out/mode, initial_hash)
            initial_hash = arms[mode]['initial_full_tensor_hash']
        dt = rig.model.opt.timestep*rig.decimation
        for backend in ('native', 'gpu'):
            candidate = arms['target_pd']['final'][backend]
            comparators = dict(torque_actor=arms['torque']['final'][backend], zero_target_pd=baseline['target_pd'][backend])
            decision[backend] = {name: comparison(candidate, value, dt) for name, value in comparators.items()}
        comparisons = [value for row in decision.values() for value in row.values()]
        learned_gain = all(v['stand_pass_gain'] > 0 or (v['stand_pass_gain'] == 0 and v['survival_rule_passed']) for v in comparisons)
    else:
        learned_gain = False
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), baseline=baseline, arms=arms,
        decision=decision, formula_probe=probe, controller_only_stand=solved_by_controller,
        intermediate_learned_gain=learned_gain, proposed_formulation_closed=not learned_gain and not solved_by_controller,
        full_learned_stand=bool(arms and all(arms['target_pd']['final'][b]['standing_passes']==8 for b in ('native','gpu'))),
        perturbations_ready=False, arena_changed=False, total_assessment_cases=32+32*len(arms))
    check_protected(protected, sources)
    save(args.out/'report.json', result)
    save(args.out/'completion-verification.json', dict(protected_inputs=len(protected), sources=len(sources),
        inputs_and_sources_unchanged=True, output_hashes={str(p): digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}))
    print(dict(completed=True, learned_gain=learned_gain, controller_only_stand=solved_by_controller), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        if arguments.out.is_dir():
            save(arguments.out/'failure.json', dict(error=traceback.format_exc()))
        raise
