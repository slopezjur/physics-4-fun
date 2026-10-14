"""Matched physical interventions through the unchanged delayed command adapter."""
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT
from .run_learned_handoff import PRIOR
from .ablate_arm_hold_queue import QueueBlindLearner
from .train_arm_hold_ppo import INITIAL, BASELINE
from .train_arm_hold_ppo_rate import DiagnosticRollout
from .target_control33 import DelayedTargetControl33

OUT = Path('logs/mimickit-training/arm-action-response-20261008-01')
PHASES = {'ramp': 90, 'hold': 210, 'return': 360}
BRANCHES = [('zero', 0., 0), ('plus-pulse', .005, 1), ('minus-pulse', -.005, 1),
            ('plus-held', .005, 12), ('minus-held', -.005, 12)]


def snapshot(adapter, active):
    env = adapter.env
    engine, c = env.engine, adapter.control
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    states = []
    for data in engine.datas:
        state = np.empty(mujoco.mj_stateSize(env.rig.model, spec))
        mujoco.mj_getState(env.rig.model, data, state, spec)
        states.append(state)
    result = dict(integration=np.stack(states), ground_forces=engine._ground_force_cache.copy(),
        active=active.numpy().copy(), solver_warning=engine.solver_warning.copy(), cursor=np.array(c.cursor),
        adapter_hold=adapter.hold_ok.numpy().copy(), adapter_return=adapter.return_ok.numpy().copy(),
        gate_hold=adapter.gate.hold.copy(), gate_return=adapter.gate.returned.copy())
    for key in ('ctrl', 'pending', 'active', 'addition', 'reference_q', 'reference_v'):
        result['control_' + key] = getattr(c, key).numpy().copy()
    for key in ('steps', 'offset', 'previous_action', 'action_cost', 'previous_com'):
        result['task_' + key] = getattr(env.task, key).numpy().copy()
    result['previous_addition'] = adapter.previous.numpy().copy()
    result['observations'] = env.task.observations().numpy().copy()
    return result


def fingerprint(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


@torch.no_grad()
def reach(adapter, net, frame, archived):
    adapter.reset()
    adapter.control.cursor = 0  # Pending slots are empty after full reset.
    active = torch.zeros(16, dtype=torch.bool)
    active[:8] = True
    for i in range(frame):
        x, *_ = adapter.observation()
        _, terminal, info = adapter.step(net(x), active)
        assert np.array_equal(active[:8].numpy(), archived['alive'][i])
        assert np.array_equal(info['qpos'][:8].numpy(), archived['qpos'][i])
        assert np.array_equal(info['issued'][:8].numpy(), archived['issued'][i])
        active &= ~terminal
    return active


@torch.no_grad()
def continuation(adapter, net, active, motor, magnitude, duration):
    env, c = adapter.env, adapter.control
    rows = []
    for offset in range(60):
        x, goals, amount, quiet, frame = adapter.observation()
        model_action = net(x)
        action = model_action.clone()
        if offset < duration:
            action[active, motor] += magnitude
        c.set_reference(*env.task.sample_reference(env.task.offset + env.task.steps * env.dt))
        source = DelayedTargetControl33.target_positions(c, adapter.anchor(x[:, :374]))
        raw = goals[:, c.q_indices] + action
        fade = ((goals[:, c.q_indices] - adapter.reference[:, c.q_indices]).abs().amax(-1) / .05).clamp(0, 1)
        desired = (raw - source).clamp(-1, 1) * amount[:, None] * fade[:, None]
        previous = adapter.previous.clone()
        before = active.clone()
        reward, terminal, info = adapter.step(action, active)
        addition = previous + (desired - previous).clamp(-.02, .02)
        addition[quiet | ~active] = 0
        assert torch.equal(addition, adapter.previous)
        expected = (source + addition).clamp(c.lower, c.upper)
        assert torch.equal(expected, info['issued'])
        assert (env.engine.substep_torque.abs() <= c.scale + 1e-4).all()
        # Counterfactual width-2 reward on precisely the same visited state.
        parts = info['components']
        broad_dense = .45 * (parts[:, 0] / .45).pow(.25) + .20 * (parts[:, 1] / .20).pow(.25) + parts[:, 2] + parts[:, 3]
        broad_reward = torch.where(info['failed'], -2., broad_dense) + parts[:, 5]
        values = dict(alive=before, qpos=info['qpos'], failed=info['failed'], supported=info['supported'],
            stand=info['stand'], goal=goals[:, c.q_indices], model_action=model_action, injected_action=action,
            raw_target=raw, source=source, authority=amount, fade=fade, desired_addition=desired,
            previous_addition=previous, addition=addition, issued=info['issued'], applied=c.active[:, :33].clone(),
            applied_velocity=c.active[:, 33:66].clone(), reward=reward, broad_reward=broad_reward,
            components=parts, tracking_rmse=info['tracking_rmse'])
        row = {k: v[:8].numpy().copy() for k, v in values.items()}
        for key in ('q', 'v', 'torque'):
            row['substep_' + key] = getattr(env.engine, 'substep_' + key)[:, :8].numpy().copy()
        rows.append(row)
        active &= ~terminal
    return {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    (OUT / 'sources').mkdir()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('ARM_ACTION_RESPONSE_PLAN.md')]
    inputs = {str(p): digest(p) for p in paths + [PARENT, PARENT.with_suffix('.amp.pt'), INITIAL, BASELINE,
        PRIOR / 'goals.npz', FOUNDATION / 'dummy.xml', FOUNDATION / 'stationary_reference.npz']}
    for p in paths:
        shutil.copy2(p, OUT / 'sources' / p.name)
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION / 'stationary_reference.npz', json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)
    env = make_environment(rig, anchor, 40.)
    engine = env.engine
    engine.record_substeps = True
    shape = (rig.decimation, 16, 33)
    engine.substep_q, engine.substep_v, engine.substep_torque = (torch.empty(shape) for _ in range(3))
    state = torch.load(INITIAL, weights_only=True)
    net = QueueBlindLearner(state['mean'], state['scale'], state['output_scale'])
    net.load_state_dict(state)
    net.eval()
    poses = np.load(PRIOR / 'goals.npz')['poses']
    adapter = DiagnosticRollout(env, anchor, poses)
    joint = rig.model.joint('UpperArm_L_ry').id
    motor = int(np.flatnonzero(adapter.qi.numpy() == rig.model.jnt_qposadr[joint])[0])
    save(OUT / 'protocol.json', dict(inputs=inputs, phases=PHASES, branches=BRANCHES, frames=60,
        dt=env.dt, physics_dt=rig.model.opt.timestep, motor=motor, qpos_index=int(adapter.qi[motor]),
        torque_limit=float(adapter.control.scale[motor]), controlled_joint='UpperArm_L_ry',
        branching='Identical deterministic prefixes; no live state overwrites', training_updates=0, arena_changed=False))
    archived = np.load(BASELINE)
    results = []
    for phase, frame in PHASES.items():
        common = None
        for name, magnitude, duration in BRANCHES:
            active = reach(adapter, net, frame, archived)
            start = snapshot(adapter, active)
            if common is None:
                common = start
                np.savez_compressed(OUT / f'{phase}-state.npz', **common)
            else:
                assert start.keys() == common.keys()
                assert all(np.array_equal(v, common[k]) for k, v in start.items()), phase
            trace = continuation(adapter, net, active.clone(), motor, magnitude, duration)
            if magnitude == 0:
                for key in ('qpos', 'issued', 'alive'):
                    assert np.array_equal(trace[key], archived[key][frame:frame + 60]), (phase, key)
            np.savez_compressed(OUT / f'{phase}-{name}.npz', **trace)
            row = dict(phase=phase, branch=name, magnitude=magnitude, duration_frames=duration,
                start_sha256=fingerprint(start), eligible_poses=np.flatnonzero(active[:8]).tolist(),
                survived_poses=np.flatnonzero(trace['alive'][-1] & ~trace['failed'][-1]).tolist())
            results.append(row)
            save(OUT / 'progress.json', results)
            print('BRANCH', row, flush=True)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT / 'report.json', dict(results=results, inputs_unchanged=True, no_training=True, arena_changed=False,
        output_hashes={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__ == '__main__':
    main()
