"""Read-only teacher counterfactuals on isolated reconstructed switch states."""
from copy import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .probe_arm_matched_feedback import OUT as BRANCHES, PRIOR, FOCUS, JOINTS
from .probe_arm_execution_distribution import setup, seal
from .train_arm_relative_proposal import load_head
from .interval_target_support import IntervalTargetSupport
from .predictive_target_support import blend, blend_pose
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-private-queue-20261008-01')
PLAN = Path(__file__).with_name('ARM_MATCHED_QUEUE_PROBE_PLAN.md')


def integration(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, values, spec)
    return values


def issue(raw, source, previous, amount, fade, lower, upper):
    desired = (raw-source).clamp(-1, 1)*amount*fade
    addition = previous+(desired-previous).clamp(-.02, .02)
    return (source+addition).clamp(lower, upper)


@torch.no_grad()
def main():
    assert json.loads((BRANCHES/'audit.json').read_text())['passed']
    branch_report = json.loads((BRANCHES/'report.json').read_text())
    protocol = json.loads((BRANCHES/'protocol.json').read_text())
    origins = [r for r in branch_report['results'] if r['config']['pulse'] is None and r['config']['origin'] == r['config']['destination']]
    inputs_paths = [PLAN, BRANCHES/'audit.json', BRANCHES/'report.json', BRANCHES/'protocol.json']
    for row in origins:
        folder = BRANCHES/row['folder']
        teacher_folder = BRANCHES/row['folder'].replace('-continue', '-teacher')
        inputs_paths += [folder/'switch-states.npz', folder/'commands.npz', teacher_folder/'commands.npz']
        for head in ('replay', 'correction05'):
            inputs_paths += [PRIOR/f'{head}-{row["config"]["seed"]}'/'final.pt', PRIOR/f'{head}-{row["config"]["seed"]}'/'architecture.json']
    inputs = seal(OUT, inputs_paths)
    save(OUT/'protocol.json', dict(inputs=inputs, states=24, expected_teacher_queries=120,
                                  queue_target_delta=.005, training_updates=0, physical_rollouts=0))
    env, anchor, _, poses, _ = setup()
    model, c = env.rig.model, env.engine.policy_control
    c.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
    proxy_task = copy(env.task)
    proxy_task.engine = copy(env.engine)
    proxy_control = copy(c)
    proxy_task.engine.policy_control = proxy_control
    proxy_control.pending = c.pending.clone()
    teacher = IntervalTargetSupport(proxy_task, objective='interval_com')
    records, parity = [], []
    for row in origins:
        config, folder = row['config'], BRANCHES/row['folder']
        states, commands = np.load(folder/'switch-states.npz'), np.load(folder/'commands.npz')
        executed = np.load(BRANCHES/row['folder'].replace('-continue', '-teacher')/'commands.npz')
        heads = {name: load_head(PRIOR/f'{name}-{config["seed"]}') for name in ('replay', 'correction05')}
        for world in FOCUS:
            frame = row['switches'][world]
            state = {k.split('__', 1)[1]: states[k] for k in states.files if k.startswith(f'{world}__')}
            data = env.engine.datas[world]
            mujoco.mj_setState(model, data, state['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
            warmstart = data.qacc_warmstart.copy()
            mujoco.mj_forward(model, data)
            data.qacc_warmstart[:] = warmstart
            assert np.array_equal(integration(model, data), state['integration'])
            c.pending[:, world] = torch.tensor(state['pending'])
            c.cursor = int(state['cursor'])
            c.reference_q[world] = torch.tensor(state['reference_q'])
            c.reference_v[world] = torch.tensor(state['reference_v'])
            proxy_control.cursor = c.cursor
            x = torch.zeros((16, 421))
            x[:8] = torch.tensor(commands['x'][frame])
            source = torch.tensor(commands['source'][frame, world])
            goal32 = torch.tensor(commands['goal'][frame, world])
            previous = torch.tensor(commands['previous'][frame, world])
            onset, ramp, hold = (1.2, 1.7, 3.5) if config['variant'] else (1., 1.5, 3.)
            now, future = frame*env.dt, frame*env.dt+2*env.dt
            ref, clear = poses[world, :2]
            goal = blend_pose(ref, clear, future-onset, ramp)
            assert onset <= now < onset+ramp
            amount = torch.tensor(float(blend(0., 1., now-onset, 1.)), dtype=torch.float32)
            fade = ((goal32-torch.tensor(ref[c.q_indices], dtype=torch.float32)).abs().max()/.05).clamp(0, 1)
            def anchor_source(observation):
                return (c.reference_q[world, c.q_indices]+.25*anchor(observation[:, :374])[world].clamp(-1, 1)).clamp(c.lower, c.upper)
            initial_anchor = anchor_source(x)
            assert (initial_anchor-source).abs().max() < 2e-6
            base_outputs = {name: net(x)[world].clone() for name, net in heads.items()}
            baseline_issued = None
            for joint, sign in [(None, 0)]+[(j, s) for j in JOINTS for s in (-1, 1)]:
                proxy_control.pending = c.pending.clone()
                altered_x = x.clone()
                delta = 0.
                if joint is not None:
                    motor = protocol['motors'][joint]
                    old = proxy_control.pending[c.cursor, world, motor].clone()
                    new = (old+sign*.005).clamp(c.lower[motor], c.upper[motor])
                    delta = float(new-old)
                    proxy_control.pending[c.cursor, world, motor] = new
                    assert abs(float(x[world, 95+motor]-old)) < 1e-7
                    altered_x[world, 95+motor] += new-old
                current_source = source+(anchor_source(altered_x)-initial_anchor)
                before = integration(model, data)
                pending_before = c.pending.clone()
                raw, _ = teacher.target(world, goal, current_source.numpy())
                assert np.array_equal(before, integration(model, data)) and torch.equal(c.pending, pending_before)
                effective = goal32+(torch.tensor(raw, dtype=torch.float32)-goal32)
                teacher_issued = issue(effective, current_source, previous, amount, fade, c.lower, c.upper)
                head_rows = {}
                for name, net in heads.items():
                    residual = net(altered_x)[world]
                    assert torch.equal(residual, base_outputs[name])
                    issued = issue(goal32+residual, current_source, previous, amount, fade, c.lower, c.upper)
                    head_rows[name] = dict(raw=(goal32+residual).tolist(), issued=issued.tolist())
                if joint is None:
                    error = float(abs(effective.numpy()-executed['raw_target'][frame, world]).max())
                    assert error < 2e-6, (folder.name, world, error)
                    baseline_issued = teacher_issued.clone()
                    parity.append(dict(folder=folder.name, world=world, maximum_raw_error=error))
                records.append(dict(folder=folder.name, world=world, frame=frame, joint=joint, sign=sign,
                    actual_queue_delta=delta, source=current_source.tolist(), teacher_raw=effective.tolist(),
                    teacher_issued=teacher_issued.tolist(), head_outputs=head_rows,
                    teacher_issued_delta_max=float((teacher_issued-baseline_issued).abs().max())))
        print('QUEUE PROBE', folder.name, 'queries', len(records), flush=True)
    assert len(records) == 120 and len(parity) == 24
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(records=records, reconstruction_parity=parity, teacher_queries=120,
        inputs_unchanged=True, original_state_unchanged=True, masked_head_raw_outputs_identical=True,
        physical_rollouts=0, training_updates=0, arena_changed=False))
    print('PRIVATE QUEUE COMPLETE', max(p['maximum_raw_error'] for p in parity), flush=True)


if __name__ == '__main__':
    main()
