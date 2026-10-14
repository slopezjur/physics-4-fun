"""Frozen controller crossovers and bounded command pulses from exact prefixes."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .train_arm_sampling_balance import OUT as PRIOR, SEEDS
from .train_arm_relative_proposal import load_head
from .probe_arm_execution_distribution import setup, seal, episode
from .probe_arm_action_response import snapshot, fingerprint
from .collect_arm_corrective_takeovers import world_snapshot, ARENA
from .train_arm_correction_ppo import CorrectionEnvironment
from .target_control33 import DelayedTargetControl33
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-matched-feedback-20261008-01')
PLAN = Path(__file__).with_name('ARM_MATCHED_FEEDBACK_PLAN.md')
FOCUS = (0, 4, 6)
JOINTS = ('UpperArm_L_ry', 'UpperArm_R_ry')


def state_for(p, i):
    state = world_snapshot(p, i)
    c, env = p.control, p.env
    for key in ('ctrl', 'active', 'addition'):
        state['control_'+key] = getattr(c, key)[i].numpy().copy()
    for key in ('steps', 'offset', 'previous_action', 'action_cost', 'previous_com'):
        state['task_'+key] = getattr(env.task, key)[i].numpy().copy()
    state.update(ground_force=env.engine._ground_force_cache[i].copy(),
        warning=np.array(env.engine.solver_warning[i]),
        hold=np.array(p.gate.hold[i]), returned=np.array(p.gate.returned[i]),
        adapter_hold=p.hold_ok[i].numpy().copy(), adapter_return=p.return_ok[i].numpy().copy())
    return state


def projected(p, x, residual, goals, amount, quiet):
    source = DelayedTargetControl33.target_positions(p.control, p.anchor(x[:, :374]))
    fade = ((goals[:, p.qi]-p.reference[:, p.qi]).abs().amax(-1)/.05).clamp(0, 1)
    desired = (goals[:, p.qi]+residual-source).clamp(-1, 1)*amount[:, None]*fade[:, None]
    addition = p.previous+(desired-p.previous).clamp(-.02, .02)
    addition[quiet] = 0
    return source, (source+addition).clamp(p.control.lower, p.control.upper)


@torch.no_grad()
def input_probes(p, heads, x, goals, amount, quiet, world, motors):
    records = []
    for name, net in heads.items():
        base = net(x)
        for joint, motor in motors.items():
            fields = (('angle', int(p.qi[motor])+8, .001),
                      ('velocity', int(p.control.v_indices[motor])+48, .01),
                      ('oldest_queue', 95+motor, .005))
            for feature, column, epsilon in fields:
                samples = []
                for sign in (-1, 1):
                    changed = x.clone()
                    changed[world, column] += sign*epsilon
                    residual = net(changed)
                    source, issued = projected(p, changed, residual, goals, amount, quiet)
                    if feature == 'oldest_queue':
                        assert torch.equal(residual, base)
                    samples.append(dict(raw=residual[world].numpy().copy(),
                                        source=source[world].numpy().copy(), issued=issued[world].numpy().copy()))
                records.append(dict(head=name, joint=joint, world=world, feature=feature, column=column,
                    epsilon=epsilon, minus={k: v.tolist() for k, v in samples[0].items()},
                    plus={k: v.tolist() for k, v in samples[1].items()},
                    derivative={k: ((samples[1][k]-samples[0][k])/(2*epsilon)).tolist() for k in samples[0]}))
    return records


class BranchProposal(torch.nn.Module):
    def __init__(self, adapter, heads, origin, destination, switches, motors, probes=False):
        super().__init__()
        from .interval_target_support import IntervalTargetSupport
        self.adapter, self.heads = adapter, heads
        self.origin, self.destination, self.switches = origin, destination, switches
        self.mean, self.scale = heads[origin].mean, heads[origin].scale
        self.teacher = IntervalTargetSupport(adapter.env.task, objective='interval_com')
        self.motors, self.probes = motors, probes
        self.last_frame, self.rows, self.states, self.probe_rows = None, [], {}, []
        self.query_count, self.last_print = 0, time.perf_counter()

    @torch.no_grad()
    def forward(self, x):
        p, env = self.adapter.physical, self.adapter.env
        _, goals, amount, quiet, frames = p.observation()
        frame = int(frames[0])
        if frame == self.last_frame:
            assert torch.equal(self.cached_x, x)
            return self.cached
        assert self.last_frame is None or frame == self.last_frame+1
        self.last_frame = frame
        residual = self.heads[self.origin](x).clone()
        active = torch.arange(16) < 8
        if self.rows:
            active[:8] &= torch.tensor(self.rows[-1]['live_next'])
        source = DelayedTargetControl33.target_positions(p.control, p.anchor(x[:, :374]))
        for i in range(8):
            if frame == self.switches[i]:
                self.states[str(i)] = state_for(p, i)
                if self.probes and i in FOCUS:
                    before = fingerprint(snapshot(p, active))
                    self.probe_rows += [dict(frame=frame, **r) for r in input_probes(p, self.heads, x, goals, amount, quiet, i, self.motors)]
                    assert before == fingerprint(snapshot(p, active))
        teacher_mask = torch.zeros(16, dtype=torch.bool)
        if self.destination != 'teacher':
            next_residual = self.heads[self.destination](x)
            for i in range(8):
                if frame >= self.switches[i]:
                    residual[i] = next_residual[i]
        else:
            from .predictive_target_support import blend_pose
            before = fingerprint(snapshot(p, active))
            now, future = frame*env.dt, frame*env.dt+2*env.dt
            for i in FOCUS:
                if frame < self.switches[i] or not active[i] or quiet[i] or amount[i] == 0:
                    continue
                ref, clear = p.poses[i, :2]
                if now < self.adapter.onset:
                    goal = ref
                elif future < self.adapter.onset+self.adapter.ramp:
                    goal = blend_pose(ref, clear, future-self.adapter.onset, self.adapter.ramp)
                elif future < self.adapter.hold_end:
                    goal = clear
                else:
                    goal = blend_pose(clear, ref, future-self.adapter.hold_end, 1.5)
                raw, _ = self.teacher.target(i, goal, source[i].numpy())
                target = torch.tensor(raw, dtype=torch.float32)
                residual[i] = target-goals[i, p.qi]
                assert (goals[i, p.qi]+residual[i]-target).abs().max() < 2e-7
                teacher_mask[i] = True
            assert before == fingerprint(snapshot(p, active))
            self.query_count += int(teacher_mask.sum())
        self.rows.append(dict(x=x[:8].numpy().copy(), source=source[:8].numpy().copy(),
            raw_target=(goals[:, p.qi]+residual)[:8].numpy().copy(),
            residual=residual[:8].numpy().copy(), goal=goals[:8, p.qi].numpy().copy(),
            previous=p.previous[:8].numpy().copy(), teacher=teacher_mask[:8].numpy().copy(),
            live_next=active[:8].numpy().copy()))
        self.cached_x, self.cached = x.clone(), residual
        if time.perf_counter()-self.last_print > 20:
            print('BRANCH PROGRESS', self.origin, self.destination, frame, 'teacher queries', self.query_count, flush=True)
            self.last_print = time.perf_counter()
        return residual


class RecordedBranch(CorrectionEnvironment):
    def step(self, action, active):
        result = super().step(action, active)
        self.proposal.rows[-1]['live_next'] = (active & ~result[1])[:8].numpy().copy()
        self.proposal.rows[-1]['qvel'] = np.stack([d.qvel.copy() for d in self.env.engine.datas[:8]])
        return result


class PulseActor(torch.nn.Module):
    def __init__(self, env, switches, motor=None, sign=0):
        super().__init__()
        self.env, self.switches, self.motor, self.sign = env, switches, motor, sign

    def forward(self, x):
        value = torch.zeros((16, 33))
        if self.motor is not None:
            for i in FOCUS:
                if int(self.env.task.steps[i]) == self.switches[i]:
                    value[i, self.motor] = self.sign*.005
        return value


def run(env, anchor, heads, poses, spec, destination, out, motors, pulse=None):
    switches, origin = np.array(spec['switches']), spec['origin']
    adapter = RecordedBranch(env, anchor, heads[origin], poses, variant=spec['variant'])
    wrapper = BranchProposal(adapter, heads, origin, destination, switches, motors,
                             probes=origin == destination and pulse is None)
    adapter.proposal = wrapper
    config = dict(mode='mean', actor=destination, origin=origin, destination=destination,
                  variant=spec['variant'], seed=spec['seed'], teacher=destination == 'teacher', pulse=pulse)
    actor = PulseActor(env, switches, None if pulse is None else motors[pulse['joint']],
                       0 if pulse is None else pulse['sign'])
    result = episode(adapter, actor, config, out)
    np.savez_compressed(out/'commands.npz', **{k: np.stack([r[k] for r in wrapper.rows]) for k in wrapper.rows[0]})
    np.savez_compressed(out/'switch-states.npz', **{f'{i}__{k}': v for i, state in wrapper.states.items() for k, v in state.items()})
    save(out/'branch.json', dict(switches=switches.tolist(), states={i: fingerprint(s) for i, s in wrapper.states.items()},
         teacher_queries=wrapper.query_count, teacher_poses=list(FOCUS) if destination == 'teacher' else [],
         input_probes=wrapper.probe_rows))
    return result


def verify_prefix(folder, baseline_folder, switches):
    z, baseline = np.load(folder/'trace.npz'), np.load(baseline_folder/'trace.npz')
    state = json.loads((folder/'branch.json').read_text())['states']
    original = json.loads((baseline_folder/'branch.json').read_text())['states']
    assert state == original
    for i, frame in enumerate(switches):
        for key in ('qpos', 'issued', 'alive', 'done'):
            assert np.array_equal(z[key][:frame, i], baseline[key][:frame, i]), (folder, key, i)


@torch.no_grad()
def main():
    diagnostics = json.loads((PRIOR/'diagnostics.json').read_text())
    assert json.loads((PRIOR/'audit.json').read_text())['passed']
    extra = [PLAN, ARENA, PRIOR/'diagnostics.json', PRIOR/'audit.json']
    specs = []
    for seed in SEEDS:
        for origin in ('replay', 'correction05'):
            extra += [PRIOR/f'{origin}-{seed}'/'final.pt', PRIOR/f'{origin}-{seed}'/'architecture.json']
            for variant in (False, True):
                name = f'{origin}-{seed}'+('-varied' if variant else '-nominal')
                extra += [PRIOR/name/'trace.npz']
                row = next(r for r in diagnostics['physical'] if r['recipe'] == 'correction05' and r['seed'] == seed and r['variant'] == variant)
                switches = [c['first_command_1mrad']['frame'] for c in row['cases']]
                specs.append(dict(seed=seed, origin=origin, variant=variant, switches=switches, name=name))
    inputs = seal(OUT, extra)
    env, anchor, _, poses, _ = setup()
    motors = {joint: int(np.flatnonzero(env.engine.policy_control.q_indices.numpy() == env.rig.model.jnt_qposadr[env.rig.model.joint(joint).id])[0]) for joint in JOINTS}
    save(OUT/'protocol.json', dict(inputs=inputs, specifications=specs, focus=FOCUS, motors=motors,
        input_epsilons=dict(angle=.001, velocity=.01, oldest_queue=.005), pulse_radians=.005,
        pulse_frames=1, training_updates=0, teacher_heldout_use='Diagnostic queries only; no exported labels'))
    results = []
    for spec in specs:
        heads = {name: load_head(PRIOR/f'{name}-{spec["seed"]}') for name in ('replay', 'correction05')}
        baseline_folder = OUT/(spec['name']+'-continue')
        other = 'correction05' if spec['origin'] == 'replay' else 'replay'
        for destination, suffix in ((spec['origin'], 'continue'), (other, 'swap'), ('teacher', 'teacher')):
            folder = OUT/(spec['name']+'-'+suffix)
            r = run(env, anchor, heads, poses, spec, destination, folder, motors)
            if suffix == 'continue':
                old, z = np.load(PRIOR/spec['name']/'trace.npz'), np.load(folder/'trace.npz')
                assert all(np.array_equal(old[k], z[k]) for k in ('qpos', 'issued', 'alive', 'done'))
            else:
                verify_prefix(folder, baseline_folder, spec['switches'])
            results.append(dict(folder=folder.name, baseline=baseline_folder.name, switches=spec['switches'], **r))
            save(OUT/'progress.json', results)
        if spec['seed'] == 711028 and spec['origin'] == 'replay' and not spec['variant']:
            for destination in ('replay', 'correction05'):
                comparison = OUT/(spec['name']+('-continue' if destination == 'replay' else '-swap'))
                for joint in JOINTS:
                    for sign in (-1, 1):
                        pulse = dict(joint=joint, sign=sign)
                        folder = OUT/f'pulse-{destination}-{joint}-{sign}'
                        r = run(env, anchor, heads, poses, spec, destination, folder, motors, pulse=pulse)
                        verify_prefix(folder, comparison, spec['switches'])
                        results.append(dict(folder=folder.name, baseline=comparison.name, switches=spec['switches'], **r))
                        save(OUT/'progress.json', results)
    assert len(results) == 32
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(results=results, training_updates=0, arena_changed=False, inputs_unchanged=True,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', len(results), 'episodes', flush=True)


if __name__ == '__main__':
    main()
