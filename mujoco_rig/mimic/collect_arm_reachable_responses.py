"""Admit paired teacher targets after real delayed command disturbances."""
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_execution_distribution import setup, seal, episode
from .probe_arm_matched_feedback import OUT as MATCHED, PRIOR, JOINTS, state_for, RecordedBranch
from .probe_arm_action_response import fingerprint
from .collect_arm_corrective_takeovers import TakeoverProposal, ARENA
from .train_arm_relative_proposal import load_head
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-reachable-response-20261008-01')
PLAN = Path(__file__).with_name('ARM_REACHABLE_RESPONSE_PLAN.md')
FOCUS = (0, 4)
OFFSETS = (0, 1, 3, 6, 12)


class ReachableProposal(TakeoverProposal):
    def __init__(self, learner, adapter, pulses):
        switches = np.full(8, adapter.physical.frames+1, dtype=np.int64)
        switches[list(FOCUS)] = pulses[list(FOCUS)]+3
        super().__init__(learner, adapter, switches, True)
        self.pulses, self.physical_states = pulses, {}

    @torch.no_grad()
    def forward(self, x):
        frame = int(self.adapter.env.task.steps[0])
        if frame != self.last_frame:
            for world in FOCUS:
                for label, event in (('pulse', self.pulses[world]), ('takeover', self.switches[world])):
                    if frame == event:
                        self.physical_states[f'{world}-{label}'] = state_for(self.adapter.physical, world)
        return super().forward(x)


class TrainingPulse(torch.nn.Module):
    def __init__(self, env, frames, motor=None, sign=0):
        super().__init__()
        self.env, self.frames, self.motor, self.sign = env, frames, motor, sign

    def forward(self, x):
        result = torch.zeros((16, 33))
        if self.motor is not None:
            for world in FOCUS:
                if int(self.env.task.steps[world]) == self.frames[world]:
                    result[world, self.motor] = self.sign*.005
        return result


def run(env, anchor, learner, poses, spec, motors, pulse):
    pulses = np.asarray(spec['switches'])
    adapter = RecordedBranch(env, anchor, learner, poses, variant=spec['variant'])
    proposal = ReachableProposal(learner, adapter, pulses)
    adapter.proposal = proposal
    suffix = 'baseline' if pulse is None else pulse['joint']+'-'+str(pulse['sign'])
    folder = OUT/(spec['name']+'-'+suffix)
    actor = TrainingPulse(env, pulses, None if pulse is None else motors[pulse['joint']], 0 if pulse is None else pulse['sign'])
    config = dict(mode='mean', actor='teacher-after-reachable-pulse', seed=spec['seed'], variant=spec['variant'],
                  teacher=True, pulse=pulse)
    result = episode(adapter, actor, config, folder)
    commands = {k: np.stack([r[k] for r in proposal.rows]) for k in proposal.rows[0]}
    assert not commands['teacher'][:, [1, 2, 3, 5, 6, 7]].any()
    np.savez_compressed(folder/'commands.npz', **commands)
    np.savez_compressed(folder/'states.npz', **{f'{event}__{k}': v for event, s in proposal.physical_states.items() for k, v in s.items()})
    metadata = dict(pulses=pulses.tolist(), switches=proposal.switches.tolist(),
        states={k: fingerprint(v) for k, v in proposal.physical_states.items()},
        teacher_queries=proposal.unchanged_queries, teacher_poses=list(FOCUS))
    save(folder/'branch.json', metadata)
    return dict(folder=folder.name, spec=spec['name'], **result), commands, metadata


def append_pairs(result, baseline, cmd, base_cmd, metadata, motor, pairs):
    folder = OUT/result['folder']
    trace, base_trace = np.load(folder/'trace.npz'), np.load(OUT/baseline['folder']/'trace.npz')
    admissions = []
    for world in FOCUS:
        pulse, switch = metadata['pulses'][world], metadata['switches'][world]
        for key in ('qpos', 'issued', 'alive', 'done'):
            assert np.array_equal(trace[key][:pulse, world], base_trace[key][:pulse, world])
        assert np.array_equal(trace['qpos'][pulse:pulse+2, world], base_trace['qpos'][pulse:pulse+2, world])
        assert np.array_equal(cmd['qvel'][pulse:pulse+2, world], base_cmd['qvel'][pulse:pulse+2, world])
        delta = float(trace['issued'][pulse, world, motor]-base_trace['issued'][pulse, world, motor])
        body = float(abs(trace['qpos'][pulse+2, world]-base_trace['qpos'][pulse+2, world]).max())
        admitted = bool(result['cases'][world]['complete'] and baseline['cases'][world]['complete']
                        and abs(delta) > 1e-5 and body > 1e-10)
        frames = []
        if admitted:
            for offset in OFFSETS:
                frame = switch+offset
                if not (cmd['teacher'][frame, world] and base_cmd['teacher'][frame, world]
                        and trace['alive'][frame, world] and base_trace['alive'][frame, world]):
                    continue
                assert np.array_equal(cmd['goal'][frame, world], base_cmd['goal'][frame, world])
                assert np.array_equal(cmd['x'][frame, world, 374:], base_cmd['x'][frame, world, 374:])
                pair = {f'{k}{side}': source[k][frame, world] for side, source in ((0, base_cmd), (1, cmd))
                        for k in ('x', 'raw_target', 'source', 'previous')}
                pair.update(goal=cmd['goal'][frame, world], world=world, frame=frame, offset=offset,
                    pulse_frame=pulse, issued_delta=delta, seed=result['config']['seed'], variant=result['config']['variant'],
                    joint=result['config']['pulse']['joint'], sign=result['config']['pulse']['sign'],
                    baseline=baseline['folder'], branch=result['folder'], validation=False)
                pairs.append(pair);frames.append(frame)
        admissions.append(dict(world=world, admitted=admitted, issued_delta=delta, native_position_delta=body,
                               baseline_complete=baseline['cases'][world]['complete'],
                               disturbed_complete=result['cases'][world]['complete'], frames=frames))
    save(folder/'admission.json', admissions)
    return admissions


@torch.no_grad()
def main():
    assert json.loads((MATCHED/'audit.json').read_text())['passed']
    protocol = json.loads((MATCHED/'protocol.json').read_text())
    specs = [s for s in protocol['specifications'] if s['origin'] == 'replay']
    extra = [PLAN, ARENA, MATCHED/'protocol.json', MATCHED/'audit.json']
    for spec in specs:
        folder = PRIOR/f'replay-{spec["seed"]}'
        extra += [folder/'final.pt', folder/'architecture.json', PRIOR/spec['name']/'trace.npz',
                  MATCHED/(spec['name']+'-continue')/'switch-states.npz']
    inputs = seal(OUT, extra)
    env, anchor, _, poses, _ = setup()
    motors = {joint: int(np.flatnonzero(env.engine.policy_control.q_indices.numpy()
              == env.rig.model.jnt_qposadr[env.rig.model.joint(joint).id])[0]) for joint in JOINTS}
    save(OUT/'protocol.json', dict(inputs=inputs, specifications=specs, focus=FOCUS, offsets=OFFSETS,
        motors=motors, requested_pulse=.005, pulse_frames=1, takeover_after=3,
        training_updates=0, heldout_teacher_queries=0, expected_physical_cases=160))
    results, pairs, admissions = [], [], []
    for spec in specs:
        learner = load_head(PRIOR/f'replay-{spec["seed"]}')
        baseline = base_cmd = base_meta = None
        for pulse in [None]+[dict(joint=j, sign=s) for j in JOINTS for s in (-1, 1)]:
            r, cmd, meta = run(env, anchor, learner, poses, spec, motors, pulse)
            z = np.load(OUT/r['folder']/'trace.npz')
            old = np.load(PRIOR/spec['name']/'trace.npz')
            for world in range(8):
                end = meta['pulses'][world] if world in FOCUS else len(z['time'])
                if pulse is None and world in FOCUS:
                    end = meta['switches'][world]
                assert all(np.array_equal(z[k][:end, world], old[k][:end, world]) for k in ('qpos', 'issued', 'alive', 'done'))
            if pulse is None:
                baseline, base_cmd, base_meta = r, cmd, meta
            else:
                assert all(meta['states'][f'{i}-pulse'] == base_meta['states'][f'{i}-pulse'] for i in FOCUS)
                admitted = append_pairs(r, baseline, cmd, base_cmd, meta, motors[pulse['joint']], pairs)
                admissions.append(dict(folder=r['folder'], cases=admitted))
            results.append(r)
            save(OUT/'progress.json', dict(results=results, admissions=admissions, paired_examples=len(pairs)))
    assert len(results) == 20
    if pairs:
        dataset = {k: np.stack([p[k] for p in pairs]) for k in pairs[0]}
        assert not dataset['validation'].any() and set(dataset['world']) <= set(FOCUS)
        np.savez_compressed(OUT/'response-pairs.npz', **dataset)
    assert all(digest(Path(path)) == h for path, h in inputs.items())
    save(OUT/'report.json', dict(results=results, admissions=admissions, paired_examples=len(pairs),
        training_updates=0, inputs_unchanged=True, arena_changed=False,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COLLECTION COMPLETE', len(pairs), 'pairs', flush=True)


if __name__ == '__main__':
    main()
