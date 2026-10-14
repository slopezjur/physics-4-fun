"""Admit teacher continuations only after exact learned-prefix replay and full gates."""
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .probe_arm_execution_distribution import setup, seal, episode
from .probe_arm_action_response import snapshot, fingerprint
from .analyze_arm_proposal_generalization import physical_results
from .challenge_arm_proposals import HEADS
from .train_arm_relative_proposal import load_head
from .train_arm_correction_ppo import CorrectionEnvironment
from .target_control33 import DelayedTargetControl33
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/arm-corrective-takeover-20261008-01')
PLAN = Path(__file__).with_name('ARM_CORRECTIVE_TAKEOVER_PLAN.md')
ARENA = Path('logs/mimickit-training/perturb-arena-learned-20261008-01/stand.onnx')


def world_snapshot(p, world):
    """Issue-time physical state and command history for one independent world."""
    data, c = p.env.engine.datas[world], p.control
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(p.env.rig.model, spec))
    mujoco.mj_getState(p.env.rig.model, data, state, spec)
    return dict(integration=state, pending=c.pending[:, world].numpy().copy(),
                previous=p.previous[world].numpy().copy(),
                observation=p.env.task.observations()[world].numpy().copy(),
                reference_q=c.reference_q[world].numpy().copy(),
                reference_v=c.reference_v[world].numpy().copy(), cursor=np.array(c.cursor))


def switches_for(adapter, old, report):
    p, dt = adapter.physical, adapter.env.dt
    onset = int(np.ceil(adapter.onset / dt))
    switches, evidence = np.full(8, p.frames, dtype=np.int64), []
    initial_xy = p.initial_xy.numpy()[:8]
    for case in report['cases']:
        i = case['pose']
        if i >= 6 or case['complete']:
            continue
        motor = np.sqrt(np.mean((old['qpos'][:, i, p.qi] - p.goals[:-1, i, p.qi].numpy()) ** 2, -1))
        com = np.linalg.norm(old['com'][:, i, :2] - p.target_com[:-1, i, :2].numpy(), axis=-1)
        feet = np.linalg.norm(old['xy'][:, i] - initial_xy[i], axis=-1).max(-1)
        bad = (motor > .05) | (com > .02) | (feet > .02) | ~old['support'][:, i] | old['done'][:, i] | ~old['alive'][:, i]
        ids = np.flatnonzero(bad & (np.arange(p.frames) >= onset))
        first = int(ids[0]) if len(ids) else None
        switches[i] = max(onset, first - int(np.ceil(.5 / dt))) if first is not None else onset
        evidence.append(dict(pose=i, first_violation=first, switch=int(switches[i]),
                             fallback_onset=onset))
    return switches, evidence


class TakeoverProposal(torch.nn.Module):
    def __init__(self, learner, adapter, switches, enabled):
        super().__init__()
        from .interval_target_support import IntervalTargetSupport
        self.learner, self.adapter = learner, adapter
        self.mean, self.scale = learner.mean, learner.scale
        self.switches, self.enabled = switches, enabled
        self.teacher = IntervalTargetSupport(adapter.env.task, objective='interval_com')
        self.last_frame, self.rows, self.states = None, [], {}
        self.unchanged_queries = 0
        self.last_print = time.perf_counter()

    @torch.no_grad()
    def forward(self, x):
        p, env = self.adapter.physical, self.adapter.env
        _, goals, amount, quiet, frames = p.observation()
        frame = int(frames[0])
        if frame == self.last_frame:
            assert torch.equal(x, self.cached_x)
            return self.cached_output
        assert self.last_frame is None or frame == self.last_frame + 1
        self.last_frame = frame
        c = p.control
        source = DelayedTargetControl33.target_positions(c, p.anchor(x[:, :374]))
        residual = self.learner(x).clone()
        raw = goals[:, p.qi] + residual
        requested = raw.clone()
        queried = torch.zeros(16, dtype=torch.bool)
        onset = int(np.ceil(self.adapter.onset / env.dt))
        for i in range(8):
            if frame in (self.switches[i], onset):
                self.states[f'{i}-{frame}'] = fingerprint(world_snapshot(p, i))
        # Failed worlds have no further live transitions; use previous native failure.
        active = torch.arange(16) < 8
        if self.rows:
            active[:8] &= torch.tensor(self.rows[-1]['live_next'])
        live_before = snapshot(p, active)
        if self.enabled:
            from .predictive_target_support import blend_pose
            now, future = frame * env.dt, (frame + 2) * env.dt
            for i in range(6):
                if frame < self.switches[i] or not active[i] or quiet[i] or amount[i] == 0:
                    continue
                ref, clear = p.poses[i, :2]
                if now < self.adapter.onset:
                    goal = ref
                elif future < self.adapter.onset + self.adapter.ramp:
                    goal = blend_pose(ref, clear, future-self.adapter.onset, self.adapter.ramp)
                elif future < self.adapter.hold_end:
                    goal = clear
                else:
                    goal = blend_pose(clear, ref, future-self.adapter.hold_end, 1.5)
                desired, _ = self.teacher.target(i, goal, source[i].numpy())
                requested[i] = torch.tensor(desired, dtype=torch.float32)
                residual[i] = requested[i] - goals[i, p.qi]
                queried[i] = True
        assert fingerprint(live_before) == fingerprint(snapshot(p, active))
        self.unchanged_queries += int(queried.sum())
        effective = goals[:, p.qi] + residual
        assert (effective - requested).abs().max() < 2e-7
        self.rows.append(dict(x=x[:8].numpy().copy(), raw_target=effective[:8].numpy().copy(),
            requested=requested[:8].numpy().copy(), residual=residual[:8].numpy().copy(),
            source=source[:8].numpy().copy(), goal=goals[:8, p.qi].numpy().copy(),
            previous=p.previous[:8].numpy().copy(), quiet=quiet[:8].numpy().copy(),
            teacher=queried[:8].numpy().copy(), live_next=active[:8].numpy().copy()))
        self.cached_x, self.cached_output = x.clone(), residual
        if time.perf_counter() - self.last_print > 20:
            print('TAKEOVER', frame, 'queries', self.unchanged_queries, flush=True)
            self.last_print = time.perf_counter()
        return residual


class RecordedEnvironment(CorrectionEnvironment):
    def step(self, action, active):
        result = super().step(action, active)
        self.proposal.rows[-1]['live_next'] = (active & ~result[1])[:8].numpy().copy()
        return result


def run(env, anchor, learner, poses, zero, variant, switches, enabled, folder):
    adapter = RecordedEnvironment(env, anchor, learner, poses, variant=variant)
    wrapper = TakeoverProposal(learner, adapter, switches, enabled)
    adapter.proposal = wrapper
    result = episode(adapter, zero, dict(mode='mean', variant=variant, teacher=enabled), folder)
    rows = {k: np.stack([r[k] for r in wrapper.rows]) for k in wrapper.rows[0]}
    np.savez_compressed(folder/'labels.npz', **rows)
    save(folder/'takeover.json', dict(switches=switches.tolist(), states=wrapper.states,
         unchanged_queries=wrapper.unchanged_queries))
    return result, rows, wrapper.states


@torch.no_grad()
def main():
    prior = physical_results()
    heads = {k: HEADS[k] for k in ('all-711028', 'all-721028')}
    extra = [PLAN, ARENA]
    for label, folder in heads.items():
        extra += [folder/'final.pt', folder/'architecture.json']
        extra += [path/name for path, _ in prior[label] for name in ('trace.npz', 'report.json')]
    inputs = seal(OUT, extra)
    env, anchor, _, poses, actors = setup()
    specifications = []
    for label, folder in heads.items():
        learner = load_head(folder)
        for archived, report in prior[label]:
            variant = report['config'].get('variant', False)
            adapter = CorrectionEnvironment(env, anchor, learner, poses, variant=variant)
            with np.load(archived/'trace.npz') as old:
                switches, evidence = switches_for(adapter, old, report)
            specifications.append(dict(actor=label, variant=variant, archived=str(archived),
                                       switches=switches.tolist(), evidence=evidence))
    save(OUT/'protocol.json', dict(inputs=inputs, specifications=specifications,
                                  training_updates=0, heldout_teacher_queries=0))
    results, datasets = [], []
    for spec in specifications:
        label, variant = spec['actor'], spec['variant']
        name = label + ('-varied' if variant else '-nominal')
        learner, switches = load_head(heads[label]), np.array(spec['switches'])
        control_folder = OUT/(name+'-control')
        control, _, states = run(env, anchor, learner, poses, actors['zero'], variant,
                                  switches, False, control_folder)
        old = np.load(Path(spec['archived'])/'trace.npz')
        control_trace = np.load(control_folder/'trace.npz')
        assert all(np.array_equal(old[k], control_trace[k]) for k in ('qpos', 'issued', 'alive', 'done'))
        pending = switches.copy()
        for attempt in ('lead', 'onset'):
            folder = OUT/(name+'-'+attempt)
            result, rows, takeover_states = run(env, anchor, learner, poses, actors['zero'],
                                                variant, pending, True, folder)
            z = np.load(folder/'trace.npz')
            for i, switch in enumerate(pending):
                for key in ('qpos', 'issued', 'alive', 'done'):
                    assert np.array_equal(z[key][:switch, i], control_trace[key][:switch, i])
                state_key = f'{i}-{switch}'
                if state_key in takeover_states:
                    assert takeover_states[state_key] == states[state_key]
            accepted = np.array([c['complete'] for c in result['cases']])
            use = rows['teacher'] & z['alive'] & accepted[None]
            frame, world = np.nonzero(use)
            sample = {k: rows[k][use] for k in ('x', 'raw_target', 'source', 'previous', 'goal', 'quiet')}
            sample.update(issued=z['issued'][use], world=world, frame=frame,
                validation=np.zeros(len(frame), dtype=bool),
                episode=np.full(len(frame), folder.name))
            assert not (world >= 6).any()
            np.savez_compressed(folder/'admitted-samples.npz', **sample)
            datasets.append(sample)
            results.append(dict(folder=folder.name, control=control_folder.name, attempt=attempt,
                                admitted_samples=len(frame), **result))
            save(OUT/'progress.json', results)
            onset = int(np.ceil((1.2 if variant else 1.) / env.dt))
            failed = [i for i in range(6) if pending[i] < len(z['time']) and not accepted[i] and pending[i] > onset]
            if not failed:
                break
            pending = np.full(8, len(z['time']), dtype=np.int64)
            pending[failed] = onset
    data = {k: np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    np.savez_compressed(OUT/'training-samples.npz', **data)
    assert all(digest(Path(p)) == h for p, h in inputs.items())
    save(OUT/'report.json', dict(results=results, inputs_unchanged=True,
        admitted_samples=len(data['world']), training_updates=0, arena_changed=False,
        outputs={str(p): digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', len(data['world']), 'admitted correction samples', flush=True)


if __name__ == '__main__':
    main()
