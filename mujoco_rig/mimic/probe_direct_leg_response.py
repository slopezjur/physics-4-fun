"""Crossed push timing and matched-prefix diagnostic leg interventions."""
import argparse
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import run_direct_recovery as prior
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT
from .stand_experiment_evidence import load_actor

OUT = Path('logs/mimickit-training/direct-leg-response-20261009-01')
PREVIOUS = Path('logs/mimickit-training/direct-recovery-20261009-01')
ONSETS = (1., 1.7)
AMPLITUDES = (-.075, -.025, .025, .075)


def motor_combinations(model, actuators):
    names = [model.joint(int(j)).name for j in model.actuator_trnid[actuators, 0]]
    index = {name: i for i, name in enumerate(names)}
    values = {}
    for name, entries in (
        ('ankles', [(f'Foot_{s}_rx', 1.) for s in ('L', 'R')]),
        ('hips', [(f'Thigh_{s}_rx', 1.) for s in ('L', 'R')]),
        ('left_flex', [('Thigh_L_rx', .5), ('Shin_L_rx', -1.), ('Foot_L_rx', .5)]),
        ('right_flex', [('Thigh_R_rx', .5), ('Shin_R_rx', -1.), ('Foot_R_rx', .5)]),
        ('transfer', [(f'{b}_{s}_rz', v) for s in ('L', 'R') for b, v in (('Thigh', 1.), ('Foot', -1.))]),
        ('split', [('Thigh_L_rx', 1.), ('Thigh_R_rx', -1.)]),
    ):
        vector = torch.zeros(33)
        for joint, value in entries:
            vector[index[joint]] = value
        values[name] = vector
    legs = [i for i, name in enumerate(names) if name.startswith(('Thigh', 'Shin', 'Foot', 'Toe'))]
    assert len(legs) == 16
    return values, legs, names


class RecordedSource(nn.Module):
    def __init__(self, anchor, legs, replay=None, replay_window=None):
        super().__init__()
        self.anchor, self.legs = anchor, legs
        self.replay, self.replay_window = replay, replay_window
        self.rows = []

    @property
    def obs_norm(self):
        return self.anchor.obs_norm

    def forward(self, obs):
        action = self.anchor(obs)
        frame = len(self.rows)
        if self.replay is not None and self.replay_window[0] <= frame < self.replay_window[1]:
            action = action.clone()
            action[:, self.legs] = torch.from_numpy(self.replay[frame, :, self.legs].T.copy())
        self.rows.append(action.detach().numpy().copy())
        return action


class Pulse:
    def __init__(self, env, vector, amplitude, issue):
        self.env, self.vector, self.amplitude, self.issue = env, vector, amplitude, issue

    def __call__(self, _):
        enabled = (self.env.task.steps >= self.issue) & (self.env.task.steps < self.issue+12)
        desired = enabled[:, None] * self.amplitude * self.vector[None]
        return torch.atanh(desired / prior.RADIUS)


def read_trace(folder):
    with np.load(folder/'trace.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compare(report, trace, baseline_report, baseline, *, issue=None):
    if issue is not None:
        for key in ('qpos', 'qvel', 'issued', 'active_target', 'force', 'alive'):
            assert np.array_equal(trace[key][:issue], baseline[key][:issue]), f'Prefix mismatch: {key}'
        for key in ('qpos', 'qvel'):
            assert np.array_equal(trace[key][:issue+2], baseline[key][:issue+2]), f'Delay mismatch: {key}'
    gains, losses = [], []
    for row, base in zip(report['cases'], baseline_report['cases'], strict=True):
        if row['settled'] and not base['settled']:
            gains.append(row['case'])
        if base['settled'] and not row['settled']:
            losses.append(row['case'])
    train = np.array([i for i in range(16) if i % 8 < 6])
    active = trace['alive'] & baseline['alive']
    addition = trace['addition']
    response = trace['nominal']-baseline['nominal']
    energy = np.square(addition).sum(-1)
    valid = active & (energy > 1e-10)
    projection = (response*addition).sum(-1)
    return dict(gained_cases=gains, lost_cases=losses,
        train_gain=sum(i % 8 < 6 for i in gains), train_loss=sum(i % 8 < 6 for i in losses),
        heldout_gain=sum(i % 8 >= 6 for i in gains), heldout_loss=sum(i % 8 >= 6 for i in losses),
        training_survival_seconds=float((trace['alive'][:, train] & ~trace['failed'][:, train]).sum()*report['dt']),
        paired_nominal_projection=float(projection[valid].sum()/energy[valid].sum()) if valid.any() else None,
        opposed_frame_cases=int((projection[valid] < 0).sum()), projected_frame_cases=int(valid.sum()))


def experiment(rig, anchor, vectors, legs, out, name, *, group=0, onset=1.,
               profile=None, amplitude=0., quiet=False, replay_folder=None, baseline_folder=None):
    dt = rig.model.opt.timestep * rig.decimation
    issue = int(np.ceil(onset/dt))+1
    window = (issue, issue+12+int(np.ceil(abs(amplitude)/prior.SLEW))+2)
    base = read_trace(baseline_folder) if baseline_folder else None
    replay = read_trace(replay_folder)['source_action'] if replay_folder else None
    source = RecordedSource(anchor, legs, replay, window)
    env = prior.Recovery(rig, source)

    def configure(target, group, quiet):
        target.reset()
        target.force.zero_()
        if not quiet:
            strength, axis = (150., 300.)[group//2], group % 2
            target.force[:8] = strength*prior.DIRECTIONS[2*axis]
            target.force[8:] = strength*prior.DIRECTIONS[2*axis+1]
        target.onset[:] = int(np.ceil(onset/target.dt))

    folder = out/name
    pulse = Pulse(env, vectors[profile], amplitude, issue) if profile else None
    with patch.object(prior, 'configure', configure):
        report = prior.rollout(env, pulse, folder, group=group, quiet=quiet)
    trace = read_trace(folder)
    trace['source_action'] = np.stack(source.rows)
    # All traces remain self-contained, including the exact source-action replay.
    np.savez_compressed(folder/'trace.npz', **trace)
    descriptor = dict(name=name, group=group, onset=onset, profile=profile, amplitude=amplitude,
        quiet=quiet, issue_frame=issue, duration_frames=12, replay_window=list(window),
        replay_folder=str(replay_folder) if replay_folder else None,
        baseline_folder=str(baseline_folder) if baseline_folder else None)
    if baseline_folder:
        base_report = json.loads((baseline_folder/'report.json').read_text())
        descriptor['paired'] = compare(report, trace, base_report, base, issue=issue)
        if replay_folder and not profile:
            assert all(np.array_equal(trace[k], base[k]) for k in base), 'Zero replay changed the baseline'
            descriptor['zero_replay_exact'] = True
    save(folder/'descriptor.json', descriptor)
    return report, descriptor


def seal_inputs(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    inputs = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(path)] = digest(path)
        shutil.copy2(path, out/'sources'/path.name)
    for path in (PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml',
                 FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
                 FOUNDATION/'protocol.json', Path(__file__).with_name('DIRECT_LEG_RESPONSE_PLAN.md'),
                 PREVIOUS/'audit.json'):
        inputs[str(path)] = digest(path)
    for path in PREVIOUS.glob('baseline-push-*/*'):
        if path.is_file():
            inputs[str(path)] = digest(path)
    save(out/'inputs.json', inputs)
    return inputs


def main():
    from .rig import Rig
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    inputs = seal_inputs(out)
    torch.set_num_threads(1)
    start = time.perf_counter()
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    vectors, legs, names = motor_combinations(rig.model, rig.actuators)
    save(out/'contract.json', dict(combinations={k: v.tolist() for k, v in vectors.items()},
        leg_indices=legs, motor_names=names, dt=rig.model.opt.timestep*rig.decimation,
        lower=rig.model.jnt_range[rig.model.actuator_trnid[rig.actuators, 0], 0].tolist(),
        upper=rig.model.jnt_range[rig.model.actuator_trnid[rig.actuators, 0], 1].tolist(),
        diagnostic_only=True, optimizer_updates=0))
    matrix = []
    for timing, onset in enumerate(ONSETS):
        for group in range(4):
            name = f'baseline-t{timing}-g{group}'
            report, _ = experiment(rig, anchor, vectors, legs, out, name, group=group, onset=onset)
            if timing == group % 2:
                old, new = read_trace(PREVIOUS/f'baseline-push-{group}'), read_trace(out/name)
                assert all(np.array_equal(old[k], new[k]) for k in old), 'Previous baseline did not reproduce'
            matrix.append(dict(timing=timing, group=group, settled=report['settled'],
                               recovered_with_step=report['recovered_with_step']))
    save(out/'crossed_matrix.json', matrix)
    candidates = []
    for profile in vectors:
        for ai, amplitude in enumerate(AMPLITUDES):
            paired = []
            for timing, onset in enumerate(ONSETS):
                name = f'probe-{profile}-a{ai}-t{timing}'
                report, descriptor = experiment(rig, anchor, vectors, legs, out, name,
                    onset=onset, profile=profile, amplitude=amplitude,
                    baseline_folder=out/f'baseline-t{timing}-g0')
                paired.append(descriptor['paired'])
            row = dict(profile=profile, amplitude=amplitude, amplitude_index=ai,
                train_gain=sum(p['train_gain'] for p in paired), train_loss=sum(p['train_loss'] for p in paired),
                heldout_gain=sum(p['heldout_gain'] for p in paired), heldout_loss=sum(p['heldout_loss'] for p in paired),
                training_survival_seconds=sum(p['training_survival_seconds'] for p in paired))
            candidates.append(row)
            save(out/'candidates.json', candidates)
            print('PROFILE', row, flush=True)
    selected = max(candidates, key=lambda c: (c['train_gain']-c['train_loss'], c['train_gain'],
        c['training_survival_seconds'], -abs(c['amplitude'])))
    save(out/'selected.json', selected)
    for timing, onset in enumerate(ONSETS):
        baseline = out/f'baseline-t{timing}-g0'
        # Zero addition uses the selected amplitude only to match replay duration.
        experiment(rig, anchor, vectors, legs, out, f'replay-control-t{timing}', onset=onset,
            amplitude=selected['amplitude'], replay_folder=baseline, baseline_folder=baseline)
        experiment(rig, anchor, vectors, legs, out, f'replay-probe-t{timing}', onset=onset,
            profile=selected['profile'], amplitude=selected['amplitude'], replay_folder=baseline, baseline_folder=baseline)
        experiment(rig, anchor, vectors, legs, out, f'stronger-t{timing}', onset=onset, group=2,
            profile=selected['profile'], amplitude=selected['amplitude'], baseline_folder=out/f'baseline-t{timing}-g2')
    experiment(rig, anchor, vectors, legs, out, 'quiet-probe', quiet=True,
        profile=selected['profile'], amplitude=selected['amplitude'])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items()), 'Protected input changed'
    save(out/'completed.json', dict(wall_seconds=time.perf_counter()-start, selected=selected,
        matrix=matrix, rollouts=63, cases=1008, optimizer_updates=0, sources_unchanged=True,
        outputs={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', selected, flush=True)


if __name__ == '__main__':
    main()
