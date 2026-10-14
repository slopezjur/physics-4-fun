"""Reconstruct correction admission and issued commands from saved evidence."""
import json
from pathlib import Path

import numpy as np
import torch

from .collect_arm_corrective_takeovers import OUT, ARENA
from .probe_arm_execution_distribution import setup
from .train_arm_correction_ppo import CorrectionEnvironment
from .train_arm_relative_proposal import load_head
from .challenge_arm_proposals import HEADS
from .run_support_skill_study import digest, save


@torch.no_grad()
def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    for p, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(p)) == h, p
    env, anchor, _, poses, _ = setup()
    c = env.engine.policy_control
    results, admitted = [], []
    for r in report['results']:
        folder = OUT/r['folder']
        spec = next(s for s in protocol['specifications'] if r['folder'].startswith(s['actor']) and s['variant'] == r['config']['variant'])
        adapter = CorrectionEnvironment(env, anchor, load_head(HEADS[spec['actor']]), poses, variant=spec['variant'])
        p = adapter.physical
        z, labels = np.load(folder/'trace.npz'), np.load(folder/'labels.npz')
        control = np.load(OUT/r['control']/'trace.npz')
        archived = np.load(Path(spec['archived'])/'trace.npz')
        for key in ('qpos', 'issued', 'alive', 'done'):
            assert np.array_equal(control[key], archived[key])
        metadata = json.loads((folder/'takeover.json').read_text())
        baseline = json.loads((OUT/r['control']/'takeover.json').read_text())
        for i, switch in enumerate(metadata['switches']):
            for key in ('qpos', 'issued', 'alive', 'done'):
                assert np.array_equal(z[key][:switch, i], control[key][:switch, i])
            key = f'{i}-{switch}'
            if key in metadata['states']:
                assert metadata['states'][key] == baseline['states'][key]
        assert not labels['teacher'][:, 6:].any()
        assert int(labels['teacher'].sum()) == metadata['unchanged_queries']
        assert np.max(abs(labels['raw_target'] - labels['requested'])) < 2e-7
        assert np.array_equal(labels['raw_target'], labels['goal'] + labels['residual'])
        raw, source, previous = (torch.tensor(labels[k]) for k in ('raw_target', 'source', 'previous'))
        scale = torch.tensor(z['scale'])
        desired = (raw-source).clamp(-1, 1)*scale[..., None]
        # The adapter multiplies authority then fade separately; reproduce that order.
        goal = torch.tensor(labels['goal'])
        fade = ((goal-p.reference[:8, p.qi]).abs().amax(-1)/.05).clamp(0, 1)
        desired = (raw-source).clamp(-1, 1)*p.authority[:-1, None, None]*fade[..., None]
        addition = previous+(desired-previous).clamp(-.02, .02)
        addition[torch.tensor(labels['quiet'] | ~z['alive'])] = 0
        issued = (source+addition).clamp(c.lower, c.upper)
        assert np.array_equal(addition.numpy(), z['addition'])
        assert np.array_equal(issued.numpy(), z['issued'])
        hold = (z['time'] > r['hold_end']-2) & (z['time'] <= r['hold_end'])
        accepted = []
        for i, case in enumerate(r['cases']):
            motor = np.sqrt(np.mean((z['qpos'][:, i, p.qi]-poses[i, 1, p.qi])**2, -1))
            com = np.linalg.norm(z['com'][:, i, :2]-p.gate.com[i], axis=-1)
            feet = np.linalg.norm(z['xy'][:, i]-p.initial_xy[i].numpy(), axis=-1).max(-1)
            stable = bool(z['alive'][hold, i].all() and z['support'][hold, i].all())
            returned = bool(z['alive'][-1, i] and not z['done'][-1, i] and z['stand'][-int(np.ceil(2/env.dt)):, i].all())
            complete = stable and returned and bool((motor[hold] <= .05).all() and (com[hold] <= .02).all() and (feet[hold] <= .02).all())
            assert (stable, returned, complete) == (case['stable_hold'], case['returned_stand'], case['complete'])
            accepted.append(complete)
        use = labels['teacher'] & z['alive'] & np.array(accepted)[None]
        frame, world = np.nonzero(use)
        data = np.load(folder/'admitted-samples.npz')
        assert np.array_equal(data['frame'], frame) and np.array_equal(data['world'], world)
        assert not data['validation'].any() and (world < 6).all()
        for key in ('x', 'raw_target', 'source', 'previous', 'goal', 'quiet'):
            assert np.array_equal(data[key], labels[key][use])
        assert np.array_equal(data['issued'], z['issued'][use])
        assert (data['episode'] == folder.name).all()
        admitted.append(dict(data))
        results.append(dict(folder=folder.name, complete=sum(accepted), admitted_samples=len(world),
                            teacher_cases=np.flatnonzero(labels['teacher'].any(0)).tolist(),
                            admitted_cases=np.unique(world).tolist()))
    merged = np.load(OUT/'training-samples.npz')
    for key in merged.files:
        assert np.array_equal(merged[key], np.concatenate([d[key] for d in admitted]))
    assert len(merged['world']) == report['admitted_samples']
    save(OUT/'audit.json', dict(passed=True, results=results, admitted_samples=len(merged['world']),
        arena_hash=digest(ARENA), training_updates=0, heldout_samples=0,
        exact_archived_replay=True, exact_prefix_and_switch_state=True, exact_command_projection=True))
    print('AUDIT PASSED', results, flush=True)


if __name__ == '__main__':
    main()
