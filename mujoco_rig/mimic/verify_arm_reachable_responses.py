"""Reconstruct real delay timing, teacher eligibility and paired-data admission."""
import json
from pathlib import Path

import numpy as np
import torch

from .collect_arm_reachable_responses import OUT, MATCHED, PRIOR, FOCUS, OFFSETS
from .probe_arm_action_response import fingerprint
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR as POSES_ROOT
from .predictive_target_support import blend


def state(z, event):
    return {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(event+'__')}


def main():
    from .rig import Rig
    torch.set_num_threads(2)
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    for path, h in {**protocol['inputs'], **report['outputs']}.items():
        assert digest(Path(path)) == h, path
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    model, dt = rig.model, rig.model.opt.timestep*4
    joints = model.actuator_trnid[rig.actuators, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    lower = torch.tensor(model.jnt_range[joints, 0], dtype=torch.float32)
    upper = torch.tensor(model.jnt_range[joints, 1], dtype=torch.float32)
    poses = np.load(POSES_ROOT/'goals.npz')['poses']
    reference = torch.tensor(poses[:8, 0, qi], dtype=torch.float32)
    outcomes, candidates, expected_pairs = [], [], []
    teacher_queries = 0
    for r in report['results']:
        folder, config = OUT/r['folder'], r['config']
        outcomes.append(dict(folder=folder.name, **gates(model, poses, folder, r)))
        z, cmd, states = (np.load(folder/name) for name in ('trace.npz', 'commands.npz', 'states.npz'))
        meta = json.loads((folder/'branch.json').read_text())
        old = np.load(PRIOR/r['spec']/'trace.npz')
        original_states = np.load(MATCHED/(r['spec']+'-continue')/'switch-states.npz')
        assert not cmd['teacher'][:, [1, 2, 3, 5, 6, 7]].any()
        assert int(cmd['teacher'].sum()) == meta['teacher_queries']
        teacher_queries += meta['teacher_queries']
        for world in range(8):
            end = len(z['time']) if world not in FOCUS else meta['switches'][world] if config['pulse'] is None else meta['pulses'][world]
            assert all(np.array_equal(z[k][:end, world], old[k][:end, world]) for k in ('qpos', 'issued', 'alive', 'done'))
            if world not in FOCUS:
                continue
            pulse, switch = meta['pulses'][world], meta['switches'][world]
            assert switch == pulse+3 and not cmd['teacher'][:switch, world].any()
            for label, frame in (('pulse', pulse), ('takeover', switch)):
                s = state(states, f'{world}-{label}')
                assert fingerprint(s) == meta['states'][f'{world}-{label}']
                chronological = np.concatenate([s['pending'][(int(s['cursor'])+slot) % 2] for slot in range(2)])
                assert np.array_equal(chronological, cmd['x'][frame, world, 95:229])
                assert np.array_equal(s['observation'], cmd['x'][frame, world, :374])
                assert np.allclose(cmd['x'][frame, world, qi+8], z['qpos'][frame-1, world, qi], atol=1e-7, rtol=0)
                assert np.allclose(cmd['x'][frame, world, vi+48], cmd['qvel'][frame-1, world, vi]*.1, atol=1e-7, rtol=0)
                if label == 'pulse':
                    archived = state(original_states, str(world))
                    assert s.keys() == archived.keys()
                    assert all(np.array_equal(s[k], archived[k]) for k in s)
        raw, source, previous, goal = (torch.tensor(cmd[k]) for k in ('raw_target', 'source', 'previous', 'goal'))
        assert np.array_equal(cmd['raw_target'], cmd['goal']+cmd['residual'])
        assert np.array_equal(cmd['goal'], cmd['x'][:, :, 374:420][:, :, qi])
        authority = torch.tensor([float(blend(0., 1., f*dt-(1.2 if config['variant'] else 1.), 1.)) for f in range(len(z['time']))])
        fade = ((goal-reference).abs().amax(-1)/.05).clamp(0, 1)
        nominal = previous+((raw-source).clamp(-1, 1)*authority[:, None, None]*fade[..., None]-previous).clamp(-.02, .02)
        correction = torch.tensor(z['correction'])
        projected = (previous+(nominal+correction*authority[:, None, None]*fade[..., None]-previous).clamp(-.02, .02)).clamp(-1, 1)
        addition = torch.where(correction != 0, projected, nominal)
        addition[torch.tensor(z['quiet'] | ~z['alive'])] = 0
        assert np.array_equal(addition.numpy(), z['addition'])
        assert np.array_equal((source+addition).clamp(lower, upper).numpy(), z['issued'])
        expected_correction = np.zeros_like(z['correction'])
        if config['pulse'] is None:
            assert not correction.any()
            continue
        motor = protocol['motors'][config['pulse']['joint']]
        baseline = next(a for a in report['results'] if a['spec'] == r['spec'] and a['config']['pulse'] is None)
        base = OUT/baseline['folder']
        bz, bc, bs = (np.load(base/name) for name in ('trace.npz', 'commands.npz', 'states.npz'))
        admissions = json.loads((folder/'admission.json').read_text())
        for world in FOCUS:
            pulse, switch = meta['pulses'][world], meta['switches'][world]
            expected_correction[pulse, world, motor] = config['pulse']['sign']*.005
            a, b = state(states, f'{world}-pulse'), state(bs, f'{world}-pulse')
            assert all(np.array_equal(a[k], b[k]) for k in a)
            for key in ('qpos', 'issued', 'alive', 'done'):
                assert np.array_equal(z[key][:pulse, world], bz[key][:pulse, world])
            assert np.array_equal(z['qpos'][pulse:pulse+2, world], bz['qpos'][pulse:pulse+2, world])
            assert np.array_equal(cmd['qvel'][pulse:pulse+2, world], bc['qvel'][pulse:pulse+2, world])
            delta = float(z['issued'][pulse, world, motor]-bz['issued'][pulse, world, motor])
            body = float(abs(z['qpos'][pulse+2, world]-bz['qpos'][pulse+2, world]).max())
            admitted = bool(r['cases'][world]['complete'] and baseline['cases'][world]['complete'] and abs(delta) > 1e-5 and body > 1e-10)
            frames = [switch+off for off in OFFSETS if admitted and cmd['teacher'][switch+off, world]
                      and bc['teacher'][switch+off, world] and z['alive'][switch+off, world] and bz['alive'][switch+off, world]]
            saved = next(a for a in admissions if a['world'] == world)
            assert (saved['admitted'], saved['issued_delta'], saved['native_position_delta'], saved['frames']) == (admitted, delta, body, frames)
            candidates.append(dict(folder=folder.name, seed=config['seed'], variant=config['variant'], **saved))
            for frame in frames:
                assert np.array_equal(cmd['x'][frame, world, 374:], bc['x'][frame, world, 374:])
                pair = {f'{k}{side}': values[k][frame, world] for side, values in ((0, bc), (1, cmd))
                        for k in ('x', 'raw_target', 'source', 'previous')}
                pair.update(goal=cmd['goal'][frame, world], world=world, frame=frame, offset=frame-switch,
                    pulse_frame=pulse, issued_delta=delta, seed=config['seed'], variant=config['variant'],
                    joint=config['pulse']['joint'], sign=config['pulse']['sign'], baseline=baseline['folder'],
                    branch=folder.name, validation=False)
                expected_pairs.append(pair)
        assert np.array_equal(expected_correction, z['correction'])
    assert len(report['results']) == 20 and len(candidates) == 32
    assert report['paired_examples'] == len(expected_pairs) <= 160
    if expected_pairs:
        dataset = np.load(OUT/'response-pairs.npz')
        assert set(dataset.files) == set(expected_pairs[0])
        for key in dataset.files:
            assert np.array_equal(dataset[key], np.stack([p[key] for p in expected_pairs])), key
        assert not dataset['validation'].any() and set(dataset['world']) <= {0, 4}
    targeted = [c for r in report['results'] for c in r['cases'] if c['pose'] in FOCUS]
    result = dict(passed=True, physical_cases=sum(r['cases'] for r in outcomes), targeted_teacher_cases=len(targeted),
        targeted_teacher_complete=sum(c['complete'] for c in targeted), pulsed_candidates=32,
        admitted_pulsed_cases=sum(c['admitted'] for c in candidates), paired_examples=len(expected_pairs),
        teacher_queries=teacher_queries, heldout_teacher_queries=0, heldout_training_rows=0,
        native_delay_verified=True, exact_prefixes=True, exact_pre_pulse_states=True, exact_command_projection=True,
        training_updates=0, ppo_transitions=0, arena_changed=False, outcomes=outcomes, candidates=candidates,
        auditor_sha256=digest(Path(__file__)))
    assert result['physical_cases'] == 160 and len(targeted) == 40
    save(OUT/'audit.json', result)
    print('AUDIT PASSED', len(expected_pairs), 'pairs;', result['targeted_teacher_complete'], '/40 targeted gates;', teacher_queries, 'queries', flush=True)


if __name__ == '__main__':
    main()
