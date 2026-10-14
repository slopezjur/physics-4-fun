"""Independent saved-evidence audit and response summaries for matched branches."""
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_matched_feedback import OUT, PRIOR, FOCUS
from .probe_arm_action_response import fingerprint
from .verify_arm_proposal_30_rounding import gates
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR as POSES_ROOT
from .predictive_target_support import blend


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
    outcomes, responses, sensitivities = [], [], []
    teacher_queries = 0
    for r in report['results']:
        folder, baseline_folder = OUT/r['folder'], OUT/r['baseline']
        gates(model, poses, folder, r)
        z, baseline = np.load(folder/'trace.npz'), np.load(baseline_folder/'trace.npz')
        cmd, original_cmd = np.load(folder/'commands.npz'), np.load(baseline_folder/'commands.npz')
        metadata = json.loads((folder/'branch.json').read_text())
        original_meta = json.loads((baseline_folder/'branch.json').read_text())
        state, original_state = np.load(folder/'switch-states.npz'), np.load(baseline_folder/'switch-states.npz')
        assert metadata['states'] == original_meta['states']
        for i, switch in enumerate(r['switches']):
            for key in ('qpos', 'issued', 'alive', 'done'):
                assert np.array_equal(z[key][:switch, i], baseline[key][:switch, i])
            assert np.array_equal(cmd['qvel'][:switch, i], original_cmd['qvel'][:switch, i])
            keys = [k for k in state.files if k.startswith(f'{i}__')]
            values = {k.split('__', 1)[1]: state[k] for k in keys}
            assert fingerprint(values) == metadata['states'][str(i)]
            assert all(np.array_equal(state[k], original_state[k]) for k in keys)
        config = r['config']
        if config['origin'] == config['destination'] and config['pulse'] is None:
            name = f'{config["origin"]}-{config["seed"]}'+('-varied' if config['variant'] else '-nominal')
            archived = np.load(PRIOR/name/'trace.npz')
            assert all(np.array_equal(z[k], archived[k]) for k in ('qpos', 'issued', 'alive', 'done'))
        assert int(cmd['teacher'].sum()) == metadata['teacher_queries']
        teacher_queries += metadata['teacher_queries']
        if config['destination'] == 'teacher':
            assert config['pulse'] is None
            for i, switch in enumerate(r['switches']):
                assert not cmd['teacher'][:switch, i].any()
                if i not in FOCUS:
                    assert not cmd['teacher'][:, i].any()
                    assert all(np.array_equal(z[k][:, i], baseline[k][:, i]) for k in ('qpos', 'issued', 'alive', 'done'))
        else:
            assert not cmd['teacher'].any()
        raw, source, previous, goal = (torch.tensor(cmd[k]) for k in ('raw_target', 'source', 'previous', 'goal'))
        assert np.array_equal(cmd['raw_target'], cmd['goal']+cmd['residual'])
        onset = 1.2 if config['variant'] else 1.
        authority = torch.tensor([float(blend(0., 1., f*dt-onset, 1.)) for f in range(len(z['time']))], dtype=torch.float32)
        fade = ((goal-reference).abs().amax(-1)/.05).clamp(0, 1)
        desired = (raw-source).clamp(-1, 1)*authority[:, None, None]*fade[..., None]
        nominal = previous+(desired-previous).clamp(-.02, .02)
        correction = torch.tensor(z['correction'])
        scaled = correction*authority[:, None, None]*fade[..., None]
        corrected = (previous+(nominal+scaled-previous).clamp(-.02, .02)).clamp(-1, 1)
        addition = torch.where(correction != 0, corrected, nominal)
        addition[torch.tensor(z['quiet'] | ~z['alive'])] = 0
        assert np.array_equal(addition.numpy(), z['addition'])
        assert np.array_equal((source+addition).clamp(lower, upper).numpy(), z['issued'])
        expected_correction = np.zeros_like(z['correction'])
        if config['pulse'] is not None:
            motor = protocol['motors'][config['pulse']['joint']]
            for i in FOCUS:
                expected_correction[r['switches'][i], i, motor] = config['pulse']['sign']*.005
        assert np.array_equal(expected_correction, z['correction'])
        for item in metadata['input_probes']:
            for key in ('raw', 'source', 'issued'):
                minus, plus = np.array(item['minus'][key], dtype=np.float32), np.array(item['plus'][key], dtype=np.float32)
                derivative = (plus-minus)/(2*item['epsilon'])
                assert np.array_equal(derivative, np.array(item['derivative'][key], dtype=np.float32))
            if item['feature'] == 'oldest_queue':
                assert np.array_equal(item['minus']['raw'], item['plus']['raw'])
            m = protocol['motors'][item['joint']]
            sensitivities.append(dict(folder=folder.name, frame=item['frame'], world=item['world'],
                head=item['head'], joint=item['joint'], feature=item['feature'],
                own_raw_derivative=item['derivative']['raw'][m],
                own_source_derivative=item['derivative']['source'][m],
                own_issued_derivative=item['derivative']['issued'][m],
                max_issued_derivative=float(np.max(np.abs(item['derivative']['issued'])))))
        outcomes.append(dict(folder=folder.name, config=config, full=r['complete'],
            focus_full=sum(r['cases'][i]['complete'] for i in FOCUS),
            focus_cases=[r['cases'][i] for i in FOCUS],
            teacher_queries=metadata['teacher_queries']))
        for i in FOCUS:
            switch = r['switches'][i]
            measures = []
            for after in (3, 12, 30, 60):
                frame = switch+after-1
                q_delta = z['qpos'][frame, i, qi]-baseline['qpos'][frame, i, qi]
                v_delta = cmd['qvel'][frame, i, vi]-original_cmd['qvel'][frame, i, vi]
                measures.append(dict(frames_after=after, alive=bool(z['alive'][frame, i] and not z['done'][frame, i]),
                    joint_delta_rms=float(np.sqrt(np.mean(q_delta**2))),
                    joint_delta_max=float(abs(q_delta).max()),
                    velocity_delta_rms=float(np.sqrt(np.mean(v_delta**2))),
                    com_delta=float(np.linalg.norm(z['com'][frame, i, :2]-baseline['com'][frame, i, :2]))))
            response = dict(folder=folder.name, pose=i, switch=switch, config=config, samples=measures)
            if config['pulse'] is not None:
                motor = protocol['motors'][config['pulse']['joint']]
                response['transmitted_addition'] = float(z['addition'][switch, i, motor]-z['nominal'][switch, i, motor])
                response['issued_delta_at_pulse'] = float(z['issued'][switch, i, motor]-baseline['issued'][switch, i, motor])
            responses.append(response)
    assert len(outcomes) == 32 and len(sensitivities) == 288
    assert len([r for r in responses if r['config']['pulse'] is not None]) == 24
    assert report['training_updates'] == 0
    save(OUT/'audit.json', dict(passed=True, outcomes=outcomes, responses=responses, sensitivities=sensitivities,
        teacher_queries=teacher_queries, teacher_diagnostic_cases=24, input_probe_pairs=len(sensitivities),
        pulse_probes=24, physical_cases=256, exact_prefixes=True, exact_switch_states=True,
        exact_command_projection=True, training_updates=0, arena_changed=False,
        auditor_sha256=digest(Path(__file__))))
    print('AUDIT PASSED', len(outcomes), 'episodes;', teacher_queries, 'teacher queries', flush=True)


if __name__ == '__main__':
    main()
