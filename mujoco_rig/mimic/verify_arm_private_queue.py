"""Audit reconstructed query parity and conditional hidden-queue target changes."""
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_private_queue import OUT, BRANCHES
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR as POSES_ROOT
from .predictive_target_support import blend


def main():
    from .rig import Rig
    protocol = json.loads((OUT/'protocol.json').read_text())
    report = json.loads((OUT/'report.json').read_text())
    for path, h in protocol['inputs'].items():
        assert digest(Path(path)) == h, path
    assert len(report['records']) == 120 and len(report['reconstruction_parity']) == 24
    assert all(p['maximum_raw_error'] < 2e-6 for p in report['reconstruction_parity'])
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    qi = rig.model.jnt_qposadr[joints]
    bounds = torch.tensor(rig.model.jnt_range[joints], dtype=torch.float32)
    dt = rig.model.opt.timestep*4
    poses = np.load(POSES_ROOT/'goals.npz')['poses']
    cases, cache = [], {}
    for row in report['records']:
        folder = row['folder']
        if folder not in cache:
            cache[folder] = dict(np.load(BRANCHES/folder/'commands.npz'))
        commands = cache[folder]
        frame, world = row['frame'], row['world']
        baseline = next(x for x in report['records'] if x['folder'] == folder and x['world'] == world and x['joint'] is None)
        previous = torch.tensor(commands['previous'][frame, world])
        source = torch.tensor(row['source'], dtype=torch.float32)
        goal = torch.tensor(commands['goal'][frame, world])
        reference = torch.tensor(poses[world, 0, qi], dtype=torch.float32)
        amount = torch.tensor(float(blend(0., 1., frame*dt-(1.2 if '-varied-' in folder else 1.), 1.)), dtype=torch.float32)
        fade = ((goal-reference).abs().max()/.05).clamp(0, 1)
        targets = {'teacher': dict(raw=row['teacher_raw'], issued=row['teacher_issued']), **row['head_outputs']}
        for name, values in targets.items():
            raw = torch.tensor(values['raw'], dtype=torch.float32)
            desired = (raw-source).clamp(-1, 1)*amount*fade
            addition = previous+(desired-previous).clamp(-.02, .02)
            issued = (source+addition).clamp(bounds[:, 0], bounds[:, 1])
            assert np.array_equal(issued.numpy(), np.array(values['issued'], dtype=np.float32))
            if name != 'teacher':
                assert values['raw'] == baseline['head_outputs'][name]['raw']
        if row['joint'] is None:
            executed = np.load(BRANCHES/folder.replace('-continue', '-teacher')/'commands.npz')
            error = np.max(abs(np.array(row['teacher_raw'], dtype=np.float32)-executed['raw_target'][frame, world]))
            assert error < 2e-6
            assert row['actual_queue_delta'] == 0
        else:
            assert abs(row['actual_queue_delta']-row['sign']*.005) < 1e-7
            teacher_raw = np.array(row['teacher_raw'])-baseline['teacher_raw']
            teacher_issued = np.array(row['teacher_issued'])-baseline['teacher_issued']
            head_changes = {head: float(np.max(abs(np.array(values['issued'])-baseline['head_outputs'][head]['issued'])))
                            for head, values in row['head_outputs'].items()}
            cases.append(dict(folder=folder, world=world, joint=row['joint'], sign=row['sign'],
                source_delta_max=float(np.max(abs(np.array(row['source'])-baseline['source']))),
                teacher_raw_delta_max=float(abs(teacher_raw).max()),
                teacher_issued_delta_max=float(abs(teacher_issued).max()),
                learned_issued_delta_max=head_changes,
                teacher_raw_l2=float(np.linalg.norm(teacher_raw))))
    assert len(cases) == 96
    metrics = {}
    for key in ('source_delta_max', 'teacher_raw_delta_max', 'teacher_issued_delta_max'):
        values = [r[key] for r in cases]
        metrics[key] = dict(min=min(values), median=float(np.median(values)), max=max(values),
                            above_1e_6=sum(v > 1e-6 for v in values), above_1mrad=sum(v > .001 for v in values))
    for head in ('replay', 'correction05'):
        values = [r['learned_issued_delta_max'][head] for r in cases]
        metrics[head+'_issued_delta_max'] = dict(min=min(values), median=float(np.median(values)), max=max(values))
    save(OUT/'audit.json', dict(passed=True, conditional_cases=cases, metrics=metrics,
        maximum_reconstruction_error=max(p['maximum_raw_error'] for p in report['reconstruction_parity']),
        masked_raw_outputs_identical=True, command_projection_exact=True, physical_rollouts=0,
        training_updates=0, report_sha256=digest(OUT/'report.json'), auditor_sha256=digest(Path(__file__))))
    print('QUEUE AUDIT PASSED', metrics, flush=True)


if __name__ == '__main__':
    main()
