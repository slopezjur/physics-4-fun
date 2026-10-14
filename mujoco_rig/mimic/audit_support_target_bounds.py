"""Check static support against the bounded target-PD command contract."""
import argparse
import json
from pathlib import Path

import numpy as np

from .diagnose_support_loss import digest
from .rig import Rig
from .static_support import solve_support, solve_target_support, target_motor_bounds
from .target_control33 import DelayedTargetControl33


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    base = Path('logs/mimickit-training')
    source = base / 'single-support-interval-20261007-01'
    rig_path = base / 'stand-ball-checkpoints-20261006-01/dummy.xml'
    rig = Rig.load(rig_path, policy_exclude=())
    model = rig.model
    joints = model.actuator_trnid[rig.actuators, 0]
    qi = model.jnt_qposadr[joints]
    names = [model.joint(int(j)).name for j in joints]
    paths = [rig_path, source/'prepared-feedback/trace.npz', source/'target-admission.json',
             Path(__file__), Path(__file__).with_name('static_support.py'),
             Path(__file__).with_name('target_control33.py'), Path(__file__).with_name('rig.py')]
    hashes = {str(p): digest(p) for p in paths}
    args.out.mkdir(parents=True, exist_ok=False)
    admitted = json.loads((source/'target-admission.json').read_text())
    cases = []
    with np.load(source/'prepared-feedback/trace.npz') as z:
        for index, world in enumerate((0, 1, 8, 9)):
            pose = z['requested_qpos'][0, index].astype(float)
            side = 'R' if world < 8 else 'L'
            ankle = names.index(f'Foot_{side}_rz')
            bounds = target_motor_bounds(rig, pose)
            ankle_only = np.tile((-1., 1.), (len(joints), 1))
            ankle_only[ankle] = bounds[ankle]
            except_ankle = bounds.copy()
            except_ankle[ankle] = (-1, 1)
            results = {}
            for label, limits in (('torque_only', None), ('all_target_bounds', bounds),
                                  ('ankle_only', ankle_only), ('except_ankle', except_ankle)):
                results[label] = solve_support(rig, pose, support_names=(f'Foot_{side}', f'Toe_{side}'),
                                               normalized_motor_bounds=limits)
            original = admitted[str(world)][-1]['static']
            assert original['feasible'] and results['torque_only']['feasible']
            assert solve_target_support(rig, pose, support_names=(f'Foot_{side}', f'Toe_{side}'))['feasible'] == results['all_target_bounds']['feasible']
            torque = np.asarray(original['normalized_motor_torque'])
            target = pose[qi] + torque/DelayedTargetControl33.gain
            q = pose[qi[ankle]]
            cases.append(dict(world=world, support_ankle=names[ankle], angle_rad=float(q),
                              target_limits_rad=model.jnt_range[joints[ankle]].tolist(),
                              original_normalized_torque=float(torque[ankle]),
                              original_target_rad=float(target[ankle]),
                              available_normalized_torque=bounds[ankle].tolist(), results=results))
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    result = dict(schema='support_target_bound_admission_audit_v1', inputs=hashes, cases=cases,
                  new_physical_cases=0, physics_substeps=0, training_updates=0,
                  scope='Necessary quasi-static support check at zero velocity. Uses the existing prospective sole-contact LP and passive-wrist tolerance; does not certify dynamic stability. Source-relative bounds can only narrow this physical-target envelope.')
    (args.out/'report.json').write_text(json.dumps(result, indent=2) + '\n')
    for path in paths[3:]:
        (args.out/path.name).write_bytes(path.read_bytes())
    for case in cases:
        print(case['world'], {k: v['feasible'] for k, v in case['results'].items()},
              case['original_normalized_torque'], case['available_normalized_torque'])


if __name__ == '__main__':
    main()
