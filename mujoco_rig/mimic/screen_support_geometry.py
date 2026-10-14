"""Bounded private-IK screen for target-PD-feasible single-support geometry."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np

from .runtime import activate
from .diagnose_support_loss import digest


def pose_metrics(model, pose, reference, side):
    data = mujoco.MjData(model)
    bodies = [model.body(name).id for name in ('Spine', 'Chest', 'Head')]
    data.qpos[:] = reference
    mujoco.mj_kinematics(model, data)
    baseline = data.xpos[bodies, 2] - data.xpos[1, 2]
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    tilts = np.rad2deg(np.arccos(np.clip(data.xmat[bodies, 8], -1, 1)))
    deficits = np.maximum(0, baseline - (data.xpos[bodies, 2] - data.xpos[1, 2]))
    pelvis_tilt = np.rad2deg(np.arccos(np.clip(data.xmat[1, 8], -1, 1)))
    clearances = []
    for suffix in ('L', 'R'):
        heights = []
        for part in ('Foot', 'Toe'):
            g = model.geom(f'g_{part}_{suffix}').id
            heights.append(data.geom_xpos[g, 2] - abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
        clearances.append(float(min(heights)))
    ankle = model.joint(f'Foot_{("R", "L")[side]}_rz').id
    angle = pose[model.jnt_qposadr[ankle]]
    margin = float(min(angle-model.jnt_range[ankle, 0], model.jnt_range[ankle, 1]-angle))
    return dict(pelvis_height_m=float(pose[2]), pelvis_tilt_degrees=float(pelvis_tilt),
                upper_tilt_degrees=tilts.tolist(), upper_clearance_deficit_m=deficits.tolist(),
                whole_foot_clearance_m=clearances, stance_ankle_margin_rad=margin,
                posture_pass=bool(pose[2] >= .75 and pelvis_tilt <= 15
                                  and (tilts <= [20, 20, 30]).all() and (deficits <= [.05, .06, .10]).all()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .support_skill_core import fit_goal
    from .support_skill_physical_teacher import clear_arms
    from .static_support import solve_target_support
    root = Path.cwd().resolve()
    foundation = Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01')
    rig = Rig.load(foundation/'dummy.xml', policy_exclude=())
    with np.load(foundation/'stationary_reference.npz') as z:
        references = z['qpos'].astype(float)
    cleared = {pose: clear_arms(rig.model, references[pose], .10) for pose in (0, 1)}
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for module in tuple(sys.modules.values()):
        path = Path(getattr(module, '__file__', '') or '').resolve()
        if path.is_file() and path.suffix == '.py' and path.is_relative_to(root) and '.venv' not in path.parts:
            relative = path.relative_to(root)
            sources[str(relative)] = digest(path)
            destination = args.out/'sources'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    inputs = {str(p): digest(p) for p in (foundation/'dummy.xml', foundation/'stationary_reference.npz')}
    protocol = dict(shifts_m=[.08, .10, .11, .12], lean_degrees=[0, 4, 8, 12], lift_m=.02,
                    poses=[0, 1], sides=[0, 1], minimum_ankle_margin_rad=.04,
                    maximum_fit_error_m=.001, maximum_penetration_m=.0002,
                    minimum_swing_clearance_m=.018, selection='largest worst ankle angle margin, then lower lean',
                    physics_substeps=0, training_updates=0, inputs=inputs, sources=sources)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    variants = []
    for shift, lean in itertools.product(protocol['shifts_m'], protocol['lean_degrees']):
        cases = []
        for side, pose_id in itertools.product((0, 1), (0, 1)):
            pose, fit = fit_goal(rig.model, cleared[pose_id], shift=shift*(1 if side == 0 else -1),
                                 lift=.02, distance=0., swing=side, lean_radians=np.deg2rad(lean))
            metrics = pose_metrics(rig.model, pose, references[pose_id], side)
            try:
                support = solve_target_support(rig, pose, support_names=tuple(f'{part}_{("R", "L")[side]}' for part in ('Foot', 'Toe')))
            except ValueError as exc:
                support = dict(feasible=False, reason=str(exc))
            gates = dict(posture=metrics['posture_pass'], fit=fit['foot_error_m'] <= .001 and fit['com_error_m'] <= .001,
                         penetration=fit['penetration_m'] <= .0002,
                         clearance=metrics['whole_foot_clearance_m'][side] >= .018,
                         ankle_margin=metrics['stance_ankle_margin_rad'] >= .04,
                         target_pd_static=support['feasible'])
            cases.append(dict(pose=pose_id, swing_side=side, fit=fit, metrics=metrics, support=support,
                              gates=gates, admitted=all(gates.values()), qpos=pose.tolist()))
        variant = dict(shift_m=shift, lean_degrees=lean, cases=cases,
                       admitted=all(c['admitted'] for c in cases),
                       minimum_ankle_margin_rad=min(c['metrics']['stance_ankle_margin_rad'] for c in cases))
        variants.append(variant)
        print(shift, lean, [(c['admitted'], [k for k, v in c['gates'].items() if not v]) for c in cases], flush=True)
    admitted = sorted((v for v in variants if v['admitted']), key=lambda v: (-v['minimum_ankle_margin_rad'], v['lean_degrees']))
    selected = {k: admitted[0][k] for k in ('shift_m', 'lean_degrees', 'minimum_ankle_margin_rad')} if admitted else None
    result = dict(completed=True, geometry_cases=64, new_physical_cases=0, selected=selected, variants=variants)
    for path, expected in {**inputs, **sources}.items():
        assert digest(Path(path)) == expected, path
    (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.out/'final-verification.json').write_text(json.dumps(dict(inputs_unchanged=True,
        outputs={p.name: digest(p) for p in args.out.iterdir() if p.is_file()}), indent=2)+'\n')
    print('Selected:', selected)


if __name__ == '__main__':
    main()
