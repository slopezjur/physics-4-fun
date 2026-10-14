"""Reflect licensed single-step diagnostics and repeat physical admission checks."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .baseline import sha256
from .runtime import activate

REFLECTION = np.diag([1., -1., 1.])


def other_side(name):
    return name.replace('_L', '_R') if '_L' in name else name.replace('_R', '_L')


def mirror_poses(model, poses):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[3] = 1.
    mujoco.mj_kinematics(model, data)
    result = poses.copy()
    result[:, 1] *= -1.
    result[:, [4, 6]] *= -1.
    for joint in range(1, model.njnt):
        target = model.joint(other_side(model.joint(joint).name)).id
        alignment = float(data.xaxis[target] @ (-REFLECTION @ data.xaxis[joint]))
        if abs(alignment) < 1. - 1e-8:
            raise ValueError('Rig hinge axes do not support an exact sagittal reflection')
        result[:, model.jnt_qposadr[target]] = np.sign(alignment) * poses[:, model.jnt_qposadr[joint]]
    return result


def mirror_error(model, source, mirrored):
    data = mujoco.MjData(model)
    pairs = [model.body(other_side(model.body(i).name)).id for i in range(1, model.nbody)]
    worst = 0.
    for before, after in zip(source, mirrored):
        data.qpos[:] = before
        mujoco.mj_kinematics(model, data)
        expected = data.xpos[1:].copy() @ REFLECTION
        data.qpos[:] = after
        mujoco.mj_kinematics(model, data)
        worst = max(worst, float(np.linalg.vector_norm(expected - data.xpos[pairs], axis=1).max()))
    return worst


def attribution_text(metadata):
    source = metadata['source']
    return ('# Diagnostic reference attribution\n\n'
        f"{source['credit']}.\n\nSource: {source['source']}\n\n"
        f"License: {source['license']} — {source['license_url']}\n\n"
        'Adaptations: procedural single-step calibration, sagittal reflection, '
        'recomputed velocities and physical revalidation. These are diagnostic '
        'poses, not recorded impact recovery.\n\n'
        f"Source diagnostic SHA256: {metadata['mirror_provenance']['source_diagnostic_sha256']}.\n")


def generate(args):
    activate(args.mimickit)
    from .force_retarget import reference_velocities
    from .placed_step import single_step_checks
    from .retarget_steps import validate_motion
    from .rig import Rig
    from .track_transfer import CalibrationReference

    if args.out.exists():
        raise ValueError('Require a new mirror directory')
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    reference = CalibrationReference(args.source, rig)
    if not reference.placed_step:
        raise ValueError('Require a physically checked single-step diagnostic')
    metadata = json.loads((args.source/'report.json').read_text())
    if 'arm_fit' in metadata:
        metadata['source_arm_fit'] = metadata.pop('arm_fit')
    with np.load(reference.path) as arrays:
        source = arrays['qpos']
        q = mirror_poses(rig.model, source)
        targets = arrays['foot_targets'][:, [1, 0]].copy() @ REFLECTION
        contact = arrays['contact'][:, [1, 0]].copy()
        soles = arrays['sole_contact'][:, [1, 0, 3, 2]].copy()
        dt = float(arrays['dt'])
    velocity = reference_velocities(rig.model, q, dt)
    error = mirror_error(rig.model, source, q)
    checks, metrics = validate_motion(rig, q, velocity, targets, contact, dt, soles)
    foot = 'R' if reference.stepping_foot == 'L' else 'L'
    index = 0 if foot == 'L' else 1
    data = mujoco.MjData(rig.model)
    body = rig.model.body('Foot_' + foot).id
    data.qpos[:] = q[0]
    mujoco.mj_kinematics(rig.model, data)
    start = float(data.xpos[body, 0])
    data.qpos[:] = q[-1]
    mujoco.mj_kinematics(rig.model, data)
    measured = float(data.xpos[body, 0]) - start
    gates = single_step_checks(checks, metrics, measured, reference.requested_step_m, index)
    gates['mirror_geometry'] = error < 1e-5
    args.out.mkdir(parents=True)
    (args.out/'mirror_step_reference.py').write_bytes(Path(__file__).read_bytes())
    diagnostic = args.out/'diagnostic_poses.npz'
    np.savez_compressed(diagnostic, qpos=q, qvel=velocity, foot_targets=targets,
        contact=contact, sole_contact=soles, dt=dt)
    metadata.update(stepping_foot=foot, checks=checks, metrics=metrics,
        measured_step_m=measured, diagnostic_sha256=sha256(diagnostic),
        placement_passed=gates['placement'], passed=all(gates.values()),
        admitted_for_training=False, mirror_gates=gates,
        mirror_provenance=dict(source_diagnostic_sha256=reference.sha256,
            source_report_sha256=reference.report_sha256, maximum_body_position_error_m=error,
            generator_sha256=sha256(Path(__file__))),
        changes=str(metadata.get('changes', ''))+'; sagittal reflection, opposite foot, recomputed velocities and physical checks',
        scope='Mirrored single-step diagnostic; dynamic tracking and unseen-hit recovery remain untested. No training admission or Arena export.')
    (args.out/'report.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.out/'ATTRIBUTION.md').write_text(attribution_text(metadata), encoding='utf-8')
    if metadata['passed']:
        CalibrationReference(args.out, rig)
    print(json.dumps(dict(foot=foot, passed=metadata['passed'], gates=gates,
        measured_step_m=measured, mirror_error_m=error)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit', 'source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    generate(parser.parse_args())
