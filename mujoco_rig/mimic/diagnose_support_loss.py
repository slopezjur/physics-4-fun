"""Diagnose saved single-support states without advancing or replacing physics."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def segments(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def kinematics(model, data, qpos, qvel):
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def contact_margin(model, data, side, point):
    """Geometric capture-point heuristic, not a support-success criterion."""
    mujoco.mj_collision(model, data)
    bodies = {model.body(f'{part}_{side}').id for part in ('Foot', 'Toe')}
    points = [c.pos[:2].copy() for c in data.contact
              if c.dist <= .001 and 0 in model.geom_bodyid[c.geom]
              and any(body in bodies for body in model.geom_bodyid[c.geom])]
    if len(points) < 3:
        return np.nan
    try:
        hull = ConvexHull(np.asarray(points))
    except QhullError:
        return np.nan
    return -np.max(hull.equations[:, :2] @ point + hull.equations[:, 2])


def analyze(model, reference, folder):
    report = read(folder / 'prepared-feedback/report.json')
    trace = folder / 'prepared-feedback/trace.npz'
    assert digest(trace) == report['trace_sha256']
    data = mujoco.MjData(model)
    bodies = [model.body(name).id for name in ('Spine', 'Chest', 'Head')]
    feet = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    joints = model.actuator_trnid[:, 0]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    names = [model.joint(int(j)).name for j in joints]
    upper = [k for k, name in enumerate(names) if name.startswith(('Spine_', 'Chest_', 'Head_'))]
    jacobian = np.zeros((3, model.nv))
    cases = []
    with np.load(trace) as z:
        times = z['time']
        dt = float(times[0])
        for i, original in enumerate(report['cases']):
            side, pose = original['swing_side'], original['pose']
            alive = z['alive'][:, i]
            kinematics(model, data, reference[pose], np.zeros(model.nv))
            reference_height = data.xpos[bodies, 2] - data.xpos[1, 2]
            goal = z['requested_qpos'][0, i]
            kinematics(model, data, goal, np.zeros(model.nv))
            anchor = data.xpos[feet[1-side], :2].copy()
            goal_com = data.subtree_com[1].copy()
            values = []
            for frame, q in enumerate(z['qpos'][:, i]):
                kinematics(model, data, q, z['qvel'][frame, i])
                tilt = np.rad2deg(np.arccos(np.clip(data.xmat[bodies, 8], -1, 1)))
                deficit = np.maximum(0, reference_height - (data.xpos[bodies, 2] - data.xpos[1, 2]))
                mujoco.mj_jacSubtreeCom(model, data, jacobian, 1)
                com = data.subtree_com[1].copy()
                velocity = jacobian @ data.qvel
                capture = com[:2] + velocity[:2] * np.sqrt(max(com[2], 1e-8) / 9.81)
                margin = contact_margin(model, data, ('L', 'R')[1-side], capture)
                values.append(np.r_[tilt, deficit, com, velocity, margin])
            values = np.asarray(values)
            assert np.max(abs(values[alive, 6:9] - z['com'][alive, i])) < 3e-6
            drift = np.linalg.norm(z['xy'][:, i, 1-side] - anchor, axis=1)
            checks = dict(clearance=z['clearance'][:, i, side] >= .015,
                          swing_unloaded=z['loads'][:, i, side] < 5,
                          stance_loaded=z['loads'][:, i, 1-side] >= 30,
                          stance_position=drift <= .02,
                          recorded_upright_support=z['supported'][:, i])
            valid = alive & np.logical_and.reduce(list(checks.values()))
            spans = segments(valid)
            duration = max((end - start for start, end in spans), default=0) * dt
            assert abs(duration - original['single_support_seconds']) < 1e-8
            recovery = next((start for start, end in spans if start >= 2 and (end-start)*dt >= .05), None)
            checks.update(pelvis_height=z['qpos'][:, i, 2] >= .75,
                          pelvis_upright=1-2*(z['qpos'][:, i, 4]**2 + z['qpos'][:, i, 5]**2) >= np.cos(np.deg2rad(15)),
                          root_horizontal_speed=np.linalg.norm(z['qvel'][:, i, :2], axis=1) <= .2,
                          root_angular_speed=np.linalg.norm(z['qvel'][:, i, 3:6], axis=1) <= 1.)
            for j, name in enumerate(('spine', 'chest', 'head')):
                checks[name + '_tilt'] = values[:, j] <= (20, 20, 30)[j]
                checks[name + '_height'] = values[:, j+3] <= (.05, .06, .10)[j]

            def first(mask, minimum=0.):
                found = np.flatnonzero(mask & alive & (times >= minimum))
                return float(times[found[0]]) if len(found) else None

            crossings = {key: dict(first_failure_seconds=first(~mask),
                                   first_failure_after_015_seconds=first(~mask, .15),
                                   first_failure_after_recovery_seconds=first(~mask, times[recovery]) if recovery is not None else None)
                         for key, mask in checks.items()}
            chest_failures = np.flatnonzero(~checks['chest_tilt'] & alive)
            f = int(chest_failures[0] if len(chest_failures) else np.flatnonzero(alive)[-1])
            q, v = z['qpos'][f, i, qi], z['qvel'][f, i, vi]
            target = z['issued'][f-2, i]
            endpoint_pd = np.clip(4*(target-q)-.08*v, -1, 1)
            motors = [dict(joint=names[k], error_rad=float(q[k]-goal[qi[k]]),
                           speed_rad_s=float(v[k]), endpoint_pd_fraction=float(endpoint_pd[k]),
                           remaining_joint_limit_rad=float(min(q[k]-model.jnt_range[joints[k], 0], model.jnt_range[joints[k], 1]-q[k])))
                      for k in upper] if len(chest_failures) and f >= 2 else []
            key_frames = sorted(set([0, 1, f, *[min(len(times)-1, int(t/dt)) for t in (.15, .3, .4, .6, .8, 1.)]]))
            samples = [dict(time_seconds=float(times[k]), clearance_m=float(z['clearance'][k, i, side]),
                            foot_loads_n=z['loads'][k, i].tolist(), stance_drift_m=float(drift[k]),
                            torso_tilt_degrees=values[k, :3].tolist(),
                            com_error_xy_m=(values[k, 6:8]-goal_com[:2]).tolist(),
                            com_velocity_xy_m_s=values[k, 9:11].tolist(),
                            capture_margin_m=float(values[k, 12]) if np.isfinite(values[k, 12]) else None)
                       for k in key_frames if alive[k]]
            cases.append(dict(pose=pose, swing_side=side, valid_support_seconds=duration,
                              valid_segments=[dict(start=float(times[a]-dt), end=float(times[b-1])) for a, b in spans],
                              threshold_crossings=crossings, chest_crossing_motors=motors,
                              first_negative_capture_margin_seconds=first(values[:, 12] < 0),
                              samples=samples))
    return dict(trace_sha256=digest(trace), cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', default=['single-support-feedback-20261007-05', 'single-support-interval-20261007-01'])
    args = parser.parse_args()
    base = Path('logs/mimickit-training')
    foundation = base / 'stand-ball-checkpoints-20261006-01'
    rig = foundation / 'dummy.xml'
    reference_path = foundation / 'stationary_reference.npz'
    runs = args.runs
    inputs = [rig, reference_path, *[base / run / name for run in runs
              for name in ('protocol.json', 'prepared-feedback/report.json', 'prepared-feedback/trace.npz')]]
    hashes = {str(p): digest(p) for p in inputs}
    args.out.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(rig))
    with np.load(reference_path) as ref:
        reference = ref['qpos'].copy()
        assert not np.any(ref['qvel'])
    result = dict(schema='saved_support_loss_diagnosis_v1', new_physical_cases=0,
                  physics_substeps=0, training_updates=0,
                  measurement_note='Saved forces/support flags; reconstructed kinematics only. Endpoint PD estimates are not recorded applied torques. Capture margin is a geometric heuristic. First crossing does not prove causality.',
                  inputs=hashes, source_sha256=digest(Path(__file__)),
                  runs={run: analyze(model, reference, base / run) for run in runs})
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    (args.out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for run, value in result['runs'].items():
        print(run)
        for case in value['cases']:
            print(case['pose'], case['swing_side'], {key: case['threshold_crossings'][key]
                  for key in ('clearance', 'chest_tilt', 'swing_unloaded', 'stance_position')},
                  'capture_exit', case['first_negative_capture_margin_seconds'])


if __name__ == '__main__':
    main()
