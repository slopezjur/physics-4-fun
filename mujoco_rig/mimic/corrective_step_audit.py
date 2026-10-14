"""Offline corrective-step measurements from saved 33-motor ball trajectories."""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class StepCriteria:
    unload_n: float = 5.
    support_n: float = 30.
    clearance_m: float = .015
    airborne_seconds: float = .05
    placement_m: float = .03
    support_drift_m: float = .02
    landing_seconds: float = .10
    settling_seconds: float = 2.
    settling_deadline_seconds: float = 2.


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)


def measure_steps(time, xy, clearance, loads, alive, valid, *, hit_time,
                  end_time=None, criteria=StepCriteria()):
    """Measure mechanics, not intent; settling and final survival remain separate.

    Arrays cover one case. Clearance is the lowest point of both foot and toe
    collision boxes relative to the ground. Translation is measured relative
    to the opposite foot as well as in world coordinates, rejecting rigid slide.
    """
    time, xy, clearance, loads = map(np.asarray, (time, xy, clearance, loads))
    alive, valid = np.asarray(alive, bool), np.asarray(valid, bool)
    n = len(time)
    if n < 2 or xy.shape != (n, 2, 2) or clearance.shape != (n, 2) or loads.shape != (n, 2):
        raise ValueError('Expected time, foot XY, whole-foot clearance and loads at every frame')
    if alive.shape != (n,) or valid.shape != (n,) or not all(
            np.isfinite(v).all() for v in (time, xy, clearance, loads)):
        raise ValueError('Malformed or nonfinite step evidence')
    dt = float(time[1] - time[0])
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-8):
        raise ValueError('Step audit requires a uniform, increasing control clock')
    if hit_time is None:
        return []
    stop = float(time[-1] + dt) if end_time is None else end_time
    active = alive & (time >= hit_time) & (time < stop)
    landing_frames = int(np.ceil(criteria.landing_seconds / dt))
    settle_frames = int(np.ceil(criteria.settling_seconds / dt))
    events = []
    for foot in (0, 1):
        opposite = 1 - foot
        for start, end in runs(active & (loads[:, foot] < criteria.unload_n)):
            # No credit for a foot already in flight at impact or trace entry.
            if start == 0 or time[start - 1] < hit_time or loads[start - 1, foot] < criteria.support_n:
                continue
            clear = clearance[start:end, foot] >= criteria.clearance_m
            longest = max((b - a for a, b in runs(clear)), default=0)
            if longest * dt + 1e-10 < criteria.airborne_seconds:
                continue
            landed = end + landing_frames
            if landed > n or not active[start:landed].all():
                continue
            if not (loads[start:landed, opposite] >= criteria.support_n).all():
                continue
            if not (loads[end:landed, foot] >= criteria.support_n).all():
                continue
            takeoff = xy[start - 1]
            landing = xy[end:landed].mean(axis=0)
            world_shift = float(np.linalg.norm(landing[foot] - takeoff[foot]))
            relative_shift = float(np.linalg.norm((landing[foot] - landing[opposite]) -
                                                  (takeoff[foot] - takeoff[opposite])))
            support_drift = float(np.linalg.norm(xy[start:landed, opposite] - takeoff[opposite], axis=-1).max())
            if min(world_shift, relative_shift) < criteria.placement_m or support_drift > criteria.support_drift_m:
                continue
            settled = None
            deadline = time[end] + criteria.settling_deadline_seconds
            for frame in range(landed - 1, n - settle_frames + 1):
                if time[frame] > deadline + dt:
                    break
                span = slice(frame, frame + settle_frames)
                if active[span].all() and valid[span].all():
                    settled = float(time[frame])
                    break
            events.append(dict(foot=('left', 'right')[foot], takeoff_seconds=float(time[start]),
                landing_seconds=float(time[end]), airborne_seconds=(end-start)*dt,
                maximum_clearance_m=float(clearance[start:end, foot].max()),
                world_placement_m=world_shift, relative_placement_m=relative_shift,
                support_drift_m=support_drift, settled_window_start=settled))
    return sorted(events, key=lambda e: e['takeoff_seconds'])


def foot_geometry(model, qpos):
    """Kinematics only: no policy evaluation, collision solve, or physics step."""
    gids = [[model.geom(f'g_{body}_{side}').id for body in ('Foot', 'Toe')] for side in ('L', 'R')]
    bids = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for pair in gids for g in pair):
        raise ValueError('Whole-foot clearance requires the authored box geometry')
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
        raise ValueError('Expected one flat ground plane')
    plane = planes[0]
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    if not np.allclose(data.geom_xmat[plane].reshape(3, 3)[:, 2], (0, 0, 1)):
        raise ValueError('Tilted ground is outside this audit contract')
    ground = data.geom_xpos[plane, 2]
    xy = np.empty((*qpos.shape[:2], 2, 2))
    clearance = np.empty((*qpos.shape[:2], 2))
    for frame in range(len(qpos)):
        for case in range(qpos.shape[1]):
            data.qpos[:] = qpos[frame, case]
            mujoco.mj_kinematics(model, data)
            xy[frame, case] = data.xpos[bids, :2]
            clearance[frame, case] = [min(data.geom_xpos[g, 2] - ground -
                np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                for g in pair) for pair in gids]
    return xy, clearance


def audit_trace(model_path, folder):
    from .audit_stand_failures import digest, read
    model = mujoco.MjModel.from_xml_path(str(model_path))
    report = read(folder / 'report.json')
    with np.load(folder / 'trace.npz', allow_pickle=False) as z:
        xy, clearance = foot_geometry(model, z['qpos_after'])
        time = z['time_s']
        valid = z['balance'] & z['posture'] & z['pre_balance'] & z['pre_posture']
        rows = []
        for row in report['cases']:
            case = row['case']
            encounters = row.get('encounters', [row])
            records = []
            for index, encounter in enumerate(encounters):
                following = encounters[index + 1] if index + 1 < len(encounters) else None
                stop = following['launch_frame'] * (time[1] - time[0]) if following else None
                events = measure_steps(time, xy[:, case], clearance[:, case], z['foot_loads_n'][:, case],
                    z['alive_before'][:, case], valid[:, case], hit_time=encounter.get('first_contact_seconds'),
                    end_time=stop)
                recovered = bool(encounter.get('passed', encounter.get('recovery_pass', False)))
                records.append(dict(encounter=index, steps=events, recovered=recovered,
                    recovered_with_step=recovered and any(e['settled_window_start'] is not None for e in events)))
            rows.append(dict(case=case, encounters=records,
                recovered_with_step=bool(row.get('recovery_pass', False)) and
                    any(e['recovered_with_step'] for e in records)))
    return dict(schema='corrective_step_audit_v1', criteria=asdict(StepCriteria()), cases=rows,
        recovered_with_step=sum(r['recovered_with_step'] for r in rows),
        measured_steps=sum(len(e['steps']) for r in rows for e in r['encounters']),
        physics_steps=0, scope='Control-frame mechanics plus existing recovery gates; not a human-naturalness score',
        inputs={str(p): digest(p) for p in (model_path, folder/'trace.npz', folder/'report.json')})
