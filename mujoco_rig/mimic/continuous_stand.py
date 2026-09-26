"""Physically admitted, time-independent standing targets and retention protocol."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .baseline import sha256
from .rig import Rig
from .weight_shift import PHYSICAL_CHECKS

REFERENCE_SCHEMA = "mimic_stationary_stand_reference_v1"
CONTRACT = "mimic_stationary_stand_target_pd_v1"
EVALUATION_SCHEMA = "mimic_stationary_stand_retention_v1"
SOURCE_SECONDS = (0., 1., 2., 3., 4., 9., 10., 11.)
CONTACT_SEATING_M = .0001
SEMANTICS = dict(target="constant_pose_per_case", target_velocity="zero",
                 phase="constant_zero", time_domain="all_nonnegative_seconds",
                 contact_seating_m=CONTACT_SEATING_M)
GATES = dict(seconds=30., cases=8, max_mean_root_error_m=.05,
             final_stable_seconds=2., min_height_m=.75, max_tilt_degrees=15.,
             max_horizontal_speed_m_s=.2, max_angular_speed_rad_s=1.,
             min_each_foot_load_n=5., max_foot_error_m=.03)


def physical_checks(rig, poses):
    """A constant reference needs one central static force/actuation check per pose."""
    import mujoco
    from .retarget_steps import validate_motion
    data = mujoco.MjData(rig.model)
    feet = [rig.model.body('Foot_' + side).id for side in ('L', 'R')]
    results = []
    for index, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(rig.model, data)
        q = np.tile(pose, (3, 1))
        v = np.zeros((3, rig.model.nv))
        targets = np.tile(data.xpos[feet], (3, 1, 1))
        checks, metrics = validate_motion(rig, q, v, targets, np.ones((3, 2), bool),
                                          .05, np.ones((3, 4), bool))
        results.append(dict(case=index, passed=all(checks[k] for k in PHYSICAL_CHECKS),
                            checks=checks, metrics=metrics))
    return results


def prepare(source, out):
    from .reference import StandReference
    from .weight_shift import validate_reference
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    reference = StandReference(source, rig, 'cpu')
    metadata = json.loads(source.with_suffix('.json').read_text())
    validate_reference(metadata)
    if reference.duration < max(SOURCE_SECONDS):
        raise ValueError('Source does not cover the fixed posture suite')
    indices = np.rint(np.array(SOURCE_SECONDS) / reference.dt).astype(int)
    if not np.allclose(indices * reference.dt, SOURCE_SECONDS, atol=1e-9, rtol=0):
        raise ValueError('Postures must coincide with source frames')
    with np.load(source, allow_pickle=False) as data:
        poses = data['qpos'][indices].copy()
    # Avoid contact/no-contact disagreement at an exactly tangent float32 reset.
    # This explicit 0.1 mm compliant contact depth is rechecked by every physical gate.
    poses[:, 2] -= CONTACT_SEATING_M
    if len(np.unique(poses, axis=0)) != GATES['cases']:
        raise ValueError('Standing suite requires eight distinct postures')
    checks = physical_checks(rig, poses)
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'stationary_reference.npz'
    np.savez_compressed(path, qpos=poses, qvel=np.zeros((len(poses), rig.model.nv)))
    result = dict(schema=REFERENCE_SCHEMA, passed=all(c['passed'] for c in checks),
                  semantics=SEMANTICS, model_sha256=sha256(rig.path), reference_sha256=sha256(path),
                  source_reference_sha256=sha256(source), source=metadata['source'],
                  source_seconds=SOURCE_SECONDS, physical_cases=checks,
                  scope='Eight independent stationary targets. No interpolation between cases. '
                        'Zero velocity/acceleration for unlimited time; static feasibility is not closed-loop stability.')
    path.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    shutil.copy2(source.parent / 'ATTRIBUTION.md', out / 'ATTRIBUTION.md')
    with (out / 'ATTRIBUTION.md').open('a', encoding='utf-8') as file:
        file.write('\nFurther adaptation: independent stationary targets from eight admitted source postures; velocities set to zero, root lowered 0.1 mm for explicit contact seating, and static physical checks repeated.\n')
    return result


class StationaryReference:
    def __init__(self, path, rig, device):
        metadata = json.loads(path.with_suffix('.json').read_text())
        if (metadata.get('schema') != REFERENCE_SCHEMA or metadata.get('passed') is not True
                or metadata.get('semantics') != SEMANTICS
                or metadata.get('model_sha256') != sha256(rig.path)
                or metadata.get('reference_sha256') != sha256(path)
                or metadata.get('source', {}).get('license') != 'CC-BY-4.0'):
            raise ValueError('Stationary reference provenance/semantics mismatch')
        with np.load(path, allow_pickle=False) as data:
            q, v = data['qpos'].copy(), data['qvel'].copy()
        if (q.shape != (GATES['cases'], rig.model.nq) or v.shape != (GATES['cases'], rig.model.nv)
                or not np.isfinite(q).all() or not np.isfinite(v).all() or np.any(v != 0)
                or not np.allclose(np.linalg.norm(q[:, 3:7], axis=-1), 1., atol=1e-8, rtol=0)
                or len(np.unique(q, axis=0)) != GATES['cases']):
            raise ValueError('Invalid stationary poses or nonzero target velocity')
        # Recompute admission instead of trusting manifest booleans.
        self.physical_cases = physical_checks(rig, q)
        if not all(c['passed'] for c in self.physical_cases):
            raise ValueError('Stationary poses fail static physical admission')
        self.path, self.sha256 = path, sha256(path)
        self.qpos = torch.tensor(q, dtype=torch.float32, device=device)
        self.qvel = torch.tensor(v, dtype=torch.float32, device=device)
        self.duration = None  # No finite clip or phase denominator.

    def sample(self, time, case_ids):
        if not torch.isfinite(time).all() or (time < 0).any():
            raise ValueError('Stationary reference requires finite nonnegative time')
        indices = case_ids % len(self.qpos)
        while indices.ndim < time.ndim:
            indices = indices.unsqueeze(-1)
        indices = indices.expand(time.shape)
        return self.qpos[indices].clone(), self.qvel[indices].clone()


def retention_passed(cases):
    if len(cases) != GATES['cases'] or [c['case'] for c in cases] != list(range(GATES['cases'])):
        raise ValueError('Retention requires the complete ordered eight-case suite')
    for case in cases:
        for key in ('survival_seconds', 'mean_root_error_m', 'final_stable_seconds'):
            if not np.isfinite(case[key]) or case[key] < 0:
                raise ValueError('Invalid retention measurements')
        if type(case['survived']) is not bool:
            raise ValueError('Retention requires an explicit survival outcome')
    return all(c['survived'] and c['survival_seconds'] >= GATES['seconds']
               and c['mean_root_error_m'] <= GATES['max_mean_root_error_m']
               and c['final_stable_seconds'] >= GATES['final_stable_seconds'] for c in cases)


def passing_cases(metrics):
    cases = metrics['cases']
    passed = retention_passed(cases)
    if (metrics.get('evaluation_schema') != EVALUATION_SCHEMA or metrics.get('gates') != GATES
            or type(metrics.get('passed')) is not bool or passed != metrics['passed']
            or metrics.get('successes') != sum(c['survived'] for c in cases)
            or not metrics.get('reference_sha256')):
        raise ValueError('Inconsistent stationary retention evidence')
    return {c['case'] for c in cases if c['survived'] and c['survival_seconds'] >= GATES['seconds']
            and c['mean_root_error_m'] <= GATES['max_mean_root_error_m']
            and c['final_stable_seconds'] >= GATES['final_stable_seconds']}


def retention_rank(metrics):
    accepted = passing_cases(metrics)
    return (metrics['passed'], len(accepted), metrics['successes'],
            min(c['survival_seconds'] for c in metrics['cases']),
            sum(c['survival_seconds'] for c in metrics['cases']) / GATES['cases'],
            -sum(c['mean_root_error_m'] for c in metrics['cases']) / GATES['cases'])


def retention_regressions(candidate, incumbent):
    if any(candidate.get(k) != incumbent.get(k) for k in ('reference_sha256', 'evaluation_schema', 'gates')):
        raise ValueError('Cannot compare different stationary retention protocols')
    return [dict(case=index, metric='settled_30_second_retention', previous=True, candidate=False)
            for index in sorted(passing_cases(incumbent) - passing_cases(candidate))]


@torch.no_grad()
def evaluate_retention(task, actor):
    """Score the first continuous episode only, including a final settled window."""
    from envs.base_env import EnvMode, DoneFlags
    if task.get_num_envs() != GATES['cases'] or task.observation_contract()['observation_contract'] != CONTRACT:
        raise ValueError('Retention requires eight worlds with the stationary contract')
    task.episode_seconds = GATES['seconds']
    task.set_mode(EnvMode.TEST)
    task.reset()
    target, _ = task.sample_reference(task.offset)
    target_positions, _ = task.mapper.from_native_qpos(target)
    active = torch.ones(8, dtype=torch.bool, device=task.device)
    survived = torch.zeros_like(active)
    lengths = torch.zeros_like(task.steps)
    stable = torch.zeros_like(lengths)
    root_error = torch.zeros_like(task.offset)
    poses, observations, masks, settled_trace = [], [], [], []
    for _ in range(int(np.ceil(GATES['seconds'] / task.dt))):
        obs = task.observations()
        observations.append(obs.cpu().numpy())
        _, _, done, _ = task.step(actor(obs))
        q = task.engine.native_qpos()
        error = torch.linalg.vector_norm(q[:, :3] - target[:, :3], dim=-1)
        feet = task.engine.get_body_pos(0)[:, task.feet]
        foot_error = torch.linalg.vector_norm(feet - target_positions[:, task.feet], dim=-1).amax(-1)
        loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
        settled = (q[:, 2] >= GATES['min_height_m']) & (
            1 - 2 * (q[:, 4].square() + q[:, 5].square()) >= np.cos(np.deg2rad(GATES['max_tilt_degrees'])))
        settled &= torch.linalg.vector_norm(task.engine.get_root_vel(0)[:, :2], dim=-1) <= GATES['max_horizontal_speed_m_s']
        settled &= torch.linalg.vector_norm(task.engine.get_root_ang_vel(0), dim=-1) <= GATES['max_angular_speed_rad_s']
        settled &= (loads[:, 0] + loads[:, 2] >= GATES['min_each_foot_load_n']) & (loads[:, 1] + loads[:, 3] >= GATES['min_each_foot_load_n'])
        settled &= (foot_error <= GATES['max_foot_error_m']) & (done != DoneFlags.FAIL.value)
        stable = torch.where(active, torch.where(settled, stable + 1, 0), stable)
        lengths += active
        root_error += error * active
        poses.append(q.cpu().numpy())
        masks.append(active.cpu().numpy().copy())
        settled_trace.append(settled.cpu().numpy())
        survived |= active & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        if not active.any():
            break
        task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    cases = [dict(case=i, survived=bool(survived[i]), survival_seconds=float(lengths[i] * task.dt),
                  mean_root_error_m=float(root_error[i] / lengths[i].clamp_min(1)),
                  final_stable_seconds=float(stable[i] * task.dt)) for i in range(8)]
    report = dict(evaluation_schema=EVALUATION_SCHEMA, gates=GATES, reference_sha256=task.reference.sha256,
                  passed=retention_passed(cases), successes=int(survived.sum()), cases=cases)
    return report, dict(qpos_after=np.array(poses), observations=np.array(observations),
                        active=np.array(masks), settled=np.array(settled_trace))


def calibrate_safeguard(rows):
    """Report separability of measured cases; never manufacture a safe KL bound."""
    accepted = [r for r in rows if r['retention']['passed']]
    rejected = [r for r in rows if not r['retention']['passed']]
    teacher_passed = rows[0]['retention']['passed']
    result = dict(schema='mimic_standing_safeguard_calibration_v1',
                  teacher_passed=teacher_passed, accepted=[r['label'] for r in accepted],
                  rejected=[r['label'] for r in rejected], recommended_kl_limit=None,
                  runtime_limit_changed=False)
    if not teacher_passed:
        result['status'] = 'no_passing_teacher'
        result['reason'] = 'The zero-deviation teacher fails retention; no KL radius can certify this behavior.'
    elif not accepted or not rejected:
        result['status'] = 'insufficient_boundary_evidence'
    else:
        high_good = max(r['anchor_kl']['max'] for r in accepted)
        low_bad = min(r['anchor_kl']['max'] for r in rejected)
        result.update(largest_passing_kl=high_good, smallest_failing_kl=low_bad)
        result['status'] = 'sampled_separation_only' if high_good < low_bad else 'not_separable_by_kl'
    result['behavioral_rule'] = 'Require all eight 30-second cases, each root-error bound, and the final two-second settled window. KL remains an optimization diagnostic; these saved policies do not establish a universally safe radius.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.out)
    print(json.dumps(dict(passed=report['passed'], cases=[(c['case'], c['passed']) for c in report['physical_cases']])))
    raise SystemExit(0 if report['passed'] else 1)
