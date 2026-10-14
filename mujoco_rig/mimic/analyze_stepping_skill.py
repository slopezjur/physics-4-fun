"""Compare archived commanded-step batches and traces without policy or physics execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path('logs/mimickit-training/stepping-skill-20261009-01')
PHASES = ('QUIET', 'WAIT', 'TRANSFER', 'LIFT', 'PLACE', 'LAND', 'SETTLE')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value))) if value.size else None


def first_time(time, mask):
    selected = np.flatnonzero(mask)
    return float(time[selected[0]]) if len(selected) else None


class Evidence:
    def __init__(self, out):
        self.out, self.hashes = out, {}
        self.completed = self.read_json('completed.json', sealed=False)
        self.outputs = {name.replace('\\', '/'): sha for name, sha in self.completed['outputs'].items()}

    def path(self, relative, sealed=True):
        path = self.out / relative
        sha = digest(path)
        if sealed:
            assert relative in self.outputs, f'Input absent from completed manifest: {relative}'
            assert sha == self.outputs[relative], f'Changed experiment output: {relative}'
        self.hashes[relative] = sha
        return path

    def read_json(self, relative, sealed=True):
        return json.loads(self.path(relative, sealed=sealed).read_text(encoding='utf-8'))

    def arrays(self, relative):
        with np.load(self.path(relative), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}


def collection_summary(z, contract):
    task = contract['task']
    goal_start = task['goal_start']
    latent_start = task['previous_latent_start']
    addition_start = latent_start - task['previous_addition_width']
    x, radius = z['x'], np.asarray(task['radius_rad'], dtype=np.float64)
    goal = x[..., goal_start:]
    phase = goal[..., :7].argmax(-1)
    previous = x[..., latent_start:goal_start].astype(np.float64)
    result = dict(observation_shape=list(x.shape),
        phase_observations={PHASES[i]: int((phase == i).sum()) for i in np.unique(phase)},
        observed_transfer_completed=int((goal[..., 32] > .5).sum()),
        observed_success_latched=int((goal[..., 31] > .5).sum()),
        failure_transitions=int(z['failed'].sum()), terminal_transitions=int(z['terminal'].sum()),
        mean_reward=float(z['reward'].mean()), positive_reward_transitions=int((z['reward'] > 0).sum()),
        windows={})
    for label, mask in (('authorized', goal[..., task['authority_goal_index']] > .5), ('wait', phase == 1)):
        if not mask.any():
            continue
        action, mean, remembered = z['action'][mask], z['mean'][mask], previous[mask]
        applied_before = x[..., addition_start:latent_start][mask] * radius
        result['windows'][label] = dict(observations=int(mask.sum()),
            sampled_latent_rms=rms(action), conditional_mean_latent_rms=rms(mean),
            sampled_innovation_rms=rms(action - .9 * remembered),
            learned_mean_innovation_rms=rms(mean - .9 * remembered),
            previous_applied_addition_rms_rad=rms(applied_before),
            previous_applied_addition_abs_max_rad=float(np.abs(applied_before).max()),
            maximum_stance_drift_observed_m=float((goal[..., 36][mask] * .02).max()),
            note=('Sampled WAIT latents are discarded by authority gating; they are not motor corrections.'
                  if label == 'wait' else 'Previous applied corrections are observed before this transition.'))
    return result


def evaluation_summary(z, report, contract):
    time, alive = z['time'], z['alive'].astype(bool)
    physical_failure = z['physical_failed'].astype(bool)
    working = alive & ~physical_failure & (z['phase_before'] == 2)
    config = contract['curriculum']
    radius = np.asarray(contract['task']['radius_rad'])
    safe = working & (z['maximum_stance_drift'] <= config['stance_drift_m']) \
        & (z['pelvis_height'] >= config['min_pelvis_height_m']) \
        & (z['pelvis_up'] >= np.cos(np.deg2rad(config['max_pelvis_tilt_degrees'])))
    side = z['side'].astype(int)
    stance_load = np.take_along_axis(z['loads'], (1 - side)[..., None], -1)[..., 0]
    share = stance_load / np.maximum(z['loads'].sum(-1), 1.)
    safe &= stance_load >= config['support_n']
    cases = []
    for case in range(16):
        mask = working[:, case]
        admitted = safe[:, case]
        progress = z['transfer_progress'][:, case]
        progressed = admitted & (progress >= config['transfer_min_progress_m'])
        drift = alive[:, case] & (z['maximum_stance_drift'][:, case] > config['stance_drift_m'])
        posture = alive[:, case] & ((z['pelvis_height'][:, case] < config['min_pelvis_height_m'])
            | (z['pelvis_up'][:, case] < np.cos(np.deg2rad(config['max_pelvis_tilt_degrees']))))
        first_active = np.flatnonzero(alive[:, case] & z['authority'][:, case])
        assert len(first_active)
        early = alive[:, case] & z['authority'][:, case] & (time < time[first_active[0]] + .5)
        addition = z['addition'][mask, case]
        cases.append(dict(case=case, pose=case % 8, side=int(side[0, case]), heldout=case % 8 >= 6,
            survived=report['cases'][case]['survived'], success=report['cases'][case]['success'],
            first_physical_failure_s=first_time(time, alive[:, case] & physical_failure[:, case]),
            first_stance_drift_over_limit_s=first_time(time, drift),
            first_posture_limit_s=first_time(time, posture),
            transfer_gate_intervals=int((mask & z['transfer_gate'][:, case]).sum()),
            transfer_completed=bool(z['transfer_completed'][-1, case]),
            max_directed_com_progress_m=float(progress[mask].max()),
            max_directed_progress_while_supported_and_before_drift_m=float(progress[admitted].max()) if admitted.any() else None,
            closest_com_target_error_m=float(z['transfer_error'][mask, case].min()),
            max_stance_share=float(share[mask, case].max()),
            progress_threshold_intervals_while_supported=int(progressed.sum()),
            closest_target_error_after_progress_threshold_m=float(z['transfer_error'][progressed, case].min()) if progressed.any() else None,
            max_stance_share_after_progress_threshold=float(share[progressed, case].max()) if progressed.any() else None,
            applied_addition_rms_rad=rms(addition), applied_addition_abs_max_rad=float(np.abs(addition).max()),
            first_500ms_addition_rms_rad=rms(z['addition'][early, case]),
            conditional_mean_innovation_rms=rms(z['latent'][mask, case] - .9 * z['previous_latent'][mask, case])))
    first = np.array([np.flatnonzero(alive[:, case] & z['authority'][:, case])[0] for case in range(16)])
    assert np.all(first == first[0]) and first[0] > 0
    frame = int(first[0])
    exact = {key: bool(np.array_equal(z[key][frame - 1, :8], z[key][frame - 1, 8:]))
             for key in ('qpos', 'qvel', 'addition', 'previous_latent')}
    exact.update({key: bool(np.array_equal(z[key][frame, :8], z[key][frame, 8:]))
                  for key in ('history_before', 'previous_latent', 'source_action', 'nominal')})
    assert all(exact.values()), 'First side contrast lacks identical physical/memory/source state'
    assert np.array_equal(z['side'][frame, :8], np.zeros(8))
    assert np.array_equal(z['side'][frame, 8:], np.ones(8))
    for key in ('level', 'enabled', 'distance', 'onset_seconds'):
        assert np.array_equal(z[key][frame, :8], z[key][frame, 8:]), key
    a, b = z['sampled_latent'][frame, :8].astype(float), z['sampled_latent'][frame, 8:].astype(float)
    magnitude = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosines = [float(np.dot(left, right) / norm) if norm > 0 else None for left, right, norm in zip(a, b, magnitude)]
    goals = z['command_before'][frame]
    changed_columns = [np.flatnonzero(goals[i] != goals[i + 8]).tolist() for i in range(8)]
    assert all(10 in columns and 11 in columns for columns in changed_columns)
    deterministic_equal = bool(np.array_equal(z['sampled_latent'][alive & z['authority']], z['latent'][alive & z['authority']]))
    assert deterministic_equal
    early = alive & z['authority'] & (time[:, None] < time[frame] + .5)
    early_addition = z['addition'][early]
    motor_rms = np.sqrt(np.mean(early_addition.astype(float) ** 2, axis=0))
    return dict(successes=report['successes'], survived=report['survived'],
        final_stand=report['final_stand'], recovered_with_step=report['recovered_with_step'],
        warning_intervals=int(z['warning'].sum()), transfer_completed_cases=int(z['transfer_completed'][-1].sum()),
        transfer_gate_intervals=int((working & z['transfer_gate']).sum()),
        first_500ms_addition_rms_rad=rms(early_addition),
        first_500ms_top_motors=[dict(motor=contract['task']['motor_names'][i], rms_rad=float(motor_rms[i]))
                               for i in np.argsort(motor_rms)[-5:][::-1]],
        applied_envelope_fraction_abs_max=float(np.abs(z['addition'][working] / radius).max()),
        first_side_contrast=dict(issue_interval=frame + 1, issue_interval_end_s=float(time[frame]),
            matching_predecision_state=exact, same_raw_commands_except_side=True,
            changed_goal_columns_by_pose=changed_columns, latent_cosines_by_pose=cosines,
            paired_latent_difference_rms=rms(a - b), latent_rms=rms(np.concatenate((a, b))),
            requested_mean_equals_realized_on_authorized_intervals=deterministic_equal,
            note='sampled_latent stores requested deterministic actor means in these evaluations; no exploration noise is added. '
                 'Goal errors legitimately change with selected swing/stance foot. Parallel first actions alone do not establish causal goal neglect.'),
        cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    destinations = (out / 'diagnostics.json', out / 'diagnostics-inputs.json')
    assert not any(path.exists() for path in destinations), 'Existing diagnostics are immutable'
    evidence = Evidence(out)
    contract = evidence.read_json('contract.json')
    assert contract['task']['goal_width'] == 37 and contract['task']['goal_start'] == 491
    inputs = evidence.read_json('inputs.json')
    for name, sha in inputs.items():
        assert digest(Path(name)) == sha, f'Changed sealed experiment source/input: {name}'
    baseline = evidence.read_json('eval-baseline-l0-t0-dpositive/report.json')
    result = dict(schema='commanded_stepping_archived_diagnostics_v1',
        status='Descriptive archived-array analysis; consult independent audit.json for admission.',
        limitations=['First/last batches contain only 1,536 of 49,152 transitions per seed; observed phases do not establish all-training rates.',
            'No model inference, optimizer execution, or physics is performed.',
            'Measured failure patterns do not establish goal impossibility or a unique causal defect.',
            'Structural quiet fallback and commanded task performance are separate from autonomous hit recovery.'],
        baseline=dict(survived=baseline['survived'], successes=baseline['successes']), seeds={})
    for seed in contract['seeds']:
        fit = f'fit-{seed}/level-0'
        report = evidence.read_json(f'{fit}/report.json')
        evaluation = f'eval-{seed}-l0-t0-dpositive'
        result['seeds'][str(seed)] = dict(training_report=report,
            saved_collections={tag: collection_summary(evidence.arrays(f'{fit}/{tag}-collection.npz'), contract)
                               for tag in ('first', 'last')},
            frozen_evaluation=evaluation_summary(evidence.arrays(f'{evaluation}/trace.npz'),
                evidence.read_json(f'{evaluation}/report.json'), contract))
    source = Path(__file__).resolve()
    source_sha = digest(source)
    destinations[0].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    manifest = dict(schema='commanded_stepping_diagnostics_provenance_v1', source=str(source),
        source_sha256=source_sha, inputs=evidence.hashes,
        outputs={'diagnostics.json': digest(destinations[0])}, physical_execution=False, policy_execution=False)
    destinations[1].write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert digest(source) == source_sha
    print(json.dumps({seed: dict(survived=data['frozen_evaluation']['survived'],
        transfer_completed=data['frozen_evaluation']['transfer_completed_cases'],
        first500ms_addition_rms=data['frozen_evaluation']['first_500ms_addition_rms_rad'])
        for seed, data in result['seeds'].items()}))


if __name__ == '__main__':
    main()
