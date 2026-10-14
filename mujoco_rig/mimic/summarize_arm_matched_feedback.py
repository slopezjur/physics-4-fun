"""Summarize controller crossovers, physical pulse response and local input probes."""
import json
from pathlib import Path

import numpy as np

from .probe_arm_matched_feedback import OUT, FOCUS
from .run_support_skill_study import digest, save


def reasons(case):
    result = []
    if not case['survived']:
        result.append('native_failure')
    if not case['stable_hold']:
        result.append('support')
    for key, threshold in (('peak_motor', .05), ('peak_com', .02), ('peak_feet', .02)):
        if case[key] > threshold:
            result.append(key)
    if not case['returned_stand']:
        result.append('return')
    return result


def main():
    audit = json.loads((OUT/'audit.json').read_text())
    assert audit['passed']
    report = json.loads((OUT/'report.json').read_text())
    matrix, pulse_cases, sensitivity = [], [], []
    inputs = [OUT/'audit.json', OUT/'report.json', Path(__file__)]
    lookup = {r['folder']: r for r in report['results']}
    for row in audit['outcomes']:
        r, config = lookup[row['folder']], row['config']
        base = lookup[r['baseline']]
        cases = []
        for i in FOCUS:
            before, after = base['cases'][i], r['cases'][i]
            cases.append(dict(pose=i, before=before['complete'], after=after['complete'],
                before_reasons=reasons(before), after_reasons=reasons(after),
                before_native_survival=before['survived'], after_native_survival=after['survived']))
        if config['pulse'] is None:
            matrix.append(dict(folder=row['folder'], **config, full=row['full'], focus_full=row['focus_full'], cases=cases))
        else:
            folder, baseline_folder = OUT/row['folder'], OUT/r['baseline']
            inputs += [folder/'trace.npz', baseline_folder/'trace.npz']
            z, b = np.load(folder/'trace.npz'), np.load(baseline_folder/'trace.npz')
            for case in cases:
                i = case['pose']
                response = next(x for x in audit['responses'] if x['folder'] == row['folder'] and x['pose'] == i)
                switch = r['switches'][i]
                assert np.array_equal(z['qpos'][switch:switch+2, i], b['qpos'][switch:switch+2, i])
                s12 = next(x for x in response['samples'] if x['frames_after'] == 12)
                s60 = next(x for x in response['samples'] if x['frames_after'] == 60)
                pulse_cases.append(dict(folder=row['folder'], head=config['destination'], pulse=config['pulse'],
                    **case, transmitted=response['transmitted_addition'], issued_delta=response['issued_delta_at_pulse'],
                    q_rms_12=s12['joint_delta_rms'], q_rms_60=s60['joint_delta_rms'],
                    v_rms_12=s12['velocity_delta_rms'], v_rms_60=s60['velocity_delta_rms'],
                    com_12=s12['com_delta'], com_60=s60['com_delta'],
                    ratio_60_to_12=s60['joint_delta_rms']/s12['joint_delta_rms'] if s12['joint_delta_rms'] > 1e-7 else None))
    for feature in ('angle', 'velocity', 'oldest_queue'):
        for head in ('replay', 'correction05'):
            rows = [r for r in audit['sensitivities'] if r['feature'] == feature and r['head'] == head]
            result = dict(feature=feature, head=head, probes=len(rows))
            for key in ('own_raw_derivative', 'own_source_derivative', 'own_issued_derivative', 'max_issued_derivative'):
                values = [r[key] for r in rows]
                result[key] = dict(min=min(values), median=float(np.median(values)), max=max(values))
            sensitivity.append(result)
    pulse_summary = dict(probes=len(pulse_cases), transmitted=sum(abs(c['transmitted']) > 1e-7 for c in pulse_cases),
        improved=sum(not c['before'] and c['after'] for c in pulse_cases),
        regressed=sum(c['before'] and not c['after'] for c in pulse_cases),
        new_native_failures=sum(c['before_native_survival'] and not c['after_native_survival'] for c in pulse_cases),
        q_rms_larger_at_60_than_12=sum(c['q_rms_60'] > c['q_rms_12'] for c in pulse_cases))
    report = dict(matrix=matrix, pulse_cases=pulse_cases, pulse_summary=pulse_summary, sensitivity=sensitivity,
        inputs={str(p): digest(p) for p in inputs},
        interpretation='One-frame command disturbances and input-only central differences are diagnostics, not a local stability certificate or learned deployment.')
    save(OUT/'summary.json', report)
    print('SUMMARY', pulse_summary, flush=True)


if __name__ == '__main__':
    main()
