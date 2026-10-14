"""Paired outcomes and ankle responses from sealed quiet-constrained evaluations."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save


OUT = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
WEIGHTED = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
ZERO = Path('logs/mimickit-training/learned-ankle-20261009-01')
SEEDS = (921009, 931009)
GROUPS = ('150N_X', '150N_Y', '300N_X', '300N_Y')
OUTCOMES = ('survived', 'settled', 'whole_stand', 'recovered_with_step')


class SealedReader:
    def __init__(self):
        self.inputs = {str(Path(__file__)): digest(Path(__file__))}
        self.manifests, self.audits = {}, {}
        for root in (OUT, WEIGHTED, ZERO):
            completed, audit = root/'completed.json', root/'audit.json'
            self.inputs[str(completed)] = digest(completed)
            self.inputs[str(audit)] = digest(audit)
            self.manifests[root] = json.loads(completed.read_text())['outputs']
            self.audits[root] = json.loads(audit.read_text())
            assert self.audits[root]['passed'] is True, root
        initial = self.read(OUT, 'inputs.json')
        assert initial[str(Path(__file__))] == self.inputs[str(Path(__file__))]

    def read(self, root, relative):
        path = root/relative
        actual = digest(path)
        assert self.manifests[root][str(Path(relative))] == actual, path
        self.inputs[str(path)] = actual
        if path.suffix == '.json':
            return json.loads(path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def report(self, root, folder):
        result = self.read(root, Path(folder)/'report.json')
        cases = result['cases']
        assert [c['case'] for c in cases] == list(range(16))
        for case in cases:
            assert case['pose'] == case['case'] % 8
            assert case['heldout'] == (case['pose'] >= 6)
            assert case['recovered_with_step'] == (case['settled'] and any(
                event['settled_window_start'] is not None for event in case['steps']))
        for key in OUTCOMES:
            assert result[key] == sum(c[key] for c in cases), (folder, key)
        assert result['measured_steps'] == sum(len(c['steps']) for c in cases)
        return result


def sign_name(case, group):
    return ('positive_' if case < 8 else 'negative_') + ('x' if group % 2 == 0 else 'y')


def pairing(current, old, *, seed, mode, comparator, timing, group):
    assert current['quiet'] == old['quiet']
    assert current['sampled'] == old['sampled']
    assert current['seconds'] == old['seconds'] and current['dt'] == old['dt']
    rows = []
    for case, previous in zip(current['cases'], old['cases'], strict=True):
        for key in ('case', 'pose', 'heldout', 'hit_seconds', 'force_n'):
            assert case[key] == previous[key], (seed, mode, comparator, key)
        row = dict(seed=seed, mode=mode, comparator=comparator, timing=timing,
            group=group, group_name=None if group is None else GROUPS[group],
            case=case['case'], pose=case['pose'], reserved_pose=case['heldout'],
            reserved_timing=not current['quiet'],
            direction=None if group is None else sign_name(case['case'], group),
            current_steps=len(case['steps']), previous_steps=len(previous['steps']))
        for key in OUTCOMES:
            now, before = bool(case[key]), bool(previous[key])
            row[key] = dict(current=now, previous=before,
                change='gain' if now and not before else 'loss' if before and not now else 'unchanged')
        rows.append(row)
    return rows


def summarize(rows, metric='settled'):
    return dict(cases=len(rows), current=sum(r[metric]['current'] for r in rows),
        previous=sum(r[metric]['previous'] for r in rows),
        gains=sum(r[metric]['change'] == 'gain' for r in rows),
        losses=sum(r[metric]['change'] == 'loss' for r in rows),
        both_pass=sum(r[metric]['current'] and r[metric]['previous'] for r in rows),
        both_fail=sum(not r[metric]['current'] and not r[metric]['previous'] for r in rows))


def stratify(rows):
    return dict(all=summarize(rows),
        seen_pose=summarize([r for r in rows if not r['reserved_pose']]),
        reserved_pose=summarize([r for r in rows if r['reserved_pose']]),
        by_timing={str(t): summarize([r for r in rows if r['timing'] == t]) for t in (0, 1)},
        by_sign={s: summarize([r for r in rows if r['direction'] == s])
                 for s in sorted({r['direction'] for r in rows})},
        survived=summarize(rows, 'survived'), recovered_with_step=summarize(rows, 'recovered_with_step'))


def zero_quiet(reader, seconds):
    """Match the quiet duration using the audited 60-second zero-correction trace."""
    source = reader.report(ZERO, 'baseline-quiet')
    if seconds == source['seconds']:
        return source
    trace = reader.read(ZERO, Path('baseline-quiet')/'trace.npz')
    count = int(np.ceil(seconds/source['dt']))
    assert seconds <= source['seconds'] and count <= len(trace['time'])
    alive = trace['alive'][:count] & ~trace['failed'][:count]
    cases = []
    for original in source['cases']:
        i = original['case']
        assert not original['steps'], 'Quiet truncation requires empty audited step events'
        survived = bool(alive[:, i].all())
        cases.append(dict(original, survived=survived,
            settled=bool(survived and trace['settled'][count-int(np.ceil(2/source['dt'])):count, i].all()),
            whole_stand=bool(survived and trace['stand'][:count, i].all())))
    return dict(source, cases=cases, seconds=seconds,
                **{key: sum(c[key] for c in cases) for key in OUTCOMES})


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    return dict(samples=values.size, mean=float(values.mean()),
                rms=float(np.sqrt(np.mean(values**2))), maximum_absolute=float(np.abs(values).max()))


def early_response(reader, root, seed, mode):
    """Report issued ankle additions, retaining failures instead of filtering cases."""
    rows, all_addition, all_quiet, all_difference = [], [], [], []
    for timing in (0, 1):
        issue = (60, 102)[timing]+1
        frames = slice(issue, issue+30)
        if root == ZERO:
            push_folder = f'baseline-t{timing}-g0'
            quiet_folder = 'baseline-quiet'
        else:
            prefix = 'matched-' if mode == 'matched' else ''
            push_folder = f'{prefix}{seed}-t{timing}-g0'
            quiet_folder = f'matched-{seed}-t{timing}-quiet' if mode == 'matched' else f'{seed}-quiet'
        push = reader.read(root, Path(push_folder)/'trace.npz')
        quiet = reader.read(root, Path(quiet_folder)/'trace.npz')
        assert np.array_equal(push['time'][frames], quiet['time'][frames])
        assert np.array_equal(push['addition'][:, :, 21], push['addition'][:, :, 29])
        assert np.array_equal(quiet['addition'][:, :, 21], quiet['addition'][:, :, 29])
        addition = push['addition'][frames, :, 21].astype(float)
        quiet_addition = quiet['addition'][frames, :, 21].astype(float)
        difference = addition-quiet_addition
        assert addition.shape == quiet_addition.shape == (30, 16)
        all_addition.append(addition)
        all_quiet.append(quiet_addition)
        all_difference.append(difference)
        for case in range(16):
            rows.append(dict(timing=timing, case=case, pose=case % 8, reserved_pose=case % 8 >= 6,
                direction=sign_name(case, 0), issue_frame=issue, intervals=30,
                push_live_intervals=int(push['alive'][frames, case].sum()),
                quiet_live_intervals=int(quiet['alive'][frames, case].sum()),
                addition_rad=moments(addition[:, case]), quiet_addition_rad=moments(quiet_addition[:, case]),
                pushed_minus_quiet_rad=moments(difference[:, case])))
    return dict(addition_rad=moments(np.concatenate(all_addition)),
        quiet_addition_rad=moments(np.concatenate(all_quiet)),
        pushed_minus_quiet_rad=moments(np.concatenate(all_difference)), cases=rows,
        window='Thirty control intervals starting at the first decision after the first force interval; about 500 ms.',
        interpretation='Pushed-minus-quiet is a closed-loop command difference, not an isolated actor derivative.')


def main():
    reader = SealedReader()
    reports = {p.parent.name: reader.report(OUT, p.parent.name) for p in sorted(OUT.glob('*/report.json'))
               if p.with_name('trace.npz').exists()}
    assert len(reports) == reader.audits[OUT]['traces'] == 45
    assert sum(len(r['cases']) for r in reports.values()) == reader.audits[OUT]['cases'] == 720
    audited_steps = sum(r['measured_steps'] for r in reports.values())
    assert audited_steps == reader.audits[OUT]['measured_steps']
    pairs, results, response = [], {}, {}
    for seed in SEEDS:
        result, responses = {}, {}
        for mode in ('continuous', 'matched', 'sampled'):
            by_comparator = {}
            for comparator, root in (('weighted', WEIGHTED), ('zero', ZERO)):
                if mode == 'sampled' and comparator == 'zero':
                    continue
                selected = []
                for timing in (0, 1):
                    for group in ((0,) if mode == 'sampled' else range(4)):
                        folder = f'{seed}-sampled-t{timing}' if mode == 'sampled' else f'{seed}-t{timing}-g{group}'
                        if mode == 'matched':
                            folder = 'matched-'+folder
                        previous = f'baseline-t{timing}-g{group}' if root == ZERO else folder
                        selected.extend(pairing(reports[folder], reader.report(root, previous),
                            seed=seed, mode=mode, comparator=comparator, timing=timing, group=group))
                pairs.extend(selected)
                by_comparator[comparator] = dict(**stratify(selected),
                    groups={GROUPS[g]: stratify([r for r in selected if r['group'] == g])
                            for g in ((0,) if mode == 'sampled' else range(4))},
                    changed_cases=[r for r in selected if r['settled']['change'] != 'unchanged'])
            result[mode] = by_comparator
        quiet_results = {}
        for mode in ('continuous', 'matched'):
            quiet_results[mode] = {}
            for comparator, root in (('weighted', WEIGHTED), ('zero', ZERO)):
                selected = []
                for timing in ((None,) if mode == 'continuous' else (0, 1)):
                    folder = f'{seed}-quiet' if mode == 'continuous' else f'matched-{seed}-t{timing}-quiet'
                    current = reports[folder]
                    old = reader.report(root, folder) if root == WEIGHTED else zero_quiet(reader, current['seconds'])
                    selected.extend(pairing(current, old, seed=seed, mode=mode+'-quiet',
                        comparator=comparator, timing=timing, group=None))
                pairs.extend(selected)
                quiet_results[mode][comparator] = dict(whole_stand=summarize(selected, 'whole_stand'),
                    settled=summarize(selected), survived=summarize(selected, 'survived'),
                    changed_cases=[r for r in selected if r['whole_stand']['change'] != 'unchanged'])
        result['quiet_retention'] = quiet_results
        result['audited_steps'] = {
            mode: dict(events=sum(r['measured_steps'] for name, r in reports.items() if
                name.startswith(f'matched-{seed}-' if mode == 'matched' else f'{seed}-')),
                recovered_cases=sum(r['recovered_with_step'] for name, r in reports.items() if
                name.startswith(f'matched-{seed}-' if mode == 'matched' else f'{seed}-')))
            for mode in ('continuous_including_sampled_and_quiet', 'matched')}
        for mode in ('continuous', 'matched'):
            responses[mode] = {name: early_response(reader, root, seed, mode)
                for name, root in (('constrained', OUT), ('weighted', WEIGHTED), ('zero', ZERO))}
        results[str(seed)], response[str(seed)] = result, responses
    assert len(pairs) == 1280
    assert all(digest(Path(path)) == sha for path, sha in reader.inputs.items())
    scope = dict(new_physical_rollouts=0, new_training=0, policy_promotion=False,
        evaluated_timings='Evaluation onset frames 60/102 are reserved. Training onsets are 40-90 excluding 55-65; 44/74 are quiet-anchor onsets. Seen_pose means poses 0-5 only.',
        comparator='Case-matched sealed weighted controls and zero correction; sampled comparisons use weighted only.',
        quiet_duplicates='The 16 quiet worlds include two copies per pose; counts are cases, not independent poses.',
        early_response='Core 150 N X pushes only. No filtering on survival in the selected 30-interval window.')
    pair_path, response_path = OUT/'comparison-pairs.json', OUT/'response-diagnostics.json'
    save(pair_path, dict(scope=scope, cases=pairs))
    save(response_path, dict(scope=scope, results=response, inputs=reader.inputs))
    save(OUT/'comparison-results.json', dict(scope=scope, inputs=reader.inputs,
        outputs={str(pair_path.relative_to(OUT)): digest(pair_path),
                 str(response_path.relative_to(OUT)): digest(response_path)},
        audited_traces=45, audited_cases=720, audited_steps=audited_steps,
        comparison_rows=len(pairs), results=results))
    compact = {seed: dict(continuous=results[seed]['continuous']['weighted']['all'],
        matched=results[seed]['matched']['weighted']['all'],
        core_continuous=results[seed]['continuous']['zero']['groups'][GROUPS[0]]['all'],
        quiet=results[seed]['quiet_retention']['continuous']['zero']['whole_stand']) for seed in results}
    print(json.dumps(dict(audited_steps=audited_steps, comparison_rows=len(pairs), results=compact)), flush=True)


if __name__ == '__main__':
    main()
