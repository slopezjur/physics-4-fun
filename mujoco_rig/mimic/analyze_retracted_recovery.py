"""Paired physical outcomes for retracted learning and three sealed controls."""
import json
from pathlib import Path

from . import analyze_quiet_constrained_recovery as shared
from .run_support_skill_study import digest, save


OUT = Path('logs/mimickit-training/recovery-retracted-learning-20261009-01')
PROJECTED = Path('logs/mimickit-training/recovery-quiet-constrained-learning-20261009-01')
WEIGHTED = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
ZERO = Path('logs/mimickit-training/learned-ankle-20261009-01')
CONTROLS = (('projected', PROJECTED), ('weighted', WEIGHTED), ('zero', ZERO))
SEEDS = (921009, 931009)


class SealedReader(shared.SealedReader):
    def __init__(self):
        sources = (Path(__file__), Path(shared.__file__))
        self.inputs = {str(path): digest(path) for path in sources}
        self.manifests, self.audits = {}, {}
        for root in (OUT, PROJECTED, WEIGHTED, ZERO):
            completed, audit = root/'completed.json', root/'audit.json'
            self.inputs[str(completed)] = digest(completed)
            self.inputs[str(audit)] = digest(audit)
            self.manifests[root] = json.loads(completed.read_text())['outputs']
            self.audits[root] = json.loads(audit.read_text())
            assert self.audits[root]['passed'] is True, root
        initial = self.read(OUT, 'inputs.json')
        assert all(initial[str(path)] == self.inputs[str(path)] for path in sources)
        assert shared.ZERO == ZERO
        contract = self.read(OUT, 'contract.json')
        assert contract['training_seeds'] == list(SEEDS)
        for root in (PROJECTED, WEIGHTED):
            previous = self.read(root, 'contract.json')
            for key in ('training_seeds', 'nominal_transitions_per_seed', 'sigma', 'training_onset_frames'):
                assert contract[key] == previous[key], (root, key)


def update_accounting(reader, root, seed):
    history = reader.read(root, Path(f'fit-{seed}')/'history.json')
    assert len(history) == 32
    assert all(row['update'] == index+1 and row['transitions'] == (index+1)*768
               for index, row in enumerate(history))
    accepted = sum(row['actor_updates'] for row in history)
    rejected = sum(row['rolled_back'] for row in history)
    assert all(0 <= row['actor_updates'] <= 9 and row['rolled_back'] in (0, 1)
               and row['actor_updates']+row['rolled_back'] <= 9 for row in history)
    active_collections = [row['update'] for row in history if row['actor_updates']]
    return dict(transitions=history[-1]['transitions'], nominal_actor_slots=288,
        accepted=accepted, rejected=rejected, skipped=288-accepted-rejected,
        last_collection_with_accepted_update=active_collections[-1] if active_collections else None,
        zero_acceptance_collections=[row['update'] for row in history if not row['actor_updates']],
        accepted_by_collection=[row['actor_updates'] for row in history])


def outcome_comparison(reader, reports, seed, mode, comparator, root):
    rows = []
    groups = (0,) if mode == 'sampled' else range(4)
    for timing in (0, 1):
        for group in groups:
            folder = f'{seed}-sampled-t{timing}' if mode == 'sampled' else f'{seed}-t{timing}-g{group}'
            if mode == 'matched':
                folder = 'matched-'+folder
            previous = f'baseline-t{timing}-g{group}' if root == ZERO else folder
            rows.extend(shared.pairing(reports[folder], reader.report(root, previous),
                seed=seed, mode=mode, comparator=comparator, timing=timing, group=group))
    result = dict(**shared.stratify(rows),
        groups={shared.GROUPS[group]: shared.stratify([row for row in rows if row['group'] == group])
                for group in groups},
        changed_cases=[row for row in rows if row['settled']['change'] != 'unchanged'])
    return rows, result


def quiet_comparison(reader, reports, seed, mode, comparator, root):
    rows = []
    for timing in ((None,) if mode == 'continuous' else (0, 1)):
        folder = f'{seed}-quiet' if mode == 'continuous' else f'matched-{seed}-t{timing}-quiet'
        current = reports[folder]
        previous = shared.zero_quiet(reader, current['seconds']) if root == ZERO else reader.report(root, folder)
        rows.extend(shared.pairing(current, previous, seed=seed, mode=mode+'-quiet',
            comparator=comparator, timing=timing, group=None))
    return rows, dict(whole_stand=shared.summarize(rows, 'whole_stand'),
        settled=shared.summarize(rows), survived=shared.summarize(rows, 'survived'),
        changed_cases=[row for row in rows if row['whole_stand']['change'] != 'unchanged'])


def footstep_accounting(reports, seed):
    result = {}
    for mode in ('continuous', 'sampled', 'quiet', 'matched_push', 'matched_quiet'):
        if mode == 'continuous':
            names = [f'{seed}-t{timing}-g{group}' for timing in (0, 1) for group in range(4)]
        elif mode == 'sampled':
            names = [f'{seed}-sampled-t{timing}' for timing in (0, 1)]
        elif mode == 'quiet':
            names = [f'{seed}-quiet']
        elif mode == 'matched_push':
            names = [f'matched-{seed}-t{timing}-g{group}' for timing in (0, 1) for group in range(4)]
        else:
            names = [f'matched-{seed}-t{timing}-quiet' for timing in (0, 1)]
        result[mode] = dict(cases=16*len(names),
            events=sum(reports[name]['measured_steps'] for name in names),
            recovered_cases=sum(reports[name]['recovered_with_step'] for name in names),
            event_cases=[dict(folder=name, case=case['case'], pose=case['pose'],
                reserved_pose=case['heldout'], settled=case['settled'],
                recovered_with_step=case['recovered_with_step'], steps=case['steps'])
                for name in names for case in reports[name]['cases'] if case['steps']])
    return result


def main():
    reader = SealedReader()
    reports = {path.parent.name: reader.report(OUT, path.parent.name)
        for path in sorted(OUT.glob('*/report.json')) if path.with_name('trace.npz').exists()}
    assert len(reports) == reader.audits[OUT]['traces'] == 45
    assert sum(len(report['cases']) for report in reports.values()) == reader.audits[OUT]['cases'] == 720
    steps = sum(report['measured_steps'] for report in reports.values())
    assert steps == reader.audits[OUT]['measured_steps']
    pairs, results, response = [], {}, {}
    for seed in SEEDS:
        result = {}
        for mode in ('continuous', 'matched', 'sampled'):
            result[mode] = {}
            for comparator, root in CONTROLS:
                if mode == 'sampled' and root == ZERO:
                    continue
                rows, comparison = outcome_comparison(reader, reports, seed, mode, comparator, root)
                pairs.extend(rows)
                result[mode][comparator] = comparison
        result['quiet_retention'] = {}
        for mode in ('continuous', 'matched'):
            result['quiet_retention'][mode] = {}
            for comparator, root in CONTROLS:
                rows, comparison = quiet_comparison(reader, reports, seed, mode, comparator, root)
                pairs.extend(rows)
                result['quiet_retention'][mode][comparator] = comparison
        result['audited_steps'] = footstep_accounting(reports, seed)
        result['actor_update_accounting'] = {name: update_accounting(reader, root, seed)
            for name, root in (('retracted', OUT), ('projected', PROJECTED), ('weighted', WEIGHTED))}
        response[str(seed)] = {mode: {name: shared.early_response(reader, root, seed, mode)
            for name, root in (('retracted', OUT), *CONTROLS)} for mode in ('continuous', 'matched')}
        results[str(seed)] = result
    assert len(pairs) == 1952
    assert all(digest(Path(path)) == sha for path, sha in reader.inputs.items())
    scope = dict(new_physical_rollouts=0, new_training=0, policy_promotion=False,
        evaluated_timings='Evaluation onset frames 60/102 are reserved. Training onsets are 40-90 excluding 55-65; 44/74 are quiet-anchor onsets. Seen_pose means poses 0-5 only.',
        comparators='Same-seed projected-only and weighted controls, plus zero correction. Sampled comparisons use projected and weighted controls at the same noise scale.',
        quiet_duplicates='Sixteen quiet worlds contain two copies per pose; these are case counts, not independent pose counts.',
        early_response='Core 150 N X pushes only; all cases retained across the selected 30-interval window. Pushed-minus-quiet includes closed-loop state differences.',
        causal_limit='Update admission and parameter movement change together. More accepted updates alone do not establish better learning or recovery.')
    pair_path, response_path = OUT/'comparison-pairs.json', OUT/'response-diagnostics.json'
    save(pair_path, dict(scope=scope, cases=pairs))
    save(response_path, dict(scope=scope, inputs=reader.inputs, results=response))
    save(OUT/'comparison-results.json', dict(scope=scope, inputs=reader.inputs,
        outputs={str(pair_path.relative_to(OUT)): digest(pair_path),
                 str(response_path.relative_to(OUT)): digest(response_path)},
        audited_traces=45, audited_cases=720, audited_steps=steps,
        comparison_rows=len(pairs), results=results))
    compact = {seed: dict(continuous=results[seed]['continuous']['zero']['all'],
        matched=results[seed]['matched']['zero']['all'],
        core_continuous=results[seed]['continuous']['zero']['groups'][shared.GROUPS[0]]['all'],
        quiet=results[seed]['quiet_retention']['continuous']['zero']['whole_stand'],
        sampled=results[seed]['sampled']['projected']['all'],
        accepted_updates=results[seed]['actor_update_accounting']['retracted']['accepted']) for seed in results}
    print(json.dumps(dict(audited_steps=steps, comparison_rows=len(pairs), results=compact)), flush=True)


if __name__ == '__main__':
    main()
