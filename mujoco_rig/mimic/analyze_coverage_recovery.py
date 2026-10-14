"""Audited 48-versus-12 quiet coverage: physical pairs and saved-command diagnostics."""
import ast
import json
from pathlib import Path

import numpy as np
import torch

from . import analyze_quiet_constrained_recovery as shared
from . import analyze_retracted_recovery as previous
from . import analyze_retracted_actor_rate as scales
from . import command_generalization_probes as probes
from . import command_generalization_traces as commands
from .run_support_skill_study import digest, save


OUT = Path('logs/mimickit-training/recovery-coverage-learning-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
ZERO = Path('logs/mimickit-training/learned-ankle-20261009-01')
BANK = Path('logs/mimickit-training/recovery-quiet-coverage-20261009-01')
SEEDS = (921009, 931009)
ONSETS = (40, 44, 48, 54, 66, 74, 82, 90)
CONTROLS = (('original12', LOW), ('zero', ZERO))
OUTPUT_NAMES = ('comparison-results.json', 'comparison-pairs.json', 'response-diagnostics.json',
                'command-generalization.json', 'update-scale-diagnostics.json')


class SealedReader(shared.SealedReader):
    def __init__(self):
        sources = [Path(__file__), *[Path(module.__file__) for module in
                   (shared, previous, scales, probes, commands)]]
        self.inputs = {str(path): digest(path) for path in sources}
        self.manifests, self.audits = {}, {}
        for root in (OUT, LOW, ZERO, BANK):
            for name in ('completed.json', 'audit.json'):
                path = root/name
                self.inputs[str(path)] = digest(path)
            self.manifests[root] = json.loads((root/'completed.json').read_text())['outputs']
            self.audits[root] = json.loads((root/'audit.json').read_text())
            assert self.audits[root]['passed'] is True, root
        initial = self.read(OUT, 'inputs.json')
        assert all(initial[str(path)] == self.inputs[str(path)] for path in sources)
        assert shared.ZERO == previous.ZERO == ZERO and scales.OUT not in (OUT, LOW)
        self.contract = self.read(OUT, 'contract.json')
        old = self.read(LOW, 'contract.json')
        for key in ('training_seeds', 'nominal_transitions_per_seed', 'sigma', 'training_onset_frames',
                    'dt', 'ankles', 'control', 'initial_actor_seed'):
            assert self.contract[key] == old[key], key
        assert self.contract['training_seeds'] == list(SEEDS)
        assert self.contract['actor_learning_rate'] == 1e-4
        assert self.contract['critic_learning_rate'] == 3e-4
        assert self.contract['coverage_source'] == str(BANK)
        assert self.contract['comparator'] == str(LOW)
        assert self.contract['quiet_projection']['anchor_count'] == 48
        assert self.contract['quiet_projection']['anchor_onsets'] == list(ONSETS)
        assert old['quiet_projection']['anchor_count'] == 12
        bank = self.read(BANK, 'bank.npz')
        current = self.read(OUT, 'quiet-anchors.npz')
        baseline = self.read(LOW, 'quiet-anchors.npz')
        assert bank['features'].shape == current['features'].shape == (48, 427)
        assert np.array_equal(current['features'], bank['features'])
        assert np.array_equal(current['target'].reshape(-1), bank['target'].reshape(-1))
        assert np.array_equal(bank['features'][bank['original_indices']], baseline['features'])
        assert np.array_equal(bank['onset'], np.repeat(ONSETS, 6))
        assert np.array_equal(bank['pose'], np.tile(np.arange(6), 8))
        self.rate_evidence = self._rates()
        for seed in SEEDS:
            report = self.read(OUT, Path(f'fit-{seed}')/'report.json')
            assert report['actor_learning_rate'] == 1e-4 and report['critic_learning_rate'] == 3e-4
            assert report['quiet_anchor_count'] == 48 and report['quiet_onsets'] == list(ONSETS)
            assert report['fixed_probes']['boundaries'] == 33

    def _rates(self):
        historical = self.read(LOW, 'inputs.json')
        source = LOW/'sources'/'quiet_constrained_ppo.py'
        original = Path(__file__).with_name(source.name)
        sha = digest(source)
        assert sha == historical[str(original)]
        self.inputs[str(source)] = sha
        tree = ast.parse(source.read_text(encoding='utf-8'))
        learn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'learn')
        rates = {}
        for node in learn.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if names in (['ao'], ['co']):
                    assert isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'Adam'
                    rates[names[0]] = ast.literal_eval(next(k.value for k in node.value.keywords if k.arg == 'lr'))
        assert rates == dict(ao=1e-4, co=3e-4)
        return dict(actor=1e-4, critic=3e-4, candidate_source='contract and both fit reports',
                    control_source=str(source), control_source_sha256=sha)


def group_ids():
    ids = np.arange(16)
    masks = dict(all=np.ones(16, dtype=bool), seen_pose=ids % 8 < 6, reserved_pose=ids % 8 >= 6,
                 positive_x=ids < 8, negative_x=ids >= 8)
    for pose_name in ('seen_pose', 'reserved_pose'):
        for direction in ('positive_x', 'negative_x'):
            masks[pose_name+'_'+direction] = masks[pose_name] & masks[direction]
    return {name: ids[mask] for name, mask in masks.items()}


def command_arrays(push, quiet, timing):
    checks = {name: commands.trace_checks(trace) for name, trace in zip(('push', 'quiet'), (push, quiet))}
    assert checks['push']['dt'] == checks['quiet']['dt']
    issue = (61, 103)[timing]
    window = slice(issue, issue+30)
    assert np.array_equal(push['time'][window], quiet['time'][window])
    arrays = {key: np.stack([trace[key][window] for trace in (push, quiet)])
              for key in ('alive', 'failed', 'latent', 'previous_latent')}
    for name, key in (('previous_addition', 'previous'), ('addition', 'addition')):
        arrays[name] = np.stack([trace[key][window][:, :, commands.ANKLES] for trace in (push, quiet)])
    arrays['memory'] = commands.RHO*arrays['previous_latent'].astype(float)
    arrays['innovation'] = arrays['latent'].astype(float)-arrays['memory']
    for key in ('latent', 'memory', 'innovation', 'addition'):
        arrays['difference_'+key] = arrays[key][0].astype(float)-arrays[key][1].astype(float)
    arrays['paired_alive'] = arrays['alive'][0] & arrays['alive'][1]
    return arrays, checks


def first_arrays(arrays):
    branch_keys = ('alive', 'failed', 'latent', 'previous_latent', 'previous_addition', 'addition', 'memory', 'innovation')
    return {key: value[:, :1] if key in branch_keys else value[:1] for key, value in arrays.items()}


def concatenate_windows(windows):
    branch_keys = ('alive', 'failed', 'latent', 'previous_latent', 'previous_addition', 'addition', 'memory', 'innovation')
    return {key: np.concatenate([window[key] for window in windows], axis=1 if key in branch_keys else 0)
            for key in windows[0]}


def command_generalization(reader, root, seed, mode):
    windows, firsts, cases, checks = [], [], [], {}
    for timing in (0, 1):
        if root == ZERO:
            push_name, quiet_name = f'baseline-t{timing}-g0', 'baseline-quiet'
        else:
            prefix = 'matched-' if mode == 'matched' else ''
            push_name = f'{prefix}{seed}-t{timing}-g0'
            quiet_name = f'matched-{seed}-t{timing}-quiet' if mode == 'matched' else f'{seed}-quiet'
        push = reader.read(root, Path(push_name)/'trace.npz')
        quiet = reader.read(root, Path(quiet_name)/'trace.npz')
        arrays, checks[str(timing)] = command_arrays(push, quiet, timing)
        first = first_arrays(arrays)
        if mode == 'matched':
            assert not first['previous_latent'].any(), 'Matched first decision must have zero action memory'
        windows.append(arrays)
        firsts.append(first)
        for case in range(16):
            cases.append(dict(timing=timing, issue_frame=(61, 103)[timing], case=case,
                pose=case % 8, reserved_pose=case % 8 >= 6, direction=shared.sign_name(case, 0),
                alive_push=arrays['alive'][0, :, case].tolist(),
                alive_quiet=arrays['alive'][1, :, case].tolist(),
                failed_push=arrays['failed'][0, :, case].tolist(),
                failed_quiet=arrays['failed'][1, :, case].tolist(),
                first_decision=commands.selection_summary(first, [case]),
                first500ms=commands.selection_summary(arrays, [case])))
    combined, combined_first = concatenate_windows(windows), concatenate_windows(firsts)
    return dict(cases=cases, checks=checks, intervals_per_case=30,
        first_quiet_memory_zero=bool(not combined_first['previous_latent'][1].any()),
        first_push_memory_zero=bool(not combined_first['previous_latent'][0].any()),
        first_decision={name: commands.selection_summary(combined_first, ids) for name, ids in group_ids().items()},
        first500ms={name: commands.selection_summary(combined, ids) for name, ids in group_ids().items()})


def compact_probe(group):
    return {key: group[key] for key in ('mean', 'paired_mean', 'paired_requested_addition',
                                      'raw_reference', 'paired_reference', 'slew_limited')}


def fixed_probe_series(reader, seed, history):
    folder = Path(f'fit-{seed}')
    records = reader.read(OUT, folder/'fixed-probes.json')
    arrays = reader.read(OUT, folder/'fixed-probes.npz')
    assert [row['update'] for row in records] == list(range(33))
    assert all(value.shape[0] == 33 for value in arrays.values())
    assert arrays['mean'].shape == (33, 36) and not arrays['mean'][0].any()
    assert not arrays['memory'].any() and not arrays['previous_latent'].any()
    for key in ('onset', 'branch', 'pose', 'quiet_index', 'reference_sign'):
        assert np.array_equal(arrays[key], np.repeat(arrays[key][:1], 33, axis=0))
    series = []
    for boundary, record in enumerate(records):
        if boundary:
            assert record['actor_sha256'] == history[boundary-1]['actor_sha256']
            assert record['optimizer_sha256'] == history[boundary-1]['optimizer_sha256']
        evidence = {key: value[boundary] for key, value in arrays.items()}
        masks = dict(quiet=evidence['branch'] == 0, push=evidence['branch'] != 0,
                     positive_x=evidence['branch'] == 1, negative_x=evidence['branch'] == 2)
        groups = {name: probes.group_summary(evidence, mask) for name, mask in masks.items()}
        assert all(group == record['summary']['groups'][name] for name, group in groups.items())
        series.append(dict(update=boundary, transitions=768*boundary,
            groups={name: compact_probe(group) for name, group in groups.items()},
            by_onset={str(onset): compact_probe(probes.group_summary(evidence, evidence['onset'] == onset))
                      for onset in (44, 74)}))
    return dict(boundaries=33, probes=36, quiet=12, pushed=24, series=series,
                control_series='Historical original12 run did not record per-collection fixed probes.',
                meaning='Same frozen inputs, seen poses/onsets, zero memory. Reference sign is not current optimality.')


def update_diagnostics(reader, root, seed):
    result = scales.update_scale(reader, root, seed)
    history = reader.read(root, Path(f'fit-{seed}')/'history.json')
    series = []
    for row in history:
        records = reader.read(root, Path(f'fit-{seed}')/f"projection-{row['update']}.json")
        series.append(dict(update=row['update'], transitions=row['transitions'],
            accepted=row['actor_updates'], rejected=row['rolled_back'], anchor_max=row['anchor_max'],
            projection_ranks=[record['rank'] for record in records],
            correction_ranks=[record['retraction']['rank'] if record['retraction'] else None for record in records],
            kl=scales.distribution(record['kl'] for record in records),
            projected_to_adam_norm=scales.distribution(scales.quotient(record['actual_norm'], record['tentative_norm'])
                                                       for record in records),
            corrected_to_adam_norm=scales.distribution(scales.quotient(record['retraction']['total_step_norm'],
                record['tentative_norm']) if record['retraction'] else None for record in records)))
    result['collection_history'] = series
    if root == OUT:
        result['fixed_probes'] = fixed_probe_series(reader, seed, history)
    return result


def main():
    assert all(not (OUT/name).exists() for name in OUTPUT_NAMES), 'Preserve existing analysis outputs'
    torch.set_num_threads(1)
    reader = SealedReader()
    reports = {path.parent.name: reader.report(OUT, path.parent.name)
        for path in sorted(OUT.glob('*/report.json')) if path.with_name('trace.npz').exists()}
    assert len(reports) == reader.audits[OUT]['traces'] == 45
    assert sum(len(report['cases']) for report in reports.values()) == reader.audits[OUT]['cases'] == 720
    steps = sum(report['measured_steps'] for report in reports.values())
    assert steps == reader.audits[OUT]['measured_steps']
    pairs, results, response, generalization, scale = [], {}, {}, {}, {}
    for seed in SEEDS:
        result = {}
        for mode in ('continuous', 'matched', 'sampled'):
            result[mode] = {}
            for comparator, root in CONTROLS:
                if mode == 'sampled' and root == ZERO:
                    continue
                rows, summary = previous.outcome_comparison(reader, reports, seed, mode, comparator, root)
                pairs.extend(rows)
                result[mode][comparator] = summary
        result['quiet_retention'] = {}
        for mode in ('continuous', 'matched'):
            result['quiet_retention'][mode] = {}
            for comparator, root in CONTROLS:
                rows, summary = previous.quiet_comparison(reader, reports, seed, mode, comparator, root)
                pairs.extend(rows)
                result['quiet_retention'][mode][comparator] = summary
        result['audited_steps'] = previous.footstep_accounting(reports, seed)
        result['actor_update_accounting'] = {name: previous.update_accounting(reader, root, seed)
            for name, root in (('expanded48', OUT), ('original12', LOW))}
        response[str(seed)] = {mode: {name: shared.early_response(reader, root, seed, mode)
            for name, root in (('expanded48', OUT), *CONTROLS)} for mode in ('continuous', 'matched')}
        generalization[str(seed)] = {mode: {name: command_generalization(reader, root, seed, mode)
            for name, root in (('expanded48', OUT), *CONTROLS)} for mode in ('continuous', 'matched')}
        scale[str(seed)] = {name: update_diagnostics(reader, root, seed)
            for name, root in (('expanded48', OUT), ('original12', LOW))}
        results[str(seed)] = result
    assert len(pairs) == 1280
    assert all(digest(Path(path)) == sha for path, sha in reader.inputs.items())
    scope = dict(new_physical_rollouts=0, new_training=0, new_policy_inference=0, policy_promotion=False,
        contrast='48 versus 12 quiet anchors; same fresh seeds, initial head, actor rate 1e-4, critic rate 3e-4 and transition budget.',
        evaluated_timings='Evaluation onsets 60/102 remain reserved. Training onsets 40-90 exclude 55-65. Seen poses 0-5; reserved poses 6/7.',
        fitted_states='Expanded anchors use training poses 0-5 at onsets 40,44,48,54,66,74,82,90; original anchors 44/74.',
        sampled='Case-paired sampled comparison against original12 only, with the same noise scale.',
        quiet_duplicates='Sixteen quiet worlds contain two copies per pose; counts are cases, not independent poses.',
        command_windows='Core 150 N X only; issue 61/103 plus 30 intervals, retaining every case and explicit live/failure masks.',
        first_decision='Matched first outputs have zero action memory; first quiet bias can precede memory accumulation.',
        causal_limit='Pushed-minus-quiet compares different closed-loop trajectories. Correlated memory/innovation moments are not causal fractions.',
        reference_sign='The 1e-7 latent band applies only to historical reference polarity; agreement is not demonstrated optimality.',
        surrogate='First/last diagnostics use each policy own collection. Later collections differ, so their objective values are not a shared performance benchmark.')
    payloads = {
        'comparison-pairs.json': dict(scope=scope, cases=pairs),
        'response-diagnostics.json': dict(scope=scope, inputs=reader.inputs, results=response),
        'command-generalization.json': dict(scope=scope, inputs=reader.inputs, results=generalization),
        'update-scale-diagnostics.json': dict(scope=scope, inputs=reader.inputs,
            learning_rates=reader.rate_evidence, results=scale),
    }
    for name, payload in payloads.items():
        save(OUT/name, payload)
    save(OUT/'comparison-results.json', dict(scope=scope, inputs=reader.inputs,
        learning_rates=reader.rate_evidence, outputs={name: digest(OUT/name) for name in payloads},
        audited_traces=45, audited_cases=720, audited_steps=steps, comparison_rows=len(pairs), results=results))
    compact = {seed: dict(continuous=result['continuous']['original12']['all'],
        matched=result['matched']['original12']['all'], core=result['continuous']['zero']['groups'][shared.GROUPS[0]]['all'],
        quiet=result['quiet_retention']['continuous']['zero']['whole_stand'],
        accepted_updates=result['actor_update_accounting']['expanded48']['accepted']) for seed, result in results.items()}
    print(json.dumps(dict(audited_steps=steps, comparison_rows=len(pairs), results=compact)), flush=True)


if __name__ == '__main__':
    main()
