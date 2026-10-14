"""Sealed actor-rate comparison: paired physical outcomes and update-scale evidence."""
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from . import analyze_quiet_constrained_recovery as shared
from . import analyze_retracted_recovery as previous
from .run_support_skill_study import digest, save


OUT = Path('logs/mimickit-training/recovery-retracted-actor-rate-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
PROJECTED, WEIGHTED, ZERO = previous.PROJECTED, previous.WEIGHTED, previous.ZERO
CONTROLS = (('lower_rate', LOW), ('projected', PROJECTED), ('weighted', WEIGHTED), ('zero', ZERO))
SEEDS = (921009, 931009)


class SealedReader(shared.SealedReader):
    def __init__(self):
        sources = (Path(__file__), Path(shared.__file__), Path(previous.__file__))
        self.inputs = {str(path): digest(path) for path in sources}
        self.manifests, self.audits = {}, {}
        for root in (OUT, LOW, PROJECTED, WEIGHTED, ZERO):
            completed, audit = root/'completed.json', root/'audit.json'
            self.inputs[str(completed)] = digest(completed)
            self.inputs[str(audit)] = digest(audit)
            self.manifests[root] = json.loads(completed.read_text())['outputs']
            self.audits[root] = json.loads(audit.read_text())
            assert self.audits[root]['passed'] is True, root
        initial = self.read(OUT, 'inputs.json')
        assert all(initial[str(path)] == self.inputs[str(path)] for path in sources)
        self.contract = self.read(OUT, 'contract.json')
        assert self.contract['actor_learning_rate'] == 1e-3
        assert self.contract['control_actor_learning_rate'] == 1e-4
        assert self.contract['critic_learning_rate'] == 3e-4
        assert self.contract['training_seeds'] == list(SEEDS)
        for root in (LOW, PROJECTED, WEIGHTED):
            control = self.read(root, 'contract.json')
            for key in ('training_seeds', 'nominal_transitions_per_seed', 'sigma', 'training_onset_frames'):
                assert self.contract[key] == control[key], (root, key)
        for seed in SEEDS:
            report = self.read(OUT, Path(f'fit-{seed}')/'report.json')
            assert report['actor_learning_rate'] == 1e-3
            assert report['critic_learning_rate'] == 3e-4
        # The prior run recorded no explicit rates: verify its immutable implementation.
        historical = self.read(LOW, 'inputs.json')
        source = LOW/'sources'/'quiet_constrained_ppo.py'
        original = Path(__file__).with_name(source.name)
        assert digest(source) == historical[str(original)]
        self.inputs[str(source)] = digest(source)
        tree = ast.parse(source.read_text(encoding='utf-8'))
        learn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'learn')
        rates = {}
        for node in learn.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if names in (['ao'], ['co']):
                    call = node.value
                    assert isinstance(call.func, ast.Attribute) and call.func.attr == 'Adam'
                    rates[names[0]] = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'lr'))
        assert rates == {'ao': 1e-4, 'co': 3e-4}
        self.rate_evidence = dict(candidate_actor=1e-3, control_actor=1e-4, critic=3e-4,
            candidate_source='contract and both fit reports',
            control_source=str(source), control_source_sha256=self.inputs[str(source)])


def distribution(values):
    sequence = list(values)
    valid = np.asarray([float(value) for value in sequence if value is not None and np.isfinite(value)], dtype=float)
    return dict(count=len(sequence), finite_count=len(valid), missing_or_nonfinite_count=len(sequence)-len(valid),
        minimum=float(valid.min()) if len(valid) else None,
        median=float(np.median(valid)) if len(valid) else None,
        maximum=float(valid.max()) if len(valid) else None)


def quotient(numerator, denominator):
    if numerator is None or denominator is None or not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return None
    return numerator/denominator


def collection_scale(batch, sigma):
    assert float(batch['sigma']) == sigma
    action = torch.from_numpy(batch['action']).reshape(-1, 1)
    old = torch.from_numpy(batch['mean']).reshape(-1, 1)
    post = torch.from_numpy(batch['post_mean']).reshape(-1, 1)
    logp = torch.from_numpy(batch['logp']).flatten()
    advantage = torch.from_numpy(batch['normalized_advantage']).flatten()
    weight = torch.from_numpy(batch['actor_weight']).flatten()
    assert len(advantage) == 768 and old.shape == post.shape == (768, 1)
    scores, ratios = [], []
    for mean in (old, post):
        ratio = (Normal(mean, sigma).log_prob(action).sum(-1)-logp).exp()
        scores.append(float((weight*torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)).mean()))
        ratios.append(ratio.numpy())
    change = (post-old).numpy().astype(float)
    rms = float(np.sqrt(np.mean(change**2)))
    kl = float(np.mean(np.sum(change**2, axis=-1)/(2*sigma**2)))
    return dict(samples=768, pre_surrogate=scores[0], post_surrogate=scores[1],
        surrogate_change=scores[1]-scores[0], mean_change_rms_latent=rms,
        mean_change_rms_over_sigma=rms/sigma, whole_collection_kl=kl,
        post_ratio=distribution(ratios[1]),
        post_ratio_outside_clip=int(((ratios[1] < .8) | (ratios[1] > 1.2)).sum()))


def update_scale(reader, root, seed):
    history = reader.read(root, Path(f'fit-{seed}')/'history.json')
    sigma = float(reader.read(root, 'contract.json')['sigma'])
    records, all_ratios, accepted_ratios, rejected_ratios = [], [], [], []
    ratio_outside, objective_clipped, weighted_clip_mass, weight_mass = 0, 0, 0., 0.
    for collection, row in enumerate(history, 1):
        folder = Path(f'fit-{seed}')
        proposals = reader.read(root, folder/f'projection-{collection}.json')
        objective = reader.read(root, folder/f'objective-{collection}.npz')
        assert len(proposals) == row['actor_updates']+row['rolled_back']
        assert sum(record['accepted'] for record in proposals) == row['actor_updates']
        assert [record['minibatch_index'] for record in proposals] == list(range(len(proposals)))
        assert objective['policy_ratio'].shape == objective['minibatch_indices'].shape == (9, 256)
        assert np.isnan(objective['policy_ratio'][len(proposals):]).all()
        advantage = objective['normalized_advantage'].astype(float)
        weights = objective['actor_weight'].ravel().astype(float)
        for index, record in enumerate(proposals):
            if root == OUT:
                assert record['actor_learning_rate'] == 1e-3
            ratio = objective['policy_ratio'][index].astype(float)
            ids = objective['minibatch_indices'][index]
            assert np.isfinite(ratio).all() and (ratio >= 0).all()
            outside = (ratio < .8) | (ratio > 1.2)
            binding = ratio*advantage[ids] > np.clip(ratio, .8, 1.2)*advantage[ids]
            ratio_outside += int(outside.sum())
            objective_clipped += int(binding.sum())
            weighted_clip_mass += float(weights[ids][binding].sum())
            weight_mass += float(weights[ids].sum())
            all_ratios.extend(ratio)
            (accepted_ratios if record['accepted'] else rejected_ratios).extend(ratio)
            records.append(record)
    accepted = [record for record in records if record['accepted']]
    rejected = [record for record in records if not record['accepted']]
    correction = [record.get('retraction') for record in records]
    accounting = previous.update_accounting(reader, root, seed)
    assert len(records) == accounting['accepted']+accounting['rejected']
    selected = {str(index): collection_scale(reader.read(root,
        Path(f'fit-{seed}')/f'collection-{index}.npz'), sigma) for index in (1, 32)}
    return dict(**accounting, attempts=len(records), sigma=sigma, kl_limit=.01,
        kl_all_attempts=distribution(record['kl'] for record in records),
        kl_accepted=distribution(record['kl'] for record in accepted),
        kl_rejected=distribution(record['kl'] for record in rejected),
        rejection_reasons={reason: sum(reason in record['reasons'] for record in rejected)
                           for reason in ('anchor', 'kl', 'nonfinite')},
        gradient_norm=distribution(record['gradient_norm'] for record in records),
        clipped_gradient_norm=distribution(record['clipped_norm'] for record in records),
        gradient_clipped_attempts=sum(record['gradient_norm'] > .5 for record in records),
        pre_attempt_policy_ratio=distribution(all_ratios),
        pre_accepted_attempt_policy_ratio=distribution(accepted_ratios),
        pre_rejected_attempt_policy_ratio=distribution(rejected_ratios),
        ratio_samples=len(all_ratios), ratio_outside_clip_samples=ratio_outside,
        clipped_surrogate_binding_samples=objective_clipped,
        clipped_surrogate_binding_weight_fraction=weighted_clip_mass/weight_mass if weight_mass else None,
        tentative_adam_norm=distribution(record['tentative_norm'] for record in records),
        projected_actual_norm=distribution(record['actual_norm'] for record in records),
        projected_actual_to_adam_norm=distribution(quotient(record['actual_norm'], record['tentative_norm']) for record in records),
        correction_norm=distribution(item['correction_norm'] if item else None for item in correction),
        correction_to_projected_norm=distribution(item['correction_to_projected_norm'] if item else None for item in correction),
        correction_to_adam_norm=distribution(quotient(item['correction_norm'], record['tentative_norm']) if item else None
                                             for item, record in zip(correction, records, strict=True)),
        corrected_total_step_norm=distribution(item['total_step_norm'] if item else None for item in correction),
        local_anchor_residual=distribution(item['local_residual_max'] if item else None for item in correction),
        attempted_zero_anchor_deviation=distribution(record['anchor_max'] for record in records),
        selected_collections=selected)


def main():
    torch.set_num_threads(1)
    reader = SealedReader()
    reports = {path.parent.name: reader.report(OUT, path.parent.name)
        for path in sorted(OUT.glob('*/report.json')) if path.with_name('trace.npz').exists()}
    assert len(reports) == reader.audits[OUT]['traces'] == 45
    assert sum(len(report['cases']) for report in reports.values()) == reader.audits[OUT]['cases'] == 720
    steps = sum(report['measured_steps'] for report in reports.values())
    assert steps == reader.audits[OUT]['measured_steps']
    pairs, results, response, scale = [], {}, {}, {}
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
            for name, root in (('higher_rate', OUT), *CONTROLS) if root != ZERO}
        response[str(seed)] = {mode: {name: shared.early_response(reader, root, seed, mode)
            for name, root in (('higher_rate', OUT), *CONTROLS)} for mode in ('continuous', 'matched')}
        scale[str(seed)] = {name: update_scale(reader, root, seed)
            for name, root in (('higher_rate', OUT), ('lower_rate', LOW))}
        results[str(seed)] = result
    assert len(pairs) == 2624
    assert all(digest(Path(path)) == sha for path, sha in reader.inputs.items())
    scope = dict(new_physical_rollouts=0, new_training=0, policy_promotion=False,
        primary_contrast='Actor Adam learning rate 0.001 versus sealed 0.0001 retracted control; same fresh seeds, initial actor, budget and other recipe.',
        evaluated_timings='Evaluation onsets 60/102 are reserved. Training onsets are 40-90 excluding 55-65; anchors use 44/74. Seen_pose means poses 0-5 only.',
        quiet_duplicates='Sixteen quiet cases include two copies per pose.',
        early_response='First 30 intervals after the first force interval for 150 N X. All cases retained. Pushed-minus-quiet includes closed-loop state differences.',
        scale_interpretation='Ratios are measured before each attempted optimizer step. Outside-clip ratios and actually binding surrogate clipping are separate. First/last surrogates compare each policy on its own saved collection; later collections differ between rates.',
        causal_limit='A larger nominal rate need not produce proportionally larger accepted movements, more recovery or faster learning.')
    pair_path, response_path, scale_path = (OUT/name for name in
        ('comparison-pairs.json', 'response-diagnostics.json', 'update-scale-diagnostics.json'))
    save(pair_path, dict(scope=scope, cases=pairs))
    save(response_path, dict(scope=scope, inputs=reader.inputs, results=response))
    save(scale_path, dict(scope=scope, inputs=reader.inputs, learning_rates=reader.rate_evidence,
        method='Float32 Normal PPO objectives from saved actions, means, old log probabilities, normalized advantages and unchanged actor weights; no forward policy or optimizer execution.',
        results=scale))
    save(OUT/'comparison-results.json', dict(scope=scope, inputs=reader.inputs,
        learning_rates=reader.rate_evidence,
        outputs={str(path.relative_to(OUT)): digest(path) for path in (pair_path, response_path, scale_path)},
        audited_traces=45, audited_cases=720, audited_steps=steps, comparison_rows=len(pairs), results=results))
    compact = {seed: dict(continuous=results[seed]['continuous']['lower_rate']['all'],
        matched=results[seed]['matched']['lower_rate']['all'],
        core=results[seed]['continuous']['zero']['groups'][shared.GROUPS[0]]['all'],
        quiet=results[seed]['quiet_retention']['continuous']['zero']['whole_stand'],
        accepted_updates=results[seed]['actor_update_accounting']['higher_rate']['accepted']) for seed in results}
    print(json.dumps(dict(audited_steps=steps, comparison_rows=len(pairs), results=compact)), flush=True)


if __name__ == '__main__':
    main()
