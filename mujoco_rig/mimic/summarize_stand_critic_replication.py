"""Verify and summarize three paired training seeds without running more physics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .audit_stand_critic import effective_config, read_json, training_summary
from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .runtime import activate
from .train_stand_imitation import tensor_hash

SEEDS = (301010, 301011, 301012)
RATES = ('1em4', '1em3')
ROOT = Path('logs/mimickit-training')


def training_path(seed, rate):
    if seed == 301010:
        return ROOT/('stand-ppo-noise-0p1-20261001-01' if rate=='1em4' else
                     'stand-ppo-critic-lr1em3-20261002-01')
    return ROOT/f'stand-ppo-critic-lr{rate}-seed{seed}-20261002-01'


def run(args):
    activate(args.mimickit)
    from .amp_recovery import validate_sidecar
    origin = read_json(training_path(301010, '1em4')/'protocol.json')
    reference_config = effective_config(training_path(301010, '1em4')/'agent.yaml')
    previous_path = ROOT/'stand-critic-audit-20261002-01/report.json'
    previous = read_json(previous_path)
    protected = {str(previous_path): sha256(previous_path), **origin['inputs']}
    rows, pairs = [], []
    shared_initializations = {}
    for seed in SEEDS:
        configs, hashes = {}, {}
        for rate in RATES:
            path = training_path(seed, rate)
            protocol = read_json(path/'protocol.json')
            initialization = read_json(path/'initialization.json')
            if protocol['seed'] != seed or initialization['seed'] != seed or not initialization['fresh']:
                raise ValueError('Fresh training seed mismatch')
            if protocol['inputs'] != origin['inputs'] or protocol['demonstrations'] != origin['demonstrations']:
                raise ValueError('Controlled training inputs differ across seeds')
            for filename in ('agent.yaml', 'protocol.json', 'report.json', 'initialization.json', 'runner.py',
                             'initial.pt', 'initial.amp.pt', 'model.pt', 'model.amp.pt',
                             'updates.jsonl', 'assessments.json', 'gpu-samples.jsonl'):
                file = path/filename
                protected[str(file)] = sha256(file)
            if sha256(path/'runner.py') != protocol['source_sha256']:
                raise ValueError('Saved training source mismatch')
            for name in ('initial', 'model'):
                full = torch.load(path/f'{name}.amp.pt', map_location='cpu', weights_only=True)
                portable = torch.load(path/f'{name}.pt', map_location='cpu', weights_only=True)
                validate_sidecar(portable, full)
                if not all(torch.isfinite(value).all() for value in full.values()):
                    raise ValueError('Nonfinite saved checkpoint tensor')
                if name == 'initial':
                    hashes[rate] = tensor_hash(full)
            if hashes[rate] != initialization['full_tensor_sha256']:
                raise ValueError('Initial checkpoint hash differs from recorded fresh state')
            configs[rate] = effective_config(path/'agent.yaml')
            training = training_summary(path)
            if training['native_at64'] is None or training['model_sha256'] != sha256(path/'model.pt'):
                raise ValueError('Missing matched-update assessment or incorrect final checkpoint')
            updates = [json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()]
            if any(not all(np.isfinite(value) for value in update.values()) for update in updates):
                raise ValueError('Nonfinite recorded training update')
            # Audit reports are already bound to their training/checkpoint file hashes.
            if seed == 301010:
                sampled = previous['comparison']['sampled_baseline' if rate=='1em4' else 'sampled_candidate']
            else:
                audit_path = args.out/f'audit-seed{seed}-lr{rate}'
                audit = read_json(audit_path/'report.json')
                audit_protocol = read_json(audit_path/'protocol.json')
                if audit['training_seed'] != seed or audit_protocol['run'] != str(path):
                    raise ValueError('Audit belongs to a different training run')
                for flag in ('initial_checkpoint_verified', 'initial_final_sidecars_valid',
                             'paired_gaussian_draws_identical', 'reset_states_identical',
                             'frozen_tensors_unchanged', 'inputs_unchanged', 'source_snapshots_verified'):
                    if not audit[flag]:
                        raise ValueError(f'Audit verification failed: {flag}')
                for file in (audit_path/'report.json', audit_path/'protocol.json'):
                    protected[str(file)] = sha256(file)
                for file, digest in audit_protocol['inputs'].items():
                    if sha256(Path(file)) != digest:
                        raise ValueError('Audited input no longer matches')
                sampled = audit['sampled_summary']
            deterministic_scoring = None if seed == 301010 else audit['deterministic']
            rows.append(dict(training_seed=seed, critic_rate=rate, training=training, sampled=sampled,
                deterministic_scoring_survival_seconds=None if deterministic_scoring is None else deterministic_scoring['mean_survival_seconds']))
        if hashes['1em4'] != hashes['1em3']:
            raise ValueError('Within-seed fresh tensors differ between critic rates')
        if configs['1em4'] != reference_config:
            raise ValueError('Baseline configuration differs across training seeds')
        expected = configs['1em4'].copy()
        expected['critic_optimizer'] = dict(expected['critic_optimizer'], learning_rate=.001)
        if expected != configs['1em3'] or configs['1em4']['critic_optimizer']['learning_rate'] != .0001:
            raise ValueError('Within-seed configuration differs beyond critic learning rate')
        shared_initializations[str(seed)] = hashes['1em4']
        lower, higher = [next(row for row in rows if row['training_seed']==seed and row['critic_rate']==rate)
                         for rate in RATES]
        pairs.append(dict(training_seed=seed,
            final_native_survival_delta_s=higher['training']['final_native']['mean_survival_seconds']-lower['training']['final_native']['mean_survival_seconds'],
            at64_survival_delta_s=higher['training']['native_at64']['mean_survival_seconds']-lower['training']['native_at64']['mean_survival_seconds'],
            sampled_survival_delta_s=higher['sampled']['mean_survival_seconds']-lower['sampled']['mean_survival_seconds'],
            sampled_task_return_delta=higher['sampled']['discounted_components']['task_reward']-lower['sampled']['discounted_components']['task_reward'],
            sampled_absolute_value_error_delta=higher['sampled']['mean_absolute_observed_return_error']-lower['sampled']['mean_absolute_observed_return_error']))
    if len(set(shared_initializations.values())) != len(SEEDS):
        raise ValueError('Additional training seeds did not produce distinct initial networks')
    summary = {}
    for rate in RATES:
        selected = [row for row in rows if row['critic_rate']==rate]
        summary[rate] = dict(training_seeds=len(selected),
            final_native_stand_passes=sum(row['training']['final_native']['standing_passes'] for row in selected),
            final_gpu_stand_passes=sum(row['training']['final_gpu']['standing_passes'] for row in selected),
            deterministic_cases_per_backend=8*len(selected),
            sampled_stand_passes=sum(row['sampled']['standing_passes'] for row in selected),
            sampled_cases=sum(row['sampled']['assessed_cases'] for row in selected),
            mean_final_native_survival_seconds=float(np.mean([row['training']['final_native']['mean_survival_seconds'] for row in selected])),
            mean_at64_survival_seconds=float(np.mean([row['training']['native_at64']['mean_survival_seconds'] for row in selected])),
            mean_sampled_survival_seconds=float(np.mean([row['sampled']['mean_survival_seconds'] for row in selected])),
            mean_sampled_task_return=float(np.mean([row['sampled']['discounted_components']['task_reward'] for row in selected])),
            mean_sampled_absolute_value_error=float(np.mean([row['sampled']['mean_absolute_observed_return_error'] for row in selected])),
            mean_training_updates=float(np.mean([row['training']['updates'] for row in selected])))
    if any(sha256(Path(file)) != digest for file, digest in protected.items()):
        raise RuntimeError('Protected artifact changed during aggregation')
    shutil.copy2(Path(__file__), args.out/'summarizer.py')
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), rows=rows, paired_deltas=pairs,
        summary=summary, shared_initial_tensor_hashes=shared_initializations,
        physical_wins=dict(final=sum(pair['final_native_survival_delta_s']>0 for pair in pairs),
                           at64=sum(pair['at64_survival_delta_s']>0 for pair in pairs),
                           sampled=sum(pair['sampled_survival_delta_s']>0 for pair in pairs)),
        inputs=protected, protected_artifacts_unchanged=True, within_seed_initializations_match=True,
        across_seed_initializations_distinct=True, only_critic_rate_differs=True,
        initial_final_sidecars_valid=True, additional_training_runs=4,
        source_sha256=sha256(Path(__file__)),
        aggregation_physics_trials=0, arena_changed=False,
        scope='Three paired training seeds, one short training budget; inference seeds are repeated assessments, not extra training replications. Own AMP snapshots differ; compare task return, survival and gates.')
    write_json(args.out/'report.json', report)
    write_json(args.out/'summary.json', dict(summary=summary, paired_deltas=pairs, physical_wins=report['physical_wins']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'stand-critic-replication-20261002-01')
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())
