"""Compare saved policies on stationary retention; no training or policy export."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .continuous_stand import evaluate_retention, calibrate_safeguard
from .evaluation_cache import actor_fingerprint
from .motion_features import install
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


def run(args):
    activate(args.mimickit)
    from .continuous_stand_task import ContinuousStandTask
    from .guarded_ppo import GuardedPPO
    from learning.distribution_gaussian_diag import DistributionGaussianDiag
    from util import mp_util

    class DiagnosticAgent(GuardedPPO):
        def _build_model(self, config):
            super()._build_model(config)
            if 'motion_feature_indices' in config:
                install(self, config['motion_feature_indices'])

        def _update_model(self):
            raise RuntimeError('This probe cannot optimize policies')

    torch.set_num_threads(1)
    mp_util.init(0, 1, 'cpu', None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    task = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.anchor_bank, allow_pickle=False) as data:
        bank = torch.tensor(data['observations'].copy(), dtype=torch.float32)
    if bank.ndim != 2 or bank.shape[1] != task.observations().shape[1] or not torch.isfinite(bank).all() or not len(bank):
        raise ValueError('Require a finite compatible standing-state bank')
    viewers = Path(__file__).resolve().parents[2] / 'logs/mimickit-viewer'
    viewer_hashes = {str(p): sha256(p) for p in (viewers / 'stand.json', viewers / 'ball.json') if p.exists()}
    report = dict(schema='mimic_stationary_retention_comparison_v1',
                  mimickit_revision=MIMICKIT_REVISION, runner_sha256=sha256(Path(__file__)),
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                                ('continuous_stand.py', 'continuous_stand_task.py', 'stand_task.py')},
                  contract=task.observation_contract(), anchor_bank_sha256=sha256(args.anchor_bank),
                  anchor_states=len(bank), teacher=str(args.checkpoint[0]), results=[],
                  viewer_hashes=viewer_hashes,
                  scope='Explicit diagnostic substitution of a new stationary reference/phase contract '
                        'into unchanged legacy policies. This measures transfer, not in-distribution retention. '
                        'All policies use the same admitted targets. No optimization/export/promotion.')
    teacher = None
    for index, checkpoint in enumerate(args.checkpoint):
        config_path = checkpoint.parent / 'agent.yaml'
        config = yaml.safe_load(config_path.read_text())
        config.setdefault('actor_max_kl', .02)
        agent = DiagnosticAgent(config, task, 'cpu').eval()
        agent.load(str(checkpoint))
        before = actor_fingerprint(agent)
        checkpoint_hash = sha256(checkpoint)
        with torch.no_grad():
            dist = agent._model.eval_actor(agent._obs_norm.normalize(bank))
            if teacher is None:
                teacher = DistributionGaussianDiag(dist.mean.clone(), dist.logstd.clone())
            kl = teacher.kl(dist).clamp_min(0)
        if not torch.isfinite(kl).all():
            raise FloatingPointError('Nonfinite policy deviation')
        retention, traces = evaluate_retention(task, StandPolicy(agent).eval())
        trace_path = args.out / f'case-{index:02d}.npz'
        np.savez_compressed(trace_path, **traces)
        assert actor_fingerprint(agent) == before and sha256(checkpoint) == checkpoint_hash
        row = dict(label=f'{checkpoint.parent.name}/{checkpoint.name}', checkpoint=str(checkpoint),
                   checkpoint_sha256=checkpoint_hash, config_sha256=sha256(config_path),
                   anchor_kl=dict(max=float(kl.max()), mean=float(kl.mean())), retention=retention,
                   trace_sha256=sha256(trace_path), state_unchanged=True)
        report['results'].append(row)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(dict(policy=row['label'], kl=row['anchor_kl'], passed=retention['passed'],
                              successes=retention['successes'], seconds=[c['survival_seconds'] for c in retention['cases']])), flush=True)
    report['calibration'] = calibrate_safeguard(report['results'])
    assert all(sha256(Path(p)) == h for p, h in viewer_hashes.items())
    report.update(viewers_unchanged=True, complete=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--anchor-bank', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True,
                        help='First checkpoint is the teacher; each parent must contain agent.yaml')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
