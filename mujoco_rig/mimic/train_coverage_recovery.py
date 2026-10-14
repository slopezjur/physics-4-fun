"""Fresh48-anchor learning through the unchanged physical experiment recipe."""
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from . import train_retracted_recovery as base
from .coverage_recovery_ppo import learn
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-coverage-learning-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
COVERAGE = Path('logs/mimickit-training/recovery-quiet-coverage-20261009-01')
ONSETS = [40, 44, 48, 54, 66, 74, 82, 90]


def provenance():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    sources = sorted(Path(__file__).parent.glob('*.py'))
    plan = Path(__file__).with_name('RECOVERY_COVERAGE_LEARNING_PLAN.md')
    for path in sources+[plan]:
        shutil.copy2(path, OUT/'sources'/path.name)
    external = Path('D:/Proyectos/Juegos/Tools/MimicKit/mimickit/learning')
    paths = sources+[plan, base.PARENT, base.PARENT.with_suffix('.amp.pt'),
        base.FOUNDATION/'dummy.xml', base.FOUNDATION/'protocol.json',
        base.FOUNDATION/'stationary_reference.npz', base.FOUNDATION/'stationary_reference.json',
        external/'normalizer.py', external/'base_agent.py']
    inputs = {str(path): digest(path) for path in paths}
    for row in base.read(base.FOUNDATION/'protocol.json')['demonstrations']:
        path = Path(row['path'])
        actual = digest(path)
        assert actual == row['sha256'], path
        inputs[str(path)] = actual
    for root in (LOW, base.PILOT, COVERAGE, base.DATA):
        assert base.read(root/'audit.json')['passed'] is True
        for name in ('completed.json', 'audit.json'):
            inputs[str(root/name)] = digest(root/name)
        outputs = base.read(root/'completed.json')['outputs']
        if root in (LOW, base.PILOT):
            selected = list(outputs)
        elif root == COVERAGE:
            selected = ['inputs.json', 'contract.json', 'bank.npz', 'bank-report.json']
            selected += [str(Path(str(seed))/'report.json') for seed in base.SEEDS]
        else:
            selected = ['inputs.json']+[str(Path(str(seed))/'dataset.npz') for seed in base.SEEDS]
        for relative in selected:
            path = root/relative
            actual = digest(path)
            assert actual == outputs[relative], path
            inputs[str(path)] = actual
    historical = {}
    for root in (LOW, COVERAGE):
        previous = base.read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared and all(inputs[path] == previous[path] for path in shared)
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', historical)
    return inputs, historical


def quiet_bank(initial):
    with np.load(COVERAGE/'bank.npz', allow_pickle=False) as archive:
        bank = {key: archive[key] for key in archive.files}
    assert bank['features'].shape == (48, 427) and bank['features'].dtype == np.float32
    assert np.array_equal(bank['onset'], np.repeat(ONSETS, 6))
    assert np.array_equal(bank['pose'], np.tile(np.arange(6), 8))
    assert np.array_equal(bank['issue'], bank['onset']+1)
    assert not bank['features'][:, -2:].any() and not bank['target'].any()
    features = torch.from_numpy(bank['features'].copy())
    with torch.no_grad():
        assert np.array_equal(initial(features).numpy(), bank['target'])
    shutil.copy2(COVERAGE/'bank.npz', OUT/'quiet-anchors.npz')
    return features


def coverage_learn(env, initial, folder, seed, sigma, features):
    with np.load(base.DATA/str(seed)/'dataset.npz', allow_pickle=False) as archive:
        probes = {key: archive[key] for key in archive.files}
    return learn(env, initial, folder, seed, sigma, features, probes)


def declared_save(path, value):
    if path == OUT/'checks.json':
        value = dict(value, quiet_anchors=dict(count=48, width=427,
            identical_source_features=True, zero_action_memory=True,
            fresh_initial_targets_zero=True, sealed_bank_exact=True))
    elif path == OUT/'contract.json':
        projection = dict(value['quiet_projection'], anchor_count=48, anchor_onsets=ONSETS)
        value = dict(value, quiet_projection=projection, comparator=str(LOW),
            additional_comparators=[str(base.PILOT)], actor_learning_rate=1e-4,
            critic_learning_rate=3e-4, coverage_source=str(COVERAGE),
            independent_linear_algebra='thin QR(J.T), then SVD(R.T)',
            fixed_probe_tracking=dict(boundaries=33, probes=36, quiet_probes=12, pushed_probes=24,
                anchor_probes=48, zero_action_memory=True, observation_only=True,
                intermediate_checkpoints=False, reference_label_training=False))
    save(path, value)


def main():
    with patch.object(base, 'OUT', OUT), patch.object(base, 'provenance', provenance), \
         patch.object(base, 'learn', coverage_learn), patch.object(base, 'save', declared_save), \
         patch.object(base.projected, 'quiet_bank', quiet_bank):
        base.main()


if __name__ == '__main__':
    main()
