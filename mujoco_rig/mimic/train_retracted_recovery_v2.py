"""Preserve the retracted experiment while accounting for one old diagnostic source."""
from pathlib import Path
import shutil
from unittest.mock import patch

from . import train_retracted_recovery as base
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
OLD_HASH = 'd3bd741568548f16a059a5f2b236161715c71f74c74563a5c48ea76370dd50d5'
CURRENT_HASH = '2d0441a09deeaaabcf5cb9096cc01a2be1c2a3234fa91f359a8b5e46adec1a8e'
DIAGNOSTIC = Path(__file__).with_name('probe_recovery_exploration.py')
ARCHIVE = base.PILOT/'sources'/'probe_recovery_exploration.py'


def provenance():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [base.PARENT, base.PARENT.with_suffix('.amp.pt'),
        base.FOUNDATION/'dummy.xml', base.FOUNDATION/'protocol.json',
        base.FOUNDATION/'stationary_reference.npz', base.FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_RETRACTED_LEARNING_PLAN.md'),
        Path(__file__).with_name('RECOVERY_RETRACTED_PROVENANCE_REPAIR.md'), ARCHIVE,
        base.TARGETS/'audit.json', base.TARGETS/'completed.json', base.DATA/'audit.json', base.DATA/'completed.json']
    assert base.read(base.TARGETS/'audit.json')['passed'] and base.read(base.DATA/'audit.json')['passed']
    assert digest(DIAGNOSTIC) == CURRENT_HASH and digest(ARCHIVE) == OLD_HASH
    for root in (base.PROJECTED, base.WEIGHTED, base.PILOT):
        assert base.read(root/'audit.json')['passed']
        paths += [root/'completed.json', root/'audit.json', root/'inputs.json']
        for relative, sha in base.read(root/'completed.json')['outputs'].items():
            path = root/relative
            assert digest(path) == sha, path
            paths.append(path)
    manifest = base.read(base.DATA/'completed.json')['outputs']
    for seed in base.SEEDS:
        path = base.DATA/str(seed)/'dataset.npz'
        assert digest(path) == manifest[str(path.relative_to(base.DATA))]
        paths.append(path)
    inputs = {str(path): digest(path) for path in paths}
    historical = {}
    for root in (base.PROJECTED, base.WEIGHTED, base.PILOT):
        previous = base.read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared
        different = [path for path in shared if inputs[path] != previous[path]]
        divergences = []
        if root == base.PILOT:
            assert different == [str(DIAGNOSTIC)]
            assert previous[str(DIAGNOSTIC)] == OLD_HASH and inputs[str(DIAGNOSTIC)] == CURRENT_HASH
            divergences.append(dict(path=str(DIAGNOSTIC), historical_sha256=OLD_HASH,
                current_sha256=CURRENT_HASH, archived_source=str(ARCHIVE)))
        else:
            assert not different
            assert previous[str(DIAGNOSTIC)] == CURRENT_HASH
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared,
                                    source_divergences=divergences)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', historical)
    return inputs, historical


def main():
    with patch.object(base, 'OUT', OUT), patch.object(base, 'provenance', provenance):
        base.main()


if __name__ == '__main__':
    main()
