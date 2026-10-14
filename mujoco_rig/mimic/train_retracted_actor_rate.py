"""Run one fresh actor-rate contrast through the sealed physical experiment."""
from pathlib import Path
import shutil
from unittest.mock import patch

from . import train_retracted_recovery as base
from . import train_retracted_recovery_v2 as repaired
from .retracted_actor_rate_ppo import ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, learn
from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-retracted-actor-rate-20261009-01')
RATE = repaired.OUT
ROOTS = (RATE, base.PROJECTED, base.WEIGHTED, base.PILOT)
CONTROL_ACTOR_LEARNING_RATE = 1e-4


def provenance():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [base.PARENT, base.PARENT.with_suffix('.amp.pt'),
        base.FOUNDATION/'dummy.xml', base.FOUNDATION/'protocol.json',
        base.FOUNDATION/'stationary_reference.npz', base.FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_RETRACTED_ACTOR_RATE_PLAN.md'),
        Path(__file__).with_name('RECOVERY_RETRACTED_PROVENANCE_REPAIR.md'), repaired.ARCHIVE,
        base.TARGETS/'audit.json', base.TARGETS/'completed.json', base.DATA/'audit.json', base.DATA/'completed.json']
    assert base.read(base.TARGETS/'audit.json')['passed'] and base.read(base.DATA/'audit.json')['passed']
    assert digest(repaired.DIAGNOSTIC) == repaired.CURRENT_HASH
    assert digest(repaired.ARCHIVE) == repaired.OLD_HASH
    for root in ROOTS:
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
    for root in ROOTS:
        previous = base.read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared
        different = [path for path in shared if inputs[path] != previous[path]]
        divergences = []
        if root == base.PILOT:
            assert different == [str(repaired.DIAGNOSTIC)]
            assert previous[str(repaired.DIAGNOSTIC)] == repaired.OLD_HASH
            assert inputs[str(repaired.DIAGNOSTIC)] == repaired.CURRENT_HASH
            divergences.append(dict(path=str(repaired.DIAGNOSTIC), historical_sha256=repaired.OLD_HASH,
                current_sha256=repaired.CURRENT_HASH, archived_source=str(repaired.ARCHIVE)))
        else:
            assert not different
            assert previous[str(repaired.DIAGNOSTIC)] == repaired.CURRENT_HASH
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared,
                                    source_divergences=divergences)
    save(OUT/'inputs.json', inputs)
    save(OUT/'historical-inputs.json', historical)
    return inputs, historical


def declared_save(path, value):
    if path == OUT/'contract.json':
        assert value['comparator'] == str(base.PROJECTED)
        value = dict(value, comparator=str(RATE),
            additional_comparators=[str(base.PROJECTED), str(base.WEIGHTED), str(base.PILOT)],
            actor_learning_rate=ACTOR_LEARNING_RATE,
            control_actor_learning_rate=CONTROL_ACTOR_LEARNING_RATE,
            critic_learning_rate=CRITIC_LEARNING_RATE)
    save(path, value)


def main():
    with patch.object(base, 'OUT', OUT), patch.object(base, 'provenance', provenance), \
         patch.object(base, 'learn', learn), patch.object(base, 'save', declared_save):
        base.main()


if __name__ == '__main__':
    main()
