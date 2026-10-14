"""Preserve the recovery audit while recording one proven historical source divergence."""
from pathlib import Path
from unittest.mock import patch

from . import verify_retracted_recovery as original
from .run_support_skill_study import digest

OUT = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
DIAGNOSTIC = Path(__file__).with_name('probe_recovery_exploration.py')
ARCHIVED = original.inherited.PILOT/'sources'/'probe_recovery_exploration.py'
HISTORICAL_SHA256 = 'd3bd741568548f16a059a5f2b236161715c71f74c74563a5c48ea76370dd50d5'
CURRENT_SHA256 = '2d0441a09deeaaabcf5cb9096cc01a2be1c2a3234fa91f359a8b5e46adec1a8e'


def provenance():
    inputs = original.read(OUT/'inputs.json')
    done = original.read(OUT/'completed.json')
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    resolved = {Path(path).resolve(): sha for path, sha in inputs.items()}
    assert digest(Path(__file__)) == resolved[Path(__file__).resolve()]
    assert resolved[DIAGNOSTIC.resolve()] == CURRENT_SHA256
    assert resolved[ARCHIVED.resolve()] == HISTORICAL_SHA256
    assert digest(ARCHIVED) == HISTORICAL_SHA256
    for path, sha in inputs.items():
        path = Path(path)
        if path.suffix == '.py' and path.resolve().parent == Path(__file__).resolve().parent:
            assert digest(OUT/'sources'/path.name) == sha

    historical, saved = {}, original.read(OUT/'historical-inputs.json')
    for root in (original.FULL, original.WEIGHTED, original.inherited.PILOT):
        assert original.read(root/'audit.json')['passed']
        outputs = original.read(root/'completed.json')['outputs']
        for path, sha in outputs.items():
            assert resolved[(root/path).resolve()] == sha, (root, path)
        previous = original.read(root/'inputs.json')
        shared = sorted(set(inputs) & set(previous))
        assert shared
        divergences = []
        for path in shared:
            if inputs[path] == previous[path]:
                continue
            assert root == original.inherited.PILOT, (root, path)
            assert Path(path).resolve() == DIAGNOSTIC.resolve(), path
            assert previous[path] == HISTORICAL_SHA256 and inputs[path] == CURRENT_SHA256
            divergences.append(dict(path=path, historical_sha256=HISTORICAL_SHA256,
                current_sha256=CURRENT_SHA256, archived_source=str(ARCHIVED)))
        assert len(divergences) == (1 if root == original.inherited.PILOT else 0)
        expected = dict(passed=True, count=len(shared), shared_paths=shared, source_divergences=divergences)
        assert saved[str(root)] == expected, root
        historical[str(root)] = dict(shared_inputs=len(shared), sealed_outputs=len(outputs),
                                    source_divergences=divergences)
    assert set(saved) == set(historical)
    for seed in original.SEEDS:
        path = original.PROBES/str(seed)/'dataset.npz'
        manifest = original.read(original.PROBES/'completed.json')['outputs']
        assert resolved[path.resolve()] == manifest[str(path.relative_to(original.PROBES))]
    return inputs, done, historical


def main():
    # Training, optimizer replay, numerical and physical checks remain the sealed implementation.
    with patch.object(original, 'OUT', OUT), patch.object(original, 'provenance', provenance):
        original.main()


if __name__ == '__main__':
    main()
