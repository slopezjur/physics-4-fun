"""Publish a validated network/contract pair, restoring the previous pair on write failure."""
from pathlib import Path


def publish_pair(staged_model: Path, destination: Path):
    staged = (staged_model, staged_model.with_suffix(".contract.json"))
    targets = (destination, destination.with_suffix(".contract.json"))
    originals = [path.read_bytes() if path.exists() else None for path in targets]
    # Read both before changing either: a missing contract cannot leave half an export installed.
    for path in staged:
        if not path.is_file():
            raise FileNotFoundError(path)
    replaced = []
    try:
        for source, target in zip(staged, targets):
            source.replace(target)
            replaced.append(target)
    except OSError:
        for target, original in zip(targets, originals):
            if target in replaced:
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(original)
        raise
