"""Run the sealed verifier with identical NPZ arrays decompressed once per file."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import verify_coordinated_ankle as original
from .run_support_skill_study import digest, save

NP_LOAD = np.load


class CachedArchive(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def cached_load(*args, **kwargs):
    loaded = NP_LOAD(*args, **kwargs)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        return loaded
    with loaded:
        return CachedArchive({key: loaded[key] for key in loaded.files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    out = parser.parse_args().out
    sample = out/'baseline-quiet'/'trace.npz'
    cached = cached_load(sample, allow_pickle=False)
    with NP_LOAD(sample, allow_pickle=False) as direct:
        assert cached.files == direct.files
        assert all(np.array_equal(cached[key], direct[key]) for key in direct.files)
    with patch.object(original.np, 'load', cached_load):
        original.verify(out)
    sources = [Path(__file__), Path(original.__file__), sample]
    save(out/'audit-harness.json', dict(npz_cache_exact=True,
        scope='Unmodified sealed verifier; only archive decompression is cached in memory.',
        inputs={str(p): digest(p) for p in sources}))


if __name__ == '__main__':
    main()
