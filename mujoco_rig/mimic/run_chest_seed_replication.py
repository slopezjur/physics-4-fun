"""Repeat a qualifying recipe with independent optimization RNG and fixed parents."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import traceback
from unittest.mock import patch

from .audit_stand_failures import read, save
from . import run_chest_learning_screen as screen


def run(args):
    settings = read(args.protocol)
    offset = settings['optimization_seed_offset']
    if not isinstance(offset, int) or not 1 <= offset <= 100000:
        raise ValueError('Explicit bounded nonzero optimization-seed offset required')
    mapping = {str(seed): seed + offset for seed in settings['seeds']}
    if settings['actual_training_seeds'] != mapping:
        raise ValueError('Declared optimization seeds do not match offset')
    original_prepare, original_protect, original_save = screen.prepare_pair, screen.protect_used, screen.save

    def prepare(rig, reference, world, source, checkpoint, source_seed, out):
        actual_seed = mapping[str(source_seed)]
        pair = original_prepare(rig, reference, world, source, checkpoint, actual_seed, out)
        save(out / 'seed-mapping.json', dict(source_parent_seed=source_seed,
            optimization_seed=actual_seed, reset_rng_seed=actual_seed + 1701,
            parent_checkpoint=str(checkpoint), matched_arms=True))
        return pair

    def protect(out, paths, protocol, cfg):
        return original_protect(out, [*paths, Path(__file__)], protocol, cfg)

    def record(path, value):
        if path == args.out / 'protocol.json':
            value = dict(value, seed_label_semantics='Result keys and training row seed identify the fixed source parent; optimization RNG uses actual_training_seeds',
                         actual_training_seeds=mapping, independent_stand_parents=False)
        original_save(path, value)

    with ExitStack() as stack:
        stack.enter_context(patch.object(screen, 'prepare_pair', prepare))
        stack.enter_context(patch.object(screen, 'protect_used', protect))
        stack.enter_context(patch.object(screen, 'save', record))
        screen.run(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = p.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
