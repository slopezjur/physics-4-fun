"""Explicit smaller policy update rate for the sealed matched onset experiment."""
import argparse
from copy import deepcopy
from pathlib import Path
import traceback
from unittest.mock import patch

from .audit_stand_failures import read, save
from . import run_chest_onset_learning_v2 as study


def run(args):
    settings = read(args.protocol)
    rate = settings['actor_learning_rate']
    assert rate in (1e-5, 3e-5, 1e-4)
    original = study.support.protect_used

    def protect(out, paths, protocol, config):
        parent, inputs, sources = original(out, [*paths, Path(__file__)], protocol, config)
        parent = deepcopy(parent)
        assert parent['config']['actor_optimizer']['learning_rate'] == 3e-4
        parent['config']['actor_optimizer']['learning_rate'] = rate
        return parent, inputs, sources

    with patch.object(study.support, 'protect_used', protect):
        study.run(args)


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
