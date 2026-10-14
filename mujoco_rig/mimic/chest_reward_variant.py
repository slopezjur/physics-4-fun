"""Prospective soft reward sensitivity; physics and assessment gates stay fixed."""
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch


@contextmanager
def reward_variant(settings):
    from . import torque_stand_task as module
    scale=float(settings.get('root_velocity_reward_scale',1.))
    if not 1 <= scale <= 25:
        raise ValueError('Root velocity reward scale outside bounded study')
    contract=deepcopy(module.REFERENCE_IMITATION)
    contract['scales']['root_velocity']=scale
    contract['coefficient_source']='Prospectively declared chest-learning protocol; all unmodified coefficients retain the pinned baseline'
    with patch.object(module,'REFERENCE_IMITATION',contract):
        yield
