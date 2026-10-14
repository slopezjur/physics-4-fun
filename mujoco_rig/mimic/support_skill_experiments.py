"""Compatibility entry point retaining the original experiment source snapshot."""
from contextlib import contextmanager
from unittest.mock import patch

from . import corrective_step_audit
from . import run_support_skill_study as study


@contextmanager
def metric_adapter():
    original = corrective_step_audit.measure_steps

    def measure(time, xy, clearance, loads, alive, valid, onset=None, **kwargs):
        # The sealed mechanical metric deliberately makes the onset keyword-only.
        if onset is not None:
            kwargs['hit_time'] = onset
        return original(time, xy, clearance, loads, alive, valid, **kwargs)

    with patch.object(corrective_step_audit, 'measure_steps', measure):
        yield


if __name__ == '__main__':
    with metric_adapter():
        study.main()
