"""Check target boundaries against analytically known terminal returns."""
import os
from pathlib import Path
import unittest

import numpy as np

from .probe_standing_credit import analyze
from .runtime import activate


class StandingCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('MIMICKIT_PATH')
        if not path:
            raise unittest.SkipTest('Set MIMICKIT_PATH to test upstream return semantics')
        activate(Path(path))

    def trace(self, ending=1):
        return dict(reward=np.array([[1.], [1.], [0.]], np.float32),
                    value=np.zeros((3, 1), np.float32),
                    next_value=np.array([[10.], [10.], [999.]], np.float32),
                    done=np.array([[0], [0], [ending]], np.int32),
                    active=np.ones((3, 1), bool))

    def test_failure_zeroes_value_but_short_chunks_bootstrap_before_failure(self):
        _, result = analyze(self.trace(), .1, .9, 1., 2)
        np.testing.assert_allclose(result['mc_return'][:, 0], [1.9, 1., 0.])
        np.testing.assert_allclose(result['td_target'][:, 0], [10., 10., 0.])

    def test_full_lambda_one_matches_monte_carlo_and_timeout_bootstraps(self):
        trace = self.trace(3)
        trace['next_value'][-1] = 2.
        _, result = analyze(trace, .1, .9, 1., 3)
        np.testing.assert_allclose(result['mc_return'][:, 0], [3.358, 2.62, 1.8], rtol=1e-6)
        np.testing.assert_allclose(result['td_target'], result['mc_return'])

    def test_lambda_zero_is_one_step_td(self):
        _, result = analyze(self.trace(), .1, .9, 0., 3)
        np.testing.assert_allclose(result['td_target'][:, 0], [10., 10., 0.])

    def test_mask_cannot_reactivate_or_end_without_terminal(self):
        trace = self.trace()
        trace['active'][1] = False
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            analyze(trace, .1, .9, .95, 2)
        trace = self.trace(0)
        with self.assertRaisesRegex(ValueError, 'end in failure or timeout'):
            analyze(trace, .1, .9, .95, 2)


if __name__ == '__main__':
    unittest.main()
