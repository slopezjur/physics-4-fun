"""Terminal-return and TD-lambda diagnostic accounting."""
import unittest

import numpy as np

from .audit_support_critic import discounted_returns, td_lambda_advantages


class SupportCriticTests(unittest.TestCase):
    def test_terminal_discounted_returns(self):
        rewards = np.array([1., 2., 0.])
        np.testing.assert_allclose(discounted_returns(rewards, .9), [2.8, 2., 0.])

    def test_td_lambda_uses_terminal_zero_and_bootstrapped_intermediate_values(self):
        values = np.array([3., 2., 1.])
        rewards = np.array([1., 2., 0.])
        np.testing.assert_allclose(td_lambda_advantages(values, rewards, .9, 1.),
                                   discounted_returns(rewards, .9) - values, atol=1e-12)
        np.testing.assert_allclose(td_lambda_advantages(values, rewards, .9, 0.),
                                   rewards + .9 * np.array([2., 1., 0.]) - values, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
