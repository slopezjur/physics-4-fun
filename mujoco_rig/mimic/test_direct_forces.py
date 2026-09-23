"""Explicit forces retain the physical envelope and exact force derivatives."""
import unittest

import numpy as np

from .direct_forces import force_balance
from .force_retarget import project_actuated_force
from .constrained_fit import constrained_least_squares


class DirectForcesTests(unittest.TestCase):
    def test_projected_seed_has_identical_residual_with_coupled_and_saturated_motors(self):
        columns = np.zeros((9, 2))
        columns[2], columns[4], columns[6] = [1., 1.], [-.1, .1], [.2, .1]
        required = np.array([0., 0., 100., 0., 3., 0., 32., 7., -5.])
        scales, bounds = np.array([10., 5., 5.]), np.array([[-1., 1.], [-1., 1.], [-1., -1.]])
        projected, coefficients = project_actuated_force(columns, required, np.arange(9), scales, 100.,
                                                         bounds, return_forces=True)
        explicit, _, _, _ = force_balance(columns, required, np.arange(9), scales, 100., coefficients, bounds)
        np.testing.assert_allclose(explicit, projected, atol=1e-10)
        self.assertTrue((coefficients >= 0).all())

    def test_exact_force_jacobians_match_independent_central_differences(self):
        rng = np.random.default_rng(33)
        columns, required = rng.normal(size=(9, 8)), rng.normal(size=9)
        scales, bounds, coefficients = np.array([10., 5., 5.]), np.tile([-.3, .4], (3, 1)), rng.uniform(.2, 1., 8)
        def evaluate(x):
            return force_balance(columns, required, np.arange(9), scales, 100., x, bounds)
        _, _, jr, jc = evaluate(coefficients)
        for column in range(8):
            delta = np.eye(8)[column] * 1e-6
            rp, cp, _, _ = evaluate(coefficients + delta)
            rm, cm, _, _ = evaluate(coefficients - delta)
            np.testing.assert_allclose(jr[:, column], (rp - rm) / 2e-6, atol=1e-7)
            np.testing.assert_allclose(jc[:, column], (cp - cm) / 2e-6, atol=1e-7)

    def test_root_support_cannot_hide_motor_overload_or_startup_semantics(self):
        columns = np.zeros((7, 1))
        columns[2] = 1.
        required = np.array([0., 0., 100., 0., 0., 0., 1.5])
        _, margins, _, _ = force_balance(columns, required, np.arange(7), np.ones(1), 100.,
                                         np.ones(1), np.array([[-1., 1.]]))
        self.assertAlmostEqual(margins.min(), -.5)
        residual, startup, _, _ = force_balance(columns, required, np.arange(7), np.ones(1), 100., np.ones(1), None)
        np.testing.assert_allclose(residual, 0., atol=1e-12)
        self.assertGreaterEqual(startup.min(), 0.)

    def test_mixed_numeric_and_analytic_columns_recover_supported_equilibrium(self):
        def evaluate(x):
            return np.array([200 * (x[0] - 1)]), np.array([1e-8 - (x[0] - x[1]), 1e-8 + (x[0] - x[1])])
        def tail(x):
            return np.zeros((1, 1)), np.array([[1.], [-1.]])
        result = constrained_least_squares(evaluate, np.array([.5, .1]), np.zeros(2), np.full(2, 2.),
                                           15, numerical_columns=1, analytic_tail=tail)
        self.assertTrue(result.constraints_satisfied)
        np.testing.assert_allclose(result.x, [1., 1.], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
