"""Hard feasibility cannot be purchased by improving the weighted objective."""
import unittest

import numpy as np

from .constrained_fit import constrained_least_squares


class ConstrainedFitTests(unittest.TestCase):
    def test_placement_limit_wins_over_a_conflicting_objective(self):
        def evaluate(x):
            return 200 * (x - [2., 0.]), np.array([1. - x @ x, x[1]])

        result = constrained_least_squares(evaluate, np.array([.1, .2]),
                                           np.full(2, -3.), np.full(2, 3.), 30)
        self.assertTrue(result.constraints_satisfied)
        np.testing.assert_allclose(result.x, [1., 0.], atol=1e-5)
        self.assertGreater(result.cost, 19000., "The lower unconstrained cost must not override placement")

    def test_impossible_support_is_reported_as_infeasible(self):
        def evaluate(x):
            return x.copy(), np.r_[x - 1., -x]

        result = constrained_least_squares(evaluate, np.array([.5]), np.array([0.]), np.array([1.]), 5)
        self.assertFalse(result.constraints_satisfied)
        self.assertGreaterEqual(result.max_constraint_violation, .5)

    def test_derivatives_stay_inside_bounds(self):
        def evaluate(x):
            self.assertTrue(((x >= 0.) & (x <= 1.)).all())
            return 200 * (x - .25), np.array([x[0] - .1])

        result = constrained_least_squares(evaluate, np.array([1.]), np.array([0.]), np.array([1.]), 10)
        self.assertTrue(result.constraints_satisfied)
        np.testing.assert_allclose(result.x, [.25], atol=1e-6)

    def test_restoration_establishes_feasibility_before_polishing(self):
        def evaluate(x):
            return 200 * (x - 1.5), np.r_[x - 1., 2. - x]

        result = constrained_least_squares(evaluate, np.zeros(2), np.full(2, -3.), np.full(2, 3.),
                                           20, restore_feasibility=True)
        self.assertTrue(result.restoration["constraints_satisfied"])
        self.assertTrue(result.constraints_satisfied)
        np.testing.assert_allclose(result.x, [1.5, 1.5], atol=1e-5)

    def test_infeasible_restoration_never_enters_polishing(self):
        def evaluate(x):
            return x.copy(), np.r_[x - 1., -x]

        result = constrained_least_squares(evaluate, np.full(2, .5), np.zeros(2), np.ones(2),
                                           5, restore_feasibility=True)
        self.assertFalse(result.constraints_satisfied)
        self.assertEqual(result.nit, 0)
        self.assertGreaterEqual(result.max_constraint_violation, .5)


if __name__ == "__main__":
    unittest.main()
