"""Recovery rehearsal must exclude failed, pre-impact and inactive states."""
import unittest

import numpy as np

from .impact_replay import select_observations


class ImpactReplayTests(unittest.TestCase):
    def test_balanced_successful_states_only(self):
        observations = np.arange(40 * 3 * 2, dtype=np.float32).reshape(40, 3, 2)
        active = np.ones((40, 3), dtype=bool)
        active[15:, 1] = False
        cases = [dict(recovered=True, hit=True, first_hit_step=5, recovered_with_step=False),
                 dict(recovered=True, hit=True, first_hit_step=8, recovered_with_step=True),
                 dict(recovered=False, hit=True, first_hit_step=2, recovered_with_step=True)]
        values, indices = select_observations(observations, active, cases, 12, .1)
        self.assertEqual((indices[:, 1] == 1).sum(), 6)
        self.assertEqual((indices[:, 1] == 0).sum(), 6)
        for t, i in indices:
            self.assertGreaterEqual(t, cases[i]["first_hit_step"])
            self.assertLess(t, cases[i]["first_hit_step"] + 30)
            self.assertTrue(active[t, i])
        np.testing.assert_array_equal(values.numpy(), observations[indices[:, 0], indices[:, 1]])

    def test_rejects_no_success(self):
        with self.assertRaisesRegex(ValueError, "no successful"):
            select_observations(np.zeros((3, 1, 2)), np.ones((3, 1), bool),
                [dict(recovered=False, hit=True, first_hit_step=0, recovered_with_step=False)], 2, .1)


if __name__ == "__main__":
    unittest.main()
