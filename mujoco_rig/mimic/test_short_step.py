"""Short-step timing must preserve a hold, one step, and a final hold."""
import unittest

import numpy as np

from .short_step import (DT, PRE_HOLD_SECONDS, POST_HOLD_SECONDS,
                         SOURCE_END_SECONDS, SOURCE_START_SECONDS,
                         phase_seconds, source_times)


class ShortStepTimingTests(unittest.TestCase):
    def test_retiming_preserves_endpoints_and_holds(self):
        motion = 2.
        wall, source = source_times(motion)
        self.assertAlmostEqual(wall[-1], PRE_HOLD_SECONDS + motion + POST_HOLD_SECONDS)
        np.testing.assert_array_equal(source[wall <= PRE_HOLD_SECONDS], SOURCE_START_SECONDS)
        np.testing.assert_array_equal(source[wall >= PRE_HOLD_SECONDS + motion],
                                      SOURCE_END_SECONDS)
        self.assertAlmostEqual(source[round((PRE_HOLD_SECONDS + motion / 2) / DT)],
                               (SOURCE_START_SECONDS + SOURCE_END_SECONDS) / 2)
        phases = phase_seconds(motion)
        self.assertLess(phases["swing_start"], phases["landing"])
        self.assertAlmostEqual(phases["swing_start"], PRE_HOLD_SECONDS + 4 / 15 * motion)

    def test_rejects_unaligned_or_too_short_duration(self):
        for duration in (.45, 1.23, float("nan")):
            with self.assertRaises(ValueError):
                source_times(duration)


if __name__ == "__main__":
    unittest.main()
