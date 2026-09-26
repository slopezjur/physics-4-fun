"""Step trigger and trajectory remain bounded and directional."""
import unittest

from .capture_step_reflex import (LIFT_M, STEP_SECONDS, capture_excess,
                                  planned_stride, swing_target)


class CaptureStepReflexTests(unittest.TestCase):
    def test_capture_excess_detects_both_sides(self):
        feet = (-.08, .08)
        self.assertEqual(capture_excess(0., 0., 1., feet), 0.)
        self.assertGreater(capture_excess(.25, 0., 1., feet), 0.)
        self.assertLess(capture_excess(-.25, 0., 1., feet), 0.)
        self.assertGreater(capture_excess(0., 1., 1., feet), 0.)

    def test_step_lifts_midflight_and_lands_at_target(self):
        start = (.1, .2)
        first = swing_target(start, -.15, 0.)
        middle = swing_target(start, -.15, STEP_SECONDS / 2)
        final = swing_target(start, -.15, STEP_SECONDS)
        self.assertAlmostEqual(first[0], .1)
        self.assertAlmostEqual(first[1], .2)
        self.assertAlmostEqual(middle[1], .2 + LIFT_M)
        self.assertAlmostEqual(final[0], -.05)
        self.assertAlmostEqual(final[1], .2)

    def test_trigger_commands_meaningful_bounded_placement(self):
        self.assertAlmostEqual(planned_stride(.021), .1)
        self.assertAlmostEqual(planned_stride(-.021), -.1)
        self.assertAlmostEqual(planned_stride(.5), .15)


if __name__ == "__main__":
    unittest.main()
