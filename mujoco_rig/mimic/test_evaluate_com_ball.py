"""Ball stress counts only confirmed hits as recovery opportunities."""
import unittest

from .evaluate_com_ball import cases, summarize


class ComBallEvaluationTests(unittest.TestCase):
    def test_protocol_covers_front_back_and_diagonals_at_both_speeds(self):
        specs = cases()
        self.assertEqual(len(specs), 8)
        self.assertEqual({case["direction"] for case in specs},
                         {"front", "back", "front_diagonal", "back_diagonal"})
        self.assertEqual({case["speed_m_s"] for case in specs}, {1.5, 2.})

    def test_miss_and_delayed_fall_are_not_recovery(self):
        rows = [dict(hit=False, survived=True, recovered=False, survival_seconds=33.),
                dict(hit=True, survived=False, recovered=False, survival_seconds=14.),
                dict(hit=True, survived=True, recovered=True, survival_seconds=33.)]
        self.assertEqual(summarize(rows), dict(cases=3, confirmed_hits=2,
                                              survived_hits=1, recovered_hits=1,
                                              delayed_falls_after_5s=1, missed_shots=1))


if __name__ == "__main__":
    unittest.main()
