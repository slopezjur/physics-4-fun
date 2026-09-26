"""Mixed distillation must reserve the fixed ballistic stress cases for evaluation."""
import unittest

from .distill_mixed_com import assert_disjoint, training_cases


class MixedComDistillationTests(unittest.TestCase):
    def test_training_impacts_are_separate_from_evaluation(self):
        cases = training_cases()
        self.assertEqual(len(cases), 8)
        assert_disjoint(cases)

    def test_duplicate_evaluation_case_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint([dict(angle_radians=0., speed_m_s=1.5,
                                  launch_seconds=2., target_body="Chest",
                                  phase_seconds=0.)])


if __name__ == "__main__":
    unittest.main()
