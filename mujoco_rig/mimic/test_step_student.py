"""Regression tests for saturated Stand inputs in motion learning."""
import tempfile
import unittest
from pathlib import Path

import torch

from .distill_recovery_step import StepStudent, TrajectoryStudent


class StepStudentTests(unittest.TestCase):
    def test_feedback_preserves_nominal_motion_and_dispatches_by_reference(self):
        source = torch.nn.Sequential(torch.nn.Linear(362, 30), torch.nn.Tanh())
        observations = torch.zeros((5, 362))
        observations[:, 217:361] = .5
        observations[:, -1] = torch.arange(5) / 5.
        actions = torch.randn(5, 30).clamp(-1., 1.)
        actor = TrajectoryStudent(source, observations, actions, observations)
        with torch.no_grad():
            actor.feedback[-1].weight.normal_(std=.1)
        torch.testing.assert_close(actor(observations), actions)
        quiet = torch.zeros_like(observations)
        torch.testing.assert_close(actor(quiet), source(quiet))
        disturbed = observations.clone()
        disturbed[:, 0] += .01
        self.assertFalse(torch.equal(actor(disturbed), actions))
        self.assertTrue(all(not p.requires_grad for p in actor.stand.parameters()))

    def test_calibration_preserves_actor_and_exposes_clipped_distinctions(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pt"
            actor = torch.nn.Sequential(torch.nn.Linear(3, 1024), torch.nn.ReLU(),
                                        torch.nn.Linear(1024, 512), torch.nn.ReLU())
            mean = torch.nn.Linear(512, 2)
            state = {"_obs_norm._mean": torch.zeros(3),
                     "_obs_norm._std": torch.full((3,), .0001),
                     "_a_norm._mean": torch.zeros(2),
                     "_a_norm._std": torch.ones(2)}
            state.update({"_model._actor_layers." + k: v for k, v in actor.state_dict().items()})
            state.update({"_model._action_dist._mean_net." + k: v for k, v in mean.state_dict().items()})
            torch.save(state, checkpoint)
            student = StepStudent(checkpoint)
            observations = torch.tensor([[.1, .2, .3], [.2, .3, .4], [0., 0., 0.]])
            before = student(observations).detach()
            torch.testing.assert_close(before[0], before[1], rtol=0, atol=1e-6)
            student.calibrate_features(observations)
            torch.testing.assert_close(student(observations), before, rtol=0, atol=1e-6)
            features = (observations - student.obs_mean) / student.feature_std
            self.assertLessEqual(float(features.abs().max()), 6.000001)
            self.assertFalse(torch.equal(features[0], features[1]))
            student(observations).sum().backward()
            self.assertGreater(float(student.actor[0].weight.grad[:, 3:].abs().sum()), 0.)
            restored = StepStudent(checkpoint)
            restored.load_student_state(student.state_dict())
            torch.testing.assert_close(restored(observations), student(observations))
            with self.assertRaises(ValueError):
                student.calibrate_features(observations)


if __name__ == "__main__":
    unittest.main()
