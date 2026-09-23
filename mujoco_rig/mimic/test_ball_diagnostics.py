"""Diagnostics must preserve motor commands and exclude reset frames from results."""
from pathlib import Path
import unittest

import numpy as np
import torch

from .ball_diagnostics import TraceTargetControl, summarize_case
from .rig import Rig
from .target_control import DelayedTargetControl


class BallDiagnosticsTests(unittest.TestCase):
    def test_instrumentation_preserves_delayed_substep_controls(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        original, probe = [cls(rig, 2, "cpu") for cls in (DelayedTargetControl, TraceTargetControl)]
        q = torch.tensor(np.tile(rig.rest, (2, 1)), dtype=torch.float32)
        v = torch.zeros(2, rig.model.nv)
        rng = torch.Generator().manual_seed(17)
        for step in range(8):
            action = torch.randn(2, 30, generator=rng)
            for control in (original, probe):
                control.set_reference(q, v)
                control.apply(action)
            for substep in range(4):
                measured = q + torch.randn(q.shape, generator=rng) * .1
                torch.testing.assert_close(original.physics_control(measured, v),
                                           probe.physics_control(measured, v), rtol=0, atol=0)
            self.assertEqual(len(probe.substeps), 4)
            if step == 3:
                original.reset(torch.tensor([0]))
                probe.reset(torch.tensor([0]))

    def test_post_hit_window_excludes_preimpact_and_posttermination_frames(self):
        traces = {key: np.zeros(shape) for key, shape in dict(
            action=(6, 1, 30), torque_fraction=(6, 4, 1, 30), joint_error=(6, 4, 1, 30),
            foot_position=(6, 1, 2, 3), foot_load=(6, 1, 2), qpos=(6, 1, 46),
            nonfoot_load=(6, 1, 18), root_velocity=(6, 1, 3)).items()}
        traces["qpos"][:, :, 2:4] = [.9, 1.]
        traces["torque_fraction"][[0, 4, 5]] = 1
        traces["torque_fraction"][1:4, :, :, 0] = 1
        traces["action"][[0, 4, 5]] = 1
        case = dict(first_hit_step=2, survival_seconds=.4, survived=True)
        result = summarize_case(case, 0, traces, .1, [str(i) for i in range(30)])
        self.assertAlmostEqual(result["post_hit_seconds"], .3)
        self.assertEqual(result["action_near_limit_fraction"], 0)
        self.assertAlmostEqual(result["torque_near_limit_fraction"], 1 / 30)
        self.assertEqual(result["highest_torque_saturation"][0]["joint"], "0")


if __name__ == "__main__":
    unittest.main()
