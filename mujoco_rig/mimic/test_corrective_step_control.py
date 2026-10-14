"""Check isolated target authority and restoration without physics integration."""
from pathlib import Path
import unittest

import numpy as np
import torch

from .runtime import activate


class CorrectiveStepControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
        from .rig import Rig
        cls.rig = Rig.load(Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01/dummy.xml'), policy_exclude=())

    def test_baseline_formula_and_wider_lower_body_only(self):
        from .corrective_step_variant import control_variant
        from .target_control33 import DelayedTargetControl33
        original = DelayedTargetControl33.target_positions
        rig = self.rig
        q = torch.tensor(rig.rest, dtype=torch.float32)[None].repeat(8, 1)
        v = torch.zeros(8, rig.model.nv)
        action = torch.linspace(-1, 1, 33)[None].repeat(8, 1)
        base = DelayedTargetControl33(rig, 8, 'cpu')
        base.set_reference(q, v)
        expected = base.target_positions(action)
        for radius in (.25, .5):
            with control_variant(radius):
                c = DelayedTargetControl33(rig, 8, 'cpu')
                c.set_reference(q, v)
                target = c.target_positions(action)
                self.assertTrue(torch.all(target >= c.lower) and torch.all(target <= c.upper))
                ranges = np.asarray(c.contract()['corrective_step_target_ranges_rad'])
                names = [rig.model.joint(int(j)).name for j in rig.model.actuator_trnid[rig.actuators, 0]]
                leg = np.asarray([n.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_')) for n in names])
                np.testing.assert_array_equal(ranges[leg], radius)
                np.testing.assert_array_equal(ranges[~leg], .25)
                if radius == .25:
                    self.assertTrue(torch.equal(target, expected))
                c.apply(action)
                self.assertFalse(c.active.any())
                c.apply(-action)
                self.assertFalse(c.active.any())
                c.apply(torch.zeros_like(action))
                self.assertTrue(torch.equal(c.active[:, :33], target))
            self.assertIs(DelayedTargetControl33.target_positions, original)


if __name__ == '__main__':
    unittest.main()
