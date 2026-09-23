"""Audit fixtures must distinguish support, transmission and sensor-timing defects."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from .foundation_audit import actuator_probe, sensor_probe, structure, settle_passive_joints
from .foundation_sensor_parity import ContactTimingProbe
from .rig import Rig
from .static_support import solve_support
from .target_control import DelayedTargetControl


class FoundationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_mass_inertia_and_rest_pose_are_well_formed(self):
        self.assertTrue(all(structure(self.rig)["checks"].values()))

    def test_every_policy_motor_has_the_intended_generalized_force(self):
        rows = actuator_probe(self.rig)
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["generalized_force_error_nm"] < 1e-9 and
                            row["positive_acceleration_response"] for row in rows))

    def test_sensor_fixture_isolates_real_toe_support(self):
        rows = sensor_probe(self.rig)
        toe = rows[-1]
        self.assertGreater(toe["complete_sole_load_n"], 10.)
        self.assertEqual(toe["actor_observed_load_n"], 0.)
        self.assertTrue(all(row["force_balance_error_n"] < 1e-3 for row in rows))

    def test_empty_support_cannot_be_declared_feasible(self):
        result = solve_support(self.rig, self.rig.rest, support_names=())
        self.assertFalse(result["feasible"])
        self.assertEqual(result["contacts"], 0)

    def test_passive_equilibrium_does_not_move_controlled_joints(self):
        pose, result = settle_passive_joints(self.rig, self.rig.rest)
        joints = self.rig.model.actuator_trnid[self.rig.actuators, 0]
        indices = self.rig.model.jnt_qposadr[joints]
        np.testing.assert_array_equal(pose[indices], self.rig.rest[indices])
        np.testing.assert_array_equal(pose[:7], self.rig.rest[:7])
        self.assertTrue(result["passed"])
        support = solve_support(self.rig, pose, passive_tolerance=.05)
        self.assertTrue(support["feasible"])

    def test_equilibrium_must_use_the_selected_support_foot(self):
        both = solve_support(self.rig, self.rig.rest, passive_tolerance=None)
        left = solve_support(self.rig, self.rig.rest, passive_tolerance=None,
                             support_names=("Foot_L", "Toe_L"))
        self.assertTrue(both["feasible"])
        self.assertFalse(left["feasible"])

    def test_last_solve_snapshot_does_not_mutate_the_physics_state(self):
        engine = ContactTimingProbe(self.rig, 1, "cpu", DelayedTargetControl)
        engine.last_solve_forces = np.zeros((1, self.rig.model.nbody - 1, 3))
        data = engine.datas[0]
        data.qpos[:] = self.rig.rest
        data.qpos[2] -= .001
        mujoco.mj_forward(self.rig.model, data)
        mujoco.mj_step(self.rig.model, data)
        before = data.qpos.copy(), data.qvel.copy(), data.qfrc_constraint.copy()
        engine._after_physics_step(0, data)
        for expected, actual in zip(before, (data.qpos, data.qvel, data.qfrc_constraint)):
            np.testing.assert_array_equal(expected, actual)
        self.assertGreater(np.linalg.norm(engine.last_solve_forces), 0)


if __name__ == "__main__":
    unittest.main()
