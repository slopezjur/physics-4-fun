"""Force planning, torque conversion and isolation checks on the actual character rig."""
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import mujoco
import numpy as np
import torch

from .dynamic_support_teacher import DynamicSupportTeacher, solve_force_qp
from .inverse_contact_control import inverse_contact_map
from .recovery_motor import RecoveryTargetControl
from .rig import Rig


class DynamicSupportTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
        with np.load('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz') as arrays:
            self.pose = arrays['qpos'][0].copy()
        self.live = mujoco.MjData(self.rig.model)
        self.live.qpos[:] = self.pose
        mujoco.mj_forward(self.rig.model, self.live)
        control = RecoveryTargetControl(self.rig, 1, 'cpu')
        control.set_reference(torch.tensor(self.pose[None], dtype=torch.float32), torch.zeros(1, self.rig.model.nv))
        self.task = SimpleNamespace(device='cpu', recovery_motor_radians=1.25, rig=self.rig,
            dt=control.dt, get_num_envs=lambda: 1, observation_contract=lambda: self.rig.contract(),
            engine=SimpleNamespace(policy_control=control, datas=[self.live]))
        self.teacher = DynamicSupportTeacher(self.task, lambda obs: torch.zeros(1, 30))

    def assert_static_plan(self, teacher):
        original = self.live.qpos.copy(), self.live.qvel.copy(), self.live.ctrl.copy()
        goals = self.live.xpos[teacher.body].copy()
        rotations = self.live.xmat[teacher.body].reshape(2, 3, 3).copy()
        started = time.monotonic()
        targets = teacher.command_targets(0, self.pose, self.pose, self.pose[teacher.indices],
            np.full(len(teacher.motors), .25), goals, rotations, 1,
            self.live.subtree_com[teacher.pelvis].copy(), self.live.xpos[teacher.pelvis].copy())
        print(f'Static force QP: {time.monotonic()-started:.3f}s', flush=True)
        solution = teacher.last_solution
        self.assertIsNotNone(solution)
        self.assertTrue(solution['diagnostics']['feasible'])
        residual = solution['dynamics'][:, :self.rig.model.nv] @ (solution['acceleration'] / 20.)
        residual += solution['dynamics'][:, self.rig.model.nv:] @ solution['coefficients']
        residual += solution['required']
        np.testing.assert_allclose(residual[teacher.free_dofs], 0., atol=.007)
        self.assertTrue(np.all(solution['coefficients'] >= -1e-8))
        self.assertTrue(np.all(abs(solution['torque']) <= teacher.scale + .002))
        self.assertTrue(np.all(abs(targets - self.pose[teacher.indices]) <= .25 + 1e-5))
        reconstructed = teacher.scale * np.clip(4. * (targets - self.pose[teacher.indices]), -1, 1)
        np.testing.assert_allclose(reconstructed, solution['torque'], atol=.003)
        for value, initial in zip((self.live.qpos, self.live.qvel, self.live.ctrl), original):
            np.testing.assert_array_equal(value, initial)

    def test_static_force_plan_respects_dynamics_motor_limits_and_live_state(self):
        self.assert_static_plan(self.teacher)

    def test_full_motor_contact_plan_preserves_all_actuator_bounds(self):
        teacher = DynamicSupportTeacher(self.task, self.teacher.actor,
                                        all_motors=True,contact_tracking_weight=20.)
        self.assertEqual(len(teacher.motors), 30)
        self.assert_static_plan(teacher)

    def test_swing_foot_cannot_supply_planned_ground_force(self):
        teacher = self.teacher
        teacher.phase[0] = 2
        teacher.foot[0] = 0
        _, feet = teacher.contact_columns(self.live, 0)
        self.assertGreater(len(feet), 0)
        self.assertTrue(np.all(feet == 1))

    def test_saturated_motor_is_eliminated_without_loosening_its_interval(self):
        dynamics = np.column_stack((np.eye(2) * 20., np.array([-10., 0.])))
        objective = np.array([[1., 0., 0.], [0., 1., 0.]])
        _, torque, diagnostics = solve_force_qp(objective, np.zeros(2), dynamics, np.array([9., 1.]),
            np.array([0]), np.array([1]), np.array([2.]), np.array([.5]), np.array([.5]), 1)
        np.testing.assert_allclose(torque, [.5], rtol=0, atol=1e-10)
        self.assertTrue(diagnostics['feasible'])

    def test_inverse_contact_map_predicts_native_motor_acceleration(self):
        model = self.rig.model
        data = mujoco.MjData(model)
        data.qpos[:] = self.pose
        data.ctrl[self.rig.actuators] = np.linspace(-.1, .1, 30)
        mujoco.mj_forward(model, data)
        acceleration = data.qacc.copy()
        force = data.qfrc_actuator.copy()
        dofs = model.jnt_dofadr[model.actuator_trnid[self.rig.actuators, 0]]
        mapping, offset, _, _, inverse, _ = inverse_contact_map(model, data, dofs, self.rig.action_scale)
        normalized = force[dofs] / self.rig.action_scale
        np.testing.assert_allclose(inverse, force, atol=1e-7)
        np.testing.assert_allclose(mapping @ normalized + offset, acceleration, atol=1e-7)
        data.ctrl[self.rig.actuators[12]] += .001
        normalized[12] += .001 / self.rig.action_scale[12]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(mapping @ normalized + offset, data.qacc, atol=1e-6)

    def test_inverse_teacher_preserves_live_state_and_motor_bounds(self):
        teacher = DynamicSupportTeacher(self.task, self.teacher.actor, all_motors=True,
            contact_tracking_weight=20., force_model='mujoco_inverse_contacts')
        original = self.live.qpos.copy(), self.live.qvel.copy(), self.live.ctrl.copy()
        targets = teacher.command_targets(0, self.pose, self.pose, self.pose[teacher.indices],
            np.full(30, .25), self.live.xpos[teacher.body].copy(),
            self.live.xmat[teacher.body].reshape(2, 3, 3).copy(), 1,
            self.live.subtree_com[teacher.pelvis].copy(), self.live.xpos[teacher.pelvis].copy())
        self.assertIsNotNone(teacher.last_solution)
        self.assertTrue(np.all(abs(targets - self.pose[teacher.indices]) <= .25 + 1e-5))
        np.testing.assert_allclose(teacher.last_solution['torque'],
            teacher.scale * np.clip(4. * (targets - self.pose[teacher.indices]), -1, 1), atol=.003)
        for value, initial in zip((self.live.qpos, self.live.qvel, self.live.ctrl), original):
            np.testing.assert_array_equal(value, initial)

    def test_refined_teacher_uses_realizable_forward_acceleration(self):
        teacher = DynamicSupportTeacher(self.task, self.teacher.actor, all_motors=True,
            contact_tracking_weight=20., force_model='mujoco_forward_refined')
        teacher.command_targets(0, self.pose, self.pose, self.pose[teacher.indices],
            np.full(30, .25), self.live.xpos[teacher.body].copy(),
            self.live.xmat[teacher.body].reshape(2, 3, 3).copy(), 1,
            self.live.subtree_com[teacher.pelvis].copy(), self.live.xpos[teacher.pelvis].copy())
        solution = teacher.last_solution
        self.assertTrue(solution['diagnostics']['feasible'])
        self.assertLessEqual(solution['diagnostics']['final_tracking_cost'],
                             solution['diagnostics']['initial_tracking_cost'])
        native = mujoco.MjData(self.rig.model)
        native.qpos[:] = self.pose
        native.ctrl[:] = self.live.ctrl
        native.ctrl[self.rig.actuators] = solution['torque']
        mujoco.mj_forward(self.rig.model, native)
        np.testing.assert_allclose(native.qacc, solution['acceleration'], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
