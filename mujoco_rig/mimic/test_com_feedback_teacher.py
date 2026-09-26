"""Native COM teacher sign, world independence, and action bounds."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from .com_feedback_teacher import ComFeedbackTeacher
from .rig import Rig


class ComFeedbackTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            cls.pose = arrays["qpos"][0].copy()

    def task(self, worlds=2):
        data = [mujoco.MjData(self.rig.model) for _ in range(worlds)]
        for world in data:
            world.qpos[:] = self.pose
            mujoco.mj_forward(self.rig.model, world)
        reference = SimpleNamespace(
            qpos=torch.tensor(np.repeat(self.pose[None, :], 2, axis=0)),
            qvel=torch.zeros((2, self.rig.model.nv)), dt=.05)
        return SimpleNamespace(rig=self.rig, engine=SimpleNamespace(rig=self.rig, datas=data),
                               reference=reference, device="cpu", dt=.05,
                               offset=torch.zeros(worlds), steps=torch.zeros(worlds),
                               get_num_envs=lambda: worlds)

    def test_opposite_com_offsets_produce_bounded_opposite_ankle_actions(self):
        task = self.task()
        task.engine.datas[0].qpos[0] += .1
        task.engine.datas[1].qpos[0] -= .1
        for data in task.engine.datas:
            mujoco.mj_forward(self.rig.model, data)
        actor = lambda observation: torch.zeros((2, len(self.rig.actuators)))
        teacher = ComFeedbackTeacher(task, actor, position_gain=2., velocity_gain=0.,
                                     max_correction=.15)
        action = teacher(torch.zeros((2, 1))).numpy()
        for ankle in teacher.ankles:
            self.assertAlmostEqual(action[0, ankle], -.15, places=6)
            self.assertAlmostEqual(action[1, ankle], .15, places=6)
        self.assertTrue(np.all(action[:, teacher.hips] == 0))
        self.assertEqual(teacher.contract()["schema"], "mimic_native_com_teacher_v1")

    def test_rejects_nonfinite_feedback_parameters(self):
        task = self.task(1)
        actor = lambda observation: torch.zeros((1, len(self.rig.actuators)))
        with self.assertRaises(ValueError):
            ComFeedbackTeacher(task, actor, position_gain=float("nan"), velocity_gain=1.)


if __name__ == "__main__":
    unittest.main()
