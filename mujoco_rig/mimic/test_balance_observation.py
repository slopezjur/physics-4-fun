import os
from pathlib import Path
import unittest

import numpy as np
import torch


class BalanceObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH") or not os.environ.get("MIMIC_STATIONARY_REFERENCE"):
            raise unittest.SkipTest("Set MimicKit and stationary reference paths")
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .rig import Rig
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        cls.reference = Path(os.environ["MIMIC_STATIONARY_REFERENCE"])

    def make_task(self, device, capture=True):
        from .balance_observation import CONTRACT, task_classes
        _, task = task_classes(dict(observation_contract=CONTRACT if capture else "old"))
        return task(self.rig, self.reference, 8, device,
                    ball_path=Path("logs/mimickit-training/reactive-ball-seed-01/ball_5.xml"))

    def test_core_and_native_com_match_and_projectile_is_excluded(self):
        import mujoco
        from .capture_reward import capture_distance
        task, old = self.make_task("cpu"), self.make_task("cpu", False)
        torch.testing.assert_close(task.observations()[:, :362], old.observations(), atol=0, rtol=0)
        task.engine.datas[0].qvel[:3] = [.3, -.2, .1]
        task.engine.datas[0].qvel[-6:-3] = [100, 200, 300]
        for data in task.engine.datas:
            mujoco.mj_forward(task.engine.rig.model, data)
            mujoco.mj_subtreeVel(task.engine.rig.model, data)
        velocity = torch.tensor(np.stack([data.subtree_linvel[1] for data in task.engine.datas]), dtype=torch.float32)
        torch.testing.assert_close(task.engine.get_center_of_mass_velocity(), velocity)
        com = torch.tensor(np.stack([data.subtree_com[1] for data in task.engine.datas]), dtype=torch.float32)
        obs = task.observations()
        expected = capture_distance(com, velocity, task.engine.get_body_pos(0)[:, task.feet], obs[:,93:95]*1000)
        torch.testing.assert_close(obs[:,362], expected, atol=2e-6, rtol=1e-5)
        self.assertEqual(task.observation_contract()["num_obs"], 363)

    @unittest.skipUnless(os.environ.get("MIMIC_TEST_GPU") == "1", "Set MIMIC_TEST_GPU=1")
    def test_native_gpu_observation_and_com_velocity_parity(self):
        from envs.base_env import EnvMode
        tasks = [self.make_task(device) for device in ("cpu", "cuda:0")]
        for task in tasks:
            task.load_recovery_resets(Path("logs/mimickit-training/step-native-exploration-2mps-01/recovery_resets.npz"))
            task.recovery_reset_fraction = 1.
            task.quiet_fraction = 0.
            task.recovery_groups = [group[:1] for group in task.recovery_groups]
            task.set_mode(EnvMode.TRAIN)
            task.reset()
        for _ in range(8):
            torch.testing.assert_close(tasks[0].observations(), tasks[1].observations().cpu(), atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(tasks[0].engine.get_center_of_mass_velocity(),
                tasks[1].engine.get_center_of_mass_velocity().cpu(), atol=2e-4, rtol=2e-4)
            for task in tasks:
                task.step(torch.zeros(8,30,device=task.device))


if __name__ == "__main__":
    unittest.main()
