"""The Perturb seed must preserve stationary inputs and release either foot after impact."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class ReactiveBallTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH") or not os.environ.get("MIMIC_STATIONARY_REFERENCE"):
            raise unittest.SkipTest("Set MimicKit and stationary reference paths")
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .rig import Rig
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        cls.reference = Path(os.environ["MIMIC_STATIONARY_REFERENCE"])

    def test_radius_variant_preserves_character_and_sphere_inertia(self):
        from .ball import load_ball_rig
        from .reactive_ball_task import write_ball_variant
        source = self.rig.path.with_name("dummy_ball.xml")
        with tempfile.TemporaryDirectory() as directory:
            variant = write_ball_variant(source, Path(directory) / "ball.xml", 6., radius_m=.12)
            world = load_ball_rig(self.rig, variant)
            body = world.model.body("ball").id
            self.assertAlmostEqual(world.model.geom("g_ball").size[0], .12)
            np.testing.assert_allclose(world.model.body_inertia[body], .4 * 6. * .12**2)
            np.testing.assert_array_equal(world.model.body_mass[:body], self.rig.model.body_mass)
            with self.assertRaises(ValueError):
                write_ball_variant(source, source, 6., radius_m=.12)

    def test_target_curricula_cover_requested_bodies_and_preserve_default_rng(self):
        from envs.base_env import EnvMode
        from .reactive_ball_task import ReactiveBallTask, TARGET_SETS
        common = dict(ball_path=self.rig.path.with_name("dummy_ball.xml"), episode_seconds=12.,
                      second_shot_fraction=1., seed=928)
        default = ReactiveBallTask(self.rig, self.reference, 32, "cpu", **common)
        explicit = ReactiveBallTask(self.rig, self.reference, 32, "cpu", target_set="torso", **common)
        for task in (default, explicit):
            task.set_mode(EnvMode.TRAIN)
            task.reset()
        for name in ("shot_target_body", "second_shot_target_body", "shot_steps", "directions", "shot_enabled"):
            torch.testing.assert_close(getattr(default, name), getattr(explicit, name), atol=0, rtol=0)
        for target_set in ("torso_head", "all"):
            task = ReactiveBallTask(self.rig, self.reference, 32, "cpu", target_set=target_set, **common)
            task.set_mode(EnvMode.TRAIN)
            observed, second = set(), set()
            for _ in range(8):
                task.reset()
                observed.update(task.shot_target_body.tolist())
                second.update(task.second_shot_target_body.tolist())
            expected = {self.rig.model.body(name).id-1 for name in TARGET_SETS[target_set]}
            self.assertEqual(observed, expected)
            self.assertEqual(second, expected)
            self.assertEqual(task.observation_contract(), default.observation_contract())
            self.assertEqual(task.experiment_contract()["target_set"], target_set)
        with self.assertRaises(ValueError):
            ReactiveBallTask(self.rig, self.reference, target_set="unknown", **common)


    def test_projectile_variant_and_stationary_observation_parity(self):
        from .baseline import sha256
        from .ball import load_ball_rig
        from .reactive_ball_task import ReactiveBallTask, write_ball_variant
        from .continuous_stand_task import ContinuousStandTask
        from .native_engine import NativeDummyEngine
        from .checkpoints import validate_contract
        source = self.rig.path.with_name("dummy_ball.xml")
        before = sha256(source)
        with tempfile.TemporaryDirectory() as directory:
            variant = write_ball_variant(source, Path(directory) / "ball.xml", 2.)
            self.assertEqual(sha256(source), before)
            world = load_ball_rig(self.rig, variant)
            self.assertEqual(float(world.model.body_mass[world.model.body("ball").id]), 2.)
            task = ReactiveBallTask(self.rig, self.reference, 8, "cpu", ball_path=variant)
            stand = ContinuousStandTask(self.rig, self.reference, 8, "cpu",
                                         engine_factory=NativeDummyEngine)
            contract = json.loads(Path("logs/mimickit-training/stand-foundation-anchor-tight-01/contract.json").read_text())
            validate_contract(contract, task.observation_contract())
            task.shot_enabled[:] = False
            for _ in range(12):
                torch.testing.assert_close(task.observations(), stand.observations(), atol=2e-5, rtol=1e-5)
                action = torch.zeros(8, 30)
                task.step(action)
                stand.step(action)

    def test_post_hit_reward_releases_fixed_foot_anchor(self):
        from .reactive_ball_task import ReactiveBallTask, write_ball_variant
        with tempfile.TemporaryDirectory() as directory:
            variant = write_ball_variant(self.rig.path.with_name("dummy_ball.xml"),
                                         Path(directory) / "ball.xml", 2.)
            task = ReactiveBallTask(self.rig, self.reference, 8, "cpu", ball_path=variant)
            task.shot_enabled[:] = True
            task._fixed_feet += 1.
            quiet = task.reward()
            self.assertTrue((quiet < 1e-10).all())
            task.engine.hit[:] = True
            recovered = task.reward()
            self.assertTrue((recovered > .1).all())

    def test_capture_reward_uses_character_com_and_keeps_quiet_actor_contract(self):
        from .reactive_ball_task import ReactiveBallTask
        from .checkpoints import validate_contract
        old = ReactiveBallTask(self.rig, self.reference, 8, "cpu",
                               ball_path=self.rig.path.with_name("dummy_ball.xml"))
        task = ReactiveBallTask(self.rig, self.reference, 8, "cpu",
                                ball_path=self.rig.path.with_name("dummy_ball.xml"), recovery_mode="capture_v3")
        validate_contract(old.observation_contract(), task.observation_contract())
        expected = torch.tensor(np.stack([data.subtree_com[1].copy() for data in task.engine.datas]), dtype=torch.float32)
        torch.testing.assert_close(task.center_of_mass(), expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(old.reward(), task.reward())
        old.shot_enabled[:] = task.shot_enabled[:] = False
        for _ in range(4):
            _, old_reward, _, _ = old.step(torch.zeros(8, 30))
            _, new_reward, _, _ = task.step(torch.zeros(8, 30))
            torch.testing.assert_close(old_reward, new_reward)
            torch.testing.assert_close(old.observations(), task.observations())

    def test_step_guidance_leaves_quiet_reward_and_observations_unchanged(self):
        from .reactive_ball_task import ReactiveBallTask
        tasks = [ReactiveBallTask(self.rig, self.reference, 8, "cpu",
                    ball_path=self.rig.path.with_name("dummy_ball.xml"), recovery_mode=mode)
                 for mode in ("capture_v3", "step_v4", "step_v5")]
        for task in tasks:
            task.shot_enabled[:] = False
        for _ in range(6):
            results = [task.step(torch.zeros(8, 30)) for task in tasks]
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
            torch.testing.assert_close(results[0][0], results[2][0], rtol=0, atol=0)
            torch.testing.assert_close(results[0][1], results[2][1], rtol=0, atol=0)

    def test_post_hit_resets_restore_motor_memory_and_never_change_test_starts(self):
        from envs.base_env import EnvMode
        from .reactive_ball_task import ReactiveBallTask
        path = Path("logs/mimickit-training/recovery-post-hit-resets-01/recovery_resets.npz")
        if not path.exists():
            self.skipTest("Generate physical post-hit reset bank first")
        task = ReactiveBallTask(self.rig, self.reference, 8, "cpu", quiet_fraction=0.,
                                ball_path=Path("logs/mimickit-training/reactive-ball-seed-01/ball_5.xml"),
                                recovery_mode="capture_v3", recovery_reset_fraction=1.)
        task.load_recovery_resets(path)
        task.set_mode(EnvMode.TRAIN)
        task.reset()
        self.assertTrue(task.engine.hit.all())
        self.assertTrue((task.shot_steps == -1).all())
        self.assertTrue((task.engine.policy_control.active[:, 60] == 1).all())
        self.assertTrue((task.engine.policy_control.pending[:, :, 60] == 1).all())
        before = task.observations().clone()
        task.reset(torch.tensor([0]))
        torch.testing.assert_close(before[1:], task.observations()[1:], rtol=0, atol=0)
        task.set_mode(EnvMode.TEST)
        task.reset()
        self.assertFalse(task.engine.hit.any())
        self.assertTrue((task.shot_steps == 90).all())
        q, _ = task.sample_reference(task.offset)
        torch.testing.assert_close(task.engine.native_qpos(), q)

    @unittest.skipUnless(os.environ.get("MIMIC_TEST_GPU") == "1", "Set MIMIC_TEST_GPU=1 for backend parity")
    def test_recovery_reset_native_gpu_parity(self):
        from envs.base_env import EnvMode
        from .reactive_ball_task import ReactiveBallTask
        tasks = []
        for device in ("cpu", "cuda:0"):
            task = ReactiveBallTask(self.rig, self.reference, 8, device, quiet_fraction=0.,
                                    ball_path=Path("logs/mimickit-training/reactive-ball-seed-01/ball_5.xml"),
                                    recovery_mode=os.environ.get("MIMIC_RECOVERY_MODE", "capture_v3"), recovery_reset_fraction=1.)
            task.load_recovery_resets(Path(os.environ.get("MIMIC_RECOVERY_RESET_BANK",
                "logs/mimickit-training/recovery-post-hit-resets-01/recovery_resets.npz")))
            task.recovery_groups = [group[:1] for group in task.recovery_groups]
            task.set_mode(EnvMode.TRAIN)
            task.reset()
            tasks.append(task)
        torch.testing.assert_close(tasks[0].observations(), tasks[1].observations().cpu(), atol=2e-5, rtol=1e-5)
        for _ in range(8):
            outputs = [task.step(torch.zeros(8, 30, device=task.device)) for task in tasks]
            torch.testing.assert_close(outputs[0][0], outputs[1][0].cpu(), atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(outputs[0][1], outputs[1][1].cpu(), atol=.003, rtol=.003)


if __name__ == "__main__":
    unittest.main()
