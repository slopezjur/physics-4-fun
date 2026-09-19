"""CPU regression tests for deployment contracts, episode isolation and scoring decisions.

Run: python -m unittest discover -s mujoco_rig/rl -p test_architecture.py -v
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco_rig" / "rl"))
sys.path.insert(0, str(ROOT / "mujoco_rig" / "scripts"))

from walk_config import WalkCommandConfig
from walk_env import WalkEnv
from perturb_env import PerturbEnv
from export_onnx import build_contract
from policy_artifacts import publish_pair
from env_config import target_probabilities
import overnight
import promote
import scoring


class WalkStateTests(unittest.TestCase):
    def test_command_configuration_rejects_invalid_ranges_and_probabilities(self):
        for kwargs in [dict(forward=(1, 1)), dict(mix=(0.5, 0.5, 0.5, 0)),
                       dict(turn=(0, float("nan")))]:
            with self.assertRaises(ValueError):
                WalkCommandConfig(**kwargs)

    def test_target_weights_reject_nonfinite_values(self):
        for value in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                target_probabilities({"Head": value})

    def test_stage_two_can_sample_reverse_and_lateral_commands(self):
        env = WalkEnv(num_envs=64, command_config=WalkCommandConfig.for_stage(2))
        self.assertTrue((env.commands[:, 0] < 0).any())
        self.assertTrue((env.commands[:, 1] != 0).any())
        np.testing.assert_array_equal(env.cmd_yaw, 0)

    def test_disabling_automatic_reset_preserves_explicit_reset(self):
        env = WalkEnv(num_envs=1, episode_seconds=0.02)
        env.auto_reset = False
        env.step(torch.zeros(1, env.num_actions))
        self.assertGreaterEqual(int(env.episode_length_buf[0]), env.max_episode_length)
        env.reset_all()
        self.assertEqual(int(env.episode_length_buf[0]), 0)

    def test_single_hit_is_episode_configuration_and_resets(self):
        env = PerturbEnv(num_envs=1)
        env.max_shots_per_episode = 1
        env.fire_ball(env.datas[0])
        first_velocity = env.datas[0].qvel.copy()
        env.fire_ball(env.datas[0])
        self.assertEqual(env.shots_fired[0], 1)
        np.testing.assert_array_equal(env.datas[0].qvel, first_velocity)
        env.reset_all()
        self.assertEqual(env.shots_fired[0], 0)
        env.fire_ball(env.datas[0])
        self.assertEqual(env.shots_fired[0], 1)

    def test_cpu_environment_does_not_import_gpu_backend(self):
        code = ("import sys; sys.path.insert(0, 'mujoco_rig/rl'); import walk_env; "
                "assert 'walk_env_warp' not in sys.modules; assert 'body_env_warp' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    def test_instances_keep_their_own_command_stage(self):
        slow = WalkEnv(num_envs=2, command_config=WalkCommandConfig.for_stage(1))
        wide = WalkEnv(num_envs=2, command_config=WalkCommandConfig.for_stage(3))
        slow._resample_commands([0, 1])
        self.assertEqual(slow.command_config.forward, (0.15, 0.35))
        self.assertEqual(wide.command_config.forward, (-0.3, 1.0))
        np.testing.assert_array_equal(slow.cmd_yaw, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            slow.command_config.forward = (0, 2)

    def test_landing_rewards_airtime_before_clearing_it(self):
        env = WalkEnv(num_envs=1)
        env.set_command(0.25)
        env.last_landed[:] = 0  # Isolate airtime from the separate alternation bonus.
        action = np.zeros(env.num_actions)
        baseline = env.reward(0, action)
        env.air_time[0] = (0.2, 0)
        env.was_airborne[0] = (True, False)
        reward = env.reward(0, action)
        self.assertAlmostEqual(reward - baseline, 1.5 * 0.2)
        np.testing.assert_array_equal(env.air_time, 0)

    def test_reset_clears_gait_history_and_heading_correction(self):
        env = WalkEnv(num_envs=2)
        env.set_command(0.25)
        env.commands[:, 2] = 0.7
        env.last_landed[:] = 0
        env.air_time[:] = 0.3
        env.command_age[:] = 4
        env.reset_idx(iter([0]))
        self.assertEqual(env.last_landed[0], 1)
        self.assertEqual(env.commands[0, 2], 0)
        self.assertEqual(env.command_age[0], 0)
        np.testing.assert_array_equal(env.air_time[0], 0)
        self.assertEqual(env.last_landed[1], 0)

    def test_export_contract_records_latency_and_checks_dimensions(self):
        env = WalkEnv(num_envs=1)
        ck = dict(num_obs=env.num_obs, num_actions=env.num_actions)
        contract = build_contract("walk", "checkpoint.pt", ck, env)
        self.assertEqual(contract["action_latency_steps"], len(env.action_queue))
        ck["num_actions"] += 1
        with self.assertRaises(ValueError):
            build_contract("walk", "checkpoint.pt", ck, env)


class ScoringTests(unittest.TestCase):
    def test_paired_test_rejects_broadcasting_and_empty_masks(self):
        for left, right in [([True, False], [True]), ([], [])]:
            with self.assertRaises(ValueError):
                scoring.paired_z(left, right)

    def test_failed_scorer_exit_cannot_supply_a_result(self):
        scorer = scoring.PerturbScorer()
        with patch.object(scoring, "run_concurrently", return_value=[1]), \
             patch.object(scorer, "read") as read:
            # Give the diagnostic path a log just as the subprocess would.
            original = pathlib.Path.read_text
            with patch.object(pathlib.Path, "read_text", lambda path, **kw:
                              "scorer failed" if path.suffix == ".log" else original(path, **kw)):
                self.assertEqual(scoring.score_all(scorer, ["ck"], {}), [None])
            read.assert_not_called()

    def test_missing_incumbent_checkpoint_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(promote, "ROOT", pathlib.Path(tmp)):
            directory = pathlib.Path(tmp) / "mujoco_rig"
            directory.mkdir()
            (directory / "balance_policy.contract.json").write_text(
                json.dumps({"source_checkpoint": "missing.pt"}))
            with self.assertRaises(ValueError):
                promote.incumbent_of(scoring.PerturbScorer())

    def test_unscorable_incumbent_never_ships_challenger(self):
        with patch.object(sys, "argv", ["promote", "--task", "perturb", "--checkpoint", "new.pt"]), \
             patch.object(promote, "incumbent_of", return_value="old.pt"), \
             patch.object(promote, "score_all", return_value=[{"survived": 80}, None]), \
             patch.object(promote, "ship") as ship:
            self.assertEqual(promote.main(), 1)
            ship.assert_not_called()

    def test_nonfinite_score_never_ships_into_empty_slot(self):
        with patch.object(sys, "argv", ["promote", "--task", "perturb", "--checkpoint", "new.pt"]), \
             patch.object(promote, "incumbent_of", return_value=None), \
             patch.object(promote, "score_all", return_value=[{"survived": float("nan")}]), \
             patch.object(promote, "ship") as ship:
            self.assertEqual(promote.main(), 1)
            ship.assert_not_called()

    def test_overnight_promotes_best_not_latest_tolerated_seed(self):
        def train(cmd, log):
            pathlib.Path(log).write_text("it 25 return 10 ep_len 1000/1200 vx 0.25\n")
            return 0

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict("os.environ", {"P4F_NIGHT": tmp}), \
             patch.object(sys, "argv", ["overnight", "--task", "walk", "--sessions", "2",
                                       "--abort_at", "0", "--ball_speed", "3"]), \
             patch.object(overnight, "run", side_effect=train), \
             patch.object(overnight, "newest_run", return_value="run"), \
             patch.object(overnight, "newest_checkpoint", side_effect=["best.pt", "later.pt"]), \
             patch.object(overnight.POLICIES["walk"], "score",
                          side_effect=[(90, "90 | detail", 3, None), (88, "88 | detail", 3, None)]), \
             patch.object(overnight.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as ship:
            self.assertEqual(overnight.main(), 0)
            arguments = ship.call_args.args[0]
            self.assertEqual(arguments[arguments.index("--checkpoint") + 1], "best.pt")

    def test_overnight_compares_first_session_against_initial_seed(self):
        def train(cmd, log):
            pathlib.Path(log).write_text("it 25 return 10 ep_len 1000/1200 vx 0.25\n")
            return 0

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict("os.environ", {"P4F_NIGHT": tmp}), \
             patch.object(sys, "argv", ["overnight", "--task", "walk", "--sessions", "1",
                                       "--abort_at", "0", "--ball_speed", "3", "--seed", "seed.pt"]), \
             patch.object(overnight, "run", side_effect=train), \
             patch.object(overnight, "newest_run", return_value="run"), \
             patch.object(overnight, "newest_checkpoint", return_value="regressed.pt"), \
             patch.object(overnight.POLICIES["walk"], "score",
                          side_effect=[(90, "90 | seed", 3, None), (40, "40 | regression", 5, None)]), \
             patch.object(overnight.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as ship:
            self.assertEqual(overnight.main(), 0)
            arguments = ship.call_args.args[0]
            self.assertEqual(arguments[arguments.index("--checkpoint") + 1], "seed.pt")


class ArtifactTests(unittest.TestCase):
    def test_partial_publish_restores_previous_model_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, target = root / "staged.onnx", root / "shipped.onnx"
            for path, content in [(source, b"new"), (target, b"old")]:
                path.write_bytes(content)
                path.with_suffix(".contract.json").write_bytes(content + b" contract")
            replace = pathlib.Path.replace

            def fail_contract(path, destination):
                if path.suffix == ".json":
                    raise OSError("injected write failure")
                return replace(path, destination)

            with patch.object(pathlib.Path, "replace", fail_contract), self.assertRaises(OSError):
                publish_pair(source, target)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(target.with_suffix(".contract.json").read_bytes(), b"old contract")


if __name__ == "__main__":
    unittest.main()
