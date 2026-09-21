"""Verify that the offline search retains the production actuator and state contract."""
from pathlib import Path
import sys
import unittest

import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_recovery_feasibility import CONTROL, Snapshot, initial_state, native_controls
from perturb_env import PerturbEnv


class RecoveryFeasibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = PerturbEnv(num_envs=1, observation_version='foundation_v2')
        self.env.auto_reset = False
        self.env.next_ball[:] = np.inf
        self.rng = np.random.default_rng(71)

    def command(self):
        return torch.tensor(self.rng.uniform(-.1, .1, (1, self.env.num_actions)), dtype=torch.float32)

    def test_snapshot_preserves_observations_queue_and_continuation(self):
        env = self.env
        for _ in range(4):
            env.step(self.command())
        snapshot = Snapshot(env)
        clone = PerturbEnv(num_envs=1, observation_version='foundation_v2')
        snapshot.restore(clone)
        np.testing.assert_array_equal(env.get_observations(), clone.get_observations())
        for _ in range(8):
            action = self.command()
            a, b = env.step(action), clone.step(action)
            for j in range(3):
                np.testing.assert_array_equal(a[j], b[j])
            np.testing.assert_array_equal(initial_state(env), initial_state(clone))

    def test_threaded_rollout_matches_delayed_production_steps(self):
        env = self.env
        for _ in range(3):
            env.step(self.command())
        # Exercise ball/body contact and the transition back to projectile gravity.
        d = env.datas[0]
        d.qpos[env.ball_q:env.ball_q+3] = d.xpos[env.pelvis] + np.array([.23, 0., .15])
        d.qvel[env.ball_v:env.ball_v+3] = (-6., 0., 0.)
        mujoco.mj_forward(env.model, d)
        env.ball_flight[:] = (int(env.episode_length_buf[0])+3)*env.dt*env.decimation
        commands = self.rng.uniform(-1.5, 1.5, (1, 12, env.num_actions)).astype(np.float32)
        controls = native_controls(env, commands)
        with Rollout(nthread=2) as pool:
            predicted, _ = pool.rollout(env.model, [mujoco.MjData(env.model) for _ in range(2)],
                initial_state(env), controls, control_spec=CONTROL, initial_warmstart=d.qacc_warmstart)
        actual = []
        for step, command in enumerate(commands[0]):
            env.step(torch.tensor(command[None]))
            actual.append(initial_state(env))
            np.testing.assert_array_equal(d.ctrl, controls[0, step*env.decimation, :env.model.nu])
            np.testing.assert_array_equal(d.xfrc_applied.reshape(-1), controls[0, step*env.decimation, env.model.nu:])
        np.testing.assert_allclose(predicted[0, env.decimation-1::env.decimation], actual, atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
