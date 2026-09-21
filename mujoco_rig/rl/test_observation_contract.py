"""Foundation sensors, delayed control state and safe checkpoint migration."""
import pathlib
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from observation_contract import LEGACY, FOUNDATION, checkpoint_version, layout
from perturb_env import PerturbEnv
from ppo import PPO
from train import PerturbTask, parse_args, save_checkpoint, seed_from_checkpoint


class FoundationTests(unittest.TestCase):
    def test_sensor_velocity_uses_native_body_origin_and_legacy_prefix_is_unchanged(self):
        legacy = PerturbEnv(num_envs=2, seed=19)
        env = PerturbEnv(num_envs=2, seed=19, observation_version=FOUNDATION)
        for i, d in enumerate(env.datas):
            d.qvel[:6] = [.3, -.2, .1, 1., -.7, .4]
            legacy.datas[i].qpos[:] = d.qpos
            legacy.datas[i].qvel[:] = d.qvel
            mujoco.mj_forward(env.model, d)
            mujoco.mj_forward(legacy.model, legacy.datas[i])
            native = np.zeros(6)
            mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_XBODY, env.pelvis, native, 0)
            actual = env.get_observations()[i].numpy()
            np.testing.assert_allclose(actual[105:108], d.xmat[env.pelvis].reshape(3, 3).T @ native[3:], atol=1e-6)
            load, _ = env.foot_contacts.read(d)
            np.testing.assert_allclose(actual[108:110], load * .001, atol=1e-6)
        torch.testing.assert_close(env.get_observations()[:, :105], legacy.get_observations(), rtol=0, atol=0)
        self.assertGreater(float(torch.linalg.vector_norm(env.get_observations()[:, 3:6]
                                                       - env.get_observations()[:, 105:108])), .01)

    def test_queue_exposes_applied_next_and_newest_actions_and_resets_selected_world(self):
        env = PerturbEnv(num_envs=2, observation_version=FOUNDATION)
        env.auto_reset = False
        first = torch.full((2, 30), .2)
        second = torch.full((2, 30), -.3)
        obs, *_ = env.step(first)
        torch.testing.assert_close(obs[:, 72:102], first)
        torch.testing.assert_close(obs[:, 110:], torch.zeros_like(first))
        obs, *_ = env.step(second)
        torch.testing.assert_close(obs[:, 72:102], second)
        torch.testing.assert_close(obs[:, 110:], first)
        env.step(torch.zeros_like(first))
        np.testing.assert_allclose(env.datas[0].ctrl[env.act_idx], first[0].numpy() * env.force_limit * env.authority, rtol=1e-6)
        env.reset_idx([0])
        obs = env.get_observations()
        torch.testing.assert_close(obs[0, 110:], torch.zeros(30))
        torch.testing.assert_close(obs[1, 110:], second[1])

    def test_migration_preserves_actor_and_std_but_refits_critic_then_resumes(self):
        old_env = PerturbEnv(num_envs=1)
        new_env = PerturbEnv(num_envs=1, observation_version=FOUNDATION)
        old, new = PPO(105, 30), PPO(140, 30)
        args = parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'old.pt'
            save_checkpoint(path, old, old_env, args, 16, 6)
            args = parse_args(['--init_from', str(path), '--critic_warmup_iters', '0'])
            seed_from_checkpoint(new, new_env, args, 'cpu')
            obs = torch.randn(128, 140)
            torch.testing.assert_close(new.net.actor(obs), old.net.actor(obs[:, :105]), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(new.net.log_std, old.net.log_std, rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(new.net.actor[0].weight[:, 105:]), 0)
            self.assertEqual(new.critic_warmup_remaining, 50)
            self.assertFalse(new.opt.state)
            self.assertFalse(new.critic_opt.state)
            self.assertFalse(torch.equal(new.net.critic[0].weight[:, :105], old.net.critic[0].weight))
            path = pathlib.Path(tmp) / 'new.pt'
            save_checkpoint(path, new, new_env, args, 16, 6)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint_version(checkpoint), FOUNDATION)
            resume_args = parse_args(['--init_from', str(path)])
            self.assertEqual(PerturbTask().env_kwargs(resume_args)['observation_version'], FOUNDATION)
            resumed = PPO(140, 30)
            seed_from_checkpoint(resumed, new_env, resume_args, 'cpu')
            for key, value in new.net.state_dict().items():
                torch.testing.assert_close(value, resumed.net.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(resumed.critic_warmup_remaining, 50)

    def test_reference_migration_requires_explicit_recollection(self):
        old_env = PerturbEnv(num_envs=1)
        new_env = PerturbEnv(num_envs=1, observation_version=FOUNDATION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'old.pt'
            save_checkpoint(path, PPO(105, 30), old_env, parse_args([]), 16, 6)
            checkpoint = torch.load(path, weights_only=False)
            # This marker must be rejected before parsing or reusing its targets.
            checkpoint['policy_reference'] = {}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'explicit recollected'):
                seed_from_checkpoint(PPO(140, 30), new_env,
                                     parse_args(['--init_from', str(path)]), 'cpu')

    def test_contract_rejects_unknown_or_unlabelled_widened_checkpoints(self):
        self.assertEqual(checkpoint_version(dict(num_obs=105, num_actions=30)), LEGACY)
        for version in (LEGACY, 'future_v9'):
            with self.assertRaises(ValueError):
                checkpoint_version(dict(num_obs=140, num_actions=30, observation_version=version))
        self.assertEqual(sum(c['width'] for c in layout(30, FOUNDATION)), 140)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires training GPU')
    def test_cpu_gpu_physical_sensors_history_and_partial_reset(self):
        from perturb_env_warp import PerturbEnvWarp
        cpu = PerturbEnv(num_envs=4, observation_version=FOUNDATION)
        gpu = PerturbEnvWarp(num_envs=4, observation_version=FOUNDATION)
        for i, d in enumerate(cpu.datas):
            d.qpos[:] = cpu.rest_qpos
            d.qpos[2] += .2 if i == 3 else 0
            d.qvel[:6] = [.3, -.1, .2, .4, -.3, .2]
            mujoco.mj_forward(cpu.model, d)
            gpu.qpos[i] = torch.as_tensor(d.qpos, device='cuda', dtype=torch.float32)
            gpu.qvel[i] = torch.as_tensor(d.qvel, device='cuda', dtype=torch.float32)
        gpu._mjw.forward(gpu._m, gpu._d)
        cpu.action_queue[0][:] = .3
        gpu.action_queue[0][:] = .3
        cpu.prev_action[:] = -.2
        gpu.prev_action[:] = -.2
        expected = cpu.get_observations()
        actual = gpu.get_observations().cpu()
        torch.testing.assert_close(actual[:, :108], expected[:, :108], rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual[:, 108:110], expected[:, 108:110], rtol=.03, atol=.005)
        torch.testing.assert_close(actual[:, 110:], expected[:, 110:])
        self.assertTrue(torch.all(actual[3, 108:110] == 0))
        gpu.reset_idx(torch.tensor([1], device='cuda'))
        actual = gpu.get_observations().cpu()
        self.assertTrue(torch.all(actual[1, 110:] == 0))
        self.assertTrue(torch.all(actual[0, 110:] == .3))
        gpu.assert_buffers_ok()


if __name__ == '__main__':
    unittest.main()
