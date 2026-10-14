"""Recovery AMP history, state transfer and discriminator sampling contracts."""
import os
from pathlib import Path
import unittest

import torch

from .runtime import activate


class AMPRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for AMP integration tests")
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .amp_recovery import MotionHistory, actor_state_dict, amp_sidecar, validate_sidecar
        cls.History = MotionHistory
        cls.actor_state_dict = staticmethod(actor_state_dict)
        cls.amp_sidecar = staticmethod(amp_sidecar)
        cls.validate_sidecar = staticmethod(validate_sidecar)

    def test_history_reset_cannot_splice_two_episodes(self):
        history = self.History(torch.tensor([[1.], [10.]]))
        self.assertEqual(history.append(torch.tensor([[2.], [20.]])).tolist(),
                         [[2., 1., 1., 1.], [20., 10., 10., 10.]])
        history.reset(torch.tensor([0]), torch.tensor([[100.], [20.]]))
        self.assertEqual(history.append(torch.tensor([[101.], [30.]])).tolist(),
                         [[101., 100., 100., 100.], [30., 20., 10., 10.]])

    def test_portable_checkpoint_preserves_actor_normalizers(self):
        state = {"_model._disc_layers.weight": 1, "_model._disc_logits.bias": 2,
                 "_disc_obs_norm._mean": 3, "_obs_norm._mean": 4,
                 "_model._actor_layers.base.weight": 5, "_model._critic_layers.weight": 6}
        self.assertEqual(self.actor_state_dict(state), {key: value for key, value in state.items()
                         if value >= 4})
        self.assertEqual(self.amp_sidecar("best.pt.tmp"), Path("best.amp.pt"))

    def test_training_sidecar_cannot_silently_replace_selected_actor(self):
        portable = {"_model._actor_layers.weight": torch.tensor([1.])}
        training = {**portable, "_model._disc_logits.weight": torch.tensor([2.])}
        self.validate_sidecar(portable, training)
        training["_model._actor_layers.weight"] = torch.tensor([3.])
        with self.assertRaisesRegex(ValueError, "hash-bound"):
            self.validate_sidecar(portable, training)

    def test_mirrored_hinges_match_physical_sagittal_reflection(self):
        import mujoco
        import numpy as np
        from .amp_recovery import mirror_state
        from .rig import Rig
        root = Path(__file__).resolve().parents[2]
        rig = Rig.load(root / "mujoco_rig/dummy.xml")
        q = torch.tensor(rig.rest)[None].repeat(2, 1)
        q[0, rig.model.joint("Thigh_L_rx").qposadr[0]] = .3
        q[1, rig.model.joint("Foot_R_rz").qposadr[0]] = -.08
        v = torch.arange(2 * rig.model.nv, dtype=q.dtype).reshape(2, -1) / 100
        mirrored_q, mirrored_v = mirror_state(q, v, rig)
        restored_q, restored_v = mirror_state(mirrored_q, mirrored_v, rig)
        torch.testing.assert_close(restored_q, q)
        torch.testing.assert_close(restored_v, v)
        left, right = mujoco.MjData(rig.model), mujoco.MjData(rig.model)
        for original, reflected in zip(q, mirrored_q):
            left.qpos[:], right.qpos[:] = original.numpy(), reflected.numpy()
            mujoco.mj_kinematics(rig.model, left)
            mujoco.mj_kinematics(rig.model, right)
            for name in ("Thigh", "Shin", "Foot", "Toe"):
                np.testing.assert_allclose(left.xpos[rig.model.body(name + "_L").id] * [1, -1, 1],
                    right.xpos[rig.model.body(name + "_R").id], atol=1e-5)

    def test_foot_cost_allows_single_step_and_detects_contact_chatter(self):
        from .foot_motion import FootMotionCost
        position = torch.zeros(1, 2, 3)
        loads = torch.full((1, 2), 100.)
        cost = FootMotionCost(position, loads, .02)
        for _ in range(4):
            value, _, _ = cost.update(position, loads)
            self.assertEqual(value.item(), 0.)
        swing = loads.clone()
        swing[:, 0] = 0
        value, _, reversal = cost.update(position, swing)
        self.assertEqual(value.item(), 0.)
        self.assertFalse(reversal.any())
        value, _, reversal = cost.update(position, loads)
        self.assertGreater(value.item(), 0.)
        self.assertTrue(reversal[0, 0])
        cost.reset(torch.tensor([0]), position, loads)
        for _ in range(4):
            value, _, _ = cost.update(position, loads)
            self.assertEqual(value.item(), 0.)
        fast = position.clone()
        fast[0, 0, 0] = .1
        value, _, _ = cost.update(fast, loads)
        self.assertGreater(value.item(), 0.)
        self.assertLessEqual(value.item(), 1.)

    def test_calibration_initializes_loaded_moments_without_changing_actor(self):
        from tempfile import TemporaryDirectory
        from learning.normalizer import Normalizer
        from learning.mp_optimizer import MPOptimizer
        from .calibrate_amp_recovery import calibrate
        from util import mp_util
        mp_util.init(0, 1, "cpu", None)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._disc_logits = torch.nn.Linear(2, 1)
                self.actor = torch.nn.Linear(2, 1)
                torch.nn.init.zeros_(self._disc_logits.weight)
                torch.nn.init.zeros_(self._disc_logits.bias)
            def eval_disc(self, values):
                return self._disc_logits(values)
            def get_disc_params(self):
                return self._disc_logits.parameters()
            def get_disc_logit_weights(self):
                return self._disc_logits.weight.flatten()
        class Agent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._device = "cpu"
                self._model = Model()
                self._disc_obs_norm = Normalizer((2,), "cpu", clip=10.)
                self._disc_grad_penalty = .1
                self._disc_logit_reg = .001
                self._disc_optimizer = MPOptimizer(dict(type="SGD", learning_rate=.01),
                                                   list(self._model.get_disc_params()))
        class Demos:
            def fetch_disc_obs_demo(self, count, split="train"):
                return torch.full((count, 2), 2.)
        agent = Agent()
        self.assertIsNone(agent._disc_obs_norm._mean_sq)
        with TemporaryDirectory() as directory:
            report = calibrate(agent, Demos(), {key: [torch.full((128, 2), -2.)]
                for key in ("train", "heldout")}, Path(directory), 16, 7)
        self.assertTrue(report["passed"])
        self.assertTrue(report["actor_and_critic_unchanged"])
        self.assertIsNotNone(agent._disc_obs_norm._mean_sq)

    def test_native_history_excludes_ball_and_preserves_actor_contract(self):
        from .amp_recovery import AMPRecoveryMixin
        from .amp_step import reference_state
        from .balance_observation import task_classes
        from .baseline import sha256
        from .rig import Rig
        import json
        root = Path(__file__).resolve().parents[2]
        source = root / "logs/mimickit-training/recovery-loop-candidate-all-target-30min-01"
        if not source.exists():
            self.skipTest("Selected recovery fixture is absent")
        contract = json.loads((source / "contract.json").read_text())
        _, Base = task_classes(contract)
        class Task(AMPRecoveryMixin, Base):
            pass
        rig = Rig.load(root / "mujoco_rig/dummy.xml")
        demos = [root / "mujoco_rig/mimic/assets" / path for path in
                 ("stand_reference.npz", "steps/forward_reference.npz", "steps/backward_reference.npz")]
        task = Task(rig, root / "logs/mimickit-steps/continuous-stand-02/stationary_reference.npz",
            8, "cpu", demonstrations=demos,
            ball_path=root / "logs/mimickit-training/recovery-loop-support-curriculum-01/mass-8-radius-0.09.xml")
        self.assertEqual(task.get_disc_obs_space().shape, (364,))
        self.assertEqual(task.observation_contract()["num_obs"], 363)
        self.assertEqual(task.observation_contract()["reference_sha256"], contract["reference_sha256"])
        demo = task.fetch_disc_obs_demo(64)
        self.assertEqual(demo.shape, (64, 364))
        self.assertTrue(torch.isfinite(demo).all())
        q = task.engine.native_qpos()
        v = torch.stack([torch.from_numpy(data.qvel[:45].copy()) for data in task.engine.datas]).float()
        torch.testing.assert_close(task._live_disc_state(), reference_state(q, v), rtol=1e-5, atol=1e-6)
        _, _, _, info = task.step(torch.zeros(8, 30))
        torch.testing.assert_close(info["disc_obs"][:, :91], task._live_disc_state())
        self.assertEqual(sha256(rig.path), contract["model_sha256"])


if __name__ == "__main__":
    unittest.main()
