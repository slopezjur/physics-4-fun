"""New motion channels must preserve initialization and remain trainable."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .motion_features import MotionFeatureNormalizer, install, validate_lineage
from .prepare_motion_normalization import migrate, validate_preparation, SCHEMA
from .baseline import sha256


class MotionFeatureTests(unittest.TestCase):
    def test_inherited_feature_layout_cannot_be_silently_changed(self):
        contract = dict(observation_layout=[dict(name="future_reference_poses_1_2_3", offset=3, width=2)])
        lineage = dict(feature_schema="mimic_dual_motion_features_v1", migration_mode="additive_features",
                       motion_feature_indices=[3, 4])
        validate_lineage(lineage, contract)
        for field, value in (("feature_schema", "unknown"), ("motion_feature_indices", [4, 3]),
                             ("migration_mode", "calibration_only")):
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_lineage({**lineage, field: value}, contract)

    def test_preparation_is_bound_to_checkpoint_reference_and_measured_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            checkpoint, contract_path, manifest = out / "candidate.pt", out / "contract.json", out / "preparation.json"
            checkpoint.write_bytes(b"candidate")
            metrics = dict(episodes=8, successes=8, mean_root_tracking_error_m=.02, mean_survival_seconds=3.)
            rows = [dict(seconds=seconds, metrics=copy.deepcopy(metrics), motion_input_clipped_fraction=0.)
                    for seconds in (3., 5., 3.)]
            report = dict(schema=SCHEMA, ready_for_training=True, checkpoint_sha256=sha256(checkpoint),
                actor_sha256="actor", calibration_reference_sha256=["quiet", "small"],
                validation=rows, initial_native=[metrics, metrics], migration_mode="additive_features",
                calibration_max_action_difference=0., motion_feature_indices=[1, 2])
            def save():
                manifest.write_text(json.dumps(report))
                contract_path.write_text(json.dumps(dict(actor_sha256="actor",
                    observation_layout=[dict(name="future_reference_poses_1_2_3", offset=1, width=2)],
                    normalization_preparation=dict(schema=SCHEMA, manifest=manifest.name, sha256=sha256(manifest)))))
            save()
            self.assertEqual(validate_preparation(contract_path, checkpoint, "small")["feature_schema"], "mimic_dual_motion_features_v1")
            with self.assertRaisesRegex(ValueError, "not admitted"):
                validate_preparation(contract_path, checkpoint, "different")
            rows[0]["metrics"]["successes"] = 7
            save()
            with self.assertRaisesRegex(ValueError, "retention"):
                validate_preparation(contract_path, checkpoint, "small")
            rows[0]["metrics"]["successes"] = 8
            save()
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "not admitted"):
                validate_preparation(contract_path, checkpoint, "small")

    def test_expanded_initialization_preserves_clipped_policy_and_gets_gradients(self):
        torch.manual_seed(7)
        normalizer = SimpleNamespace(_mean=torch.zeros(3), _std=torch.ones(3), _count=torch.tensor([10]))
        model = SimpleNamespace(_actor_layers=torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh()),
                                _critic_layers=torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh()))
        actor, critic = copy.deepcopy(model._actor_layers), copy.deepcopy(model._critic_layers)
        agent = SimpleNamespace(_obs_norm=normalizer, _model=model)
        install(agent, [1, 2])
        agent._obs_norm._motion_std[:] = torch.tensor([1., 10., 10.])
        inputs = torch.tensor([[.2, 20., -30.], [.2, 40., -50.]])
        features = agent._obs_norm.normalize(inputs)
        self.assertTrue(torch.equal(features[0, :3], features[1, :3]))
        self.assertFalse(torch.equal(features[0, 3:], features[1, 3:]))
        torch.testing.assert_close(model._actor_layers(features), actor(inputs.clamp(-10, 10)), rtol=0, atol=1e-7)
        torch.testing.assert_close(model._critic_layers(features), critic(inputs.clamp(-10, 10)), rtol=0, atol=1e-7)
        model._actor_layers(features).sum().backward()
        self.assertGreater(float(model._actor_layers[0].weight.grad[:, 3:].abs().max()), 0.)
        self.assertEqual(agent._obs_norm.get_count().item(), 10)
        self.assertTrue(torch.equal(agent._obs_norm._std, torch.ones(3)))

    def test_expanded_normalization_state_roundtrips(self):
        base = SimpleNamespace(_mean=torch.zeros(3), _std=torch.ones(3), _count=torch.tensor([10]))
        source, target = MotionFeatureNormalizer(base, [0, 2]), MotionFeatureNormalizer(base, [0, 2])
        source._motion_std[:] *= 5
        target.load_state_dict(source.state_dict(), strict=True)
        values = torch.tensor([[30., 1., -20.]])
        torch.testing.assert_close(source.normalize(values), target.normalize(values))

    def test_compensated_diagnostic_changes_only_selected_scales_and_input_weights(self):
        state = {"_obs_norm._mean": torch.tensor([.1, -.2]), "_obs_norm._std": torch.ones(2),
                 "_model._actor_layers.0.weight": torch.tensor([[2., 3.]]),
                 "_model._critic_layers.0.weight": torch.tensor([[4., 5.]])}
        saved = copy.deepcopy(state)
        result = migrate(state, torch.tensor([[0., 100.]]), torch.tensor([1]))
        for key in state:
            self.assertTrue(torch.equal(state[key], saved[key]))
        self.assertEqual(result["_obs_norm._std"][0], 1.)
        raw = torch.tensor([[.2, .3]])
        old = (raw - state["_obs_norm._mean"]) / state["_obs_norm._std"]
        new = (raw - state["_obs_norm._mean"]) / result["_obs_norm._std"]
        for branch in ("actor", "critic"):
            key = f"_model._{branch}_layers.0.weight"
            torch.testing.assert_close(old @ state[key].T, new @ result[key].T)


if __name__ == "__main__":
    unittest.main()
