"""Keep diagnostic interventions isolated and correctly measured."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch

from .audit_weight_shift import normalization_summary, remove_observation_clip, scaled_reference
from .reference import StandReference
from .rig import Rig


class WeightShiftAuditTests(unittest.TestCase):
    def test_normalization_reports_original_coordinate_indices(self):
        groups = [dict(name="state", offset=0, width=2), dict(name="reference", offset=2, width=2)]
        summary = normalization_summary(np.array([[0., 11., -12., 2.], [0., 1., 0., 3.]]),
                                        np.zeros(4), np.ones(4), groups)
        self.assertEqual(summary[0]["clipped_indices"], [1])
        self.assertEqual(summary[1]["clipped_indices"], [2])
        self.assertEqual(summary[1]["clipped_fraction"], .25)
        with self.assertRaises(ValueError):
            normalization_summary(np.zeros((2, 4)), np.zeros(4), np.zeros(4), groups)

    def test_clip_intervention_preserves_source_and_action_limits(self):
        constants = [numpy_helper.from_array(np.array(value, np.float32), name)
                     for name, value in (("obs_norm._std", 1.), ("lo", -10.), ("hi", 10.),
                                         ("action_lo", -1.), ("action_hi", 1.))]
        nodes = [helper.make_node("Div", ["observation", "obs_norm._std"], ["z"]),
                 helper.make_node("Clip", ["z", "lo", "hi"], ["normalized"]),
                 helper.make_node("Clip", ["normalized", "action_lo", "action_hi"], ["action"])]
        info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [2])
        model = helper.make_model(helper.make_graph(nodes, "test", [info("observation")],
                                                    [info("normalized"), info("action")], constants),
                                  opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        original = model.SerializeToString()
        modified = remove_observation_clip(model)
        self.assertEqual(original, model.SerializeToString())
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(modified.SerializeToString(), options, providers=["CPUExecutionProvider"])
        normalized, actions = session.run(None, {"observation": np.array([-20., 20.], np.float32)})
        np.testing.assert_array_equal(normalized, [-20., 20.])
        np.testing.assert_array_equal(actions, [-1., 1.])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            remove_observation_clip(modified)

    def test_amplitude_intervention_preserves_original_and_quaternions(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        reference = StandReference(Path(__file__).parent / "assets/stand_reference.npz", rig, "cpu")
        source = SimpleNamespace(qpos=reference.qpos[:5].clone(), qvel=reference.qvel[:5].clone(), dt=reference.dt)
        saved = source.qpos.clone()
        hold = scaled_reference(source, rig, 0.)
        np.testing.assert_allclose(hold.qpos, source.qpos[:1].repeat(5, 1), atol=1e-7)
        np.testing.assert_allclose(hold.qvel, 0., atol=1e-7)
        full = scaled_reference(source, rig, 1.)
        np.testing.assert_allclose(full.qpos, source.qpos, atol=1e-7)
        half = scaled_reference(source, rig, .5)
        np.testing.assert_allclose(torch.linalg.vector_norm(half.qpos[:, 3:7], dim=1), 1., atol=1e-7)
        self.assertTrue(torch.equal(source.qpos, saved))


if __name__ == "__main__":
    unittest.main()
