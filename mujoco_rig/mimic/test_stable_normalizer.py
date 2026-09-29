"""Near-constant pose coordinates must retain their measured variance."""
import os
from pathlib import Path
import unittest

import torch


class StableMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for the upstream normalizer interface")
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .stable_normalizer import StableNormalizer
        cls.normalizer_type = StableNormalizer

    def test_stationary_height_variance_matches_double_precision(self):
        values = torch.tensor([.90275496, .90283513, .90314209, .90344912,
                               .90352923, .90264136, .90220630, .90177119]).reshape(-1, 1)
        normalizer = self.normalizer_type((1,), "cpu", clip=10.)
        for _ in range(64):
            for _ in range(32):
                normalizer.record(values.repeat(256, 1))
            normalizer.update()
        expected = values.double().std(dim=0, correction=0).float()
        torch.testing.assert_close(normalizer.get_std(), expected, atol=1e-8, rtol=1e-5)
        self.assertLess(float(normalizer.normalize(values).abs().max()), 3.)

    def test_checkpoint_inference_and_resumed_statistics(self):
        torch.manual_seed(17)
        values = torch.randn(100, 3) * .001 + .9
        normalizer = self.normalizer_type((3,), "cpu", clip=10.)
        normalizer.record(values)
        normalizer.update()
        resumed = self.normalizer_type((3,), "cpu", clip=10.)
        resumed.load_state_dict(normalizer.state_dict())
        torch.testing.assert_close(resumed.normalize(values), normalizer.normalize(values), rtol=0, atol=0)
        self.assertEqual(set(normalizer.state_dict()), {"_mean", "_std", "_count"})
        for instance in (normalizer, resumed):
            instance.record(values + .002)
            instance.update()
        torch.testing.assert_close(resumed.get_mean(), normalizer.get_mean(), atol=1e-7, rtol=0)
        torch.testing.assert_close(resumed.get_std(), normalizer.get_std(), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
