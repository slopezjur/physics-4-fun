"""Randomized test coverage and schedule isolation from the training task."""
from collections import Counter
import copy
import os
from pathlib import Path
import unittest

import torch

from .ball_generalization import SCHEMA, make_cases, make_task, validate_protocol
from .baseline import sha256
from .ball_protocol import TARGET_BODIES


class BallGeneralizationTests(unittest.TestCase):
    def protocol(self, seed=230923):
        return dict(schema=SCHEMA, control_timestep=.016668, reference_duration=5.966786,
                    cases=make_cases(seed, 120, .016668, 5.966786))

    def test_frozen_seed_balances_bodies_and_varies_continuous_parameters(self):
        cases = self.protocol()["cases"]
        self.assertEqual(cases, self.protocol()["cases"])
        self.assertNotEqual(cases, self.protocol(230924)["cases"])
        self.assertEqual(Counter(c["target_body"] for c in cases), {body: 10 for body in TARGET_BODIES})
        for key in ("angle_radians", "speed", "phase"):
            self.assertEqual(len({c[key] for c in cases}), 120)
        self.assertGreater(len({c["launch_step"] for c in cases}), 30)
        validate_protocol(self.protocol())

    def test_rejects_duplicate_names_and_reference_overrun(self):
        for change in (lambda p: p["cases"].append(copy.deepcopy(p["cases"][0])),
                       lambda p: p["cases"][0].update(phase=2.)):
            protocol = self.protocol()
            change(protocol)
            with self.assertRaises(ValueError):
                validate_protocol(protocol)
        with self.assertRaises(ValueError):
            make_cases(1, 121, .016668, 5.966786)

    def test_native_resets_apply_frozen_cases_and_preserve_peers(self):
        checkout = os.environ.get("MIMICKIT_PATH")
        if not checkout:
            self.skipTest("Set MIMICKIT_PATH for native integration")
        from .runtime import activate
        activate(Path(checkout))
        from envs.base_env import EnvMode
        from .rig import Rig
        from .ball import load_ball_rig
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        protocol = self.protocol()
        protocol.update(model_sha256=sha256(rig.path), world_model_sha256=sha256(load_ball_rig(rig).path),
                        reference_sha256=sha256(root / "assets/stand_reference.npz"))
        task = make_task(rig, root / "assets/stand_reference.npz", protocol)
        task.set_mode(EnvMode.TEST)
        obs, _ = task.reset()
        torch.testing.assert_close(obs, task.observations())
        for i, case in enumerate(protocol["cases"]):
            self.assertAlmostEqual(float(task.offset[i]), case["phase"], places=6)
            self.assertAlmostEqual(float(task.shot_speeds[i]), case["speed"], places=6)
            self.assertEqual(int(task.shot_steps[i]), case["launch_step"])
            self.assertEqual(int(task.shot_target_body[i]), rig.model.body(case["target_body"]).id - 1)
            angle = torch.tensor(case["angle_radians"])
            torch.testing.assert_close(task.directions[i], torch.tensor([angle.cos(), angle.sin(), 0.]))
        task.steps[1] = 17
        task.engine.hit[1] = True
        peer = task.observations()[1].clone()
        task.reset(torch.tensor([0, 119]))
        torch.testing.assert_close(task.observations()[1], peer)
        self.assertEqual(int(task.steps[1]), 17)
        self.assertTrue(bool(task.engine.hit[1]))


if __name__ == "__main__":
    unittest.main()
