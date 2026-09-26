"""Experimental residual artifact must reload with intact source provenance."""
from pathlib import Path
import tempfile
import unittest

import torch

from .baseline import sha256
from .com_residual_policy import (MIXED_SCHEMA, PILOT_SCHEMA, ComResidualPolicy,
                                  build_residual_network, load_experimental_residual)
from .rig import Rig


class ComResidualPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_round_trip_and_source_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            contract, checkpoint, artifact = (directory / name for name in
                                              ("contract.json", "source.pt", "residual.pt"))
            contract.write_text("source observation contract", encoding="utf-8")
            checkpoint.write_bytes(b"source actor")
            names = [self.rig.model.actuator(int(a)).name for a in self.rig.actuators]
            ankles = [names.index(name) for name in ("Foot_L_rx", "Foot_R_rx")]
            model = build_residual_network(362)
            with torch.no_grad():
                model[-2].weight.zero_()
                model[-2].bias.fill_(.2)
            torch.save(dict(schema=PILOT_SCHEMA, model=model.state_dict(),
                            mean=torch.zeros(362), scale=torch.ones(362),
                            residual_max=.2, ankle_indices=ankles,
                            source_contract_sha256=sha256(contract),
                            source_checkpoint_sha256=sha256(checkpoint)), artifact)
            num_actions = len(self.rig.actuators)

            class ZeroActor(torch.nn.Module):
                def forward(self, obs):
                    return torch.zeros((len(obs), num_actions))

            actor = ZeroActor()
            original = ComResidualPolicy(actor, model, torch.zeros(362),
                                         torch.ones(362), ankles, .2)
            loaded = load_experimental_residual(artifact, actor, self.rig, contract, checkpoint)
            obs = torch.zeros((2, 362))
            torch.testing.assert_close(loaded(obs), original(obs))
            saved = torch.load(artifact, weights_only=True)
            saved["schema"] = MIXED_SCHEMA
            torch.save(saved, artifact)
            mixed = load_experimental_residual(artifact, actor, self.rig, contract, checkpoint)
            torch.testing.assert_close(mixed(obs), original(obs))
            checkpoint.write_bytes(b"different actor")
            with self.assertRaises(ValueError):
                load_experimental_residual(artifact, actor, self.rig, contract, checkpoint)


if __name__ == "__main__":
    unittest.main()
