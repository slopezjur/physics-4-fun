"""COM reconstruction must use only fields present in the Stand observation."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from .observation_com import ObservationCom
from .rig import Rig


class ObservationComTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        names = ("root_position_world", "root_rotation_tangent_normal",
                 "root_linear_velocity_world", "root_angular_velocity_world",
                 "all_hinge_positions", "all_hinge_velocities_scaled_0.1")
        widths = (3, 6, 3, 3, 39, 39)
        offset = 0
        layout = []
        for name, width in zip(names, widths):
            layout.append(dict(name=name, offset=offset, width=width))
            offset += width
        cls.contract = dict(observation_contract="mimic_stand_target_pd_v2",
                            model_sha256=cls.rig.contract()["model_sha256"],
                            num_obs=362, observation_layout=layout)

    def test_reconstruction_matches_native_nonzero_velocity_and_rotation(self):
        data = mujoco.MjData(self.rig.model)
        data.qpos[:] = self.rig.rest
        data.qpos[0] += .075
        data.qpos[3:7] = [np.cos(.15), 0., np.sin(.15), 0.]
        data.qpos[9] += .12
        data.qvel[:6] = [.13, -.03, .02, .1, -.08, .05]
        data.qvel[7] = -.22
        mujoco.mj_forward(self.rig.model, data)
        mujoco.mj_subtreeVel(self.rig.model, data)
        rotation = data.xmat[1].reshape(3, 3)
        observation = np.zeros((1, 362))
        observation[0, :3] = data.qpos[:3]
        observation[0, 3:6] = rotation[:, 0]
        observation[0, 6:9] = rotation[:, 2]
        observation[0, 9:12] = data.qvel[:3]
        observation[0, 12:15] = rotation @ data.qvel[3:6]
        observation[0, 15:54] = data.qpos[7:]
        observation[0, 54:93] = data.qvel[6:] * .1
        measured = ObservationCom(self.rig, self.contract)(observation)[0]
        np.testing.assert_allclose(measured, [data.subtree_com[1, 0],
                                               data.subtree_linvel[1, 0]], atol=1e-7)

    def test_legacy_observation_is_rejected(self):
        legacy = dict(self.contract, observation_contract="mimic_stand_target_pd_v1")
        with self.assertRaises(ValueError):
            ObservationCom(self.rig, legacy)


if __name__ == "__main__":
    unittest.main()
