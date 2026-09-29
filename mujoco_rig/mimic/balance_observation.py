"""Versioned instantaneous COM capture feedback; no projectile or timing inputs."""
import torch

from .capture_reward import capture_distance, rotate_xyzw

CONTRACT = "mimic_stationary_capture_target_pd_v2"
SPEC = dict(schema="instantaneous_loaded_capture_v1", sole_radius_m=.06, gravity_m_s2=9.81,
            minimum_foot_load_n=5., minimum_com_height_above_feet_m=.05,
            unsupported_error_m=1., velocity="instantaneous_character_com_world",
            projectile="excluded", ground_height="minimum_foot_origin_height")


class CaptureObservationMixin:
    def observations(self):
        observation = super().observations()
        if not hasattr(self, "_capture_mass"):
            if abs(float(self.rig.model.opt.gravity[2]) + SPEC["gravity_m_s2"]) > 1e-6:
                raise ValueError("Unsupported capture-observation gravity")
            self._capture_mass = torch.tensor(self.rig.model.body_mass[1:], dtype=torch.float32, device=self.device)
            self._capture_ipos = torch.tensor(self.rig.model.body_ipos[1:], dtype=torch.float32, device=self.device)
        positions = self.engine.get_body_pos(0)
        rotation = self.engine.get_body_rot(0)
        centers = positions + rotate_xyzw(rotation, self._capture_ipos)
        com = (centers * self._capture_mass[None, :, None]).sum(1) / self._capture_mass.sum()
        error = capture_distance(com, self.engine.get_center_of_mass_velocity(), positions[:, self.feet],
                                 observation[:, 93:95] * 1000., gravity=float(-self.rig.model.opt.gravity[2]))
        return torch.cat((observation, error[:, None]), -1)

    def observation_contract(self):
        result = super().observation_contract()
        if result["num_obs"] != 362:
            raise ValueError("Capture feedback requires the stationary 362-channel core")
        result.update(observation_contract=CONTRACT, num_obs=363, balance_observation=SPEC)
        result["observation_layout"].append(dict(name="loaded_capture_error_m", offset=362, width=1))
        return result


def task_classes(contract):
    from .continuous_stand_task import ContinuousStandTask
    from .reactive_ball_task import ReactiveBallTask
    if contract.get("observation_contract") != CONTRACT:
        return ContinuousStandTask, ReactiveBallTask

    class CaptureStandTask(CaptureObservationMixin, ContinuousStandTask):
        pass

    class CaptureBallTask(CaptureObservationMixin, ReactiveBallTask):
        pass

    return CaptureStandTask, CaptureBallTask
