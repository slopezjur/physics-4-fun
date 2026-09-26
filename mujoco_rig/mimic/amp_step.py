"""AMP motion observations for the existing MuJoCo/Newton dummy contract."""
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
import torch
from util.torch_util import quat_rotate, quat_to_tan_norm

from .reference import StandReference
from .stand_task import StandTask
from .track_transfer import CalibrationReference


DISC_CONTRACT = "mimic_amp_state_pair_v1"


def discriminator_state(qpos, root_vel, root_ang_vel, dof_vel):
    """Root-translation-invariant state; angular velocity is in world coordinates."""
    return torch.cat((qpos[..., 2:3], quat_to_tan_norm(qpos[..., [4, 5, 6, 3]]),
                      qpos[..., 7:], root_vel, root_ang_vel, dof_vel), dim=-1)


def reference_state(qpos, qvel):
    root_rot = qpos[..., [4, 5, 6, 3]]
    return discriminator_state(qpos, qvel[..., :3],
                               quat_rotate(root_rot, qvel[..., 3:6]), qvel[..., 6:])


def load_demonstrations(paths, rig, device):
    if not paths:
        raise ValueError("AMP requires at least one licensed motion demonstration")
    references = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            ref = CalibrationReference(path, rig, device)
            if not ref.placed_step:
                raise ValueError("Only validated placed-step diagnostics may be AMP demos")
        else:
            ref = StandReference(path, rig, device)
        references.append(ref)
    return references


class AMPStepTask(StandTask):
    """Keep the actor's support_v2 observation; supply AMP with motion pairs."""
    def __init__(self, rig, reference, demonstrations, num_envs=128, device="cuda:0",
                 seed=210921, engine_factory=None):
        self.demonstrations = load_demonstrations(demonstrations, rig, device)
        # Pose(1+6+all hinge qpos) and velocity(3+3+all hinge qvel), twice.
        state_width = 7 + rig.model.nq - 7 + rig.model.nv
        self._disc_obs_space = spaces.Box(-np.inf, np.inf, (2 * state_width,), dtype=np.float32)
        self._demo_rng = torch.Generator(device=device).manual_seed(seed + 1907)
        self._body_mass = torch.as_tensor(rig.model.body_mass[1:], dtype=torch.float32, device=device)
        options = dict(seed=seed, control_mode="target_pd", contact_mode="support_v2",
                       reference_factory=CalibrationReference)
        if engine_factory is not None:
            options["engine_factory"] = engine_factory
        super().__init__(rig, reference, num_envs, device, **options)
        if not self.reference.placed_step:
            raise ValueError("AMP step task requires a validated placed-step reference")
        self.episode_seconds = 8.0

    def get_disc_obs_space(self):
        return self._disc_obs_space

    def reward(self):
        """Support transfer and step placement, leaving whole-body style to AMP."""
        target_q, _ = self.sample_reference(self.offset + self.steps * self.dt)
        target_pos, _ = self.mapper.from_native_qpos(target_q)
        body_pos = self.engine.get_body_pos(0)
        root = self.engine.native_qpos()
        upright = 1. - 2. * (root[:, 4].square() + root[:, 5].square())
        height = torch.exp(-10. * (root[:, 2] - target_q[:, 2]).square())
        attitude = torch.exp(-5. * (1. - upright).square())
        foot_distance = torch.linalg.vector_norm(body_pos[:, self.feet] - target_pos[:, self.feet], dim=-1)
        swing_goal = torch.exp(-10. * foot_distance[:, 0])
        support_goal = torch.exp(-15. * foot_distance[:, 1])
        mass = self._body_mass[None, :, None]
        actual_com = (body_pos * mass).sum(dim=1) / self._body_mass.sum()
        target_com = (target_pos * mass).sum(dim=1) / self._body_mass.sum()
        com_error = torch.linalg.vector_norm(actual_com[:, :2] - target_com[:, :2], dim=-1)
        transfer_goal = torch.exp(-com_error / .05)
        root_speed = torch.linalg.vector_norm(self.engine.get_root_vel(0)[:, :2], dim=-1)
        phase = self.offset + self.steps * self.dt
        settling = torch.where(phase >= 15., torch.exp(-4. * root_speed), 1.)
        self._diagnostics["step_foot_error_m"] = foot_distance[:, 0].mean()
        self._diagnostics["support_com_error_m"] = com_error.mean()
        return .2 * height + .15 * attitude + .25 * transfer_goal + .2 * swing_goal + .1 * support_goal + .1 * settling

    def _live_disc_state(self):
        engine = self.engine
        return discriminator_state(engine.native_qpos(), engine.get_root_vel(0),
                                   engine.get_root_ang_vel(0), engine.get_dof_vel(0))

    def step(self, action):
        previous = self._live_disc_state()
        obs, reward, done, info = super().step(action)
        pair = torch.cat((self._live_disc_state(), previous), dim=-1)
        if pair.shape[1:] != self._disc_obs_space.shape:
            raise RuntimeError("AMP discriminator contract changed")
        info["disc_obs"] = torch.nan_to_num(pair, nan=0., posinf=0., neginf=0.).clamp(-1e4, 1e4)
        return obs, reward, done, info

    def fetch_disc_obs_demo(self, num_samples):
        if num_samples < 1:
            raise ValueError("Expected a positive demonstration batch")
        selection = torch.randint(len(self.demonstrations), (num_samples,),
                                  generator=self._demo_rng, device=self.device)
        result = torch.empty((num_samples, *self._disc_obs_space.shape),
                             device=self.device, dtype=torch.float32)
        for index, ref in enumerate(self.demonstrations):
            chosen = torch.nonzero(selection == index).flatten()
            if not len(chosen):
                continue
            time = self.dt + torch.rand(len(chosen), generator=self._demo_rng,
                                        device=self.device) * (ref.duration - self.dt)
            q, v = ref.sample(time)
            prev_q, prev_v = ref.sample(time - self.dt)
            result[chosen] = torch.cat((reference_state(q, v), reference_state(prev_q, prev_v)), dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite AMP motion demonstration")
        return result

    def discriminator_contract(self):
        return dict(schema=DISC_CONTRACT, width=self._disc_obs_space.shape[0],
                    sampling="current post-step state followed by prior pre-step state, one control dt apart",
                    features=("pelvis_height", "root_rotation_tangent_normal", "all_hinge_positions",
                              "root_linear_velocity_world", "root_angular_velocity_world",
                              "all_hinge_velocities"),
                    demonstrations=[dict(path=str(ref.path), sha256=ref.sha256)
                                    for ref in self.demonstrations])
