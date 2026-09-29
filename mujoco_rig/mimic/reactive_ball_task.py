"""Stationary standing with physical ball hits and either-foot recovery."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from envs.base_env import EnvMode

from .ball import launch_state
from .ball_native import BallNativeEngine
from .ball_newton import BallNewtonEngine
from .ball_protocol import TARGET_BODIES as ALL_TARGET_BODIES
from .continuous_stand_task import ContinuousStandTask
from .recovery_reward import RecoveryReward
from .capture_reward import CaptureRecoveryReward, capture_distance, rotate_xyzw
from .step_reward import StepRecoveryReward, TransferRecoveryReward, swing_quality
from .perturb import RECOVERY
from .stand_foundation_task import FootAnchoredStandTask


TARGET_BODIES = ("Chest", "Spine", "Pelvis")
TARGET_SETS = {"torso": TARGET_BODIES, "torso_head": TARGET_BODIES + ("Head",),
               "all": ALL_TARGET_BODIES}


def write_ball_variant(source: Path, out: Path, mass_kg: float, *, radius_m=None):
    """Change only projectile mass/optional sphere radius in an isolated plant copy."""
    if not np.isfinite(mass_kg) or not .1 <= mass_kg <= 8.:
        raise ValueError("Projectile mass must be within 0.1–8 kg")
    if radius_m is not None and (not np.isfinite(radius_m) or not .04 <= radius_m <= .2):
        raise ValueError("Projectile radius must be within 0.04–0.2 m")
    if source.resolve() == out.resolve():
        raise ValueError("Projectile variants require a distinct output path")
    tree = ET.parse(source)
    ball = tree.find("./worldbody/body[@name='ball']/geom[@name='g_ball']")
    if ball is None:
        raise ValueError("Projectile geom is missing")
    ball.set("mass", format(mass_kg, ".9g"))
    if radius_m is not None:
        if ball.get("type") != "sphere":
            raise ValueError("Radius variation requires a spherical projectile")
        ball.set("size", format(radius_m, ".9g"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="unicode")
    return out


class ReactiveBallTask(FootAnchoredStandTask):
    def __init__(self, rig, reference, num_envs=8, device="cpu", *, ball_path,
                 speed_min=1., speed_max=2., quiet_fraction=.25, seed=270927,
                 foot_anchor_sigma_m=.005, episode_seconds=8., second_shot_fraction=0.,
                 recovery_mode="settling_v2", forward_fraction=0., recovery_reset_fraction=0.,
                 target_set="torso"):
        if target_set not in TARGET_SETS:
            raise ValueError("Unknown projectile target curriculum")
        if not (1. <= speed_min <= speed_max <= 8. and 0. <= quiet_fraction < 1.
                and 5. <= episode_seconds <= 30. and 0. <= second_shot_fraction <= 1.
                and (second_shot_fraction == 0. or episode_seconds >= 11.)
                and 0. <= forward_fraction <= 1. and 0. <= recovery_reset_fraction <= 1.):
            raise ValueError("Invalid stationary ball curriculum")
        self.ball_path = Path(ball_path)
        self.speed_min, self.speed_max = float(speed_min), float(speed_max)
        self.quiet_fraction, self.ball_episode_seconds = float(quiet_fraction), float(episode_seconds)
        self.second_shot_fraction = float(second_shot_fraction)
        self.forward_fraction = float(forward_fraction)
        self.recovery_reset_fraction = float(recovery_reset_fraction)
        self.recovery_resets = None
        if recovery_mode not in ("settling_v2", "capture_v3", "step_v4", "step_v5"):
            raise ValueError("Unknown recovery objective")
        self.recovery_mode = recovery_mode
        self.recovery_reward = (TransferRecoveryReward() if recovery_mode == "step_v5" else
                                StepRecoveryReward() if recovery_mode == "step_v4" else
                                CaptureRecoveryReward() if recovery_mode == "capture_v3" else
                                RecoveryReward(sustained_settling_weight=.25))
        self.body_mass = torch.tensor(rig.model.body_mass[1:], device=device, dtype=torch.float32)
        self.body_ipos = torch.tensor(rig.model.body_ipos[1:], device=device, dtype=torch.float32)
        self.previous_com = None
        self.target_set = target_set
        self.target_bodies = TARGET_SETS[target_set]
        self.target_indices = None

        def engine_factory(character, worlds, where, control_factory):
            engine = BallNativeEngine if where == "cpu" else BallNewtonEngine
            return engine(character, worlds, where, control_factory, self.ball_path)

        super().__init__(rig, reference, num_envs, device, seed=seed,
                         engine_factory=engine_factory, foot_anchor_sigma_m=foot_anchor_sigma_m)
        self.episode_seconds = self.ball_episode_seconds
        self.target_indices = torch.tensor([rig.model.body(name).id - 1 for name in self.target_bodies],
                                           device=device, dtype=torch.long)
        self.shot_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.shot_speeds = torch.zeros(num_envs, device=device)
        self.directions = torch.zeros((num_envs, 3), device=device)
        self.shot_target_body = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.shot_enabled = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.second_shot_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.second_shot_speeds = torch.zeros(num_envs, device=device)
        self.second_directions = torch.zeros((num_envs, 3), device=device)
        self.second_shot_target_body = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.second_shot_enabled = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.settled_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        reference_q, _ = self.sample_reference(self.offset)
        reference_positions, _ = self.mapper.from_native_qpos(reference_q)
        self.sole_reference_height = reference_positions[:, self.allowed_ground][:, [0, 2, 1, 3], 2].reshape(num_envs, 2, 2)
        self.reset()

    def load_recovery_resets(self, path):
        from .baseline import sha256
        with np.load(path, allow_pickle=False) as bank:
            for key, expected in (("schema", "physical_post_hit_resets_v1"),
                                  ("model_sha256", sha256(self.rig.path)),
                                  ("reference_sha256", self.reference.sha256),
                                  ("ball_sha256", sha256(self.ball_path))):
                if str(bank[key]) != expected:
                    raise ValueError("Recovery reset bank mismatch: " + key)
            count = len(bank["posture"])
            shapes = dict(qpos=(count, 46), qvel=(count, 45), ground=(count, 18, 3),
                          pending=(count, len(self.engine.policy_control.pending), 61),
                          control_active=(count, 61))
            if any(bank[key].shape != shape or not np.isfinite(bank[key]).all() for key, shape in shapes.items()):
                raise ValueError("Malformed recovery reset state")
            if set(bank["posture"].tolist()) != set(range(8)):
                raise ValueError("Recovery resets must cover all eight postures")
            self.recovery_resets = {key: torch.as_tensor(bank[key].copy(), dtype=torch.float32,
                                                         device=self.device) for key in shapes}
            self.recovery_groups = [torch.as_tensor(np.flatnonzero(bank["posture"] == i), device=self.device)
                                    for i in range(8)]
            weights = np.ones(count)
            if "direction" in bank:
                directions = bank["direction"]
                if directions.shape != (count,) or not np.isin(directions, range(4)).all():
                    raise ValueError("Invalid recovery reset direction strata")
                forces = np.abs(bank["ground"][:, self.allowed_ground.cpu().numpy(), 2])
                single = np.minimum(forces[:, 0] + forces[:, 2], forces[:, 1] + forces[:, 3]) < 5.
                strata = bank["posture"] * 8 + directions * 2 + single
                for stratum in np.unique(strata):
                    indices = strata == stratum
                    weights[indices] = 1. / indices.sum()
            self.recovery_weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

    def _restore_recovery_resets(self, ids):
        bank = self.recovery_resets
        if bank is None or not self.recovery_reset_fraction:
            return
        selected = ids[self.shot_enabled[ids] & (torch.rand(len(ids), generator=self.rng,
                       device=self.device) < self.recovery_reset_fraction)]
        world_groups, row_groups = [], []
        for posture, group in enumerate(self.recovery_groups):
            worlds = selected[selected % 8 == posture]
            if not len(worlds):
                continue
            rows = group[torch.multinomial(self.recovery_weights[group], len(worlds), replacement=True, generator=self.rng)]
            world_groups.append(worlds)
            row_groups.append(rows)
        if not world_groups:
            return
        worlds, rows = torch.cat(world_groups), torch.cat(row_groups)
        self.engine.reset_envs(worlds, bank["qpos"][rows], bank["qvel"][rows])
        control = self.engine.policy_control
        control.active[worlds] = bank["control_active"][rows]
        for slot in range(len(control.pending)):
            control.pending[(control.cursor + slot) % len(control.pending), worlds] = bank["pending"][rows, slot]
        if hasattr(self.engine, "_ground_force_cache"):
            self.engine._ground_force_cache[worlds.cpu().numpy(), :18] = bank["ground"][rows].cpu().numpy()
        else:
            self.engine.get_ground_contact_forces(0)[worlds] = bank["ground"][rows]
        # Each saved state follows a measured hit with the projectile separated.
        # Park that spent projectile; subsequent shots still use physical flight.
        self.engine.hit[worlds] = True
        self.shot_steps[worlds] = -1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.target_indices is None:
            return self.observations(), {}
        ids = self.ids if env_ids is None else env_ids
        if not len(ids):
            return self.observations(), {}
        self.settled_steps[ids] = 0
        if self._mode == EnvMode.TRAIN:
            self.shot_steps[ids] = torch.randint(60, 121, (len(ids),), generator=self.rng, device=self.device)
            self.shot_speeds[ids] = self.speed_min + (self.speed_max - self.speed_min) * torch.rand(
                len(ids), generator=self.rng, device=self.device)
            angle = 2 * math.pi * torch.rand(len(ids), generator=self.rng, device=self.device)
            if self.forward_fraction:
                focused = torch.rand(len(ids), generator=self.rng, device=self.device) < self.forward_fraction
                jitter = (torch.rand(len(ids), generator=self.rng, device=self.device) * 2 - 1) * math.pi / 9
                angle = torch.where(focused, jitter, angle)
            target = torch.randint(0, len(self.target_bodies), (len(ids),), generator=self.rng, device=self.device)
            self.shot_enabled[ids] = torch.rand(len(ids), generator=self.rng,
                                                device=self.device) >= self.quiet_fraction
            self.second_shot_enabled[ids] = self.shot_enabled[ids] & (
                torch.rand(len(ids), generator=self.rng, device=self.device) < self.second_shot_fraction)
            self.second_shot_steps[ids] = torch.randint(360, 421, (len(ids),), generator=self.rng,
                                                       device=self.device)
            self.second_shot_speeds[ids] = self.speed_min + (self.speed_max - self.speed_min) * torch.rand(
                len(ids), generator=self.rng, device=self.device)
            second_angle = 2 * math.pi * torch.rand(len(ids), generator=self.rng, device=self.device)
            second_target = torch.randint(0, len(self.target_bodies), (len(ids),), generator=self.rng,
                                          device=self.device)
        else:
            self.shot_steps[ids] = 90
            self.shot_speeds[ids] = self.speed_max
            angle = ((ids // 8) % 4) * (math.pi / 2)
            target = (ids // 32) % len(self.target_bodies)
            self.shot_enabled[ids] = True
            self.second_shot_enabled[ids] = False
        self.shot_target_body[ids] = self.target_indices[target]
        self.directions[ids] = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), -1)
        if self._mode == EnvMode.TRAIN:
            self.second_shot_target_body[ids] = self.target_indices[second_target]
            self.second_directions[ids] = torch.stack((second_angle.cos(), second_angle.sin(),
                                                        torch.zeros_like(second_angle)), -1)
            self._restore_recovery_resets(ids)
        if self.previous_com is not None:
            self.previous_com[ids] = self.center_of_mass()[ids]
        return self.observations(), {}

    def step(self, action):
        if self.recovery_mode in ("capture_v3", "step_v4", "step_v5"):
            self.previous_com = self.center_of_mass()
        for steps, enabled, bodies, directions, speeds in (
                (self.shot_steps, self.shot_enabled, self.shot_target_body,
                 self.directions, self.shot_speeds),
                (self.second_shot_steps, self.second_shot_enabled, self.second_shot_target_body,
                 self.second_directions, self.second_shot_speeds)):
            ids = torch.nonzero((self.steps == steps) & enabled).flatten()
            if len(ids):
                target = self.engine.get_body_pos(0)[ids, bodies[ids]].cpu().numpy()
                position, velocity = launch_state(target, directions[ids].cpu().numpy(),
                                                  speeds[ids].cpu().numpy(),
                                                  self.rig.model.opt.gravity)
                self.engine.launch_ball(ids, position, velocity)
        transition = super().step(action)
        self.last_transition_reward = transition[1]
        return transition

    def center_of_mass(self):
        count = len(self.body_mass)
        positions = self.engine.get_body_pos(0)[:, :count]
        rotation = self.engine.get_body_rot(0)[:, :count]
        centers = positions + rotate_xyzw(rotation, self.body_ipos)
        return (centers * self.body_mass[None, :, None]).sum(1) / self.body_mass.sum()

    def reward(self):
        quiet = super().reward()
        q = self.engine.native_qpos()
        target, _ = self.sample_reference(self.offset + self.steps * self.dt)
        loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
        feet = torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1)
        up_z = 1 - 2 * (q[:, 4].square() + q[:, 5].square())
        velocity = self.engine.get_root_vel(0)
        angular_velocity = self.engine.get_root_ang_vel(0)
        settled = ((q[:, 2] >= RECOVERY["height_m"])
                   & (up_z >= math.cos(math.radians(RECOVERY["tilt_degrees"])))
                   & (feet.amin(-1) >= RECOVERY["each_foot_load_n"])
                   & (torch.linalg.vector_norm(velocity[:, :2], dim=-1) <= RECOVERY["horizontal_speed_m_s"])
                   & (torch.linalg.vector_norm(angular_velocity, dim=-1) <= RECOVERY["angular_speed_rad_s"])
                   & self.engine.hit & self.shot_enabled)
        self.settled_steps = torch.where(settled, self.settled_steps + 1, 0)
        imitation = self.reference_reward(track_root=False)
        if self.recovery_mode in ("capture_v3", "step_v4", "step_v5"):
            com = self.center_of_mass()
            com_velocity = (com - self.previous_com) / self.dt if self.previous_com is not None else velocity
            error = capture_distance(com, com_velocity, self.engine.get_body_pos(0)[:, self.feet], feet,
                                     gravity=float(-self.rig.model.opt.gravity[2]))
            reward_args = (imitation, q[:, 2], target[:, 2], up_z, velocity, angular_velocity, error)
            if self.recovery_mode in ("step_v4", "step_v5"):
                points = self.engine.get_body_pos(0)[:, self.allowed_ground][:, [0, 2, 1, 3]].reshape(-1, 2, 2, 3)
                swing = swing_quality(com, com_velocity, points, self.sole_reference_height, feet,
                                      gravity=float(-self.rig.model.opt.gravity[2]),
                                      preparation_weight=.3 if self.recovery_mode == "step_v5" else 0.)
                recovered = (self.recovery_reward(*reward_args, swing, feet) if self.recovery_mode == "step_v5"
                             else self.recovery_reward(*reward_args, swing))
                self._diagnostics["swing_quality"] = swing.mean()
            else:
                recovered = self.recovery_reward(*reward_args)
            self._diagnostics["capture_error_m"] = error.mean()
        else:
            recovered = self.recovery_reward(imitation, q[:, 2], target[:, 2],
                                             up_z, velocity, angular_velocity, feet, self.settled_steps * self.dt)
        return torch.where(self.engine.hit & self.shot_enabled, recovered, quiet)

    def experiment_contract(self):
        from .baseline import sha256
        world = self.engine.rig.model
        return dict(schema="mimic_reactive_stationary_ball_seed_v1",
                    reference_sha256=self.reference.sha256, ball_model_sha256=sha256(self.ball_path),
                    ball_mass_kg=float(world.body_mass[world.body("ball").id]),
                    speed_min=self.speed_min, speed_max=self.speed_max,
                    quiet_fraction=self.quiet_fraction,
                    episode_seconds=self.episode_seconds, target_bodies=self.target_bodies,
                    target_set=self.target_set,
                    angles="uniform 0..2pi", launch_steps=[60, 120],
                    forward_fraction=self.forward_fraction, forward_jitter_degrees=20.,
                    second_shot_fraction=self.second_shot_fraction,
                    second_launch_steps=[360, 420],
                    shot_observation="none; body feedback only", post_hit_foot="either foot may move",
                    quiet_foot_anchor_sigma_m=self.foot_anchor_sigma_m,
                    post_hit_reward=self.recovery_reward.contract())
