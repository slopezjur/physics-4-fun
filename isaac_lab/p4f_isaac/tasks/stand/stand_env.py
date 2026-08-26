"""Stand: hold an upright, settled pose from a standing start.

The Godot equivalent is `UprightProgressReward` + `UprightTermination`, and the success criteria
are ported verbatim. The reward itself is not a line-for-line port, and the difference is
deliberate. That reward is built around potential-based shaping on head height because its hard
case is the *get-up*, where the body starts prone and needs a gradient that exists from the floor.
Starting from a standing pose, the dominant problem is instead staying up and staying still, so
the terms here are the ones that measure that directly. The get-up task will want the shaping
potential back, which is why the criteria constants live in the config, shared.

One more deviation, also deliberate: success does not end the episode. Godot's `UprightTermination`
absorbs on success to distinguish "stood up" from "stood up then fell at t=7.9 s", and pays
`StandingBonus` calibrated to beat the forfeited per-tick reward. Here every episode runs its full
window and the per-step upright term does the scoring, which expresses "stayed up for eight
seconds" without the break-even arithmetic - the same reasoning `UprightTermination` gives for
turning absorption off in the perturbation task.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from p4f_isaac.assets import ACTUATED_JOINTS, CONTACT_BONES, REST_PELVIS_HEIGHT

from .stand_env_cfg import (
    FALL_HEAD_HEIGHT,
    FALL_TILT_DEG,
    REST_HEAD_HEIGHT,
    StandEnvCfg,
)


class StandEnv(DirectRLEnv):
    cfg: StandEnvCfg

    def __init__(self, cfg: StandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve the frozen action order against whatever order PhysX ended up giving the DOFs.
        # preserve_order is the whole point: without it the ids come back sorted and every action
        # lands on a different joint than the contract says, which trains perfectly happily and
        # produces a policy that is nonsense the moment Godot applies it in contract order.
        self._actuated_ids, _ = self.robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        self._actuated_ids = torch.tensor(self._actuated_ids, device=self.device, dtype=torch.long)

        limits = self.robot.data.soft_joint_pos_limits[0]  # (num_joints, 2), identical per env
        self._act_lower = limits[self._actuated_ids, 0]
        self._act_upper = limits[self._actuated_ids, 1]
        self._act_default = self.robot.data.default_joint_pos[0, self._actuated_ids]

        self._head_id, _ = self.robot.find_bodies("Head")
        self._contact_ids, _ = self.contact_sensor.find_bodies(CONTACT_BONES, preserve_order=True)

        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._action = torch.zeros_like(self._previous_action)
        self._raw_action = torch.zeros_like(self._previous_action)

        # Velocity command (vx, vy, yaw_rate), constant zero for Stand and reserved for Walk.
        #
        # Three wasted floats now, and they buy the only thing that makes Walk cheap: Walk is
        # bootstrapped from a Stand checkpoint, which is possible only while the observation widths
        # match. A command has to reach the input layer - there is no other way to express "go that
        # way" to a trained network - so the slots must exist in the shared observation BEFORE
        # walking uses them, or adding them later orphans every Stand checkpoint. This is the same
        # reasoning BodyStateObservation gives for its reserved joystick block, and the reason that
        # block's width is documented there as frozen.
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        self._effort_limit = self.robot.data.joint_effort_limits[0, self._actuated_ids].clamp(min=1e-6)

        self._fall_cos = math.cos(math.radians(FALL_TILT_DEG))
        # _get_dones runs before _get_rewards each step and fills this in, but the buffer has to
        # exist first: a reset can call into the reward path before any done has been evaluated.
        self._fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-term episode sums, mirroring UprightProgressReward.EpisodeComponentTotals: reading a
        # single scalar reward tells you nothing about *which* term a policy is actually farming.
        #
        # Populated lazily from whatever _reward_terms() actually returns, rather than a key list
        # kept in step with it by hand. Every subclass that changed the terms previously had to
        # remember to restate this list too, and a stale entry logs a term that is no longer summed
        # - which misattributes where the reward is coming from without failing.
        self._episode_sums: dict[str, torch.Tensor] = {}

    def _resample_commands(self, env_ids) -> None:
        """Draw a new velocity command for the given envs. Stand's command is always zero.

        The hook exists on the base task rather than only on Walk so both share one reset path:
        Walk inherits this environment wholesale, and a bootstrap from a Stand checkpoint is only
        valid while the two agree on what every observation slot means.
        """
        self._command[env_ids] = 0.0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                # Godot uses 1.2 friction on the feet and 0.9 on the body; a single ground value of
                # 1.0 sits between them. Friction is the parameter most likely to need randomising
                # before the policy survives the trip back, so it is written here to be found.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ stepping

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_action = self._action.clone()
        self._raw_action = actions
        self._action = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # [-1, 1] -> [lower, upper], piecewise-linear about the REST pose rather than about the
        # midpoint of the range.
        #
        # The affine mapping JointLimitedActionSpace uses puts a=0 at the centre of each joint's
        # limits, which for the knee ([-2.6, 0.1]) is -1.25 rad. A freshly initialised policy
        # outputs approximately zero, so the body spawns standing and is instantly commanded into
        # a deep crouch: measured here as a mean episode length of 22 steps out of 480, i.e. it
        # collapsed in 0.37 s before learning anything.
        #
        # This keeps the property that motivated the per-axis scaling in the first place - the full
        # [-1, 1] range maps onto the reachable range of that specific axis and nothing is wasted
        # past a hard stop - while making the neutral action the neutral pose. The two halves have
        # different slopes because the limits are asymmetric, which is exactly the point.
        # `action_scale` shrinks the authority of a unit action without changing its shape. At 1.0
        # the full [-1, 1] spans the joint's whole range, which sounds desirable and is not: the
        # knee's range is 2.7 rad, so an action of 0.5 commands a 1.35 rad swing, and against a
        # stiffness of 1600 Nm/rad that pins the actuator at its effort limit. The action space
        # becomes effectively bang-bang and balance needs fine adjustments. Measured at 1.0: the
        # policy plateaued at 48-step episodes while a zero-action policy from the same pose
        # survives 168 - i.e. acting was worse than doing nothing.
        span = torch.where(self._action >= 0.0, self._act_upper - self._act_default, self._act_default - self._act_lower)
        target = self._act_default + self.cfg.action_scale * self._action * span
        self.robot.set_joint_position_target(target, joint_ids=self._actuated_ids)

    # ------------------------------------------------------------------ observation

    def _get_observations(self) -> dict:
        data = self.robot.data
        projected_gravity = data.projected_gravity_b
        lin_vel_b = quat_rotate_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w)
        ang_vel_b = data.root_ang_vel_b

        forces = self.contact_sensor.data.net_forces_w[:, self._contact_ids, :]
        contacts = (forces.norm(dim=-1) > 1.0).float()

        obs = torch.cat(
            [
                projected_gravity,
                lin_vel_b,
                ang_vel_b,
                data.root_pos_w[:, 2].unsqueeze(-1) - self.scene.env_origins[:, 2].unsqueeze(-1),
                data.joint_pos - self._default_joint_pos,
                data.joint_vel,
                contacts,
                self._action,
                self._command,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ reward

    def _upright(self) -> torch.Tensor:
        """1.0 when the torso's up axis matches the world's, 0 on its side.

        Clamped at 0 so being upside-down is worth nothing rather than negative, which keeps this
        from competing with the explicit termination penalty.
        """
        return (-self.robot.data.projected_gravity_b[:, 2]).clamp(min=0.0)

    def _head_height(self) -> torch.Tensor:
        """Head centre of mass above this environment's ground.

        `body_com_pos_w`, not `body_pos_w`. The latter is the link *frame*, which this URDF places
        on each joint's pivot - for the head that is the neck, 14 cm low. Godot's "head height" is
        the Head rigid body's own origin, i.e. its centre, so the COM is the matching quantity and
        the ported 1.35 m threshold means the same thing in both engines.
        """
        return self.robot.data.body_com_pos_w[:, self._head_id[0], 2] - self.scene.env_origins[:, 2]

    def _height_reward(self) -> torch.Tensor:
        """Gaussian around the rest-pose head height. Shared so Walk and Run cannot drift from it."""
        return torch.exp(-((self._head_height() - REST_HEAD_HEIGHT) / 0.25) ** 2)

    def _action_clip_penalty(self) -> torch.Tensor:
        """Barrier on how far the RAW policy output strays outside the usable range.

        `_action` is already clamped, so this reads the unclamped tensor. Without it PPO's Gaussian
        mean drifts to infinity: the environment clamps to [-1, 1], so once a component is outside,
        pushing it further changes nothing that executes and therefore costs nothing. Measured over
        4000 iterations, 97% of action components ended up saturated and 34 of 36 joints sat
        permanently past the boundary - a bang-bang policy that stands only because one rigid pose
        happens to be stable, with no graded response left for a disturbance to act on.

        Zero inside [-1, 1], so it never distorts legitimate behaviour; it only forbids the
        degenerate region.
        """
        return torch.sum(torch.relu(self._raw_action.abs() - 1.0) ** 2, dim=1)

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Named reward contributions, already weighted.

        **Subclasses extend this rather than replacing `_get_rewards`.** That is not a style
        preference: `WalkEnv` previously overrode `_get_rewards` wholesale and silently dropped the
        `action_clip` barrier, which would have let its policy degenerate to bang-bang exactly as
        the first Stand run did - invisible in every quality metric. Terms defined here are
        inherited by every task unless a subclass deliberately deletes them by key.
        """
        data = self.robot.data
        dt = self.step_dt
        upright = self._upright()

        return {
            "alive": self.cfg.rew_alive * dt * torch.ones_like(upright),
            "upright": self.cfg.rew_upright * dt * upright,
            "head_height": self.cfg.rew_head_height * dt * self._height_reward(),
            "lin_vel": self.cfg.rew_lin_vel * dt * torch.sum(data.root_lin_vel_b[:, :2] ** 2, dim=1),
            "ang_vel": self.cfg.rew_ang_vel * dt * torch.sum(data.root_ang_vel_b**2, dim=1),
            "action_rate": self.cfg.rew_action_rate
            * torch.sum((self._action - self._previous_action) ** 2, dim=1),
            "action_clip": self.cfg.rew_action_clip * self._action_clip_penalty(),
            "joint_vel": self.cfg.rew_joint_vel * dt * torch.sum(data.joint_vel**2, dim=1),
            "effort": self.cfg.rew_effort * dt * self._effort_fraction(),
            "termination": self.cfg.rew_termination * self._fell.float(),
        }

    def _effort_fraction(self) -> torch.Tensor:
        """Mean squared torque as a fraction of each joint's limit."""
        applied = self.robot.data.applied_torque[:, self._actuated_ids]
        return torch.sum((applied / self._effort_limit) ** 2, dim=1) / len(ACTUATED_JOINTS)

    def _get_rewards(self) -> torch.Tensor:
        """Sums `_reward_terms()` and accumulates each one for per-term episode logging.

        Deliberately final in spirit - override `_reward_terms` instead. Everything that has to
        happen for every term (logging, summation) lives here exactly once.
        """
        terms = self._reward_terms()
        for key, value in terms.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[key] += value
        return torch.stack(list(terms.values()), dim=0).sum(dim=0)

    # ------------------------------------------------------------------ termination

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        # body_com_pos_w, not body_pos_w. The latter is the link *frame*, which this URDF places on
        # each joint's pivot - for the head that is the neck, 14 cm low. Godot's "head height" is
        # the Head rigid body's own origin, i.e. its centre, so the COM is the matching quantity
        # and the ported 1.35 m threshold means the same thing in both engines.
        head_height = data.body_com_pos_w[:, self._head_id[0], 2] - self.scene.env_origins[:, 2]
        upright = -data.projected_gravity_b[:, 2]

        # Cached because _get_rewards runs after _get_dones and needs the same mask for its
        # termination penalty; recomputing it there would be a second source of truth.
        self._fell = (head_height < FALL_HEAD_HEIGHT) | (upright < self._fall_cos)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._fell, timed_out

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.1, 0.1)
        joint_pos = joint_pos.clamp(
            self.robot.data.soft_joint_pos_limits[env_ids, :, 0],
            self.robot.data.soft_joint_pos_limits[env_ids, :, 1],
        )
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        # A little starting height and velocity noise so the policy cannot learn one exact
        # trajectory out of one exact pose; without it a standing start is nearly deterministic.
        root_state[:, 2] += torch.empty(len(env_ids), device=self.device).uniform_(-0.02, 0.02)
        root_state[:, 7:10] += torch.empty(len(env_ids), 3, device=self.device).uniform_(-0.1, 0.1)

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._action[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._resample_commands(env_ids)

        extras = {}
        for key, buffer in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = torch.mean(buffer[env_ids]).item()
            buffer[env_ids] = 0.0
        self.extras["log"] = extras
