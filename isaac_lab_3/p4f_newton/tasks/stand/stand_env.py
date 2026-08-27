"""Stand under Newton/XPBD: hold an upright, settled pose from a standing start.

The task is a port of `isaac_lab/p4f_isaac/tasks/stand/stand_env.py` and the objective is
unchanged, so the numbers are comparable. Every difference is forced by the solver, and all of them
come from one fact:

**Under XPBD, `joint_pos`, `joint_vel`, `applied_torque`, `root_quat_w`, every `root_*` velocity
and `projected_gravity_b` are frozen and read a clean, plausible zero.** That is not a bug — it is
what a maximal-coordinate solver is. See `p4f_newton/state.py`, which is the only place allowed to
read state, precisely so no observation term can quietly come from a stale buffer.

The failure this guards against is not subtle in its consequences: a frozen `projected_gravity_b`
reads `(0, 0, -1)` — an `upright` of exactly 1.000 — for a body lying face-down, so the reward pays
full marks for standing while the dummy is flat on the floor, and every log agrees.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from p4f_newton.assets import ACTUATED_JOINTS, CONTACT_BONES, REST_PELVIS_HEIGHT
from p4f_newton.state import NewtonRigState, quat_rotate_inverse, yaw_only

from .stand_env_cfg import FALL_HEAD_HEIGHT, FALL_TILT_DEG, REST_HEAD_HEIGHT, StandEnvCfg

# A foot/hand is "in contact" below this height above its environment's ground.
#
# Newton's contact reporting does not surface through Isaac Lab's `ContactSensor` on this backend,
# so the four contact flags in observation slice [100:104] are derived geometrically. The threshold
# is the foot box's own half-height (0.04 m) plus a small margin — a foot resting on the floor sits
# at 0.040, so this is "the sole is down", which is what Godot's flag means.
#
# It is a PROXY and is recorded as one: it cannot distinguish a foot resting on the ground from a
# foot passing 6 cm above it, and it will read true for a hand near the floor that is not touching.
# For Stand, where the feet are planted and the hands hang at 0.62 m, that is a good approximation.
# It would need revisiting for get-up, where hands genuinely bear load.
CONTACT_HEIGHT = 0.06

# Absolute bound on any observation component. See `StandEnv._sanitised` - this exists to stop a
# diverged environment corrupting the observation normaliser, not to shape the observation.
OBS_LIMIT = 100.0


class StandEnv(DirectRLEnv):
    cfg: StandEnvCfg

    def __init__(self, cfg: StandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve the frozen action order against whatever order the backend gave the DOFs.
        # `preserve_order` is the whole point: without it the ids come back sorted, every action
        # lands on a different joint than the contract names, and it trains perfectly happily into
        # a policy that is nonsense the moment Godot applies it in contract order.
        actuated, _ = self.robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        self._actuated_ids = torch.tensor(actuated, device=self.device, dtype=torch.long)

        limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self._act_lower = limits[self._actuated_ids, 0]
        self._act_upper = limits[self._actuated_ids, 1]
        self._act_default = self.robot.data.default_joint_pos.torch[0, self._actuated_ids]

        self._head_id = self.robot.body_names.index("Head")
        self._pelvis_id = self.robot.body_names.index("Pelvis")
        self._contact_ids = [self.robot.body_names.index(name) for name in CONTACT_BONES]
        self._foot_ids = [self.robot.body_names.index(n) for n in ("Foot_L", "Foot_R")]
        # EMA state for the balance assist's damping term. See `_apply_balance_assist`.
        self._balance_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self._default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self._effort_limit = self.robot.data.joint_effort_limits.torch[0].clamp(min=1e-6)
        # Per-episode torque budget scale. Resampled in `_reset_idx`; see `effort_scale_range`.
        self._effort_scale = torch.ones(self.num_envs, 1, device=self.device)
        self._stiffness = self.robot.data.joint_stiffness.torch[0]
        self._damping = self.robot.data.joint_damping.torch[0]

        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._action = torch.zeros_like(self._previous_action)
        self._raw_action = torch.zeros_like(self._previous_action)
        self._joint_target = self._default_joint_pos.clone()

        # Velocity command (vx, vy, yaw_rate), constant zero for Stand and reserved for Walk. Three
        # wasted floats now, and they buy the only thing that makes Walk cheap: Walk bootstraps from
        # a Stand checkpoint, which works only while the observation widths match.
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        self._fall_cos = math.cos(math.radians(FALL_TILT_DEG))
        # `_get_dones` runs before `_get_rewards` and fills this in, but the buffer has to exist
        # first: a reset can reach the reward path before any done has been evaluated.
        self._fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums: dict[str, torch.Tensor] = {}

        # Non-finite / impossible states seen since the last log. Surfaced through `extras` so a
        # rising rate is visible in TensorBoard rather than only in a crash hours later.
        self._divergences = 0

        # Built lazily: `NewtonManager._model` does not exist until the simulation has been
        # initialised, which happens inside `super().__init__` after `_setup_scene`.
        self._state: NewtonRigState | None = None
        self._joint_pos = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self._joint_vel = torch.zeros_like(self._joint_pos)

    # ------------------------------------------------------------------ state

    @property
    def rig(self) -> NewtonRigState:
        if self._state is None:
            self._state = NewtonRigState(self.robot, self.num_envs, self.robot.num_joints)
            self._verify_joint_alignment()
        return self._state

    def _verify_joint_alignment(self) -> None:
        """Does `eval_ik` index 0 mean the same joint as `robot.joint_names[0]`?

        Two independent orderings meet here — Isaac Lab's joint list, which the action tensor is
        written through, and Newton's DOF order, which `eval_ik` returns. If they disagree, actions
        land on the wrong joints while the observation reports the wrong angles, and BOTH errors are
        invisible: at the rest pose every joint sits near zero, so nothing numeric can see a
        permutation. That is the exact defect class `IsaacParityTest` exists for on the Godot side,
        and the one `obs_action_contract.md` was wrong about in four separate ways.

        This cannot be checked by comparing names, because `eval_ik` returns bare numbers. So it is
        checked causally: the count has to match, and the recovered vector has to respond where the
        command was applied. The second half runs on the first real step, in `_read_joint_state`.
        """
        q, qd = self.rig.joint_state()
        if q.shape[1] != self.robot.num_joints:
            raise RuntimeError(
                f"eval_ik returned {q.shape[1]} coordinates for {self.robot.num_joints} joints"
            )
        if qd.shape[1] != self.robot.num_joints:
            raise RuntimeError(
                f"eval_ik returned {qd.shape[1]} velocities for {self.robot.num_joints} joints"
            )

    def _read_joint_state(self) -> None:
        """One `eval_ik` per step, cached. Both the observation and the reward need it."""
        self._joint_pos, self._joint_vel = self.rig.joint_state()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                # Godot uses 1.2 friction on the feet and 0.9 on the body; a single ground value of
                # 1.0 sits between them.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    # ------------------------------------------------------------------ stepping

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # `copy_` into persistent buffers rather than rebinding to the incoming tensor.
        #
        # Rebinding looks equivalent and is not: under `torch.inference_mode()` — which is how a
        # policy is evaluated in the arena — `actions` is an *inference tensor*, and so is anything
        # cloned from it. `_reset_idx` then fails with "Inplace update to inference tensor outside
        # InferenceMode is not allowed" the first time an episode ends. Training never sees it,
        # because rsl_rl collects rollouts under `no_grad` instead, so this only breaks playback,
        # and only once a body falls.
        self._previous_action.copy_(self._action)
        self._raw_action.copy_(actions)
        limited = actions.clamp(-1.0, 1.0)
        if self.cfg.action_rate_limit > 0.0:
            # **In ACTION space and once per POLICY step**, which is the only place Godot can mirror
            # exactly. `_apply_action` runs once per PHYSICS tick, so limiting there would apply the
            # cap `decimation` times per policy step and silently double the real rate.
            #
            # The cap is on the commanded change, not on the joint's actual motion - the joint may
            # still be dragged faster by the body. It removes the STEP from the command, which is
            # what the two engines disagree about executing.
            step = self.cfg.action_rate_limit
            limited = self._action + (limited - self._action).clamp(-step, step)
        self._action.copy_(limited)

    def _apply_action(self) -> None:
        # [-1, 1] -> [lower, upper], piecewise-linear about the REST pose rather than the midpoint
        # of the range. The affine mapping puts a=0 at the centre of each joint's limits, which for
        # the knee is -1.25 rad: a freshly initialised policy outputs ~0, so the body spawns
        # standing and is instantly commanded into a deep crouch. Measured at 22 steps of 480.
        span = torch.where(
            self._action >= 0.0, self._act_upper - self._act_default, self._act_default - self._act_lower
        )
        target = self._act_default + self.cfg.action_scale * self._action * span
        if self.cfg.enforce_effort_limit:
            target = self._effort_limited(target)
        # Kept for the effort proxy below, which has no `applied_torque` to read.
        self._joint_target[:, self._actuated_ids] = target
        self.robot.set_joint_position_target_index(target=target, joint_ids=self._actuated_ids)

        if self.cfg.balance_assist > 0.0:
            self._apply_balance_assist()

    def _apply_balance_assist(self) -> None:
        """Pelvis attitude PD, ported verbatim from Godot's `PelvisStabilizationModule`.

        The pelvis is the unactuated skeletal root: without this it has no attitude control at all,
        in either engine. Godot supplies it from its balance layer and Isaac supplies nothing, which
        is why "zero action" means *stand* in one and *fall* in the other - and why a policy trained
        here learned to command a crouch that destabilises a Godot dummy which was standing fine.

        Applied from `_apply_action`, i.e. once per PHYSICS substep, because the solver zeroes
        `State.body_f` every step.

        The counter-torque goes into the feet rather than into nothing. That matters physically: an
        attitude torque on a free root with no reaction is angular momentum from nowhere, and the
        body would happily spin up. Godot distributes it across the grounded feet; here it is split
        across both, which is the same thing whenever the dummy is standing on two feet and a mild
        approximation when it is not.
        """
        rig = self.rig
        up = rig.up_axis_w()
        world_up = torch.zeros_like(up)
        world_up[:, 2] = 1.0
        # Axis-angle attitude error; magnitude ~ sin(angle), zero when perfectly upright.
        attitude_error = torch.cross(up, world_up, dim=-1)

        alpha = self.cfg.balance_filter_alpha
        self._balance_ang_vel += (rig.root_ang_vel_w - self._balance_ang_vel) * alpha

        torque = self.cfg.balance_gain * attitude_error - self.cfg.balance_damping * self._balance_ang_vel
        cap = self.cfg.balance_max_torque * self.cfg.balance_assist
        norm = torch.linalg.vector_norm(torque, dim=-1, keepdim=True)
        torque = torque * (cap / norm.clamp(min=1e-6)).clamp(max=1.0)

        rig.add_body_torque([self._pelvis_id], torque)
        if self.cfg.balance_reaction:
            # Only into feet that are actually down, as Godot does - the reaction has to reach the
            # ground through a planted sole, not spin an airborne leg.
            grounded = self._contacts()[:, 2:4]
            share = grounded / grounded.sum(dim=1, keepdim=True).clamp(min=1.0)
            for slot, body_id in enumerate(self._foot_ids):
                rig.add_body_torque([body_id], -torque * share[:, slot : slot + 1])

    def _effort_limited(self, target: torch.Tensor) -> torch.Tensor:
        """Pull each target back to the furthest one this joint's torque budget can actually ask for.

        **`SolverXPBD` silently ignores `joint_effort_limit`** - `assets.py` records this as a known
        gap - so without this the policy trains against drives that can deliver unbounded torque.
        That is not a small mismatch, it is the one that decides transfer: measured against a
        checkpoint scoring 100% here, the learned solution was to move into a single braced pose,
        pin `max |action|` at exactly 1.00, and hold it with `joint_vel` decaying to 0.18 rad/s -
        a statue, not a balance controller. It satisfies every standing criterion because nothing in
        the criterion ever pushes it. In Godot the same commanded pose sags, because Godot's
        actuators have a real ceiling, and a policy that only knows how to HOLD is out of
        distribution the moment the body starts to move.

        The commanded angle is clamped rather than the torque, because the drive is a position
        target and the torque is the solver's to compute. Inverting the PD law for the target that
        produces exactly the effort limit gives a joint that behaves like an effort-limited
        actuator - the same law, and the same clamp, Godot applies.

        `_joint_pos` / `_joint_vel` are one policy step old, which is deliberate: that is exactly
        the information the policy itself acted on, so the clamp cannot use knowledge the policy
        did not have.
        """
        q = self._joint_pos[:, self._actuated_ids]
        qd = self._joint_vel[:, self._actuated_ids]
        kp = self._stiffness[self._actuated_ids]
        kd = self._damping[self._actuated_ids]
        effort = self._effort_limit[self._actuated_ids] * self._effort_scale

        torque = kp * (target - q) - kd * qd

        # **Clamped per BONE, as a 3-axis vector - not per axis.** Godot's `ActiveBone` bounds
        # `totalTorque.Length()` against one `MaxTorque`, so a bone's three axes share a single
        # budget. A per-axis clamp lets Isaac spend up to `effort * sqrt(3)` on the same joint -
        # 693 N.m where Godot delivers 400 - and the difference is not academic: every limb's twist
        # axis is limited to about +/-0.10 rad and chatters against its stops in Godot, so the y and
        # z axes burn budget that the knee's x axis needs to hold the body up. The policy learns to
        # rely on torque Godot will never give it, and the legs buckle in about a second.
        #
        # `AxesPerBone` groups are contiguous in action order, which is what makes the reshape valid
        # - the contract defines index 3*i+{0,1,2} as bone i's three axes.
        axes = 3
        bones = torque.shape[1] // axes
        grouped = torque.view(-1, bones, axes)
        norm = torch.linalg.vector_norm(grouped, dim=-1, keepdim=True)
        budget = effort.view(-1, bones, axes)[..., 0].unsqueeze(-1)
        budget = budget * self._hill_scale(grouped, qd.view(-1, bones, axes), norm)
        scale = (budget / norm.clamp(min=1e-6)).clamp(max=1.0)
        allowed = (grouped * scale).view_as(torque)

        # Invert the PD law for the target that delivers exactly the allowed torque.
        reachable = q + (allowed + kd * qd) / kp.clamp(min=1e-6)

        # `isfinite` is load-bearing, not defensive. Roughly one environment in 256 diverges under
        # XPBD, and `_diverged` resets it - but that happens at the END of the step, so a non-finite
        # `_joint_pos` reaches this clamp first. Without the guard a NaN passes straight into the
        # position target, where it is no longer one bad environment: it is a NaN action, a NaN
        # gradient, and a run that keeps writing checkpoints of a destroyed policy. Caught as
        # `mean |action| = nan` in an evaluation that had reported a clean 0.456 the day before.
        limited = torch.where(torch.isfinite(reachable), reachable, target)

        # **Clamp back into the joint's own range**, because inverting the PD law can leave it far
        # outside. `reachable = q + (allowed + kd*qd) / kp` divides by the gain, so a low-gain joint
        # carrying a high velocity produces an enormous target: at kp=60, kd=6 and qd=100 rad/s that
        # is 16.7 rad on an axis limited to +/-0.5. Commanding it is what turns a fast joint into a
        # diverged one, which is the likeliest source of the roughly one environment in 256 that goes
        # non-finite. A target outside the limits is meaningless anyway - the solver constrains the
        # joint there regardless.
        return torch.clamp(limited, self._act_lower, self._act_upper)

    # ------------------------------------------------------------------ observation

    def _get_observations(self) -> dict:
        self._read_joint_state()
        rig = self.rig

        quat = rig.root_quat_w
        projected_gravity = rig.projected_gravity_b()
        lin_vel_b = quat_rotate_inverse(yaw_only(quat), rig.root_lin_vel_w)
        ang_vel_b = quat_rotate_inverse(quat, rig.root_ang_vel_w)

        obs = torch.cat(
            [
                projected_gravity,
                lin_vel_b,
                ang_vel_b,
                (rig.root_pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1),
                self._joint_pos - self._default_joint_pos,
                self._observed_joint_vel(),
                self._contacts(),
                self._action,
                self._command,
            ],
            dim=-1,
        )
        return {"policy": self._sanitised(obs)}

    def _sanitised(self, obs: torch.Tensor) -> torch.Tensor:
        """Guarantee nothing non-finite ever leaves this environment.

        **`_diverged` cannot do this job, because it is structurally one step late.** `_joint_pos`
        and `_joint_vel` are refreshed here, in `_get_observations`, which runs at the END of a step;
        `_get_dones` runs before it and therefore tests the PREVIOUS step's values. So the step in
        which a joint first goes non-finite produces a NaN observation, hands it to rsl_rl, and only
        gets terminated on the step after.

        rsl_rl does not tolerate that: it raises *"The observation group 'policy' returned by the
        environment contains NaN values"* and the process exits. It killed two unattended chains
        last night, at 06:24 and 07:24, costing about an hour and three quarters of GPU time between
        them - and the guard was reporting `Diagnostics/divergences: 0.0000` throughout, because the
        environments it would have counted had already crashed the run.

        Replacing with zero is safe precisely because the environment is about to be reset anyway:
        the value is used for one policy step and then thrown away. The count is surfaced so this
        stays visible rather than becoming a silent scrub - if it climbs, the physics needs
        attention.
        """
        bad = ~torch.isfinite(obs)
        if bad.any():
            self._divergences += int(bad.any(dim=1).sum().item())
            obs = torch.where(bad, torch.zeros_like(obs), obs)

        # **Bounded, not just finite.** Replacing NaN is not enough: a huge FINITE observation is
        # just as destructive, because `obs_normalization` accumulates a running mean and variance
        # from these values and one excursion poisons the statistics for every environment and every
        # step afterwards. It surfaces far from the cause, as
        # `RuntimeError: normal expects all elements of std >= 0.0` once the actor's own parameters
        # have gone non-finite - which is how Perturbation failed at a 25 N.s impulse ceiling while
        # training happily at 12, and why switching the std to log space did not help.
        #
        # Every slice in the contract is bounded by construction: gravity is a unit vector, joint
        # velocity is clipped, contacts are flags, actions and commands are in [-1, 1]. Nothing
        # legitimate approaches this bound, so it only ever truncates a blow-up.
        return obs.clamp(-OBS_LIMIT, OBS_LIMIT)

    def _observed_joint_vel(self) -> torch.Tensor:
        """The joint-velocity slice as the POLICY sees it: clipped, and noisy during training.

        See `StandEnvCfg.obs_joint_vel_clip`. Godot's tightly-limited twist axes chatter against
        their stops at up to 68 rad/s where Isaac's peak across all 45 DOF is 7.5, so a policy that
        trusts this channel precisely cannot survive the crossing. The noise is the part that
        matters - it teaches the policy to lean on gravity, height and joint ANGLE instead, all of
        which the two engines already agree on.

        Noise is training-only. An evaluation or an export must measure the policy, not the
        sampling, and Godot adds no noise of its own.
        """
        vel = self._joint_vel
        if self.cfg.obs_joint_vel_noise > 0.0 and not self.cfg.playback:
            vel = vel + torch.randn_like(vel) * self.cfg.obs_joint_vel_noise
        clip = self.cfg.obs_joint_vel_clip
        return vel.clamp(-clip, clip) if clip > 0.0 else vel

    def _contacts(self) -> torch.Tensor:
        """Hand_L, Hand_R, Foot_L, Foot_R ground flags — see CONTACT_HEIGHT for why this is height."""
        com = self.robot.data.body_com_pos_w.torch
        heights = com[:, self._contact_ids, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)
        return (heights < CONTACT_HEIGHT).float()

    # ------------------------------------------------------------------ reward

    def _head_height(self) -> torch.Tensor:
        """Head centre of mass above this environment's ground.

        `body_com_pos_w`, not `body_link_pos_w`: the latter is the link frame, which sits on the
        joint pivot — for the head that is the neck, 14 cm low. Godot's "head height" is the Head
        rigid body's own centre, so the COM is the matching quantity and the ported 1.35 m
        threshold means the same thing in both engines.
        """
        return self.robot.data.body_com_pos_w.torch[:, self._head_id, 2] - self.scene.env_origins[:, 2]

    def _height_reward(self) -> torch.Tensor:
        return torch.exp(-((self._head_height() - REST_HEAD_HEIGHT) / 0.25) ** 2)

    def _action_clip_penalty(self) -> torch.Tensor:
        """Barrier on how far the RAW policy output strays outside the usable range.

        `_action` is already clamped, so this reads the unclamped tensor. Without it PPO's Gaussian
        mean drifts to infinity: the environment clamps, so once a component is outside, pushing it
        further changes nothing that executes and therefore costs nothing.
        """
        return torch.sum(torch.relu(self._raw_action.abs() - 1.0) ** 2, dim=1)

    def _effort_fraction(self) -> torch.Tensor:
        """Mean squared PD torque as a fraction of each joint's limit.

        **A proxy.** `applied_torque` is one of the frozen buffers under XPBD — it reads exactly 0
        for every joint on every step — so the actual delivered torque is not observable through
        Isaac Lab here. This recomputes what the drive is being asked for, `kp*(target - q) - kd*qd`,
        from the gains the solver actually holds and the angles `eval_ik` recovers.

        It is the COMMANDED torque, not the delivered one, so it does not see XPBD clamping to the
        effort limit. That makes it an upper bound on effort, which is the right direction for a
        penalty term: it never under-reports work the policy is doing.
        """
        error = self._joint_target - self._joint_pos
        torque = self._stiffness * error - self._damping * self._joint_vel
        return torch.sum((torque / self._effort_limit) ** 2, dim=1) / self.robot.num_joints

    def _hill_scale(
        self, torque: torch.Tensor, joint_vel: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        """Hill force-velocity ceiling per bone, in [0, 1]. Ported from `ActiveBone`.

        Only the component of joint velocity ALONG the commanded torque counts as shortening. A
        joint being braked - velocity opposing the torque - is an eccentric contraction, where
        muscle is stronger than isometric, so it keeps full authority rather than being penalised.
        **That asymmetry is what makes this safe for balance:** arresting a limb is braking, and
        arresting a limb is most of what standing consists of. What it removes is the ability to
        keep driving a joint that is already spinning.

        Linear falloff, not Hill's hyperbola - matching Godot exactly, which chose the linear form
        because the quantity being bounded is peak power and it cannot go negative.
        """
        vmax = self.cfg.hill_max_shortening_velocity
        if vmax <= 0.0:
            return torch.ones_like(norm)

        shortening = (joint_vel * torque).sum(dim=-1, keepdim=True) / norm.clamp(min=1e-6)
        derated = (1.0 - shortening / vmax).clamp(0.0, 1.0)
        # Eccentric (shortening <= 0) keeps the full ceiling.
        return torch.where(shortening > 0.0, derated, torch.ones_like(derated))

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Named reward contributions, already weighted.

        **Subclasses extend this rather than replacing `_get_rewards`.** `WalkEnv` previously
        overrode `_get_rewards` wholesale in the 2.3.2 tree and silently dropped the `action_clip`
        barrier, which would have let its policy degenerate to bang-bang exactly as the first Stand
        run did — invisible in every quality metric.
        """
        dt = self.step_dt
        rig = self.rig
        upright = rig.upright().clamp(min=0.0)
        lin_vel = rig.root_lin_vel_w
        ang_vel = rig.root_ang_vel_w

        return {
            "alive": self.cfg.rew_alive * dt * torch.ones_like(upright),
            "upright": self.cfg.rew_upright * dt * upright,
            "head_height": self.cfg.rew_head_height * dt * self._height_reward(),
            "lin_vel": self.cfg.rew_lin_vel * dt * torch.sum(lin_vel[:, :2] ** 2, dim=1),
            "ang_vel": self.cfg.rew_ang_vel * dt * torch.sum(ang_vel**2, dim=1),
            "action_rate": self.cfg.rew_action_rate * torch.sum((self._action - self._previous_action) ** 2, dim=1),
            "action_clip": self.cfg.rew_action_clip * self._action_clip_penalty(),
            "joint_vel": self.cfg.rew_joint_vel * dt * torch.sum(self._joint_vel**2, dim=1),
            "effort": self.cfg.rew_effort * dt * self._effort_fraction(),
            "termination": self.cfg.rew_termination * self._fell.float(),
        }

    def _get_rewards(self) -> torch.Tensor:
        """Summed reward terms, with every term scrubbed of non-finite values first.

        **The observation is not the only way a NaN escapes this environment.** rsl_rl checks the
        rewards too, and refuses the rollout with *"The rewards returned by the environment contain
        NaN values"* - which killed a chain at 09:30 after the observation path had already been
        sanitised. Every term here is built from `_joint_pos`, `_joint_vel` or a body pose, so a
        single diverged environment poisons its own reward, its episode sums, and through the sums
        the logged averages for all 8192.

        Scrubbing per TERM rather than only the total keeps `_episode_sums` finite as well; a NaN
        that reached a sum would stay there for the rest of the episode and be reported as the
        environment's contribution. The count is folded into the same divergence counter, so this
        stays visible instead of becoming a silent repair.
        """
        terms = self._reward_terms()
        for key, value in terms.items():
            bad = ~torch.isfinite(value)
            if bad.any():
                self._divergences += int(bad.sum().item())
                value = torch.where(bad, torch.zeros_like(value), value)
                terms[key] = value
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[key] += value
        return torch.stack(list(terms.values()), dim=0).sum(dim=0)

    # ------------------------------------------------------------------ termination

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        head_height = self._head_height()
        upright = self.rig.upright()

        # Cached because `_get_rewards` runs after `_get_dones` and needs the same mask for its
        # termination penalty; recomputing it there would be a second source of truth.
        self._fell = (head_height < FALL_HEAD_HEIGHT) | (upright < self._fall_cos)

        diverged = self._diverged(head_height, upright)
        if diverged.any():
            # Terminate rather than penalise. A diverged environment is not a policy failure and
            # must not be scored as one - it is a solver blow-up, and the fall penalty would teach
            # the policy to avoid whatever it happened to be doing.
            self._fell = self._fell & ~diverged
            self._divergences += int(diverged.sum().item())

        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._fell | diverged, timed_out

    def _diverged(self, head_height: torch.Tensor, upright: torch.Tensor) -> torch.Tensor:
        """Environments whose state has gone non-finite or physically impossible.

        **Measured at roughly 1 environment in 256 over 8 seconds.** Tolerable in playback, where a
        single NaN only poisons a mean. Not tolerable in an unattended run: a NaN observation
        produces a NaN gradient, and rsl_rl does not necessarily stop - it propagates into the
        weights and the run continues writing checkpoints of a destroyed policy for hours. The 2.3.2
        track lost a run to exactly this when `IdealPDActuator` diverged.

        Resetting the environment bounds the damage to one episode. The counter is logged so a
        rising divergence rate is visible rather than silent - if it climbs, the physics needs
        attention, and that is a different problem from the policy plateauing.
        """
        # **Parenthesise the negation.** `.any()` binds tighter than `~`, so
        # `~torch.isfinite(x).any(dim=1)` reads "NOT (any element is finite)" - it only fires when
        # EVERY joint has gone non-finite, which essentially never happens. The guard therefore
        # reported `Diagnostics/divergences: 0.0000` for hours while NaNs propagated freely, until
        # rsl_rl finally refused the rollout with "The observation group 'policy' ... contains NaN
        # values" and killed a 12-segment chain mid-run.
        bad = ~torch.isfinite(head_height) | ~torch.isfinite(upright)
        bad |= (~torch.isfinite(self._joint_pos)).any(dim=1)
        bad |= (~torch.isfinite(self._joint_vel)).any(dim=1)
        # A head above 3 m has not been reached by standing on a 1.54 m body; it is a launch.
        bad |= head_height > 3.0
        return bad

    # ------------------------------------------------------------------ reset

    def _resample_commands(self, env_ids) -> None:
        """Draw a new velocity command for the environments being reset.

        Zero for Stand, and a seam rather than a literal so `WalkEnv` does not have to override
        `_reset_idx` to change one line. The 2.3.2 tree learned this the expensive way with
        `_get_rewards`: a subclass that copies a whole method to change part of it silently drops
        whatever the parent adds later.
        """
        self._command[env_ids] = 0.0

    def _starting_push(self, env_ids) -> torch.Tensor | None:
        """A random shove applied at spawn, as a root twist. `None` disables it.

        **This is what stops the policy solving the task by standing still.** With a deterministic
        start and no disturbance, the optimal policy is a statue: move once into a braced pose and
        freeze. Measured on a checkpoint scoring 100% of the strict criterion - `joint_vel` decayed
        to 0.18 rad/s and `max |action|` sat at exactly 1.00 forever. That satisfies head height,
        tilt and speed perfectly and teaches nothing about balance, so the moment it meets an engine
        whose actuators sag it has no recovery behaviour to fall back on.

        A push at spawn is deliberately cheap rather than clever: it needs no external-force API on
        a backend where contact reporting already does not surface, and it costs nothing per step.
        The body has to arrive at its own equilibrium from somewhere different every episode, which
        is the property a statue does not have.

        The magnitudes are small on purpose - a shove, not a launch. `docs/RL-TRAINING.md` records a
        perturbation curriculum overshooting into a regime where falling was unavoidable, at which
        point the policy correctly learned that nothing it did mattered.
        """
        if self.cfg.push_velocity <= 0.0:
            return None
        count = len(env_ids)
        twist = torch.zeros(count, 6, device=self.device)

        pushed = torch.rand(count, device=self.device) < self.cfg.push_probability
        # Uniform in direction, uniform in magnitude - not a normal draw, whose tail would
        # occasionally deliver a shove nothing could recover from.
        angle = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * math.pi)
        speed = torch.empty(count, device=self.device).uniform_(0.0, self.cfg.push_velocity)
        speed = speed * pushed.float()
        twist[:, 0] = speed * torch.cos(angle)
        twist[:, 1] = speed * torch.sin(angle)

        spin = torch.empty(count, 3, device=self.device).uniform_(-1.0, 1.0)
        twist[:, 3:6] = spin * self.cfg.push_ang_velocity * pushed.float().unsqueeze(-1)
        return twist

    def _reset_idx(self, env_ids) -> None:
        # Only when None. The 2.3.2 version also substituted `_ALL_INDICES` for a full-size
        # `env_ids`, which is wrong here: under Newton `_ALL_INDICES` is a `wp.array`, and the base
        # class indexes `episode_length_buf` with it — warp arrays do not support item indexing, so
        # a full reset raised `RuntimeError: Item indexing is not supported on wp.array objects`.
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        noise = self.cfg.reset_joint_noise
        if noise > 0.0:
            joint_pos += torch.empty_like(joint_pos).uniform_(-noise, noise)
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])
        joint_vel = torch.zeros_like(joint_pos)

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        # A little starting height noise so the policy cannot learn one exact trajectory out of one
        # exact pose; without it a standing start is nearly deterministic.
        if self.cfg.reset_height_noise > 0.0:
            h = self.cfg.reset_height_noise
            root_pose[:, 2] += torch.empty(len(env_ids), device=self.device).uniform_(-h, h)

        # Straight to Newton's state, NOT through `write_root_pose_to_sim_index` — that call does
        # not round-trip here and respawns the body 0.82 m in the air on every reset after the
        # first. See NewtonRigState.reset_to.
        #
        # `default_root_pose` stores its rotation in the same order the config declares it, which is
        # Isaac Lab 3's xyzw — the same order Newton's free joint wants, so it copies directly.
        self.rig.reset_to(
            env_ids=env_ids,
            root_pos=root_pose[:, :3],
            root_quat_xyzw=root_pose[:, 3:7],
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_vel=self._starting_push(env_ids),
        )

        # Nominal budget when measuring: an evaluation or an export has to report the policy
        # against the actuator the contract describes, not against a random draw from it.
        low, high = (1.0, 1.0) if self.cfg.playback else self.cfg.effort_scale_range
        self._effort_scale[env_ids] = torch.empty(
            len(env_ids), 1, device=self.device
        ).uniform_(low, high)

        self._balance_ang_vel[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._joint_target[env_ids] = self._default_joint_pos[env_ids]
        self._resample_commands(env_ids)

        extras = {}
        for key, buffer in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = torch.mean(buffer[env_ids]).item()
            buffer[env_ids] = 0.0
        extras["Diagnostics/divergences"] = float(self._divergences)
        self._divergences = 0
        self.extras["log"] = extras
