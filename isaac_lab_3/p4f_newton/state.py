"""Reading the dummy's state under a maximal-coordinate solver.

**Half of `ArticulationData` is frozen under XPBD and reads a clean, plausible zero.** Nothing
errors. Measured over 180 steps of a body visibly falling over:

| live | frozen |
|---|---|
| `body_com_pos_w`, `body_link_pos_w` | `root_quat_w` |
| `body_quat_w`, `body_link_quat_w` | `root_lin_vel_w/b`, `root_ang_vel_w/b` |
| `body_com_lin_vel_w`, `body_ang_vel_w` | `projected_gravity_b` |
| | `joint_pos`, `joint_vel`, `applied_torque` |

The split is not arbitrary and it is not a bug: **everything per-BODY is live, everything
joint-level or derived from the articulation's root state is frozen.** That is what a
maximal-coordinate solver *is*. XPBD integrates body poses directly and treats joints as
constraints between them, so generalized joint coordinates are never part of its state, and neither
is a distinguished "root" pose. A reduced-coordinate solver — PhysX, MuJoCo, Featherstone — carries
joint angle as *the* state variable, which is exactly the distinction this whole track set out to
test. Godot is on this side of the line too, and derives joint angles the same way.

So this module exists to make the frozen half unreachable. Every quantity the task needs is built
here from a live source:

* **root pose and velocity** come from the **Pelvis body**, not from `root_*`.
* **joint angles and velocities** come from `newton.eval_ik`, which recovers them from
  `State.body_q` / `body_qd`.
* **projected gravity** is computed from the pelvis quaternion.

The danger this guards against is specific and severe: a frozen `projected_gravity_b` reads
`(0, 0, -1)` — an `upright` of exactly 1.000 — for a body lying face-down on the floor, and a frozen
`joint_pos` reads exactly 0.0000, which is the healthy rest pose. A reward built on those pays a
policy for standing perfectly while the dummy is flat on the ground, and every log agrees.
"""

from __future__ import annotations

import newton
import torch
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager

# Per-environment coordinate/DOF layout of Newton's flat joint arrays. The root FREE joint comes
# first and carries 7 coordinates (position + quaternion) but only 6 DOF (linear + angular
# velocity) — the counts genuinely differ, and using one for the other silently shifts every joint.
ROOT_COORDS = 7
ROOT_DOFS = 6

# Offset of the ANGULAR half of a spatial wrench in `State.body_f`. Measured with
# `scripts/probe_twist.py --wrench`; see `NewtonRigState.add_body_torque`.
WRENCH_ANGULAR = 3


class NewtonRigState:
    """Live state for one articulation, read the only way XPBD actually supports.

    Construct after `sim.reset()`, once `NewtonManager._model` exists.
    """

    def __init__(self, robot, num_envs: int, num_dofs: int, pelvis_body: str = "Pelvis") -> None:
        self._robot = robot
        self._num_envs = num_envs
        self._num_dofs = num_dofs
        self._pelvis_id = robot.body_names.index(pelvis_body)

        model = NewtonManager._model
        self._model = model
        self._joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=model.device)
        self._joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)

        # `joint_q`/`joint_qd` are single flat arrays over EVERY environment, in per-env blocks.
        # Stripping only the leading root entries removes environment 0's root and leaves every
        # other environment's root inside the result — which puts world POSITIONS into a joint-angle
        # array. Measured: max |q| reads 10.5 rad against joints limited to +/-0.5, and the mean
        # drifts with `num_envs` and `env_spacing` rather than with the physics.
        self._coords_per_env = ROOT_COORDS + num_dofs
        self._dofs_per_env = ROOT_DOFS + num_dofs
        if model.joint_coord_count != num_envs * self._coords_per_env:
            raise RuntimeError(
                f"joint_coord_count {model.joint_coord_count} != {num_envs} x {self._coords_per_env}"
            )
        if model.joint_dof_count != num_envs * self._dofs_per_env:
            raise RuntimeError(
                f"joint_dof_count {model.joint_dof_count} != {num_envs} x {self._dofs_per_env}"
            )

    # ------------------------------------------------------------------ joints

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint angles (rad) and velocities (rad/s), each shaped (num_envs, num_dofs).

        One `eval_ik` per call, so callers should take both at once and cache them for the step
        rather than asking twice.
        """
        newton.eval_ik(self._model, NewtonManager._state_0, self._joint_q, self._joint_qd)
        q = wp.to_torch(self._joint_q).view(self._num_envs, self._coords_per_env)[:, ROOT_COORDS:]
        qd = wp.to_torch(self._joint_qd).view(self._num_envs, self._dofs_per_env)[:, ROOT_DOFS:]
        return q, qd

    def sync_bodies_from_joints(self) -> None:
        """Recompute body poses from the joint coordinates.

        **Tried against the reset bug and it did NOT help — kept only so nobody spends the time
        again.** The reasoning was sound: a maximal-coordinate solver's state is `body_q`, joint
        coordinates are a derived view, so a joint-position write updates a buffer the solver does
        not integrate and `eval_fk` should be needed to make the two agree. Measured with and
        without, the reset still places the pelvis at 1.640 instead of 0.820, identically.

        The real cause is upstream, in `write_root_*_pose_to_sim_index` — see README §3.
        """
        newton.eval_fk(
            self._model,
            NewtonManager._state_0.joint_q,
            NewtonManager._state_0.joint_qd,
            NewtonManager._state_0,
        )

    def reset_to(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_vel: torch.Tensor | None = None,
    ) -> None:
        """Place the given environments at a pose, writing the state the solver actually integrates.

        **Isaac Lab's `write_root_*_pose_to_sim_index` does not round-trip on this backend.** The
        first reset lands correctly; every one after it pins the pelvis at 1.640 m — the authored
        0.82 plus the written 0.82 — and the written height is then ignored entirely. Writing 0.0
        gives 1.640 too, and so does `write_root_link_pose_to_sim_index`.

        Left alone that is not a visible crash. The dummy respawns 0.82 m in the air every episode,
        falls, respawns, and the aggregate reads as a body GAINING height: pelvis 0.820 -> 1.582,
        head to 2.302. It therefore never trips the fall threshold, `Episode_Reward/termination`
        stays exactly 0.0000, and mean episode length pins at 479 of 480 at every environment count.
        That reads as a policy which has learned to stand perfectly, in 25 iterations.

        So the reset goes straight to Newton's own state instead. `joint_q` is one flat array over
        every environment in blocks of ``7 + num_dofs``: three position coordinates, four quaternion
        coordinates in **xyzw**, then one per joint. `joint_qd` uses blocks of ``6 + num_dofs``,
        with the root's six twist entries first — the counts differ, and using one layout for the
        other silently shifts every joint.

        `eval_fk` then propagates those coordinates into `body_q`, which is what XPBD integrates.
        Without it the bodies keep their old poses while `joint_q` claims otherwise, and the solver
        turns that disagreement into energy.
        """
        state = NewtonManager._state_0
        q = wp.to_torch(state.joint_q).view(self._num_envs, self._coords_per_env)
        qd = wp.to_torch(state.joint_qd).view(self._num_envs, self._dofs_per_env)

        q[env_ids, 0:3] = root_pos
        q[env_ids, 3:7] = root_quat_xyzw
        q[env_ids, ROOT_COORDS:] = joint_pos

        # Start at rest unless the caller asks for a shove: a respawning body carries no momentum
        # from the episode that just ended.
        #
        # `root_vel` is the root FREE joint's six-entry twist, **linear in [0:3] and angular in
        # [3:6]** - measured with `scripts/probe_twist.py`, not read off a docstring. Writing 2.0
        # into [0] gives the pelvis lin (+2.000, 0, 0) and ang (0, 0, 0); into [3] it gives lin
        # (0, 0, 0) and ang (+1.999, 0, 0). warp's spatial vectors are conventionally the other way
        # round (angular first), so this is worth having measured: getting it backwards does not
        # raise, it spins the body where a push was intended, and both look like a working
        # perturbation.
        qd[env_ids, :ROOT_DOFS] = 0.0 if root_vel is None else root_vel
        qd[env_ids, ROOT_DOFS:] = joint_vel

        # **`indices` is mandatory, not an optimisation.** Unmasked, `eval_fk` rewrites `body_q`
        # for EVERY environment from `joint_q` — and XPBD does not maintain `joint_q` while it
        # simulates, so for every environment that is not resetting those coordinates are stale,
        # frozen at whatever was last written. One environment resetting therefore teleports all
        # of them back to a near-rest, perfectly upright pose.
        #
        # Measured with 256 environments and zero actions: `upright` never left 0.99999, mean head
        # height floated at 1.55-2.18 m, and only 10 of 256 bodies fell in 8 seconds. Episodes
        # never terminated, so `ep_len` pinned at 479 of 480 and `Episode_Reward/upright` at
        # 15.9666 — a training curve that looks like a policy holding a flawless stand. The policy
        # trained against it for 334M steps and learned to fall over roughly twice as fast as
        # commanding nothing at all.
        indices = wp.from_torch(env_ids.to(torch.int32).contiguous(), dtype=wp.int32)
        newton.eval_fk(self._model, state.joint_q, state.joint_qd, state, indices=indices)

    # ------------------------------------------------------------------ root, via the pelvis body

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Pelvis orientation, wxyz. Replaces the frozen `root_quat_w`."""
        return self._robot.data.body_quat_w.torch[:, self._pelvis_id]

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Pelvis centre of mass in world.

        Also replaces `root_pos_w` for a second reason beyond staleness: that field reports 1.640
        for a pelvis the physics puts at 0.820, because it adds the articulation's authored root
        offset on top of the free joint's own translation. Plausible, stable, and exactly double.
        """
        return self._robot.data.body_com_pos_w.torch[:, self._pelvis_id]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._robot.data.body_com_lin_vel_w.torch[:, self._pelvis_id]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._robot.data.body_ang_vel_w.torch[:, self._pelvis_id]

    def add_body_torque(self, body_ids: list[int], torque: torch.Tensor) -> None:
        """Accumulate an external torque on the given bodies, one row of `torque` per environment.

        Writes `State.body_f`, which Newton documents as "an external wrench in world frame with the
        body's centre of mass as reference point". **The solver zeroes it every step**, so this has
        to be called from `_apply_action` - once per physics substep - not once per policy step.

        The wrench's angular/linear split is measured, not assumed: `scripts/probe_twist.py
        --wrench` writes a known value into each half and reports which one spins the body. warp's
        spatial vectors are conventionally angular-first while `joint_qd` measured linear-first on
        this build, so the two are not safe to infer from each other.
        """
        state = NewtonManager._state_0
        if state.body_f is None:
            raise RuntimeError("State.body_f is None; external wrenches are unavailable")
        wrench = wp.to_torch(state.body_f).view(self._num_envs, -1, 6)
        ids = torch.as_tensor(body_ids, device=torque.device, dtype=torch.long)
        wrench[:, ids, WRENCH_ANGULAR : WRENCH_ANGULAR + 3] += torque.unsqueeze(1)

    def add_body_wrench(
        self, body_idx: torch.Tensor, force: torch.Tensor, torque: torch.Tensor
    ) -> None:
        """External force and torque on ONE body per environment, chosen per environment.

        `body_idx` is `(num_envs,)`, indexing bodies within an environment; `force` and `torque` are
        `(num_envs, 3)` in world frame about the body's centre of mass.

        Used where the target varies per episode - a perturbation shot picks a different bone each
        time - which `add_body_torque` cannot express because it applies the same bodies to every
        environment.

        **`robot.set_external_force_and_torque` is not the route here.** That path goes through the
        PhysX view, which does not exist on a Newton backend; `State.body_f` is the buffer this
        solver actually integrates. It is also zeroed every step, which is exactly what a
        perturbation wants - write a force for one step and it behaves as an impulse rather than a
        sustained wind.
        """
        state = NewtonManager._state_0
        if state.body_f is None:
            raise RuntimeError("State.body_f is None; external wrenches are unavailable")
        wrench = wp.to_torch(state.body_f).view(self._num_envs, -1, 6)
        env_idx = torch.arange(self._num_envs, device=body_idx.device)
        wrench[env_idx, body_idx, :WRENCH_ANGULAR] += force
        wrench[env_idx, body_idx, WRENCH_ANGULAR : WRENCH_ANGULAR + 3] += torque

    def body_com_pos_w(self, body_id: int) -> torch.Tensor:
        return self._robot.data.body_com_pos_w.torch[:, body_id]

    # ------------------------------------------------------------------ derived

    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity direction in the pelvis frame. `(0, 0, -1)` upright — the contract's slice [0:3].

        Replaces the frozen `projected_gravity_b`, which reads exactly `(0, 0, -1)` for a body lying
        on the floor and would pay an `upright` reward of 1.000 for a total failure.
        """
        return quat_rotate_inverse(self.root_quat_w, _gravity_dir(self.root_quat_w))

    def up_axis_w(self) -> torch.Tensor:
        """The pelvis body's own up axis, expressed in world. `(0, 0, 1)` when perfectly upright.

        The third column of the rotation matrix for a wxyz quaternion. Local +Z is the body's up
        axis because the D6 builder keeps bodies in conventional USD Z-up - the Godot-to-USD
        rotation rides on the JOINT frames, not the bodies. Same convention `upright()` uses.
        """
        q = self.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack(
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1
        )

    def upright(self) -> torch.Tensor:
        """+1 standing, 0 on its side, -1 inverted.

        The pelvis body's local +Z expressed in world. The D6 builder keeps bodies in conventional
        USD Z-up — the Godot-to-USD rotation rides on the JOINT frames — so local +Z is the body's
        up axis.
        """
        q = self.root_quat_w
        x, y = q[:, 1], q[:, 2]
        return 1.0 - 2.0 * (x * x + y * y)


def _gravity_dir(like: torch.Tensor) -> torch.Tensor:
    g = torch.zeros(like.shape[0], 3, device=like.device, dtype=like.dtype)
    g[:, 2] = -1.0
    return g


def quat_rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` into the frame described by `quat_wxyz`.

    Local rather than `isaaclab.utils.math.quat_rotate_inverse` only so this module has no opinion
    about which quaternion order that helper expects this release; the inputs here are known-wxyz
    because they come from `body_quat_w`.
    """
    w = quat_wxyz[:, 0:1]
    xyz = quat_wxyz[:, 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)


def yaw_only(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """The heading component of a rotation, as a wxyz quaternion.

    Used for the heading-relative linear velocity in observation slice [3:6], so the policy reads
    "forward" rather than a world axis.
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    out = torch.zeros_like(quat_wxyz)
    out[:, 0] = torch.cos(half)
    out[:, 3] = torch.sin(half)
    return out
