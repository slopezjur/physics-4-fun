"""Feasibility gate: can Newton's XPBD solver hold the Godot dummy's rest pose?

This is the cheapest possible test of the one hypothesis the 2.3.2 track never got to try, and it
is designed to be killable in ten minutes. Read `isaac_lab/HANDOFF.md` §1 and §5 first.

**The hypothesis.** Seven attempts failed to transfer an Isaac-trained policy into Godot, and the
diagnosis was a solver-CLASS mismatch: Isaac/PhysX solves a reduced-coordinate articulation, where
joint angles ARE the state and a joint cannot come apart; Godot/Jolt solves maximal-coordinate
constraints, where each body carries a full 6-DOF pose and joints are satisfied approximately, by
iteration, so they sag under load. No parameter range crosses that boundary, because Godot is not a
point in the space PhysX parameterises — it is a different model.

XPBD is position-based dynamics: the same maximal-coordinate family Jolt is in. If the trainer and
the game engine are in the same family, NVIDIA's own result says transfer needs nothing but joint
reordering, which is already done and verified exact (0 of 45 differ on the D6 rig).

**The gate is the zero-action hold, and nothing downstream is worth measuring until it passes.**
Commanding all-zero actions is defined by the contract as commanding the rest pose. Measured on the
same body from the same pose:

    Isaac 2.3.2 / PhysX, URDF rig    98.4% still standing at 8 s
    Isaac 2.3.2 / PhysX, D6 rig      ~84% still standing at 8 s
    Godot / Jolt                     0% — on the floor in under 2 s

Godot is the number XPBD has to reproduce. A rest pose that holds like PhysX's means XPBD is
solving the articulation the way PhysX does and the gap is unchanged. A rest pose that collapses
like Godot's is the first evidence in this project that the two engines agree about statics, and
everything already built — the D6 rig, the 143/36 contract, the parity harness, the Godot-side
runtime — is waiting behind it.

**Read the structural check first.** If XPBD silently drops the D6 joints or reports a different
DOF count, the hold measurement is describing a different machine and means nothing.

**One known gap, measured rather than assumed.** `SolverXPBD`'s own docstring lists
`joint_effort_limit`, `joint_velocity_limit` and `joint_armature` as NOT supported. The per-joint
torque ceilings are part of the Godot contract, so if they do not bind, the body may hold its pose
by bracing at torque Godot could never produce. The probe reports peak applied torque against the
contract's ceilings so that failure is visible rather than flattering.

**No `AppLauncher` here.** Isaac Lab 3 with a Newton solver runs kit-less — no Isaac Sim, no Kit, no
RTX, no Vulkan — and `AppLauncher` is the Kit entry point: it reads `EXP_PATH` and fails with a bare
`KeyError` when Kit is not installed. `SimulationContext` on a fresh stage is the whole runtime.
This also removes the Windows teardown deadlock that forced `os._exit(0)` on the 2.3.2 scripts.

    python isaac_lab_3/scripts/probe_xpbd.py
    python isaac_lab_3/scripts/probe_xpbd.py --solver mjwarp        # reduced-coordinate control
    python isaac_lab_3/scripts/probe_xpbd.py --iterations 2 4 8 16  # sweep solver effort
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import isaaclab.sim as sim_utils
import newton
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, XPBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from p4f_newton.assets import DUMMY_CFG, RIG, TOTAL_MASS  # noqa: E402

# The same bar the Godot arena, the 2.3.2 evaluators and `UprightTermination` all use. Kept
# identical so this number is comparable to every figure in HANDOFF.md without a conversion.
STANDING_HEAD_HEIGHT = 1.35
STANDING_UPRIGHT = 0.86

# What the D6 rig must report. From `build_d6_usd.py`, which asserts them at build time.
EXPECT_BODIES = 16
EXPECT_JOINTS = 15
EXPECT_DOF = 45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["xpbd", "mjwarp"],
        help="xpbd is the hypothesis; mjwarp is the reduced-coordinate control that should behave "
        "like the 2.3.2 PhysX result and confirm the rig itself is fine.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[2],
        help="XPBD solver iterations to sweep. The direct analogue of Jolt's position/velocity "
        "steps, which measurably changed how gracefully the Godot body degraded (2/10 -> 16/30 "
        "moved the zero-action head height at t=6 from 0.211 to 0.608) without ever making the "
        "rest pose stable.",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="Hold duration. 8 s matches the 2.3.2 probe.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Physics step. 1/120 matches both engines.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


args = parse_args()


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    """Ground plus the dummy. No lights — nothing renders."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = DUMMY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def build_physics_cfg(iterations: int) -> NewtonCfg:
    if args.solver == "xpbd":
        return NewtonCfg(solver_cfg=XPBDSolverCfg(iterations=iterations), num_substeps=1)
    return NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)


def check_structure(robot) -> bool:
    """Did XPBD keep the D6 joints, or quietly solve a different machine?

    A mismatch here invalidates the hold measurement entirely, so it runs first and its result is
    printed whether it passes or fails.
    """
    bodies, joints = robot.num_bodies, robot.num_joints
    dof = robot.data.joint_pos.torch.shape[1]
    mass = robot.data.default_mass.torch[0].sum().item()

    ok = True
    print(f"  {'quantity':<12}{'measured':>12}{'expected':>12}")
    for name, got, want in (("bodies", bodies, EXPECT_BODIES), ("DOF", dof, EXPECT_DOF)):
        good = got == want
        ok &= good
        print(f"  {name:<12}{got:>12}{want:>12}   {'ok' if good else 'MISMATCH'}")

    # `num_joints` counts articulation joints, which some backends report as DOF. Accept either
    # reading rather than failing the gate on a naming convention.
    joints_ok = joints in (EXPECT_JOINTS, EXPECT_DOF)
    ok &= joints_ok
    print(f"  {'joints':<12}{joints:>12}{EXPECT_JOINTS:>12}   {'ok' if joints_ok else 'MISMATCH'}")

    mass_ok = abs(mass - TOTAL_MASS) < 0.05
    ok &= mass_ok
    print(f"  {'mass (kg)':<12}{mass:>12.2f}{TOTAL_MASS:>12.2f}   {'ok' if mass_ok else 'MISMATCH'}")

    # Joint ordering by NAME, not by number. At the rest pose every joint sits near zero, so a
    # permutation is invisible to any numeric comparison — this is the defect the parity harness
    # was built to catch on the Godot side.
    contract_order = RIG["physx_dof_order"]
    live_order = list(robot.joint_names)
    if len(live_order) == len(contract_order):
        differing = sum(1 for a, b in zip(live_order, contract_order) if a != b)
        print(f"  {'DOF order':<12}{differing:>12} of {len(contract_order)} differ from physx_dof_order")
        if differing:
            print("               (expected under a new solver — record the new order, never "
                  "reorder the contract to match)")
    else:
        print(f"  {'DOF order':<12}{'n/a':>12}   length mismatch, cannot compare")
    return ok


class JointAngleReader:
    """Recovers joint angles from body poses, because XPBD does not maintain them.

    **`robot.data.joint_pos` is unusable under XPBD and reads a convincing exactly-zero.** That is
    not a bug; it is what a maximal-coordinate solver *is*. XPBD integrates body poses directly and
    treats joints as constraints between them, so generalized joint coordinates are never part of
    its state. A reduced-coordinate solver — PhysX, MuJoCo, Featherstone — carries joint angle AS
    the state variable, which is exactly the distinction this whole track is testing for.

    So the read is real evidence, not an obstacle: a solver that cannot tell you its joint angles
    without an inverse-kinematics pass is in Jolt's family, not PhysX's. Godot derives joint angles
    the same way, by decomposing bone-local rotations.

    The danger is that zero looks like a healthy, perfectly-held rest pose. Every joint reads 0.0000
    while the body folds up and hits the floor, and the effort telemetry agrees, so a run reports a
    rigid body that never moved a joint. `newton.eval_ik` recovers the true coordinates from
    `State.body_q`, which IS updated.
    """

    # Coordinates per environment: the root FREE joint carries 7 (position + quaternion), then one
    # per rig DOF. `joint_q` is a single flat array over every environment, so these blocks repeat.
    ROOT_COORDS = 7

    def __init__(self, num_envs: int) -> None:
        model = NewtonManager._model
        self._model = model
        self._num_envs = num_envs
        self._joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=model.device)
        self._joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)

        self._per_env = model.joint_coord_count // num_envs
        expected = self.ROOT_COORDS + EXPECT_DOF
        if self._per_env != expected or model.joint_coord_count % num_envs:
            raise RuntimeError(
                f"joint_coord_count {model.joint_coord_count} is not {num_envs} x {expected}; "
                "the per-env coordinate layout changed and the slice below would read the wrong DOF"
            )

    def angles(self) -> torch.Tensor:
        """Every rig joint's angle in radians, shaped (num_envs, 45).

        Stripping only the leading 7 coordinates is the trap here: it removes environment 0's root
        joint and leaves every OTHER environment's root in the result. Those carry world positions,
        so the array picks up values like 10.5 rad against joints limited to +/-0.5, and the mean
        drifts with `--num_envs` and `env_spacing` rather than with the physics.
        """
        newton.eval_ik(self._model, NewtonManager._state_0, self._joint_q, self._joint_qd)
        per_env = wp.to_torch(self._joint_q).view(self._num_envs, self._per_env)
        return per_env[:, self.ROOT_COORDS :]


def upright_from_pelvis(robot, pelvis_id: int) -> torch.Tensor:
    """How upright the pelvis is: +1 standing, 0 on its side, -1 inverted.

    Deliberately NOT `projected_gravity_b`. That is derived from the articulation's root quaternion,
    which Isaac Lab does not refresh under XPBD — measured, it reports a rock-steady 1.000 for a
    body whose head is at 0.140 m, because the underlying `root_quat_w` never leaves identity while
    Newton's own `State.body_q` shows 82 degrees of pitch. Reading the pelvis BODY's quaternion
    avoids the stale root entirely.

    The body's local +Z is its up axis (the D6 builder keeps bodies in conventional USD Z-up), so
    the world-frame z component of that axis is the uprightness.
    """
    quat = robot.data.body_quat_w.torch[:, pelvis_id]  # wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Third component of the rotated basis vector (0, 0, 1): 1 - 2(x^2 + y^2).
    return 1.0 - 2.0 * (x * x + y * y)


def reset_to_rest_pose(scene: InteractiveScene) -> None:
    """Place every body at its authored rest pose.

    `sim.reset()` starts the simulation; it does NOT write an articulation's default root state.
    Without this the root sits at its env origin — z = 0 rather than the rest pelvis height — so the
    dummy spawns buried in the ground plane, the solver resolves the overlap explosively, and the
    hold measures a body being ejected from the floor rather than a body standing on it. It reads as
    a collapse, which is exactly the answer the gate is looking for, so getting this wrong would have
    produced a convincing false positive.
    """
    robot = scene["robot"]

    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())

    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())

    scene.reset()
    scene.write_data_to_sim()
    scene.update(args.dt)


def report_spawn_state(scene: InteractiveScene) -> None:
    """Is the body actually standing before the clock starts?

    Invariant #2 of the project's own session rules: verify the physics before spending time on
    anything downstream. A rest pose that is already wrong makes every later row meaningless.
    """
    robot = scene["robot"]
    head_id = robot.body_names.index("Head")
    pelvis_id = robot.body_names.index("Pelvis")
    data = robot.data

    head = (data.body_com_pos_w.torch[:, head_id, 2] - scene.env_origins[:, 2]).mean().item()
    # Pelvis height comes from the BODY, not from `root_pos_w`. Measured: `root_pos_w` reports
    # 1.640 for a pelvis that every other reading — and the physics — puts at 0.820, i.e. it adds
    # the articulation's authored root offset on top of the free joint's own translation. It reads
    # as a plausible height, never errors, and is exactly double.
    pelvis = (data.body_com_pos_w.torch[:, pelvis_id, 2] - scene.env_origins[:, 2]).mean().item()
    upright = upright_from_pelvis(robot, pelvis_id)[0].item()


    print(f"  head           {head:>8.3f} m   (Godot rest 1.540)")
    print(f"  pelvis         {pelvis:>8.3f} m   (Godot rest 0.820)")
    print(f"  upright        {upright:>8.3f}     (standing is +1.000)")




def hold(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Command the rest pose for `--seconds` and watch whether the body stays up."""
    robot = scene["robot"]
    head_id = robot.body_names.index("Head")
    pelvis_id = robot.body_names.index("Pelvis")

    # Zero action is the rest pose, by the contract's definition. Commanding the articulation's own
    # default joint positions is exactly that, and it reads the pose from the rig rather than
    # assuming zero is the neutral angle.
    target = robot.data.default_joint_pos.torch.clone()

    # The rest-pose distance from pelvis to head, in 3D. This is the column that separates the two
    # ways a body can end up on the floor, and they mean opposite things:
    #
    #   * a RAGDOLL collapse — joints sag under load, the chain folds, and this distance SHRINKS.
    #     That is Godot's failure and the one this whole track is trying to reproduce.
    #   * a RIGID topple — the joints never move and the whole assembly tips over as one piece.
    #     This distance stays constant while head height still drops to the floor.
    #
    # Head height alone cannot tell them apart: both end with the head at ~0.14 m. Without this
    # column a locked articulation reads as a perfect reproduction of the Godot result.
    rest_span = torch.linalg.vector_norm(
        robot.data.body_com_pos_w.torch[:, head_id] - robot.data.body_com_pos_w.torch[:, pelvis_id],
        dim=-1,
    ).mean().item()

    steps = int(args.seconds / args.dt)
    report_every = max(1, int(0.5 / args.dt))
    peak_torque = torch.zeros_like(target)
    joints = JointAngleReader(scene.num_envs)
    peak_joint = 0.0

    print(f"  pelvis->head span at rest: {rest_span:.3f} m  (constant => rigid topple, not a collapse)")
    print(
        f"\n  {'t (s)':>7}{'head (m)':>11}{'pelvis (m)':>12}{'span (m)':>11}"
        f"{'mean|q|':>10}{'max|q|':>9}{'standing':>11}"
    )
    for step in range(steps):
        robot.set_joint_position_target_index(target=target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(args.dt)

        peak_torque = torch.maximum(peak_torque, robot.data.applied_torque.torch.abs())

        if step % report_every == 0 or step == steps - 1:
            data = robot.data
            com = data.body_com_pos_w.torch
            head = com[:, head_id, 2] - scene.env_origins[:, 2]
            pelvis = com[:, pelvis_id, 2] - scene.env_origins[:, 2]
            span = torch.linalg.vector_norm(com[:, head_id] - com[:, pelvis_id], dim=-1)
            # Uprightness from the pelvis body's own orientation. `projected_gravity_b` is frozen
            # under XPBD for the same reason `joint_pos` is — it is derived from a root quaternion
            # Isaac Lab never refreshes — and reads 1.000 for a body lying on the floor.
            upright = upright_from_pelvis(robot, pelvis_id)
            q = joints.angles().abs()
            peak_joint = max(peak_joint, q.max().item())
            standing = ((head >= STANDING_HEAD_HEIGHT) & (upright >= STANDING_UPRIGHT)).float().mean().item()
            print(
                f"  {step * args.dt:>7.2f}{head.mean().item():>11.3f}{pelvis.mean().item():>12.3f}"
                f"{span.mean().item():>11.3f}{q.mean().item():>10.4f}{q.max().item():>9.4f}"
                f"{standing * 100.0:>10.1f}%"
            )

    # A joint set that never leaves zero is not a soft ragdoll — it is a locked one, and every
    # number above then describes a rigid body falling over rather than an articulation sagging.
    if peak_joint < 1e-4:
        print(
            f"\n  WARNING: max |joint angle| over the whole run was {peak_joint:.2e}. No joint ever"
            "\n  moved. Whatever fell above did so as a RIGID body — this is not a ragdoll collapse"
            "\n  and must not be read as reproducing Godot's."
        )

    # Does the effort limit bind? XPBD says it does not support one. If peak torque sits above the
    # contract's ceiling, the body is being held up by torque Godot cannot produce, and any
    # apparent success here is measuring a stronger machine.
    ceilings = torch.tensor(
        [RIG["joints"][name]["effort"] for name in robot.joint_names],
        device=peak_torque.device,
        dtype=peak_torque.dtype,
    )
    if peak_torque.max().item() <= 0.0:
        # Same class of stale buffer as `joint_pos` and `projected_gravity_b`: Isaac Lab never
        # refreshes it under XPBD, so it reads a clean zero rather than erroring. Say so, instead of
        # printing "effort limits bind" — which is what a zero here would otherwise imply, and would
        # be an unearned reassurance about the one parameter XPBD documents as unsupported.
        print(
            "\n  applied_torque read exactly 0 for the whole run — the buffer is not refreshed under"
            "\n  XPBD, so effort-limit binding is UNMEASURED here, not confirmed. Newton's model does"
            f"\n  carry joint_effort_limit = {ceilings.min().item():.0f}..{ceilings.max().item():.0f} N.m."
        )
        return

    ratio = peak_torque.max(dim=0).values / ceilings
    over = int((ratio > 1.01).sum().item())
    print(
        f"\n  peak |torque| vs contract ceiling: max {ratio.max().item():.2f}x, "
        f"{over} of {len(ceilings)} DOF above ceiling"
    )
    if over:
        print("  -> effort limits are NOT binding. Expected for XPBD; the hold above is optimistic.")
    else:
        print("  -> effort limits bind, or the pose never demanded that much torque.")


def main() -> None:
    for iterations in args.iterations:
        label = args.solver + (f", iterations={iterations}" if args.solver == "xpbd" else "")
        print("=" * 68)
        print(f"  XPBD FEASIBILITY GATE - {label}, {args.num_envs} bodies, dt={args.dt:.5f}")
        print("=" * 68)

        sim_utils.create_new_stage()
        sim_cfg = sim_utils.SimulationCfg(device=args.device, dt=args.dt, physics=build_physics_cfg(iterations))
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(ProbeSceneCfg(num_envs=args.num_envs, env_spacing=3.0))
        sim.reset()

        print("\n  -- structure --")
        if not check_structure(scene["robot"]):
            print("\n  STRUCTURE MISMATCH. The hold below would describe a different machine.")

        print("\n  -- spawn state --")
        reset_to_rest_pose(scene)
        report_spawn_state(scene)

        print("\n  -- zero-action hold --")
        hold(sim, scene)

        sim.clear_instance()
        print()


if __name__ == "__main__":
    main()
