"""Which rotation decomposition do PhysX's D6 joint DOF correspond to?

The last named structural difference between the engines. Godot's `IsaacObservation` recovers a
joint's three angles by taking the child's rotation relative to its parent and decomposing it as
XYZ Euler. PhysX reports three DOF per D6 joint, but nothing documents which decomposition those
DOF ARE - twist-swing and the six Euler orders all produce different per-axis numbers for the same
physical rotation.

The rest pose cannot tell them apart, because every angle there is near zero and every convention
agrees at zero. Neither can a single-axis rotation. Only a COMBINED rotation discriminates, which
is what this commands: three distinct angles on one joint at once, then read back the resulting
relative rotation and ask which convention reproduces the commanded triple.

Joint state is written directly rather than driven to a target, so there is no PD tracking error
between what was commanded and what is measured.

    P4F_RIG=d6 python isaac_lab/scripts/probe_dof_convention.py
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_isaac.tasks  # noqa: F401, E402
from p4f_isaac.tasks.stand.stand_env_cfg import StandEnvCfg  # noqa: E402

# The joint to probe, and three deliberately distinct angles. Distinct matters: equal angles would
# make several conventions agree by coincidence.
PROBE_JOINT = "Spine"
PROBE_ANGLES = (0.30, -0.20, 0.45)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.tensor([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q: torch.Tensor) -> list[list[float]]:
    w, x, y, z = [float(v) for v in q]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def axis_matrix(axis: str, angle: float) -> list[list[float]]:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == "y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_diff(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))


def main() -> None:
    # Read matched snapshots from the running env rather than trying to command and hold a pose.
    #
    # Two earlier attempts failed: writing joint state without stepping leaves `body_quat_w` stale
    # (the deviation quaternion differed between identical runs), and a bare SimulationContext
    # hangs on this machine the same way convert_asset.py does. Neither is needed - any moment of
    # ordinary motion provides a (joint_pos, relative rotation) PAIR, and the convention is
    # whatever maps one onto the other. Random actions just spread those pairs away from rest,
    # where every convention agrees and nothing is discriminated.
    cfg = StandEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("P4F-Dummy-Stand-Direct-v0", cfg=cfg)
    env.reset()
    base = env.unwrapped
    robot = base.robot

    names = list(robot.joint_names)
    dof = [names.index(f"joint_{PROBE_JOINT}:{i}") for i in range(3)]
    body_names = list(robot.body_names)
    child_id = body_names.index(PROBE_JOINT)
    parent_id = body_names.index("Pelvis")

    def relative_quat() -> torch.Tensor:
        q = robot.data.body_quat_w[0]
        return quat_mul(quat_conj(q[parent_id].cpu()), q[child_id].cpu())

    rest = relative_quat()

    # Drive the body well away from rest so the three angles are large and distinct.
    torch.manual_seed(0)
    action = torch.zeros(1, base.cfg.action_space, device=base.device)
    for i, d in enumerate(dof):
        action[0, names.index(f"joint_{PROBE_JOINT}:{i}") % base.cfg.action_space] = 0.0
    for step in range(200):
        if step % 40 == 0:
            action = (torch.rand(1, base.cfg.action_space, device=base.device) * 2.0 - 1.0)
        with torch.inference_mode():
            env.step(action)

    measured = [float(robot.data.joint_pos[0, d]) for d in dof]
    current = relative_quat()
    deviation_world = quat_mul(quat_conj(rest), current)

    # The DOF are about the JOINT frame's axes, not the world's. build_d6_usd.py gives every joint
    # frame the Godot-to-USD rotation, so the deviation has to be conjugated into that frame before
    # any decomposition means anything. Skipping this defeats every Euler order at once and looks
    # exactly like "the convention is twist-swing" - which is what it did on the first run here.
    frame = torch.tensor([-0.5, -0.5, 0.5, 0.5])  # joint_frame_rotation(), (w, x, y, z)
    deviation = quat_mul(quat_conj(frame), quat_mul(deviation_world, frame))


    print("\n" + "=" * 68)
    print(f"  DOF CONVENTION PROBE - joint_{PROBE_JOINT}")
    print("=" * 68)
    print(f"measured   : {tuple(round(v, 4) for v in measured)}")
    print(f"deviation q: w={deviation[0]:+.5f} x={deviation[1]:+.5f} "
          f"y={deviation[2]:+.5f} z={deviation[3]:+.5f}")

    target = quat_to_matrix(deviation)

    # Every intrinsic Euler order, each with every sign assignment. The sign sweep matters because
    # the joint frame's axes may point opposite to the body axes the angle is measured about.
    print(f"\n{'order':>8}{'signs':>10}{'max error':>14}")
    results = []
    for order in itertools.permutations("xyz"):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for axis, sign, angle in zip(order, signs, measured):
                m = mat_mul(m, axis_matrix(axis, sign * angle))
            results.append((mat_diff(m, target), "".join(order), signs))

    results.sort()
    for err, order, signs in results[:5]:
        sign_str = "".join("+" if s > 0 else "-" for s in signs)
        print(f"{order:>8}{sign_str:>10}{err:>14.6f}")

    # Twist-swing, which is what PhysX articulations actually use for a spherical joint: a twist
    # about X composed with a swing about an axis lying in the YZ plane. It is not expressible as
    # any Euler order, which is why the sweep above cannot match it.
    def axis_angle(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
        x, y, z = axis
        c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
        return [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]

    print(f"\n{'twist-swing variant':>26}{'max error':>14}")
    ts_results = []
    for twist_first in (True, False):
        for sign in (1, -1):
            a0, a1, a2 = (sign * a for a in measured)
            twist = axis_angle((1.0, 0.0, 0.0), a0)
            mag = math.hypot(a1, a2)
            swing = (
                axis_angle((0.0, a1 / mag, a2 / mag), mag)
                if mag > 1e-9
                else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )
            m = mat_mul(twist, swing) if twist_first else mat_mul(swing, twist)
            label = f"{'twist*swing' if twist_first else 'swing*twist'} {'+' if sign > 0 else '-'}"
            ts_results.append((mat_diff(m, target), label))

    ts_results.sort()
    for err, label in ts_results:
        print(f"{label:>26}{err:>14.6f}")

    if ts_results[0][0] < results[0][0]:
        results = [(ts_results[0][0], ts_results[0][1], ())]

    best_err, best_order, best_signs = results[0]
    print()
    if best_err < 1e-3:
        sign_str = "".join("+" if s > 0 else "-" for s in best_signs)
        print(f"[OK] PhysX D6 DOF decompose as intrinsic {best_order.upper()} with signs {sign_str}")
        print(f"     Godot's IsaacObservation must use this, not its current EulerOrder.Xyz +++")
    else:
        print(f"[FAIL] no Euler order matches (best {best_err:.4f}) - likely twist-swing, "
              "which no EulerOrder can express")



if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
