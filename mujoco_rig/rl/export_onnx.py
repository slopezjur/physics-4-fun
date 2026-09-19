"""Export a trained actor to ONNX, with the contract the C# side needs to feed it.

Godot runs the policy through `MjPolicyDriver`, which builds the observation from the contract
written here. What has caused trouble before is the CONTRACT rather than the network: the Isaac
track's `obs_action_contract` was wrong four separate ways, and the fix was to read the generated
artefact instead of prose. So the observation layout is emitted here, from the environment itself,
next to the weights.

    python mujoco_rig/rl/export_onnx.py --checkpoint logs/mujoco/<run>/model_N.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from env_config import ACTION_LATENCY_STEPS, POLICY_FAMILY  # noqa: E402
from perturb_env import PerturbEnv  # noqa: E402
from ppo import ActorCritic  # noqa: E402
from policy_artifacts import publish_pair  # noqa: E402
from walk_env import WalkEnv  # noqa: E402
from walk_config import HEADING_GAIN  # noqa: E402

# The one-world CPU environment each task's contract is read from.
CONTRACT_ENV = {
    "perturb": lambda: PerturbEnv(num_envs=1, model="dummy_ball.xml"),
    "walk": lambda: WalkEnv(num_envs=1),
}

# How a task fills the command channel, where that is not simply the raw command. For walk, the yaw
# slot is not the raw command when the commanded yaw is zero: the env holds the heading the command
# began on, and a driver that writes the raw 0 instead runs a different policy from the one that was
# scored. Every walk checkpoint trained after 2026-09-10 22:00 was trained with it.
COMMAND_FILL = {
    "perturb": {"source": "zero", "yaw_mode": "raw"},
    "walk": {
        "source": "velocity_command",
        "yaw_mode": "heading_hold",
        "heading_gain": HEADING_GAIN,
        "formula": "yaw = clip(heading_gain * wrap(held - heading, -pi, pi), -1, 1) while the "
                   "commanded yaw is 0; heading = atan2(R[1,0], R[0,0]) of the pelvis",
        "latch": "held = the heading when a zero-yaw command begins; re-latched on any "
                 "command change or reset",
    },
}


class Actor(torch.nn.Module):
    """Deterministic mean action. Exploration noise is a training-time device only."""

    def __init__(self, net):
        super().__init__()
        self.actor = net.actor

    def forward(self, obs):
        return self.actor(obs)


def load_actor(checkpoint):
    """The checkpoint's metadata and its network, ready for inference."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(ck["num_obs"], ck["num_actions"])
    net.load_state_dict(ck["model"])
    net.eval()
    return ck, net


def export_actor(net, num_obs, out):
    """Write the deterministic actor to `out`, weights inline."""
    torch.onnx.export(Actor(net).eval(), torch.zeros(1, num_obs), str(out),
                      input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17)

    # **Inline the weights.** Torch splits tensors into a sibling `.onnx.data` file, and
    # OnnxRuntime resolves that path relative to the PROCESS working directory - so a Godot scene
    # loading the model from a byte buffer throws "file_size: cannot find walk_policy.onnx.data"
    # and the policy silently never loads. The network is ~300 KB; external data buys nothing here.
    external = out.with_suffix(".onnx.data")
    try:
        import onnx
        onnx.save_model(onnx.load(str(out)), str(out), save_as_external_data=False)
        if external.exists():
            external.unlink()
    except ImportError as error:
        raise RuntimeError("onnx is required to inline policy weights for Godot") from error


def observation_layout(env, num_obs):
    """Every observation channel, its offset and its width, in the order the env writes them."""
    channels = [("projected_gravity", 3), ("pelvis_linear_velocity", 3),
                ("pelvis_angular_velocity", 3), ("pelvis_height", 1),
                ("joint_position", env.num_actions),
                ("joint_velocity_scaled_0.1", env.num_actions),
                ("foot_contact_L_R", 2), ("previous_action", env.num_actions),
                # Present for EVERY task. Walk fills it with (vx, vy, yaw) in the pelvis frame;
                # perturb leaves it at zero. One shared layout is what lets a perturb brain seed a
                # walk one directly.
                ("command_vx_vy_yaw", 3)]
    layout, at = [], 0
    for name, width in channels:
        layout.append({"name": name, "offset": at, "width": width})
        at += width
    assert at == num_obs, f"layout {at} != {num_obs}"
    return layout


def action_to_control(env):
    """How the C# side turns a policy output into a control.

    TORQUE is the current plant: the action IS newton-metres, so a zero action is zero muscle and
    the body falls limp. The position form is kept for a model rebuilt with
    build_mjcf.ACTUATOR_MODE = "position".
    """
    if env.torque_mode:
        return {"mode": "torque",
                "formula": "tau_nm = action * force_limit_nm * authority",
                "authority": env.authority,
                "force_limit_nm": [float(x) for x in env.force_limit]}
    return {"mode": "position",
            "formula": "target_rad = action * span * authority, "
                       "span = upper if action >= 0 else -lower",
            "authority": env.authority,
            "lower_rad": [float(x) for x in env.lo],
            "upper_rad": [float(x) for x in env.hi]}


def build_contract(task, checkpoint, ck, env):
    """The contract, generated from the ENVIRONMENT and never hand-written: the Isaac track's
    `obs_action_contract.md` was wrong four separate ways, and the fix was to stop writing prose."""
    if ck["num_obs"] != env.num_obs or ck["num_actions"] != env.num_actions:
        raise ValueError("Checkpoint dimensions differ from the deployment environment")
    return {
        "task": task,
        "source_checkpoint": checkpoint,
        "num_obs": ck["num_obs"],
        "num_actions": ck["num_actions"],
        "policy_hz": round(1.0 / (env.dt * env.decimation), 3),
        "sim_timestep": env.dt,
        "decimation": env.decimation,
        "action_latency_steps": ACTION_LATENCY_STEPS,
        "observation_layout": observation_layout(env, ck["num_obs"]),
        # The POLICY's actuators, not the model's - the neck is actuated but excluded, and a
        # contract listing it would have Godot drive 33 outputs from a 30-wide policy.
        "joint_order": list(env.act_names),
        **({"command": COMMAND_FILL[task]} if task in COMMAND_FILL else {}),
        "action_to_control": action_to_control(env),
        "notes": [
            "Frames are MuJoCo's: Z up. Godot -> MuJoCo is (-z, -x, y).",
            "Pelvis velocities are expressed in the PELVIS frame (R.T @ world).",
            "Foot contact is foot-origin height < 0.05 m, left then right.",
            "The pelvis balance assist must be OFF; this policy replaces it.",
            "Joint order below is mjModel actuator order; read joint angles with "
            "MjBridge.JointPosition, which indexes qpos through jnt_qposadr.",
            "Reset to the model's `rest` keyframe, not to qpos0: at qpos0 the arms hang "
            "inside the legs.",
        ],
    }


def check_parity(net, env, out):
    """ONNX against torch on a real observation, not a zero vector."""
    obs = env.get_observations()
    with torch.no_grad():
        ref = net.actor(obs).numpy()
    import onnxruntime as ort
    got = ort.InferenceSession(str(out)).run(None, {"obs": obs.numpy()})[0]
    if got.shape != ref.shape or not np.isfinite(got).all():
        raise ValueError("Exported policy has invalid output")
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)
    print(f"onnx vs torch max abs diff: {np.abs(got - ref).max():.3e}")


def main() -> int:
    # `torch.onnx.export` prints a U+2705 on success, which a default Windows console (cp1252)
    # cannot encode - the export then dies in the logging call, after the graph is already built.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", choices=("auto",) + tuple(CONTRACT_ENV), default="auto",
                   help="auto reads the task the checkpoint recorded")
    # Deliberately NOT Models/. Promoting a checkpoint into the shipped model set is a decision to
    # be made explicitly (scripts/promote.py), not a side effect of exporting one.
    p.add_argument("--out", default="")
    args = p.parse_args()

    ck, net = load_actor(args.checkpoint)
    task = args.task if args.task != "auto" else ck.get("task", "perturb")
    out = pathlib.Path(args.out or f"mujoco_rig/{POLICY_FAMILY.get(task, task)}_policy.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = CONTRACT_ENV[task]()
    contract = build_contract(task, args.checkpoint, ck, env)
    with tempfile.TemporaryDirectory(prefix="policy_export_", dir=out.parent) as scratch:
        staged = pathlib.Path(scratch) / out.name
        export_actor(net, ck["num_obs"], staged)
        staged.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        check_parity(net, env, staged)
        publish_pair(staged, out)
    cpath = out.with_suffix(".contract.json")
    print(f"wrote {out} and {cpath.name}  ({ck['num_obs']} obs -> {ck['num_actions']} actions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
