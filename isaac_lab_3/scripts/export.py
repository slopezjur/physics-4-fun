"""Export a trained Newton policy to ONNX, with the contract Godot needs to consume it.

Writes two files side by side into `exported/`:

* **`<brain>_policy.onnx`** — `obs[1,143] -> actions[1,36]`, with the observation normalizer folded
  into the graph. Godot feeds raw observations and must not normalise them itself.
* **`<brain>_policy.contract.json`** — everything about the policy that is not in the graph.

**Named after the BRAIN, not the task** (see `BRAIN`). Stand and Perturb are one brain - `PerturbEnv`
inherits `StandEnv` and reuses its reward dictionary unchanged - so `stand_policy.onnx` and
`perturb_policy.onnx` were two names for interchangeable artifacts. That invented a choice Godot had
to make, and the choice went wrong: a measured-dead perturb export sat in `Models/` dropping the
dummies while the working brain sat beside it. Walk and Run ARE a different brain, so there are two
stems rather than one.

**The contract file is not optional, and this is the part that differs from the 2.3.2 export.**
A policy trained here reads its joint slices in NEWTON's DOF order, which differs from the
`physx_dof_order` recorded in `dummy_rig.json` in **42 of 45 slots**. `Source/RL/Isaac/
IsaacRigContract.cs` reads that file, so a Godot side that keeps using it would feed this network
a permuted observation — and at the rest pose every joint sits near zero, so the permutation is
invisible to any numeric check. It trains and runs perfectly happily and produces a dummy that
flails. That is precisely the defect class `obs_action_contract.md` was wrong about four separate
ways, and why `IsaacParityTest` compares resolved NAMES rather than values.

The Newton order is not restated by hand here either. It is read from the live articulation and
verified causally first: `probe_dof_order` drives one actuated joint at a time with gravity off and
checks the recovered coordinate responds at its own index. 35/36 respond exactly; the one that does
not is a 0.001 rad difference on an axis with almost no range.

    python isaac_lab_3/scripts/export.py --checkpoint <abs path to model_N.pt>
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium
from p4f_newton.assets import ACTUATED_JOINTS  # noqa: E402

ISAAC3_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT = ISAAC3_ROOT.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--checkpoint", type=str, required=True, help="Full path to a model_*.pt.")
    p.add_argument("--name", type=str, default="", help="Output stem. Defaults to the task's short name.")
    p.add_argument(
        "--promote",
        action="store_true",
        help="Also copy the pair into the Godot project's Models/ so the engine can load it.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# Which artifact a task exports into. Tasks sharing a brain share a file, because their policies are
# interchangeable: identical observation and action layout, identical joint order, and - for Stand
# and Perturb - literally the same reward dictionary.
#
# Walk and Run are separate because Walk deletes five of Stand's reward terms and adds four. A
# locomotion policy dropped into a standing scene loads fine and behaves wrongly, which is the same
# silent failure this rename exists to remove.
#
# **Mirrored by `Get-Isaac3Brain` in scripts/config.ps1**, which resolves the same map to find the
# promoted checkpoint. Change one and change the other.
BRAIN = {
    "stand": "balance",
    "perturb": "balance",
    "walk": "locomotion",
    "run": "locomotion",
}


def brain_of(task: str) -> str:
    """Artifact stem for a gym id such as `P4F-Dummy-Stand-Newton-v0`.

    Falls back to the task's own short name rather than guessing a brain: an unmapped task is a new
    one, and quietly filing it under `balance` would overwrite a working policy.
    """
    short = task.split("-")[2].lower()
    return BRAIN.get(short, short)


def run_config(checkpoint: str) -> dict:
    """The env config the checkpoint's own run was trained with, from its `params/env.yaml`.

    **The registry default is not good enough for anything that changes inference.** A run may set
    `action_rate_limit` on the command line; the task default is 0, so exporting from the registry
    writes a contract saying "no limit" for a policy that only works WITH one. Godot then drives it
    with un-limited step commands - a different controller, and one that fails silently rather than
    erroring. Caught exactly that way: a rate-limited policy scored 25.5% with its own limit applied
    and 0.0% without.
    """
    run = pathlib.Path(checkpoint).resolve().parent
    env_yaml = run / "params" / "env.yaml"
    if not env_yaml.is_file():
        print(f"[export] WARNING: no {env_yaml}; falling back to task defaults for the contract.")
        return {}
    import yaml

    with open(env_yaml, encoding="utf-8") as fh:
        return yaml.unsafe_load(fh) or {}


def main() -> None:
    args = parse_args()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Two environments, not one: enough to build the articulation and read the live joint order,
    # small enough to start in seconds.
    env_cfg.scene.num_envs = 2
    env_cfg.sim.device = args.device
    # Anything that changes what the policy expects at inference has to come from the RUN, not the
    # registry default. See run_config().
    trained = run_config(args.checkpoint)
    for field in ("action_rate_limit", "obs_joint_vel_clip", "action_scale"):
        if field in trained:
            setattr(env_cfg, field, trained[field])
            print(f"[export] {field} = {trained[field]}  (from the run's env.yaml)")
    env_cfg.playback = True  # measure the policy, not the observation noise
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)

    base = wrapped.unwrapped
    newton_dof_order = list(base.robot.body_names and base.robot.joint_names)

    stem = args.name or brain_of(args.task)
    out_dir = ISAAC3_ROOT / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_name = f"{stem}_policy.onnx"

    # Exported through the model's own `as_onnx`, NOT `isaaclab_rl.rsl_rl.export_policy_as_onnx`.
    # That helper still assumes the actor is an `nn.Sequential` and dies on
    # `self.actor[0].in_features`; in rsl-rl 5.x the actor is an `MLPModel` wrapping a normalizer
    # and an MLP, which is not subscriptable.
    #
    # Two properties this relies on, both verified below rather than assumed:
    #   * `MLPModel.forward` is DETERMINISTIC by default — `stochastic_output` defaults to False,
    #     so the graph emits the distribution mean. That is what you want in-engine; a sampling
    #     policy would make the dummy jitter differently on every run.
    #   * `obs_normalization=True` puts the empirical normalizer INSIDE the actor, ahead of the
    #     MLP, so it is already folded into the exported graph. Godot feeds raw observations.
    #     Normalising again on either side would hand the policy inputs it has never seen.
    actor = runner.alg.actor
    exportable = actor.as_onnx(verbose=False).cpu().eval()
    dummy_obs = torch.zeros(1, env_cfg.observation_space)
    torch.onnx.export(
        exportable,
        dummy_obs,
        str(out_dir / onnx_name),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    onnx_path = out_dir / onnx_name
    embed_external_data(onnx_path)

    verify(onnx_path, env_cfg)

    contract = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "task": args.task,
        "engine": "isaac-lab-3 / newton / xpbd",
        "solver_iterations": env_cfg.sim.physics.solver_cfg.iterations,
        "observation_size": env_cfg.observation_space,
        "action_size": env_cfg.action_space,
        "action_scale": env_cfg.action_scale,
        "policy_hz": round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        "physics_hz": round(1.0 / env_cfg.sim.dt),
        # **Godot MUST apply this same clip to observation slice [55:100].** Godot's tightly-limited
        # twist axes chatter against their stops at up to 68 rad/s where Isaac's peak across all 45
        # DOF is 7.5; unclipped, 45 of the 143 floats arrive an order of magnitude outside anything
        # training produced, and the baked-in normaliser amplifies that straight into a saturated
        # action. IsaacObservation.JointVelocityClip reads this field.
        "joint_velocity_clip": env_cfg.obs_joint_vel_clip,
        # **Godot MUST apply this same cap**, in action space, once per policy step. Zero means the
        # policy was trained without one. `IsaacActionSpace.ActionRateLimit` reads this field.
        "action_rate_limit": env_cfg.action_rate_limit,
        # The action vector's joint order. Same as the 2.3.2 contract - actions are resolved through
        # find_joints(preserve_order=True) against this list, so it is unchanged by the backend.
        # The 36 joints the action vector drives, in action order. NOT the full 45-DOF list -
        # `robot.joint_names` returns every joint including the 9 passive Head/Hand axes, which
        # would publish a 45-entry action contract for a 36-entry action vector.
        "actuated_joints": list(ACTUATED_JOINTS),
        # THE ONE THAT CHANGED. Observation slices [10:55] and [55:100] are in this order, which is
        # NOT dummy_rig.json's physx_dof_order.
        "newton_dof_order": newton_dof_order,
        "observation_layout": {
            "0:3": "projected gravity, pelvis frame, (0,0,-1) upright",
            "3:6": "root linear velocity, yaw-frame, m/s",
            "6:9": "root angular velocity, pelvis frame, rad/s",
            "9:10": "pelvis height above ground, m (rest 0.82)",
            "10:55": "joint position - default, rad, in newton_dof_order",
            "55:100": "joint velocity, rad/s, in newton_dof_order, clipped to joint_velocity_clip",
            "100:104": "contact flags Hand_L, Hand_R, Foot_L, Foot_R",
            "104:140": "previous action, pre-scaling, in [-1,1]",
            "140:143": "velocity command (vx, vy, yaw_rate), zero for Stand",
        },
        "notes": [
            "Contact flags are derived from body height, not a contact sensor - Newton's contact "
            "reporting does not surface through Isaac Lab's ContactSensor on this backend. See "
            "CONTACT_HEIGHT in stand_env.py.",
            "Uprightness and root velocities come from the PELVIS BODY, not root_* fields, which "
            "are frozen under XPBD.",
        ],
    }
    contract_path = out_dir / f"{stem}_policy.contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    print(f"\n[export] {onnx_path}")
    print(f"[export] {contract_path}")

    if args.promote:
        models = PROJECT_ROOT / "Models"
        models.mkdir(exist_ok=True)
        for src in (onnx_path, contract_path):
            shutil.copy2(src, models / src.name)
        print(f"[export] promoted to {models}")

    env.close()


def embed_external_data(onnx_path: pathlib.Path) -> None:
    """Fold the weights back into the .onnx so it is a single self-contained file.

    **torch's dynamo ONNX exporter writes the tensors to a sidecar `<name>.onnx.data` and leaves
    the .onnx holding only the graph.** Here that is a 13.8 KB .onnx beside a 971 KB .data. The
    pair works in Python, because `onnx.load` follows the reference from the same directory — so
    every check passes locally and nothing looks wrong.

    It breaks the moment the file is moved on its own, which is exactly what shipping it to Godot
    does: `Models/` gets the 13.8 KB graph pointing at a sidecar that is not there. The 2.3.2
    export was a single 950 KB file, so a size check is the cheap tell.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)

    sidecar = onnx_path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()


def verify(onnx_path: pathlib.Path, env_cfg) -> None:
    """Check the graph before anyone trusts it.

    The 2.3.2 track shipped a policy whose raw output had drifted to +/-18 while every quality gate
    read healthy; it was caught only by running the exported graph on a plausible observation and
    looking at the numbers. So that check happens here, at export, rather than being left to
    whoever loads it.
    """
    import numpy as np
    import onnx
    import onnxruntime

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"  graph: {inp.name} {inp.shape} -> {out.name} {out.shape}")

    expected_in, expected_out = env_cfg.observation_space, env_cfg.action_space
    if inp.shape[-1] != expected_in or out.shape[-1] != expected_out:
        raise SystemExit(
            f"  SHAPE MISMATCH: expected [{expected_in}] -> [{expected_out}]. "
            "Godot fails this at load with an ONNX shape error, which is the good case; the bad "
            "case is a width that happens to match and means something else."
        )

    # A rest-pose observation: gravity down, everything else zero. The action mapping defines a=0
    # as the rest pose, so a healthy policy answers with small corrections, not saturated commands.
    obs = np.zeros((1, expected_in), dtype=np.float32)
    obs[0, 2] = -1.0
    obs[0, 9] = 0.82
    actions = sess.run(None, {inp.name: obs})[0]
    lo, hi = float(actions.min()), float(actions.max())
    saturated = int((np.abs(actions) > 1.0).sum())
    print(f"  rest-pose response: range [{lo:+.3f}, {hi:+.3f}], {saturated}/{expected_out} outside [-1,1]")
    if not np.isfinite(actions).all():
        raise SystemExit("  NON-FINITE OUTPUT - do not ship this checkpoint.")
    if saturated > expected_out // 2:
        print("  WARNING: over half the components are outside the clip range. That is the "
              "degenerate bang-bang signature from RESULTS.md; check rew_action_clip.")


if __name__ == "__main__":
    main()
