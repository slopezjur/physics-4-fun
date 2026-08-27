"""Dump a rest-pose reference observation and its policy action, for Godot parity testing.

The Isaac side of the transfer check. `obs_action_contract.md` is a document, and this project has
already learned that a document can be confidently wrong in three separate places at once - a
permuted action order, an inverted roll sign and an undocumented DOF ordering all shipped in it
while every trained policy behaved perfectly, because Isaac reads the rig contract rather than the
prose. None of those defects would have surfaced as an error in Godot either: they produce a body
that flails, which reads as "the policy didn't transfer".

So rather than trusting either implementation, this writes down what Isaac actually computes at a
known pose, and Godot is checked against it slot by slot. At rest the expectations are sharp and
independently knowable - gravity (0,0,-1), pelvis height 0.82, every joint angle ~0 - so a
mismatch localises to a specific slice instead of being a vague behavioural difference.

    python isaac_lab/scripts/dump_reference.py --checkpoint <path> [--task stand]

Writes `isaac_lab/assets/reference_obs.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="stand", choices=["stand", "walk", "perturb", "run"])
parser.add_argument("--output", type=pathlib.Path, default=ASSETS_DIR / "reference_obs.json")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=0,
    help="Policy steps to run before capturing. 0 captures the spawn pose, which is the one Godot "
    "can reproduce exactly; anything higher diverges between engines and is only useful for eyeballing.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_isaac.tasks  # noqa: F401, E402

TASKS = {
    "stand": ("P4F-Dummy-Stand-Direct-v0", "stand.stand_env_cfg", "StandEnvCfg", "stand.agents.rsl_rl_ppo_cfg", "StandPPORunnerCfg"),
    "walk": ("P4F-Dummy-Walk-Direct-v0", "walk.walk_env_cfg", "WalkEnvCfg", "walk.agents.rsl_rl_ppo_cfg", "WalkPPORunnerCfg"),
    "perturb": ("P4F-Dummy-Perturb-Direct-v0", "perturb.perturb_env_cfg", "PerturbEnvCfg", "perturb.agents.rsl_rl_ppo_cfg", "PerturbPPORunnerCfg"),
    "run": ("P4F-Dummy-Run-Direct-v0", "run.run_env_cfg", "RunEnvCfg", "run.agents.rsl_rl_ppo_cfg", "RunPPORunnerCfg"),
}


def main() -> None:
    import importlib

    task_id, cfg_mod, cfg_cls, agent_mod, agent_cls = TASKS[args.task]
    env_cfg = getattr(importlib.import_module(f"p4f_isaac.tasks.{cfg_mod}"), cfg_cls)()
    env_cfg.scene.num_envs = 1

    agent_cfg = getattr(importlib.import_module(f"p4f_isaac.tasks.{agent_mod}"), agent_cls)()
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.robot

    def tensor_of(o):
        """`get_observations` returns a bare tensor, a (obs, extras) tuple, a {"policy": ...} dict
        or a TensorDict, depending on the wrapper stack version. Normalise rather than assuming.

        TensorDict is the one that bites: it is mapping-like but does NOT pass `isinstance(o, dict)`,
        so a dict check alone falls through and returns the TensorDict itself. Indexing that yields
        another TensorDict rather than a row, and `.tolist()` on it yields a dict - which surfaces
        forty lines later as `KeyError: 0` on what looked like a plain list.
        """
        if isinstance(o, (tuple, list)):
            return tensor_of(o[0])
        if hasattr(o, "keys") and not torch.is_tensor(o):
            keys = list(o.keys())
            return o["policy"] if "policy" in keys else o[keys[0]]
        return o

    obs = env.get_observations()
    for _ in range(args.settle_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))

    with torch.inference_mode():
        action = policy(obs)

    vec = tensor_of(obs)[0].detach().cpu().tolist()
    act = action[0].detach().cpu().tolist()

    payload = {
        "task": args.task,
        "checkpoint": str(pathlib.Path(args.checkpoint).name),
        "settle_steps": args.settle_steps,
        # Written alongside the vectors so a Godot-side mismatch can be localised to a slice by
        # name rather than by counting indices by hand.
        "layout": {
            "projected_gravity": [0, 3],
            "root_lin_vel_yaw": [3, 6],
            "root_ang_vel_b": [6, 9],
            "pelvis_height": [9, 10],
            "joint_pos": [10, 55],
            "joint_vel": [55, 100],
            "contacts": [100, 104],
            "prev_action": [104, 140],
            "command": [140, 143],
        },
        # The authoritative orderings, restated here so the reference file is self-contained: a
        # Godot test reading this does not also have to resolve the rig contract to interpret it.
        "physx_dof_order": list(robot.joint_names),
        "actuated_joints": list(base._actuated_ids.detach().cpu().numpy().astype(int).tolist()),
        "actuated_joint_names": [robot.joint_names[i] for i in base._actuated_ids.detach().cpu().tolist()],
        "contact_bones": [robot.body_names[i] for i in base._contact_ids],
        "observation": vec,
        "action": act,
    }

    # Printed BEFORE the write: a serialisation failure here still leaves the numbers on screen,
    # which is the whole reason the run was worth its two minutes.
    print(f"  gravity      : {vec[0]:+.4f} {vec[1]:+.4f} {vec[2]:+.4f}   (upright expects 0, 0, -1)")
    print(f"  pelvisHeight : {vec[9]:.4f}                        (rest expects 0.82)")
    print(f"  maxJointPos  : {max(abs(v) for v in vec[10:55]):.4f}")
    print(f"  maxJointVel  : {max(abs(v) for v in vec[55:100]):.4f}")
    print(f"  contacts     : {vec[100:104]}   (Hand_L, Hand_R, Foot_L, Foot_R)")
    print(f"  maxAction    : {max(abs(v) for v in act):.4f}")

    # Several of the fields above come straight off the env and can still be torch tensors or numpy
    # scalars depending on how the attribute happens to be stored; coerce rather than enumerating
    # which ones, so adding a field later cannot reintroduce the failure.
    def plain(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=plain) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
