"""Dump a rest-pose reference observation and its policy action, for Godot parity testing.

The Isaac side of the transfer check, ported from the 2.3.2 track. `obs_action_contract.md` is a
document, and this project has already learned that a document can be confidently wrong in three
separate places at once — a permuted action order, an inverted roll sign and an undocumented DOF
ordering all shipped in it while every trained policy behaved perfectly, because Isaac reads the rig
contract rather than the prose. None of those defects would have surfaced as an error in Godot
either: they produce a body that flails, which reads as "the policy didn't transfer".

So rather than trusting either implementation, this writes down what Isaac actually computes at a
known pose, and Godot is checked against it slot by slot. At rest the expectations are sharp and
independently knowable — gravity (0,0,-1), pelvis height 0.82, every joint angle ~0 — so a mismatch
localises to a specific slice instead of being a vague behavioural difference.

**Two things differ from the 2.3.2 dumper**, and both are consequences of the solver:

* The DOF ordering written out is **`newton_dof_order`**, read live from the articulation. It
  differs from `physx_dof_order` in 42 of 45 slots, and a Godot side using the wrong one permutes
  90 of the 143 floats invisibly.
* The observation is built through `NewtonRigState`, because `joint_pos`, `joint_vel` and
  `projected_gravity_b` are frozen under XPBD and read a plausible zero.

    python isaac_lab_3/scripts/dump_reference.py --checkpoint <abs path to model_N.pt>

Writes `isaac_lab_3/assets/reference_obs_newton.json`.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium
from p4f_newton.assets import ACTUATED_JOINTS  # noqa: E402

ISAAC3_ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS_DIR = ISAAC3_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=pathlib.Path, default=ASSETS_DIR / "reference_obs_newton.json")
    p.add_argument(
        "--settle_steps",
        type=int,
        default=0,
        help="Policy steps to run before capturing. 0 captures the spawn pose, which is the one "
        "Godot can reproduce exactly; anything higher diverges between engines and is only useful "
        "for eyeballing.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 2
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    base = wrapped.unwrapped
    obs, _ = wrapped.reset()
    for _ in range(args.settle_steps):
        with torch.no_grad():
            obs = wrapped.step(policy(obs))[0]

    with torch.no_grad():
        action = policy(obs)

    vec = obs[0].detach().cpu().tolist()
    act = action[0].detach().cpu().tolist()

    payload = {
        "task": args.task,
        "engine": "isaac-lab-3 / newton / xpbd",
        "checkpoint": str(pathlib.Path(args.checkpoint).name),
        "settle_steps": args.settle_steps,
        "solver_iterations": env_cfg.sim.physics.solver_cfg.iterations,
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
        #
        # `newton_dof_order`, NOT `physx_dof_order` - see the module docstring.
        "newton_dof_order": list(base.robot.joint_names),
        "actuated_joint_names": list(ACTUATED_JOINTS),
        "contact_bones": [base.robot.body_names[i] for i in base._contact_ids],
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

    def plain(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=plain) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.output}")
    env.close()


if __name__ == "__main__":
    main()
