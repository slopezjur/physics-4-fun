"""Train a policy under Newton/XPBD with rsl_rl.

Kit-less: no `AppLauncher`, no Isaac Sim, no Vulkan. `SimulationContext` is created by the
environment itself when `gym.make` builds it, so this script only has to register the task and hand
it to rsl_rl.

Normally driven by `train.ps1`, which supplies every argument from `config.ps1`.

    python isaac_lab_3/scripts/train.py --task P4F-Dummy-Stand-Newton-v0 --num_envs 4096 --max_minutes 15
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--iterations", type=int, default=0, help="0 uses the task's own max_iterations.")
    p.add_argument(
        "--max_minutes",
        type=float,
        default=0.0,
        help="Wall-clock cap. Training stops at whichever comes first, this or --iterations. The "
        "run always ends on a checkpoint boundary, so it stays resumable.",
    )
    p.add_argument("--run_name", type=str, default="", help="Suffix appended to the run directory.")
    p.add_argument("--resume", type=str, default="", help="Path to a model_*.pt to continue from.")
    p.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Seed the networks from another task's checkpoint, without its optimizer state.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Override the agent config's experiment_name, i.e. which logs/rsl_rl subtree "
        "this run writes into. Use it to keep a variant off a chained run's checkpoint path: "
        "night.py resumes from the newest checkpoint in its experiment tree, so two runs "
        "sharing a name will silently adopt each other's weights.",
    )
    p.add_argument(
        "--push",
        type=float,
        default=-1.0,
        help="Spawn push in m/s to TRAIN against. -1 keeps the task default, 0 disables it. "
        "Staging matters: effort limits and a shove introduced together took the fall rate "
        "to 100%% and the policy learned nothing, which is the overshoot docs/RL-TRAINING.md "
        "warns about.",
    )
    p.add_argument(
        "--action_rate_limit",
        type=float,
        default=-1.0,
        help="Cap the per-policy-step change in each action component. -1 keeps the task "
        "default (0, disabled). A contract change: a policy trained with it MUST be driven "
        "with it, and Godot reads the value from the exported contract.",
    )
    p.add_argument(
        "--action_scale",
        type=float,
        default=-1.0,
        help="Fraction of each joint's range the policy may command. -1 keeps the task "
        "default. A SMALL value trains a policy that makes corrections around the rest pose "
        "rather than commanding a different pose - which is what composes with Godot's "
        "balance layer instead of fighting it.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.seed = args.seed
    if args.iterations:
        agent_cfg.max_iterations = args.iterations
    if args.experiment:
        agent_cfg.experiment_name = args.experiment
    if args.push >= 0.0:
        env_cfg.push_velocity = args.push
        env_cfg.push_ang_velocity = args.push * (5.0 / 3.0)
    if args.action_rate_limit >= 0.0:
        env_cfg.action_rate_limit = args.action_rate_limit
    if args.action_scale > 0.0:
        env_cfg.action_scale = args.action_scale

    # Relative, matching the 2.3.2 layout so `logs/rsl_rl/<experiment>` means the same thing to the
    # PowerShell run-resolution helpers. That makes the working directory load-bearing: run this
    # from `isaac_lab_3/`, which is what `Invoke-Isaac` in config.ps1 guarantees.
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root = os.path.abspath(log_root)
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_dir += f"_{args.run_name}"
    log_dir = os.path.join(log_root, run_dir)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[train] task     : {args.task}")
    print(f"[train] envs     : {env_cfg.scene.num_envs}")
    print(f"[train] log dir  : {log_dir}")

    # Reconciles the config schema against whatever rsl-rl-lib is actually installed. Without it
    # the runner dies on `TypeError: MLPModel.__init__() got an unexpected keyword argument
    # 'stochastic'` — isaaclab_rl emits fields this build of rsl_rl does not accept. The stock 3.0
    # train script calls this too; it is not optional.
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume and args.init_from:
        raise SystemExit("--resume continues one run; --init_from seeds a new one. Pick one.")
    if args.resume:
        print(f"[train] resuming from {args.resume}")
        runner.load(args.resume)
    elif args.init_from:
        # Networks only. The optimizer state belongs to a DIFFERENT reward scale - Walk's product
        # reward is roughly an order of magnitude smaller than Stand's additive one - and Adam's
        # second-moment estimates carried across would shrink the effective learning rate for
        # exactly as long as it takes to look like a plateau rather than a bug. The iteration
        # counter is dropped for the same reason: an adaptive LR schedule resuming at iteration
        # 4000 of a run that never happened is not a continuation of anything.
        print(f"[train] seeding networks from {args.init_from} (no optimizer, no iteration)")
        # `strict=False` because a task may parameterise its std differently from the checkpoint
        # it seeds from - `std_param` vs `log_std_param` - and that single key is exactly the one
        # being deliberately reset below anyway. Every weight tensor still has to match.
        runner.load(
            args.init_from,
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
            strict=False,
        )
        # **Re-inflate the exploration noise, or the bootstrap smothers itself.**
        #
        # `std_param` is part of the actor's state dict, so a converged Stand checkpoint carries its
        # collapsed exploration noise across with the weights - measured at 0.14 after 15 minutes,
        # against the 0.4 the Walk config asks for. The seeded policy would then start at a sharp
        # local optimum (it holds a very good stand), sampling almost deterministically, in a reward
        # that pays nothing for standing still under a movement command. Every action scores the
        # same near-zero and there is no gradient pointing out of it.
        #
        # `init_std` from the config is what the run was configured to explore at; honour it rather
        # than whatever the donor run happened to decay to.
        import math

        distribution = runner.alg.actor.distribution
        target = agent_cfg.actor.distribution_cfg.init_std
        # Handles both parameterisations: `scalar` stores std directly, `log` stores its logarithm.
        with torch.no_grad():
            if hasattr(distribution, "log_std_param"):
                was = float(distribution.log_std_param.exp().mean().item())
                distribution.log_std_param.fill_(math.log(target))
            else:
                was = float(distribution.std_param.mean().item())
                distribution.std_param.fill_(target)
        print(f"[train] exploration std reset {was:.3f} -> {target:.3f}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    total = agent_cfg.max_iterations
    deadline = time.time() + args.max_minutes * 60.0 if args.max_minutes > 0 else None

    if deadline is None:
        runner.learn(num_learning_iterations=total, init_at_random_ep_len=True)
    else:
        # Chunked so the wall-clock cap lands on a checkpoint boundary rather than killing the
        # process mid-iteration. `save_interval` is the natural chunk: rsl_rl writes a checkpoint
        # every `save_interval` iterations, so a run stopped here is always resumable.
        chunk = max(1, agent_cfg.save_interval)
        done = 0
        started = time.time()
        while done < total and time.time() < deadline:
            todo = min(chunk, total - done)
            runner.learn(num_learning_iterations=todo, init_at_random_ep_len=(done == 0))
            done += todo
            elapsed = time.time() - started
            remaining = (deadline - time.time()) / 60.0
            print(
                f"[train] {done}/{total} iterations, {elapsed / 60.0:.1f} min elapsed, "
                f"{max(0.0, remaining):.1f} min left"
            )
        print(f"[train] stopped after {done} iterations ({(time.time() - started) / 60.0:.1f} min)")

    env.close()


if __name__ == "__main__":
    main()
