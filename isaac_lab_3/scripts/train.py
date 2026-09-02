"""Train a policy under Newton/XPBD with rsl_rl.

Kit-less: no `AppLauncher`, no Isaac Sim, no Vulkan. `SimulationContext` is created by the
environment itself when `gym.make` builds it, so this script only has to register the task and hand
it to rsl_rl.

Normally driven by `train.ps1`, which supplies every argument from `config.ps1`.

    python isaac_lab_3/scripts/train.py --task P4F-Dummy-Stand-Newton-v0 --num_envs 4096 --max_minutes 15
"""

from __future__ import annotations

import argparse
import json
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
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one env-cfg field, e.g. --set balance_max_torque=75. Preferred over the "
        "P4F_* environment variables: it is explicit at the call site, it lands in the run's own "
        "params/env.yaml, and an unknown key is an error rather than a silent no-op. Three separate "
        "bugs on this project came from an env var quietly falling back to its default.",
    )
    p.add_argument(
        "--max_action_std",
        type=float,
        default=0.4,
        help="Upper bound on the Gaussian exploration std. 0 disables the bound. **Not a tuning "
        "knob - it is the fix for a measured runaway.** rsl_rl leaves `log_std_param` a free "
        "parameter with no ceiling, and when a curriculum pushes reward negative the entropy term "
        "is the only one left with a clear gradient, so std inflates, actions get wilder, reward "
        "gets worse, and it inflates further. Measured over 13 chained segments: every segment that "
        "ended with std >= 0.91 scored 0%% and every segment that ended <= 0.32 scored 96-100%%, "
        "with std peaking at 6.25.",
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
    from run_conditions import apply_overrides, restore

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
    # A resume continues an experiment, so it must continue that experiment's PLANT. Without this
    # the reaction, the solver iterations and the action scale all revert to task defaults and the
    # run silently trains a different body than the checkpoint came from.
    if args.resume:
        restore(env_cfg, args.resume, label="train")
    # **Carry the curriculum ceiling across the restart.** Written by this script at the end of a
    # run, beside the checkpoints, and read back here - so resume.ps1, night.py and a hand-typed
    # --resume all pick it up rather than only one of them. Read BEFORE apply_overrides so an
    # explicit `--set push_curriculum_start=` still wins.
    # Which recorded value feeds which cfg field, and how to describe it. One table so a new
    # curriculum needs an entry rather than another copy of this block.
    CURRICULUM_CARRY = {
        "impulse_ceiling": ("push_curriculum_start", "_impulse_ceiling", "N.s"),
        "cmd_speed_ceiling": ("cmd_speed_start", "_cmd_speed_ceiling", "m/s"),
    }

    if args.resume:
        _carry = pathlib.Path(args.resume).resolve().parent / "curriculum.json"
        if _carry.is_file():
            try:
                _saved = json.loads(_carry.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                print(f"[train] WARNING could not read {_carry}: {exc}; every ramp restarts from "
                      "its task default and will re-hunt the level it already found.")
                _saved = {}
            for _key, (_field, _attr, _unit) in CURRICULUM_CARRY.items():
                if _key not in _saved or not hasattr(env_cfg, _field):
                    continue
                try:
                    _prev = float(_saved[_key])
                except (TypeError, ValueError):
                    continue
                if _prev > 0.0:
                    setattr(env_cfg, _field, _prev)
                    print(f"[train] curriculum resumes at {_prev:.3f} {_unit}")

    apply_overrides(env_cfg, args.set, label="train")
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

    # **Bound the exploration std after every update.**
    #
    # Wrapped around `alg.update` rather than patched into rsl_rl: the clamp has to land after the
    # optimizer step that moved the parameter, and this is the only seam that sees every step
    # without a fork. Clamping the PARAMETER, not the sampled action - capping actions after the
    # fact would leave the runaway inside the policy and merely hide it from the environment, and
    # `deterministic_output` returns the MEAN, so a policy whose mean has been dragged out of
    # [-1, 1] stays broken at evaluation time where no sampling happens at all.
    if args.max_action_std > 0.0:
        import math

        _dist = runner.alg.actor.distribution
        _ceiling = args.max_action_std
        _inner = runner.alg.update

        if hasattr(_dist, "log_std_param"):
            _limit = math.log(_ceiling)

            def _clamp():
                _dist.log_std_param.clamp_(max=_limit)
        else:
            def _clamp():
                _dist.std_param.clamp_(max=_ceiling)

        def _update_then_clamp(*a, **kw):
            out = _inner(*a, **kw)
            with torch.no_grad():
                _clamp()
            return out

        runner.alg.update = _update_then_clamp
        print(f"[train] exploration std bounded at {_ceiling}")

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

    # Record where the ramp got to, for whatever resumes from these checkpoints. Written before
    # env.close() while the env is still alive, and failure here must not lose the training run -
    # the checkpoints are already on disk and are the thing that matters.
    _state = {}
    for _key, (_field, _attr, _unit) in CURRICULUM_CARRY.items():
        _reached = getattr(env.unwrapped, _attr, None)
        if _reached is not None:
            _state[_key] = float(_reached)
            print(f"[train] curriculum ceiling {float(_reached):.3f} {_unit} recorded for the next run")
    if _state:
        try:
            (pathlib.Path(log_dir) / "curriculum.json").write_text(
                json.dumps(_state, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            print(f"[train] WARNING could not record the curriculum ceiling: {exc}")

    env.close()


if __name__ == "__main__":
    main()
