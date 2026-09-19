"""Train the dummy's policies with PPO on the MuJoCo plant.

Two tasks: `perturb` - survive ball impacts using its LEGS - and `walk` - follow a velocity command.
The first training on this project where the plant the policy learns on is the plant it ships on,
so there is no transfer gap to lose it to - which was the failure mode of every previous attempt.

The pelvis balance assist is absent from the environment. Measured 2026-09-09, it was carrying the
perturbation recovery: up to 90.7 N.m of external torque written straight onto the pelvis, without
which the body fell to 33.4% upright. A policy here has to keep its feet under its centre of mass.

Sessions are normally chained and scored by `scripts/overnight.py`. One by hand:

    python mujoco_rig/rl/train.py --task perturb --num_envs 4096 --steps 16 --max_minutes 15
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from env_config import FALL_PENALTY  # noqa: E402
from ppo import PPO  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=tuple(TASKS), default="perturb",
                   help="perturb = survive ball impacts; walk = follow a velocity command")
    p.add_argument("--backend", choices=("cpu", "warp"), default="warp",
                   help="warp = mujoco_warp on the GPU (training only); cpu = MuJoCo C")
    p.add_argument("--num_envs", type=int, default=8192)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--steps", type=int, default=24, help="rollout steps per env per iteration")
    p.add_argument("--seconds", type=float, default=12.0, help="episode length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default="perturb")
    p.add_argument("--max_minutes", type=float, default=0.0)
    p.add_argument("--init_std", type=float, default=0.03)
    p.add_argument("--entropy_coef", type=float, default=0.0005)
    # A GPU batch is ~28x the CPU one, so its gradient is far less noisy - but PPO still moves
    # the policy only as far as the KL cap allows, which is why raising num_envs alone bought
    # bigger batches and NOT faster wall-clock learning. This is the knob that spends them.
    p.add_argument("--desired_kl", type=float, default=0.01)
    # Continue from an existing checkpoint; its optimiser state comes too when the shapes match
    # (see seed_from_checkpoint).
    # **Exploration does not always travel with the weights.** A seed's std comes with it by
    # default, which is right when continuing the SAME task - a converged policy has a small std and
    # re-inflating it throws its behaviour away. It is wrong when the seed was optimised for a
    # different task: the walk was seeded from a balance brain whose std had collapsed to ~0.08
    # after hours of learning to stand perfectly still, so it began unable to explore away from
    # standing, and eleven sessions drove entropy from -28.3 to -29.8 while it failed to move.
    p.add_argument("--reset_std", action="store_true",
                   help="re-inflate the seed's exploration std to --init_std. Use when the seed "
                        "was trained on a DIFFERENT task.")
    p.add_argument("--init_from", default="",
                   help="path to a model_*.pt to seed the networks from")
    p.add_argument("--lr_max", type=float, default=1e-2,
                   help="ceiling on the adaptive learning rate")
    p.add_argument("--walk_stage", type=int, default=3, choices=(1, 2, 3, 4, 5),
                   help="walk only: how much of the command space to ask for. "
                        "See walk_config.STAGES.")
    # Gradient steps per iteration are `epochs x minibatches`, and they are nearly FREE:
    # collecting a 262,144-sample batch takes ~12 s, the update on it ~10 ms. Raising
    # either is the only lever that buys more learning per hour without shrinking the
    # batch, which is what collapses a run.
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    # **Batch size is scheduled, not fixed.** The useful batch is the one near the gradient's
    # CRITICAL size - below it a bigger batch buys proportionally more progress per update, so
    # small and noisy is strictly faster in wall-clock; above it the extra samples buy almost
    # nothing and the compute is wasted. That threshold is not a constant: it RISES as the policy
    # improves, because a sharpened policy has a subtler, noisier gradient. One fixed batch is
    # therefore wrong at one end of a run or the other. WorldSchedule says what grows, and why.
    p.add_argument("--envs_schedule", default="",
                   help="comma-separated world counts to grow through, e.g. 4096,8192,16384. "
                        "Empty keeps --num_envs fixed.")
    p.add_argument("--schedule_dwell", type=int, default=25,
                   help="iterations at a world count before it may grow")
    p.add_argument("--schedule_gain", type=float, default=0.02,
                   help="fractional improvement in episode length over one dwell window that "
                        "counts as still making progress. Below it, the batch grows.")
    # The authored 2-4 s made the task impossible for ANY controller: a single 10 kg
    # impact is survivable 75% of the time unaided, but the same ball every 2-4 s topples
    # 8/8 in 6.8 s, because the next one lands mid-recovery.
    p.add_argument("--ball_every", type=float, nargs=2, default=(4.0, 7.0),
                   help="seconds between shots")
    p.add_argument("--target_weights", default="",
                   help="perturb: aim more shots at some bones, e.g. Head=5.5,Chest=4.7 - unnamed "
                        "bones weigh 1. Training only; the scorers always shoot uniformly.")
    # The curriculum band, in ball speed. The ball is 8 kg (build_mjcf.BALL_MASS), so 2.0 m/s is
    # 16 N.s, and the 6.0 m/s the Godot scene fires is 48 N.s - 0.69 m/s of COM change, inside the
    # ~1.0 m/s a stepping recovery can absorb.
    p.add_argument("--speed_start", type=float, default=2.0)
    p.add_argument("--speed_end", type=float, default=6.0)
    p.add_argument("--promote_at", type=float, default=0.93,
                   help="fraction of the episode survived before the ball gets faster")
    p.add_argument("--speed_step", type=float, default=1.10)
    # Dwell is counted in EPISODES, not iterations. A GPU iteration carries ~85x the experience
    # of a CPU one, so an iteration count means something completely different on each backend
    # and would make the curriculum either instant or unreachable depending on the flag.
    p.add_argument("--stage_min_episodes", type=int, default=400,
                   help="episodes finished at a difficulty before it can be promoted")
    p.add_argument("--stage_min_iters", type=int, default=15,
                   help="iterations at a difficulty before it can be promoted; the episode count "
                        "alone is ~1.5 iterations at 16,384 envs")
    p.add_argument("--promote_cooldown", type=int, default=8,
                   help="iterations after a promotion during which the learning rate is capped, "
                        "so the critic can re-fit before the policy is allowed to move far")
    p.add_argument("--promote_lr", type=float, default=1e-4,
                   help="the learning-rate cap applied during that cooldown")
    return p.parse_args(argv)


# ---------------------------------------------------------------- tasks
class Task:
    """What the trainer needs from one task. A new task is a new one of these, nothing else."""

    envs: dict[str, str] = {}      # backend -> "module.Class"

    def prepare(self, args):
        """Anything that must be set before the environment is built."""

    def env_kwargs(self, args):
        """Constructor arguments beyond the ones every environment takes."""
        return {}

    def progress(self, env, curriculum):
        """This task's field in the per-iteration log line, which `overnight.py` parses."""
        raise NotImplementedError


class PerturbTask(Task):
    envs = {"cpu": "perturb_env.PerturbEnv", "warp": "perturb_env_warp.PerturbEnvWarp"}

    def env_kwargs(self, args):
        weights = {}
        for item in args.target_weights.split(","):
            if item.strip():
                bone, weight = item.split("=")
                weights[bone.strip()] = float(weight)
        return {"ball_every": tuple(args.ball_every), "target_weights": weights or None}

    def progress(self, env, curriculum):
        return f"speed {curriculum.speed:4.2f} ({curriculum.impulse:4.0f}N.s)"


class WalkTask(Task):
    envs = {"cpu": "walk_env.WalkEnv", "warp": "walk_env_warp.WalkEnvWarp"}

    def prepare(self, args):
        from walk_config import STAGES
        s = STAGES[args.walk_stage]
        print(f"[train] walk stage {args.walk_stage}: forward {s['forward']} "
              f"lateral {s['lateral']} turn {s['turn']}", flush=True)

    def env_kwargs(self, args):
        from walk_config import WalkCommandConfig
        return {"command_config": WalkCommandConfig.for_stage(args.walk_stage)}

    def progress(self, env, curriculum):
        return f"vx {getattr(env, 'vx_ema', 0.0):+5.2f}"


TASKS = {"perturb": PerturbTask(), "walk": WalkTask()}


def env_factory(task, args, device):
    """A function building the task's environment for this backend at `n` worlds. The world
    schedule calls it again to grow the world count."""
    module, name = task.envs[args.backend].rsplit(".", 1)
    # Imported on demand: the warp modules pull in CUDA, which a CPU run never needs.
    env_class = getattr(importlib.import_module(module), name)
    kwargs = dict(device=device, episode_seconds=args.seconds, seed=args.seed,
                  **task.env_kwargs(args))
    return lambda n: env_class(num_envs=n, **kwargs)


# ---------------------------------------------------------------- checkpoints
def seed_from_checkpoint(algo, env, args, device):
    """Start `algo` from `args.init_from`: its weights, its exploration std unless `--reset_std`,
    and its optimiser state when the shapes allow."""
    seed_ck = torch.load(args.init_from, map_location=device, weights_only=False)
    state = seed_ck["model"]
    if seed_ck["num_actions"] != env.num_actions:
        raise SystemExit(
            f"--init_from action mismatch: checkpoint drives {seed_ck['num_actions']} joints, "
            f"this env has {env.num_actions}. That is a different rig, not a different task.")
    if seed_ck["num_obs"] != env.num_obs:
        # **Widen the input layer instead of refusing.** A walk policy is a perturb policy with
        # three command channels appended, and on the compliant plant a walk started from
        # scratch has to learn to STAND before it can learn to walk - the body falls in 2.5 s
        # unaided. Seeding from a brain that already stands and zeroing the new columns hands it
        # that for free: at zero command the widened network computes exactly what the seed did.
        if seed_ck["num_obs"] > env.num_obs:
            raise SystemExit(
                f"--init_from is WIDER than this env ({seed_ck['num_obs']} > {env.num_obs}); "
                f"dropping observation channels would silently change what the policy reads.")
        own = algo.net.state_dict()
        widened = []
        for key in ("actor.0.weight", "critic.0.weight"):
            old = state[key]
            new = own[key].clone()
            new[:, :old.shape[1]] = old
            new[:, old.shape[1]:] = 0.0
            state[key] = new
            widened.append(f"{key} {tuple(old.shape)} -> {tuple(new.shape)}")
        print(f"[train] widened {seed_ck['num_obs']} -> {env.num_obs} obs, new channels zeroed: "
              + "; ".join(widened), flush=True)
    algo.net.load_state_dict(state)
    seed_std = algo.net.log_std.detach().exp().mean().item()
    # The seed's exploration std comes with it. A converged policy has a small std, and
    # continuing from one with --init_std set high would throw its behaviour away on step one.
    if args.reset_std:
        with torch.no_grad():
            algo.net.log_std.fill_(float(np.log(args.init_std)))
        print(f"[train] exploration std RESET to {args.init_std} "
              f"(seed was {seed_std:.3f}) - the seed is from another task", flush=True)
    # The optimiser state travels too (see save_checkpoint) - but only when the shapes match: a
    # widened input layer has different parameter tensors, and Adam's moments are per-tensor.
    carried = "weights only"
    if "optimizer" in seed_ck and seed_ck.get("num_obs") == env.num_obs:
        try:
            algo.opt.load_state_dict(seed_ck["optimizer"])
            algo.lr = float(seed_ck.get("lr", algo.lr))
            for g in algo.opt.param_groups:
                g["lr"] = algo.lr
            carried = f"with optimiser state, lr {algo.lr:.2e}"
        except ValueError as exc:
            print(f"[train] optimiser state not carried: {exc}", flush=True)
    print(f"[train] seeded from {args.init_from} "
          f"(task {seed_ck.get('task', '?')}, std {algo.net.log_std.exp().mean():.3f}, "
          f"{carried})", flush=True)


def save_checkpoint(path, algo, env, args, steps, speed):
    """Write the weights with everything a later run or a scorer needs to continue from them.

    The curriculum stage travels WITH the weights. A policy scored at a difficulty it never reached
    is not a measurement of that policy, and the stage is not recoverable afterwards.
    **The optimiser state travels too.** Adam's moment estimates are what scale the first steps of
    an update; restoring weights without them makes the first update after a seed badly scaled,
    which measurably costs the run. Three continuations in a row opened with KL ~0.06 against a 0.01
    target, the adaptive controller clamped the learning rate to its floor in response, and episode
    length sat flat for ~8 iterations before recovering. On a 5-minute session that is half the
    budget.
    """
    torch.save({"model": algo.net.state_dict(), "optimizer": algo.opt.state_dict(),
                "lr": algo.lr, "num_obs": env.num_obs,
                "steps": steps, "num_envs": env.num_envs,
                "task": args.task, "ball_speed": speed,
                "ball_every": tuple(args.ball_every),
                "num_actions": env.num_actions}, path)


# ---------------------------------------------------------------- episodes and rollouts
class EpisodeStats:
    """Per-world return and length accumulators, and a bounded history of finished episodes."""

    def __init__(self, num_envs, device):
        self.device = device
        self.returns: list[float] = []
        self.lengths: list[float] = []
        self.resize(num_envs)

    def resize(self, num_envs):
        """Restart the per-world accumulators at a new world count; the history is kept.

        They are carried across iterations, so after a rebuild they are the same trap as the
        trainer's own observation: still the old width.
        """
        self._return = torch.zeros(num_envs, device=self.device)
        self._length = torch.zeros(num_envs, device=self.device)

    def record(self, reward, done):
        """Accumulate one step of every world; returns how many episodes it finished."""
        self._return += reward
        self._length += 1
        # Vectorised: a per-element loop forced a GPU->host sync PER ENV PER STEP, which at 8,192
        # envs costs more than the physics it is bookkeeping for.
        finished = torch.nonzero(done).flatten()
        if not finished.numel():
            return 0
        self.returns.extend(self._return[finished].tolist())
        self.lengths.extend(self._length[finished].tolist())
        self._return[finished] = 0.0
        self._length[finished] = 0.0
        # Bound the history; a GPU run finishes millions of episodes.
        if len(self.lengths) > 4000:
            del self.lengths[:-2000]
            del self.returns[:-2000]
        return int(finished.numel())

    def mean_return(self, last):
        return np.mean(self.returns[-last:]) if self.returns else float("nan")

    def mean_length(self, last, empty=float("nan")):
        return float(np.mean(self.lengths[-last:])) if self.lengths else empty

    def clear(self):
        self.returns.clear()
        self.lengths.clear()


def collect_rollout(env, algo, obs, steps, episodes, device):
    """Step every world `steps` times under the current policy.

    Returns the rollout buffers, the last observation, how many episodes finished, and how many
    non-finite rewards had to be replaced.
    """
    # Rollout buffers live on the SAME device as the env. Allocating them on the host would
    # copy 196k x 120 floats across PCIe every step and undo the entire point of the port.
    z = dict(device=device)
    buf = {"obs": torch.zeros(steps, env.num_envs, env.num_obs, **z),
           "act": torch.zeros(steps, env.num_envs, env.num_actions, **z),
           "logp": torch.zeros(steps, env.num_envs, **z),
           "rew": torch.zeros(steps, env.num_envs, **z),
           "done": torch.zeros(steps, env.num_envs, **z),
           "val": torch.zeros(steps, env.num_envs, **z)}
    finished = 0
    nonfinite = torch.zeros((), device=device)
    for t in range(steps):
        action, logp, value = algo.act(obs)
        buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, action, logp, value
        obs, reward, done, _ = env.step(action)
        # **Last line of defence.** The envs retire a diverged world and replace its reward, but
        # one non-finite reward reaching the buffer turns the whole update into NaN - which is
        # how walk training died twice on 2026-09-10. Replace it, count it, and say so.
        bad = ~torch.isfinite(reward)
        nonfinite = nonfinite + bad.sum()
        reward = torch.where(bad, torch.full_like(reward, -FALL_PENALTY), reward)
        buf["rew"][t], buf["done"][t] = reward, done.float()
        finished += episodes.record(reward, done)
    return buf, obs, finished, int(nonfinite)


# ---------------------------------------------------------------- what changes during a run
class Curriculum:
    """Impact strength, raised one step each time the current one is MASTERED.

    **Exploration cannot reach a protective step.** Runs 3 and 4 both converged to the rest-pose
    hold and stopped: 29.9% upright against the unaided plant's 30.1%. The reason is measured, not
    guessed - std 0.03 across 36 joints is 0.75% of each joint's range, which will never stumble
    onto a coordinated step, and any larger uniform noise topples the body unaided (probe_noise.py).
    A behaviour that large has to be GROWN: keep each increment within reach of the small noise
    around the current policy.

    Impulse is m*dv, so ball SPEED buys the same ramp as ball mass with no model recompile - and
    recompiling is the only correct way to change mass here, since writing `body_mass` on a live
    model leaves the contact solver's invweight constants stale. The ball no longer falls (see
    PerturbEnv._park_ball), so speed maps onto impulse linearly and the log reports the impulse.
    """

    def __init__(self, args, env):
        self.speed = args.speed_start
        self.has_ball = env.ball >= 0
        # No projectile in this model, so the impact curriculum has nothing to ramp.
        self.end = args.speed_end if self.has_ball else args.speed_start
        self.ball_kg = float(env.model.body_mass[env.ball]) if self.has_ball else 0.0
        self.step, self.promote_at = args.speed_step, args.promote_at
        self.min_episodes, self.min_iters = args.stage_min_episodes, args.stage_min_iters
        self.cooldown, self.cooldown_lr = args.promote_cooldown, args.promote_lr
        self.stage_episodes = 0     # episodes finished since this difficulty began
        self.stage_iters = 0
        # **A seeded start is a promotion in every way that matters.** The critic is about to see
        # data from a policy it did not fit, so the first advantages are wrong in the same
        # direction, and on a sharpened policy the first update is enormous: measured KL +0.06,
        # +0.44, +0.78, +5.97 and +23.0 across successive chained sessions, against a 0.01 target
        # - each one worse than the last as the policy gets more deterministic. One of those cost
        # a session outright (episode length 764 -> 116). The cooldown protects this too.
        self.promoted_at = 0 if args.init_from else -10 ** 9

    @property
    def impulse(self):
        return self.speed * self.ball_kg

    def cap_lr(self, it, algo):
        """The cooldown after a promotion. The adaptive controller reacts to a bad step AFTER it
        has been taken; this is the only thing that stops the step being taken at all."""
        if it - self.promoted_at < self.cooldown and algo.lr > self.cooldown_lr:
            algo.lr = self.cooldown_lr
            for g in algo.opt.param_groups:
                g["lr"] = algo.lr

    def maybe_promote(self, it, env, steps, episodes):
        """Raise the difficulty one step if the current one is mastered.

        **The gate has to be strict, or the curriculum is decorative.** The first attempt promoted
        on "75% of an episode survived" with no dwell time and went 3.0 -> 6.0 m/s in 150
        iterations (2.2 min); episode length then fell straight back to 733, exactly the plateau
        the curriculum existed to escape. A stage the policy has not MASTERED teaches it nothing
        before it is taken away, so promotion needs near-perfect survival AND a minimum dwell.

        **Dwell in ITERATIONS as well as episodes, and the LR is capped afterwards.** Two defects,
        both measured. At 16,384 envs one iteration finishes ~262 episodes, so a 400-episode dwell
        was 1.5 iterations - and the history refills instantly with episodes that were already in
        flight at the EASIER difficulty, so they read as mastery of a stage the policy has never
        faced. A run went 2.00 -> 2.20 -> 2.42 between two log lines. Then the promotion itself
        destroyed the policy: the critic is calibrated for the old returns, so the first update
        after a promotion saw KL +11.88 (a thousand times target), and episode length fell
        1023 -> 102 and never recovered in 84 further iterations.

        **The dwell is at least one full episode.** A world that times out a few iterations after
        a promotion played most of its episode at the EASIER difficulty: with a 15-iteration dwell
        a surviving episode had spent 60 of its 75 iterations there, and read as mastery.
        """
        dwell = max(self.min_iters, -(-env.max_episode_length // steps))
        target = self.promote_at * env.max_episode_length
        if not (self.has_ball and self.speed < self.end
                and self.stage_episodes >= self.min_episodes
                and self.stage_iters >= dwell
                and len(episodes.lengths) >= 100
                and episodes.mean_length(100) > target):
            return
        # What the gate actually saw. Every session on 2026-09-10 promoted at exactly iterations
        # 75 and 150, which reads like a clock rather than a mastery test.
        gate = (f"on ep_len {episodes.mean_length(100):.0f} > {target:.0f} after "
                f"{self.stage_episodes} episodes / {self.stage_iters} iters")
        self.speed = min(self.end, self.speed * self.step)
        env.ball_speed = self.speed
        self.stage_episodes = 0
        self.stage_iters = 0
        self.promoted_at = it
        episodes.clear()            # re-earn the promotion at the new difficulty
        print(f"[curriculum] ball speed -> {self.speed:.2f} m/s ({self.impulse:.0f} N.s) at it "
              f"{it}  {gate}; learning rate capped for {self.cooldown} iterations", flush=True)


class WorldSchedule:
    """Grows the world count when progress over one dwell window has flattened.

    **Measured, not timed**: a clock cannot know when the policy sharpened, and every guessed
    threshold on this project has had to be re-measured once the plant changed underneath it.

    **It grows the worlds, never the rollout length.** At an equal 65,536-sample batch, 4,096 x 16
    peaked at 1,196 episode steps and 16,384 x 4 at 373: a short rollout is too little experience
    for the critic to see a step's consequences, whatever the batch size around it. Worlds are also
    the cheap axis - measured, ms per policy step = 35.8 + 0.04157 x envs, where every extra rollout
    step costs the full 36 ms. A rebuild costs ~20 s of Warp warm-up, 2% of a 15-minute session.
    """

    def __init__(self, args):
        self.sizes = [int(x) for x in args.envs_schedule.split(",") if x.strip()]
        self.dwell, self.gain = args.schedule_dwell, args.schedule_gain
        self.at, self.mark = 0, None

    def first(self, default):
        return self.sizes[0] if self.sizes else default

    def grow(self, it, num_envs, episodes, steps):
        """The world count to rebuild at, or None to keep the current one."""
        if not self.sizes or num_envs == self.sizes[-1] or it - self.at < self.dwell:
            return None
        now = episodes.mean_length(200, empty=0.0)
        grown = None
        if self.mark is not None and now < self.mark * (1.0 + self.gain):
            grown = self.sizes[self.sizes.index(num_envs) + 1]
            print(f"[train] progress flattened ({self.mark:.0f} -> {now:.0f} over "
                  f"{self.dwell} iterations): {grown:,} envs, batch {grown * steps:,}", flush=True)
        self.mark, self.at = now, it
        return grown


# ---------------------------------------------------------------- the loop
def main() -> int:
    args = parse_args()
    task = TASKS[args.task]
    task.prepare(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # **Backend, measured on this machine (RTX 4080 SUPER, Ryzen 7 7800X3D):**
    #
    #     cpu    96 envs      2,693 policy SPS      10,773 physics steps/s     45x realtime
    #     warp 8,192 envs    71,522 policy SPS     286,089 physics steps/s   1,192x realtime
    #
    # The CPU path steps every MjData sequentially in a Python loop - one core of eight. The GPU
    # path is the same MuJoCo model batched by mujoco_warp, and the network follows it there
    # because at the resulting 196k-sample batch a PPO pass is 346 ms on CPU against 7.9 ms on CUDA.
    #
    # `warp` is float32 where the C engine is float64, so it is a TRAINING backend only. Scoring
    # stays on CPU MuJoCo, which is what Godot drives through P/Invoke - see eval.py and
    # parity_gpu.py. A checkpoint is never judged by the engine that trained it.
    device = "cuda" if args.backend == "warp" else "cpu"
    make_env = env_factory(task, args, device)
    schedule = WorldSchedule(args)
    env = make_env(schedule.first(args.num_envs))
    algo = PPO(env.num_obs, env.num_actions, device=device,
               init_std=args.init_std, entropy_coef=args.entropy_coef,
               desired_kl=args.desired_kl, epochs=args.epochs,
               minibatches=args.minibatches, lr_max=args.lr_max)
    if args.init_from:
        seed_from_checkpoint(algo, env, args, device)

    log_dir = pathlib.Path("logs/mujoco") / (time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] task={args.task} {env.num_envs} envs, {env.num_obs} obs, "
          f"{env.num_actions} actions, episode {env.max_episode_length} steps, NO balance assist, "
          f"init_std {args.init_std}, "
          f"ball every {args.ball_every[0]:.0f}-{args.ball_every[1]:.0f}s -> {log_dir}", flush=True)

    curriculum = Curriculum(args, env)
    env.ball_speed = curriculum.speed
    steps = args.steps
    if schedule.sizes:
        print(f"[train] batch schedule: {schedule.sizes} envs x {steps} steps "
              f"= {[n * steps for n in schedule.sizes]} samples", flush=True)

    obs = env.reset_all()
    # Spread the episode clocks, or every survivor times out in the same iteration: the curriculum
    # gate then reads only survivors, and for the first 75 iterations the logged episode length
    # counts only worlds that FELL - read once as "the dummy falls in 3 s" when it stood 100%.
    env.randomize_episode_phase()
    episodes = EpisodeStats(env.num_envs, device)
    started = time.time()

    for it in range(1, args.iterations + 1):
        buf, obs, finished, nonfinite = collect_rollout(env, algo, obs, steps, episodes, device)
        curriculum.stage_episodes += finished
        if nonfinite:
            print(f"[train] it {it}: {nonfinite} non-finite rewards reached the trainer and were "
                  f"replaced by the fall penalty - an env guard is missing", flush=True)
        with torch.no_grad():
            last_value = algo.net.value(obs)
        adv, ret = algo.compute_returns(buf["rew"], buf["done"], buf["val"], last_value)
        curriculum.stage_iters += 1

        grown = schedule.grow(it, env.num_envs, episodes, steps)
        if grown:
            env = make_env(grown)
            env.ball_speed = curriculum.speed
            # **Everything sized by the world count has to be re-fetched.** The rollout buffers
            # are allocated per rollout and follow automatically, but the trainer's own `obs` is
            # carried across iterations and is still the old width - which is exactly how the first
            # schedule step died, one iteration after a rebuild that had itself worked perfectly.
            obs = env.reset_all()
            env.randomize_episode_phase()
            episodes.resize(env.num_envs)
        curriculum.cap_lr(it, algo)

        stats = algo.update((
            buf["obs"].reshape(-1, env.num_obs),
            buf["act"].reshape(-1, env.num_actions),
            buf["logp"].reshape(-1),
            adv.reshape(-1),
            ret.reshape(-1),
            buf["val"].reshape(-1)))
        curriculum.maybe_promote(it, env, steps, episodes)

        if it % 5 == 0 or it == 1:
            minutes = (time.time() - started) / 60.0
            print(f"it {it:4d}  return {episodes.mean_return(200):8.1f}  "
                  f"ep_len {episodes.mean_length(200):6.1f}/{env.max_episode_length}  "
                  f"{task.progress(env, curriculum)}  "
                  f"kl {stats['kl']:+.4f}  lr {algo.lr:.2e}  entropy {stats['entropy']:6.2f}"
                  + (f"  n{env.num_envs}" if schedule.sizes else "")
                  + f"  {minutes:5.1f} min", flush=True)

        out_of_time = args.max_minutes > 0.0 and (time.time() - started) / 60.0 >= args.max_minutes
        if it % 25 == 0 or it == args.iterations or out_of_time:
            save_checkpoint(log_dir / f"model_{it}.pt", algo, env, args, steps, curriculum.speed)
        if out_of_time:
            print(f"[train] stopped after {it} iterations ({args.max_minutes:.0f} min budget)",
                  flush=True)
            break

    print(f"[train] done -> {log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
