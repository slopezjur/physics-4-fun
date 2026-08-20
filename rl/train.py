"""PPO training entry point for the ragdoll RL track.

Exists because the packaged `gdrl` console script cannot set `n_parallel`: its argparse has no
such flag, and the `extras` it collects are never forwarded to the environment constructor
(see godot_rl/wrappers/stable_baselines_wrapper.py::stable_baselines_training). Process-based
parallelism is the framework's own vectorization mechanism and the whole point of M5, so we
drive StableBaselinesGodotEnv directly instead.

Everything else deliberately mirrors what the bundled trainer does, so numbers stay comparable.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor


def _git(*args: str) -> str:
    """Run a git command, returning "" if git is unavailable or this is not a repo."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


class TimeLimitCallback(BaseCallback):
    """Stops training after a wall-clock duration.

    --timesteps is a step budget, and throughput varies a lot with --n_parallel, --speedup and
    especially --viz (a rendering instance gates the whole batch), so "how many steps is 15
    minutes" is not knowable in advance. Returning False from _on_step is SB3's documented way to
    end learn() early, and it stops cleanly - the final checkpoint and manifest are still written.
    """

    def __init__(self, max_seconds: float):
        super().__init__()
        self._max_seconds = max_seconds
        self._deadline = 0.0

    def _on_training_start(self) -> None:
        self._deadline = time.time() + self._max_seconds

    def _on_step(self) -> bool:
        if self._max_seconds <= 0:
            return True
        if time.time() >= self._deadline:
            print(f"reached time limit of {self._max_seconds:.0f}s - stopping")
            return False
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """Saves a matched .zip + .onnx pair into the run directory on a wall-clock interval.

    Time-based rather than SB3's step-based CheckpointCallback because throughput varies with
    --n_parallel and --speedup, so "every N steps" means an unpredictable amount of real time -
    and what you actually want to bound is how much work a crash can destroy.

    Both formats are written together and stamped with the same step count. They were allowed to
    diverge once during development (an .onnx exported from a checkpoint that was later
    overwritten), leaving a .zip and .onnx in the same folder that came from different models.
    Saving them as a pair makes that impossible.

    Nothing is ever overwritten: filenames carry the step count, and the run directory itself is
    auto-incremented per session, so every session's history stays recoverable.
    """

    def __init__(self, interval_seconds: float):
        super().__init__()
        self._interval = interval_seconds
        self._last_save = 0.0
        self._dir = ""
        self.saved: list = []

    def _on_training_start(self) -> None:
        self._dir = self.logger.get_dir() or "."
        self._last_save = time.time()

    def save(self, tag: str) -> None:
        if not self._dir:
            return
        stem = os.path.join(self._dir, f"{tag}_{self.model.num_timesteps:09d}")
        self.model.save(stem + ".zip")

        onnx_ok = False
        # godot_rl's exporter starts with `model.policy.to("cpu")`, and nn.Module.to() mutates in
        # place - so exporting mid-training silently strands the LIVE policy on the CPU while SB3
        # still believes it is on CUDA. The next forward pass then dies with "Expected all tensors
        # to be on the same device". Restoring the device afterwards is what makes periodic export
        # safe; without it every run crashed immediately after its first checkpoint.
        device = self.model.device
        try:
            from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

            # torch warns "Exporting a model while it is in training mode". For this MLP policy
            # (no dropout, no batchnorm) it almost certainly changes nothing, but "almost
            # certainly" is not something you want standing between the .zip and the .onnx that
            # is supposed to reproduce it. Restored in the finally below alongside the device.
            self.model.policy.set_training_mode(False)
            export_model_as_onnx(self.model, stem + ".onnx")
            _embed_onnx_weights(stem + ".onnx")
            onnx_ok = True
        except Exception as exc:  # noqa: BLE001 - a failed export must not kill a long run
            print(f"warning: onnx export failed for {stem}: {exc}")
        finally:
            self.model.policy.to(device)
            self.model.policy.set_training_mode(True)

        self.saved.append(os.path.basename(stem))
        print(f"checkpoint -> {stem}.zip" + (" (+.onnx)" if onnx_ok else " (.onnx FAILED)"))

    def _on_step(self) -> bool:
        if self._interval > 0 and (time.time() - self._last_save) >= self._interval:
            self._last_save = time.time()
            self.save("checkpoint")
        return True

    def _on_training_end(self) -> None:
        self.save("final")


def _embed_onnx_weights(path: str) -> None:
    """Rewrite an exported .onnx so its weights are inside the file.

    torch.onnx writes tensors to a sibling .onnx.data file, but the in-engine loader
    (ONNXInference.LoadModel) reads only the .onnx bytes and would fail to resolve the external
    reference.
    """
    import onnx

    model = onnx.load(path)  # resolves the sidecar into memory
    onnx.save_model(model, path + ".tmp", save_as_external_data=False)
    os.replace(path + ".tmp", path)
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")


class RewardDecompositionCallback(BaseCallback):
    """Logs each reward term separately, so a flat total curve is diagnosable.

    A single scalar reward cannot distinguish "no term is producing signal" from "two terms are
    cancelling out" - both look like a line near zero. Reading a near-zero MEAN as though it were
    a near-zero GRADIENT is an easy and confidently-wrong inference to make from the total alone,
    so the terms are logged individually instead of being reasoned about.

    The Godot bridge emits per-term episode totals on each terminal transition (see
    RagdollRLBridge.GetStepInfo). They are signed to sum to the episode reward, so
    reward/shaping + reward/upright + reward/effort + reward/terminal should track
    rollout/ep_rew_mean - if it drifts, one of the two sides is wrong and that is worth knowing.

    Also counts WHY episodes end. ep_len_mean pinned at its maximum implies everything is timing
    out, but implication is not observation: explicit per-reason fractions distinguish "never
    succeeds" from "succeeds occasionally and the mean hides it".

    Finally, splits the success rate by START POSE. The run trains a spread of tasks of wildly
    different difficulty at once, and every aggregate number is a mixture of them - ep_rew_mean most
    of all. A prone policy that is genuinely improving would stay invisible behind the already-solved
    standing end. The keys needed for the split (episode_end_reason and start_pose_t) already arrive
    in the same info dict on a terminal transition, so it costs nothing extra on the wire.

    Buckets, deliberately not a uniform partition:

    * ``start_standing/*`` - began at exactly upright. The refresher that keeps the goal state alive.
    * ``start_prone/*``    - began exactly flat. THE headline: this is the get-up, and its definition
                             is unchanged since the first run so the whole history stays comparable.
    * ``curriculum/*``     - everything strictly between, i.e. the reverse-curriculum poses.
    * ``pose_0.0/`` ... ``pose_1.0/`` - per-decile, which shows WHICH level the agent is stuck at
                             rather than only that the aggregate stopped moving.
    """

    def __init__(self) -> None:
        super().__init__()
        self._terms: dict[str, list[float]] = {}
        self._stand: dict[str, list[float]] = {}
        self._balls: dict[str, list[float]] = {}
        self._reasons: dict[str, int] = {}
        # start pose -> {"episodes": n, "successes": n}
        self._by_start: dict[str, dict[str, int]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for key, value in info.items():
                if key.startswith("rew_"):
                    self._terms.setdefault(key[4:], []).append(float(value))
                # Success-criterion sub-condition rates. Logged under their own namespace so
                # "which check blocks Standing" is answerable without reading the C# thresholds.
                elif key.startswith("stand_"):
                    self._stand.setdefault(key[6:], []).append(float(value))
                elif key.startswith("act_"):
                    self._stand.setdefault(key, []).append(float(value))
                elif key in ("started_standing", "start_pose_t", "curriculum_t"):
                    self._stand.setdefault(key, []).append(float(value))
                # Perturbation telemetry. Own namespace because it describes the ENVIRONMENT, not
                # the policy: ball/hits below ball/shots means shots are missing, which inflates
                # every balance metric and is invisible in all of them.
                elif key.startswith("ball_"):
                    self._balls.setdefault(key[5:], []).append(float(value))
            reason = info.get("episode_end_reason")
            if reason is not None:
                self._reasons[str(reason)] = self._reasons.get(str(reason), 0) + 1

                # Bucket on the CONTINUOUS start pose, not on started_standing.
                #
                # The reverse curriculum turned the start pose into a continuum, and the old
                # two-way split silently breaks under it: started_standing is only true at exactly
                # t=1, so everything else - including near-standing curriculum poses the agent
                # solves easily - would land in "start_prone" and inflate the one number this
                # project is judged on. start_prone/success has to keep meaning "began flat on the
                # floor" or 32M steps of history stop being comparable.
                #
                # So the endpoints keep their exact historical definitions and the interior gets
                # its own bucket, plus a per-decile breakdown showing which level is the wall.
                pose = info.get("start_pose_t")
                if pose is not None:
                    pose = float(pose)
                    if pose >= 1.0:
                        bucket = "start_standing"
                    elif pose <= 0.0:
                        bucket = "start_prone"
                    else:
                        bucket = "curriculum"

                    # Two decimals, matching CurriculumStep. At one decimal every level the
                    # curriculum actually visits early on (0.99 down to 0.90) rounds into a single
                    # "pose_1.0" bucket, so the per-level breakdown - the whole point of it, since
                    # it says WHICH rung the agent is stuck on - reads as one flat line. Only levels
                    # actually sampled get a tag, so this stays sparse rather than emitting 100.
                    won = str(reason) == "Standing"
                    for key in (bucket, f"pose_{round(pose, 2):.2f}"):
                        counts = self._by_start.setdefault(key, {"episodes": 0, "successes": 0})
                        counts["episodes"] += 1
                        if won:
                            counts["successes"] += 1
        return True

    def _on_rollout_end(self) -> None:
        # Recorded here rather than per step because SB3 dumps the logger once per rollout; a
        # per-step record would just be overwritten by the next one before ever being flushed.
        for term, values in self._terms.items():
            if values:
                self.logger.record(f"reward/{term}", sum(values) / len(values))

        for term, values in self._stand.items():
            if values:
                self.logger.record(f"standing/{term}", sum(values) / len(values))

        for term, values in self._balls.items():
            if values:
                self.logger.record(f"ball/{term}", sum(values) / len(values))
        shots = self._balls.get("shots")
        hits = self._balls.get("hits")
        if shots and hits and sum(shots):
            self.logger.record("ball/hit_rate", sum(hits) / sum(shots))
        small = self._balls.get("small_shots")
        if shots and small and sum(shots):
            self.logger.record("ball/small_share", sum(small) / sum(shots))

        total = sum(self._reasons.values())
        for reason, count in self._reasons.items():
            self.logger.record(f"episode_end/{reason}", count / total)
        if total:
            self.logger.record("episode_end/episodes", total)

        # Emitted per bucket rather than as one ratio so that "no prone episodes ran at all" reads
        # as a missing/zero episode count rather than silently as a 0% success rate.
        for bucket, counts in self._by_start.items():
            if counts["episodes"]:
                self.logger.record(f"{bucket}/success", counts["successes"] / counts["episodes"])
                self.logger.record(f"{bucket}/episodes", counts["episodes"])

        self._terms.clear()
        self._stand.clear()
        self._balls.clear()
        self._reasons.clear()
        self._by_start.clear()


class RunManifestCallback(BaseCallback):
    """Writes a manifest.json into the run's TensorBoard directory.

    Exists because neither artefact SB3 produces answers "what was this trained on". The .zip
    stores step count and PPO hyperparameters; TensorBoard stores curves. Neither records the
    reward function, termination thresholds, action range, bone set, or the code revision - i.e.
    exactly the things that differ between experiments. Without it, comparing two runs weeks apart
    is guesswork.

    The environment-side half is not duplicated here: it is read from the env's info dict, which
    the Godot bridge fills from each strategy's own Describe(), so it cannot drift from the build.
    """

    def __init__(self, args):
        super().__init__()
        self._args = args
        self._env_config: dict = {}
        self._started = 0.0
        self._start_timesteps = 0
        self._path = ""

    def _manifest(self) -> dict:
        elapsed = time.time() - self._started
        rewards = [ep["r"] for ep in (self.model.ep_info_buffer or [])]
        lengths = [ep["l"] for ep in (self.model.ep_info_buffer or [])]
        return {
            "experiment_name": self._args.experiment_name,
            "started_utc": datetime.fromtimestamp(self._started, timezone.utc).isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "timesteps_done": int(self.model.num_timesteps),
            "timesteps_requested": self._args.timesteps,
            # Steps gained THIS session over this session's clock. Using cumulative num_timesteps
            # overstated throughput on every resumed run - and by a growing factor with each
            # resume, since the restored count is carried but the elapsed time is not.
            "steps_this_session": int(self.model.num_timesteps - self._start_timesteps),
            "steps_per_second": (
                round((self.model.num_timesteps - self._start_timesteps) / elapsed, 1)
                if elapsed > 0 else None
            ),
            "restored_from": self._args.restore or None,
            "cli_args": vars(self._args),
            "trainer": {
                "algo": "PPO",
                "policy": "MultiInputPolicy",
                "gamma": self.model.gamma,
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "n_epochs": self.model.n_epochs,
                "target_kl": self.model.target_kl,
                "ent_coef": self.model.ent_coef,
                # Read through the schedule, not self.learning_rate: the schedule is what PPO
                # actually applies, so this reports the value in effect even if the two diverge.
                "learning_rate": self.model.lr_schedule(1.0),
                "device": str(self.model.device),
                "observation_space": str(self.model.observation_space),
                "action_space": str(self.model.action_space),
            },
            # Filled from the Godot side - see RagdollRLBridge.GetStepInfo.
            "environment": self._env_config,
            "code": {
                "git_commit": _git("rev-parse", "HEAD"),
                "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                # A dirty tree means the commit alone does NOT identify what ran.
                "git_dirty": bool(_git("status", "--porcelain")),
            },
            "latest_metrics": {
                "ep_rew_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
                "ep_len_mean": round(sum(lengths) / len(lengths), 1) if lengths else None,
                "episodes_seen": len(rewards),
            },
        }

    def _write(self) -> None:
        if not self._path:
            return
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(self._manifest(), handle, indent=2)

    def _on_training_start(self) -> None:
        self._started = time.time()
        self._start_timesteps = self.model.num_timesteps
        log_dir = self.logger.get_dir() or self._args.experiment_dir
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "manifest.json")

        # Resuming continues INTO the checkpoint's own run directory (SB3 configure_logger:
        # "Continue training in the same directory"), so a fixed filename would silently replace
        # the previous session's provenance - the very record that exists to survive. Archive it
        # under the step count it reached, then let manifest.json mean "most recent session".
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as handle:
                    previous = json.load(handle)
                archived = os.path.join(
                    log_dir, f"manifest_{int(previous.get('timesteps_done', 0)):09d}.json")
                if not os.path.exists(archived):
                    os.replace(self._path, archived)
                    print(f"archived previous manifest -> {os.path.basename(archived)}")
            except (OSError, ValueError) as exc:
                print(f"warning: could not archive previous manifest: {exc}")

        self._write()
        print(f"manifest -> {self._path}")

    def _on_step(self) -> bool:
        # Match on a marker key rather than "first non-empty dict": VecMonitor injects its own
        # {"episode": ...} entry when an episode ends, which would otherwise be captured instead
        # of the Godot config.
        if not self._env_config:
            for info in self.locals.get("infos", []):
                if isinstance(info, dict) and "control_model" in info:
                    # A terminal transition can land inside the config-reporting window, so the
                    # per-episode diagnostic keys have to be stripped - the manifest records what
                    # the run was CONFIGURED with, not one arbitrary episode's numbers.
                    #
                    # terminal_observation especially: TruncationBootstrapWrapper puts a numpy
                    # array there, which json.dump cannot serialise, so letting it through would
                    # not merely pollute the manifest - it would kill every write for the run.
                    self._env_config = {
                        k: v
                        for k, v in info.items()
                        if k != "episode"
                        and not k.startswith("rew_")
                        and not k.startswith("stand_")
                        and not k.startswith("act_")
                        and k not in (
                            "episode_end_reason",
                            "started_standing",
                            "terminal_observation",
                            "TimeLimit.truncated",
                        )
                    }
                    self._write()
                    break
        return True

    def _on_rollout_end(self) -> None:
        # Refreshed each rollout so a crashed or Ctrl+C'd run still leaves current numbers.
        self._write()

    def _on_training_end(self) -> None:
        self._write()


class SelectivelyVisibleGodotEnv(StableBaselinesGodotEnv):
    """Vec env where only the first N Godot processes open a window.

    StableBaselinesGodotEnv forwards one show_window value to every process, so the stock options
    are "all 40 windows" or "none". Rendering one instance is what you actually want while
    training: a live view you can fly the camera around in, with the other 39 headless.

    Only __init__ is overridden; every other method is inherited unchanged. It does not call
    super().__init__ because that method's whole body IS the env construction we are replacing -
    calling it would launch a second set of processes.
    """

    def __init__(self, env_path, n_parallel=1, seed=0, visible_count=0, **kwargs):
        port = kwargs.pop("port", GodotEnv.DEFAULT_PORT)
        self.envs = [
            GodotEnv(
                env_path=env_path,
                convert_action_space=True,
                port=port + p,
                seed=seed + p,
                show_window=(p < visible_count),
                **kwargs,
            )
            for p in range(n_parallel)
        ]
        self.n_parallel = n_parallel
        self._check_valid_action_space()
        self.results = None


class TruncationBootstrapWrapper(VecEnvWrapper):
    """Restores value bootstrapping for episodes that end on the time limit.

    godot_rl computes a truncation flag and then throws it away: StableBaselinesGodotEnv.step
    collects `all_trunc` but returns only `all_term`, and GodotEnv.step_recv fills both slots from
    the same "done" boolean under a standing `# TODO update API to term, trunc`. So neither
    "TimeLimit.truncated" nor "terminal_observation" ever reaches SB3 - and PPO bootstraps only
    when BOTH keys are present (on_policy_algorithm.py, "Handle timeout by bootstrapping with value
    function"). Every episode end was therefore trained as absorbing, with V(s_T) = 0.

    Measured: episode_end/TimeLimit was 1.0 across all 12M steps of getup_v4_2, i.e. 100% of
    episode ends were mislabelled. That stayed nearly harmless only by accident - an 8 s episode
    always ended with the body prone, whose true continuation value really is about zero, which is
    why explained_variance still sat at 0.92. Shortening the episode moves truncation into states
    that DO still have value, which is precisely when the missing bootstrap starts to cost
    something. Hence this landing together with MaxEpisodeSeconds 8 -> 3, not after it.

    The terminal observation needs no reconstruction. RagdollRLBridge defers the physical reset by
    one tick specifically so that the obs Python receives on a done step is the pre-reset one (see
    its _resetPending branch and the ClearDone doc), so the obs already in hand IS the terminal obs.

    Sits inside VecMonitor, which computes ep_rew_mean before collect_rollouts adds gamma*V(s_T) to
    the reward array - so rollout/ep_rew_mean stays a clean measure of real environment reward
    rather than silently absorbing the bootstrap term.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self._announced = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for idx, done in enumerate(dones):
            if not done:
                continue

            # Timeouts only. "Standing" and "Inverted" are genuine absorbing states where V = 0 is
            # the correct target, and bootstrapping them would pay the agent for value it can never
            # collect. A missing reason key means the reward function has no IRlRewardDiagnostics,
            # so fall through to the old behaviour rather than guessing - which also guarantees we
            # never mutate the aliased `default_info` list godot_env.py builds ([{}] * n, every
            # element the same object) when the plugin sends no info at all.
            if infos[idx].get("episode_end_reason") != "TimeLimit":
                continue

            infos[idx]["terminal_observation"] = {key: value[idx].copy() for key, value in obs.items()}
            infos[idx]["TimeLimit.truncated"] = True

            # Announced once, not per episode: a wrapper that silently never engages is
            # indistinguishable from one that works, and this exact class of bug (a flag that is
            # computed and then dropped) is what it exists to fix.
            if not self._announced:
                self._announced = True
                print("truncation bootstrapping active - TimeLimit episodes now carry gamma*V(s_T)", flush=True)

        return obs, rewards, dones, infos


def main() -> None:
    # torch.onnx prints check-mark emoji during export. On Windows the console defaults to cp1252,
    # which raises UnicodeEncodeError mid-export and silently loses the .onnx half of a checkpoint.
    # errors="replace" keeps output readable without letting encoding kill a two-hour run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", required=True, help="Path to the exported Godot executable.")
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of Godot processes to run.")
    parser.add_argument("--speedup", type=int, default=8, help="Physics speed multiplier.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    # Anchored to this file, NOT the caller's working directory. A relative default silently
    # wrote logs to <cwd>/rl/runs, so launching from rl/scripts produced rl/scripts/rl/runs and
    # TensorBoard (pointed at the project-root path) showed an empty dashboard.
    _default_runs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    parser.add_argument("--experiment_dir", default=_default_runs)
    parser.add_argument("--experiment_name", default="ragdoll")
    parser.add_argument("--n_steps", type=int, default=256, help="PPO rollout length per env.")
    # 2048, not 256. With n_steps=256 across 40 envs a rollout holds 10,240 samples, so a
    # 256-sample batch means 40 minibatches x 10 epochs = 400 gradient steps over the SAME data.
    # Measured consequence across two runs: clip_fraction 0.43-0.45 and approx_kl ~0.05, against
    # healthy ranges of ~0.1-0.2 and ~0.01-0.02 - i.e. PPO's trust region was being blown through
    # on nearly half of all samples.
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.02,
        help="Abort an epoch early once the policy has moved this far in KL. Backstop for the "
        "same problem batch_size addresses, since the ideal batch size shifts with n_parallel.",
    )
    # 0.001, not the 0.0001 this ran at for its first 12M steps. train/std fell 1.0005 -> 0.6671
    # (t = -141) and was still steepening (-0.0148/1M, t = -294) with nothing holding it open,
    # while the goal state had never once been reached - i.e. exploration was closing before there
    # was anything to converge onto. Not 0.01: entropy for this 36-dim policy at std 0.667 is 36.5,
    # so 0.01 would contribute 0.365 against a total loss of 0.14 and simply swamp it. Raise to
    # 0.003 if std keeps falling.
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.001,
        help="Entropy bonus weight. Higher keeps the action distribution wide for longer, which "
        "matters while the task's reward has never been reached and there is nothing to exploit.",
    )
    # 1.5e-4, half of SB3's 3e-4 default. At 3e-4 the trust region was the binding constraint on
    # every single iteration: approx_kl sat at 0.028-0.032 against target_kl 0.02, so SB3's
    # 1.5 * target_kl abort fired at "Early stopping at step 3" every rollout and 7 of 10 epochs
    # were discarded. Widening target_kl would silence the message by removing the safety valve;
    # halving the step size instead makes the updates fit inside the region PPO was configured
    # with, so all 10 epochs run on well-conditioned gradients.
    #
    # This does NOT promise faster learning - it makes the optimisation well-posed. The wall-clock
    # question is whether 10 small epochs beat 3 large ones; judge it on the standing/icp slope
    # over ~2M steps, not on approx_kl alone.
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.5e-4,
        help="Adam step size. Lower keeps approx_kl inside target_kl so PPO runs all n_epochs "
        "instead of aborting early every iteration.",
    )
    parser.add_argument("--save_model_path", default="", help="Optional .zip checkpoint path.")
    parser.add_argument("--onnx_export_path", default="", help="Optional .onnx export path (M6).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Render ONE training instance in a window (the other processes stay headless). "
        "Note: step() waits for every env each tick, so the rendering instance gates the whole "
        "batch's throughput - expect a slowdown. It also runs at --speedup like the rest, so "
        "lower --speedup if you want watchable motion.",
    )
    parser.add_argument(
        "--restore",
        default="",
        help="Path to a .zip checkpoint to continue training from instead of starting fresh.",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds of wall clock, whatever --timesteps says. 0 disables. "
        "Use this when you want a run of a known DURATION, since throughput is not predictable.",
    )
    parser.add_argument(
        "--save_every_seconds",
        type=float,
        default=60.0,
        help="Wall-clock interval between checkpoint saves (matched .zip + .onnx). 0 disables. "
        "Checkpoints are step-stamped inside the run directory and never overwrite each other.",
    )
    args = parser.parse_args()

    env = SelectivelyVisibleGodotEnv(
        env_path=args.env_path,
        n_parallel=args.n_parallel,
        speedup=args.speedup,
        seed=args.seed,
        visible_count=1 if args.viz else 0,
    )
    # Order matters: the truncation fix sits INSIDE VecMonitor so that ep_rew_mean is measured on
    # the environment's own reward, before PPO adds the bootstrap term to it.
    env = VecMonitor(TruncationBootstrapWrapper(env))

    if args.restore:
        # Loads policy AND optimizer state, so training genuinely continues rather than restarting
        # with a warm-started policy.
        model = PPO.load(
            args.restore,
            env=env,
            tensorboard_log=args.experiment_dir,
        )
        # PPO.load restores the hyperparameters saved INTO the checkpoint, so without this a
        # resume would silently keep the old batch_size/target_kl/ent_coef and undo the fix.
        # All three are plain attributes read fresh at loss time (ppo.py train()), so assigning
        # them post-load is sufficient - no re-initialisation needed.
        model.batch_size = args.batch_size
        model.target_kl = args.target_kl
        model.ent_coef = args.ent_coef

        # learning_rate is NOT one of those, and assigning it alone is a silent no-op. PPO reads
        # the step size through self.lr_schedule(self._current_progress_remaining) in
        # OnPolicyAlgorithm._update_learning_rate; self.learning_rate is only the value that
        # _setup_lr_schedule() converts INTO that callable. Verified against SB3 2.4.0 on a real
        # checkpoint: setting the attribute left lr_schedule(1.0) at the restored 3e-4, and only
        # the re-setup below moved it. Getting this wrong looks exactly like the change working.
        model.learning_rate = args.learning_rate
        model._setup_lr_schedule()
        print(
            f"restored from {args.restore} at {model.num_timesteps} timesteps "
            f"(batch_size={model.batch_size}, target_kl={model.target_kl}, "
            f"ent_coef={model.ent_coef}, learning_rate={model.lr_schedule(1.0):g})"
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            verbose=2,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            target_kl=args.target_kl,
            tensorboard_log=args.experiment_dir,
        )

    checkpoints = PeriodicCheckpointCallback(args.save_every_seconds)
    try:
        # reset_num_timesteps=False keeps the restored step count so TensorBoard curves continue
        # from where the previous run stopped instead of overwriting from zero.
        model.learn(
            args.timesteps,
            tb_log_name=args.experiment_name,
            reset_num_timesteps=not args.restore,
            callback=[
                RunManifestCallback(args),
                RewardDecompositionCallback(),
                checkpoints,
                TimeLimitCallback(args.max_seconds),
            ],
        )
    except KeyboardInterrupt:
        # SB3's learn() has no exception handling around its rollout loop, so a Ctrl+C propagates
        # straight out and callback.on_training_end() - which writes the final checkpoint - never
        # runs. Measured: an interrupted 52-minute session left no final_* file at all, and would
        # have lost everything since the last periodic save. Saving here makes stopping safe at any
        # moment rather than only at a save boundary.
        print("\ninterrupted - saving before exit", flush=True)
        try:
            checkpoints.save("interrupted")
        except Exception as exc:  # noqa: BLE001 - never let the rescue save mask the interrupt
            print(f"warning: interrupt save failed: {exc}")
    finally:
        if args.save_model_path:
            model.save(args.save_model_path)
            print(f"saved model to {args.save_model_path}")
        if args.onnx_export_path:
            from godot_rl.wrappers.onnx.stable_baselines_export import (
                export_model_as_onnx,
                verify_onnx_export,
            )

            export_model_as_onnx(model, args.onnx_export_path)
            # Numerically compares ONNX inference against the torch policy, so a silently
            # wrong export is caught here rather than looking like bad behaviour in-engine.
            verify_onnx_export(model, args.onnx_export_path)
            print(f"exported onnx to {args.onnx_export_path}")
        env.close()


if __name__ == "__main__":
    main()
