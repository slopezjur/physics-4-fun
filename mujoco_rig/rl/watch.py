"""Watch a policy run, in a window, on the engine that scores it.

**Why this exists.** Every judgement about this dummy has come from a log line or a 40-second
headless report. The Isaac track had `isaac_lab_3/scripts/watch.ps1` for exactly this reason and it
paid for itself repeatedly - a body that measures well and moves wrongly is a common and expensive
failure, and no scalar catches it.

It runs `PerturbEnv` / `WalkEnv` on MuJoCo's **C engine**, which is:

  * the engine `eval.py` scores on, so what is on screen is what the numbers describe, and
  * the engine Godot drives through P/Invoke, so it is also what ships.

The GPU trainer's `mujoco_warp` is deliberately not an option here. It is float32, it diverges from
the C engine (`parity_gpu.py` measures how far), and watching the engine that trained a policy tells
you nothing about the one that will run it.

    python mujoco_rig/rl/watch.py --checkpoint logs/mujoco/<run>/model_300.pt
    python mujoco_rig/rl/watch.py --zero_action            # the unpowered body, the baseline
    python mujoco_rig/rl/watch.py --reload                 # follow a training run

Keys, matching what `MujocoDummy._UnhandledInput` binds so the two surfaces behave alike:

    R      reset the episode          B      fire a ball now
    Space  shove the pelvis           P      pause / resume
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mujoco                                                        # noqa: E402
import mujoco.viewer                                                 # noqa: E402

from env_config import FALL_FRACTION                                 # noqa: E402
from perturb_env import PerturbEnv                                   # noqa: E402
from observation_contract import LEGACY, checkpoint_version          # noqa: E402
from ppo import ActorCritic                                          # noqa: E402
from walk_env import WalkEnv                                         # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The same definitions MjGaitMetrics uses on the C# side, so the two surfaces cannot report
# different numbers for the same run.
FOOT_LIFT = 0.02          # metres above its rest height before a foot counts as airborne
TELEMETRY_HZ = 4.0


def load_policy(path, num_obs=None):
    """The actor from a checkpoint, plus what it was trained on."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_version(ck)
    net = ActorCritic(ck["num_obs"], ck["num_actions"])
    net.load_state_dict(ck["model"])
    net.eval()
    if num_obs is not None and ck["num_obs"] != num_obs:
        raise SystemExit(f"{pathlib.Path(path).name} is {ck['num_obs']}-wide, the environment is "
                         f"{num_obs}. The policy was trained on a different rig.")
    return net, ck


class Telemetry:
    """The console block, carrying what RagdollTelemetryHud shows for the Jolt body."""

    def __init__(self, env, checkpoint, zero_action):
        self.env = env
        self.checkpoint = pathlib.Path(checkpoint).name if checkpoint else "none (zero action)"
        self.zero_action = zero_action
        self.loaded_at = time.strftime("%H:%M:%S")
        self.reloads = 0
        self.resets = 0
        self.samples = 0
        self.upright_samples = 0
        self.single = 0
        self.flight = 0
        self.shots = 0
        self.last_target = "-"
        self.last_impulse = 0.0
        self.peak_impulse = 0.0
        self.started = time.perf_counter()
        self.sim_time = 0.0
        self._rest_foot = None
        self._lines = 0

    def sample(self, d, tau, dt):
        env = self.env
        self.sim_time += dt
        self.samples += 1
        if d.xpos[env.pelvis][2] >= FALL_FRACTION * env.rest_pelvis_z:
            self.upright_samples += 1
        if self._rest_foot is None:
            self._rest_foot = (d.xpos[env.foot_l][2], d.xpos[env.foot_r][2])
        up_l = d.xpos[env.foot_l][2] - self._rest_foot[0] > FOOT_LIFT
        up_r = d.xpos[env.foot_r][2] - self._rest_foot[1] > FOOT_LIFT
        self.single += up_l ^ up_r
        self.flight += up_l and up_r
        self.tau = tau

    def on_reset(self):
        self.resets += 1
        self.sim_time = 0.0

    def render(self, d, paused):
        env = self.env
        m = env.model
        up = d.xmat[env.pelvis].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        tilt = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
        pel_v = np.linalg.norm(d.cvel[env.pelvis][3:6])
        n = max(1, self.samples)

        # Weight transfer, from the vertical force each foot's contacts carry.
        f_l, f_r = self._foot_load(d)
        total = max(f_l + f_r, 1e-6)

        tau = np.abs(getattr(self, "tau", np.zeros(1)))
        peak_human = float(np.max(env.force_limit))
        realtime = self.sim_time / max(time.perf_counter() - self.started, 1e-6)

        block = [
            "--- TELEMETRY ---" + ("   [PAUSED]" if paused else ""),
            f"Brain:      {self.checkpoint}   (loaded {self.loaded_at}"
            + (f", reloads {self.reloads}" if self.reloads else "") + ")",
            f"Engine:     MuJoCo {mujoco.__version__} CPU @ {1.0 / m.opt.timestep:.0f} Hz"
            f"   x{realtime:.2f} realtime",
            f"Episode:    {self.sim_time:5.1f} s   resets {self.resets}"
            f"   upright {100.0 * self.upright_samples / n:5.1f}%",
            f"Pelvis:     {d.xpos[env.pelvis][2]:.3f} m (rest {env.rest_pelvis_z:.3f})"
            f"   speed {pel_v:4.2f} m/s   tilt {tilt:5.1f} deg"
            f"   (falls below {FALL_FRACTION * env.rest_pelvis_z:.3f} m)",
            f"Feet:       L {'ON ' if f_l > 1.0 else 'AIR'}  R {'ON ' if f_r > 1.0 else 'AIR'}"
            f"   weight L {100.0 * f_l / total:3.0f}% / R {100.0 * f_r / total:3.0f}%"
            f"   single {100.0 * self.single / n:4.1f}%   flight {100.0 * self.flight / n:4.1f}%",
            f"Muscle:     mean {tau.mean():5.1f} N.m   peak {tau.max():5.1f}"
            f"   {100.0 * tau.mean() / peak_human:4.1f}% of human strength",
        ]
        if env.ball >= 0:
            nxt = max(0.0, float(env.next_ball[0]) - self.sim_time)
            block.append(
                f"Ball gun:   shot {self.shots} -> {self.last_target:<10}"
                f" delivered {self.last_impulse:5.1f} N.s (peak {self.peak_impulse:5.1f})"
                f"   next in {nxt:4.1f} s")
        block.append("Keys:       R reset   B ball   Space shove   P pause")

        if self._lines:
            sys.stdout.write(f"\033[{self._lines}A")
        for line in block:
            sys.stdout.write("\033[2K" + line + "\n")
        sys.stdout.flush()
        self._lines = len(block)

    def _foot_load(self, d):
        """Vertical contact force under each foot, for the weight-transfer readout."""
        env = self.env
        m = env.model
        loads = [0.0, 0.0]
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for body, slot in ((env.foot_l, 0), (env.foot_r, 1)):
                if m.geom_bodyid[c.geom1] == body or m.geom_bodyid[c.geom2] == body:
                    mujoco.mj_contactForce(m, d, i, wrench)
                    loads[slot] += abs(wrench[0])
        return loads[0], loads[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    # **The task comes from the CHECKPOINT by default, not from config.ps1.** config's $Task
    # is the TRAINING task, and the first run of this viewer opened the balance brain on the
    # walk task because of it - no ball gun, wrong question, and nothing on screen said so.
    # A checkpoint records what it was trained for; that is the only honest source.
    p.add_argument("--task", choices=("auto", "perturb", "walk"), default="auto")
    p.add_argument("--zero_action", action="store_true",
                   help="watch the UNPOWERED body - the baseline every result is read against")
    p.add_argument("--reload", action="store_true",
                   help="reload the checkpoint when it changes, to follow a training run")
    p.add_argument("--ball_speed", type=float, default=0.0)
    p.add_argument("--ball_every", type=float, nargs=2, default=(3.0, 3.0))
    p.add_argument("--command", type=float, nargs=3, default=(0.6, 0.0, 0.0),
                   help="walk only: forward m/s, lateral m/s, turn rad/s")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.checkpoint and not args.zero_action:
        raise SystemExit("give --checkpoint, or --zero_action to watch the unpowered body")

    ck = None
    if args.checkpoint:
        _, ck = load_policy(args.checkpoint)
    task = args.task
    if task == "auto":
        task = (ck or {}).get("task", "perturb")
    # The difficulty the checkpoint actually reached travels with the weights: watching a policy at
    # a difficulty it never trained at is not a view of that policy.
    speed = args.ball_speed or (float(ck.get("ball_speed", 6.0)) if ck else 6.0)

    if task == "walk":
        env = WalkEnv(num_envs=1, episode_seconds=1e6, seed=args.seed)
        env.set_command(*args.command)
    else:
        env = PerturbEnv(num_envs=1, episode_seconds=1e6, seed=args.seed,
                         model="dummy_ball.xml", ball_every=tuple(args.ball_every),
                         ball_speed=speed, observation_version=checkpoint_version(ck) if ck else LEGACY)
    net = None
    if args.checkpoint:
        net, _ = load_policy(args.checkpoint, num_obs=env.num_obs)

    obs = env.reset_all()
    m, d = env.model, env.datas[0]
    tel = Telemetry(env, args.checkpoint, args.zero_action)
    step_dt = env.dt * env.decimation
    state = {"paused": False, "quit": False}

    def on_key(code):
        # GLFW key codes; the viewer forwards everything it does not consume itself.
        if code == ord("R"):
            env.reset_idx([0])
            tel.on_reset()
        elif code == ord("B") and env.ball >= 0:
            env.fire_ball(d, 0)
            tel.shots += 1
            tel.last_target = "manual"
        elif code == ord(" "):
            d.xfrc_applied[env.pelvis, 0] += 400.0
        elif code == ord("P"):
            state["paused"] = not state["paused"]

    ball_geoms = ([g for g in range(m.ngeom) if m.geom_bodyid[g] == env.ball]
                  if env.ball >= 0 else [])
    print(f"[watch] {args.checkpoint or 'zero action'} on CPU MuJoCo, task {task}"
          + (f", ball {speed:.2f} m/s every {args.ball_every[0]:.0f}-{args.ball_every[1]:.0f}s"
             if env.ball >= 0 else ""))
    print("[watch] close the window to exit\n")

    mtime = pathlib.Path(args.checkpoint).stat().st_mtime if args.checkpoint else 0.0
    last_draw = 0.0
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        while viewer.is_running():
            frame = time.perf_counter()
            if not state["paused"]:
                shots_before = env.next_ball[0]
                p_before = env.com_velocity(d) * env.total_mass
                with torch.no_grad():
                    action = (torch.zeros(1, env.num_actions) if net is None
                              else net.actor(obs))
                obs, _, dones, _ = env.step(action)
                if bool(dones[0]):
                    # The env resets a fallen body itself; the counter has to see that, or the
                    # episode clock runs on across a fall and every per-episode number is wrong.
                    tel.on_reset()
                # The pelvis shove is a one-shot impulse, not a permanent force.
                d.xfrc_applied[env.pelvis, :3] = 0.0
                tel.sample(d, d.actuator_force[env.act_idx], step_dt)

                if env.ball >= 0:
                    if env.next_ball[0] != shots_before:
                        tel.shots += 1
                        tel.last_impulse = 0.0
                    touching = any(
                        (c.geom1 in ball_geoms or c.geom2 in ball_geoms)
                        and m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
                        and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE
                        for c in d.contact[:d.ncon])
                    if touching:
                        p_after = env.com_velocity(d) * env.total_mass
                        hit = float(np.linalg.norm(p_after - p_before))
                        tel.last_impulse = max(tel.last_impulse, hit)
                        tel.peak_impulse = max(tel.peak_impulse, hit)

            viewer.sync()
            now = time.perf_counter()
            if now - last_draw > 1.0 / TELEMETRY_HZ:
                tel.render(d, state["paused"])
                last_draw = now
            # Run at wall-clock speed; the point is to WATCH it.
            sleep = step_dt - (time.perf_counter() - frame)
            if sleep > 0:
                time.sleep(sleep)

            if args.reload and args.checkpoint:
                current = pathlib.Path(args.checkpoint).stat().st_mtime
                if current != mtime:
                    mtime = current
                    try:
                        net, _ = load_policy(args.checkpoint, num_obs=env.num_obs)
                        tel.reloads += 1
                        tel.loaded_at = time.strftime("%H:%M:%S")
                    except Exception as exc:                     # a half-written file
                        print(f"\n[watch] reload skipped: {exc}")

    print("\n[watch] window closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
