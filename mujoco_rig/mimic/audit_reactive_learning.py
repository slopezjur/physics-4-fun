"""Measure reward activation, input clipping and action authority on failed hits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from envs.base_env import DoneFlags, EnvMode
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .reactive_ball_task import ReactiveBallTask
    torch.set_num_threads(1)
    mp_util.init(0, 1, args.device, None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, 8, args.device, ball_path=args.ball_model,
                            speed_max=1.5, episode_seconds=8., recovery_mode=args.recovery_mode)
    task.set_mode(EnvMode.TEST)
    task.reset()
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    agent = PPOAgent(config, task, args.device)
    agent.load(str(args.checkpoint))
    task.reset()
    task.shot_target_body[:] = task.target_indices[task.ids % 3]
    launch_targets = [task.rig.model.body(int(i)+1).name for i in task.shot_target_body.cpu()]
    actor = StandPolicy(agent).eval()
    active = torch.ones(8, dtype=torch.bool, device=args.device)
    rows = {k: [] for k in ("observation", "action", "hit", "reward", "settled_steps", "active", "done", "qpos")}
    resets = {k: [] for k in ("qpos", "qvel", "ground", "pending", "control_active", "posture")}
    if args.save_recovery_resets and args.device != "cpu":
        raise ValueError("Recovery reset capture requires native physical state")
    with torch.no_grad():
        for _ in range(480):
            obs = task.observations()
            action = actor(obs)
            _, reward, done, _ = task.step(action)
            if args.save_recovery_resets and 120 <= int(task.steps.max()) <= 168:
                control = task.engine.policy_control
                for i, data in enumerate(task.engine.datas):
                    if not active[i] or not task.engine.hit[i] or task.steps[i] % 4 or done[i] != DoneFlags.NULL.value:
                        continue
                    ball = task.engine.ball_body
                    contacting = any(ball in task.engine.rig.model.geom_bodyid[c.geom] for c in data.contact[:data.ncon])
                    if contacting:
                        continue
                    state = dict(qpos=data.qpos[:46].copy(), qvel=data.qvel[:45].copy(),
                                 ground=task.engine.get_ground_contact_forces(0)[i].numpy().copy(),
                                 pending=np.stack([control.pending[(control.cursor+s) % len(control.pending), i].numpy().copy()
                                                   for s in range(len(control.pending))]),
                                 control_active=control.active[i].numpy().copy(), posture=i)
                    for key, value in state.items():
                        resets[key].append(value)
            for key, value in dict(observation=obs, action=action, hit=task.engine.hit,
                                   reward=reward, settled_steps=task.settled_steps,
                                   active=active, done=done, qpos=task.engine.native_qpos()).items():
                rows[key].append(value.cpu().numpy().copy())
            active &= done == DoneFlags.NULL.value
            if not active.any():
                break
            failed = torch.nonzero(done != DoneFlags.NULL.value).flatten()
            if len(failed):
                task.reset(failed)
                task.shot_enabled[failed] = False
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    mask = arrays["active"] & arrays["hit"]
    normalized = (arrays["observation"] - actor.obs_norm._mean.cpu().numpy()) / actor.obs_norm._std.cpu().numpy()
    clipped = np.abs(normalized) >= 10.
    contract = task.observation_contract()
    report = dict(checkpoint_sha256=sha256(args.checkpoint), device=args.device,
                  recovery_mode=args.recovery_mode, reward_contract=task.recovery_reward.contract(),
                  targets=launch_targets,
                  observed_hits=int(np.any(mask, axis=0).sum()),
                  post_hit_samples=int(mask.sum()),
                  post_hit_settled_fraction=float(np.mean(arrays["settled_steps"][mask] > 0)),
                  maximum_settled_seconds=float(arrays["settled_steps"][mask].max() * task.dt),
                  post_hit_reward_mean=float(arrays["reward"][mask].mean()),
                  post_hit_action_saturation_fraction=float((np.abs(arrays["action"][mask]) >= .999).mean()),
                  clipping_by_group={g["name"]: float(clipped[..., g["offset"]:g["offset"]+g["width"]][mask].mean())
                                     for g in contract["observation_layout"]},
                  saturation_by_actuator={name: float((np.abs(arrays["action"][mask, i]) >= .999).mean())
                                          for i, name in enumerate(contract["actuator_names"])},
                  cases=[dict(posture=i, hit=bool(arrays["hit"][:, i].any()),
                              seconds=float(arrays["active"][:, i].sum() * task.dt),
                              max_settled_seconds=float(arrays["settled_steps"][:, i].max() * task.dt))
                         for i in range(8)])
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "trace.npz", **arrays)
    if args.save_recovery_resets:
        if set(resets["posture"]) != set(range(8)):
            raise RuntimeError("Recovery reset bank must cover every posture")
        np.savez_compressed(args.out / "recovery_resets.npz", **{k: np.asarray(v) for k, v in resets.items()},
                            schema="physical_post_hit_resets_v1", model_sha256=sha256(rig.path),
                            reference_sha256=sha256(args.reference), ball_sha256=sha256(args.ball_model),
                            checkpoint_sha256=sha256(args.checkpoint))
        report["recovery_reset_samples"] = len(resets["posture"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recovery-mode", choices=("settling_v2", "capture_v3"), default="settling_v2")
    parser.add_argument("--save-recovery-resets", action="store_true")
    run(parser.parse_args())
