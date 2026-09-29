"""Native 2 m/s step assay and balanced physical post-hit reset capture."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from envs.base_env import DoneFlags
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .policy import StandPolicy
    from .balance_observation import task_classes
    _, ReactiveBallTask = task_classes(json.loads((args.checkpoint.parent / "contract.json").read_text()))
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                            speed_max=args.speed, episode_seconds=12., recovery_mode=args.recovery_mode)
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.checkpoint))
    bank = {k: [] for k in ("qpos", "qvel", "ground", "pending", "control_active", "posture", "direction")}
    actor = StandPolicy(agent).eval()
    feedback = None
    if args.feedback:
        with np.load(args.feedback) as data:
            for key, path in (("checkpoint_sha256", args.checkpoint), ("reference_sha256", args.reference),
                              ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
                if data[key].item() != sha256(path):
                    raise ValueError("Feedback provenance mismatch: " + key)
            indices = data["actuator_indices"].copy()
            gains = torch.from_numpy(data["gains"].copy())
            if gains.shape != (len(indices), 4) or not torch.isfinite(gains).all():
                raise ValueError("Malformed feedback gains")
            feedback = indices, gains
    exploration = None
    chosen = None
    authority = None
    if args.authority_report:
        if args.exploration_report or args.feedback:
            raise ValueError("Authority verification cannot combine independent discovery probes")
        probe = json.loads(args.authority_report.read_text())
        if probe["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Authority probe uses another actor")
        for key,path in (("reference_sha256",args.reference),("ball_model_sha256",args.ball_model)):
            if probe[key] != sha256(path):
                raise ValueError("Authority probe provenance mismatch: " + key)
        with np.load(args.authority_report.with_name("successful_exploration.npz")) as trace:
            width=int(trace["per_setting"])
            worlds=trace["world_indices"].tolist()
            selected=[]
            for i in range(32):
                pool=[w for w in worlds if w%32==i and probe["settings"][w//width]["residual_radians"]==.25]
                selected.append(max(pool,key=lambda w:probe["cases"][w]["final_settled_seconds"]) if pool else None)
            exploration=torch.zeros(len(trace["perturbation"]),32,30)
            extended=torch.zeros(32,dtype=torch.bool)
            for i,w in enumerate(selected):
                if w is not None:
                    exploration[:,i]=torch.from_numpy(trace["perturbation"][:,worlds.index(w)].copy())
                    extended[i]=probe["settings"][w//width]["support_activation"]
            authority=extended
            chosen=selected
    if args.exploration_report:
        probe = json.loads(args.exploration_report.read_text())
        if probe["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Exploration probe uses another actor")
        chosen = []
        for direction in range(4):
            for posture in range(8):
                pool = [i for i,c in enumerate(probe["cases"]) if c["posture"] == posture and c["direction"] == direction]
                chosen.append(max(pool, key=lambda i: (probe["cases"][i]["recovered_with_step"],
                    sum(probe["cases"][i]["steps_by_foot"]) > 0, probe["cases"][i]["survival_seconds"])))
        with np.load(args.exploration_report.with_name("actions.npz")) as trace:
            exploration = torch.from_numpy(trace["exploration"][:, chosen].copy())
    observed, commands, masks, rewards = [], [], [], []
    control_step = 0

    def policy(obs):
        nonlocal control_step
        action = actor(obs)
        if authority is not None:
            from .probe_recovery_authority import recovery_action, support_gate
            alternative=recovery_action(actor,obs,support_gate(obs))
            action=torch.where(authority[:,None],alternative,action)
        if feedback is not None:
            indices, gains = feedback
            features = torch.cat((obs[:, 9:11] / .5, obs[:, 12:14] / 2.), -1).clamp(-2, 2)
            correction = (features @ gains.T).clamp(-.8, .8)
            if args.feedback_gated:
                speed = torch.linalg.vector_norm(obs[:, 9:11], dim=-1)
                rotation = torch.linalg.vector_norm(obs[:, 12:14], dim=-1)
                gate = torch.maximum((speed - .2) / .2, (rotation - .5) / .5).clamp(0, 1)
                correction *= gate[:, None]
            action[:, indices] += correction
            action = action.clamp(-1, 1)
        if exploration is not None and control_step < len(exploration):
            action = action + exploration[control_step]
        observed.append(obs.numpy().copy())
        commands.append(action.numpy().copy())
        control_step += 1
        return action

    def capture(task, step, active, done):
        masks.append(active.numpy().copy())
        rewards.append(task.last_transition_reward.numpy().copy())
        if not args.capture or step % 8 or not 112 <= step <= 240:
            return
        control = task.engine.policy_control
        for i, data in enumerate(task.engine.datas):
            if not active[i] or not task.engine.hit[i] or done[i] != DoneFlags.NULL.value:
                continue
            if data.qpos[2] < .73 or 1 - 2 * (data.qpos[4]**2 + data.qpos[5]**2) < .9:
                continue
            if any(task.engine.ball_body in task.engine.rig.model.geom_bodyid[c.geom]
                   for c in data.contact[:data.ncon]):
                continue
            values = dict(qpos=data.qpos[:46].copy(), qvel=data.qvel[:45].copy(),
                ground=task.engine.get_ground_contact_forces(0)[i].numpy().copy(),
                pending=np.stack([control.pending[(control.cursor+s) % len(control.pending), i].numpy().copy()
                                  for s in range(len(control.pending))]),
                control_active=control.active[i].numpy().copy(), posture=i % 8, direction=i // 8)
            for key, value in values.items():
                bank[key].append(value)

    report, _ = evaluate(task, policy, selection_cases(args.speed, 12., task.dt), capture)
    args.out.mkdir(parents=True, exist_ok=False)
    report.update(checkpoint_sha256=sha256(args.checkpoint), model_sha256=sha256(rig.path),
                  reference_sha256=sha256(args.reference), ball_sha256=sha256(args.ball_model))
    if args.feedback:
        report["feedback_sha256"] = sha256(args.feedback)
        report["feedback_gated"] = args.feedback_gated
    if exploration is not None:
        report["exploration_probe_sha256"] = sha256(args.exploration_report or args.authority_report)
        report["exploration_worlds"] = chosen
        if authority is not None:
            report["authority_probe"] = str(args.authority_report)
            report["authority_support_activation"] = authority.tolist()
            report["authority_scope"] = "Native replay at original 0.25 rad range; stochastic discovery, not policy admission"
    report["recovery_mode"] = args.recovery_mode
    report["source_hashes"] = {name: sha256(Path(__file__).with_name(name)) for name in
                              ("audit_recovery_steps.py", "reactive_ball_task.py", "step_reward.py", "gated_recovery_net.py",
                               "probe_recovery_authority.py")}
    np.savez_compressed(args.out / "trace.npz", observation=np.asarray(observed), action=np.asarray(commands),
                        active=np.asarray(masks), reward=np.asarray(rewards))
    if args.capture:
        if set(bank["posture"]) != set(range(8)):
            raise RuntimeError("Incomplete posture coverage")
        np.savez_compressed(args.out / "recovery_resets.npz", **{k: np.asarray(v) for k,v in bank.items()},
            schema="physical_post_hit_resets_v1", **{k: report[k] for k in
            ("checkpoint_sha256", "model_sha256", "reference_sha256", "ball_sha256")})
        report["reset_samples"] = len(bank["posture"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("recovered_hits", "recovered_with_step")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--speed", type=float, default=2.)
    parser.add_argument("--recovery-mode", choices=("capture_v3", "step_v4", "step_v5"), default="capture_v3")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--exploration-report", type=Path)
    parser.add_argument("--authority-report", type=Path)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--feedback-gated", action="store_true")
    run(parser.parse_args())
