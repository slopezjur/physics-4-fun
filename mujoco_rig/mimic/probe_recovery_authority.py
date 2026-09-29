"""Discovery assay: current-state activation and motor target range, with no step clock."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def support_gate(observation):
    capture = ((observation[..., 362] - .001) / .019).clamp(0, 1)
    loads = observation[..., 93:95] * 1000
    transfer = ((30 - loads.amin(-1)) / 25).clamp(0, 1)
    transfer *= (loads.amax(-1) / 50).clamp(0, 1)
    return torch.maximum(capture, transfer)


def recovery_action(actor, observation, gate):
    normalized = actor.obs_norm.normalize(observation)
    network = actor.actor
    raw = normalized * network.observation_std + network.observation_mean
    features = torch.cat((raw[..., 9:11] / .5, raw[..., 12:14] / 2.), -1).clamp(-2, 2)
    correction = (network.velocity(features) + network.recovery(normalized)).clamp(-.8, .8)
    correction *= network.leg_mask * gate[..., None] / network.action_std
    mean = network.base_mean(network.base(normalized[..., :362])) + correction
    return actor.action_norm.unnormalize(actor.mean(mean)).clamp(-1, 1)


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import CONTRACT, task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    if contract["observation_contract"] != CONTRACT:
        raise ValueError("Require the capture-observation seed")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    _, Task = task_classes(contract)
    settings = [(radius, supported, sigma) for radius in (.25, .5) for supported in (False, True)
                for sigma in (0., .3, .6, .9)]
    replicas = 4
    per_setting = 32 * replicas
    count = len(settings) * per_setting
    task = Task(rig, args.reference, count, "cuda:0", ball_path=args.ball_model,
                speed_max=2., episode_seconds=12., recovery_mode="step_v5")
    validate_contract(contract, task.observation_contract())
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, task.device)
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    residual = torch.tensor([r for r,g,s in settings], device=task.device).repeat_interleave(per_setting)[:,None]
    extended = torch.tensor([g for r,g,s in settings], device=task.device).repeat_interleave(per_setting)
    sigma = torch.tensor([s for r,g,s in settings], device=task.device).repeat_interleave(per_setting)[:,None]
    task.engine.policy_control.residual_radians = residual
    noise = torch.zeros(count, 30, device=task.device)
    rho = float(np.exp(-task.dt / .2))
    perturbations = []

    @torch.no_grad()
    def policy(observation):
        nonlocal noise
        capture = ((observation[:,362]-.001)/.019).clamp(0,1)
        gate = torch.where(extended, support_gate(observation), capture)
        action = recovery_action(actor, observation, gate)
        paired = torch.randn(per_setting, 30, device=task.device).repeat(len(settings), 1)
        noise = rho * noise + np.sqrt(1-rho*rho) * paired
        perturbation = noise * sigma * actor.actor.leg_mask * gate[:,None]
        perturbations.append(perturbation.cpu().numpy())
        # Frozen nominal actions retain their original physical target displacement.
        # Only exploratory authority changes with the tested controller range.
        return (action * (.25 / residual) + perturbation).clamp(-1,1)

    cases = [dict(c, name=f"setting-{g}-replica-{j}-{c['name']}") for g in range(len(settings))
             for j in range(replicas) for c in selection_cases(2.,12.,task.dt)]
    report, _ = evaluate(task, policy, cases)
    summaries = []
    for i,(radius,supported,sigma_value) in enumerate(settings):
        rows = report["cases"][i*per_setting:(i+1)*per_setting]
        summaries.append(dict(setting=i,residual_radians=radius,support_activation=supported,
            exploration_std=sigma_value,episodes=len(rows),recovered=sum(c["recovered"] for c in rows),
            lifted=sum(any(c["lifted_feet"]) for c in rows),
            recovered_with_step=sum(c["recovered_with_step"] for c in rows)))
    report.update(settings=summaries, checkpoint_sha256=sha256(args.checkpoint),
        reference_sha256=sha256(args.reference), ball_model_sha256=sha256(args.ball_model),
        seed=args.seed, correlation_seconds=.2,
        scope="Stochastic physical discovery only; no deployment, selected foot, hit flag or action clock")
    args.out.mkdir(parents=True)
    indices = [i for i,c in enumerate(report["cases"]) if c["recovered_with_step"]]
    np.savez_compressed(args.out / "successful_exploration.npz", world_indices=indices,
                        perturbation=np.asarray(perturbations)[:,indices],
                        settings=np.asarray(settings,dtype=float), per_setting=per_setting)
    (args.out/"report.json").write_text(json.dumps(report,indent=2))
    (args.out/"runner.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summaries),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit","checkpoint","reference","ball-model","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--seed",type=int,default=280912)
    run(parser.parse_args())
