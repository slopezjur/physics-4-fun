"""Native development audit of articulation and grounded preparation credit."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .capture_reward import capture_distance
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate, selection_cases
    from .recovery_quality import RecoveryQuality
    from .rig import Rig
    from .step_reward import swing_quality
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    _, Task = task_classes(contract)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    results = []
    args.out.mkdir(parents=True, exist_ok=False)
    for speed in args.speeds:
        task = Task(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                    recovery_mode=args.recovery_mode, speed_min=2., speed_max=max(2., speed), episode_seconds=12.)
        agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
        agent.load(str(args.checkpoint))
        quality = RecoveryQuality(task, StandPolicy(agent).eval())
        totals = torch.zeros(32, 3)
        def record(task, step, active, done):
            quality.record(task, step, active, done)
            com = task.center_of_mass()
            vel = (com-task.previous_com)/task.dt
            loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
            feet = torch.stack((loads[:, 0]+loads[:, 2], loads[:, 1]+loads[:, 3]), -1)
            points = task.engine.get_body_pos(0)[:, task.allowed_ground][:, [0, 2, 1, 3]].reshape(32, 2, 2, 3)
            error = capture_distance(com, vel, task.engine.get_body_pos(0)[:, task.feet], feet)
            old = swing_quality(com, vel, points, task.sole_reference_height, feet, preparation_weight=.3)
            lift = swing_quality(com, vel, points, task.sole_reference_height, feet)
            mask = active & task.engine.hit
            risk = (error/.06).clamp(0, 1)
            grounded = ((points[..., 2]-task.sole_reference_height).amin(-1) < .015).all(-1)
            totals[:, 0] += old*risk*mask*task.dt
            totals[:, 1] += old*risk*mask*grounded*task.dt
            totals[:, 2] += lift*risk*mask*task.dt
        metrics, _ = evaluate(task, quality, selection_cases(speed, 12., task.dt), record)
        metrics['motion_quality'] = quality.report()
        metrics['speed'] = speed
        metrics['swing_audit'] = dict(total_risk_weighted_signal_seconds=float(totals[:, 0].sum()),
            grounded_signal_fraction=float(totals[:, 1].sum()/totals[:, 0].sum().clamp_min(1e-9)),
            lift_only_signal_seconds=float(totals[:, 2].sum()),
            scope="Development cardinal torso cases; reward signal, not causal proof or unseen validation")
        results.append(metrics)
        (args.out/'report.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(args.checkpoint),
            source_sha256=sha256(Path(__file__)), results=results), indent=2))
        print(json.dumps({k:metrics[k] for k in ('speed','recovered_hits','survived_hits','recovered_with_step','swing_audit')}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','checkpoint','reference','ball-model','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--speeds', type=float, nargs='+', default=[2.,3.])
    p.add_argument('--recovery-mode',choices=('step_v5','step_v6'),default='step_v5')
    run(p.parse_args())
