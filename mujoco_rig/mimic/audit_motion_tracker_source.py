"""Isolate a moving-reference actor from reset seating and correction authority."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from envs.base_env import EnvMode, DoneFlags
    from .evaluate_motion_tracker import evaluate
    from .motion_tracker_task import MotionTrackingTask
    from .native_engine import NativeDummyEngine
    from .rig import Rig

    if args.out.exists():
        raise ValueError('Require a new audit directory')
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    options = dict(rig=rig, stationary=args.stationary, motions=[args.motion],
        num_envs=8, quiet_fraction=0., engine_factory=NativeDummyEngine)
    runtime_options = ort.SessionOptions()
    runtime_options.intra_op_num_threads = runtime_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.source_onnx), runtime_options,
        providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def actor(observation):
        return torch.from_numpy(session.run(None, {input_name:observation[:, :363].numpy()})[0])

    results = {}
    for seating in (0., .0001):
        task = MotionTrackingTask(**options)
        reference = task.bank.references[0]
        reference.qpos[:] = reference.qpos[0].clone()
        reference.qpos[:, 2] -= seating
        reference.qvel.zero_()
        if seating:
            original_reset = task.engine.reset_envs

            def reset(ids, q, v):
                q = q.clone()
                q[:, 2] += seating  # Task already seats physical resets once.
                return original_reset(ids, q, v)

            task.engine.reset_envs = reset
        report, _ = evaluate(task, actor)
        results[f'fixed_reference_seating_{seating}'] = report
        print(json.dumps(dict(seating=seating, survived=report['successes'],
            survival=[c['survival_seconds'] for c in report['cases']])), flush=True)

    task = MotionTrackingTask(**options)
    task.set_mode(EnvMode.TEST)
    task.reset()
    checkpoint = torch.load(args.source_checkpoint, weights_only=False, map_location='cpu')
    saturated = total = 0
    max_correction = future_peak = 0.
    with np.load(args.trace) as trace:
        for actions, active in zip(trace['action'], trace['active']):
            observation = task.observations()
            source = actor(observation)
            mask = torch.tensor(active)
            saturated += int(((source.abs() > .99) & mask[:, None]).sum())
            total += int(active.sum() * 30)
            if active.any():
                max_correction = max(max_correction, float((torch.tensor(actions)-source)[mask].abs().max()))
                future = ((observation[:,217:361] - checkpoint['_obs_norm._mean'][217:361]) /
                    checkpoint['_obs_norm._std'][217:361])
                future_peak = max(future_peak, float(future[mask].abs().max()))
            _, _, done, _ = task.step(torch.tensor(actions))
            task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    results['action_diagnosis'] = dict(source_saturated_fraction=saturated/max(1,total),
        maximum_realized_correction=max_correction, maximum_normalized_future_feature=future_peak)
    results['provenance'] = dict(source_checkpoint_sha256=sha256(args.source_checkpoint),
        source_onnx_sha256=sha256(args.source_onnx), trace_sha256=sha256(args.trace),
        task_sha256=sha256(Path(__file__).with_name('motion_tracker_task.py')),
        scope='Native fixed-reference ablation and replay; no training, rig change or Arena promotion')
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results['action_diagnosis']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('mimickit', 'stationary', 'motion', 'source-onnx', 'source-checkpoint', 'trace', 'out'):
        parser.add_argument('--'+option, required=True, type=Path)
    run(parser.parse_args())
