"""One-minute PPO/AMP throughput measurements, without checkpoint or Arena promotion."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate
from .train_torque_stand import build_agent, prepare_plant, sample_gpu, write_json


def measure(args):
    activate(args.mimickit)
    from util import mp_util
    from .rig import Rig
    from .torque_stand_task import AMPTorqueStandTask
    from .amp_recovery import amp_sidecar, validate_sidecar
    torch.set_num_threads(1)
    torch.manual_seed(300994)
    np.random.seed(300994)
    mp_util.init(0, 1, 'cuda:0', None)
    args.out.mkdir(parents=True, exist_ok=False)
    startup = time.perf_counter()
    source = Path('mujoco_rig/dummy.xml')
    prepare_plant(source, args.out/'dummy.xml')
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    parent = args.checkpoint.parent
    contract = json.loads((parent/'contract.json').read_text())
    source_report = json.loads((parent/'report.json').read_text())
    if (contract['model_sha256'] != sha256(rig.path)
            or source_report['final_checkpoint_sha256'] != sha256(args.checkpoint)
            or contract['reference_sha256'] != sha256(args.reference)):
        raise ValueError('Benchmark source provenance does not match')
    portable = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    full = torch.load(amp_sidecar(args.checkpoint), map_location='cpu', weights_only=True)
    validate_sidecar(portable, full)
    del portable, full
    config = yaml.safe_load((parent/'agent.yaml').read_text())
    original_buffer_size = config['disc_buffer_size']
    # Preserve replay capacity per environment, and fit a complete initial rollout.
    config['disc_buffer_size'] = max(original_buffer_size,
        int(original_buffer_size*args.envs[0]/2048), config['steps_per_iter']*args.envs[0])
    task = AMPTorqueStandTask(rig, args.reference, args.envs[0], 'cuda:0', seed=300994,
        demonstrations=[Path('mujoco_rig/mimic/assets/stand_reference.npz'),
                        Path('logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz')],
        demonstration_rig=Rig.load(source))
    agent = build_agent(config, task, 'cuda:0', args.checkpoint)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    phases = defaultdict(float)
    for name in ('_rollout_train', '_build_train_data', '_update_model', '_update_normalizers'):
        original = getattr(agent, name)
        def timed(*values, _original=original, _name=name, **kw):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = _original(*values, **kw)
            torch.cuda.synchronize()
            phases[_name] += time.perf_counter()-tick
            return result
        setattr(agent, name, timed)
    # Warmup performs one complete update, including inference, resets and all optimizers.
    agent._train_iter()
    agent._sample_count = agent._update_sample_count()
    agent._iter += 1
    torch.cuda.synchronize()
    startup_seconds = time.perf_counter()-startup
    phases.clear()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cat([p.detach().flatten() for p in agent._model._actor_layers.parameters()]+
                       [p.detach().flatten() for p in agent._model._action_dist._mean_net.parameters()]).clone()
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, args.out/'gpu-samples.jsonl'), daemon=True)
    sampler.start()
    started, cpu_started = time.perf_counter(), time.process_time()
    updates, iteration_seconds = 0, []
    try:
        while time.perf_counter()-started < args.seconds:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            info = agent._train_iter()
            torch.cuda.synchronize()
            iteration_seconds.append(time.perf_counter()-tick)
            if not all(np.isfinite(float(value)) for value in info.values()):
                raise FloatingPointError('Nonfinite PPO/AMP benchmark update')
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            updates += 1
            print(json.dumps(dict(envs=args.envs[0], updates=updates,
                                  measured_seconds=round(time.perf_counter()-started, 3))), flush=True)
        elapsed, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu_started
    finally:
        stop.set()
        sampler.join(timeout=6)
    samples = [json.loads(line) for line in (args.out/'gpu-samples.jsonl').read_text().splitlines()]
    after = torch.cat([p.detach().flatten() for p in agent._model._actor_layers.parameters()]+
                      [p.detach().flatten() for p in agent._model._action_dist._mean_net.parameters()])
    transitions = updates*config['steps_per_iter']*args.envs[0]
    gpu = [row['gpu_percent'] for row in samples]
    report = dict(schema='mimic_torque_throughput_v1', envs=args.envs[0], steps_per_iteration=config['steps_per_iter'],
        requested_seconds=args.seconds, measured_seconds=elapsed, startup_and_warmup_seconds=startup_seconds,
        measured_updates=updates, transitions=transitions, transitions_per_second=transitions/elapsed,
        updates_per_minute=updates/elapsed*60, median_update_seconds=statistics.median(iteration_seconds),
        phases_seconds=dict(phases), process_cpu_seconds=cpu_seconds,
        gpu_mean_percent=statistics.mean(gpu), gpu_peak_percent=max(gpu), gpu_samples=len(gpu),
        gpu_device_memory_peak_mib=max(row['memory_mib'] for row in samples),
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        actor_parameter_change_l2=float(torch.linalg.vector_norm(after-before)),
        checkpoint_sha256=sha256(args.checkpoint), amp_sha256=sha256(amp_sidecar(args.checkpoint)),
        model_sha256=sha256(rig.path), observation_contract=task.observation_contract(), agent_config=config,
        original_disc_buffer_size=original_buffer_size,
        scope='Discarded PPO/AMP updates; CUDA-synchronized wall timing; one warmup update excluded; '
              'finish complete update after 60-second boundary; phase synchronization adds overhead; '
              'whole-device GPU samples; no standing evaluation or evidence of learning quality',
        arenas_modified=False, checkpoint_promoted=False)
    write_json(args.out/'report.json', report)
    print(json.dumps({key:report[key] for key in ('envs', 'measured_seconds', 'transitions_per_second',
                                                'gpu_mean_percent', 'updates_per_minute')}), flush=True)


def run(args):
    if args.worker:
        measure(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for count in args.envs:
        target = args.out/f'envs-{count}'
        command = [sys.executable, '-B', '-u', '-m', 'mujoco_rig.mimic.benchmark_torque_throughput',
            '--worker', '--mimickit', str(args.mimickit), '--checkpoint', str(args.checkpoint),
            '--reference', str(args.reference), '--out', str(target), '--seconds', str(args.seconds),
            '--envs', str(count)]
        subprocess.run(command, check=True)
        reports.append(json.loads((target/'report.json').read_text()))
    write_json(args.out/'report.json', dict(schema='mimic_torque_throughput_comparison_v1', measurements=reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--envs', type=int, nargs='+', default=[2048, 4096])
    parser.add_argument('--worker', action='store_true')
    run(parser.parse_args())
