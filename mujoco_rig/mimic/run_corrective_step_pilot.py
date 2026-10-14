"""Prospective four-arm corrective-step screen, with quiet preflight and fixed budgets."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from . import ball_study_support as support
from .corrective_step_audit import audit_trace
from .corrective_step_variant import control_variant, learning_variant
from .evaluate_stand_ball_boundary import evaluate as native_ball
from .evaluate_stand_ball_gpu import evaluate as gpu_ball, compare as compare_backends
from .evaluate_stand_sustained import evaluate_duration
from .policy import StandPolicy
from .prepare_stand_ball_pair import prepare_pair
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .run_stand_ball_pilot import train_arm
from .stand_experiment_evidence import load_actor, seal
from .stand_start_guard import compare as compare_starts


def rehearse(agent, folder):
    """Same quiet examples and auxiliary-loss weight in all admitted arms."""
    with np.load(folder/'trace.npz', allow_pickle=False) as z:
        keep = z['time_s'] <= 16
        if not (z['balance'][keep] & z['posture'][keep] & z['pre_balance'][keep] & z['pre_posture'][keep]).all():
            raise ValueError('Quiet rehearsal contains an invalid standing frame')
        obs = torch.tensor(z['observation'][keep].reshape(-1, 374), device=agent._device)
        action = torch.tensor(z['action'][keep].reshape(-1, 33), device=agent._device)
    if tensor_hash(StandPolicy(agent).state_dict()) != read(folder/'report.json')['actor_hash']:
        raise ValueError('Rehearsal actor mismatch')
    original = agent._compute_actor_loss
    count = dict(calls=0, examples=0)
    index = torch.arange(256, device=agent._device)

    def loss(batch):
        result = original(batch)
        ids = (index*31 + count['calls']*257) % len(obs)
        mean = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs[ids])).mode)
        mse = (mean-action[ids]).square().mean()
        result['rehearsal_mse'] = mse.detach()
        result['actor_loss'] = result['actor_loss']+100*mse
        count['calls'] += 1
        count['examples'] += len(ids)
        return result

    agent._compute_actor_loss = loss
    return count


def assess(actor, rig, reference, world, folder, settings, *, impacts):
    folder.mkdir()
    result = dict(quiet={}, impacts={}, parity={}, step_audits={})
    actor.eval().requires_grad_(False)
    for backend in ('native', 'gpu'):
        result['quiet'][backend] = evaluate_duration(rig, reference, 'target_pd', backend, actor,
            folder/f'quiet-{backend}', seconds=12.)
    result['quiet_pass'] = all(r['standing_passes'] == 8 for r in result['quiet'].values())
    if impacts:
        for speed, sign in ((1.6, 1), (1.6, -1), (2., 1), (2., -1)):
            name = f'{speed:g}-{sign:+d}'
            pair = {}
            for backend, evaluator in (('native', native_ball), ('gpu', gpu_ball)):
                out = folder/f'{name}-{backend}'
                journal = {}
                observer = support.body_observer(journal, int(np.ceil(12/.016668))) if backend == 'native' else nullcontext()
                with observer:
                    pair[backend] = evaluator(rig, reference, actor, world, out, settings, speed, sign)
                if backend == 'native':
                    bodies = support.save_bodies(out, journal)
                    pair[backend]['chest_first_all'] = all(c['clusters'] and 'Chest' in c['clusters'][0]['bodies']
                                                         for c in bodies['cases'])
                audit = audit_trace(rig.path, out)
                save(out/'corrective-step.json', audit)
                result['step_audits'][f'{name}-{backend}'] = audit
            result['impacts'][name] = pair
            result['parity'][name] = compare_backends(pair['native'], pair['gpu'], .016668)
            save(folder/'progress.json', result)
    result['physical_admission'] = all(r['passed'] for r in result['parity'].values()) and all(
        pair['native']['chest_first_all'] for pair in result['impacts'].values())
    save(folder/'report.json', result)
    return result


def run(args):
    s = read(args.protocol)
    if s['schema'] != 'corrective_step_pilot_v1' or not __debug__:
        raise ValueError('Expected the preregistered corrective-step pilot')
    if s['updates_per_arm'] != 100 or s['maximum_updates'] != 800 or s['maximum_transitions'] != 3276800:
        raise ValueError('Unexpected experiment budget')
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    deps = [Path(__file__), Path(__file__).with_name('corrective_step_variant.py'),
            Path(__file__).with_name('corrective_step_audit.py'), Path(__file__).with_name('step_reward.py'),
            Path(__file__).with_name('run_chest_stronger_onset_learning.py')]
    for seed in s['seeds']:
        item = s['checkpoints'][str(seed)]
        for root, paths in ((Path(item['root']), [Path(item['checkpoint']), Path(item['checkpoint']).with_suffix('.amp.pt')]),
                            (Path(s['rehearsal_root']), [Path(s['rehearsal_folders'][str(seed)])/name for name in ('trace.npz', 'report.json', 'contract.json')])):
            proof = read(root/'final-verification.json')
            if not proof['verified']:
                raise ValueError('Unverified source')
            known = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
            if any(known.get(str(p.resolve())) != digest(p) for p in paths):
                raise ValueError('Source hash mismatch')
    with patch.object(support, 'checkpoint', lambda seed, _: Path(s['checkpoints'][str(seed)]['checkpoint'])):
        parent, inputs, sources = support.protect_used(args.out, deps, args.protocol, s)
    parent = deepcopy(parent)
    parent['config']['actor_optimizer']['learning_rate'] = 3e-5
    save(args.out/'protocol.json', dict(settings=s, inputs=inputs, sources=sources,
        config=parent['config'], demonstrations=parent['demonstrations'],
        interpretation='Identical weights/RNG; wider target range changes initial behavior and must pass zero-update quiet admission.',
        automatic_promotion=False))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    world = args.out/'ball-5kg.xml'
    write_world(rig.path, support.FOUNDATION/'ball-5kg.xml', world, 5)
    results, updates, evaluations, presentations = {}, 0, 0, 0
    for seed in s['seeds']:
        root = args.out/str(seed)
        root.mkdir()
        checkpoint = Path(s['checkpoints'][str(seed)]['checkpoint'])
        seed_results = dict(preflight={}, arms={})
        for leg_range in (.25, .5):
            with control_variant(leg_range):
                source_actor = load_actor(rig, reference, parent, checkpoint)
                seed_results['preflight'][str(leg_range)] = assess(source_actor, rig, reference, world,
                    root/f'initial-{leg_range}', s, impacts=leg_range == .25)
            evaluations += 80 if leg_range == .25 else 16
            del source_actor
            gc.collect()
            torch.cuda.empty_cache()
        if not seed_results['preflight']['0.25']['quiet_pass']:
            raise RuntimeError('Source failed quiet preflight; no training is admitted')
        # Reverse the sequence for the second parent, without changing learner RNG.
        arms = s['arms'] if seed == s['seeds'][0] else list(reversed(s['arms']))
        for arm in arms:
            name, leg_range, reward = arm['name'], arm['leg_range_rad'], arm['recovery_reward']
            if not seed_results['preflight'][str(leg_range)]['quiet_pass']:
                seed_results['arms'][name] = dict(status='gated_out', reason='Changed controller fails zero-update quiet Stand', updates=0)
                continue
            with control_variant(leg_range), learning_variant(s, reward):
                starts = root/f'{name}-starts'
                pair = prepare_pair(rig, reference, world, parent, checkpoint,
                                    seed+s['optimization_seed_offset'], starts)
                agent, _, _ = pair['single_ball']
                # The sibling is never trained; all admitted arms start from the source.
                counter = rehearse(agent, Path(s['rehearsal_folders'][str(seed)]))
                peer = next((root/f'{a["name"]}-starts' for a in s['arms']
                             if a['name'] != name and (root/f'{a["name"]}-starts').exists()), None)
                if peer:
                    admission = compare_starts(peer/'single_ball', starts/'single_ball')
                    save(starts/'cross-arm-admission.json', admission)
                    if not admission['passed']:
                        raise RuntimeError('Cross-arm start admission failed')
                training = train_arm(pair, 'single_ball', seed, starts, root/name, s)
                if counter['calls'] != 48*s['updates_per_arm']:
                    raise RuntimeError('Rehearsal/optimizer accounting mismatch')
                actor = StandPolicy(agent)
                updates += training['updates']
                presentations += counter['examples']
                del agent, pair
                gc.collect()
                torch.cuda.empty_cache()
            with control_variant(leg_range):
                final = assess(actor, rig, reference, world, root/name/'evaluation', s, impacts=True)
            evaluations += 80
            seed_results['arms'][name] = dict(status='complete', training=training, rehearsal=counter, evaluation=final)
            del actor
            gc.collect()
            torch.cuda.empty_cache()
            results[str(seed)] = seed_results
            save(args.out/'progress.json', dict(results=results, training_updates=updates, evaluation_cases=evaluations))
        results[str(seed)] = seed_results
    result = dict(results=results, training_updates=updates, transitions=updates*4096,
        evaluation_cases=evaluations, rehearsal_example_presentations=presentations,
        promoted=False, scope='Bounded discovery; no naturalness or 10kg/5mps readiness claim')
    save(args.out/'report.json', result)
    seal(args.out, inputs, sources, verified=True, training_updates=updates, evaluation_cases=evaluations, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc(), promoted=False))
        raise
