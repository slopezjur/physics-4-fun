"""Frozen inference on saved teacher-labelled states; no simulator or optimizer."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .audit_stand_failures import digest, read, save
from .policy import StandPolicy
from .runtime import activate
from .train_stand_imitation import tensor_hash


MODELS = dict(imitation=Path('logs/mimickit-training/stand-imitation-pilot-20261001-01'),
    correction_imitation=Path('logs/mimickit-training/stand-correction-pilot-20261001-01'),
    ppo_0=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0'),
    ppo_025=Path('logs/mimickit-training/stand-angular-reward-pilot-20261002-01/weight-0.25'))
GROUPS = ('actor-0-fraction-0',)+tuple(f'actor-{weight}-fraction-{fraction}'
    for weight in ('0','0.25') for fraction in ('0.01','0.05','0.1'))


def load_actor(path):
    from learning.nets.net_builder import build_net
    from learning.normalizer import Normalizer
    from .amp_recovery import validate_sidecar
    from .stable_normalizer import StableNormalizer
    state = torch.load(path/'model.amp.pt',map_location='cpu',weights_only=True)
    portable = torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    validate_sidecar(portable,state)
    assert digest(path/'model.pt')==read(path/'report.json')['model_sha256']
    config = yaml.safe_load((path/'agent.yaml').read_text())
    layers,_ = build_net(config['model']['actor_net'],dict(obs=np.zeros(306,np.float32)),torch.nn.ReLU)
    mean = torch.nn.Linear(512,33)
    obs = StableNormalizer((306,),'cpu',clip=10.)
    action = Normalizer((33,),'cpu')
    for module,prefix in ((layers,'_model._actor_layers.'),(mean,'_model._action_dist._mean_net.'),
                          (obs,'_obs_norm.'),(action,'_a_norm.')):
        module.load_state_dict({key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)})
    actor = StandPolicy(SimpleNamespace(_obs_norm=obs,_a_norm=action,
        _model=SimpleNamespace(_actor_layers=layers,_action_dist=SimpleNamespace(_mean_net=mean))))
    actor.eval().requires_grad_(False)
    assert torch.isfinite(actor.obs_norm.get_std()).all() and (actor.obs_norm.get_std()>0).all()
    return actor,state


def normalizers(actor):
    return dict(observation_count=int(actor.obs_norm.get_count()),action_count=int(actor.action_norm.get_count()),
        observation_pending=int(actor.obs_norm._pending_count),observation_clip=float(actor.obs_norm._clip),
        action_pending=int(actor.action_norm._new_count))


def metrics(error, clipped, mask, scale):
    cases = []
    for case in range(8):
        selected = mask[:,case]
        count = int(selected.sum())
        row = dict(case=case,samples=count)
        if count:
            e = error[selected,case].astype(np.float64)
            absolute = np.abs(e)
            row.update(normalized_command_rmse=float(np.sqrt(np.mean(e**2))),
                torque_rmse_nm=float(np.sqrt(np.mean((e*scale)**2))),
                max_absolute_command_error=float(absolute.max()),
                absolute_command_error_p95=float(np.quantile(absolute,.95)),
                absolute_command_error_p99=float(np.quantile(absolute,.99)),
                frame_rmse_p99=float(np.quantile(np.sqrt(np.mean(e**2,axis=-1)),.99)),
                clipped_feature_fraction=float(clipped[selected,case].mean()),
                frames_with_clipping_fraction=float(clipped[selected,case].any(-1).mean()),
                per_motor_rmse=np.sqrt(np.mean(e**2,axis=0)).tolist(),
                per_motor_max_absolute_error=absolute.max(0).tolist(),
                per_observation_feature_clipping_fraction=clipped[selected,case].mean(0).tolist())
        cases.append(row)
    populated = [row for row in cases if row['samples']]
    equal_pose = {key:np.mean([row[key] for row in populated],axis=0).tolist()
                  for key in populated[0] if key not in ('case','samples')} if populated else {}
    return dict(samples=int(mask.sum()),poses_with_samples=len(populated),cases=cases,
                equal_pose_means=equal_pose,weighting='Each populated pose equally weighted; no pooling across live durations')


def subsets(trace,dt):
    live = trace['alive_before']
    reset = np.zeros_like(live)
    reset[0] = True
    time = trace['time_s']-dt
    previous_valid = np.zeros_like(live)
    previous_valid[1:] = trace['balance'][:-1] & trace['posture'][:-1]
    return dict(all_live=live,reset=live & reset,first_two_seconds=live & (time[:,None]<2.),
        later=live & (time[:,None]>=2.),balanced_and_posture_valid=live & previous_valid & ~reset,
        departures=live & ~previous_valid & ~reset)


@torch.no_grad()
def predictions(actor,trace):
    live = trace['alive_before']
    proposed = np.zeros_like(trace['teacher_only_command'])
    clipped = np.zeros_like(trace['obs'],dtype=bool)
    for frame,observation in enumerate(trace['obs']):
        obs = torch.from_numpy(observation)
        valid = live[frame]
        proposed[frame,valid] = actor(obs).numpy()[valid]
        standardized = (obs-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
        clipped[frame,valid] = (standardized.abs()>actor.obs_norm._clip).numpy()[valid]
    error = np.zeros_like(proposed)
    error[live] = proposed[live]-trace['teacher_only_command'][live]
    assert np.isfinite(proposed).all() and np.isfinite(error).all()
    return dict(predicted_command=proposed,error_from_live_teacher=error,
        clipped_features=clipped,clipped_feature_fraction=clipped.mean(-1).astype(np.float32),
        alive_before=live,time_s=trace['time_s'])


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    prior = read(args.prior/'protocol.json')
    complete = read(args.prior/'completion-verification.json')
    prior_report = read(args.prior/'report.json')
    protected = dict(prior['inputs'])
    protected.update(complete['output_hashes'])
    protected[str(args.prior/'completion-verification.json')] = digest(args.prior/'completion-verification.json')
    paths = [Path(__file__)]+[Path(s['path']) for s in prior['sources']]
    for s in prior['sources']:
        for key in ('path','snapshot'): protected[s[key]] = s['sha256']
    paths.extend(args.mimickit/'mimickit'/name for name in
        ('learning/normalizer.py','learning/nets/net_builder.py','learning/nets/fc_2layers_1024units.py',
         'learning/base_model.py','learning/ppo_model.py'))
    paths.extend(Path('mujoco_rig/mimic')/name for name in ('amp_recovery.py','amp_step.py','stable_normalizer.py'))
    paths = list({str(path.resolve()):path for path in paths}.values())
    for path in paths:
        value = digest(path)
        assert str(path) not in protected or protected[str(path)]==value
        protected[str(path)] = value
    for path in MODELS.values():
        for filename in ('model.pt','model.amp.pt','agent.yaml','contract.json','protocol.json','report.json'):
            file = path/filename
            value = digest(file)
            assert str(file) not in protected or protected[str(file)]==value
            protected[str(file)] = value
        contract = read(path/'contract.json')
        for key in ('model_sha256','reference_sha256','actuator_names','observation_layout','torque_scale_nm'):
            assert contract[key]==prior['contract'][key], f'Incompatible {path}: {key}'
        assert contract['num_obs']==306 and len(contract['actuator_names'])==33
    assert all(digest(Path(path))==value for path,value in protected.items()), 'Historical input/source changed'
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'sources').mkdir()
    sources = []
    for path in paths:
        snapshot = args.out/'sources'/path.name
        assert not snapshot.exists()
        shutil.copy2(path,snapshot)
        sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=digest(path)))
    traces = {}
    keys = ('obs','teacher_only_command','actor_only_command','alive_before','balance','posture','time_s')
    for group in GROUPS:
        with np.load(args.prior/group/'trace.npz',allow_pickle=False) as archive:
            traces[group] = {key:archive[key] for key in keys}
    observations = sum(int(t['alive_before'].sum()) for t in traces.values())
    assert observations==32061
    save(args.out/'protocol.json',dict(inputs=protected,sources=sources,prior=str(args.prior),
        models={key:str(value) for key,value in MODELS.items()},groups=list(GROUPS),
        contract=prior['contract'],unique_live_observations=observations,recorded_live_predictions=observations*4,
        batch_scope='Eight rows per saved frame preserve original PPO floating-point kernels; inactive outputs discarded',
        phase_partition='Observation issue time = ending time - control timestep; first two seconds use issue time < 2',
        physical_partition='Pre-command validity from previous saved endpoint; reset excluded from validity/departure groups',
        observation_clip=10.,normalizers='Each checkpoint mean/std/count, unchanged',
        simulation_steps=0,optimizers_created=0,training_updates=0,arena_changed=False))
    results, frozen = {}, {}
    scale = np.asarray(prior['contract']['torque_scale_nm'])
    dt = prior['contract']['control_timestep']
    for label,path in MODELS.items():
        actor,state = load_actor(path)
        before = tensor_hash(actor.state_dict())
        counts = normalizers(actor)
        assert not counts['observation_pending'] and not counts['action_pending']
        if label.startswith('ppo_'):
            weight = '0' if label=='ppo_0' else '0.25'
            assert before==prior_report['frozen'][weight]['tensor_hashes_before']['actor']
        model_out = args.out/label
        model_out.mkdir()
        groups = {}
        for group,trace in traces.items():
            out = model_out/group
            out.mkdir()
            arrays = predictions(actor,trace)
            own = ((label=='ppo_0' and group.startswith('actor-0-')) or
                   (label=='ppo_025' and group.startswith('actor-0.25-')))
            if own:
                live = trace['alive_before']
                assert np.array_equal(arrays['predicted_command'][live],trace['actor_only_command'][live]), 'Stored own PPO proposal differs'
            masks = subsets(trace,dt)
            arrays.update({f'subset_{key}':value for key,value in masks.items()})
            np.savez_compressed(out/'predictions.npz',**arrays)
            report = dict(model=label,trace=group,own_ppo_proposals_verified_exactly=own,
                subsets={key:metrics(arrays['error_from_live_teacher'],arrays['clipped_features'],mask,scale)
                         for key,mask in masks.items()},simulation_steps=0,training_updates=0)
            save(out/'report.json',report)
            groups[group] = report
        after = tensor_hash(actor.state_dict())
        assert before==after and normalizers(actor)==counts
        frozen[label] = dict(actor_hash_before=before,actor_hash_after=after,
            full_checkpoint_tensor_hash=tensor_hash(state),normalizer_counts_before=counts,
            normalizer_counts_after=normalizers(actor),portable_sidecar_match=True)
        results[label] = groups
        save(model_out/'report.json',dict(groups=groups,frozen=frozen[label],simulation_steps=0,training_updates=0))
        save(args.out/'progress.json',dict(models_completed=len(results),total_models=4))
        print(dict(model=label,live_predictions=observations,frozen=True),flush=True)
    assert all(digest(Path(path))==value for path,value in protected.items())
    assert all(digest(Path(s[key]))==s['sha256'] for s in sources for key in ('path','snapshot'))
    save(args.out/'report.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),models=results,frozen=frozen,
        unique_live_observations=observations,recorded_live_predictions=observations*4,
        forward_padding_rows_discarded=(sum(t['alive_before'].size for t in traces.values())-observations)*4,
        inputs_unchanged=True,sources_and_snapshots_unchanged=True,frozen_tensors_and_normalizers_unchanged=True,
        simulation_steps=0,simulators_created=0,optimizers_created=0,training_updates=0,arena_changed=False,
        interpretation='Counterfactual predictions on recorded states, not new actor trajectories or causal clipping effects'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,default=Path('logs/mimickit-training/stand-whole-body-blend-20261002-01'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise
