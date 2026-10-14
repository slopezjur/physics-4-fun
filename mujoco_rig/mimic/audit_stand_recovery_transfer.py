"""Frozen student commands and native-pool coverage on saved autonomous states."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np
from scipy.spatial.distance import cdist
import torch

from .audit_stand_failures import digest, read, save
from .audit_stand_student_errors import load_actor, metrics, normalizers, predictions
from .runtime import activate
from .train_stand_imitation import tensor_hash


MODELS = Path('logs/mimickit-training/stand-recovery-supervision-20261002-01')
STATES = Path('logs/mimickit-training/stand-student-states-20261002-01')
RECOVERY = Path('logs/mimickit-training/stand-early-recovery-20261002-01')
TEACHER = Path('logs/mimickit-training/stand-teacher-20261001-01/native/dataset.npz')
CORRECTIONS = Path('logs/mimickit-training/stand-corrections-20261001-01/native/dataset.npz')
GROUPS = ('control-native', 'control-gpu', 'augmented-native', 'augmented-gpu')
FIELDS = dict(full=(0, 306), physical=(0, 95), pending=(95, 161), reference=(161, 306))


def protect(out):
    prior, complete = read(MODELS/'protocol.json'), read(MODELS/'final-verification.json')
    protected = {**prior['inputs'], **complete['output_hashes']}
    protected[str(MODELS/'final-verification.json')] = digest(MODELS/'final-verification.json')
    paths = [Path(__file__)]+[Path(source['path']) for source in prior['sources']]
    for source in prior['sources']:
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        value = digest(path)
        assert str(path) not in protected or protected[str(path)] == value
        protected[str(path)] = value
    assert all(digest(Path(path)) == value for path, value in protected.items())
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources, prior['contract']


def load_catalog():
    observations, actions, metadata, sources = [], [], [], []

    def append(path, obs, action, rows, cases, frames, episodes):
        source_id = len(sources)
        sources.append(dict(path=str(path), sha256=digest(path)))
        observations.append(obs)
        actions.append(action)
        metadata.append(dict(source_id=np.full(len(obs), source_id, dtype=np.int64),
            row=np.asarray(rows), case=np.asarray(cases), frame=np.asarray(frames), episode=np.asarray(episodes)))

    with np.load(TEACHER, allow_pickle=False) as data:
        obs, action = data['observation'].reshape(-1, 306), data['action'].reshape(-1, 33)
        rows = np.arange(len(obs), dtype=np.int64)
        append(TEACHER, obs, action, rows, rows % 8, rows//8, np.zeros(len(obs), dtype=np.int64))
    with np.load(CORRECTIONS, allow_pickle=False) as data:
        rows = np.arange(len(data['observation']), dtype=np.int64)
        append(CORRECTIONS, data['observation'], data['action'], rows, data['case'], data['step'], data['episode'])
    original_count = sum(len(obs) for obs in observations)
    for actor_name in ('control', 'augmented'):
        path = RECOVERY/f'{actor_name}-native'/'admitted-dataset.npz'
        report = read(path.parent/'report.json')
        with np.load(path, allow_pickle=False) as data:
            rows = []
            for case in report['accepted_cases']:
                selected = np.flatnonzero(data['case'] == case)
                selected = selected[np.argsort(data['step'][selected])]
                assert len(selected) == 720 and np.array_equal(data['step'][selected], np.arange(720))
                rows.extend(selected.tolist())
            rows = np.asarray(rows, dtype=np.int64)
            append(path, data['observation'][rows], data['action'][rows], rows,
                data['case'][rows], data['step'][rows], np.zeros(len(rows), dtype=np.int64))
    obs, action = np.concatenate(observations), np.concatenate(actions)
    catalog = {key: np.concatenate([row[key] for row in metadata]) for key in metadata[0]}
    assert original_count == 47520 and len(obs) == 57600
    assert np.isfinite(obs).all() and np.isfinite(action).all() and (np.abs(action) <= 1.).all()
    return obs, action, catalog, sources, original_count


def partitions(trace):
    live = trace['alive_before']
    frame = np.broadcast_to(np.arange(len(live))[:, None], live.shape)
    before_first = np.zeros_like(live)
    for case in range(8):
        first = int(np.flatnonzero(live[:, case] & ~(trace['balance'][:, case] & trace['posture'][:, case]))[0])
        before_first[:first+1, case] = True
    pre_valid = live & trace['pre_balance'] & trace['pre_posture'] & before_first
    initial = live & (frame == 0)
    early = live & (frame > 0) & (frame < 6)
    later_valid = pre_valid & (frame >= 6)
    departed = live & ~(initial | early | later_valid)
    assert np.array_equal(initial.astype(int)+early+later_valid+departed, live.astype(int))
    return dict(all_live=live, initial=initial, early_noninitial=early,
        later_valid_before_first_departure=later_valid, departed=departed,
        all_valid_before_first_departure=pre_valid)


@torch.no_grad()
def normalized(actor, obs):
    return actor.obs_norm.normalize(torch.from_numpy(obs)).numpy().astype(np.float64)


def nearest(query, pool, begin, end):
    indices, distances = [], []
    for offset in range(0, len(query), 64):
        matrix = cdist(query[offset:offset+64, begin:end], pool[:, begin:end], metric='sqeuclidean')
        index = matrix.argmin(axis=1)
        indices.append(index)
        distances.append(np.sqrt(matrix[np.arange(len(index)), index]/(end-begin)))
    return np.concatenate(indices), np.concatenate(distances)


def coverage(actor, traces, pool_obs, pool_actions, split, out):
    pool = normalized(actor, pool_obs)
    queries = np.concatenate([trace['obs'][trace['alive_before']] for trace in traces.values()])
    query = normalized(actor, queries)
    results = {}
    for field, (begin, end) in FIELDS.items():
        old_index, old_distance = nearest(query, pool[:split], begin, end)
        new_index, new_distance = nearest(query, pool[split:], begin, end)
        new_index += split
        choose_new = new_distance < old_distance
        combined_index = np.where(choose_new, new_index, old_index)
        combined_distance = np.minimum(old_distance, new_distance)
        results[field] = dict(original_index=old_index, original_rms=old_distance,
            recovery_index=new_index, recovery_rms=new_distance,
            combined_index=combined_index, combined_rms=combined_distance)
        print(dict(coverage_field=field, live_queries=len(query), complete=True), flush=True)
    (out/'coverage').mkdir()
    offset = 0
    reports = {}
    for name, trace in traces.items():
        count = int(trace['alive_before'].sum())
        selection = slice(offset, offset+count)
        target = trace['teacher_only_command'][trace['alive_before']]
        arrays = dict(frame=np.nonzero(trace['alive_before'])[0], case=np.nonzero(trace['alive_before'])[1])
        for field, values in results.items():
            for key, value in values.items():
                arrays[f'{field}_{key}'] = value[selection]
            for pool_name in ('original', 'recovery', 'combined'):
                ids = arrays[f'{field}_{pool_name}_index']
                error = pool_actions[ids]-target
                arrays[f'{field}_{pool_name}_nearest_label_rmse'] = np.sqrt(np.mean(error.astype(np.float64)**2, axis=-1))
        np.savez_compressed(out/'coverage'/f'{name}.npz', **arrays)
        report = {}
        for partition, mask in partitions(trace).items():
            selected = mask[trace['alive_before']]
            report[partition] = dict(samples=int(selected.sum()), fields={field: {
                pool_name: dict(mean_normalized_rms=float(arrays[f'{field}_{pool_name}_rms'][selected].mean()),
                    p95_normalized_rms=float(np.quantile(arrays[f'{field}_{pool_name}_rms'][selected], .95)),
                    mean_nearest_label_rmse=float(arrays[f'{field}_{pool_name}_nearest_label_rmse'][selected].mean()))
                for pool_name in ('original', 'recovery', 'combined')} for field in FIELDS})
        save(out/'coverage'/f'{name}.json', report)
        reports[name] = report
        offset += count
    assert offset == 1956
    return reports


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(2)
    protected, sources, contract = protect(args.out)
    traces = {}
    for name in GROUPS:
        with np.load(STATES/name/'trace.npz', allow_pickle=False) as data:
            traces[name] = {key: data[key] for key in ('obs', 'teacher_only_command', 'alive_before',
                'balance', 'posture', 'pre_balance', 'pre_posture', 'time_s', 'pending_commands_oldest_first',
                'diagnostic_label_numerically_valid')}
        trace = traces[name]
        assert trace['diagnostic_label_numerically_valid'][trace['alive_before']].all()
        assert np.array_equal(trace['obs'][..., 95:161], trace['pending_commands_oldest_first'])
    assert sum(int(trace['alive_before'].sum()) for trace in traces.values()) == 1956
    pool_obs, pool_actions, catalog, pool_sources, split = load_catalog()
    for source in pool_sources:
        assert source['path'] in protected and protected[source['path']] == source['sha256']
    np.savez_compressed(args.out/'pool-catalog.npz', **catalog)
    save(args.out/'pool-sources.json', pool_sources)
    models, frozen = {}, {}
    for label in ('control', 'augmented'):
        actor, full = load_actor(MODELS/label)
        models[label] = actor
        frozen[label] = dict(actor_hash=tensor_hash(actor.state_dict()), full_checkpoint_hash=tensor_hash(full),
            normalizer_counts=normalizers(actor))
    assert all(torch.equal(value, models['augmented'].obs_norm.state_dict()[key])
        for key, value in models['control'].obs_norm.state_dict().items())
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, contract=contract,
        states=str(STATES), models=str(MODELS), groups=list(GROUPS), frozen=frozen,
        labelled_live_states=1956, frozen_live_predictions=3912, cases_per_frame=8,
        original_native_pairs=split, recovery_native_pairs=len(pool_obs)-split, fields=FIELDS,
        normalization='Shared frozen Phase 32 standardization with inference clip 10; nearest RMS in float64 over these clipped normalized inputs',
        distance='Exact scipy sqeuclidean search; each field independently minimized; tie uses first catalog row, combined tie favors original',
        provenance='Catalog source_id + original flat row + case + frame + episode; recovery ordering matches Phase 36 native training pool',
        partitions='Initial frame 0; early noninitial frames 1..5; later valid pre-command frames >=6 up to first ending departure; departed remainder. Extra all-valid partition overlaps.',
        interpretation='Counterfactual frozen predictions on older actors, not new Phase 36 rollouts; finite private proposals are diagnostic only',
        simulators_created=0, physics_steps=0, optimizers_created=0, training_updates=0,
        dataset_admissions=0, new_amp_clips=0, arena_changed=False))
    covered = coverage(models['control'], traces, pool_obs, pool_actions, split, args.out)
    results = {}
    scale = np.asarray(contract['torque_scale_nm'])
    for label, actor in models.items():
        arm = args.out/label
        arm.mkdir()
        groups = {}
        for name, trace in traces.items():
            out = arm/name
            out.mkdir()
            arrays = predictions(actor, trace)
            masks = partitions(trace)
            arrays.update({f'subset_{key}': value for key, value in masks.items()})
            np.savez_compressed(out/'predictions.npz', **arrays)
            measures = {key: metrics(arrays['error_from_live_teacher'], arrays['clipped_features'], mask, scale)
                for key, mask in masks.items()}
            group = dict(group=name, source_trace=str(STATES/name/'trace.npz'),
                source_trace_sha256=digest(STATES/name/'trace.npz'), partitions=measures,
                simulation_steps=0, training_updates=0)
            save(out/'report.json', group)
            groups[name] = group
        assert tensor_hash(actor.state_dict()) == frozen[label]['actor_hash']
        assert normalizers(actor) == frozen[label]['normalizer_counts']
        results[label] = groups
        print(dict(actor=label, frozen_live_predictions=1956, complete=True), flush=True)
    assert all(digest(Path(path)) == value for path, value in protected.items())
    assert all(digest(Path(source[key])) == source['sha256'] for source in sources for key in ('path', 'snapshot'))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        actors=results, coverage=covered, frozen=frozen, live_predictions=3912,
        inputs_and_sources_unchanged=True, actors_and_normalizers_unchanged=True,
        simulators_created=0, physics_steps=0, optimizers_created=0, training_updates=0,
        dataset_admissions=0, new_amp_clips=0, arena_changed=False))
    save(args.out/'completion-verification.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        protected_input_count=len(protected), source_count=len(sources), frozen_live_predictions=3912,
        inputs_and_sources_verify=True, frozen_actors_verify=True,
        output_hashes={str(path): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
