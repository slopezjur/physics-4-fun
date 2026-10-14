"""Repeat the sealed recovery protocol at fixed student source frame six."""
import argparse
from pathlib import Path
import shutil
import traceback

import numpy as np

from . import admit_stand_student_recovery as sealed
from .audit_stand_failures import digest, read, save


FRAME = 6
LATE = Path('logs/mimickit-training/stand-student-recovery-20261002-01')
ORIGINAL_PROTECT = sealed.protect_sources


def select_states(name):
    trace_path = sealed.PRIOR/name/'trace.npz'
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    records = []
    for case in range(8):
        assert trace['alive_before'][FRAME, case]
        assert trace['pre_balance'][FRAME, case] and trace['pre_posture'][FRAME, case]
        assert trace['diagnostic_label_numerically_valid'][FRAME, case]
        first = int(np.flatnonzero(trace['alive_before'][:, case] &
            ~(trace['balance'][:, case] & trace['posture'][:, case]))[0])
        issue_time = float(trace['time_s'][FRAME]-trace['time_s'][0])
        assert abs(issue_time-0.100008) < 1e-12 and FRAME < first
        records.append(dict(case=case, source_trace=str(trace_path), source_trace_sha256=digest(trace_path),
            source_frame=FRAME, source_issue_time_s=issue_time,
            source_first_departure_ending_time_s=float(trace['time_s'][first]),
            source_pre_balance=True, source_pre_posture=True, source_numerically_valid_teacher_label=True))
    start = dict(qpos=trace['qpos_before'][FRAME].copy(), qvel=trace['qvel_before'][FRAME].copy(),
        pending_commands_oldest_first=trace['pending_commands_oldest_first'][FRAME].copy(),
        held_torque_nm=trace['held_torque_nm'][FRAME-1].copy(),
        previous_issued_action=trace['command'][FRAME-1].copy(), source_observation=trace['obs'][FRAME].copy(),
        source_teacher_proposal=trace['teacher_only_command'][FRAME].copy())
    measurements = {key[4:]: value[FRAME].copy() for key, value in trace.items() if key.startswith('pre_')}
    return start, measurements, records


def protect_sources(out):
    # Bind the sealed late comparison before creating any new recovery starts.
    late_protocol, late_verified = read(LATE/'protocol.json'), read(LATE/'final-verification.json')
    late_inputs = {**late_protocol['inputs'], **late_verified['output_hashes']}
    late_inputs[str(LATE/'final-verification.json')] = digest(LATE/'final-verification.json')
    assert all(digest(Path(path)) == value for path, value in late_inputs.items())
    protected, sources, contract = ORIGINAL_PROTECT(out)
    for path, value in late_inputs.items():
        assert path not in protected or protected[path] == value
        protected[path] = value
    path = Path(__file__)
    snapshot = out/'sources'/f'{len(sources):02d}-{path.name}'
    shutil.copy2(path, snapshot)
    protected[str(path)] = digest(path)
    sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources, contract


def save_protocol(path, value):
    if path.name == 'protocol.json':
        value = dict(value, source_selection='Fixed live balanced/posture-valid pre-command source frame 6',
            source_frame=FRAME, source_issue_time_s=0.100008, late_comparison=str(LATE),
            implementation='Sealed Phase 34 recovery implementation; only source selection and provenance wrapper differ')
    save(path, value)


def run(args):
    # Reuse the sealed simulation, teacher, restoration, queue and admission logic.
    sealed.select_states = select_states
    sealed.protect_sources = protect_sources
    sealed.save = save_protocol
    sealed.run(args)


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
