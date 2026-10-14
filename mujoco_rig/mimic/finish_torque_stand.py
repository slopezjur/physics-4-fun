"""Silently select a completed torque Stand window in the primary Stand Arena."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time

from .baseline import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def finish(run):
    deadline = time.monotonic()+7200
    state_path = run/'completion-state.json'
    while time.monotonic()<deadline:
        if state_path.exists():
            state = read(state_path)
            if state['status']=='failed':
                print(json.dumps(dict(status='failed', error=state['error'])), flush=True)
                return
            if state['status']=='final_preview_ready':
                if state.get('artifact_finalization_complete'):
                    return
                break
        time.sleep(10)
    else:
        raise TimeoutError('No completion state after two hours; trainer was not stopped')
    preview = run/'arena-preview'
    bundle = 'res://'+preview.resolve().relative_to(Path.cwd().resolve()).as_posix()
    scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicStandTorqueArena.tscn')
    # Normalize the absolute --out used by the PowerShell launcher in the first run.
    wrong = 'res://'+preview.as_posix()
    scene.write_text(scene.read_text(encoding='utf-8').replace(wrong, bundle), encoding='utf-8')
    selection = read(preview/'arena-selection.json')
    report = read(run/'report.json')
    selection.update(reward_mode=report.get('reward_mode'),
                     posture_curriculum=report.get('posture_curriculum'),
                     final_quiet_retention=report.get('final_quiet_retention'),
                     stop_reason=report.get('stop_reason'))
    viewer = Path('logs/mimickit-viewer/stand.json')
    previous = read(viewer) if viewer.exists() else None
    selection.update(bundle=bundle, previous_stand_selection=previous,
                     selected_at_utc=datetime.now(timezone.utc).isoformat(), scene_sha256=sha256(scene))
    stand_scene = Path('Scenes/RL/Isaac3/MuJoCo/MimicStandArena.tscn')
    import re
    prior_scene = preview/'prior-stand-arena.tscn'
    if not prior_scene.exists():
        prior_scene.write_bytes(stand_scene.read_bytes())
    text = stand_scene.read_text(encoding='utf-8')
    text = re.sub(r'^ModelPath = ".*"', f'ModelPath = "{bundle}/dummy.xml"', text, flags=re.M)
    text = re.sub(r'^BundleDirectory = ".*"', f'BundleDirectory = "{bundle}"', text, flags=re.M)
    if 'RunIndefinitely = ' not in text:
        text = text.replace('TrialSeconds = 5.0', 'TrialSeconds = 30.0\nRunIndefinitely = true\nUseLatestExport = false')
    text = re.sub(r'^RunIndefinitely = .*', 'RunIndefinitely = true', text, flags=re.M)
    text = re.sub(r'^UseLatestExport = .*', 'UseLatestExport = false', text, flags=re.M)
    short_hash = selection['checkpoint_sha256'][:8]
    gate = f"standing {report['final']['settled_cases']}/8"
    help_text = ('R: restart · P: pause\\nDirect learned torques · no joint springs or standing-pose servo\\n'
                 f'Candidate {short_hash} · experimental · {gate}')
    text = re.sub(r'(\[node name="Help"[^\]]*\]\s*layout_mode = 2\s*)text = ".*"',
                  lambda match: match[1]+f'text = "{help_text}"', text)
    stand_scene.write_text(text, encoding='utf-8')
    selection['stand_arena_scene'] = str(stand_scene)
    selection['stand_arena_scene_sha256'] = sha256(stand_scene)
    samples = [json.loads(line) for line in (run/'gpu-samples.jsonl').read_text().splitlines() if line.strip()]
    values = sorted(s['gpu_percent'] for s in samples)
    selection['gpu'] = dict(samples=len(values), mean_percent=statistics.mean(values),
                            p95_percent=values[int(.95*(len(values)-1))], peak_percent=max(values),
                            scope='Whole-device samples including setup and built-in assessments; not exclusive PPO utilization')
    write(preview/'arena-selection.json', selection)
    write(viewer, dict(schema='mimic_viewer_selection_v1', task='stand', bundle=bundle,
                      checkpoint_sha256=selection['checkpoint_sha256'], experimental_preview=True,
                      previous=previous['bundle'] if previous else None))
    doc = Path('mujoco_rig/mimic/AMP_RECOVERY.md')
    continuation = report.get('continuation')
    initialization = (f"continues checkpoint `{continuation['checkpoint_sha256'][:8]}` with its actor, critic, "
                      'discriminator and normalization; optimizer/RNG state starts a new window'
                      if continuation else 'starts a fresh MimicKit AMP actor, critic, discriminator and normalization')
    motors = report.get('policy_actions', 30)
    neck = ('Neck torque is learned by the policy. ' if motors == 33 else
            'Neck position actuators become zero-command torque motors. ')
    doc.open('a', encoding='utf-8').write(
        f'\n## Spring-free direct-torque Stand window ({datetime.now().astimezone().date().isoformat()})\n\n'
        f'Run `{run.name}` {initialization}. '
        'All passive rest-pose springs are removed. '+neck+
        f'The {motors} policy motors use direct bounded torques, with no standing-pose PD servo. '
        'Damping, joint/torque limits and collision physics remain. Licensed standing and weight-shift '
        'clips supply a style prior; their original dynamic admission does not transfer to this plant.\n\n'
        f'Completed {report["iterations"]} updates / {report["samples"]:,} transitions in '
        f'{report["training_iteration_wall_seconds"]:.3f} synchronized training-update seconds, '
        f'excluding built-in assessments. Cumulative training: '
        f'{report.get("cumulative_training_seconds", report["training_iteration_wall_seconds"])/60:.3f} minutes. '
        f'Final settled standing: {report["final"]["settled_cases"]}/8. '
        f'Conditional 600-second gate passed: {report["quiet_long_hold_passed"]}. '
        'Natural movement is not admitted. The final checkpoint is an experimental preview in '
        '`MimicStandArena.tscn`, with unlimited duration and no balls. '
        'C# contract support was edited; no Godot/C# compilation or standalone tests were run.\n')
    curriculum = report.get('posture_curriculum')
    if curriculum:
        doc.open('a', encoding='utf-8').write(
            f'\nStaged posture ended at `{curriculum["final_stage_name"]}`. '
            f'Stop reason: `{report["stop_reason"]}`. '
            f'Final quiet-retention result: `{report["final_quiet_retention"]}`. '
            'The fixed upright gate remains unchanged. Per-body tilt/clearance, '
            'foot loads, contacts and failed criteria are recorded in report.json. '
            'The retention checkpoint is preserved separately from final model.pt.\n')
    state.update(arena_preview=selection, artifact_finalization_complete=True)
    write(state_path, state)
    print(json.dumps(dict(status='ready', bundle=bundle,
                          training_minutes=report['training_iteration_wall_seconds']/60,
                          settled_cases=report['final']['settled_cases'],
                          stop_reason=report.get('stop_reason'))), flush=True)


def shorten(run, trainer_pid, seconds):
    """Stop only this run after a complete saved update; finalize without more PPO."""
    import shutil
    from .train_torque_stand import complete_saved_window
    requested = dict(training_seconds=seconds, reason='User shortened active Stand run to 15 minutes',
                     trainer_pid=trainer_pid, changed_at_utc=datetime.now(timezone.utc).isoformat())
    write(run/'budget-change.json', requested)
    updates_path = run/'updates.jsonl'
    while True:
        rows = [json.loads(line) for line in updates_path.read_text().splitlines() if line.strip()]
        last = rows[-1]
        if last['training_iteration_wall_seconds']>=seconds:
            # latest.pt is saved every eight updates; evaluation_N is saved every 64.
            complete = [r for r in rows if r['iteration']%8==0 and r['training_iteration_wall_seconds']>=seconds]
            if complete:
                row = complete[-1]
                path = run/(f'evaluation_{row["iteration"]:06d}.pt' if row['iteration']%64==0 else 'latest.pt')
                if path.exists() and path.stat().st_mtime>=updates_path.stat().st_mtime-1:
                    time.sleep(1)
                    snapshot = run/'shortened-final.pt'
                    shutil.copy2(path, snapshot)
                    shutil.copy2(path.with_suffix('.amp.pt'), snapshot.with_suffix('.amp.pt'))
                    # Verify command-line identity before stopping the specific training process.
                    script = f'$p=Get-CimInstance Win32_Process -Filter "ProcessId={trainer_pid}"; if ($p -and $p.CommandLine -match "mujoco_rig.mimic.train_torque_stand") {{ Stop-Process -Id {trainer_pid} }} else {{ throw "Trainer identity changed" }}'
                    subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True)
                    write(run/'shortened-window.json', dict(request=requested, saved_update=row,
                        observed_update_at_stop=last, source_checkpoint=str(path),
                        checkpoint_sha256=sha256(snapshot),
                        scope='Final checkpoint saved at an update boundary; later unsaved updates are excluded from checkpoint training time.'))
                    complete_saved_window(run, snapshot, row)
                    break
        time.sleep(1)
    finish(run)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int)
    parser.add_argument('--seconds', type=float, default=900.)
    args=parser.parse_args()
    if args.trainer_pid:
        shorten(args.run, args.trainer_pid, args.seconds)
    else:
        finish(args.run)
