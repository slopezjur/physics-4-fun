"""Pair frozen native recovery evidence without hiding failed trajectories."""
import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

from .baseline import sha256


def paired(source, candidate):
    for field in ("seed", "seconds", "target_bodies", "character_model_sha256",
                  "ball_model_sha256", "reference_sha256", "recovery_criteria", "step_criteria"):
        if source.get(field) != candidate.get(field):
            raise ValueError(f"Paired validation mismatch: {field}")
    if len(source["cases"]) != len(candidate["cases"]):
        raise ValueError("Paired validation needs equal case counts")
    for old, new in zip(source["cases"], candidate["cases"]):
        for field in ("name", "posture", "target_body", "angle_radians", "speed", "phase", "launch_step", "trial_steps"):
            if old.get(field) != new.get(field):
                raise ValueError(f"Paired case mismatch: {field}")
    gains = [i for i, (a, b) in enumerate(zip(source["cases"], candidate["cases"])) if b["recovered"] and not a["recovered"]]
    losses = [i for i, (a, b) in enumerate(zip(source["cases"], candidate["cases"])) if a["recovered"] and not b["recovered"]]
    survival_losses = [i for i, (a, b) in enumerate(zip(source["cases"], candidate["cases"])) if a["survived"] and not b["survived"]]
    common = [i for i, (a, b) in enumerate(zip(source["cases"], candidate["cases"])) if a["recovered"] and b["recovered"]]
    quality = {}
    for label, ids in (("all_cases", list(range(len(source["cases"])))), ("common_recovered", common)):
        quality[label] = dict(cases=len(ids), metrics={})
        if not ids:
            continue
        for field in ("contact_slip_distance_by_foot_m", "action_variation_per_second",
                      "action_second_difference_per_second", "load_switches_by_foot",
                      "mean_foot_acceleration_by_foot_m_s2", "rapid_contact_reversals_by_foot"):
            values = [float(np.mean([report["motion_quality"]["cases"][i][field] for i in ids]))
                      for report in (source, candidate)]
            quality[label]["metrics"][field] = dict(source=values[0], candidate=values[1],
                change_percent=100 * (values[1] / values[0] - 1) if values[0] else None)
        if all('articulation_first_5s' in report['motion_quality']['cases'][i]
               for report in (source,candidate) for i in ids):
            for group in ('hip','knee','ankle_toe'):
                for field in ('range_rad','travel_rad'):
                    values = [float(np.mean([report['motion_quality']['cases'][i]
                        ['articulation_first_5s'][group][field] for i in ids])) for report in (source,candidate)]
                    quality[label]['metrics'][f'{group}_{field}_first_5s'] = dict(source=values[0],candidate=values[1],
                        change_percent=100*(values[1]/values[0]-1) if values[0] else None)
    return dict(seed=source["seed"], seconds=source["seconds"], cases=len(source["cases"]),
                source_recovered=source["recovered_hits"], candidate_recovered=candidate["recovered_hits"],
                source_survived=source["survived_hits"], candidate_survived=candidate["survived_hits"],
                source_recovered_steps=source["recovered_with_step"], candidate_recovered_steps=candidate["recovered_with_step"],
                all_hits_confirmed=candidate["confirmed_hits"] == len(candidate["cases"]),
                intended_target_contacts=candidate["intended_target_contacts"],
                recovery_gains=gains, recovery_losses=losses, survival_losses=survival_losses,
                retains_each_source_success=not losses and not survival_losses, quality=quality)


def gpu_usage(run, gpu_csv):
    path = run / "updates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # The last update's append time anchors its monotonic elapsed clock.
    started = datetime.fromtimestamp(path.stat().st_mtime) - timedelta(seconds=rows[-1]["elapsed_seconds"])
    intervals, previous = [], 0.
    for row in rows:
        duration = row["training_iteration_wall_seconds"] - previous
        end = started + timedelta(seconds=row["elapsed_seconds"])
        intervals.append((end - timedelta(seconds=duration), end))
        previous = row["training_iteration_wall_seconds"]
    utilization = []
    with gpu_csv.open() as file:
        for sample in csv.DictReader(file, skipinitialspace=True):
            stamp = datetime.strptime(sample["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
            if any(begin <= stamp <= end for begin, end in intervals):
                utilization.append(float(sample["utilization.gpu [%]"].strip().removesuffix('%').strip()))
    return dict(samples=len(utilization), mean_utilization_percent=float(np.mean(utilization)) if utilization else None,
                peak_utilization_percent=max(utilization) if utilization else None,
                p95_utilization_percent=float(np.percentile(utilization, 95)) if utilization else None,
                scope="Device utilization sampled only during synchronized policy update intervals; native evaluation and calibration excluded")


def run(args):
    source, candidate, quiet, training, calibration = [json.loads(path.read_text()) for path in
        (args.source_report, args.candidate_report, args.quiet_report,
         args.training_run / "report.json", args.training_run / "discriminator-calibration.json")]
    if quiet["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
        raise ValueError("Quiet and hit reports must bind the same frozen actor")
    comparison = paired(source, candidate)
    rows = [json.loads(line) for line in (args.training_run / "updates.jsonl").read_text().splitlines()]
    report = dict(schema="amp_recovery_smooth_followup_v1", source_checkpoint=source["checkpoint"],
        source_sha256=source["checkpoint_sha256"], candidate_checkpoint=candidate["checkpoint"],
        candidate_sha256=candidate["checkpoint_sha256"], iterations=training["iterations"],
        transitions=training["samples"], training_seconds=training["training_iteration_wall_seconds"],
        total_trainer_seconds=training["elapsed_seconds"], paired_unseen=comparison,
        quiet_600_passed=quiet["passed"], quiet_600_successes=quiet["successes"],
        calibration=calibration, final_training_discriminator={key: rows[-1][key] for key in
            ("disc_demo_acc", "disc_agent_acc", "disc_demo_logit", "disc_agent_logit")},
        gpu_training=gpu_usage(args.training_run, args.gpu_csv),
        fixed_initial_recovered=training["initial"]["recovered_hits"],
        fixed_final_recovered=training["final"]["recovered_hits"],
        admitted=False, natural_stepping_gate=False, arena_changed=False,
        scope="Paired physical retention and chatter diagnostics; naturalness still requires visual assessment",
        evidence_hashes={str(path): sha256(path) for path in
            (args.source_report, args.candidate_report, args.quiet_report, args.training_run / "report.json")})
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(quiet_600_passed=report["quiet_600_passed"],
                         paired_unseen={key: value for key, value in comparison.items() if key != "quality"},
                         quality=comparison["quality"], gpu_training=report["gpu_training"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--quiet-report", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
