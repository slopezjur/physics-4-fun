"""Explicit contract checks and native-evaluation-based checkpoint retention."""
import copy
import json
import math
from pathlib import Path

from .baseline import sha256
from .ball_protocol import TARGET_BODIES
from .motion_protocol import tracking_checks


BALL_SELECTION = "directional_retention_v1"
CASE_FIELDS = ("name", "target_body", "direction", "speed", "phase", "launch_step", "trial_steps")
TORSO_BODIES = ("Chest", "Spine", "Pelvis")


def ball_selection_groups(metrics):
    """Validate measured cases before constructing direction and torso retention floors."""
    cases = metrics.get("cases", [])
    if len(cases) != metrics["episodes"] or not cases:
        raise ValueError("Ball selection requires complete per-case evidence")
    signatures = []
    groups = {}
    for case in cases:
        if any(key not in case for key in CASE_FIELDS) or type(case["direction"]) is not int \
                or case["direction"] not in range(4) or case["target_body"] not in TARGET_BODIES:
            raise ValueError("Invalid ball selection case")
        if any(type(case.get(key)) is not bool for key in ("hit", "survived", "recovered")) \
                or case["recovered"] and not (case["hit"] and case["survived"]):
            raise ValueError("Invalid ball case outcomes")
        if not all(math.isfinite(float(case[key])) for key in ("speed", "phase", "launch_step", "trial_steps")):
            raise ValueError("Invalid ball case timing/speed")
        signatures.append(tuple(case[key] for key in CASE_FIELDS))
        labels = [f"direction_{case['direction']}"]
        if case["target_body"] in TORSO_BODIES:
            labels.append(f"direction_{case['direction']}_torso")
        for label in labels:
            counts = groups.setdefault(label, dict(confirmed_hits=0, survived_hits=0, recovered_hits=0))
            counts["confirmed_hits"] += case["hit"]
            counts["survived_hits"] += case["hit"] and case["survived"]
            counts["recovered_hits"] += case["recovered"]
    # Fixed-speed curricula share display names across the two distinct start phases.
    if len(set(signatures)) != len(cases) or len(groups) != 8:
        raise ValueError("Ball selection requires unique cases covering every direction and torso group")
    for key, value in dict(confirmed_hits=sum(c["hit"] for c in cases),
                           survived_hits=sum(c["hit"] and c["survived"] for c in cases),
                           recovered_hits=sum(c["recovered"] for c in cases),
                           successes=sum(c["survived"] for c in cases)).items():
        if metrics[key] != value:
            raise ValueError("Ball aggregate disagrees with per-case outcomes")
    return sorted(signatures), groups


def ball_regressions(candidate, incumbent):
    cases, groups = ball_selection_groups(candidate)
    previous_cases, previous_groups = ball_selection_groups(incumbent)
    if cases != previous_cases or candidate.get("recovery_criteria") != incumbent.get("recovery_criteria"):
        raise ValueError("Cannot compare checkpoints evaluated on different suites")
    return [dict(group=group, metric=key, previous=previous_groups[group][key], candidate=value)
            for group, counts in groups.items() for key, value in counts.items()
            if value < previous_groups[group][key]]


def validate_contract(actual, expected):
    differences = [key for key, value in expected.items()
                   if key != "deployment_status" and actual.get(key) != value]
    if differences:
        raise ValueError("Checkpoint contract mismatch: " + ", ".join(differences))


def standing_passed(metrics):
    return metrics["successes"] == metrics["episodes"] and metrics["mean_root_tracking_error_m"] <= 0.05


def checkpoint_rank(metrics):
    values = [metrics[key] for key in ("episodes", "successes", "mean_survival_seconds", "mean_root_tracking_error_m")]
    count = metrics["episodes"]
    if not all(math.isfinite(float(v)) for v in values) or type(count) is not int or count < 1 \
            or type(metrics["successes"]) is not int or not 0 <= metrics["successes"] <= count or metrics["mean_survival_seconds"] < 0 \
            or metrics["mean_root_tracking_error_m"] < 0:
        raise ValueError("Invalid evaluation metrics")
    if "quiet_standing_passed" in metrics:
        if type(metrics["quiet_standing_passed"]) is not bool:
            raise ValueError("Ball selection requires a measured quiet-standing gate")
        for key in ("confirmed_hits", "survived_hits"):
            if type(metrics.get(key)) is not int or not 0 <= metrics[key] <= count:
                raise ValueError("Ball selection requires measured native hit counts")
        recovered = metrics.get("recovered_hits", 0)
        if type(recovered) is not int or not 0 <= recovered <= metrics["survived_hits"] <= min(metrics["confirmed_hits"], metrics["successes"]):
            raise ValueError("Inconsistent ball survival/recovery counts")
        return (metrics["quiet_standing_passed"], metrics["survived_hits"], recovered, metrics["successes"],
                metrics["mean_survival_seconds"], -metrics["mean_root_tracking_error_m"])
    if count != 8:
        raise ValueError("Stand selection requires eight phases")
    if "motion_tracking" in metrics:
        tracking = metrics["motion_tracking"]
        errors = [(c["mean_foot_error_m"], c["joint_rmse_rad"]) for c in tracking["cases"]]
        if len(errors) != count or not all(math.isfinite(v) and v >= 0 for pair in errors for v in pair):
            raise ValueError("Motion selection requires finite measured tracking errors for every phase")
        if any(len(c["foot_lift_range_m"]) != 2 or not all(math.isfinite(v) and v >= 0 for v in c["foot_lift_range_m"])
               for c in tracking["cases"]):
            raise ValueError("Motion selection requires measured lift for both feet")
        passed = all(tracking_checks(metrics, tracking["cases"]).values())
        if type(tracking["passed"]) is not bool or tracking["passed"] != passed:
            raise ValueError("Inconsistent motion tracking gate")
        return (passed, metrics["successes"], metrics["mean_survival_seconds"],
                -sum(foot for foot, _ in errors) / count, -metrics["mean_root_tracking_error_m"])
    return (standing_passed(metrics), metrics["successes"], metrics["mean_survival_seconds"],
            -metrics["mean_root_tracking_error_m"])


class BestCheckpoint:
    def __init__(self, directory: Path, contract):
        self.path = directory / "best.pt"
        self.metadata_path = directory / "best.json"
        self.contract = contract
        self.metadata = None
        self.decisions = []

    def consider(self, save, metrics, *, iteration, samples):
        rank = checkpoint_rank(metrics)
        ball = "quiet_standing_passed" in metrics
        groups = ball_selection_groups(metrics)[1] if ball else None
        if self.metadata is not None and any(metrics.get(key) != self.metadata["metrics"].get(key)
                                             for key in ("episodes", "evaluation_schema")):
            raise ValueError("Cannot compare checkpoints evaluated on different suites")
        regressions = ball_regressions(metrics, self.metadata["metrics"]) if ball and self.metadata is not None else []
        improved = self.metadata is None or rank > checkpoint_rank(self.metadata["metrics"])
        decision = dict(iteration=iteration, samples=samples, accepted=improved and not regressions,
                        reason="directional_regression" if regressions else "improved_rank" if improved else "rank_not_improved",
                        regressions=regressions)
        if not decision["accepted"]:
            self.decisions.append(decision)
            return False
        temporary = self.path.with_suffix(".pt.tmp")
        save(str(temporary))
        digest = sha256(temporary)
        temporary.replace(self.path)
        self.metadata = {"checkpoint_sha256": digest, "iteration": iteration, "samples": samples,
                         "metrics": copy.deepcopy(metrics), "contract": self.contract,
                         "selection_policy": BALL_SELECTION if ball else "motion_tracking_v1" if "motion_tracking" in metrics else "standing_rank_v1",
                         "retention_groups": groups,
                         "selection": ("retain confirmed hits, survival and recovery in every direction and torso subgroup; then quiet standing gate, survived confirmed hits, settled recovery, successes, survival, root error"
                                       if ball else
                                       "tracking gate, successes, survival, lower foot error, lower root error; fixed eight phases"
                                       if "motion_tracking" in metrics else
                                       "standing gate, successes, survival, then lower root error; fixed eight phases")}
        temp_metadata = self.metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        temp_metadata.replace(self.metadata_path)
        self.decisions.append(decision)
        return True
