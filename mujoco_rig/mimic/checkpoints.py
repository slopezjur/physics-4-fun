"""Explicit contract checks and native-evaluation-based checkpoint retention."""
import json
import math
from pathlib import Path

from .baseline import sha256


def validate_contract(actual, expected):
    differences = [key for key, value in expected.items()
                   if key != "deployment_status" and actual.get(key) != value]
    if differences:
        raise ValueError("Checkpoint contract mismatch: " + ", ".join(differences))


def standing_passed(metrics):
    return metrics["successes"] == metrics["episodes"] and metrics["mean_root_tracking_error_m"] <= 0.05


def checkpoint_rank(metrics):
    values = [metrics[key] for key in ("episodes", "successes", "mean_survival_seconds", "mean_root_tracking_error_m")]
    if not all(math.isfinite(float(v)) for v in values) or metrics["episodes"] != 8 \
            or not 0 <= metrics["successes"] <= 8 or metrics["mean_survival_seconds"] < 0 \
            or metrics["mean_root_tracking_error_m"] < 0:
        raise ValueError("Invalid eight-phase evaluation metrics")
    if "quiet_standing_passed" in metrics:
        for key in ("confirmed_hits", "survived_hits"):
            if not isinstance(metrics.get(key), int) or not 0 <= metrics[key] <= 8:
                raise ValueError("Ball selection requires measured native hit counts")
        return (metrics["quiet_standing_passed"], metrics["survived_hits"], metrics["successes"],
                metrics["mean_survival_seconds"], -metrics["mean_root_tracking_error_m"])
    return (standing_passed(metrics), metrics["successes"], metrics["mean_survival_seconds"],
            -metrics["mean_root_tracking_error_m"])


class BestCheckpoint:
    def __init__(self, directory: Path, contract):
        self.path = directory / "best.pt"
        self.metadata_path = directory / "best.json"
        self.contract = contract
        self.metadata = None

    def consider(self, save, metrics, *, iteration, samples):
        rank = checkpoint_rank(metrics)
        if self.metadata is not None and rank <= checkpoint_rank(self.metadata["metrics"]):
            return False
        temporary = self.path.with_suffix(".pt.tmp")
        save(str(temporary))
        digest = sha256(temporary)
        temporary.replace(self.path)
        self.metadata = {"checkpoint_sha256": digest, "iteration": iteration, "samples": samples,
                         "metrics": metrics, "contract": self.contract,
                         "selection": ("quiet standing gate, survived confirmed hits, successes, survival, then root error"
                                       if "quiet_standing_passed" in metrics else
                                       "standing gate, successes, survival, then lower root error; fixed eight phases")}
        temp_metadata = self.metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        temp_metadata.replace(self.metadata_path)
        return True
