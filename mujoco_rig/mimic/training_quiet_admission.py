"""Verify long standing evidence for an experimental projectile curriculum."""
import math

from .baseline import sha256
from .validate_stand_foundation import case_passed


def validate_quiet_initialization(report, checkpoint, reference, rig_path):
    for key, path in (("checkpoint_sha256", checkpoint), ("reference_sha256", reference),
                      ("model_sha256", rig_path)):
        if report.get(key) != sha256(path):
            raise ValueError("Quiet initialization provenance mismatch: " + key)
    seconds = report.get("seconds", 0.)
    cases = report.get("cases", [])
    if (not math.isfinite(seconds) or seconds < 600. or not report.get("passed")
            or len(cases) != 8 or any(not case_passed(case, seconds) for case in cases)):
        raise ValueError("Experimental curriculum requires eight passing 600-second standing cases")

