"""Load an isolated diagnostic as an initialization, never as admitted motion."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .baseline import sha256


def rotation_offsets(original, seeded):
    """World rotation vectors that pre-multiply native wxyz root rotations."""
    return (Rotation.from_quat(seeded[:, [1, 2, 3, 0]]) *
            Rotation.from_quat(original[:, [1, 2, 3, 0]]).inv()).as_rotvec()


def load_seed_probe(directory, reference_hash, model_hash, poses, dt, window):
    report_path, path = directory / "report.json", directory / "diagnostic_poses.npz"
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "mimic_transition_probe_v1" or
            report.get("source_reference_sha256") != reference_hash or report.get("model_sha256") != model_hash):
        raise ValueError("Seed probe must match the exact source reference and rig")
    start, stop = report["source_frame_start"], report["source_frame_stop_exclusive"]
    if (type(start) is not int or type(stop) is not int or
            not window[0] <= start < stop <= window[1] or not 0 <= start < stop <= len(poses)):
        raise ValueError("Seed probe must lie inside the first editable window")
    with np.load(path, allow_pickle=False) as source:
        seed = source["qpos"]
        if (seed.shape != (stop - start, poses.shape[1]) or not np.isfinite(seed).all()
                or float(source["dt"]) != dt or not np.allclose(np.linalg.norm(seed[:, 3:7], axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError("Seed probe has invalid poses or a different timestep")
    content_hash = sha256(path)
    if report.get("diagnostic_sha256") not in (None, content_hash):
        raise ValueError("Seed probe content hash mismatch")
    result = poses.copy()
    result[start:stop] = seed
    return result, dict(source_frame_start=start, source_frame_stop_exclusive=stop,
                        diagnostic_sha256=content_hash, report_sha256=sha256(report_path),
                        historical_content_hash_verified=report.get("diagnostic_sha256") is not None,
                        probe_anchor_offsets_not_applied=report.get("fit", {}).get("stance_anchor_offsets_xy"),
                        scope="Initialization only; source targets, bounds, anchors and admission remain authoritative")
