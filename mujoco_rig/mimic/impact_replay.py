"""Select provenance-checked successful native reactions for PPO teacher replay."""
import json

import numpy as np
import torch

from .baseline import sha256


def select_observations(observations, active, cases, samples, dt):
    if samples < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Impact replay requires at least two samples and a positive dt")
    if observations.ndim != 3 or active.shape != observations.shape[:2] or len(cases) != active.shape[1]:
        raise ValueError("Impact replay trace dimensions disagree")
    pools = [[], []]
    for index, case in enumerate(cases):
        if not case["recovered"] or not case["hit"] or case["first_hit_step"] < 0:
            continue
        start = case["first_hit_step"]
        end = min(len(active), start + round(3. / dt))
        times = np.arange(start, end)[active[start:end, index].astype(bool)]
        pools[int(case["recovered_with_step"])].extend((int(t), index) for t in times)
    if not any(pools):
        raise ValueError("Impact replay has no successful post-hit observations")
    # Reserve half the replay for measured stepped recoveries when available.
    quotas = [samples - samples // 2, samples // 2] if all(pools) else [
        samples if pool else 0 for pool in pools]
    selected = []
    for pool, quota in zip(pools, quotas):
        if quota:
            selected.extend(pool[i] for i in np.linspace(0, len(pool)-1, min(quota, len(pool)), dtype=int))
    indices = np.asarray(selected, dtype=np.int64)
    values = observations[indices[:, 0], indices[:, 1]].copy()
    if not np.isfinite(values).all():
        raise ValueError("Impact replay observations must be finite")
    return torch.from_numpy(values), indices


def load_replay(trace, checkpoint, reference, ball_model, rig_path, samples, dt):
    report_path = trace.with_name("report.json")
    report = json.loads(report_path.read_text())
    for key, path in (("checkpoint_sha256", checkpoint), ("reference_sha256", reference),
                      ("ball_sha256", ball_model), ("model_sha256", rig_path)):
        if report.get(key) != sha256(path):
            raise ValueError("Impact replay provenance mismatch: " + key)
    with np.load(trace) as data:
        observations, indices = select_observations(data["observation"], data["active"],
                                                    report["cases"], samples, dt)
    return observations, indices, dict(trace=str(trace), trace_sha256=sha256(trace),
        report_sha256=sha256(report_path), samples=len(observations),
        case_names=sorted({report["cases"][i]["name"] for i in indices[:, 1]}),
        selection="First three seconds after confirmed hits in recovered cases; half reserved for stepped recoveries",
        scope="Training rehearsal; these cases are not held-out validation")
