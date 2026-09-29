"""Select a training seed from completed, hash-bound native demonstration-fit results."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from .baseline import sha256
from .checkpoints import BestCheckpoint, ball_regressions, recovery_step_rank
from .continuous_stand import passing_cases


def run(args):
    report_path = args.fit / "report.json"
    report = json.loads(report_path.read_text())
    protocol = json.loads((args.fit / "protocol.json").read_text())
    if protocol != report["protocol"]:
        raise ValueError("Fit report does not match its protocol")
    contract = json.loads((args.fit / "contract.json").read_text())
    if contract != protocol["contract"]:
        raise ValueError("Fit contract mismatch")
    candidates = []
    for row in report["history"]:
        path = args.fit / f"candidate_{row['iteration']:06d}.pt"
        if sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("Candidate changed after native evaluation")
        metrics = copy.deepcopy(row["metrics"])
        metrics.update(quiet_standing_passed=len(passing_cases(row["quiet"])) == 8,
                       quiet_native=row["quiet"])
        regressions = ball_regressions(metrics, report["baseline"])
        if metrics["quiet_standing_passed"] and not regressions and row.get("eligible_for_further_validation", True):
            candidates.append((row, path, metrics))
    if not candidates:
        raise ValueError("No candidate retains quiet standing and all directional recovery groups")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("agent.yaml", "contract.json"):
        shutil.copyfile(args.fit / name, args.out / name)

    best = BestCheckpoint(args.out, contract, rank_function=recovery_step_rank,
        selection_label="Training seed: retain quiet and every directional survival/recovery group; "
                        "use stepped recoveries after survival/recovery count ties. Not Arena admission.")
    selected_source = None
    for row, path, metrics in candidates:
        if best.consider(lambda destination: shutil.copyfile(path, destination), metrics,
                         iteration=row["iteration"], samples=0):
            selected_source = path
    result = dict(schema="native_demonstration_training_seed_v1", experiment=protocol["experiment"],
        contract=contract, source_fit_report=str(report_path), source_fit_report_sha256=sha256(report_path),
        source_fit_protocol_sha256=sha256(args.fit / "protocol.json"),
        selected_source=str(selected_source), selected_checkpoint_sha256=sha256(args.out / "best.pt"),
        selector_sha256=sha256(Path(__file__)), decisions=best.decisions,
        godot_selected=False, long_quiet_validated=False, unseen_validated=False,
        scope="Same-mass training initialization only; long quiet, unseen and physical parity gates outstanding.")
    (args.out / "protocol.json").write_text(json.dumps(result, indent=2))
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    shutil.copyfile(Path(__file__), args.out / "selector.py")
    print(json.dumps(dict(selected_source=str(selected_source),
        recovered=best.metadata["metrics"]["recovered_hits"],
        recovered_steps=best.metadata["metrics"]["recovered_with_step"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
