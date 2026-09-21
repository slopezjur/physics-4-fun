"""Snapshot the current working tree and policies without committing or stashing."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = Path("logs/mujoco/2026-09-20_19-14-28_contact_v3_full5m/model_78.pt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(destination: Path, mimickit: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode().split("\0")
    paths = {Path(p) for p in tracked if p and (ROOT / p).is_file()}
    paths.add(ACCEPTED)
    inventory = {}
    for relative in sorted(paths):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        inventory[relative.as_posix()] = sha256(source)
        target = destination / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "project_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "mimickit_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mimickit).decode().strip(),
        "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
        "accepted_checkpoint": ACCEPTED.as_posix(),
        "sha256": inventory,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mimickit", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.out, args.mimickit)
    print(f"Preserved {len(result['sha256'])} files in {args.out}")
