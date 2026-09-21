"""Load the external, pinned MimicKit checkout without editing its sources."""
from pathlib import Path
import subprocess
import sys

MIMICKIT_REVISION = "2ed1e6c"


def activate(mimickit: Path) -> None:
    mimickit = mimickit.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mimickit).decode().strip()
    if not revision.startswith(MIMICKIT_REVISION):
        raise RuntimeError(f"Revalidate MimicKit integration before using revision {revision}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=mimickit
    ).decode().strip()
    if dirty:
        raise RuntimeError("MimicKit has tracked local changes; revalidate a clean checkout first")
    if not (mimickit / "mimickit" / "engines" / "newton_engine.py").is_file():
        raise FileNotFoundError(f"Missing MimicKit engine under {mimickit}")
    sys.path.insert(0, str(mimickit / "mimickit"))
