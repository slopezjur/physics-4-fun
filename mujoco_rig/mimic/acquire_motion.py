"""Fetch pinned, explicitly licensed idle or walking sources and preserve provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PAGE = "https://www.ianxmason.com/100style/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
FILES = {
    "Neutral_ID.bvh": ("1Vnz5wnFmOztNzaSBHjSt9v6y0pl0FK8X", "HIERARCHY",
                       "71aead2e3246bdb47a5b839cb3e60e0574e9a1a046c071cc147e2922a35f259f"),
    "Frame_Cuts.csv": ("1d0VM8k4UjA4dDmaviuZMjAf-WQNLUwnZ", "STYLE_NAME",
                       "e3c1d2beab6486e083622cef6a5402fe9b20edc75a9963ea30675c67ec29ee82"),
}
STEP_FILES = {
    "Neutral_FW.bvh": ("1-TqZdIJvpr-QVvAmYJREUZkuAgpAQSfd", "HIERARCHY",
                       "2f28800baa299547c42c257d1c926172542a1e69b46a3d3e6aa0aded0e9b40f6"),
    "Neutral_BW.bvh": ("1pmJ3GAceoi04kCmXsHvC56TJzKDEHh02", "HIERARCHY",
                       "cfa225ff3a29a38d11ecd1b72fa0d917ca6c3dc3b24b3d09a4cb5c28ee9dc5b9"),
    "Frame_Cuts.csv": FILES["Frame_Cuts.csv"],
}


def verify_source(source: Path, files=None):
    for name, (_, _, expected) in (FILES if files is None else files).items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source content changed; review provenance again: {name}")


def acquire(out: Path, *, steps=False):
    out.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for name, (file_id, prefix, expected) in (STEP_FILES if steps else FILES).items():
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        path = out / name
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(url, timeout=60).read()
        if not data.decode("utf-8").startswith(prefix):
            raise ValueError(f"Unexpected content for {name}")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source content changed; review provenance again: {name}")
        if not path.exists():
            path.write_bytes(data)
        inventory[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    provenance = {"dataset": "100STYLE", "author": "Ian Mason", "source": PAGE,
                  "license": "CC-BY-4.0", "license_url": LICENSE,
                  "credit": "The 100STYLE Dataset - Ian Mason", "verified_date": "2026-09-23" if steps else "2026-09-21",
                  "permission_basis": "Author explicitly permits creative/commercial use with attribution; CC BY 4.0 permits adaptation.",
                  "files": inventory}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", action="store_true", help="Acquire forward/back walking priors instead of idle")
    args = parser.parse_args()
    acquire(args.out, steps=args.steps)
