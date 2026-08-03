#!/usr/bin/env python3
"""Download and verify the five final DBGFN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


TAG = "v1.0.0-reproducibility"
BASE_URL = (
    "https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network"
    f"/releases/download/{TAG}"
)
DESTINATION = Path(__file__).resolve().parent

FINAL = {
    "fold0_best.pt": (
        "fold0_best.pt",
        "e132bda12b09002157d43f530b2880e44c0b6ee50a92eeb40450fefff9023951",
    ),
    "fold1_best.pt": (
        "fold1_best.pt",
        "04cdd21b3fbd7762d6c34f650cb09a4ef06ddb3e2a1289cede3c6c08d58a690d",
    ),
    "fold2_best.pt": (
        "fold2_best.pt",
        "74c6b17bf4b74b64405a2981166daa88c5e958a739c2109b752f420edacde258",
    ),
    "fold3_best.pt": (
        "fold3_best.pt",
        "48953425ddb4fca24cbf0b12c1903a9f5863694f80bf324f92d8f02374cb55e6",
    ),
    "fold4_best.pt": (
        "fold4_best.pt",
        "06a32edfb9471af7aed198bdce771612329ca527b234392bd77402a10b91f162",
    ),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(relative_path: str, asset_name: str, expected_hash: str) -> None:
    destination = DESTINATION / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_hash:
        print(f"verified  {relative_path}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{asset_name}"
    print(f"download  {asset_name} -> {relative_path}")
    request = urllib.request.Request(url, headers={"User-Agent": "DBGFN-reproducer"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if total:
                print(f"\r          {received / total:6.1%}", end="", flush=True)
        if total:
            print()

    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {asset_name}: {actual_hash} != {expected_hash}"
        )
    os.replace(temporary, destination)
    print(f"verified  {relative_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the five final integrated DBGFN checkpoints."
    )
    parser.parse_args()
    try:
        for relative_path, (asset_name, expected_hash) in FINAL.items():
            fetch(relative_path, asset_name, expected_hash)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("All requested checkpoint files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
