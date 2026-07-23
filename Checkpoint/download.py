#!/usr/bin/env python3
"""Download and verify DBGFN checkpoint assets from the GitHub release."""

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

INITIALIZERS = {
    "mace_pretrained_best.pt": (
        "mace_pretrained_best.pt",
        "665a16daf36243a4021fcba8917b163cf122574aa6229da2f90857698c173aed",
    ),
    "ALIGNN/fold0_best.pt": (
        "alignn_fold0_best.pt",
        "7a973170b12c0acf634fd088e43a6dec8834926660ac302d8e22d8f2526ed98e",
    ),
    "ALIGNN/fold1_best.pt": (
        "alignn_fold1_best.pt",
        "76eb4cba1808132c67a0e2759584743bfce6d893b5185635569f92af3869e5af",
    ),
    "ALIGNN/fold2_best.pt": (
        "alignn_fold2_best.pt",
        "f32e5968b5c40c1718788b9a76304d47b266efa71f7dc43bb302b4a5ede422e9",
    ),
    "ALIGNN/fold3_best.pt": (
        "alignn_fold3_best.pt",
        "b0a5884f8f061c04893a7369337c3c7549544442050ae3d388928392a88fc97e",
    ),
    "ALIGNN/fold4_best.pt": (
        "alignn_fold4_best.pt",
        "23ec83c962072925987025aed9a86eb494f8778e42f168de243b0fc32a471531",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-initializers",
        action="store_true",
        help="also download the MACE and fold-specific ALIGNN initialization weights",
    )
    args = parser.parse_args()

    assets = dict(FINAL)
    if args.training_initializers:
        assets.update(INITIALIZERS)
    try:
        for relative_path, (asset_name, expected_hash) in assets.items():
            fetch(relative_path, asset_name, expected_hash)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("All requested checkpoint files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
