"""Print a POST /v1/predict request for one manifest row (the W2 slice: one cached image).

    uv run python -m ms.service.example [--manifest data/manifests/ibean_v1.jsonl]
        [--split test] [--index 0 | --image-id <id>] > request.json

The frame is what piece 2's emulator would send when it replays an open-set image: frame_uid
= the row's image_id; uri = its path under data/raw; sha256, width and height from the row.
The metadata is synthetic (platform emulator, pose source synthetic, timestamp = now), and
bbch is null with not_recorded, as for every open-set frame (DECISIONS 30).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ms.data.manifests import load_manifest, manifest_path


def request_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(uuid.uuid4()),
        "contract_version": "v0",
        "frame": {
            "frame_uid": row["image_id"],
            "uri": row["path"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
        },
        "metadata": {
            "site_id": row["dataset"],
            "plot_id": "unknown",
            "zone_id": "open",
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "frame_id": "plot",
                "source": "synthetic",
            },
            "platform": "emulator",
            "camera": {
                "model": "unknown",
                "exposure_us": None,
                "gain": None,
                "flash": False,
                "white_balance": "auto",
            },
            "illumination_mode": "ambient",
            "sowing_batch": "unknown",
            "bbch": None,
            "bbch_null_reason": "not_recorded",
        },
        "options": {"coverage_target": 0.9, "return_patch_map": False},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ms.service.example", description=__doc__.split("\n")[0]
    )
    p.add_argument(
        "--manifest", type=Path, default=None, help="default: data/manifests/ibean_v1.jsonl"
    )
    p.add_argument("--split", default="test")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--index", type=int, default=0)
    which.add_argument("--image-id", default=None)
    args = p.parse_args(argv)
    rows = load_manifest(args.manifest or manifest_path("ibean"), args.split, "serve").rows
    if args.image_id:
        rows = [r for r in rows if r["image_id"] == args.image_id]
        if not rows:
            print(f"no row {args.image_id} in split {args.split}", file=sys.stderr)
            return 2
        row = rows[0]
    else:
        row = rows[args.index]
    print(json.dumps(request_for(row), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
