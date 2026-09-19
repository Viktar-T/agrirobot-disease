"""Rebuild the CPU fixture tests/fixtures/ibean_30/ (W1 "Fixture"; spec 001 US-1.1, spec 002).

    uv run python model_service/tests/fixtures/make_ibean_30.py                  # images + manifest
    uv run python model_service/tests/fixtures/make_ibean_30.py --manifest-only  # manifest only

Images: 10 per iBean class from data/raw/ibean/, namely the source's own train .0-.5,
validation .0-.1 and test .0-.1. They are copied byte for byte and checked against
data/raw/ibean/SHA256SUMS; SHA256SUMS, DOWNLOAD.json and the MIT licence go with them.
--manifest-only reads the committed images and needs no data/raw/.

Manifest: ibean_v1.jsonl and ibean_v1.meta.json, written by the spec 001 builder
(`python -m ms.data.manifests build --dataset ibean --raw-root <this folder> --out <this
folder>`). Until the builder existed (W1), this script wrote the rows itself; the builder
reproduced every field of them except `split`, its own seeded draw (2026-09-19).

Idempotent: unchanged images and manifest are not rewritten, and the sidecar is rewritten
only when something other than built_at and builder changes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "ibean_30"
REPO_ROOT = HERE.parents[2]
RAW = REPO_ROOT / "data" / "raw" / "ibean"

CLASSES = ("healthy", "angular_leaf_spot", "bean_rust")
PICK = {"train": ("train", range(6)), "validation": ("val", range(2)), "test": ("test", range(2))}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def copy_images() -> None:
    """The 30 images, SHA256SUMS, DOWNLOAD.json and the licence, from data/raw/ibean/."""
    sums = {}
    for line in (RAW / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        sums[rel] = digest
    lines = []
    for cls in CLASSES:
        for split, (tag, indices) in PICK.items():
            for i in indices:
                rel = f"extracted/{split}/{cls}/{cls}_{tag}.{i}.jpg"
                data = (RAW / rel).read_bytes()
                if sha256(data) != sums[rel]:
                    sys.exit(f"{RAW / rel} differs from data/raw/ibean/SHA256SUMS")
                _write_if_changed(FIXTURE / "ibean" / rel, data)
                lines.append(f"{sums[rel]}  {rel}\n")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    _write_if_changed(FIXTURE / "ibean" / "SHA256SUMS", "".join(lines).encode())
    _write_if_changed(FIXTURE / "ibean" / "DOWNLOAD.json", (RAW / "DOWNLOAD.json").read_bytes())
    licence = (REPO_ROOT / "data" / "ibean-master" / "LICENSE").read_text(encoding="utf-8")
    _write_if_changed(FIXTURE / "LICENSE-MIT", licence.replace("\r\n", "\n").encode())


def write_manifest() -> Path:
    """ibean_v1.jsonl and ibean_v1.meta.json for the committed images, by the builder."""
    from ms.data.manifests import build

    if build("ibean", raw_root=FIXTURE, out=FIXTURE) != 0:
        sys.exit("the manifest builder failed on the fixture")
    return FIXTURE / "ibean_v1.jsonl"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest-only", action="store_true", help="keep the committed images")
    args = p.parse_args(argv)
    if not args.manifest_only:
        copy_images()
    manifest = write_manifest()
    print(f"{manifest}: {len(manifest.read_text(encoding='utf-8').splitlines())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
