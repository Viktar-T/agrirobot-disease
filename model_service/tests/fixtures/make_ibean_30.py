"""Rebuild the CPU fixture tests/fixtures/ibean_30/ (W1 "Fixture"; spec 001 US-1.1, spec 002).

    uv run python model_service/tests/fixtures/make_ibean_30.py                  # images + manifest
    uv run python model_service/tests/fixtures/make_ibean_30.py --manifest-only  # manifest only

Images: 10 per iBean class from data/raw/ibean/, namely the source's own train .0-.5,
validation .0-.1 and test .0-.1. They are copied byte for byte and checked against
data/raw/ibean/SHA256SUMS; SHA256SUMS, DOWNLOAD.json and the MIT licence go with them.
--manifest-only reads the committed images and needs no data/raw/.

Manifest: ibean_v1.jsonl and ibean_v1.meta.json, named as spec 001 names the builder's
output. These are the rows spec 001 prescribes for this set (unblocked, no duplicates).
The builder (ms.data.manifests) arrives in W2, so for now this script writes them. The
content fields follow FR-003, FR-006 (the pinned pHash) and the class map. The split is
this script's own seeded draw: 7/1/2 per trained class, angular leaf spot held out. The
script fails on exact or near duplicates, which only the builder may group.

W2: once the builder exists, it writes the manifest:
`python -m ms.data.manifests build --dataset ibean --raw-root <this folder> --out <this
folder>`. test_manifests.py compares its rows with the committed ones.

Idempotent: unchanged images and manifest are not rewritten, and the sidecar is rewritten
only when something other than built_at and builder changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "ibean_30"
REPO_ROOT = HERE.parents[2]
RAW = REPO_ROOT / "data" / "raw" / "ibean"
CONFIGS = REPO_ROOT / "model_service" / "configs"

CLASSES = ("healthy", "angular_leaf_spot", "bean_rust")
PICK = {"train": ("train", range(6)), "validation": ("val", range(2)), "test": ("test", range(2))}
TARGET = {"train": 0.7, "val": 0.1, "test": 0.2}
TRAINED = ("healthy", "rust", "anthracnose")
FIELDS = (
    "manifest_version",
    "image_id",
    "dataset",
    "source_record",
    "source_version",
    "path",
    "dup_paths",
    "sha256",
    "phash",
    "width",
    "height",
    "format",
    "class_raw",
    "class_km2",
    "boxes",
    "group_keys",
    "split_group",
    "split",
    "split_rule",
    "licence",
    "attribution",
    "notes",
)
GROUP_KEYS = ("district", "subcounty", "date", "region", "session", "variety", "plant_age")

# spec 001 FR-006: orthonormal DCT-II basis at 32 x 32
_K, _N = np.arange(32)[:, None], np.arange(32)[None, :]
_DCT = np.sqrt(2 / 32) * np.cos(np.pi * (2 * _N + 1) * _K / 64)
_DCT[0] /= np.sqrt(2)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phash(path: Path) -> str:
    """Spec 001 FR-006: greyscale, 32 x 32 Lanczos, 2-D DCT-II, top-left 8 x 8 against their
    median, row-major, first bit most significant, 16 hex digits."""
    with Image.open(path) as im:
        grey = ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coef = (_DCT @ np.asarray(grey, dtype=np.float64) @ _DCT.T)[:8, :8]
    bits = "".join("1" if c > np.median(coef) else "0" for c in coef.flatten())
    return f"{int(bits, 2):016x}"


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


def _git_sha() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or None


def write_manifest() -> Path:
    """ibean_v1.jsonl and ibean_v1.meta.json for the committed images (spec 001)."""
    recipe_path = CONFIGS / "manifests" / "ibean.yaml"
    cmap_path = CONFIGS / "class_map_v1.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    cmap = yaml.safe_load(cmap_path.read_text(encoding="utf-8"))
    download = json.loads((FIXTURE / "ibean" / "DOWNLOAD.json").read_text(encoding="utf-8"))
    source = download["source"]
    record = f"https://doi.org/{source['doi']}" if source.get("doi") else source["record_url"]
    version = source.get("version") or (source.get("published") or "")[:10]

    files = sorted(p for p in (FIXTURE / "ibean" / "extracted").rglob("*") if p.is_file())
    rows = []
    for p in files:
        data = p.read_bytes()
        digest = sha256(data)
        with Image.open(p) as im:
            im.load()
            fmt = im.format.lower()
            width, height = ImageOps.exif_transpose(im).size
        class_raw = p.parent.name
        rows.append(
            {
                "manifest_version": "1",
                "image_id": f"ibean_{digest[:16]}",
                "dataset": "ibean",
                "source_record": record,
                "source_version": version,
                "path": p.relative_to(FIXTURE).as_posix(),
                "dup_paths": [],
                "sha256": digest,
                "phash": phash(p),
                "width": width,
                "height": height,
                "format": fmt,
                "class_raw": class_raw,
                "class_km2": cmap["map"]["ibean"][class_raw],
                "boxes": None,
                "group_keys": {k: None for k in GROUP_KEYS},
                "split_group": None,
                "split": None,
                "split_rule": None,
                "licence": download["licence"]["id"],
                "attribution": download["attribution"],
                "notes": None,
            }
        )
    rows.sort(key=lambda r: r["image_id"])

    if len({r["sha256"] for r in rows}) != len(rows):
        sys.exit("exact duplicates: group them with the builder (ms.data.manifests)")
    threshold = recipe["phash_threshold"]
    for i, a in enumerate(rows):
        for b in rows[:i]:
            if (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= threshold:
                sys.exit(f"near duplicates {a['path']} ~ {b['path']}: use the builder")
    for r in rows:
        r["group_keys"]["phash_group"] = r["split_group"] = r["image_id"]

    rng = random.Random(recipe["seed"])
    for cls in sorted({r["class_km2"] for r in rows}):
        group = [r for r in rows if r["class_km2"] == cls]
        if cls not in TRAINED:
            for r in group:
                r["split"] = r["split_rule"] = "holdout_unknown"
            continue
        rng.shuffle(group)
        n_val = max(1, round(TARGET["val"] * len(group)))
        n_test = max(1, round(TARGET["test"] * len(group)))
        n_train = len(group) - n_val - n_test
        for k, r in enumerate(group):
            r["split"] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
            r["split_rule"] = recipe["rule"]
    rows = [{k: r[k] for k in FIELDS} for r in rows]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = FIXTURE / "ibean_v1.jsonl"
    _write_if_changed(manifest, text.encode("utf-8"))

    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(r[key] for r in rows).items()))

    by_split: dict[str, Counter] = {}
    for r in rows:
        by_split.setdefault(r["split"], Counter())[r["class_km2"]] += 1
    trained_total = Counter(r["class_km2"] for r in rows if r["class_km2"] in TRAINED)
    meta = {
        "meta_json_version": 1,
        "manifest": "ibean",
        "version": recipe["version"],
        "role": recipe["role"],
        "rule": recipe["rule"],
        "seed": recipe["seed"],
        "phash_threshold": threshold,
        "class_map": {
            "path": "model_service/configs/class_map_v1.yaml",
            "version": cmap["version"],
            "sha256": sha256(cmap_path.read_bytes()),
        },
        "recipe_sha256": sha256(recipe_path.read_bytes()),
        "members": {"ibean": {"files": len(files), "distinct": len(rows), "shared_with": {}}},
        "counts": {
            "rows": len(rows),
            "class_raw": counts("class_raw"),
            "class_km2": counts("class_km2"),
            "split": counts("split"),
            "split_rule": counts("split_rule"),
            "split_x_class_km2": {s: dict(sorted(c.items())) for s, c in sorted(by_split.items())},
        },
        "fractions": {
            s: {c: round(by_split[s][c] / n, 4) for c, n in sorted(trained_total.items())}
            for s in ("train", "val", "test")
        },
        "splits": None,
        "excluded": [],
        "warnings": ["unblocked"],
        "licence_source": "ibean/DOWNLOAD.json",
        "allow_unknown_licence": False,
        "manifest_sha256": sha256(text.encode("utf-8")),
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": {
            "module": "model_service/tests/fixtures/make_ibean_30.py",
            "git_sha": _git_sha(),
        },
    }
    sidecar = FIXTURE / "ibean_v1.meta.json"
    volatile = ("built_at", "builder")
    if sidecar.exists():
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k not in volatile} == {
            k: v for k, v in meta.items() if k not in volatile
        }:
            return manifest
    sidecar.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


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
