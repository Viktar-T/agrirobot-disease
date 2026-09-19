"""Synthetic manifests and caches for the CPU tests of specs 003 and 005 (test_heads.py,
test_n_table.py). No image, backbone or GPU is needed, as in test_probe.py.

A manifest here has the row fields that the heads and the eval read (spec 001 FR-003:
image_id, dataset, sha256, class_km2, split, split_rule, group_keys), a sidecar with its
role, and, when frozen, a line in the FROZEN.jsonl beside it. Its cache (spec 002 FR-005,
FR-006) holds features with a planted class signal: noise plus a fixed direction per class,
the same in every manifest, so a head can learn the class and a held-out class has its own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ms.cache import cache_files, cache_key
from ms.data.manifests import load_manifest

BACKBONE = "bb"
RES = 28
DIM = 16
#: split -> class -> rows; imbalanced like the real sets, anthracnose the rarest trained class
COUNTS = {
    "train": {"healthy": 60, "rust": 30, "anthracnose": 12},
    "val": {"healthy": 10, "rust": 6, "anthracnose": 4},
    "test": {"healthy": 16, "rust": 8, "anthracnose": 6},
    "holdout_unknown": {"unknown_als": 10},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_row(dataset: str, i: int, klass: str, split: str) -> dict:
    image_id = f"{dataset}_{i:05d}"
    return {
        "image_id": image_id,
        "dataset": dataset,
        "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
        "class_km2": klass,
        "split": split,
        "split_rule": "holdout_unknown" if split == "holdout_unknown" else "blocked:date",
        "group_keys": {"date": None, "phash_group": image_id},
    }


def direction(klass: str) -> np.ndarray:
    """A fixed unit vector per class, the same in every manifest."""
    rng = np.random.default_rng(int(hashlib.sha256(klass.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def make_manifest(
    root: Path,
    name: str = "alpha",
    *,
    counts: dict[str, dict[str, int]] | None = None,
    role: str = "train_eval",
    frozen: bool = True,
    signal: float = 3.0,
    seed: int = 0,
    backbone: str = BACKBONE,
) -> Path:
    """<root>/manifests/<name>_v1.jsonl with its sidecar (and, if frozen, its FROZEN.jsonl
    line), and its cache under <root>/cache; returns the manifest's path."""
    rows: list[dict] = []
    for split, per_class in (counts or COUNTS).items():
        for klass, n in per_class.items():
            rows += [image_row(name, len(rows) + i, klass, split) for i in range(n)]
    path = root / "manifests" / f"{name}_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    meta = {"manifest": name, "version": 1, "role": role}
    path.with_name(f"{name}_v1.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if frozen:
        record = {
            "manifest": path.name,
            "sha256": sha256(path),
            "frozen_at": "2026-09-19T00:00:00Z",
            "git_sha": None,
        }
        with (path.parent / "FROZEN.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(rows), DIM))
    x += signal * np.stack([direction(r["class_km2"]) for r in rows])
    write_cache(root / "cache", path, x, backbone=backbone)
    return path


def write_cache(
    cache_root: Path, manifest: Path, x: np.ndarray, *, backbone: str = BACKBONE
) -> Path:
    """The cache of `manifest` for `backbone` at RES, with the arrays and sidecar fields that
    load_cache checks and a head run records (spec 002 FR-005, FR-006); returns the .npz."""
    m = load_manifest(manifest, None, "extract")
    n, d = x.shape
    fields = {
        "backbone_id": backbone,
        "weights_sha256": hashlib.sha256(backbone.encode()).hexdigest(),
        "resolution": RES,
        "preprocess_string": f"resize_short={RES};center_crop={RES};norm=imagenet",
        "compute_dtype": "float32",
    }
    key = cache_key(**fields)
    npz, side, _ = cache_files(cache_root, backbone, RES, key, m.name)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz,
        cls=x.astype(np.float16),
        meanpatch=x[:, ::-1].astype(np.float16),
        image_id=np.array([r["image_id"] for r in m.rows]),
        sha256=np.array([r["sha256"] for r in m.rows]),
    )
    meta = {
        **fields,
        "cache_key": key,
        "hf_id": f"test/{backbone}",
        "revision": None,
        "research_only_until_c5": False,
        "token_types": ["cls", "meanpatch"],
        "storage_dtype": "float16",
        "embed_dim": d,
        "n_images": n,
        "shapes": {"cls": [n, d], "meanpatch": [n, d]},
        "manifest": manifest.name,
        "manifest_sha256": m.sha256,
    }
    side.write_text(json.dumps(meta), encoding="utf-8")
    return npz


def rows_of(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
