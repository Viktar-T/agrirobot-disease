"""Dataset manifests - spec 001 (model_service/specs/001-manifests/spec.md).

    python -m ms.data.manifests build --dataset ibean     # make manifests DS=ibean
    python -m ms.data.manifests validate data/manifests/ibean_v1.jsonl
    python -m ms.data.manifests freeze data/manifests/ibean_v1.jsonl
    python -m ms.data.manifests overlap data/manifests/a_v1.jsonl data/manifests/b_v1.jsonl
    python -m ms.data.manifests crops data/manifests/makerere_v1.jsonl --margin 0.10

One manifest per (manifest id, version): data/manifests/<manifest>_v<N>.jsonl, one row per
distinct image (FR-003), with its sidecar <manifest>_v<N>.meta.json (FR-005). The recipe is
configs/manifests/<manifest>.yaml and the class map configs/class_map_v1.yaml.

Readers: iBean, Makerere, Tanzania (tz155k + tz59k) and SWM (the R-SWM originals); a
manifest without one (ACRE, a stretch goal) stops with `no_reader`.

`crops` cuts the boxes of a manifest into <manifest>_crops_v<N> (US-6, FR-012): JPEG files
in data/derived/<manifest>_crops/, the root its sidecar names.

Exit codes: build 0 (written or unchanged), 2 (build error, nothing written), 3 (would change
a frozen manifest); validate 0, 2 (content violation), 3 (freeze violation); freeze 0 or 2;
overlap 0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

import ms

#: repository root: model_service/src/ms/data/manifests.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "model_service" / "configs"
CLASS_MAP = CONFIG_DIR / "class_map_v1.yaml"
RAW_ROOT = REPO_ROOT / "data" / "raw"
#: derived images (crops, US-6): reproducible from data/raw and a manifest
DERIVED_ROOT = REPO_ROOT / "data" / "derived"
MANIFEST_ROOT = REPO_ROOT / "data" / "manifests"
FROZEN_LIST = "FROZEN.jsonl"

#: FR-003, in the order the rows are written
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
GROUP_KEYS = (
    "district",
    "subcounty",
    "date",
    "region",
    "session",
    "variety",
    "plant_age",
    "phash_group",
)
TRAINED = ("healthy", "rust", "anthracnose")
UNKNOWN = ("unknown_als", "unknown_wm")
UNLABELLED = "none"
KM2 = (*TRAINED, *UNKNOWN, UNLABELLED)
SPLITS = ("train", "val", "test")
HOLDOUT = "holdout_unknown"
TEST_ONLY = "test_only"
UNBLOCKED = "unblocked:random_by_phash_group"
#: FR-008: the share of each trained class's rows that each split aims for
TARGETS = {"train": 0.70, "val": 0.10, "test": 0.20}
LICENCES = ("CC-BY-4.0", "CC0-1.0", "MIT", "unknown")
FORMATS = {"JPEG": "jpeg", "PNG": "png"}
#: FR-004
RULE = re.compile(
    rf"blocked:({'|'.join(k for k in GROUP_KEYS if k != 'phash_group')})"
    rf"|{re.escape(UNBLOCKED)}|{TEST_ONLY}|{HOLDOUT}"
)
#: FR-011: who may read which split
PURPOSES = ("train", "select", "evaluate", "serve", "extract")
RESTRICTED_SPLITS = ("test", HOLDOUT)
RESTRICTED_READERS = ("evaluate", "serve", "extract")

HEX64 = re.compile(r"[0-9a-f]{64}")
HEX16 = re.compile(r"[0-9a-f]{16}")
WORKERS = min(8, os.cpu_count() or 1)

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# --- errors -----------------------------------------------------------------------------------


class ManifestError(Exception):
    """A build or validation failure, carrying its FR-009 reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class TestSplitAccessError(PermissionError):
    """FR-011: test and holdout rows are readable only for evaluate, serve and extract."""

    __test__ = False  # not a pytest class


class FrozenManifestModified(ManifestError):
    """FR-011: a manifest listed in FROZEN.jsonl whose bytes no longer match."""

    def __init__(self, message: str) -> None:
        super().__init__("frozen_manifest_modified", message)


# --- small helpers ----------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def repo_relative(path: Path) -> str:
    """POSIX path relative to the repository root, or the absolute POSIX path outside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_recipe(manifest: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    path = config_dir / "manifests" / f"{manifest}.yaml"
    if not path.exists():
        raise ManifestError("missing_file", f"no recipe {repo_relative(path)} for {manifest!r}")
    return _yaml(path)


def manifest_path(manifest: str, root: Path = MANIFEST_ROOT, version: int | None = None) -> Path:
    """data/manifests/<manifest>_v<N>.jsonl, N from the recipe unless given."""
    if version is None:
        version = int(load_recipe(manifest)["version"])
    return Path(root) / f"{manifest}_v{version}.jsonl"


def sidecar_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.stem + ".meta.json")


def read_frozen(path: Path) -> dict[str, dict[str, Any]]:
    """FROZEN.jsonl as {manifest file name: its freeze record}; an absent list is empty."""
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records.setdefault(record["manifest"], record)
    return records


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _parse_rows(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- pHash (FR-006, pinned bit for bit) ---------------------------------------------------------

_K, _N = np.arange(32)[:, None], np.arange(32)[None, :]
_DCT = np.sqrt(2 / 32) * np.cos(np.pi * (2 * _N + 1) * _K / 64)
_DCT[0] /= np.sqrt(2)


def phash(upright: Image.Image) -> str:
    """64-bit pHash of an orientation-applied image, as 16 hex digits: greyscale (PIL L),
    32 x 32 Lanczos, orthonormal 2-D DCT-II, the top-left 8 x 8 coefficients (DC included)
    against their median, row by row, first bit most significant."""
    grey = upright.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coef = (_DCT @ np.asarray(grey, dtype=np.float64) @ _DCT.T)[:8, :8]
    bits = "".join("1" if c > np.median(coef) else "0" for c in coef.flatten())
    return f"{int(bits, 2):016x}"


def phash_pairs(a: list[str], b: list[str] | None, threshold: int) -> list[tuple[int, int]]:
    """Index pairs (i, j) with Hamming(a[i], b[j]) <= threshold; within `a` when b is None,
    then i < j. Exact all-pairs, in chunks of at most 2^24 comparisons (about 150 MB)."""
    codes_a = np.array([int(h, 16) for h in a], dtype=np.uint64)
    codes_b = codes_a if b is None else np.array([int(h, 16) for h in b], dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    chunk = max(1, min(1024, (1 << 24) // max(1, len(codes_b))))
    for start in range(0, len(codes_a), chunk):
        block = codes_a[start : start + chunk]
        dist = np.bitwise_count(block[:, None] ^ codes_b[None, :])
        for i, j in zip(*np.nonzero(dist <= threshold), strict=True):
            i = int(i) + start
            if b is None and i >= j:
                continue
            pairs.append((i, int(j)))
    return pairs


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


# --- provenance and readers ---------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """What a row copies from its member's DOWNLOAD.json (FR-003)."""

    source_record: str
    source_version: str | None
    licence: str
    attribution: str


def read_provenance(member_dir: Path) -> Provenance:
    path = member_dir / "DOWNLOAD.json"
    if not path.exists():
        raise ManifestError("missing_file", f"{repo_relative(path)} (run make download)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    source = doc["source"]
    record = f"https://doi.org/{source['doi']}" if source.get("doi") else source["record_url"]
    version = source.get("version") or (source.get("published") or "")[:10] or None
    licence = doc["licence"]["id"]
    if licence not in LICENCES:
        raise ManifestError("bad_value", f"{repo_relative(path)}: licence {licence!r}")
    return Provenance(record, version, licence, doc["attribution"])


def read_sha256sums(member_dir: Path) -> dict[str, str]:
    """<member>/SHA256SUMS as {path relative to the member folder: sha256}; absent -> {}."""
    path = member_dir / "SHA256SUMS"
    if not path.exists():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            sums[rel.lstrip("*")] = digest
    return sums


@dataclass(frozen=True)
class Item:
    """A file a reader labelled: its path under the raw root, its label and what the
    source publishes about it. With date_from_exif, the date is the image's EXIF
    DateTimeOriginal, read when the image is decoded."""

    path: str
    class_raw: str
    keys: dict[str, Any] = field(default_factory=dict)
    boxes: list[dict[str, Any]] | None = None
    date_from_exif: bool = False


@dataclass
class Scan:
    """What a reader found under one member: labelled items, annotation files it consumed,
    and files it set aside with a reason (US-1.4: every file lands in exactly one place)."""

    items: list[Item] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)


def _files(folder: Path, member_dir: Path) -> list[Path]:
    """Every file under folder, in POSIX path order (the same on every OS)."""
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(member_dir).as_posix())


def read_ibean(member_dir: Path, member: str) -> Scan:
    """extracted/<split>/<class>/<file>: class_raw is the class folder, verbatim. The source's
    own split stays visible in `path` and is not used (spec 001, dataset table)."""
    root = member_dir / "extracted"
    scan = Scan()
    for p in _files(root, member_dir):
        rel = f"{member}/{p.relative_to(member_dir).as_posix()}"
        parts = p.relative_to(root).parts
        if len(parts) == 3:
            scan.items.append(Item(rel, parts[1]))
        else:
            scan.excluded.append((rel, "no_label"))
    return scan


#: the offset of Makerere's XML datetimes (Uganda, UTC+03:00); file names are read in it
EAT = timezone(timedelta(hours=3))


def _text(element: ET.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip() or None


def _number(value: str | None) -> int | None:
    try:
        return round(float(value)) if value is not None else None
    except ValueError:
        return None


def voc_boxes(annotation: ET.Element) -> list[dict[str, Any]]:
    """Pascal VOC objects as {x, y, w, h, class_raw}, in the file's order and coordinates."""
    boxes = []
    for obj in annotation.iter("object"):
        name, box = _text(obj, "name"), obj.find("bndbox")
        if name is None or box is None:
            continue
        x0, y0, x1, y1 = (_number(_text(box, t)) for t in ("xmin", "ymin", "xmax", "ymax"))
        if None in (x0, y0, x1, y1):
            continue
        boxes.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "class_raw": name})
    return boxes


def _day_from_ms(stem: str) -> str | None:
    """A file named by its capture time in Unix milliseconds: its day in UTC+03:00."""
    if not stem.isdigit() or len(stem) != 13:
        return None
    return datetime.fromtimestamp(int(stem) / 1000, EAT).date().isoformat()


def _day_from_iso(value: str | None) -> str | None:
    """The local calendar day of an ISO 8601 datetime, as written (its own offset)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None


def read_makerere(member_dir: Path, member: str) -> Scan:
    """extracted/<archive>/<archive>/<unix ms>.jpg. Angular leaf spot and rust images have an
    XML beside them: class, district, subcounty, datetime (the date), variety, age (plant_age)
    and boxes. Healthy images have none: class_raw is their folder without its chunk digits,
    and the date is the file name read as Unix milliseconds in UTC+03:00, never EXIF, which
    was rewritten after the trip (spec 001, Edge cases; DECISIONS 17). An image without XML
    outside a healthy folder has no label."""
    root = member_dir / "extracted"
    scan = Scan()
    files = _files(root, member_dir)
    xml = {p.with_suffix(""): p for p in files if p.suffix.lower() == ".xml"}
    for p in files:
        rel = f"{member}/{p.relative_to(member_dir).as_posix()}"
        if p.suffix.lower() == ".xml":
            scan.annotations.append(rel)
            continue
        annotation = None
        if p.with_suffix("") in xml:
            try:
                annotation = ET.parse(xml[p.with_suffix("")]).getroot()
            except ET.ParseError:
                annotation = None
        if annotation is not None and _text(annotation, "class"):
            keys = {
                "district": _text(annotation, "district"),
                "subcounty": _text(annotation, "subcounty"),
                "date": _day_from_iso(_text(annotation, "datetime")),
                "variety": _text(annotation, "variety"),
                "plant_age": _number(_text(annotation, "age")),
            }
            scan.items.append(Item(rel, _text(annotation, "class"), keys, voc_boxes(annotation)))
        elif re.sub(r"_\d+$", "", p.parent.name) == "healthy":
            scan.items.append(Item(rel, "healthy", {"date": _day_from_ms(p.stem)}))
        else:
            scan.excluded.append((rel, "no_label"))
    return scan


def read_tanzania(member_dir: Path, member: str) -> Scan:
    """extracted/<archive>/<class folder>/*.jpg (tz155k and tz59k alike): class_raw is the
    class folder without its chunk digits (healthy1..healthy11 -> healthy); the date is EXIF
    DateTimeOriginal, never IFD0 DateTime, and never the file name, an upload time."""
    root = member_dir / "extracted"
    scan = Scan()
    for p in _files(root, member_dir):
        rel = f"{member}/{p.relative_to(member_dir).as_posix()}"
        parts = p.relative_to(root).parts
        if len(parts) == 3:
            scan.items.append(Item(rel, re.sub(r"\d+$", "", parts[1]), date_from_exif=True))
        else:
            scan.excluded.append((rel, "no_label"))
    return scan


SWM_BASE = ("extracted", "r-swm-dataset", "r-swm-dataset")


def read_swm(member_dir: Path, member: str) -> Scan:
    """The R-SWM originals, original-images/<split>/images/<name>.jpg, labelled by the Pascal
    VOC file pascal-voc/<split>/<name>.xml: class_raw is its object names, distinct, sorted,
    joined " + ". The renderings in original-images/<split>/bbox/ have the boxes drawn in
    (drawn_boxes). date and session come from the name, ds-<date>-<place>-<image>. The
    300-px SWM crops are swm_crops (US-6), not rows here."""
    base = member_dir.joinpath(*SWM_BASE)
    scan = Scan()
    for p in _files(base / "pascal-voc", member_dir):
        scan.annotations.append(f"{member}/{p.relative_to(member_dir).as_posix()}")
    originals = base / "original-images"
    for p in _files(originals, member_dir):
        rel = f"{member}/{p.relative_to(member_dir).as_posix()}"
        parts = p.relative_to(originals).parts
        if len(parts) == 3 and parts[1] == "bbox":
            scan.excluded.append((rel, "drawn_boxes"))
            continue
        voc = base / "pascal-voc" / parts[0] / f"{p.stem}.xml" if len(parts) == 3 else None
        if parts[1:2] != ("images",) or voc is None or not voc.is_file():
            scan.excluded.append((rel, "no_label"))
            continue
        try:
            boxes = voc_boxes(ET.parse(voc).getroot())
        except ET.ParseError:
            boxes = []
        names = sorted({b["class_raw"] for b in boxes})
        if not names:
            scan.excluded.append((rel, "no_label"))
            continue
        named = re.match(r"ds-(\d{4}-\d{2}-\d{2})-", p.stem)
        keys = {
            "date": named.group(1) if named else None,
            "session": p.stem.rsplit("-", 1)[0] if named else None,
        }
        scan.items.append(Item(rel, " + ".join(names), keys, boxes))
    return scan


#: manifest id -> reader(member folder, member id)
READERS: dict[str, Callable[[Path, str], Scan]] = {
    "ibean": read_ibean,
    "makerere": read_makerere,
    "tanzania": read_tanzania,
    "swm": read_swm,
}


# --- inspecting files ---------------------------------------------------------------------------


@dataclass
class Seen:
    """One scanned file after hashing and decoding."""

    member: str
    path: str
    sha256: str
    item: Item | None  # None: set aside by the reader
    reason: str | None = None  # not_an_image, decode_error, or the reader's reason
    format: str | None = None
    width: int | None = None
    height: int | None = None
    phash: str | None = None
    exif_date: str | None = None


def _exif_day(image: Image.Image) -> str | None:
    """EXIF DateTimeOriginal ("YYYY:MM:DD HH:MM:SS", local time) as its calendar day."""
    try:
        taken = image.getexif().get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal)
    except Exception:  # noqa: BLE001 - a broken EXIF block means no date, not no image
        return None
    found = re.match(r"(\d{4}):(\d{2}):(\d{2})", str(taken or "").strip("\x00 "))
    if not found:
        return None
    try:
        return date(*map(int, found.groups())).isoformat()
    except ValueError:
        return None


def _inspect(raw_root: Path, member: str, item: Item) -> Seen:
    data = (raw_root / item.path).read_bytes()
    seen = Seen(member, item.path, sha256_bytes(data), item)
    try:
        im = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError):
        seen.reason = "not_an_image"
        return seen
    if im.format not in FORMATS:
        seen.reason = "not_an_image"
        return seen
    if item.date_from_exif:
        seen.exif_date = _exif_day(im)
    try:
        im.load()
        upright = ImageOps.exif_transpose(im)
        seen.width, seen.height = upright.size
        seen.phash = phash(upright)
    except Exception:  # noqa: BLE001 - any failure to decode is the same verdict
        seen.reason = "decode_error"
        return seen
    seen.format = FORMATS[im.format]
    return seen


def _hash_only(raw_root: Path, member: str, path: str, reason: str) -> Seen:
    return Seen(member, path, sha256_file(raw_root / path), None, reason)


# --- splits -------------------------------------------------------------------------------------


#: FR-008: up to this many groups every assignment is scored (3^12 = 531,441) ...
EXHAUSTIVE_GROUPS = 12
#: ... and up to this many, the greedy one is improved by moves and swaps
LOCAL_SEARCH_GROUPS = 500


def assign_splits(groups: dict[str, Counter], seed: int) -> tuple[dict[str, str], dict]:
    """FR-008: whole groups into train/val/test, as close to TARGETS as can be for every
    trained class, with every class present in every split.

    The distance is the sum over splits and classes of ((rows - target rows) / class rows)^2.
    Up to 12 groups (a blocked split: Makerere has 11 blocks), every assignment is scored and
    the closest is taken, ties broken by the seed. Beyond that, a seeded greedy (groups
    shuffled, then largest first, each into the split that brings its classes closest to
    their targets), repair moves for a class a split lacks and, up to 500 groups, single
    moves and pairwise swaps while they bring the split closer. ManifestError
    `class_missing_from_split` when no assignment puts every class in every split.
    Returns ({group: split}, {split: Counter of rows per class}).
    """
    total: Counter = Counter()
    for counts in groups.values():
        total.update(counts)
    classes = [c for c in TRAINED if total[c]]
    names = sorted(groups)
    for c in classes:
        holding = sum(1 for g in names if groups[g][c])
        if holding < len(SPLITS):
            raise ManifestError(
                "class_missing_from_split",
                f"{c}: its {total[c]} rows lie in {holding} group(s), which cannot reach "
                "train, val and test all at once",
            )
    if len(names) <= EXHAUSTIVE_GROUPS:
        where = _closest(groups, names, classes, total, seed)
    else:
        where = _greedy(groups, names, classes, total, seed)
    have: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for g, s in where.items():
        have[s].update(groups[g])
    if EXHAUSTIVE_GROUPS < len(names) <= LOCAL_SEARCH_GROUPS:
        _improve(groups, names, classes, total, where, have)
    return where, have


def _closest(
    groups: dict[str, Counter], names: list[str], classes: list[str], total: Counter, seed: int
) -> dict[str, str]:
    """Score every assignment of the groups to the splits; the closest that covers."""
    counts = np.array([[groups[g][c] for c in classes] for g in names], dtype=float)
    rows = np.array([total[c] for c in classes], dtype=float)
    options = np.array(list(itertools.product(range(len(SPLITS)), repeat=len(names))), np.int8)
    cost = np.zeros(len(options))
    covers = np.ones(len(options), dtype=bool)
    for k, s in enumerate(SPLITS):
        have = (options == k).astype(float) @ counts
        cost += (((have - TARGETS[s] * rows) / rows) ** 2).sum(axis=1)
        covers &= (have > 0).all(axis=1)
    if not covers.any():
        raise ManifestError(
            "class_missing_from_split",
            "no assignment of the whole groups puts every class in every split: "
            + ", ".join(
                f"{c} in {sum(1 for g in names if groups[g][c])} group(s)" for c in classes
            ),
        )
    best = np.flatnonzero(covers & (cost <= cost[covers].min() + 1e-12))
    pick = best[random.Random(seed).randrange(len(best))]
    return {g: SPLITS[options[pick, i]] for i, g in enumerate(names)}


def _cost(counts: dict[str, int], s: str, classes: list[str], total: Counter) -> float:
    return sum(((counts.get(c, 0) - TARGETS[s] * total[c]) / total[c]) ** 2 for c in classes)


def _greedy(
    groups: dict[str, Counter], names: list[str], classes: list[str], total: Counter, seed: int
) -> dict[str, str]:
    """Seeded greedy, largest group first, then repair moves for a class a split lacks."""
    have: dict[str, Counter] = {s: Counter() for s in SPLITS}
    want = {s: {c: TARGETS[s] * total[c] for c in classes} for s in SPLITS}

    def err(s: str, c: str, n: float) -> float:
        return ((n - want[s][c]) / total[c]) ** 2

    def change(g: str, s: str, sign: int) -> float:
        # per class, as the first builder did: the same float rounding keeps its splits
        # (ibean_v1, the fixture) bit for bit
        return sum(
            err(s, c, have[s][c] + sign * k) - err(s, c, have[s][c]) for c, k in groups[g].items()
        )

    order = list(names)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sum(groups[g].values()))  # stable: equal sizes keep the draw
    where: dict[str, str] = {}
    for g in order:
        s = min(SPLITS, key=lambda s: change(g, s, +1))
        where[g] = s
        have[s].update(groups[g])

    while gaps := [(c, s) for c in classes for s in SPLITS if not have[s][c]]:
        c, s = gaps[0]
        best: tuple[float, str] | None = None
        for g in names:
            src = where[g]
            if src == s or not groups[g][c]:
                continue
            if any(have[src][k] - n <= 0 for k, n in groups[g].items() if n):
                continue  # the move would open a gap in the split it leaves
            delta = change(g, s, +1) + change(g, src, -1)
            if best is None or delta < best[0]:
                best = (delta, g)
        if best is None:
            raise ManifestError(
                "class_missing_from_split",
                f"{c} has no rows in {s}, and no whole group can move there without "
                "emptying another split",
            )
        g = best[1]
        have[where[g]].subtract(groups[g])
        where[g] = s
        have[s].update(groups[g])
    return where


def _improve(
    groups: dict[str, Counter],
    names: list[str],
    classes: list[str],
    total: Counter,
    where: dict[str, str],
    have: dict[str, Counter],
) -> None:
    """Best single move or pairwise swap while one brings the split closer to its targets
    and keeps every class in every split (in place)."""
    while True:
        now = {s: _cost(have[s], s, classes, total) for s in SPLITS}
        best_delta, best_move = -1e-12, ()
        for move, changes in _moves(groups, names, classes, where, have):
            if any(not all(new[c] > 0 for c in classes) for new in changes.values()):
                continue
            delta = sum(_cost(new, s, classes, total) - now[s] for s, new in changes.items())
            if delta < best_delta:
                best_delta, best_move = delta, move
        if not best_move:
            return
        for g, s in zip(best_move[::2], best_move[1::2], strict=True):
            have[where[g]].subtract(groups[g])
            where[g] = s
            have[s].update(groups[g])


def _moves(
    groups: dict[str, Counter],
    names: list[str],
    classes: list[str],
    where: dict[str, str],
    have: dict[str, Counter],
) -> Iterable[tuple[tuple[str, ...], dict[str, dict[str, int]]]]:
    """Every single move and pairwise swap, as ((group, split, ...), the new class counts of
    the splits it changes)."""
    for g in names:
        src = where[g]
        for dst in SPLITS:
            if dst != src:
                yield (
                    (g, dst),
                    {
                        src: {c: have[src][c] - groups[g][c] for c in classes},
                        dst: {c: have[dst][c] + groups[g][c] for c in classes},
                    },
                )
    for i, g1 in enumerate(names):
        for g2 in names[i + 1 :]:
            s1, s2 = where[g1], where[g2]
            if s1 != s2:
                d = {c: groups[g2][c] - groups[g1][c] for c in classes}
                yield (
                    (g1, s2, g2, s1),
                    {
                        s1: {c: have[s1][c] + d[c] for c in classes},
                        s2: {c: have[s2][c] - d[c] for c in classes},
                    },
                )


# --- build --------------------------------------------------------------------------------------


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build(
    dataset: str,
    *,
    raw_root: Path = RAW_ROOT,
    out: Path = MANIFEST_ROOT,
    version: int | None = None,
    seed: int | None = None,
    rule: str | None = None,
    phash_threshold: int | None = None,
    allow_unknown_licence: bool = False,
    config_dir: Path = CONFIG_DIR,
    log: Log = _print,
) -> int:
    """Build data/manifests/<dataset>_v<N>.jsonl and its sidecar; the exit code."""
    try:
        return _build(
            dataset,
            raw_root=Path(raw_root),
            out=Path(out),
            version=version,
            seed=seed,
            rule=rule,
            phash_threshold=phash_threshold,
            allow_unknown_licence=allow_unknown_licence,
            config_dir=Path(config_dir),
            log=log,
        )
    except ManifestError as exc:
        log(str(exc))
        return 2


def _build(
    dataset: str,
    *,
    raw_root: Path,
    out: Path,
    version: int | None,
    seed: int | None,
    rule: str | None,
    phash_threshold: int | None,
    allow_unknown_licence: bool,
    config_dir: Path,
    log: Log,
) -> int:
    recipe_file = config_dir / "manifests" / f"{dataset}.yaml"
    recipe = load_recipe(dataset, config_dir)
    version = int(recipe["version"] if version is None else version)
    seed = int(recipe["seed"] if seed is None else seed)
    rule = recipe["rule"] if rule is None else rule
    threshold = int(recipe["phash_threshold"] if phash_threshold is None else phash_threshold)
    members: list[str] = list(recipe["members"])
    if not RULE.fullmatch(rule):
        raise ManifestError("bad_split_rule", f"{rule!r} is not in the FR-004 vocabulary")
    reader = READERS.get(dataset)
    if reader is None:
        raise ManifestError(
            "no_reader",
            f"manifest {dataset!r} has no reader (readers: {', '.join(READERS)})",
        )
    class_map_file = config_dir / "class_map_v1.yaml"
    class_map = _yaml(class_map_file)
    table: dict[str, str] = class_map["map"].get(dataset, {})

    # scan, hash and decode every file of every member, in member order then path order
    provenance: dict[str, Provenance] = {}
    seen: list[Seen] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for member in members:
            member_dir = raw_root / member
            if not member_dir.is_dir():
                raise ManifestError(
                    "missing_file",
                    f"{member_dir} does not exist (make download / make extract DS={member})",
                )
            provenance[member] = read_provenance(member_dir)
            scan = reader(member_dir, member)
            jobs = [pool.submit(_inspect, raw_root, member, item) for item in scan.items]
            jobs += [pool.submit(_hash_only, raw_root, member, p, r) for p, r in scan.excluded]
            found = sorted((j.result() for j in jobs), key=lambda s: s.path)
            sums = read_sha256sums(member_dir)
            for s in found:
                expected = sums.get(s.path.split("/", 1)[1])
                if expected is not None and expected != s.sha256:
                    raise ManifestError(
                        "sha256_mismatch",
                        f"{s.path}: sha256 {s.sha256}, {member}/SHA256SUMS says {expected}",
                    )
            seen += found

    distinct = {m: {s.sha256 for s in seen if s.member == m} for m in members}
    stats = {
        m: {
            "files": sum(1 for s in seen if s.member == m),
            "distinct": len(distinct[m]),
            "shared_with": {o: len(distinct[m] & distinct[o]) for o in members if o != m},
        }
        for m in members
    }
    excluded = [(s.path, s.reason) for s in seen if s.reason]
    images = [s for s in seen if not s.reason]
    unmapped = sorted({s.item.class_raw for s in images} - set(table))
    if unmapped:
        raise ManifestError(
            "unmapped_class",
            f"{', '.join(map(repr, unmapped))} not in {repo_relative(class_map_file)} "
            f"map.{dataset}",
        )

    copies: dict[str, list[Seen]] = defaultdict(list)
    for s in images:  # member order, then path order: the first copy is the row's path
        copies[s.sha256].append(s)
    rows: list[dict[str, Any]] = []
    for digest, group in copies.items():
        labels = {s.item.class_raw for s in group}
        if len(labels) > 1:
            excluded += [(s.path, "label_conflict") for s in group]
            continue
        (class_raw,) = labels
        km2 = table[class_raw]
        if km2 == "excluded":
            excluded += [(s.path, "excluded_by_class_map") for s in group]
            continue
        first, prov = group[0], provenance[group[0].member]
        keys = {k: first.item.keys.get(k) for k in GROUP_KEYS}
        if first.item.date_from_exif:
            keys["date"] = first.exif_date
        rows.append(
            {
                "manifest_version": str(version),
                "image_id": f"{dataset}_{digest[:16]}",
                "dataset": dataset,
                "source_record": prov.source_record,
                "source_version": prov.source_version,
                "path": first.path,
                "dup_paths": sorted(s.path for s in group[1:]),
                "sha256": digest,
                "phash": first.phash,
                "width": first.width,
                "height": first.height,
                "format": first.format,
                "class_raw": class_raw,
                "class_km2": km2,
                "boxes": first.item.boxes,
                "group_keys": keys,
                "split_group": None,
                "split": None,
                "split_rule": None,
                "licence": prov.licence,
                "attribution": prov.attribution,
                "notes": None,
            }
        )
    unknown_licence = sorted({r["path"].split("/")[0] for r in rows if r["licence"] == "unknown"})
    if unknown_licence and not allow_unknown_licence:
        raise ManifestError(
            "licence_unknown",
            f"member(s) {', '.join(unknown_licence)} state no licence; build with "
            "--allow-unknown-licence (recorded in the sidecar), which freeze still refuses",
        )
    rows.sort(key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate_image_id", "two images share a sha256 prefix of 16 digits")

    # near duplicates: pHash components; the group id is the smallest image_id in it
    uf = _UnionFind(len(rows))
    for i, j in phash_pairs([r["phash"] for r in rows], None, threshold):
        uf.union(i, j)
    members_of: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        members_of[uf.find(i)].append(i)
    warnings: list[str] = []
    for component in members_of.values():
        gid = rows[component[0]]["image_id"]  # rows are sorted: the first is the smallest
        for i in component:
            rows[i]["group_keys"]["phash_group"] = gid
        if len({rows[i]["class_km2"] for i in component}) > 1:
            warnings.append("mixed_class_phash_group")

    splits_listing = _split(rows, rule, seed)
    if rule.startswith("blocked:") and any(r["split_rule"] == UNBLOCKED for r in rows):
        warnings.append("partly_unblocked")
    if rule == UNBLOCKED and any(r["split_rule"] == UNBLOCKED for r in rows):
        warnings.append("unblocked")
    rows = [{k: r[k] for k in FIELDS} for r in rows]
    text = _jsonl(rows)
    digest = sha256_bytes(text.encode("utf-8"))

    target = out / f"{dataset}_v{version}.jsonl"
    frozen = read_frozen(out / FROZEN_LIST).get(target.name)
    if frozen is not None and frozen["sha256"] != digest:
        log(
            f"frozen_manifest_modified: {target.name} is frozen (sha256 {frozen['sha256']}); "
            f"this build would change it. Build --version {version + 1} instead."
        )
        return 3

    trained = [r for r in rows if r["class_km2"] in TRAINED]
    total = Counter(r["class_km2"] for r in trained)
    by_split: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_split[r["split"]][r["class_km2"]] += 1
    meta = {
        "meta_json_version": 1,
        "manifest": dataset,
        "version": version,
        "role": recipe["role"],
        "rule": rule,
        "seed": seed,
        "phash_threshold": threshold,
        "class_map": {
            "path": repo_relative(class_map_file),
            "version": class_map["version"],
            "sha256": sha256_file(class_map_file),
        },
        "recipe_sha256": sha256_file(recipe_file),
        "members": stats,
        "counts": {
            "rows": len(rows),
            "class_raw": _counts(r["class_raw"] for r in rows),
            "class_km2": _counts(r["class_km2"] for r in rows),
            "split": _counts(r["split"] for r in rows),
            "split_rule": _counts(r["split_rule"] for r in rows),
            "split_x_class_km2": {s: dict(sorted(c.items())) for s, c in sorted(by_split.items())},
        },
        "fractions": {
            s: {c: round(by_split[s][c] / n, 4) for c, n in sorted(total.items())}
            for s in SPLITS
            if any(r["split"] == s for r in trained)
        },
        "splits": splits_listing,
        "excluded": [{"path": p, "reason": r} for p, r in sorted(excluded)],
        "warnings": sorted(set(warnings)),
        "licence_source": ", ".join(f"{m}/DOWNLOAD.json" for m in members),
        "allow_unknown_licence": allow_unknown_licence,
        "manifest_sha256": digest,
        "built_at": _now(),
        "builder": {"module": "ms.data.manifests", "version": ms.__version__, "git_sha": git_sha()},
    }

    written = _write_if_changed(target, text.encode("utf-8"))
    _write_sidecar(sidecar_path(target), meta)
    state = "written" if written else "unchanged"
    log(
        f"{repo_relative(target)}: {len(rows)} rows ({state}); class_km2 "
        f"{meta['counts']['class_km2']}; split {meta['counts']['split']}; excluded "
        f"{len(excluded)}; warnings {meta['warnings'] or 'none'}; sha256 {digest}"
    )
    return 0


def _split(rows: list[dict[str, Any]], rule: str, seed: int) -> dict | None:
    """Set split, split_rule and split_group on every row (FR-004, FR-008); the sidecar's
    `splits` listing (blocked rules only)."""
    trained = []
    for r in rows:
        r["split_group"] = r["group_keys"]["phash_group"]
        if r["class_km2"] in UNKNOWN or rule == HOLDOUT:
            r["split"] = r["split_rule"] = HOLDOUT
        elif r["class_km2"] == UNLABELLED or rule == TEST_ONLY:
            r["split"], r["split_rule"] = "test", TEST_ONLY
        else:
            trained.append(r)
    if not trained:
        return None
    for r in trained:
        r["split_rule"] = UNBLOCKED
    if rule.startswith("blocked:"):
        key = rule.removeprefix("blocked:")
        if not any(r["group_keys"].get(key) is not None for r in rows):
            raise ManifestError("rule_not_applicable", f"{rule}: no row has group_keys.{key}")
        _blocks(rows, trained, key, rule)
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in trained:
        groups[r["split_group"]][r["class_km2"]] += 1
    where, _ = assign_splits(groups, seed)
    for r in trained:
        r["split"] = where[r["split_group"]]
    if not rule.startswith("blocked:"):
        return None
    listing: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for r in trained:
        if r["split_rule"] == rule:
            listing[r["split"]][r["split_group"]] = listing[r["split"]].get(r["split_group"], 0) + 1
    return {s: dict(sorted(blocks.items())) for s, blocks in listing.items()}


def _blocks(rows: list[dict[str, Any]], trained: list[dict[str, Any]], key: str, rule: str) -> None:
    """FR-006: the blocks are the connected components of the trained rows under three links:
    the same key value; the same phash_group; and, for a row without the key, the key values
    that rows of any class (held-out ones too) carry on its capture date (US-3.2). A block's
    id is <key>:<its values, sorted, joined by +>. A row no key value reaches keeps its
    phash_group and split_rule unblocked:random_by_phash_group."""
    on_date: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        value, day = r["group_keys"].get(key), r["group_keys"].get("date")
        if value is not None and day:
            on_date[day].add(str(value))
    values = sorted(
        {str(r["group_keys"][key]) for r in trained if r["group_keys"].get(key) is not None}
        | {v for vs in on_date.values() for v in vs}
    )
    node = {v: len(trained) + i for i, v in enumerate(values)}
    uf = _UnionFind(len(trained) + len(values))
    first_of_group: dict[str, int] = {}
    for i, r in enumerate(trained):
        value, day = r["group_keys"].get(key), r["group_keys"].get("date")
        if value is not None:
            uf.union(i, node[str(value)])
        else:
            for v in on_date.get(day, ()) if day else ():
                uf.union(i, node[v])
        g = r["group_keys"]["phash_group"]
        if g in first_of_group:
            uf.union(i, first_of_group[g])
        else:
            first_of_group[g] = i
    reached: dict[int, set[str]] = defaultdict(set)
    for v, k in node.items():
        reached[uf.find(k)].add(v)
    for i, r in enumerate(trained):
        found = reached.get(uf.find(i))
        if found:
            r["split_group"], r["split_rule"] = f"{key}:{'+'.join(sorted(found))}", rule


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


VOLATILE = ("built_at", "builder")


def _write_sidecar(path: Path, meta: dict[str, Any]) -> bool:
    """FR-005: rewritten only when something other than built_at and the builder changed."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k not in VOLATILE} == {
            k: v for k, v in meta.items() if k not in VOLATILE
        }:
            return False
    text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    return _write_if_changed(path, text.encode("utf-8"))


# --- validate -----------------------------------------------------------------------------------


def check_rows(
    rows: list[dict[str, Any]], meta: dict[str, Any] | None, raw_root: Path | None
) -> list[tuple[str, str]]:
    """FR-009 content checks, in the order they are printed: (reason, message)."""
    problems: list[tuple[str, str]] = []
    meta = meta or {}

    def bad(reason: str, message: str) -> None:
        problems.append((reason, message))

    ok_rows: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    previous = None
    for n, r in enumerate(rows, 1):
        missing = [f for f in FIELDS if f not in r]
        if isinstance(r.get("group_keys"), dict):
            missing += [f"group_keys.{k}" for k in GROUP_KEYS if k not in r["group_keys"]]
        if missing:
            bad("missing_field", f"row {n} lacks {', '.join(missing)}")
            continue
        wrong = _bad_values(r)
        if wrong:
            bad("bad_value", f"row {n}: {'; '.join(wrong)}")
        if not isinstance(r["split_rule"], str) or not RULE.fullmatch(r["split_rule"]):
            bad("bad_split_rule", f"row {n}: {r['split_rule']!r}")
        if r["image_id"] in seen_ids:
            bad("duplicate_image_id", f"row {n}: {r['image_id']} appears twice")
        elif previous is not None and str(r["image_id"]) < previous:
            bad("rows_not_sorted", f"row {n}: {r['image_id']} after {previous}")
        seen_ids.add(r["image_id"])
        previous = str(r["image_id"])
        ok_rows.append((n, r))

    if raw_root is not None:
        paths = [(n, r["path"], r["sha256"]) for n, r in ok_rows]

        def check_file(item: tuple[int, str, str]) -> tuple[str, str] | None:
            n, path, digest = item
            file = raw_root / path
            if not file.is_file():
                return "missing_file", f"row {n}: {path}"
            actual = sha256_file(file)
            if actual != digest:
                return "sha256_mismatch", f"row {n}: {path} has sha256 {actual}, not {digest}"
            return None

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            problems += [p for p in pool.map(check_file, paths) if p]

    for kind in ("phash_group", "split_group"):
        splits_of: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for n, r in ok_rows:
            gid = r["group_keys"]["phash_group"] if kind == "phash_group" else r["split_group"]
            if r["split"] in SPLITS:
                splits_of[gid][r["split"]].append(n)
        for gid, where in splits_of.items():
            if len(where) > 1:
                parts = ", ".join(
                    f"{s} (rows {' '.join(map(str, where[s]))})" for s in SPLITS if s in where
                )
                bad("group_straddles_split", f"{kind} {gid} has rows in {parts}")

    for n, r in ok_rows:
        if r["class_km2"] in UNKNOWN and r["split"] in ("train", "val"):
            bad("unknown_in_train", f"row {n}: {r['class_km2']} in {r['split']}")
        if r["class_km2"] == UNLABELLED and r["split"] in ("train", "val"):
            bad("unlabelled_in_train", f"row {n}: class none in {r['split']}")

    within = [
        r
        for _, r in ok_rows
        if r["class_km2"] in TRAINED and str(r["split_rule"]).startswith(("blocked:", "unblocked:"))
    ]
    for c in TRAINED:
        present = {r["split"] for r in within if r["class_km2"] == c}
        if present and not set(SPLITS) <= present:
            lacking = ", ".join(s for s in SPLITS if s not in present)
            bad("class_missing_from_split", f"{c} has no rows in {lacking}")

    if not meta.get("allow_unknown_licence"):
        unknown = [str(n) for n, r in ok_rows if r["licence"] == "unknown"]
        if unknown:
            bad("licence_unknown", f"rows {' '.join(unknown[:20])} (built without the allowance)")
    return problems


def _bad_values(r: dict[str, Any]) -> list[str]:
    wrong = []
    if not str(r["manifest_version"]).isdigit():
        wrong.append(f"manifest_version {r['manifest_version']!r}")
    if not isinstance(r["sha256"], str) or not HEX64.fullmatch(r["sha256"]):
        wrong.append(f"sha256 {r['sha256']!r}")
    elif r["image_id"] != f"{r['dataset']}_{r['sha256'][:16]}":
        wrong.append(f"image_id {r['image_id']!r} is not <dataset>_<sha256[:16]>")
    if not isinstance(r["phash"], str) or not HEX16.fullmatch(r["phash"]):
        wrong.append(f"phash {r['phash']!r}")
    if r["class_km2"] not in KM2:
        wrong.append(f"class_km2 {r['class_km2']!r}")
    if r["split"] not in (*SPLITS, HOLDOUT):
        wrong.append(f"split {r['split']!r}")
    if r["licence"] not in LICENCES:
        wrong.append(f"licence {r['licence']!r}")
    if r["format"] not in FORMATS.values():
        wrong.append(f"format {r['format']!r}")
    for dim in ("width", "height"):
        if not isinstance(r[dim], int) or r[dim] < 1:
            wrong.append(f"{dim} {r[dim]!r}")
    if not isinstance(r["path"], str) or "\\" in r["path"] or r["path"].startswith("/"):
        wrong.append(f"path {r['path']!r}")
    if not isinstance(r["dup_paths"], list):
        wrong.append("dup_paths is not a list")
    if r["boxes"] is not None and not isinstance(r["boxes"], list):
        wrong.append("boxes is neither null nor a list")
    if r["class_km2"] in UNKNOWN and r["split"] == "test":
        wrong.append(f"{r['class_km2']} in test, not {HOLDOUT}")
    return wrong


def validate(
    manifest: Path,
    *,
    raw_root: Path | None = None,
    frozen_list: Path | None = None,
    log: Log = _print,
) -> int:
    """FR-009; the files are looked up under raw_root, by default the folder the
    sidecar names as `root` (crops) or data/raw."""
    manifest = Path(manifest)
    data = manifest.read_bytes()
    digest = sha256_bytes(data)
    frozen = read_frozen(Path(frozen_list) if frozen_list else manifest.parent / FROZEN_LIST)
    record = frozen.get(manifest.name)
    freeze_broken = record is not None and record["sha256"] != digest
    if freeze_broken:
        log(
            f"frozen_manifest_modified: {manifest.name} was frozen with sha256 "
            f"{record['sha256']} on {record['frozen_at']}; it is now {digest}"
        )
    try:
        rows = _parse_rows(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log(f"bad_value: not JSON Lines in UTF-8 ({exc})")
        return 3 if freeze_broken else 2
    side = sidecar_path(manifest)
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    problems = check_rows(rows, meta, Path(raw_root) if raw_root else file_root(manifest))
    if meta and meta.get("kind") == "crops":
        problems += check_crops(rows, meta, manifest)
    for reason, message in problems:
        log(f"{reason}: {message}")
    if freeze_broken:
        return 3
    if problems:
        return 2
    log(f"ok: {manifest.name}, {len(rows)} rows, sha256 {digest}")
    return 0


# --- freeze -------------------------------------------------------------------------------------


def freeze(manifest: Path, *, log: Log = _print) -> int:
    """US-5.1: append {manifest, sha256, frozen_at, git_sha} to FROZEN.jsonl beside the file."""
    manifest = Path(manifest)
    data = manifest.read_bytes()
    digest = sha256_bytes(data)
    rows = _parse_rows(data.decode("utf-8"))
    if any(r.get("licence") == "unknown" for r in rows):
        log(f"licence_unknown: {manifest.name} has rows without a licence; it cannot be frozen")
        return 2
    side = sidecar_path(manifest)
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    problems = check_rows(rows, meta, raw_root=None)
    if problems:
        for reason, message in problems:
            log(f"{reason}: {message}")
        return 2
    frozen_list = manifest.parent / FROZEN_LIST
    record = read_frozen(frozen_list).get(manifest.name)
    if record is not None:
        if record["sha256"] != digest:
            log(
                f"frozen_manifest_modified: {manifest.name} was frozen with sha256 "
                f"{record['sha256']}; it is now {digest}"
            )
            return 2
        log(f"already frozen on {record['frozen_at']}: {manifest.name} sha256 {digest}")
        return 0
    record = {
        "manifest": manifest.name,
        "sha256": digest,
        "frozen_at": _now(),
        "git_sha": git_sha(),
    }
    with frozen_list.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"frozen: {repo_relative(frozen_list)} += {manifest.name}")
    log(
        f"for DECISIONS.md: `{manifest.name}` frozen {record['frozen_at'][:10]}, "
        f"sha256 `{digest}`, {len(rows)} rows (git {record['git_sha']})"
    )
    return 0


# --- the loader (FR-011) ------------------------------------------------------------------------


@dataclass
class Manifest:
    path: Path
    rows: list[dict[str, Any]]
    sha256: str
    frozen: bool
    meta: dict[str, Any] | None

    @property
    def name(self) -> str:
        """<manifest>_v<N>, the file name without .jsonl."""
        return self.path.stem


def load_manifest(path: Path | str, split: str | Iterable[str] | None, purpose: str) -> Manifest:
    """The rows of `split` (a split, several, or None for every split) for `purpose`.

    test and holdout_unknown are readable only for evaluate, serve and extract
    (TestSplitAccessError otherwise); a frozen manifest whose bytes changed raises
    FrozenManifestModified.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    wanted = None if split is None else {split} if isinstance(split, str) else set(split)
    if purpose not in RESTRICTED_READERS and (wanted is None or wanted & set(RESTRICTED_SPLITS)):
        raise TestSplitAccessError(
            f"purpose {purpose!r} may not read {sorted(wanted) if wanted else 'every split'}: "
            f"test and {HOLDOUT} are for {', '.join(RESTRICTED_READERS)} only"
        )
    path = Path(path)
    data = path.read_bytes()
    digest = sha256_bytes(data)
    record = read_frozen(path.parent / FROZEN_LIST).get(path.name)
    if record is not None and record["sha256"] != digest:
        raise FrozenManifestModified(
            f"{path.name} was frozen with sha256 {record['sha256']}; it is now {digest}"
        )
    rows = [r for r in _parse_rows(data.decode("utf-8")) if wanted is None or r["split"] in wanted]
    side = sidecar_path(path)
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    return Manifest(path, rows, digest, record is not None, meta)


# --- overlap ------------------------------------------------------------------------------------


def overlap(a: Path, b: Path, *, log: Log = _print) -> int:
    """US-2.6: images shared by two manifests, exact (sha256) and near (pHash); nothing merged."""
    rows_a = _parse_rows(Path(a).read_text(encoding="utf-8"))
    rows_b = _parse_rows(Path(b).read_text(encoding="utf-8"))
    side = sidecar_path(Path(a))
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    threshold = int(meta.get("phash_threshold", 6))
    exact = len({r["sha256"] for r in rows_a} & {r["sha256"] for r in rows_b})
    pairs = phash_pairs([r["phash"] for r in rows_a], [r["phash"] for r in rows_b], threshold)
    near = sum(1 for i, j in pairs if rows_a[i]["sha256"] != rows_b[j]["sha256"])
    report = {
        "a": Path(a).name,
        "b": Path(b).name,
        "exact": exact,
        "near": near,
        "phash_threshold": threshold,
    }
    log(json.dumps(report))
    return 0


# --- crops (US-6, FR-012) -----------------------------------------------------------------------

#: a crops row: FR-003's fields, then its box: the parent, the box's index there and the
#: rectangle cut (the box and its margin, in the parent's stored pixels)
CROP_FIELDS = (*FIELDS, "parent_image_id", "box_index", "crop")
CROP_ENCODING = "JPEG, quality 95, no chroma subsampling"
CROP_FRAME = (
    "per box: on an EXIF-turned image, the one frame whose bounds hold the box (the stored "
    "pixels or the upright image; neither decided: box_frame_unknown); cut there, then upright"
)
#: EXIF orientation -> the transposition that shows stored pixels upright (ImageOps.exif_transpose)
UPRIGHT = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def file_root(manifest: Path) -> Path:
    """The folder a manifest's paths are relative to: the `root` its sidecar names (crops:
    data/derived), else data/raw."""
    side = sidecar_path(Path(manifest))
    if side.exists():
        root = json.loads(side.read_text(encoding="utf-8")).get("root")
        if root:
            return Path(root) if Path(root).is_absolute() else REPO_ROOT / root
    return RAW_ROOT


def grown(box: dict[str, Any], margin: Fraction) -> tuple[int, int, int, int]:
    """The box grown by margin x its side on each side, outward to whole pixels:
    (x0, y0, x1, y1). The margin is exact (a Fraction): 10 - 0.1 * 30 is 7, not 6.999..."""
    x, y, w, h = (Fraction(box[k]) for k in ("x", "y", "w", "h"))
    return (
        math.floor(x - margin * w),
        math.floor(y - margin * h),
        math.ceil(x + w + margin * w),
        math.ceil(y + h + margin * h),
    )


def crop_rect(
    box: dict[str, Any], margin: Fraction, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """grown(), clipped to width x height; None when nothing of it lies inside."""
    x0, y0, x1, y1 = grown(box, margin)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _fits(box: dict[str, Any], width: int, height: int) -> bool:
    return (
        box["x"] >= 0
        and box["y"] >= 0
        and box["x"] + box["w"] <= width
        and box["y"] + box["h"] <= height
    )


def box_frame(
    box: dict[str, Any], stored: tuple[int, int], upright: tuple[int, int], turned: bool
) -> str | None:
    """US-6.4: the frame a box was drawn in. An image that is not turned has one frame,
    "stored". On an EXIF-turned image, the one frame whose bounds hold the box, "stored" or
    "upright"; None when both hold it or neither does: the Makerere annotation tool drew in
    either frame, image by image and even box by box (W2 scan, DECISIONS 44)."""
    if not turned:
        return "stored"
    in_stored, in_upright = _fits(box, *stored), _fits(box, *upright)
    if in_stored == in_upright:
        return None
    return "stored" if in_stored else "upright"


def _cut(raw_root: Path, parent: dict[str, Any], margin: Fraction) -> tuple[list, list, int]:
    """The crops of one parent row: ([(box_index, rect, frame, jpeg bytes, (w, h), phash)],
    [(path#box<k>, reason)], clipped boxes). Each box is cut in its frame (box_frame), and
    a crop cut from the stored pixels is then turned upright, as whole frames are."""
    with Image.open(raw_root / parent["path"]) as im:
        orientation = im.getexif().get(ExifTags.Base.Orientation, 1) or 1
        stored = im.convert("RGB")
    turn = UPRIGHT.get(orientation)
    upright = stored.transpose(turn) if turn else stored
    crops, skipped, clipped = [], [], 0
    for k, box in enumerate(parent["boxes"] or []):
        frame = box_frame(box, stored.size, upright.size, turn is not None)
        if frame is None:
            skipped.append((f"{parent['path']}#box{k}", "box_frame_unknown"))
            continue
        source = stored if frame == "stored" else upright
        rect = crop_rect(box, margin, *source.size)
        if rect is None:
            skipped.append((f"{parent['path']}#box{k}", "box_outside_image"))
            continue
        clipped += rect != grown(box, margin)
        crop = source.crop(rect)
        if frame == "stored" and turn:
            crop = crop.transpose(turn)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=95, subsampling=0)
        crops.append((k, rect, frame, buf.getvalue(), crop.size, phash(crop)))
    return crops, skipped, clipped


def build_crops(
    parent_file: Path,
    *,
    margin: float = 0.10,
    raw_root: Path | None = None,
    out: Path | None = None,
    crop_root: Path = DERIVED_ROOT,
    config_dir: Path = CONFIG_DIR,
    log: Log = _print,
) -> int:
    """US-6: <parent>_crops_v<N>.jsonl beside the parent (or in `out`), and the crops in
    crop_root/<parent>_crops/. Exit 0 written or unchanged, 2 error, 3 frozen."""
    try:
        return _build_crops(
            Path(parent_file),
            margin=margin,
            raw_root=Path(raw_root) if raw_root else None,
            out=Path(out) if out else None,
            crop_root=Path(crop_root),
            config_dir=Path(config_dir),
            log=log,
        )
    except ManifestError as exc:
        log(str(exc))
        return 2


def _build_crops(
    parent_file: Path,
    *,
    margin: float,
    raw_root: Path | None,
    out: Path | None,
    crop_root: Path,
    config_dir: Path,
    log: Log,
) -> int:
    parent = load_manifest(parent_file, None, "extract")
    meta_p = parent.meta or {}
    if not parent.rows:
        raise ManifestError("bad_value", f"{parent_file.name} has no rows")
    dataset = parent.rows[0]["dataset"]
    crops_id = f"{dataset}_crops"
    version = int(meta_p.get("version", 1))
    raw_root = raw_root or file_root(parent_file)
    out = out or parent_file.parent
    class_map_file = config_dir / "class_map_v1.yaml"
    class_map = _yaml(class_map_file)
    table: dict[str, str] = class_map["map"].get(dataset, {})
    exact = Fraction(str(margin))
    boxed = [r for r in parent.rows if r["boxes"]]

    staging = crop_root / f"{crops_id}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    made: list[dict[str, Any]] = []
    excluded: list[tuple[str, str]] = []
    unmapped: set[str] = set()
    clipped = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for p, (cut, skipped, n_clipped) in zip(
            boxed, pool.map(lambda r: _cut(raw_root, r, exact), boxed), strict=True
        ):
            excluded += skipped
            clipped += n_clipped
            for k, (x0, y0, x1, y1), frame, data, (w, h), code in cut:
                label = p["boxes"][k]["class_raw"]
                km2 = table.get(label)
                if km2 is None:
                    unmapped.add(label)
                    continue
                if km2 == "excluded":
                    excluded.append((f"{p['path']}#box{k}", "excluded_by_class_map"))
                    continue
                name = f"{p['image_id']}_b{k}.jpg"
                (staging / name).write_bytes(data)
                held = km2 in UNKNOWN
                made.append(
                    {
                        "manifest_version": str(version),
                        "image_id": None,
                        "dataset": crops_id,
                        "source_record": p["source_record"],
                        "source_version": p["source_version"],
                        "path": f"{crops_id}/{name}",
                        "dup_paths": [],
                        "sha256": sha256_bytes(data),
                        "phash": code,
                        "width": w,
                        "height": h,
                        "format": "jpeg",
                        "class_raw": label,
                        "class_km2": km2,
                        "boxes": None,
                        "group_keys": dict(p["group_keys"]),
                        "split_group": p["split_group"],
                        "split": HOLDOUT if held else p["split"],
                        "split_rule": HOLDOUT if held else p["split_rule"],
                        "licence": p["licence"],
                        "attribution": p["attribution"],
                        "notes": None,
                        "parent_image_id": p["image_id"],
                        "box_index": k,
                        "crop": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "frame": frame},
                    }
                )
    if unmapped:
        shutil.rmtree(staging, ignore_errors=True)
        raise ManifestError(
            "unmapped_class",
            f"box labels {', '.join(map(repr, sorted(unmapped)))} not in "
            f"{repo_relative(class_map_file)} map.{dataset}",
        )
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(made, key=lambda r: r["path"]):  # identical crops: one row (FR-006)
        by_sha[r["sha256"]].append(r)
    rows = []
    for digest, same in by_sha.items():
        row = same[0]
        row["image_id"] = f"{crops_id}_{digest[:16]}"
        row["dup_paths"] = [r["path"] for r in same[1:]]
        rows.append({k: row[k] for k in CROP_FIELDS})
    rows.sort(key=lambda r: r["image_id"])
    text = _jsonl(rows)
    digest = sha256_bytes(text.encode("utf-8"))

    target = out / f"{crops_id}_v{version}.jsonl"
    frozen = read_frozen(out / FROZEN_LIST).get(target.name)
    if frozen is not None and frozen["sha256"] != digest:
        shutil.rmtree(staging, ignore_errors=True)
        log(
            f"frozen_manifest_modified: {target.name} is frozen (sha256 {frozen['sha256']}); "
            "these crops would change it"
        )
        return 3
    # publish: identical crops stay untouched, changed ones are replaced, stale ones go
    final = crop_root / crops_id
    final.mkdir(parents=True, exist_ok=True)
    fresh = {f.name for f in staging.iterdir()}
    for f in staging.iterdir():
        dst = final / f.name
        if dst.exists() and dst.read_bytes() == f.read_bytes():
            f.unlink()
        else:
            os.replace(f, dst)
    for f in final.iterdir():
        if f.name not in fresh:
            f.unlink()
    staging.rmdir()

    by_split: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_split[r["split"]][r["class_km2"]] += 1
    meta = {
        "meta_json_version": 1,
        "manifest": crops_id,
        "version": version,
        "kind": "crops",
        "role": meta_p.get("role"),
        "rule": meta_p.get("rule"),
        "parent": {"manifest": parent_file.name, "sha256": parent.sha256, "frozen": parent.frozen},
        "margin": margin,
        "encoding": CROP_ENCODING,
        "frame": CROP_FRAME,
        "root": repo_relative(crop_root),
        "class_map": {
            "path": repo_relative(class_map_file),
            "version": class_map["version"],
            "sha256": sha256_file(class_map_file),
        },
        "boxes": {
            "parents": len(boxed),
            "boxes": sum(len(r["boxes"]) for r in boxed),
            "clipped": clipped,
        },
        "counts": {
            "rows": len(rows),
            "class_raw": _counts(r["class_raw"] for r in rows),
            "class_km2": _counts(r["class_km2"] for r in rows),
            "split": _counts(r["split"] for r in rows),
            "split_rule": _counts(r["split_rule"] for r in rows),
            "split_x_class_km2": {s: dict(sorted(c.items())) for s, c in sorted(by_split.items())},
        },
        "excluded": [{"path": p, "reason": r} for p, r in sorted(excluded)],
        "warnings": [],
        "licence_source": f"{parent_file.name}, row by row",
        "allow_unknown_licence": bool(meta_p.get("allow_unknown_licence", False)),
        "manifest_sha256": digest,
        "built_at": _now(),
        "builder": {"module": "ms.data.manifests", "version": ms.__version__, "git_sha": git_sha()},
    }
    written = _write_if_changed(target, text.encode("utf-8"))
    _write_sidecar(sidecar_path(target), meta)
    log(
        f"{repo_relative(target)}: {len(rows)} crops ({'written' if written else 'unchanged'}) "
        f"of {meta['boxes']['boxes']} boxes on {len(boxed)} images, margin {margin}; class_km2 "
        f"{meta['counts']['class_km2']}; split {meta['counts']['split']}; clipped {clipped}; "
        f"excluded {len(excluded)}; files in {repo_relative(final)}; sha256 {digest}"
    )
    return 0


def check_crops(
    rows: list[dict[str, Any]], meta: dict[str, Any], manifest: Path
) -> list[tuple[str, str]]:
    """US-6.2: every parent_image_id is in the parent manifest (beside this one), and the
    crops of one parent do not straddle splits."""
    parent_file = Path(manifest).parent / meta["parent"]["manifest"]
    if not parent_file.exists():
        return [("parent_missing", f"{parent_file.name} is not beside {Path(manifest).name}")]
    ids = {r["image_id"] for r in _parse_rows(parent_file.read_text(encoding="utf-8"))}
    problems = []
    missing = [str(n) for n, r in enumerate(rows, 1) if r.get("parent_image_id") not in ids]
    if missing:
        problems.append(
            ("parent_missing", f"rows {' '.join(missing[:20])}: not in {parent_file.name}")
        )
    splits_of: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for n, r in enumerate(rows, 1):
        if r.get("split") in SPLITS:
            splits_of[r.get("parent_image_id")][r["split"]].append(n)
    for pid, where in splits_of.items():
        if len(where) > 1:
            parts = ", ".join(f"{s} (rows {' '.join(map(str, where[s]))})" for s in where)
            problems.append(("group_straddles_split", f"the crops of {pid} lie in {parts}"))
    return problems


# --- CLI ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ms.data.manifests", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build data/manifests/<dataset>_v<N>.jsonl + sidecar")
    b.add_argument("--dataset", required=True, help="manifest id (configs/manifests/<id>.yaml)")
    b.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    b.add_argument("--out", type=Path, default=MANIFEST_ROOT)
    b.add_argument("--version", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--rule", default=None)
    b.add_argument("--phash-threshold", type=int, default=None)
    b.add_argument("--allow-unknown-licence", action="store_true")

    v = sub.add_parser("validate", help="check a manifest (FR-009)")
    v.add_argument("manifest", type=Path)
    v.add_argument("--raw-root", type=Path, default=None, help="default: the sidecar's root")
    v.add_argument("--frozen-list", type=Path, default=None)

    f = sub.add_parser("freeze", help="append the manifest's sha256 to FROZEN.jsonl")
    f.add_argument("manifest", type=Path)

    o = sub.add_parser("overlap", help="images shared by two manifests (exact, near)")
    o.add_argument("a", type=Path)
    o.add_argument("b", type=Path)

    c = sub.add_parser("crops", help="cut the boxes of a manifest into <manifest>_crops_v<N>")
    c.add_argument("manifest", type=Path)
    c.add_argument("--margin", type=float, default=0.10, help="share of the box side")
    c.add_argument("--raw-root", type=Path, default=None, help="default: the parent's root")
    c.add_argument("--out", type=Path, default=None, help="default: beside the parent")
    c.add_argument("--crop-root", type=Path, default=DERIVED_ROOT)

    args = p.parse_args(argv)
    if args.command == "build":
        return build(
            args.dataset,
            raw_root=args.raw_root,
            out=args.out,
            version=args.version,
            seed=args.seed,
            rule=args.rule,
            phash_threshold=args.phash_threshold,
            allow_unknown_licence=args.allow_unknown_licence,
        )
    if args.command == "validate":
        return validate(args.manifest, raw_root=args.raw_root, frozen_list=args.frozen_list)
    if args.command == "freeze":
        return freeze(args.manifest)
    if args.command == "crops":
        return build_crops(
            args.manifest,
            margin=args.margin,
            raw_root=args.raw_root,
            out=args.out,
            crop_root=args.crop_root,
        )
    return overlap(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
