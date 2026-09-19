"""Dataset manifests - spec 001 (model_service/specs/001-manifests/spec.md).

    python -m ms.data.manifests build --dataset ibean     # make manifests DS=ibean
    python -m ms.data.manifests validate data/manifests/ibean_v1.jsonl
    python -m ms.data.manifests freeze data/manifests/ibean_v1.jsonl
    python -m ms.data.manifests overlap data/manifests/a_v1.jsonl data/manifests/b_v1.jsonl

One manifest per (manifest id, version): data/manifests/<manifest>_v<N>.jsonl, one row per
distinct image (FR-003), with its sidecar <manifest>_v<N>.meta.json (FR-005). The recipe is
configs/manifests/<manifest>.yaml and the class map configs/class_map_v1.yaml.

Readers exist for iBean (the W2 vertical slice). Makerere, Tanzania and SWM bring theirs
with their W2 tasks, and blocked splits come with the first of them; until then `build`
stops with `no_reader` for those manifests.

Exit codes: build 0 (written or unchanged), 2 (build error, nothing written), 3 (would change
a frozen manifest); validate 0, 2 (content violation), 3 (freeze violation); freeze 0 or 2;
overlap 0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

import ms

#: repository root: model_service/src/ms/data/manifests.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "model_service" / "configs"
CLASS_MAP = CONFIG_DIR / "class_map_v1.yaml"
RAW_ROOT = REPO_ROOT / "data" / "raw"
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
    source publishes about it."""

    path: str
    class_raw: str
    keys: dict[str, Any] = field(default_factory=dict)
    boxes: list[dict[str, Any]] | None = None


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


#: manifest id -> reader(member folder, member id)
READERS: dict[str, Callable[[Path, str], Scan]] = {"ibean": read_ibean}


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


def assign_splits(groups: dict[str, Counter], seed: int) -> tuple[dict[str, str], dict]:
    """FR-008: whole groups into train/val/test, close to TARGETS for every trained class,
    and every class present in every split.

    Seeded greedy (groups shuffled, then largest first; each goes where it brings the
    per-class shares closest to their targets), then repair moves for any class a split
    lacks. ManifestError `class_missing_from_split` when no move can fill the gap.
    Returns ({group: split}, {split: Counter of rows per class}).
    """
    total: Counter = Counter()
    for counts in groups.values():
        total.update(counts)
    classes = [c for c in TRAINED if total[c]]
    want = {s: {c: TARGETS[s] * total[c] for c in classes} for s in SPLITS}
    have: dict[str, Counter] = {s: Counter() for s in SPLITS}

    def err(s: str, c: str, n: float) -> float:
        return ((n - want[s][c]) / total[c]) ** 2

    def joining(g: str, s: str) -> float:
        return sum(err(s, c, have[s][c] + k) - err(s, c, have[s][c]) for c, k in groups[g].items())

    def leaving(g: str, s: str) -> float:
        return sum(err(s, c, have[s][c] - k) - err(s, c, have[s][c]) for c, k in groups[g].items())

    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sum(groups[g].values()))  # stable: equal sizes keep the draw
    where: dict[str, str] = {}
    for g in order:
        s = min(SPLITS, key=lambda s: joining(g, s))
        where[g] = s
        have[s].update(groups[g])

    while gaps := [(c, s) for c in classes for s in SPLITS if not have[s][c]]:
        c, s = gaps[0]
        best: tuple[float, str] | None = None
        for g in sorted(groups):
            src = where[g]
            if src == s or not groups[g][c]:
                continue
            if any(have[src][k] - n <= 0 for k, n in groups[g].items() if n):
                continue  # the move would open a gap in the split it leaves
            delta = joining(g, s) + leaving(g, src)
            if best is None or delta < best[0]:
                best = (delta, g)
        if best is None:
            n_groups = sum(1 for counts in groups.values() if counts[c])
            raise ManifestError(
                "class_missing_from_split",
                f"{c} has no rows in {s}: its {total[c]} rows lie in {n_groups} group(s), "
                "which cannot reach train, val and test all at once",
            )
        g = best[1]
        have[where[g]].subtract(groups[g])
        where[g] = s
        have[s].update(groups[g])
    return where, have


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
            f"manifest {dataset!r} has no reader yet: iBean is the W2 slice; Makerere, "
            "Tanzania and SWM bring theirs with their W2 tasks",
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
    if rule.startswith("blocked:"):
        key = rule.removeprefix("blocked:")
        if not any(r["group_keys"].get(key) is not None for r in rows):
            raise ManifestError("rule_not_applicable", f"{rule}: no row has group_keys.{key}")
        raise ManifestError(
            "rule_not_implemented",
            f"{rule}: blocked splits arrive with the first reader that publishes {key}",
        )
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in trained:
        groups[r["split_group"]][r["class_km2"]] += 1
    where, _ = assign_splits(groups, seed)
    for r in trained:
        r["split"], r["split_rule"] = where[r["split_group"]], UNBLOCKED
    return None


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
    raw_root: Path | None = RAW_ROOT,
    frozen_list: Path | None = None,
    log: Log = _print,
) -> int:
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
    problems = check_rows(rows, meta, Path(raw_root) if raw_root is not None else None)
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
    v.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    v.add_argument("--frozen-list", type=Path, default=None)

    f = sub.add_parser("freeze", help="append the manifest's sha256 to FROZEN.jsonl")
    f.add_argument("manifest", type=Path)

    o = sub.add_parser("overlap", help="images shared by two manifests (exact, near)")
    o.add_argument("a", type=Path)
    o.add_argument("b", type=Path)

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
    return overlap(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
