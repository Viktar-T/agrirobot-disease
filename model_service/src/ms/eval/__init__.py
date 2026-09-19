"""The N-table (S4.5; specs/005-results-table/spec.md): model_service/results/n_table.jsonl +
n_table.md.

One JSON line per number, written by code (`make eval`, `make probe`), never by hand, and
checked before it is appended. Every row carries the split rule and the hashes of both
manifests (D-9). The five seeds of a number give one aggregate row (seed null): their mean
and a 95 % Student t interval. n_table.md is rendered from the JSON lines and never edited.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ms.data.manifests import HOLDOUT, RULE, UNKNOWN

#: repository root: model_service/src/ms/eval/__init__.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS = REPO_ROOT / "model_service" / "results"
N_TABLE = RESULTS / "n_table.jsonl"
N_TABLE_MD = RESULTS / "n_table.md"

#: the row, in the order it is written (spec 005 FR-001): the work plan's S4.5 fields and the
#: two manifest hashes, then what makes the row traceable (run, model_version, cache key)
FIELDS = (
    "ts",
    "number",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "backbone_id",
    "res",
    "token_type",
    "head",
    "seed",
    "train_manifest",
    "train_manifest_sha256",
    "train_split",
    "test_manifest",
    "test_manifest_sha256",
    "test_split",
    "split_rule",
    "coverage",
    "classes",
    "run_id",
    "model_version",
    "cache_key",
    "quotable",
    "notes",
)
#: what makes two rows the same number (FR-004)
IDENTITY = ("run_id", "number", "metric", "test_manifest_sha256", "test_split", "coverage")
NUMBERS = {
    "N1": "within-dataset",
    "N2": "cross-dataset",
    "N3": "unknown recall / AUROC",
    "N4": "site-prediction probe",
    "N5": "negative control",
    "N6": "latency",
    "N7": "compute cost",
}
#: metric names (FR-005) and their qualifier: None (no qualifier), "class" (one of the row's
#: classes) or "optional" (a target, or nothing)
METRICS = {"macro_f1": None, "recall": "class", "balanced_accuracy": "optional"}
#: the seeds of a reported number, and the Student t quantile for their 95 % interval
SEEDS = (0, 1, 2, 3, 4)
T_975_4 = 2.776445
#: the fields in which the seeds of one number differ (US-2.1)
PER_SEED = ("ts", "value", "ci_low", "ci_high", "seed", "run_id", "quotable", "notes")

#: the splits a number is scored on (FR-001): a test split, the held-out unknowns, or every row
#: of a test-only target (N2's iBean); cross-validation folds are cv<k>
SCORED = ("test", HOLDOUT, "all")
CV_RULE = re.compile(r"unblocked:[1-9][0-9]*fold_by_phash_group")
CV_SPLIT = re.compile(r"cv[1-9][0-9]*")
HEX64 = re.compile(r"[0-9a-f]{64}")
TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class NTableError(ValueError):
    """A row that fails validate_row; `reason` is the first problem's reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def read_rows(path: Path | str = N_TABLE) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def identity(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in IDENTITY)


# --- validation (US-1, US-4.2, US-4.3) -----------------------------------------------------------


def _number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _integer(x: Any, low: int) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= low


def _text(x: Any) -> bool:
    return isinstance(x, str) and bool(x)


def split_rule_ok(rule: Any) -> bool:
    """FR-003: spec 001's rules and N4's cross-validation rule; several are distinct, sorted
    and joined with +."""
    if not _text(rule):
        return False
    parts = rule.split("+")
    return parts == sorted(set(parts)) and all(
        RULE.fullmatch(p) or CV_RULE.fullmatch(p) for p in parts
    )


def _metric_ok(metric: Any, classes: Any) -> bool:
    if not _text(metric):
        return False
    name, sep, qualifier = metric.partition(":")
    if name not in METRICS:
        return False
    kind = METRICS[name]
    if kind is None:
        return not sep
    if kind == "class":
        return bool(sep) and isinstance(classes, list) and qualifier in classes
    return not sep or _text(qualifier)


def validate_row(row: dict[str, Any], frozen: set[str] | None = None) -> list[str]:
    """The row's problems, each '<reason>: <what>'; empty when the row is valid (spec 005
    FR-001 to FR-005 and US-4). With `frozen` (manifest sha256s), a quotable row's manifests
    must all be in it."""
    missing = [k for k in FIELDS if k not in row]
    if missing:
        return [f"missing_field: {', '.join(missing)}"]
    problems = []
    if list(row) != list(FIELDS):
        problems.append("bad_field_order: the fields are not FIELDS, in order")

    def bad(reason: str, field: str) -> None:
        problems.append(f"{reason}: {field} = {row[field]!r}")

    if not (isinstance(row["ts"], str) and TS.fullmatch(row["ts"])):
        bad("bad_value", "ts")
    if row["number"] not in NUMBERS:
        bad("bad_number", "number")
    if not _metric_ok(row["metric"], row["classes"]):
        bad("bad_metric", "metric")
    if not _number(row["value"]):
        bad("bad_value", "value")
    lo, hi = row["ci_low"], row["ci_high"]
    if not ((lo is None and hi is None) or (_number(lo) and _number(hi) and lo <= hi)):
        problems.append(f"bad_value: ci_low, ci_high = {lo!r}, {hi!r}")
    if not _integer(row["n"], 1):
        bad("bad_value", "n")
    for field in ("backbone_id", "token_type", "head", "train_split", "run_id"):
        if not _text(row[field]):
            bad("bad_value", field)
    if not _integer(row["res"], 1):
        bad("bad_value", "res")
    if not (row["seed"] is None or _integer(row["seed"], 0)):
        bad("bad_value", "seed")
    for side in ("train", "test"):
        paths, shas = row[f"{side}_manifest"], row[f"{side}_manifest_sha256"]
        ok = _text(paths) and isinstance(shas, str)
        if ok:
            parts = shas.split("+")
            ok = len(parts) == len(paths.split("+")) and all(HEX64.fullmatch(p) for p in parts)
        if not ok:
            bad("bad_hash", f"{side}_manifest_sha256")
    split = row["test_split"]
    if not (split in SCORED or (isinstance(split, str) and CV_SPLIT.fullmatch(split))):
        bad("bad_value", "test_split")
    if not split_rule_ok(row["split_rule"]):
        bad("bad_split_rule", "split_rule")
    if not (_number(row["coverage"]) and 0 < row["coverage"] <= 1):
        bad("bad_value", "coverage")
    classes = row["classes"]
    if not (isinstance(classes, list) and classes and all(_text(c) for c in classes)):
        bad("bad_value", "classes")
    for field in ("model_version", "cache_key", "notes"):
        if not (row[field] is None or isinstance(row[field], str)):
            bad("bad_value", field)
    if not isinstance(row["quotable"], bool):
        bad("bad_value", "quotable")
    if row["number"] == "N3" and not (
        split == HOLDOUT and isinstance(classes, list) and set(classes) <= set(UNKNOWN)
    ):
        problems.append(
            f"n3_not_held_out: test_split {split!r}, classes {classes!r}: N3 scores held-out "
            f"classes ({', '.join(UNKNOWN)}) only"
        )
    if frozen is not None and row["quotable"] is True:
        shas = [
            h
            for side in ("train", "test")
            if isinstance(row[f"{side}_manifest_sha256"], str)
            for h in row[f"{side}_manifest_sha256"].split("+")
        ]
        loose = [h[:12] for h in shas if h not in frozen]
        if loose:
            problems.append(f"manifest_not_frozen: {', '.join(loose)} (a quotable row)")
    return problems


def append_rows(
    rows: Iterable[dict[str, Any]], path: Path | str = N_TABLE, frozen: set[str] | None = None
) -> int:
    """Append the rows whose identity is not in the table yet; how many were added. Every row
    is checked first: one invalid row raises NTableError and nothing is appended."""
    rows = list(rows)
    for r in rows:
        problems = validate_row(r, frozen)
        if problems:
            reason, _, what = problems[0].partition(": ")
            raise NTableError(reason, f"{r.get('run_id')}: {what}")
    p = Path(path)
    known = {identity(r) for r in read_rows(p)}
    new = []
    for r in rows:
        if identity(r) not in known:
            known.add(identity(r))
            new.append(r)
    if new:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as fh:
            for r in new:
                fh.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
    return len(new)


# --- five seeds (US-2) ---------------------------------------------------------------------------


def number_key(row: dict[str, Any]) -> str:
    """What the seeds of one number share: every field but PER_SEED."""
    return json.dumps({k: row.get(k) for k in FIELDS if k not in PER_SEED}, sort_keys=True)


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per number whose per-seed rows cover exactly seeds 0-4: seed null, the mean,
    the 95 % Student t interval (4 df), the five run ids joined with + in seed order."""
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for r in rows:
        if r.get("seed") is not None:
            groups.setdefault(number_key(r), {})[r["seed"]] = r
    out = []
    for key in sorted(groups):
        by_seed = groups[key]
        if sorted(by_seed) != list(SEEDS):
            continue
        five = [by_seed[s] for s in SEEDS]
        v = np.array([r["value"] for r in five], dtype=np.float64)
        mean = float(v.mean())
        half = T_975_4 * float(v.std(ddof=1)) / math.sqrt(len(v))
        row = {k: five[0][k] for k in FIELDS}
        row.update(
            ts=max(r["ts"] for r in five),
            value=round(mean, 4),
            ci_low=round(mean - half, 4),
            ci_high=round(mean + half, 4),
            seed=None,
            run_id="+".join(r["run_id"] for r in five),
            quotable=all(r["quotable"] is True for r in five),
            notes=f"mean of seeds {SEEDS[0]}-{SEEDS[-1]}; 95 % Student t interval (4 df)",
        )
        out.append(row)
    return out


def unpaired(rows: Iterable[dict[str, Any]]) -> list[tuple]:
    """(backbone_id, res, token_type, head, the missing number) for every quotable N1
    aggregate without an N2 one, and the reverse (US-4.1)."""
    have: dict[str, set[tuple]] = {"N1": set(), "N2": set()}
    for r in rows:
        if r.get("seed") is None and r.get("quotable") is True and r.get("number") in have:
            have[r["number"]].add((r["backbone_id"], r["res"], r["token_type"], r["head"]))
    return sorted(
        [(*k, "N2") for k in have["N1"] - have["N2"]]
        + [(*k, "N1") for k in have["N2"] - have["N1"]]
    )


# --- the rendered table (US-5) --------------------------------------------------------------------


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _manifests(joined: str, split: str) -> str:
    """'ibean_v1 (train)', or 'a_v1+b_v1 (train)' for a row over several manifests (N4)."""
    return f"{'+'.join(Path(p).stem for p in joined.split('+'))} ({split})"


def _metric(r: dict[str, Any]) -> str:
    """The metric, and for N4 (a probe over k sites) its chance level 1/k."""
    if r["number"] == "N4" and r.get("classes"):
        return f"{r['metric']} (chance {1 / len(r['classes']):.3f})"
    return r["metric"]


def _order(r: dict[str, Any]) -> tuple:
    return (
        r["test_manifest"],
        r["train_manifest"],
        r["backbone_id"],
        r["res"],
        r["token_type"],
        r["head"],
        r["coverage"],
        r["metric"],
        -1 if r["seed"] is None else r["seed"],
    )


def render(rows: list[dict[str, Any]]) -> str:
    """n_table.md: one section per number; every aggregate row (seeds 0–4) and every per-seed
    row that no aggregate covers, the split rule on every line."""
    covered = {number_key(r) for r in rows if r.get("seed") is None}
    shown = [r for r in rows if r.get("seed") is None or number_key(r) not in covered]
    out = [
        "# N-table",
        "",
        "Rendered from `n_table.jsonl` by `make eval` (`ms.eval.run`) and `make probe`",
        "(`ms.eval.probe`); do not edit. Every number carries its split rule (D-9). Seeds `0–4`",
        "mark the mean over five seeds with its 95 % Student t interval; the per-seed rows",
        "behind it are in `n_table.jsonl`. A row marked *not quotable* only exercises the",
        "pipeline (for example the W2 slice, trained on a test-only manifest).",
    ]
    for number, title in NUMBERS.items():
        these = sorted((r for r in shown if r["number"] == number), key=_order)
        if not these:
            continue
        out += [
            "",
            f"## {number} — {title}",
            "",
            "| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI "
            "| coverage | n | split rule | |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in these:
            train = _manifests(r["train_manifest"], r["train_split"])
            test = _manifests(r["test_manifest"], r["test_split"])
            ci = "—"
            if r["ci_low"] is not None:
                ci = f"{_fmt(r['ci_low'])}–{_fmt(r['ci_high'])}"
            seed = f"{SEEDS[0]}–{SEEDS[-1]}" if r["seed"] is None else r["seed"]
            flag = "" if r.get("quotable") else "*not quotable*"
            out.append(
                f"| {r['backbone_id']} | {r['res']} | {r['token_type']} | {r['head']} "
                f"| {seed} "
                f"| {train} → {test} | {_metric(r)} | {_fmt(r['value'])} | {ci} "
                f"| {_fmt(r['coverage'], 2)} | {r['n']} | `{r['split_rule']}` | {flag} |"
            )
    return "\n".join(out) + "\n"


def write_md(path_md: Path | str = N_TABLE_MD, path_jsonl: Path | str = N_TABLE) -> bool:
    """Render the table; write it only when it changed."""
    text = render(list(read_rows(path_jsonl)))
    p = Path(path_md)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return True
