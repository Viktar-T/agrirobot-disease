"""The N-table (S4.5; spec 005 comes in W3): model_service/results/n_table.jsonl + n_table.md.

One JSON line per number, written by `python -m ms.eval.run` (make eval), never by hand.
Every row carries the split rule and the hashes of both manifests (D-9); n_table.md is
rendered from the JSON lines and is never edited.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

#: repository root: model_service/src/ms/eval/__init__.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS = REPO_ROOT / "model_service" / "results"
N_TABLE = RESULTS / "n_table.jsonl"
N_TABLE_MD = RESULTS / "n_table.md"

#: the row, in the order it is written: the work plan's S4.5 fields and the two manifest
#: hashes, then what makes the row traceable (run, model_version, cache key)
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
#: what makes two rows the same number
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


def read_rows(path: Path | str = N_TABLE) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def identity(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in IDENTITY)


def append_rows(rows: Iterable[dict[str, Any]], path: Path | str = N_TABLE) -> int:
    """Append rows whose identity is not in the table yet; how many were added."""
    p = Path(path)
    known = {identity(r) for r in read_rows(p)}
    new = [r for r in rows if identity(r) not in known]
    if new:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as fh:
            for r in new:
                fh.write(json.dumps({k: r.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
    return len(new)


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render(rows: list[dict[str, Any]]) -> str:
    """n_table.md: one section per number, one line per row, the split rule on every line."""
    out = [
        "# N-table",
        "",
        "Rendered from `n_table.jsonl` by `python -m ms.eval.run` (`make eval`); do not edit.",
        "Every number carries its split rule (D-9). A row marked *not quotable* only exercises",
        "the pipeline (for example the W2 slice, trained on a test-only manifest).",
    ]
    for number, title in NUMBERS.items():
        these = [r for r in rows if r["number"] == number]
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
            train = f"{Path(r['train_manifest']).stem} ({r['train_split']})"
            test = f"{Path(r['test_manifest']).stem} ({r['test_split']})"
            ci = "—"
            if r["ci_low"] is not None:
                ci = f"{_fmt(r['ci_low'])}–{_fmt(r['ci_high'])}"
            flag = "" if r.get("quotable") else "*not quotable*"
            out.append(
                f"| {r['backbone_id']} | {r['res']} | {r['token_type']} | {r['head']} "
                f"| {r['seed']} "
                f"| {train} → {test} | {r['metric']} | {_fmt(r['value'])} | {ci} "
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
