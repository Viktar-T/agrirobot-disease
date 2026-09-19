"""Evaluate trained heads into the N-table (`make eval`; S4.5, specs/005-results-table/spec.md).

    python -m ms.eval.run [--run <run_id> ...] [--heads-root data/heads] [--cache-root data/cache]
        [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
        [--compute-log model_service/results/compute_log.jsonl]

N1 (US-3): for every head run (or the ones named) whose rows are not in the table yet, reads
the test split of the run's training manifest (purpose=evaluate), scores the cached features
with the head and appends one macro_f1 row and one recall:<class> row per class, each with
the split rule and both manifest hashes. A row is quotable when its run is and both manifests
are frozen. Then every number whose seeds 0-4 are all in the table gets its aggregate row
(the mean and a 95 % Student t interval), and n_table.md is rendered. Running it again adds
nothing. No abstention before S4.4 (W4): coverage is 1.0, and one seed has no interval.

Exit codes: 0; 2 input error or an invalid row, nothing appended.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ms import compute_log
from ms.cache import CACHE_ROOT, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import load_manifest, repo_relative
from ms.eval import (
    N_TABLE,
    N_TABLE_MD,
    NTableError,
    aggregate,
    append_rows,
    identity,
    read_rows,
    unpaired,
    write_md,
)
from ms.heads import HEADS_ROOT, Run, features, list_runs, load_run, resolve_path
from ms.heads.train import macro_f1


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def n1_metrics(classes: list[str]) -> list[str]:
    return ["macro_f1", *(f"recall:{c}" for c in classes)]


def n1_rows(run: Run, cache_root: Path) -> list[dict[str, Any]]:
    """N1: the head on the test split of its own training manifest (within-dataset):
    macro-F1 over its classes, and each class's recall."""
    meta = run.meta
    manifest = load_manifest(resolve_path(meta["train"]["manifest"]), "test", "evaluate")
    rows = [r for r in manifest.rows if r["class_km2"] in run.classes]
    bb = meta["backbone"]
    npz = find_cache(
        cache_root,
        bb["backbone_id"],
        bb["res"],
        manifest.name,
        manifest.sha256,
        key=bb["cache_key"],
    )
    cache = load_cache(npz, manifest.sha256)
    where = {str(i): k for k, i in enumerate(cache.image_id)}
    x = features(cache, np.array([where[r["image_id"]] for r in rows]), run.tokens)
    y = np.array([run.classes.index(r["class_km2"]) for r in rows])
    missing = [c for k, c in enumerate(run.classes) if not (y == k).any()]
    if missing:
        raise ValueError(f"{manifest.path.name} has no test rows of {missing}")
    pred = run.predict_proba(x).argmax(axis=1)
    quotable = bool(meta["quotable"]) and bool(meta["train"]["frozen"]) and manifest.frozen
    notes = "one seed; no abstention yet (S4.4, W4)"
    if not meta["quotable"]:
        notes += "; " + (meta.get("notes") or "not quotable")
    elif not quotable:
        notes += "; a manifest is not frozen: not quotable"
    values = [(macro_f1(y, pred, len(run.classes)), len(rows))]
    values += [
        (float((pred[y == k] == k).mean()), int((y == k).sum())) for k in range(len(run.classes))
    ]
    return [
        {
            "ts": _now(),
            "number": "N1",
            "metric": metric,
            "value": round(value, 4),
            "ci_low": None,
            "ci_high": None,
            "n": n,
            "backbone_id": bb["backbone_id"],
            "res": bb["res"],
            "token_type": run.tokens,
            "head": meta["head"],
            "seed": meta["seed"],
            "train_manifest": meta["train"]["manifest"],
            "train_manifest_sha256": meta["train"]["manifest_sha256"],
            "train_split": meta["train"]["split"],
            "test_manifest": repo_relative(manifest.path),
            "test_manifest_sha256": manifest.sha256,
            "test_split": "test",
            "split_rule": "+".join(sorted({r["split_rule"] for r in rows})),
            "coverage": 1.0,
            "classes": run.classes,
            "run_id": run.run_id,
            "model_version": meta["model_version"],
            "cache_key": bb["cache_key"],
            "quotable": quotable,
            "notes": notes,
        }
        for metric, (value, n) in zip(n1_metrics(run.classes), values, strict=True)
    ]


def expected(run: Run) -> set[tuple]:
    """The identities of a run's N1 rows (FR-004), to skip a run already in the table."""
    return {
        identity(
            {
                "run_id": run.run_id,
                "number": "N1",
                "metric": metric,
                "test_manifest_sha256": run.meta["train"]["manifest_sha256"],
                "test_split": "test",
                "coverage": 1.0,
            }
        )
        for metric in n1_metrics(run.classes)
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ms.eval.run", description=__doc__.split("\n")[0])
    p.add_argument(
        "--run", action="append", default=None, help="run_id (repeatable); default: every run"
    )
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--md", type=Path, default=N_TABLE_MD)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    args = p.parse_args(argv)

    started = time.perf_counter()
    folders = list_runs(args.heads_root)
    if args.run:
        by_id = {f.name: f for f in folders}
        missing = [r for r in args.run if r not in by_id]
        if missing:
            print(f"missing_run: {', '.join(missing)} not under {args.heads_root}", flush=True)
            return 2
        folders = [by_id[r] for r in args.run]
    table = list(read_rows(args.n_table))
    known = {identity(r) for r in table}
    new: list[dict[str, Any]] = []
    scored = []
    for folder in folders:
        run = load_run(folder)
        if expected(run) <= known:
            continue
        try:
            rows = n1_rows(run, args.cache_root)
        except (FileNotFoundError, CacheMismatchError, ValueError) as exc:
            print(f"cannot_evaluate: {run.run_id}: {exc}", flush=True)
            return 2
        new += rows
        scored.append(run.run_id)
        flag = "" if rows[0]["quotable"] else " (not quotable)"
        recalls = ", ".join(f"{r['metric'][7:]} {r['value']:.3f}" for r in rows[1:])
        print(
            f"N1 {run.run_id}: macro-F1 {rows[0]['value']:.4f} on {rows[0]['n']} test rows "
            f"(recall {recalls}), {rows[0]['split_rule']}{flag}",
            flush=True,
        )
    aggregates = aggregate(table + new)
    try:
        added = append_rows(new + aggregates, args.n_table)
    except NTableError as exc:
        print(str(exc), flush=True)
        return 2
    rendered = write_md(args.md, args.n_table)
    if added:
        compute_log.log_row(
            step="eval",
            n_images_or_runs=len(scored),
            wallclock_s=round(time.perf_counter() - started, 3),
            device=f"CPU ({platform.processor() or platform.machine()})",
            notes=f"N1: {len(scored)} run(s) scored, {added} row(s) appended with the aggregates",
            path=args.compute_log,
        )
    for missing_pair in unpaired(read_rows(args.n_table)):
        *key, number = missing_pair
        print(f"unpaired: {' '.join(map(str, key))} has no {number} yet (spec 005 US-4.1)")
    print(
        f"{repo_relative(args.n_table)}: {added} row(s) added; "
        f"{repo_relative(args.md)} {'rewritten' if rendered else 'unchanged'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
