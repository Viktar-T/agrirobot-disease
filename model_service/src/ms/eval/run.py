"""Evaluate trained heads into the N-table (`make eval`; S4.5 slice version, spec 005 in W3).

    python -m ms.eval.run [--run <run_id> ...] [--heads-root data/heads] [--cache-root data/cache]
        [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
        [--compute-log model_service/results/compute_log.jsonl]

For every head run (or the ones named) without its N1 row yet: reads the test split of the
run's training manifest (purpose=evaluate), scores the cached features with the head, and
appends one N1 row, macro-F1 over the head's classes, carrying the split rule and both
manifest hashes. Then renders n_table.md. Running it again adds nothing.

N1 is one seed and has no abstention yet (S4.4, W4): ci_low and ci_high stay null (the
interval is over five seeds, W3), and coverage is 1.0. Exit codes: 0; 2 input error.
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
from ms.eval import N_TABLE, N_TABLE_MD, append_rows, identity, read_rows, write_md
from ms.heads import HEADS_ROOT, Run, features, list_runs, load_run, resolve_path
from ms.heads.train import macro_f1


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def n1_row(run: Run, cache_root: Path) -> dict[str, Any]:
    """N1: the head on the test split of its own training manifest (within-dataset)."""
    meta = run.meta
    manifest = load_manifest(resolve_path(meta["train"]["manifest"]), "test", "evaluate")
    rows = [r for r in manifest.rows if r["class_km2"] in run.classes]
    if not rows:
        raise ValueError(f"{manifest.path.name} has no test rows of {run.classes}")
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
    pred = run.predict_proba(x).argmax(axis=1)
    return {
        "ts": _now(),
        "number": "N1",
        "metric": "macro_f1",
        "value": round(macro_f1(y, pred, len(run.classes)), 4),
        "ci_low": None,
        "ci_high": None,
        "n": len(rows),
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
        "quotable": bool(meta["quotable"]),
        "notes": "one seed; no abstention yet (S4.4, W4)"
        + ("" if meta["quotable"] else "; " + (meta.get("notes") or "not quotable")),
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
    known = {identity(r) for r in read_rows(args.n_table)}
    new = []
    for folder in folders:
        run = load_run(folder)
        probe = {
            "run_id": run.run_id,
            "number": "N1",
            "metric": "macro_f1",
            "test_manifest_sha256": run.meta["train"]["manifest_sha256"],
            "test_split": "test",
            "coverage": 1.0,
        }
        if identity(probe) in known:
            continue
        try:
            row = n1_row(run, args.cache_root)
        except (FileNotFoundError, CacheMismatchError, ValueError) as exc:
            print(f"cannot_evaluate: {run.run_id}: {exc}", flush=True)
            return 2
        new.append(row)
        flag = "" if row["quotable"] else " (not quotable)"
        print(
            f"N1 {row['run_id']}: macro-F1 {row['value']:.4f} on {row['n']} test rows, "
            f"{row['split_rule']}{flag}",
            flush=True,
        )
    added = append_rows(new, args.n_table)
    rendered = write_md(args.md, args.n_table)
    if added:
        compute_log.log_row(
            step="eval",
            n_images_or_runs=added,
            wallclock_s=round(time.perf_counter() - started, 3),
            device=f"CPU ({platform.processor() or platform.machine()})",
            notes=f"N1 rows for {', '.join(r['run_id'] for r in new)}",
            path=args.compute_log,
        )
    print(
        f"{repo_relative(args.n_table)}: {added} row(s) added; "
        f"{repo_relative(args.md)} {'rewritten' if rendered else 'unchanged'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
