"""Evaluate trained heads into the N-table (`make eval`; S4.5, specs/005-results-table/spec.md).

    python -m ms.eval.run [--run <run_id> ...] [--heads-root data/heads] [--cache-root data/cache]
        [--manifest-root data/manifests] [--config model_service/configs/eval.yaml]
        [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
        [--compute-log model_service/results/compute_log.jsonl]

For every head run (or the ones named) whose rows are not all in the table yet:

- N1 (US-3), for a run trained on one manifest: the test split of that manifest
  (purpose=evaluate), one macro_f1 row and one recall:<class> row per class.
- N2 (US-7), for every direction of configs/eval.yaml whose `train` manifests are the run's:
  the direction's target (its test split, or every row of a test-only target), deciding among
  the direction's classes only; macro_f1 and one recall per class, or the direction's metrics.

Every row carries the split rule and both manifest hashes, and is quotable when its run is and
every manifest is frozen. Then every number whose seeds 0-4 are all in the table gets its
aggregate row (the mean and a 95 % Student t interval), and n_table.md is rendered. Running it
again adds nothing. No abstention before S4.4 (W4): coverage is 1.0; one seed has no interval.

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
import yaml

from ms import compute_log
from ms.cache import CACHE_ROOT, Cache, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import MANIFEST_ROOT, TRAINED, Manifest, load_manifest, repo_relative
from ms.eval import (
    N_TABLE,
    N_TABLE_MD,
    REPO_ROOT,
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

EVAL_CONFIG = REPO_ROOT / "model_service" / "configs" / "eval.yaml"
N2_SPLITS = ("test", "all")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_directions(path: Path) -> list[dict[str, Any]]:
    """configs/eval.yaml's N2 directions, checked; ValueError on a malformed one."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    directions = cfg.get("n2") or []
    for d in directions:
        ok = (
            isinstance(d.get("train"), list)
            and d["train"]
            and isinstance(d.get("test"), str)
            and d.get("split") in N2_SPLITS
            and isinstance(d.get("classes"), list)
            and len(d["classes"]) >= 2
            and set(d["classes"]) <= set(TRAINED)
            and (d.get("metrics") is None or isinstance(d["metrics"], list))
        )
        if not ok:
            raise ValueError(f"{path}: a malformed N2 direction {d!r}")
    return directions


def decide(proba: np.ndarray, classes: list[str], among: list[str]) -> np.ndarray:
    """The argmax over the columns of `among` only: an index into `among` per row (US-7.3)."""
    missing = [c for c in among if c not in classes]
    if missing:
        raise ValueError(f"the head has no class {missing} (its classes: {classes})")
    return proba[:, [classes.index(c) for c in among]].argmax(axis=1)


def train_stems(run: Run) -> list[str]:
    """The run's training manifests as <manifest>_v<N>."""
    return [Path(p).stem for p in run.meta["train"]["manifest"].split("+")]


class Store:
    """Manifests and caches read once per call (Tanzania's cache is half a gigabyte)."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.manifests: dict[tuple, Manifest] = {}
        self.caches: dict[Path, Cache] = {}

    def manifest(self, path: Path, split: str | None) -> Manifest:
        key = (Path(path).resolve(), split)
        if key not in self.manifests:
            self.manifests[key] = load_manifest(path, split, "evaluate")
        return self.manifests[key]

    def features(self, run: Run, manifest: Manifest, rows: list[dict]) -> np.ndarray:
        bb = run.meta["backbone"]
        npz = find_cache(
            self.cache_root,
            bb["backbone_id"],
            bb["res"],
            manifest.name,
            manifest.sha256,
            key=bb["cache_key"],
        )
        if npz not in self.caches:
            self.caches[npz] = load_cache(npz, manifest.sha256)
        cache = self.caches[npz]
        where = {str(i): k for k, i in enumerate(cache.image_id)}
        return features(cache, np.array([where[r["image_id"]] for r in rows]), run.tokens)


def score(y: np.ndarray, pred: np.ndarray, classes: list[str], metrics: list[str]) -> list:
    """(metric, value, n) for macro_f1 (over `classes`) and recall:<class>."""
    out = []
    for metric in metrics:
        if metric == "macro_f1":
            absent = [c for k, c in enumerate(classes) if not (y == k).any()]
            if absent:
                raise ValueError(f"macro_f1 needs rows of every class; none of {absent}")
            out.append((metric, macro_f1(y, pred, len(classes)), len(y)))
        elif metric.startswith("recall:") and metric[7:] in classes:
            k = classes.index(metric[7:])
            if not (y == k).any():
                raise ValueError(f"{metric} has no rows")
            out.append((metric, float((pred[y == k] == k).mean()), int((y == k).sum())))
        else:
            raise ValueError(f"no such metric here: {metric!r} (classes {classes})")
    return out


def rows_for(
    run: Run,
    number: str,
    target: Manifest,
    split: str,
    classes: list[str],
    metrics: list[str],
    store: Store,
    note: str,
) -> list[dict[str, Any]]:
    """The rows of one run on one target, deciding among `classes` (N1: the head's own)."""
    rows = [r for r in target.rows if r["class_km2"] in classes]
    x = store.features(run, target, rows)
    y = np.array([classes.index(r["class_km2"]) for r in rows])
    pred = decide(run.predict_proba(x), run.classes, classes)
    meta, bb = run.meta, run.meta["backbone"]
    quotable = bool(meta["quotable"]) and bool(meta["train"]["frozen"]) and target.frozen
    notes = f"{note}; one seed; no abstention yet (S4.4, W4)"
    if not meta["quotable"]:
        notes += "; " + (meta.get("notes") or "not quotable")
    elif not quotable:
        notes += "; a manifest is not frozen: not quotable"
    return [
        {
            "ts": _now(),
            "number": number,
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
            "test_manifest": repo_relative(target.path),
            "test_manifest_sha256": target.sha256,
            "test_split": split,
            "split_rule": "+".join(sorted({r["split_rule"] for r in rows})),
            "coverage": 1.0,
            "classes": classes,
            "run_id": run.run_id,
            "model_version": meta["model_version"],
            "cache_key": bb["cache_key"],
            "quotable": quotable,
            "notes": notes,
        }
        for metric, value, n in score(y, pred, classes, metrics)
    ]


def parts(run: Run, directions: list[dict], manifest_root: Path, store: Store) -> list[dict]:
    """What a run is scored on: N1 on its own test split when it trained on one manifest, and
    N2 on the target of every direction whose training manifests are its own."""
    stems = train_stems(run)
    out = []
    if len(stems) == 1:
        target = store.manifest(resolve_path(run.meta["train"]["manifest"]), "test")
        out.append(
            {
                "number": "N1",
                "target": target,
                "split": "test",
                "classes": run.classes,
                "metrics": ["macro_f1", *(f"recall:{c}" for c in run.classes)],
                "note": "within-dataset",
            }
        )
    for d in directions:
        if sorted(d["train"]) != sorted(stems):
            continue
        path = Path(manifest_root) / f"{d['test']}.jsonl"
        target = store.manifest(path, "test" if d["split"] == "test" else None)
        out.append(
            {
                "number": "N2",
                "target": target,
                "split": d["split"],
                "classes": list(d["classes"]),
                "metrics": d.get("metrics") or ["macro_f1", *(f"recall:{c}" for c in d["classes"])],
                "note": f"cross-dataset {'+'.join(stems)} -> {d['test']}; decision among "
                f"{', '.join(d['classes'])} (the classes both sides have)",
            }
        )
    return out


def identities(run: Run, part: dict) -> set[tuple]:
    return {
        identity(
            {
                "run_id": run.run_id,
                "number": part["number"],
                "metric": metric,
                "test_manifest_sha256": part["target"].sha256,
                "test_split": part["split"],
                "coverage": 1.0,
            }
        )
        for metric in part["metrics"]
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ms.eval.run", description=__doc__.split("\n")[0])
    p.add_argument(
        "--run", action="append", default=None, help="run_id (repeatable); default: every run"
    )
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    p.add_argument("--config", type=Path, default=EVAL_CONFIG)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--md", type=Path, default=N_TABLE_MD)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    args = p.parse_args(argv)

    started = time.perf_counter()
    try:
        directions = load_directions(args.config) if args.config.exists() else []
    except ValueError as exc:
        print(f"bad_config: {exc}", flush=True)
        return 2
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
    store = Store(args.cache_root)
    new: list[dict[str, Any]] = []
    scored = set()
    for folder in folders:
        run = load_run(folder)
        try:
            for part in parts(run, directions, args.manifest_root, store):
                if identities(run, part) <= known:
                    continue
                rows = rows_for(
                    run,
                    part["number"],
                    part["target"],
                    part["split"],
                    part["classes"],
                    part["metrics"],
                    store,
                    part["note"],
                )
                new += rows
                scored.add(run.run_id)
                flag = "" if rows[0]["quotable"] else " (not quotable)"
                values = ", ".join(f"{r['metric']} {r['value']:.4f}" for r in rows)
                print(
                    f"{part['number']} {run.run_id} -> {part['target'].name} ({part['split']}): "
                    f"{values} on {rows[0]['n']} rows, {rows[0]['split_rule']}{flag}",
                    flush=True,
                )
        except (FileNotFoundError, CacheMismatchError, ValueError) as exc:
            print(f"cannot_evaluate: {run.run_id}: {exc}", flush=True)
            return 2
    aggregates = aggregate(table + new)
    try:
        added = append_rows(new + aggregates, args.n_table)
    except NTableError as exc:
        print(str(exc), flush=True)
        return 2
    rendered = write_md(args.md, args.n_table)
    if added:
        numbers = "/".join(sorted({r["number"] for r in new})) or "aggregates"
        compute_log.log_row(
            step="eval",
            n_images_or_runs=len(scored),
            wallclock_s=round(time.perf_counter() - started, 3),
            device=f"CPU ({platform.processor() or platform.machine()})",
            notes=f"{numbers}: {len(scored)} run(s) scored, {added} row(s) appended with the "
            "aggregates",
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
