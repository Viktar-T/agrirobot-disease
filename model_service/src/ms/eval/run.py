"""Evaluate trained heads into the N-table (`make eval`; S4.5, specs/005-results-table/spec.md).

    python -m ms.eval.run [--run <run_id> ...] [--heads-root data/heads] [--cache-root data/cache]
        [--manifest-root data/manifests] [--config model_service/configs/eval.yaml]
        [--abstain-config model_service/configs/abstain.yaml]
        [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
        [--compute-log model_service/results/compute_log.jsonl]

For every head run (or the ones named) whose rows are not all in the table yet:

- N1 (US-3), for a run trained on one manifest: the test split of that manifest
  (purpose=evaluate), one macro_f1 row and one recall:<class> row per class.
- N2 (US-7), for every direction of configs/eval.yaml whose `train` manifests are the run's:
  the direction's target (its test split, or every row of a test-only target), deciding among
  the direction's classes only; macro_f1 and one recall per class, or the direction's metrics.
- N3 and the operating points (spec 004 US-6 to US-9), for a run that `make abstain` has
  fitted: the unknown recall of each held-out set at each declared coverage and one AUROC per
  score at coverage 1.0; N1 and N2 again at each declared coverage, over the accepted rows,
  with the selective risk and the abstention rate beside them; and the calibration error of
  the temperature-scaled probabilities, in domain and out.

Every row carries the split rule and both manifest hashes, and is quotable when its run is and
every manifest is frozen. Then every number whose seeds 0-4 are all in the table gets its
aggregate row (the mean and a 95 % Student t interval), and n_table.md is rendered. Running it
again adds nothing. A run with no abstain.json keeps its coverage 1.0 rows and nothing else.

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

from ms import abstain, compute_log
from ms.cache import CACHE_ROOT, Cache, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import (
    HOLDOUT,
    MANIFEST_ROOT,
    TRAINED,
    Manifest,
    load_manifest,
    repo_relative,
)
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
from ms.eval.verdict import VerdictError, beside, write_verdict
from ms.heads import HEADS_ROOT, Run, features, list_runs, load_run, resolve_path
from ms.heads.train import macro_f1

EVAL_CONFIG = REPO_ROOT / "model_service" / "configs" / "eval.yaml"
N2_SPLITS = ("test", "all")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def retired(table: list[dict[str, Any]]) -> set[str]:
    """The runs whose every row in the table is marked superseded (spec 005 US-8.5). A
    replaced recipe does not gain new numbers when a later task adds metrics: its rows are
    not quoted, and scoring it again would only fill the table with rows to mark."""
    seen: dict[str, bool] = {}
    for r in table:
        for run_id in str(r.get("run_id") or "").split("+"):
            seen[run_id] = seen.get(run_id, True) and bool(r.get("superseded"))
    return {run_id for run_id, marked in seen.items() if marked}


def order(folders: list[Path]) -> list[Path]:
    """The runs, grouped by the features their fit shares, so that each bank is built once
    (Store.fit). A run with no fit, or an unreadable one, keeps its place at the end."""

    def key(folder: Path) -> tuple:
        try:
            return (0, str(abstain.bank_key(abstain.load_fit(folder, rebuild=False))), folder.name)
        except (abstain.AbstainError, OSError, KeyError):
            return (1, "", folder.name)

    return sorted(folders, key=key)


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


def train_manifests(run: Run) -> list[str]:
    """The run's training manifests, as run.json records their paths."""
    return run.meta["train"]["manifest"].split("+")


def train_stems(run: Run) -> list[str]:
    """The run's training manifests as <manifest>_v<N>."""
    return [Path(p).stem for p in train_manifests(run)]


class Store:
    """Manifests, caches and feature-space scores read or computed once per call (Tanzania's
    cache is half a gigabyte, and its kNN bank is 78k rows)."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.manifests: dict[tuple, Manifest] = {}
        self.caches: dict[Path, Cache] = {}
        self.banks: dict[tuple, tuple] = {}
        self.distances: dict[tuple, dict] = {}

    def fit(self, folder: Path) -> Any:
        """The run's fit, with the bank and the Gaussians of its group: every head and seed
        trained on the same features shares them (spec 004 FR-013).

        One group's bank at a time. Tanzania's is 78k x 1,024 floats and its cache half a
        gigabyte, so holding six of them would cost more memory than this machine has; the
        runs are scored group by group (`order`), so each is built exactly once."""
        fit = abstain.load_fit(folder, self.cache_root, rebuild=False)
        key = abstain.bank_key(fit)
        if key not in self.banks:
            self.banks.clear()
            self.distances.clear()
            self.caches.clear()
            fit.rebuild(self.cache_root)
            self.banks[key] = (fit.bank, fit.gaussians)
        fit.bank, fit.gaussians = self.banks[key]
        return fit

    def scores(self, fit: Any, run: Run, what: tuple, x: np.ndarray) -> dict:
        """The four scores of one run on one set of rows. The two distances depend on the
        bank, not on the head, so they are computed once per (group, set of rows)."""
        key = (abstain.bank_key(fit), what)
        if key not in self.distances:
            self.distances[key] = abstain.distance_scores(fit, x)
        return abstain.scores(fit, run, x, self.distances[key])

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


def score(
    y: np.ndarray,
    pred: np.ndarray,
    classes: list[str],
    metrics: list[str],
    accepted: np.ndarray | None = None,
) -> tuple[list[tuple[str, float, int]], list[str]]:
    """(metric, value, n) per metric, and the reasons for the ones not written.

    `n` counts the **scored** rows, never the accepted ones (spec 004 US-7.1): it is not one
    of spec 005's per-seed fields, so an n that followed the accepted rows would keep the five
    seeds of a number from ever aggregating. With `accepted`, the values are computed over the
    accepted rows alone, and a metric with no row to stand on is skipped with its reason
    instead of failing the run (spec 004 US-7.3). At coverage 1.0 it still fails."""
    keep = np.ones(len(y), dtype=bool) if accepted is None else np.asarray(accepted, dtype=bool)
    out: list[tuple[str, float, int]] = []
    skipped: list[str] = []

    def missing(what: str) -> None:
        if accepted is None:
            raise ValueError(what)
        skipped.append(what)

    for metric in metrics:
        name, _, qualifier = metric.partition(":")
        if name == "macro_f1":
            absent = [c for k, c in enumerate(classes) if not (y[keep] == k).any()]
            if absent:
                missing(f"macro_f1 needs rows of every class; none accepted of {absent}")
                continue
            out.append((metric, macro_f1(y[keep], pred[keep], len(classes)), len(y)))
        elif name == "recall" and qualifier in classes:
            k = classes.index(qualifier)
            if not (y[keep] == k).any():
                missing(f"{metric} has no accepted row")
                continue
            take = keep & (y == k)
            out.append((metric, float((pred[take] == k).mean()), int((y == k).sum())))
        elif name == "selective_risk":
            if not keep.any():
                missing("selective_risk has no accepted row")
                continue
            out.append((metric, float((pred[keep] != y[keep]).mean()), len(y)))
        elif name == "abstention_rate" and not qualifier:
            out.append((metric, float((~keep).mean()), len(y)))
        elif name == "abstention_rate" and qualifier in classes:
            k = classes.index(qualifier)
            if not (y == k).any():
                missing(f"{metric} has no row")
                continue
            out.append((metric, float((~keep[y == k]).mean()), int((y == k).sum())))
        else:
            raise ValueError(f"no such metric here: {metric!r} (classes {classes})")
    return out, skipped


def _quotable(run: Run, targets: list[Manifest]) -> tuple[bool, str]:
    """Whether a row of this run on these targets may be quoted, and what to add to its
    notes when it may not (spec 005 FR-007)."""
    meta = run.meta
    ok = bool(meta["quotable"]) and bool(meta["train"]["frozen"]) and all(t.frozen for t in targets)
    if not meta["quotable"]:
        return ok, "; " + (meta.get("notes") or "not quotable")
    if not ok:
        return ok, "; a manifest is not frozen: not quotable"
    return ok, ""


def _row(
    run: Run,
    number: str,
    metric: str,
    value: float,
    n: int,
    *,
    targets: list[Manifest],
    split: str,
    split_rule: str,
    coverage: float,
    classes: list[str],
    notes: str,
) -> dict[str, Any]:
    """One row of the N-table (spec 005 FR-001), in FIELDS order."""
    meta, bb = run.meta, run.meta["backbone"]
    quotable, why = _quotable(run, targets)
    return {
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
        "test_manifest": "+".join(repo_relative(t.path) for t in targets),
        "test_manifest_sha256": "+".join(t.sha256 for t in targets),
        "test_split": split,
        "split_rule": split_rule,
        "coverage": coverage,
        "classes": classes,
        "run_id": run.run_id,
        "model_version": meta["model_version"],
        "cache_key": bb["cache_key"],
        "quotable": quotable,
        "superseded": None,
        "notes": notes + why,
    }


def metrics_at(part: dict, coverage: float, fit: Any) -> list[str]:
    """What a part is scored on at one coverage (spec 004 US-7.1, US-8.1). Without a fit,
    only the part's own metrics at coverage 1.0, as before S4.4."""
    if fit is None:
        return list(part["metrics"]) if coverage == 1.0 else []
    if coverage == 1.0:
        return [*part["metrics"], "selective_risk"]
    out = [*part["metrics"], "selective_risk", "abstention_rate"]
    if part["number"] == "N1":
        out += [f"abstention_rate:{c}" for c in part["classes"]]
    return out


def rows_for(
    run: Run,
    part: dict,
    store: Store,
    *,
    fit: Any = None,
    coverages: tuple[float, ...] = (),
    log: Any = print,
) -> list[dict[str, Any]]:
    """The rows of one run on one target, deciding among the part's classes (N1: the head's
    own), at coverage 1.0 and at each declared coverage.

    The scores are computed once and read at every coverage: the kNN distance to a 78k-row
    bank is the expensive part of an operating point, and it does not depend on the
    threshold."""
    target, classes = part["target"], part["classes"]
    rows = [r for r in target.rows if r["class_km2"] in classes]
    x = store.features(run, target, rows)
    y = np.array([classes.index(r["class_km2"]) for r in rows])
    pred = decide(run.predict_proba(x), run.classes, classes)
    split_rule = "+".join(sorted({r["split_rule"] for r in rows}))
    what = (target.sha256, part["split"], tuple(classes))
    scores = store.scores(fit, run, what, x) if fit is not None else None
    out: list[dict[str, Any]] = []
    for coverage in (1.0, *coverages):
        metrics = metrics_at(part, coverage, fit)
        if not metrics:
            continue
        accepted = None
        notes = f"{part['note']}; one seed"
        if coverage < 1.0:
            accepted = ~abstain.decisions(fit, scores, coverage)
            notes += (
                f"; {int(accepted.sum())} of {len(y)} rows accepted at a declared coverage "
                f"of {coverage:.2f}"
            )
        elif fit is None:
            notes += "; no abstention: the run has no abstain.json (S4.4)"
        values, skipped = score(y, pred, classes, metrics, accepted)
        for reason in skipped:
            log(f"not_written: {run.run_id} -> {target.name} at coverage {coverage:.2f}: {reason}")
        out += [
            _row(
                run,
                part["number"],
                metric,
                value,
                n,
                targets=[target],
                split=part["split"],
                split_rule=split_rule,
                coverage=coverage,
                classes=classes,
                notes=notes,
            )
            for metric, value, n in values
        ]
    if fit is not None:
        proba = abstain.probabilities(fit, run, x)[:, [run.classes.index(c) for c in classes]]
        proba = proba / proba.sum(axis=1, keepdims=True)
        out.append(
            _row(
                run,
                part["number"],
                "ece",
                abstain.ece(proba, y, int(fit.meta["ece_bins"])),
                len(y),
                targets=[target],
                split=part["split"],
                split_rule=split_rule,
                coverage=1.0,
                classes=classes,
                notes=f"{part['note']}; one seed; {fit.meta['ece_bins']} equal-width bins over "
                f"the largest probability at T = {fit.temperature:.4f}",
            )
        )
    return out


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


def n3_rows(
    run: Run,
    fit: Any,
    sources: list[dict],
    store: Store,
    manifest_root: Path,
    coverages: tuple[float, ...],
    log: Any = print,
) -> list[dict[str, Any]]:
    """N3 (spec 004 US-6): the unknown recall of each held-out set at each declared coverage,
    and one AUROC per score at coverage 1.0, an AUROC needing no threshold.

    The knowns are the run's own in-domain test rows — the rows N1 scores — so that the two
    numbers stand on one population, and an AUROC row names both manifests."""
    knowns = [store.manifest(resolve_path(p), "test") for p in train_manifests(run)]
    known_rows = {m.name: [r for r in m.rows if r["class_km2"] in run.classes] for m in knowns}
    out: list[dict[str, Any]] = []
    known_scores: dict[str, np.ndarray] | None = None
    if all(known_rows.values()):
        parts_ = [store.features(run, m, known_rows[m.name]) for m in knowns]
        what = tuple(m.sha256 for m in knowns) + ("test", "known")
        known_scores = store.scores(fit, run, what, np.concatenate(parts_))
    else:
        log(f"no_known_rows: {run.run_id} has no in-domain test row: no AUROC (spec 004 US-6.5)")
    for source in sources:
        target = store.manifest(Path(manifest_root) / f"{source['manifest']}.jsonl", HOLDOUT)
        rows = [r for r in target.rows if r["class_km2"] in source["classes"]]
        if not rows:
            log(
                f"no_unknown_rows: {target.name} holds no {', '.join(source['classes'])} row "
                "(spec 004 US-6.5)"
            )
            continue
        classes = list(source["classes"])
        what = (target.sha256, HOLDOUT, tuple(classes))
        scores = store.scores(fit, run, what, store.features(run, target, rows))
        rule = "+".join(sorted({r["split_rule"] for r in rows}))
        where = "a dataset this run trained on" if target.name in known_rows else "a dataset new"
        note = f"held-out {'/'.join(classes)} from {target.name}, {where} to this run"
        for coverage in coverages:
            out.append(
                _row(
                    run,
                    "N3",
                    "unknown_recall",
                    float(abstain.decisions(fit, scores, coverage).mean()),
                    len(rows),
                    targets=[target],
                    split=HOLDOUT,
                    split_rule=rule,
                    coverage=coverage,
                    classes=classes,
                    notes=f"{note}; one seed",
                )
            )
        if known_scores is None:
            continue
        both = [*knowns, target]
        known_rule = sorted({r["split_rule"] for m in knowns for r in known_rows[m.name]})
        n_known = sum(len(v) for v in known_rows.values())
        for name in abstain.SCORES:
            if name == "energy" and run.meta["head"] == "mix":
                log(
                    f"constant_energy: {run.run_id} is a mix run, whose logits are normalised "
                    "log-probabilities: its energy is zero on every row, so no auroc:energy "
                    "row (spec 004 US-6.1)"
                )
                continue
            out.append(
                _row(
                    run,
                    "N3",
                    f"auroc:{name}",
                    abstain.auroc(
                        abstain.strangeness(name, known_scores[name]),
                        abstain.strangeness(name, scores[name]),
                    ),
                    n_known + len(rows),
                    targets=both,
                    split=HOLDOUT,
                    split_rule="+".join(sorted({*known_rule, rule})),
                    coverage=1.0,
                    classes=classes,
                    notes=f"{note}; {len(rows)} unknown rows against {n_known} known rows of "
                    f"{'+'.join(m.name for m in knowns)} (test); one seed",
                )
            )
    return out


def n3_identities(
    run: Run,
    fit: Any,
    sources: list[dict],
    store: Store,
    manifest_root: Path,
    coverages: tuple[float, ...],
) -> set[tuple]:
    """Every N3 row `n3_rows` would write, without computing one: the same rules, on the
    manifests alone, so that a second `make eval` does not score the held-out sets again."""
    out: set[tuple] = set()
    knowns = [store.manifest(resolve_path(p), "test") for p in train_manifests(run)]
    has_knowns = all(any(r["class_km2"] in run.classes for r in m.rows) for m in knowns)
    for source in sources:
        target = store.manifest(Path(manifest_root) / f"{source['manifest']}.jsonl", HOLDOUT)
        if not any(r["class_km2"] in source["classes"] for r in target.rows):
            continue
        base = {"run_id": run.run_id, "number": "N3", "test_split": HOLDOUT}
        for coverage in coverages:
            out.add(
                identity(
                    {
                        **base,
                        "metric": "unknown_recall",
                        "test_manifest_sha256": target.sha256,
                        "coverage": coverage,
                    }
                )
            )
        if not has_knowns:
            continue
        joined = "+".join(m.sha256 for m in [*knowns, target])
        for name in abstain.SCORES:
            if name == "energy" and run.meta["head"] == "mix":
                continue
            out.add(
                identity(
                    {
                        **base,
                        "metric": f"auroc:{name}",
                        "test_manifest_sha256": joined,
                        "coverage": 1.0,
                    }
                )
            )
    return out


def identities(run: Run, part: dict, coverages: tuple[float, ...], fit: Any) -> set[tuple]:
    """Every row this part would write, as spec 005 FR-004 identifies it: the part is scored
    again only when one of them is missing."""
    return {
        identity(
            {
                "run_id": run.run_id,
                "number": part["number"],
                "metric": metric,
                "test_manifest_sha256": part["target"].sha256,
                "test_split": part["split"],
                "coverage": coverage,
            }
        )
        for coverage in (1.0, *coverages)
        for metric in [
            *metrics_at(part, coverage, fit),
            *(["ece"] if coverage == 1.0 and fit is not None else []),
        ]
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
    p.add_argument("--abstain-config", type=Path, default=abstain.ABSTAIN_CONFIG)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--md", type=Path, default=N_TABLE_MD)
    p.add_argument(
        "--verdict",
        type=Path,
        default=None,
        help="default: verdict.md beside --n-table, so a verdict never leaves its own table",
    )
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    args = p.parse_args(argv)

    started = time.perf_counter()
    try:
        directions = load_directions(args.config) if args.config.exists() else []
        cfg = abstain.load_abstain_config(args.abstain_config)
    except ValueError as exc:
        print(f"bad_config: {exc}", flush=True)
        return 2
    coverages = tuple(sorted(float(c) for c in cfg["coverages"]))
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
    old = retired(table)
    skipped = [f for f in folders if f.name in old]
    folders = order([f for f in folders if f.name not in old])
    if skipped:
        print(
            f"superseded: {len(skipped)} run(s) are not scored again; every row they have is "
            "marked (spec 005 US-8.5)",
            flush=True,
        )
    new: list[dict[str, Any]] = []
    scored = set()
    unfitted = 0
    for folder in folders:
        run = load_run(folder)
        try:
            fit = store.fit(folder)
        except abstain.AbstainError as exc:
            fit, unfitted = None, unfitted + 1
            if exc.reason != "missing_fit":
                print(f"not_fitted: {run.run_id}: {exc}", flush=True)
        try:
            for part in parts(run, directions, args.manifest_root, store):
                if identities(run, part, coverages, fit) <= known:
                    continue
                rows = rows_for(run, part, store, fit=fit, coverages=coverages)
                new += rows
                scored.add(run.run_id)
                full = [r for r in rows if r["coverage"] == 1.0]
                flag = "" if rows[0]["quotable"] else " (not quotable)"
                values = ", ".join(f"{r['metric']} {r['value']:.4f}" for r in full)
                extra = f" (+{len(rows) - len(full)} rows at the declared coverages)" if fit else ""
                print(
                    f"{part['number']} {run.run_id} -> {part['target'].name} ({part['split']}): "
                    f"{values} on {full[0]['n']} rows, {full[0]['split_rule']}{flag}{extra}",
                    flush=True,
                )
            want = (
                n3_identities(run, fit, cfg["n3"], store, args.manifest_root, coverages)
                if fit is not None and cfg["n3"]
                else set()
            )
            if want and not want <= known:
                rows = n3_rows(run, fit, cfg["n3"], store, args.manifest_root, coverages)
                fresh = [r for r in rows if identity(r) not in known]
                if fresh:
                    new += fresh
                    scored.add(run.run_id)
                    at = cfg["default_coverage"]
                    recall = {
                        r["classes"][0]: r["value"]
                        for r in fresh
                        if r["metric"] == "unknown_recall" and r["coverage"] == at
                    }
                    auroc = {
                        r["classes"][0]: r["value"] for r in fresh if r["metric"] == "auroc:knn"
                    }
                    print(
                        f"N3 {run.run_id}: unknown recall at {at:.2f} "
                        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(recall.items()))
                        + "; auroc:knn "
                        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(auroc.items())),
                        flush=True,
                    )
        except (FileNotFoundError, CacheMismatchError, ValueError) as exc:
            print(f"cannot_evaluate: {run.run_id}: {exc}", flush=True)
            return 2
    if unfitted:
        print(
            f"not_fitted: {unfitted} run(s) have no abstain.json: their rows stay at coverage "
            "1.0 (`make abstain` fits them; spec 004)",
            flush=True,
        )
    # A row whose identity is already in the table is not appended (spec 005 FR-004), so it
    # must not reach `aggregate` either: a re-scored row of a superseded run carries no mark,
    # and would otherwise stand in for the current seed of that number and put its run id in
    # a current aggregate (spec 005 US-8.3; DECISIONS 99).
    new = [r for r in new if identity(r) not in known]
    aggregates = aggregate(table + new)
    try:
        added = append_rows(new + aggregates, args.n_table)
    except NTableError as exc:
        print(str(exc), flush=True)
        return 2
    rendered = write_md(args.md, args.n_table)
    table = list(read_rows(args.n_table))
    missing_pairs = unpaired(table)
    for missing_pair in missing_pairs:
        *key, number = missing_pair
        print(f"unpaired: {' '.join(map(str, key))} has no {number} yet (spec 005 US-4.1)")
    verdict_path = args.verdict or beside(args.n_table)
    if missing_pairs:
        print(
            f"verdict: refused while {len(missing_pairs)} number(s) are unpaired "
            f"(spec 005 US-6.1); {repo_relative(verdict_path)} not written",
            flush=True,
        )
    else:
        try:
            written, v = write_verdict(table, directions, verdict_path)
        except VerdictError as exc:
            print(f"verdict: refused, {exc.reason}: {exc}", flush=True)
        else:
            print(
                f"{repo_relative(verdict_path)}: {v['winner']} wins - {v['why']} "
                f"({'rewritten' if written else 'unchanged'})",
                flush=True,
            )
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
    print(
        f"{repo_relative(args.n_table)}: {added} row(s) added; "
        f"{repo_relative(args.md)} {'rewritten' if rendered else 'unchanged'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
