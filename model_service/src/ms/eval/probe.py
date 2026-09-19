"""The site-prediction probe, N4 (`make probe`; docs/piece4-work-plan.md W2 Fri).

    python -m ms.eval.probe [--backbone <id> ...] [--res 224] [--target dataset|district ...]
        [--datasets ibean makerere tanzania] [--within makerere] [--seed 0] [--folds 5]
        [--manifest-root data/manifests] [--cache-root data/cache]
        [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
        [--compute-log model_service/results/compute_log.jsonl] [--mlflow-uri URI | --no-mlflow]

A warning light, not a disease model: how easily a linear model reads, from the cached CLS
features alone, where a picture was taken. If it reads that easily, a disease head can learn
the place instead of the disease.

- `dataset`: which bean set (iBean, Makerere, Tanzania; chance 1/3). The probe is fitted on
  the train splits and scored on the test splits, so the scored rows come from districts and
  dates it never saw. Only the classes that every set has (healthy, rust) take part.
- `district`: within Makerere, the district (chance 1/12), on the rows that record one:
  rust and angular leaf spot, from every split; healthy rows carry no district. The
  districts are the split's blocks, so the train split holds none of the test's, and the
  probe is scored by 5-fold cross-validation grouped by phash_group instead.

The probe is multinomial logistic regression (lbfgs, C = 1) on the L2-normalised CLS,
standardised; nothing is tuned. Every (target, class) cell weighs the same in the fit and in
the score, so a set's class mix cannot stand in for the set. The number is balanced
accuracy: each target's recall, its classes weighted equally, averaged over the targets. Its
interval is a bootstrap over the scored rows within their cells (1,000 draws); the probe has
no seed to vary apart from the folds.

Each (backbone, target) gives one N-table row, one MLflow run (MLFLOW_TRACKING_URI, else the
local store mlruns/mlflow.db) and one compute-log row. Running it again does nothing.
Exit codes: 0; 2 input error, nothing written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ms
from ms import compute_log
from ms.cache import BACKBONE_CONFIGS, CACHE_ROOT, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import (
    MANIFEST_ROOT,
    TRAINED,
    Manifest,
    ManifestError,
    git_sha,
    load_manifest,
    manifest_path,
    repo_relative,
)
from ms.eval import (
    N_TABLE,
    N_TABLE_MD,
    REPO_ROOT,
    append_rows,
    identity,
    read_rows,
    validate_row,
    write_md,
)
from ms.heads import features

PROBE_VERSION = 1
TARGETS = ("dataset", "district")
DATASETS = ("ibean", "makerere", "tanzania")
WITHIN = "makerere"
TOKENS = "cls"
#: fixed before the first number, not tuned (DECISIONS 54)
CLASSIFIER = {
    "model": "logistic_regression",
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 5000,
    "features": "cls, L2-normalised, standardised",
    "weights": "each (target, class) cell the same",
}
FOLDS = 5
DRAWS = 1000
EXPERIMENT = "piece4-n4-site-probe"
#: the local MLflow store until piece 3 runs the server (git-ignored)
LOCAL_MLFLOW = REPO_ROOT / "mlruns"


class ProbeError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the arithmetic ----------------------------------------------------------------------------


def cell_weights(target: np.ndarray, klass: np.ndarray) -> np.ndarray:
    """Every target weighs the same, and within a target each of its classes: a row weighs
    1 / (rows in its cell x classes of its target), scaled to mean 1."""
    w = np.zeros(len(target))
    for t in np.unique(target):
        in_t = target == t
        present = np.unique(klass[in_t])
        for c in present:
            cell = in_t & (klass == c)
            w[cell] = 1.0 / (cell.sum() * len(present))
    return w * (len(w) / w.sum())


def fit_predict(
    x_fit: np.ndarray, t_fit: np.ndarray, c_fit: np.ndarray, x_score: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    """(the predicted targets of x_score, lbfgs iterations, converged)."""
    w = cell_weights(t_fit, c_fit)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=CLASSIFIER["C"], solver=CLASSIFIER["solver"], max_iter=CLASSIFIER["max_iter"]
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            x_fit, t_fit, standardscaler__sample_weight=w, logisticregression__sample_weight=w
        )
    converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
    return model.predict(x_score), int(np.max(model[-1].n_iter_)), converged


def folds(target: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """The fold of every row: k folds stratified by target, a phash group never split."""
    fold = np.full(len(target), -1)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, idx) in enumerate(splitter.split(np.zeros(len(target)), target, groups)):
        fold[idx] = f
    return fold


def score(
    target: np.ndarray, klass: np.ndarray, pred: np.ndarray, seed: int, draws: int = DRAWS
) -> dict[str, Any]:
    """Balanced accuracy (the mean over targets of their class-averaged recall), its
    bootstrap interval (rows resampled within their cells) and the recall of each cell."""
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for t, c in sorted(set(zip(target.tolist(), klass.tolist(), strict=True))):
        m = (target == t) & (klass == c)
        cells[(t, c)] = (int(m.sum()), int((pred[m] == t).sum()))

    def balanced(recall: dict[tuple[int, int], Any]) -> Any:
        by_target: dict[int, list[Any]] = {}
        for (t, _), r in recall.items():
            by_target.setdefault(t, []).append(r)
        return np.mean([np.mean(rs, axis=0) for rs in by_target.values()], axis=0)

    rng = np.random.default_rng(seed)
    boot = balanced({k: rng.binomial(n, ok / n, size=draws) / n for k, (n, ok) in cells.items()})
    low, high = np.percentile(boot, [2.5, 97.5])
    recall_t: dict[int, list[float]] = {}
    for (t, _), (n, ok) in cells.items():
        recall_t.setdefault(t, []).append(ok / n)
    return {
        "value": float(balanced({k: ok / n for k, (n, ok) in cells.items()})),
        "ci_low": float(low),
        "ci_high": float(high),
        "recall": {t: float(np.mean(rs)) for t, rs in recall_t.items()},
        "cells": cells,
    }


# --- the two probes -----------------------------------------------------------------------------


@dataclass
class Part:
    """One manifest's share of a probe: its rows to fit and to score, and its cache."""

    manifest: Manifest
    npz: Path
    fit_rows: list[dict[str, Any]] = field(default_factory=list)
    score_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Plan:
    """A probe, identified before anything is computed."""

    target: str
    backbone_id: str
    res: int
    cache_key: str
    research_only: bool
    names: list[str]  # the target's values, in label order
    classes: list[str]  # the disease classes that take part
    parts: list[Part]
    seed: int
    folds: int | None
    probe_id: str = ""

    @property
    def metric(self) -> str:
        return f"balanced_accuracy:{self.target}"

    @property
    def manifests(self) -> tuple[str, str]:
        """(paths, sha256s), each '+'-joined in the order of the parts."""
        return (
            "+".join(repo_relative(p.manifest.path) for p in self.parts),
            "+".join(p.manifest.sha256 for p in self.parts),
        )

    @property
    def splits(self) -> tuple[str, str]:
        return (f"cv{self.folds}", f"cv{self.folds}") if self.folds else ("train", "test")

    def stub(self) -> dict[str, Any]:
        """What identifies this probe's N-table row (ms.eval.IDENTITY)."""
        return {
            "run_id": self.probe_id,
            "number": "N4",
            "metric": self.metric,
            "test_manifest_sha256": self.manifests[1],
            "test_split": self.splits[1],
            "coverage": 1.0,
        }


class Manifests:
    """Manifests by id, read once per (split, purpose)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._memo: dict[tuple[str, str | None, str], Manifest] = {}

    def get(self, manifest_id: str, split: str | None, purpose: str) -> Manifest:
        key = (manifest_id, split, purpose)
        if key not in self._memo:
            try:
                path = manifest_path(manifest_id, self.root)
                if not path.exists():
                    raise ProbeError("missing_file", f"no manifest {repo_relative(path)}")
                self._memo[key] = load_manifest(path, split, purpose)
            except ManifestError as exc:
                raise ProbeError(exc.reason, str(exc)) from None
        return self._memo[key]


def _cache(cache_root: Path, backbone_id: str, res: int, m: Manifest) -> Path:
    try:
        return find_cache(cache_root, backbone_id, res, m.name, m.sha256)
    except (FileNotFoundError, CacheMismatchError) as exc:
        raise ProbeError("missing_cache", str(exc)) from None


def _sidecar(npz: Path) -> dict[str, Any]:
    return json.loads(npz.with_name(npz.stem + ".meta.json").read_text(encoding="utf-8"))


def plan_dataset(
    manifests: Manifests, datasets: list[str], backbone_id: str, res: int, cache_root: Path
) -> tuple[list[Part], list[str], list[str]]:
    """The dataset probe's parts: each set's train rows to fit, test rows to score, of the
    classes that every set has in both splits."""
    parts, shared = [], set(TRAINED)
    for d in datasets:
        train = manifests.get(d, "train", "train")
        test = manifests.get(d, "test", "evaluate")
        for side in (train, test):
            shared &= {r["class_km2"] for r in side.rows}
        parts.append(
            Part(train, _cache(cache_root, backbone_id, res, train), train.rows, test.rows)
        )
    classes = [c for c in TRAINED if c in shared]
    if not classes:
        raise ProbeError("bad_value", f"{', '.join(datasets)} share no class in train and test")
    for p in parts:
        p.fit_rows = [r for r in p.fit_rows if r["class_km2"] in classes]
        p.score_rows = [r for r in p.score_rows if r["class_km2"] in classes]
    return parts, list(datasets), classes


def plan_district(
    manifests: Manifests, within: str, backbone_id: str, res: int, cache_root: Path
) -> tuple[list[Part], list[str], list[str]]:
    """The district probe's one part: the rows that record a district, every split."""
    m = manifests.get(within, None, "evaluate")
    rows = [r for r in m.rows if (r.get("group_keys") or {}).get("district")]
    if not rows:
        raise ProbeError("bad_value", f"{m.path.name} has no row with a district")
    names = sorted({r["group_keys"]["district"] for r in rows})
    if len(names) < 2:
        raise ProbeError("bad_value", f"{m.path.name} records one district only: {names}")
    classes = sorted({r["class_km2"] for r in rows})
    return [Part(m, _cache(cache_root, backbone_id, res, m), rows, rows)], names, classes


def make_plan(
    args: argparse.Namespace, manifests: Manifests, backbone_id: str, target: str
) -> Plan:
    if target == "dataset":
        parts, names, classes = plan_dataset(
            manifests, args.datasets, backbone_id, args.res, args.cache_root
        )
        n_folds = None
    else:
        parts, names, classes = plan_district(
            manifests, args.within, backbone_id, args.res, args.cache_root
        )
        n_folds = args.folds
    metas = [_sidecar(p.npz) for p in parts]
    keys = sorted({meta["cache_key"] for meta in metas})
    if len(keys) != 1:
        raise ProbeError(
            "bad_value", f"{backbone_id} @ {args.res}: the caches come from several keys {keys}"
        )
    plan = Plan(
        target=target,
        backbone_id=backbone_id,
        res=args.res,
        cache_key=keys[0],
        research_only=any(bool(meta.get("research_only_until_c5")) for meta in metas),
        names=names,
        classes=classes,
        parts=parts,
        seed=args.seed,
        folds=n_folds,
    )
    inputs = {
        "probe_version": PROBE_VERSION,
        "target": target,
        "backbone_id": backbone_id,
        "res": args.res,
        "cache_key": plan.cache_key,
        "tokens": TOKENS,
        "manifests": [[p.manifest.name, p.manifest.sha256] for p in parts],
        "names": names,
        "classes": classes,
        "classifier": CLASSIFIER,
        "folds": n_folds,
        "seed": args.seed,
        "draws": DRAWS,
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    plan.probe_id = f"n4-{target}-{backbone_id}-{args.res}-{TOKENS}-{digest[:8]}"
    return plan


@dataclass
class Result:
    score: dict[str, Any]
    n_fit: int
    n_scored: int
    n_iter: int
    converged: bool
    split_rule: str
    counts: dict[str, dict[str, int]]  # target name -> class -> scored rows
    confusion: list[list[int]]  # scored rows: true target x predicted target
    wallclock_s: float


def compute(plan: Plan) -> Result:
    started = time.perf_counter()
    x_fit: list[np.ndarray] = []
    x_score: list[np.ndarray] = []
    t_fit, c_fit, t_score, c_score, groups = [], [], [], [], []
    rules: set[str] = set()
    for i, part in enumerate(plan.parts):
        cache = load_cache(part.npz, part.manifest.sha256)
        where = {str(image_id): k for k, image_id in enumerate(cache.image_id)}

        def matrix(rows: list[dict[str, Any]], _where=where, _cache=cache) -> np.ndarray:
            return features(_cache, np.array([_where[r["image_id"]] for r in rows]), TOKENS)

        def label(r: dict[str, Any], _i=i) -> int:
            return plan.names.index(r["group_keys"]["district"]) if plan.folds else _i

        if plan.folds:  # one part, fitted and scored by cross-validation
            x_score.append(matrix(part.score_rows))
            groups += [r["group_keys"]["phash_group"] for r in part.score_rows]
        else:
            x_fit.append(matrix(part.fit_rows))
            t_fit += [label(r) for r in part.fit_rows]
            c_fit += [plan.classes.index(r["class_km2"]) for r in part.fit_rows]
            x_score.append(matrix(part.score_rows))
            rules |= {r["split_rule"] for r in part.score_rows}
        t_score += [label(r) for r in part.score_rows]
        c_score += [plan.classes.index(r["class_km2"]) for r in part.score_rows]
    xs, ts, cs = np.concatenate(x_score), np.array(t_score), np.array(c_score)

    if plan.folds:
        fold = folds(ts, np.array(groups), plan.folds, plan.seed)
        pred = np.full(len(ts), -1)
        n_iter, converged = 0, True
        for f in range(plan.folds):
            out = fold == f
            pred[out], it, ok = fit_predict(xs[~out], ts[~out], cs[~out], xs[out])
            n_iter, converged = max(n_iter, it), converged and ok
        n_fit = len(ts)
        split_rule = f"unblocked:{plan.folds}fold_by_phash_group"
    else:
        pred, n_iter, converged = fit_predict(
            np.concatenate(x_fit), np.array(t_fit), np.array(c_fit), xs
        )
        n_fit = len(t_fit)
        split_rule = "+".join(sorted(rules))

    k = len(plan.names)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (ts, pred), 1)
    counts = {
        name: {c: int(((ts == t) & (cs == j)).sum()) for j, c in enumerate(plan.classes)}
        for t, name in enumerate(plan.names)
    }
    return Result(
        score=score(ts, cs, pred, plan.seed),
        n_fit=n_fit,
        n_scored=len(ts),
        n_iter=n_iter,
        converged=converged,
        split_rule=split_rule,
        counts=counts,
        confusion=confusion.tolist(),
        wallclock_s=round(time.perf_counter() - started, 3),
    )


def n4_row(plan: Plan, result: Result) -> dict[str, Any]:
    paths, shas = plan.manifests
    s = result.score
    chance = 1 / len(plan.names)
    recall = ", ".join(f"{plan.names[t]} {r:.3f}" for t, r in sorted(s["recall"].items()))
    if plan.folds:
        how = (
            f"on the {result.n_scored} rows with a recorded district ({', '.join(plan.classes)}), "
            f"every split, {plan.folds}-fold CV grouped by phash_group (seed {plan.seed})"
        )
    else:
        how = (
            f"fitted on the train splits' {' + '.join(plan.classes)} rows ({result.n_fit}), "
            f"scored on the test splits' ({result.n_scored})"
        )
    notes = (
        f"logistic regression (C = {CLASSIFIER['C']:g}) on standardised L2-normalised CLS, "
        f"{how}; every ({plan.target}, class) cell weighs the same; chance {chance:.3f}; "
        f"recall {recall}; interval: bootstrap over rows within cells ({DRAWS} draws), not seeds"
        + ("" if result.converged else f"; lbfgs stopped at {result.n_iter} iterations")
    )
    train_split, test_split = plan.splits
    return {
        "ts": _now(),
        "number": "N4",
        "metric": plan.metric,
        "value": round(s["value"], 4),
        "ci_low": round(s["ci_low"], 4),
        "ci_high": round(s["ci_high"], 4),
        "n": result.n_scored,
        "backbone_id": plan.backbone_id,
        "res": plan.res,
        "token_type": TOKENS,
        "head": "logreg",
        "seed": plan.seed,
        "train_manifest": paths,
        "train_manifest_sha256": shas,
        "train_split": train_split,
        "test_manifest": paths,
        "test_manifest_sha256": shas,
        "test_split": test_split,
        "split_rule": result.split_rule,
        "coverage": 1.0,
        "classes": plan.names,
        "run_id": plan.probe_id,
        "model_version": None,
        "cache_key": plan.cache_key,
        "quotable": True,
        "notes": notes,
    }


# --- MLflow -------------------------------------------------------------------------------------


class Tracker:
    """One MLflow run per probe, in the experiment piece4-n4-site-probe. The store is
    --mlflow-uri, else MLFLOW_TRACKING_URI, else mlruns/mlflow.db (git-ignored) until piece 3
    runs the server; a local SQLite store keeps its artifacts beside it."""

    def __init__(self, uri: str | None) -> None:
        from mlflow.tracking import MlflowClient

        uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            LOCAL_MLFLOW.mkdir(parents=True, exist_ok=True)
            uri = "sqlite:///" + (LOCAL_MLFLOW / "mlflow.db").as_posix()
        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)
        exp = self.client.get_experiment_by_name(EXPERIMENT)
        if exp is not None:
            self.experiment_id = exp.experiment_id
        else:
            artifacts = None
            if uri.startswith("sqlite:///"):
                artifacts = (Path(uri[len("sqlite:///") :]).parent / "artifacts").as_uri()
            self.experiment_id = self.client.create_experiment(
                EXPERIMENT, artifact_location=artifacts
            )

    def run_of(self, probe_id: str) -> str | None:
        runs = self.client.search_runs(
            [self.experiment_id], filter_string=f"tags.probe_id = '{probe_id}'", max_results=1
        )
        return runs[0].info.run_id if runs else None

    def log(self, plan: Plan, result: Result, row: dict[str, Any]) -> str:
        from mlflow.entities import Metric, Param

        s = result.score
        tags = {
            "probe_id": plan.probe_id,
            "number": "N4",
            "target": plan.target,
            "backbone_id": plan.backbone_id,
            "res": str(plan.res),
            "cache_key": plan.cache_key,
            "research_only_until_c5": str(plan.research_only).lower(),
            "n_table": "model_service/results/n_table.jsonl",
            "git_sha": git_sha() or "",
        }
        params = {
            "tokens": TOKENS,
            **{f"classifier.{k}": v for k, v in CLASSIFIER.items()},
            "targets": ",".join(plan.names),
            "classes": ",".join(plan.classes),
            "manifests": row["test_manifest"],
            "manifest_sha256": row["test_manifest_sha256"],
            "fit": row["train_split"],
            "score": row["test_split"],
            "split_rule": row["split_rule"],
            "folds": plan.folds or 0,
            "seed": plan.seed,
            "bootstrap_draws": DRAWS,
            "probe_version": PROBE_VERSION,
        }
        metrics = {
            "balanced_accuracy": s["value"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "chance": 1 / len(plan.names),
            "n_fit": result.n_fit,
            "n_scored": result.n_scored,
            "lbfgs_iterations": result.n_iter,
            **{f"recall_{plan.names[t]}": r for t, r in s["recall"].items()},
        }
        run = self.client.create_run(self.experiment_id, run_name=plan.probe_id, tags=tags)
        rid = run.info.run_id
        now = int(time.time() * 1000)
        self.client.log_batch(
            rid,
            metrics=[Metric(k, float(v), now, 0) for k, v in metrics.items()],
            params=[Param(k, str(v)) for k, v in params.items()],
        )
        self.client.log_dict(rid, detail(plan, result, row), "probe.json")
        self.client.set_terminated(rid)
        return rid


def detail(plan: Plan, result: Result, row: dict[str, Any]) -> dict[str, Any]:
    """The probe in full: the N-table row, the cells, the confusion matrix."""
    cells = [
        {
            "target": plan.names[t],
            "class": plan.classes[c],
            "n": n,
            "correct": ok,
            "recall": round(ok / n, 6),
        }
        for (t, c), (n, ok) in result.score["cells"].items()
    ]
    return {
        "row": row,
        "targets": plan.names,
        "classes": plan.classes,
        "counts": result.counts,
        "cells": cells,
        "confusion": {"rows": "true target", "columns": "predicted", "matrix": result.confusion},
        "classifier": CLASSIFIER,
        "lbfgs_iterations": result.n_iter,
        "converged": result.converged,
        "wallclock_s": result.wallclock_s,
        "builder": {"module": "ms.eval.probe", "version": ms.__version__, "git_sha": git_sha()},
    }


# --- the command --------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ms.eval.probe", description=__doc__.split("\n")[0])
    p.add_argument(
        "--backbone", action="append", default=None, help="repeatable; default: every config"
    )
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--target", action="append", choices=TARGETS, default=None)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), help="manifest ids")
    p.add_argument("--within", default=WITHIN, help="the manifest of the district probe")
    p.add_argument("--seed", type=int, default=0, help="the folds and the bootstrap")
    p.add_argument("--folds", type=int, default=FOLDS)
    p.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--config-dir", type=Path, default=BACKBONE_CONFIGS)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--md", type=Path, default=N_TABLE_MD)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    tracking = p.add_mutually_exclusive_group()
    tracking.add_argument("--mlflow-uri", default=None)
    tracking.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except ProbeError as exc:
        print(f"{exc.reason}: {exc}", flush=True)
        return 2


def run(args: argparse.Namespace) -> int:
    backbones = args.backbone or sorted(p.stem for p in Path(args.config_dir).glob("*.yaml"))
    targets = args.target or list(TARGETS)
    if len(set(args.datasets)) < 2:
        raise ProbeError("bad_value", "--datasets needs two manifests at least")
    if args.folds < 2:
        raise ProbeError("bad_value", "--folds needs 2 at least")
    manifests = Manifests(args.manifest_root)
    # every probe is planned, and every input checked, before anything is computed
    plans = [make_plan(args, manifests, bb, t) for bb in backbones for t in targets]
    tracker = None if args.no_mlflow else Tracker(args.mlflow_uri)
    known = {identity(r) for r in read_rows(args.n_table)}
    new = []
    for plan in plans:
        in_table = identity(plan.stub()) in known
        in_mlflow = tracker is None or tracker.run_of(plan.probe_id) is not None
        if in_table and in_mlflow:
            print(f"N4 {plan.probe_id}: already there, nothing to do", flush=True)
            continue
        result = compute(plan)
        row = n4_row(plan, result)
        problems = validate_row(row)  # before MLflow and the compute log (spec 005 US-1.2)
        if problems:
            reason, _, what = problems[0].partition(": ")
            raise ProbeError(reason, f"{plan.probe_id}: {what}")
        if not in_table:
            new.append(row)
        mlflow_run = None if in_mlflow else tracker.log(plan, result, row)
        compute_log.log_row(
            step="eval",
            backbone_id=plan.backbone_id,
            res=plan.res,
            dataset="+".join(p.manifest.name.rsplit("_v", 1)[0] for p in plan.parts),
            n_images_or_runs=result.n_scored if plan.folds else result.n_fit + result.n_scored,
            wallclock_s=result.wallclock_s,
            device=f"CPU ({platform.processor() or platform.machine()})",
            notes=f"N4 {plan.probe_id}: {result.n_fit} rows fitted, {result.n_scored} scored",
            extra={
                "probe": {
                    "probe_id": plan.probe_id,
                    "target": plan.target,
                    "n_fit": result.n_fit,
                    "n_scored": result.n_scored,
                    "folds": plan.folds,
                    "lbfgs_iterations": result.n_iter,
                    "converged": result.converged,
                    "mlflow_run_id": mlflow_run,
                }
            },
            path=args.compute_log,
        )
        s = result.score
        print(
            f"N4 {plan.target} {plan.backbone_id} @ {plan.res}: balanced accuracy "
            f"{s['value']:.4f} ({s['ci_low']:.4f}-{s['ci_high']:.4f}) against chance "
            f"{1 / len(plan.names):.3f}, {result.n_scored} rows scored; "
            + ", ".join(f"{plan.names[t]} {r:.3f}" for t, r in sorted(s["recall"].items()))
            + f"; {result.wallclock_s:.1f} s"
            + (f"; MLflow run {mlflow_run}" if mlflow_run else ""),
            flush=True,
        )
    added = append_rows(new, args.n_table)
    rendered = write_md(args.md, args.n_table)
    print(
        f"{repo_relative(args.n_table)}: {added} row(s) added; "
        f"{repo_relative(args.md)} {'rewritten' if rendered else 'unchanged'}"
        + (f"; MLflow {tracker.uri}" if tracker else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
