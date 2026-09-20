"""Fit a trained head's abstention thresholds and its temperature (`make abstain`; S4.4,
specs/004-abstention/spec.md).

    python -m ms.abstain.fit [--run <run_id> ...] [--heads-root data/heads]
        [--cache-root data/cache] [--config model_service/configs/abstain.yaml]
        [--compute-log model_service/results/compute_log.jsonl] [--force]

For every run (or the ones named), on the run's own **in-domain validation slice** and
nothing else (US-2): the kNN bank and the Gaussians from its training rows, then a threshold
per score at each declared coverage, then the temperature. The result is
`data/heads/<run_id>/abstain.json` (FR-002) and one compute-log row (N7). A run already
fitted with the same inputs is left alone; `--force` fits it again.

Exit codes: 0 when every run is fitted or already there. 2 on an input error: every input is
checked before the first fit (FR-011), so nothing is written and no row is appended.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import ms
from ms import compute_log
from ms.abstain import (
    ABSTAIN_CONFIG,
    ABSTAIN_JSON_VERSION,
    FLAT,
    AbstainError,
    coverage_key,
    fit_temperature,
    gaussians,
    inputs_sha256,
    knn_distance,
    load_abstain_config,
    logsumexp,
)
from ms.cache import CACHE_ROOT, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import (
    FrozenManifestModified,
    Manifest,
    TestSplitAccessError,
    git_sha,
    load_manifest,
    repo_relative,
    sha256_file,
)
from ms.heads import HEADS_ROOT, features, list_runs, load_run, resolve_path


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Plan:
    """One run's fit, with everything read and checked before anything is written."""

    folder: Path
    meta: dict[str, Any]
    train: list[Manifest]
    val: list[Manifest]
    on: dict[str, Any]
    npzs: list[Path]
    config_sha: str
    digest: str

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def group(self) -> tuple:
        """What two runs share when their bank and their slice are the same features: the
        kNN distances and the Gaussians are computed once for the group."""
        bb = self.meta["backbone"]
        return (
            bb["backbone_id"],
            bb["res"],
            bb["cache_key"],
            self.meta["tokens"],
            "+".join(m.sha256 for m in self.train),
            self.meta["train"]["split"],
            "+".join(m.sha256 for m in self.val),
            self.meta["val"]["split"],
            tuple(self.meta["classes"]),
        )

    @property
    def dataset(self) -> str:
        return "+".join(m.rows[0]["dataset"] for m in self.train if m.rows)


def _record(meta: dict[str, Any], train: list[Manifest], val: list[Manifest], classes) -> dict:
    """abstain.json's `fitted_on` (FR-002): the bank's manifests and split, the slice's, and
    the slice's classes."""
    val_rows = [r for m in val for r in m.rows if r["class_km2"] in classes]
    return {
        "manifest": "+".join(repo_relative(m.path) for m in train),
        "manifest_sha256": "+".join(m.sha256 for m in train),
        "frozen": all(m.frozen for m in train) and all(m.frozen for m in val),
        "train_split": meta["train"]["split"],
        "n_train": sum(1 for m in train for r in m.rows if r["class_km2"] in classes),
        "val_split": meta["val"]["split"],
        "n_val": len(val_rows),
        "class_counts": dict(sorted(Counter(r["class_km2"] for r in val_rows).items())),
    }


def read_plans(args: argparse.Namespace, cfg: dict[str, Any], cfg_sha: str) -> list[Plan]:
    """Every run's inputs, read and checked (FR-011). Nothing is written here."""
    folders = list_runs(args.heads_root)
    if args.run:
        by_id = {f.name: f for f in folders}
        missing = [r for r in args.run if r not in by_id]
        if missing:
            raise AbstainError(
                "missing_run", f"{', '.join(missing)} is not under {args.heads_root}"
            )
        folders = [by_id[r] for r in args.run]
    slice_needed = max(math.ceil(round(1 / (1 - c), 9)) for c in cfg["coverages"])
    loaded: dict[tuple, Manifest] = {}

    def manifests(side: str, purpose: str, meta: dict) -> list[Manifest]:
        out = []
        for path in meta[side]["manifest"].split("+"):
            key = (path, meta[side]["split"], purpose)
            if key not in loaded:
                loaded[key] = load_manifest(resolve_path(path), meta[side]["split"], purpose)
            out.append(loaded[key])
        return out

    plans = []
    for folder in folders:
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        try:
            train = manifests("train", "train", meta)
            val = manifests("val", "select", meta)
        except TestSplitAccessError as exc:
            raise AbstainError("test_split_access", f"{folder.name}: {exc}") from None
        except FrozenManifestModified as exc:
            raise AbstainError("frozen_manifest_modified", f"{folder.name}: {exc}") from None
        except FileNotFoundError as exc:
            raise AbstainError("missing_file", f"{folder.name}: {exc}") from None
        bb, npzs = meta["backbone"], {}
        for m in {mm.sha256: mm for mm in [*train, *val]}.values():
            try:
                npzs[m.sha256] = find_cache(
                    args.cache_root, bb["backbone_id"], bb["res"], m.name, m.sha256,
                    key=bb["cache_key"],
                )  # fmt: skip
            except (FileNotFoundError, CacheMismatchError) as exc:
                raise AbstainError("missing_cache", f"{folder.name}: {exc}") from None
        on = _record(meta, train, val, meta["classes"])
        if on["n_val"] < slice_needed:
            raise AbstainError(
                "too_few_rows",
                f"{folder.name}: {on['n_val']} validation rows; coverage "
                f"{max(cfg['coverages']):.2f} needs {slice_needed}",
            )
        if on["n_train"] < cfg["k"] + 1:
            raise AbstainError(
                "too_few_rows",
                f"{folder.name}: {on['n_train']} training rows for k = {cfg['k']}",
            )
        skeleton = {
            "run_id": meta["run_id"],
            "tokens": meta["tokens"],
            "fitted_on": on,
            "config": {"sha256": cfg_sha},
            "cache": {"cache_key": bb["cache_key"]},
        }
        bank_npzs = [npzs[m.sha256] for m in train]
        plans.append(
            Plan(folder, meta, train, val, on, bank_npzs, cfg_sha, inputs_sha256(skeleton))
        )
    return plans


def matrix(plan: Plan, side: list[Manifest], cache_root: Path, *, labels: bool):
    """The features of one side of a run, as ms.heads.features gives them, and the class
    index of each row. Rows of a class the head does not have are left out."""
    classes, bb = plan.meta["classes"], plan.meta["backbone"]
    xs, ys = [], []
    for m in side:
        npz = find_cache(
            cache_root, bb["backbone_id"], bb["res"], m.name, m.sha256, key=bb["cache_key"]
        )
        cache = load_cache(npz, m.sha256)
        where = {str(i): k for k, i in enumerate(cache.image_id)}
        rows = [r for r in m.rows if r["class_km2"] in classes]
        index = np.array([where[r["image_id"]] for r in rows])
        xs.append(features(cache, index, plan.meta["tokens"]))
        ys.append(np.array([classes.index(r["class_km2"]) for r in rows], dtype=np.int64))
    x = np.concatenate(xs)
    return (x, np.concatenate(ys)) if labels else x


def fit_run(plan: Plan, shared: dict[str, Any], cfg: dict[str, Any], cfg_path: Path) -> dict:
    """One run's abstain.json payload. `shared` holds what its group already computed: the
    slice's knn and maha scores, which do not depend on the head."""
    run = load_run(plan.folder)
    z = run.logits(shared["xv"]).astype(np.float64)
    slice_scores = {
        "conf": z.max(axis=1),
        "knn": shared["knn"],
        "maha": shared["maha"],
        "energy": -logsumexp(z),
    }
    for score in ("conf", "knn", "maha"):
        values = slice_scores[score]
        if float(values.max() - values.min()) <= FLAT:
            raise AbstainError(
                "degenerate_score",
                f"{plan.run_id}: {score} is constant over the validation slice "
                f"({float(values[0]):.6g}): it has no threshold to give",
            )
    thresholds = {}
    # the distance scores keep the same share of the slice at every coverage: H8 §6.6's
    # "threshold at TPR 95 % on in-domain validation" (US-3.1, DECISIONS 94)
    tau_knn = float(np.quantile(slice_scores["knn"], cfg["distance_tpr"]))
    tau_maha = float(np.quantile(slice_scores["maha"], cfg["distance_tpr"]))
    for coverage in sorted(cfg["coverages"]):
        tau_conf = float(np.quantile(slice_scores["conf"], 1 - coverage))
        kept = (slice_scores["conf"] >= tau_conf) & (slice_scores["knn"] <= tau_knn)
        thresholds[coverage_key(coverage)] = {
            "tau_conf": round(tau_conf, 6),
            "tau_knn": round(tau_knn, 6),
            "tau_maha": round(tau_maha, 6),
            "coverage_achieved": round(float(kept.mean()), 6),
        }
    bb, on = plan.meta["backbone"], plan.on
    return {
        "abstain_json_version": ABSTAIN_JSON_VERSION,
        "run_id": plan.run_id,
        "head": plan.meta["head"],
        "tokens": plan.meta["tokens"],
        "seed": plan.meta["seed"],
        "classes": plan.meta["classes"],
        "input_dim": plan.meta["input_dim"],
        "model_version": plan.meta["model_version"],
        "quotable": plan.meta["quotable"],
        "fitted_on": on,
        "scores": {
            "conf": {"higher_is_stranger": False},
            "knn": {"higher_is_stranger": True, "k": int(cfg["k"])},
            "maha": {"higher_is_stranger": True, "epsilon": float(cfg["epsilon"])},
            "energy": {"higher_is_stranger": True},
        },
        "thresholds": thresholds,
        "temperature": fit_temperature(z, shared["yv"]),
        "ece_bins": int(cfg["ece_bins"]),
        "config": {"path": repo_relative(cfg_path), "sha256": plan.config_sha, **cfg},
        "cache": {
            "backbone_id": bb["backbone_id"],
            "res": bb["res"],
            "cache_key": bb["cache_key"],
            "npz": "+".join(repo_relative(p) for p in plan.npzs),
        },
        "inputs_sha256": plan.digest,
        "wallclock_s": None,
        "device": f"CPU ({platform.processor() or platform.machine()})",
        "created_at": _now(),
        "builder": {"module": "ms.abstain.fit", "version": ms.__version__, "git_sha": git_sha()},
        "notes": None
        if plan.meta["quotable"]
        else "the run is not quotable (spec 003 US-5.1); its operating point is not either",
    }


def write_fit(folder: Path, payload: dict[str, Any]) -> None:
    """FR-001: written under a temporary name and renamed, so a half-written file is never
    read as a fit."""
    tmp = folder / "abstain.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                   newline="\n")  # fmt: skip
    tmp.replace(folder / "abstain.json")


def run_fits(args: argparse.Namespace) -> int:
    cfg = load_abstain_config(args.config)
    cfg_sha = sha256_file(Path(args.config))
    plans = read_plans(args, cfg, cfg_sha)
    todo = []
    for plan in plans:
        path = plan.folder / "abstain.json"
        fitted = (
            path.exists()
            and json.loads(path.read_text(encoding="utf-8")).get("inputs_sha256") == plan.digest
        )
        if fitted and not args.force:
            print(f"fit exists, nothing to do: {repo_relative(plan.folder)}", flush=True)
        else:
            todo.append(plan)
    if not todo:
        print(f"{len(plans)} run(s), nothing to fit", flush=True)
        return 0

    groups: dict[tuple, list[Plan]] = {}
    for plan in todo:
        groups.setdefault(plan.group, []).append(plan)
    print(
        f"{len(todo)} run(s) to fit in {len(groups)} group(s) of shared features "
        f"(the bank and the slice do not depend on the head)",
        flush=True,
    )
    fitted = 0
    for group in groups.values():
        first = group[0]
        started = time.perf_counter()
        bank, y_bank = matrix(first, first.train, args.cache_root, labels=True)
        xv, yv = matrix(first, first.val, args.cache_root, labels=True)
        shared = {
            "xv": xv,
            "yv": yv,
            "knn": knn_distance(bank, xv, int(cfg["k"])),
            "maha": gaussians(
                bank, y_bank, len(first.meta["classes"]), float(cfg["epsilon"])
            ).distance(xv),
        }
        print(
            f"  {first.dataset} @ {first.meta['backbone']['backbone_id']}/"
            f"{first.meta['backbone']['res']}: bank {len(bank)}, slice {len(xv)} rows, "
            f"{time.perf_counter() - started:.1f} s for {len(group)} run(s)",
            flush=True,
        )
        for plan in group:
            one = time.perf_counter()
            payload = fit_run(plan, shared, cfg, Path(args.config))
            payload["wallclock_s"] = round(time.perf_counter() - one, 3)
            write_fit(plan.folder, payload)
            fitted += 1
            tau = payload["thresholds"][coverage_key(cfg["default_coverage"])]
            compute_log.log_row(
                step="fit_abstain",
                backbone_id=payload["cache"]["backbone_id"],
                res=payload["cache"]["res"],
                dataset=plan.dataset,
                n_images_or_runs=1,
                wallclock_s=payload["wallclock_s"],
                device=payload["device"],
                vram_gb=None,
                notes=f"{plan.run_id}; {payload['fitted_on']['n_val']} validation rows",
                extra={
                    "abstain": {
                        "run_id": plan.run_id,
                        "k": int(cfg["k"]),
                        "n_train": payload["fitted_on"]["n_train"],
                        "n_val": payload["fitted_on"]["n_val"],
                        "T": payload["temperature"]["T"],
                    }
                },
                path=args.compute_log,
            )
            print(
                f"    {plan.run_id}: T {payload['temperature']['T']:.4f}, at coverage "
                f"{cfg['default_coverage']:.2f} tau_conf {tau['tau_conf']:.4f} tau_knn "
                f"{tau['tau_knn']:.4f}, kept {tau['coverage_achieved']:.3f}",
                flush=True,
            )
    print(f"{fitted} fit(s) written under {repo_relative(Path(args.heads_root))}", flush=True)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ms.abstain.fit", description=__doc__.split("\n")[0])
    p.add_argument(
        "--run", action="append", default=None, help="run_id (repeatable); default: every run"
    )
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--config", type=Path, default=ABSTAIN_CONFIG)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    p.add_argument("--force", action="store_true", help="fit again a run that is already fitted")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_fits(args)
    except AbstainError as exc:
        print(f"{exc.reason}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
