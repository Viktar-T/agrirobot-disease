"""Train heads on cached features (S4.3; specs/003-heads/spec.md).

    python -m ms.heads.train --backbone dinov2_l14_reg --res 224 \\
        --train-manifest data/manifests/makerere_v1.jsonl [<more manifests>] [--split train]
        [--head linear proto mix] [--seed 0 1 2 3 4] [--val-manifest <file>] [--val-split val]
        [--tokens cls] [--cache-key K] [--allow-test-only]
        [--cache-root data/cache] [--heads-root data/heads] [--config-dir configs/heads]
        [--compute-log model_service/results/compute_log.jsonl]

One run per (head, seed); the defaults are the W3 protocol, every head for seeds 0-4. The
train split is read with purpose=train and the validation split with purpose=select (spec
001 FR-011), both from the training manifests themselves (in domain). Several training
manifests (N2's Makerere + iBean) are read side by side: the first one's role decides, and
a further one may be test_only. Every input is checked before the first run trains, so an
input error writes nothing.

linear and proto are fitted with the recipe of configs/heads/<head>.yaml: AdamW, focal loss
with label smoothing, class-balanced sampling, early stopping on the validation split's
per-class-averaged cross-entropy. proto starts from per-class k-means. mix trains nothing:
it is the fixed 0.5/0.5 mixture of the linear and proto runs with the same inputs and seed,
which are trained first when missing. Same inputs and seed, same weights; the run_id is
derived from every input, so running again does nothing. A manifest whose role is not
train_eval trains only with --allow-test-only (the W2 slice on iBean), never quotable.

Exit codes: 0 trained or already there; 2 input error, nothing written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

import ms
from ms import compute_log
from ms.cache import CACHE_ROOT, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import (
    TRAINED,
    FrozenManifestModified,
    Manifest,
    TestSplitAccessError,
    git_sha,
    load_manifest,
    repo_relative,
    sha256_file,
)
from ms.heads import (
    HEAD_CONFIGS,
    HEADS,
    HEADS_ROOT,
    TOKEN_SPECS,
    MixHead,
    ProtoHead,
    build_head,
    features,
    load_run,
    model_version,
)

#: an input of every run id: 3 since k-means runs on one thread (DECISIONS 67)
TRAINER_VERSION = 3
DEFAULT_HEADS = ("linear", "proto", "mix")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
#: the heads that are fitted; mix mixes them
FITTED = ("linear", "proto")
SAMPLING = ("class_balanced", "uniform")
#: what a fitted head's config gets when it does not say (spec 003 US-4.1)
DEFAULTS = {"sampling": "class_balanced"}


class TrainError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the recipe ---------------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor, y: torch.Tensor, gamma: float, smoothing: float
) -> torch.Tensor:
    """Focal loss against label-smoothed targets: mean over rows of
    -sum_c q_c (1 - p_c)^gamma log p_c, with q = (1 - s) one-hot + s / C."""
    logp = F.log_softmax(logits, dim=1)
    n_classes = logits.shape[1]
    q = torch.full_like(logp, smoothing / n_classes)
    q.scatter_(1, y[:, None], 1 - smoothing + smoothing / n_classes)
    return -(q * (1 - logp.exp()) ** gamma * logp).sum(dim=1).mean()


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def balanced_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    """Cross-entropy averaged per class, then over the classes present: as continuous as a
    loss, and weighted like macro-F1 (every class counts the same)."""
    ce = F.cross_entropy(logits, y, reduction="none")
    return float(np.mean([ce[y == c].mean().item() for c in range(n_classes) if (y == c).any()]))


#: validation scores that `select` may name; higher is better, the second breaks ties
SELECT = {
    "balanced_val_loss": lambda f1, loss, bal: (-bal, f1),
    "macro_f1": lambda f1, loss, bal: (f1, -loss),
}


def epoch_order(y: np.ndarray, sampling: str, generator: torch.Generator) -> torch.Tensor:
    """One epoch's row indices. class_balanced draws len(y) rows with replacement, a row's
    probability proportional to 1 / its class's count, so every class is drawn equally often
    in expectation; uniform is a permutation of the rows (spec 003 US-4.1)."""
    y = np.asarray(y)
    if sampling == "class_balanced":
        weights = torch.from_numpy(1.0 / np.bincount(y)[y])
        return torch.multinomial(weights, len(y), replacement=True, generator=generator)
    if sampling == "uniform":
        return torch.randperm(len(y), generator=generator)
    raise ValueError(f"sampling must be one of {SAMPLING}, got {sampling!r}")


def init_prototypes(x: np.ndarray, y: np.ndarray, n_classes: int, k: int, seed: int) -> np.ndarray:
    """float32 [n_classes, k, D]: per class, the k-means centroids (k-means++, seeded) of its
    L2-normalised rows, themselves L2-normalised (spec 003 US-2.2).

    k-means runs on one thread: on several, it adds the threads' partial cluster sums in the
    order they finish, so the same seed gave other centroids from call to call (DECISIONS 67).
    """
    x = np.asarray(x, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.asarray(y)
    out = np.empty((n_classes, k, x.shape[1]), dtype=np.float32)
    with threadpool_limits(limits=1):
        for c in range(n_classes):
            rows = x[y == c]
            if len(rows) < k:
                raise ValueError(f"class {c} has {len(rows)} rows, fewer than k = {k}")
            km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(rows)
            centres = km.cluster_centers_.astype(np.float32)
            out[c] = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    return out


def _scores(model: torch.nn.Module, xv: torch.Tensor, yv: torch.Tensor, n_classes: int):
    """(macro-F1, cross-entropy, per-class-averaged cross-entropy) on the validation rows."""
    model.eval()
    with torch.no_grad():
        logits = model(xv)
        loss = F.cross_entropy(logits, yv).item()
        bal = balanced_loss(logits, yv, n_classes)
    return macro_f1(yv.numpy(), logits.argmax(dim=1).numpy(), n_classes), loss, bal


def fit(
    head: str,
    x: np.ndarray,
    y: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    n_classes: int,
    cfg: dict,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """linear or proto with the recipe of its config, on CPU, deterministic for a seed; the
    model at its best epoch, its constructor params, the history and the summary."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    if head == "proto":
        params = {
            "prototypes_per_class": int(cfg["prototypes_per_class"]),
            "tau_init": float(cfg["tau_init"]),
        }
        model = build_head("proto", x.shape[1], n_classes, **params)
        start = init_prototypes(x, y, n_classes, params["prototypes_per_class"], seed)
        with torch.no_grad():
            model.prototypes.copy_(torch.from_numpy(start))
        # tau is not decayed: decay would pull log tau to 0, tau to 1 (spec 003 FR-006)
        groups = [{"params": [model.prototypes]}, {"params": [model.log_tau], "weight_decay": 0.0}]
    else:
        params = {}
        model = build_head("linear", x.shape[1], n_classes)
        groups = [{"params": list(model.parameters())}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    xvt, yvt = torch.from_numpy(xv), torch.from_numpy(yv)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stale = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = epoch_order(y, cfg["sampling"], gen)
        total = 0.0
        for k in range(0, len(order), cfg["batch_size"]):
            idx = order[k : k + cfg["batch_size"]]
            loss = focal_loss(model(xt[idx]), yt[idx], cfg["focal_gamma"], cfg["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_f1, val_loss, val_bal = _scores(model, xvt, yvt, n_classes)
        entry = {
            "epoch": epoch,
            "train_loss": round(total / len(order), 6),
            "val_loss": round(val_loss, 6),
            "val_balanced_loss": round(val_bal, 6),
            "val_macro_f1": round(val_f1, 6),
        }
        if isinstance(model, ProtoHead):
            entry["tau"] = round(model.tau, 6)
        history.append(entry)
        score = SELECT[cfg["select"]](val_f1, val_loss, val_bal)
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    assert best is not None
    model.load_state_dict(best["state"])
    model.eval()
    chosen = history[best["epoch"] - 1]
    summary = {
        "select": cfg["select"],
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "stopped_early": len(history) < cfg["max_epochs"],
        "val_macro_f1": chosen["val_macro_f1"],
        "val_loss": chosen["val_loss"],
        "val_balanced_loss": chosen["val_balanced_loss"],
    }
    if isinstance(model, ProtoHead):
        summary["tau"] = round(model.tau, 6)
    return model, params, history, summary


def mix(
    components: dict[str, Any], weights: list[float], xv: np.ndarray, yv: np.ndarray, n: int
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """The fixed mixture of the loaded linear and proto runs; its params and its validation
    scores. Nothing is trained."""
    saved = {h: torch.load(r.folder / "head.pt", weights_only=True) for h, r in components.items()}
    params = {
        "weights": [float(w) for w in weights],
        "linear": saved["linear"].get("params", {}),
        "proto": saved["proto"].get("params", {}),
    }
    model = build_head("mix", xv.shape[1], n, **params)
    assert isinstance(model, MixHead)
    model.linear.load_state_dict(saved["linear"]["state_dict"])
    model.proto.load_state_dict(saved["proto"]["state_dict"])
    val_f1, val_loss, val_bal = _scores(model, torch.from_numpy(xv), torch.from_numpy(yv), n)
    summary = {
        "select": None,
        "epochs_run": 0,
        "best_epoch": None,
        "stopped_early": False,
        "val_macro_f1": round(val_f1, 6),
        "val_loss": round(val_loss, 6),
        "val_balanced_loss": round(val_bal, 6),
    }
    return model, params, summary


# --- configs ------------------------------------------------------------------------------------


def _number(cfg: dict, key: str, low: float, *, integer: bool = False, above: bool = False):
    value = cfg.get(key)
    ok = isinstance(value, int if integer else int | float) and not isinstance(value, bool)
    if not ok or value < low or (above and value == low):
        kind = "an integer" if integer else "a number"
        raise ValueError(f"{key} must be {kind} {'>' if above else '>='} {low}, got {value!r}")
    return value


def check_config(head: str, cfg: dict) -> None:
    """The keys of spec 003 FR-003, with values the recipe can run; ValueError otherwise."""
    if head == "mix":
        if cfg.get("components") != list(FITTED):
            raise ValueError(f"components must be {list(FITTED)}, got {cfg.get('components')!r}")
        w = cfg.get("weights")
        if not (
            isinstance(w, list)
            and len(w) == 2
            and all(isinstance(v, int | float) and v >= 0 for v in w)
            and math.isclose(sum(w), 1.0)
        ):
            raise ValueError(f"weights must be two shares summing to 1, got {w!r}")
        return
    if cfg.get("optimizer") != "adamw" or cfg.get("loss") != "focal":
        raise ValueError("optimizer must be adamw and loss focal (H8 6.5)")
    _number(cfg, "lr", 0, above=True)
    _number(cfg, "weight_decay", 0)
    for key in ("batch_size", "max_epochs", "patience"):
        _number(cfg, key, 1, integer=True)
    _number(cfg, "focal_gamma", 0)
    if not 0 <= _number(cfg, "label_smoothing", 0) < 1:
        raise ValueError("label_smoothing must be below 1")
    if cfg.get("sampling") not in SAMPLING:
        raise ValueError(f"sampling must be one of {SAMPLING}, got {cfg.get('sampling')!r}")
    if cfg.get("select") not in SELECT:
        raise ValueError(f"select must be one of {tuple(SELECT)}, got {cfg.get('select')!r}")
    if head == "proto":
        _number(cfg, "prototypes_per_class", 1, integer=True)
        _number(cfg, "tau_init", 0, above=True)


def load_config(config_dir: Path, head: str) -> tuple[Path, dict[str, Any]]:
    """configs/heads/<head>.yaml with its defaults filled in (the effective recipe)."""
    path = Path(config_dir) / f"{head}.yaml"
    if not path.is_file():
        raise TrainError("bad_config", f"{path} is missing")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainError("bad_config", f"{path}: {exc}") from None
    if not isinstance(raw, dict) or raw.get("head") != head:
        found = raw.get("head") if isinstance(raw, dict) else None
        raise TrainError("bad_config", f"{path} is for head {found!r}, not {head!r}")
    cfg = {**DEFAULTS, **raw} if head in FITTED else dict(raw)
    try:
        check_config(head, cfg)
    except ValueError as exc:
        raise TrainError("bad_config", f"{path}: {exc}") from None
    return path, cfg


# --- inputs -------------------------------------------------------------------------------------


@dataclass
class Side:
    """The training or the validation side of a call: its manifests, their caches and the
    split read. Several manifests (N2's Makerere + iBean) join with + (spec 003 FR-004)."""

    manifests: list[Manifest]
    npzs: list[Path]
    split: str

    @property
    def sha256(self) -> str:
        return "+".join(m.sha256 for m in self.manifests)

    def record(self, classes: list[str]) -> dict[str, Any]:
        """run.json's `train` or `val`."""
        rows = [r for m in self.manifests for r in m.rows]
        roles = [(m.meta or {}).get("role") for m in self.manifests]
        return {
            "manifest": "+".join(repo_relative(m.path) for m in self.manifests),
            "manifest_sha256": self.sha256,
            "frozen": all(m.frozen for m in self.manifests),
            "role": roles[0] if len(roles) == 1 else "+".join(map(str, roles)),
            "split": self.split,
            "split_rules": sorted({r["split_rule"] for r in rows}),
            "n": sum(1 for r in rows if r["class_km2"] in classes),
            "class_counts": dict(sorted(Counter(r["class_km2"] for r in rows).items())),
            "cache": "+".join(repo_relative(p) for p in self.npzs),
        }


@dataclass
class Inputs:
    """What every run of one call shares, read and checked once (spec 003 FR-008)."""

    train: Side
    val: Side
    role: str | None
    classes: list[str]
    cache_meta: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    xv: np.ndarray
    yv: np.ndarray

    @property
    def quotable(self) -> bool:
        """The first training manifest decides (spec 003 US-5.1)."""
        return self.role == "train_eval"

    @property
    def dataset(self) -> str:
        return "+".join(m.rows[0]["dataset"] for m in self.train.manifests)


def read_inputs(args: argparse.Namespace, heads: list[str], configs: dict) -> Inputs:
    paths = list(dict.fromkeys(args.train_manifest))
    try:
        train_ms = [load_manifest(p, args.split, "train") for p in paths]
        val_ms = [load_manifest(p, args.val_split, "select") for p in paths]
        chosen = (
            load_manifest(args.val_manifest, args.val_split, "select")
            if args.val_manifest
            else None
        )
    except TestSplitAccessError as exc:
        raise TrainError("test_split_access", str(exc)) from None
    except FrozenManifestModified as exc:
        raise TrainError("frozen_manifest_modified", str(exc)) from None
    except FileNotFoundError as exc:
        raise TrainError("missing_file", str(exc)) from None
    names = ", ".join(m.path.name for m in train_ms)
    if chosen is not None:
        val_ms = [m for m in val_ms if m.sha256 == chosen.sha256]
        if not val_ms:
            raise TrainError(
                "val_not_in_domain",
                f"{chosen.path.name} is not a training manifest ({names}): early stopping reads "
                "the training manifests' own validation splits (spec 003 US-3.2)",
            )
    roles = [(m.meta or {}).get("role") for m in train_ms]
    if roles[0] == "holdout_unknown" or (roles[0] != "train_eval" and not args.allow_test_only):
        raise TrainError(
            "role_not_trainable",
            f"{train_ms[0].path.name} has role {roles[0]!r} (spec 001 FR-001): heads train on "
            "train_eval manifests; --allow-test-only trains on a test_only one for the W2 slice, "
            "never quoted",
        )
    for m, role in zip(train_ms[1:], roles[1:], strict=True):
        if role not in ("train_eval", "test_only"):
            raise TrainError(
                "role_not_trainable",
                f"{m.path.name} has role {role!r}: a further training manifest is train_eval or "
                "test_only (spec 003 US-5.1)",
            )
    rows = [r for m in train_ms for r in m.rows]
    strays = sorted({r["class_km2"] for r in rows} - set(TRAINED))
    if strays:
        raise TrainError("bad_value", f"{names} {args.split}: untrainable classes {strays}")
    classes = [c for c in TRAINED if any(r["class_km2"] == c for r in rows)]
    if len(classes) < 2:
        raise TrainError("bad_value", f"{names} {args.split} hold {classes}: two classes at least")
    val_rows = [[r for r in m.rows if r["class_km2"] in classes] for m in val_ms]
    present = {r["class_km2"] for part in val_rows for r in part}
    missing = [c for c in classes if c not in present]
    if missing:
        raise TrainError("bad_value", f"{names} {args.val_split} have no rows of {missing}")
    if "proto" in heads:
        k = configs["proto"][1]["prototypes_per_class"]
        counts = Counter(r["class_km2"] for r in rows)
        few = {c: counts[c] for c in classes if counts[c] < k}
        if few:
            raise TrainError(
                "too_few_rows", f"{names} {args.split}: {few} train rows, fewer than the {k} "
                "prototypes per class of proto",
            )  # fmt: skip

    caches, npzs = {}, {}
    try:
        for m in train_ms:
            npzs[m.sha256] = find_cache(
                args.cache_root, args.backbone, args.res, m.name, m.sha256, key=args.cache_key
            )
            caches[m.sha256] = load_cache(npzs[m.sha256], m.sha256)
    except (FileNotFoundError, CacheMismatchError) as exc:
        raise TrainError("missing_cache", str(exc)) from None
    keys = {c.meta["cache_key"] for c in caches.values()}
    if len(keys) > 1:
        raise TrainError(
            "missing_cache", f"the training manifests' features come from cache keys {sorted(keys)}"
        )

    def matrix(parts):
        xs, ys = [], []
        for m, part in parts:
            cache = caches[m.sha256]
            where = {str(i): k for k, i in enumerate(cache.image_id)}
            xs.append(features(cache, np.array([where[r["image_id"]] for r in part]), args.tokens))
            ys.append(np.array([classes.index(r["class_km2"]) for r in part], dtype=np.int64))
        return np.concatenate(xs), np.concatenate(ys)

    x, y = matrix([(m, m.rows) for m in train_ms])
    xv, yv = matrix(list(zip(val_ms, val_rows, strict=True)))
    return Inputs(
        Side(train_ms, [npzs[m.sha256] for m in train_ms], args.split),
        Side(val_ms, [npzs[m.sha256] for m in val_ms], args.val_split),
        roles[0],
        classes,
        next(iter(caches.values())).meta,
        x,
        y,
        xv,
        yv,
    )


# --- the runs -----------------------------------------------------------------------------------


def plan_heads(requested: list[str]) -> list[str]:
    """The heads to run, in order: mix after its components, which it needs."""
    wanted = set(requested) | (set(FITTED) if "mix" in requested else set())
    return [h for h in HEADS if h in wanted]


def run_identity(
    args: argparse.Namespace, inp: Inputs, head: str, cfg_sha: str, seed: int, components
) -> tuple[str, str]:
    """(run_id, the sha256 of its inputs) (spec 003 FR-002)."""
    inputs: dict[str, Any] = {
        "trainer_version": TRAINER_VERSION,
        "backbone_id": args.backbone,
        "res": args.res,
        "cache_key": inp.cache_meta["cache_key"],
        "tokens": args.tokens,
        "head_config_sha256": cfg_sha,
        "seed": seed,
        "train": [inp.train.sha256, args.split],
        "val": [inp.val.sha256, args.val_split],
    }
    if components is not None:
        inputs["components"] = [c["run_id"] for c in components]
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    tokens = args.tokens.replace("+", "_")
    return f"{args.backbone}-{args.res}-{head}-{tokens}-s{seed}-{digest[:8]}", digest


def train_run(
    args: argparse.Namespace, inp: Inputs, head: str, seed: int, configs: dict, done: dict
) -> str:
    """One run: trained, mixed or already there; its run_id."""
    cfg_path, cfg = configs[head]
    head_config = {"path": repo_relative(cfg_path), "sha256": sha256_file(cfg_path), **cfg}
    components = None
    if head == "mix":
        components = [{"head": h, "run_id": done[(h, seed)]} for h in cfg["components"]]
    run_id, digest = run_identity(args, inp, head, head_config["sha256"], seed, components)
    folder = Path(args.heads_root) / run_id
    if (folder / "run.json").exists():
        print(f"run exists, nothing to do: {repo_relative(folder)}", flush=True)
        return run_id

    started = time.perf_counter()
    n_classes = len(inp.classes)
    if head == "mix":
        loaded = {c["head"]: load_run(Path(args.heads_root) / c["run_id"]) for c in components}
        model, params, summary = mix(loaded, cfg["weights"], inp.xv, inp.yv, n_classes)
        history: list[dict[str, Any]] = []
    else:
        model, params, history, summary = fit(
            head, inp.x, inp.y, inp.xv, inp.yv, n_classes, cfg, seed
        )
    bb = inp.cache_meta
    run = {
        "run_json_version": 2,
        "run_id": run_id,
        "head": head,
        "tokens": args.tokens,
        "seed": seed,
        "classes": inp.classes,
        "input_dim": int(inp.x.shape[1]),
        "model_version": model_version(args.backbone, args.res, head, inp.train.sha256),
        "quotable": inp.quotable,
        "allow_test_only": bool(args.allow_test_only),
        "backbone": {
            "backbone_id": args.backbone,
            "res": args.res,
            "cache_key": bb["cache_key"],
            "weights_sha256": bb["weights_sha256"],
            "preprocess_string": bb["preprocess_string"],
            "compute_dtype": bb["compute_dtype"],
            "hf_id": bb["hf_id"],
            "revision": bb["revision"],
            "research_only_until_c5": bb["research_only_until_c5"],
        },
        "train": inp.train.record(inp.classes),
        "val": inp.val.record(inp.classes),
        "head_config": head_config,
        "fit": {**summary, "history": history},
        "components": components,
        "inputs_sha256": digest,
        "wallclock_s": None,
        "device": f"CPU ({platform.processor() or platform.machine()})",
        "torch_version": torch.__version__,
        "created_at": _now(),
        "builder": {"module": "ms.heads.train", "version": ms.__version__, "git_sha": git_sha()},
        "notes": None
        if inp.quotable
        else f"trained on a {inp.role} manifest (--allow-test-only): pipeline check, never quoted",
    }
    saved = {
        "head": head,
        "classes": inp.classes,
        "tokens": args.tokens,
        "input_dim": int(inp.x.shape[1]),
        "params": params,
        "state_dict": model.state_dict(),
    }
    if components is not None:
        saved["components"] = components
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save(saved, tmp / "head.pt")
    wallclock = round(time.perf_counter() - started, 3)
    run["wallclock_s"] = wallclock
    (tmp / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, folder)

    head_row: dict[str, Any] = {
        "run_id": run_id,
        "head": head,
        "tokens": args.tokens,
        "seed": seed,
        "n_train": len(inp.y),
        "n_val": len(inp.yv),
        "epochs_run": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
    }
    if components is not None:
        head_row["components"] = [c["run_id"] for c in components]
    what = "mixed" if head == "mix" else f"{summary['epochs_run']} epochs"
    compute_log.log_row(
        step="train_head",
        backbone_id=args.backbone,
        res=args.res,
        dataset=inp.dataset,
        n_images_or_runs=1,
        wallclock_s=wallclock,
        device=run["device"],
        vram_gb=None,
        notes=f"{run_id}; {len(inp.y)} train rows, {what}",
        extra={"head": head_row},
        path=args.compute_log,
    )
    best = (
        "" if head == "mix" else f"best epoch {summary['best_epoch']} of {summary['epochs_run']}, "
    )
    tau = f", tau {summary['tau']:.4f}" if "tau" in summary else ""
    print(
        f"{repo_relative(folder)}: {head} on {args.tokens}, {len(inp.y)} train / {len(inp.yv)} "
        f"val rows, {best}val macro-F1 {summary['val_macro_f1']:.4f}{tau}, {wallclock:.1f} s; "
        f"{run['model_version']}" + ("" if inp.quotable else " (not quotable)"),
        flush=True,
    )
    return run_id


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ms.heads.train", description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--train-manifest", type=Path, nargs="+", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--head", nargs="+", choices=tuple(HEADS), default=list(DEFAULT_HEADS))
    p.add_argument("--seed", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--val-manifest", type=Path, default=None, help="default: every one")
    p.add_argument("--val-split", default="val")
    p.add_argument("--tokens", default="cls", choices=TOKEN_SPECS)
    p.add_argument("--cache-key", default=None, help="when several caches of a manifest match")
    p.add_argument(
        "--allow-test-only", action="store_true", help="train on a test_only manifest (W2 slice)"
    )
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--config-dir", type=Path, default=HEAD_CONFIGS)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return train(args)
    except TrainError as exc:
        print(f"{exc.reason}: {exc}", flush=True)
        return 2


def train(args: argparse.Namespace) -> int:
    heads = plan_heads(args.head)
    seeds = list(dict.fromkeys(args.seed))
    configs = {h: load_config(args.config_dir, h) for h in heads}
    inp = read_inputs(args, heads, configs)
    done: dict[tuple[str, int], str] = {}
    for head in heads:
        for seed in seeds:
            done[(head, seed)] = train_run(args, inp, head, seed, configs, done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
