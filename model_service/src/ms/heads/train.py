"""Train one head on cached features (S4.3, W2 slice version; spec 003 in W3).

    python -m ms.heads.train --backbone dinov2_l14_reg --res 224 --head linear --seed 0 \\
        --train-manifest data/manifests/ibean_v1.jsonl --split train --val-split val \\
        [--val-manifest <file>] [--tokens cls] [--cache-key K] [--allow-test-only]
        [--cache-root data/cache] [--heads-root data/heads] [--config configs/heads/<head>.yaml]
        [--compute-log model_service/results/compute_log.jsonl]

Reads the train split with purpose=train and the validation split with purpose=select
(spec 001 FR-011, so no test row can reach training), and their features from the cache
(spec 002). Fits the head with the recipe in configs/heads/<head>.yaml: AdamW, focal loss
with label smoothing, class-balanced sampling, early stopping on the validation split
(`select`: the per-class-averaged cross-entropy). Same inputs and seed, same weights. The
run_id is derived from every input, so running again does nothing. A manifest whose role
is not train_eval trains only with --allow-test-only (the W2 slice on iBean), and the run
is marked not quotable.

Exit codes: 0 trained or already there; 2 input error, nothing written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.nn import functional as F

import ms
from ms import compute_log
from ms.cache import CACHE_ROOT, CacheMismatchError, find_cache, load_cache
from ms.data.manifests import TRAINED, git_sha, load_manifest, repo_relative, sha256_file
from ms.heads import HEAD_CONFIGS, HEADS_ROOT, TOKEN_SPECS, build_head, features, model_version

TRAINER_VERSION = 1


class TrainError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def fit(
    x: np.ndarray,
    y: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    n_classes: int,
    cfg: dict,
    seed: int,
) -> tuple[torch.nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """The recipe of configs/heads/<head>.yaml, on CPU, deterministic for a seed."""
    if cfg["select"] not in SELECT:
        raise TrainError(
            "bad_config", f"select must be one of {tuple(SELECT)}, got {cfg['select']!r}"
        )
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_head(cfg["head"], x.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    xvt, yvt = torch.from_numpy(xv), torch.from_numpy(yv)
    counts = np.bincount(y, minlength=n_classes)
    weights = torch.from_numpy(1.0 / counts[y])
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stale = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        if cfg["sampling"] == "class_balanced":
            order = torch.multinomial(weights, len(y), replacement=True, generator=gen)
        else:
            order = torch.randperm(len(y), generator=gen)
        total = 0.0
        for k in range(0, len(order), cfg["batch_size"]):
            idx = order[k : k + cfg["batch_size"]]
            loss = focal_loss(model(xt[idx]), yt[idx], cfg["focal_gamma"], cfg["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(xvt)
            val_loss = F.cross_entropy(logits, yvt).item()
            val_bal = balanced_loss(logits, yvt, n_classes)
        val_f1 = macro_f1(yv, logits.argmax(dim=1).numpy(), n_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / len(order), 6),
                "val_loss": round(val_loss, 6),
                "val_balanced_loss": round(val_bal, 6),
                "val_macro_f1": round(val_f1, 6),
            }
        )
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
        "val_macro_f1": chosen["val_macro_f1"],
        "val_loss": chosen["val_loss"],
        "val_balanced_loss": chosen["val_balanced_loss"],
        "stopped_early": len(history) < cfg["max_epochs"],
    }
    return model, history, summary


def _split_side(
    manifest, split: str, cache_path: Path, classes: list[str] | None
) -> dict[str, Any]:
    rows = manifest.rows
    return {
        "manifest": repo_relative(manifest.path),
        "manifest_sha256": manifest.sha256,
        "frozen": manifest.frozen,
        "role": (manifest.meta or {}).get("role"),
        "split": split,
        "split_rules": sorted({r["split_rule"] for r in rows}),
        "n": sum(1 for r in rows if classes is None or r["class_km2"] in classes),
        "class_counts": dict(sorted(Counter(r["class_km2"] for r in rows).items())),
        "cache": repo_relative(cache_path),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ms.heads.train", description=__doc__.split("\n")[0])
    p.add_argument("--backbone", required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--head", required=True, choices=("linear",), help="proto and mix come in W3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--val-manifest", type=Path, default=None, help="default: the train manifest")
    p.add_argument("--val-split", default="val")
    p.add_argument("--tokens", default="cls", choices=TOKEN_SPECS)
    p.add_argument("--cache-key", default=None, help="when several caches of a manifest match")
    p.add_argument(
        "--allow-test-only", action="store_true", help="train on a test_only manifest (W2 slice)"
    )
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--config", type=Path, default=None, help="default: configs/heads/<head>.yaml")
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
    started = time.perf_counter()
    cfg_path = args.config or HEAD_CONFIGS / f"{args.head}.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if cfg.get("head") != args.head:
        raise TrainError(
            "bad_config", f"{cfg_path} is for head {cfg.get('head')!r}, not {args.head!r}"
        )

    train_m = load_manifest(args.train_manifest, args.split, "train")
    val_m = load_manifest(args.val_manifest or args.train_manifest, args.val_split, "select")
    role = (train_m.meta or {}).get("role")
    if role == "holdout_unknown" or (role != "train_eval" and not args.allow_test_only):
        raise TrainError(
            "role_not_trainable",
            f"{train_m.path.name} has role {role!r} (spec 001 FR-001): heads train on train_eval "
            "manifests; --allow-test-only trains on a test_only one for the W2 slice, never quoted",
        )
    strays = sorted({r["class_km2"] for r in train_m.rows} - set(TRAINED))
    if strays:
        raise TrainError(
            "bad_value", f"{train_m.path.name} {args.split}: untrainable classes {strays}"
        )
    classes = [c for c in TRAINED if any(r["class_km2"] == c for r in train_m.rows)]
    if len(classes) < 2:
        raise TrainError(
            "bad_value", f"{train_m.path.name} {args.split} holds {classes}: two classes at least"
        )
    val_rows = [r for r in val_m.rows if r["class_km2"] in classes]
    if not val_rows:
        raise TrainError(
            "bad_value", f"{val_m.path.name} {args.val_split} has no rows of {classes}"
        )

    try:
        train_npz = find_cache(
            args.cache_root,
            args.backbone,
            args.res,
            train_m.name,
            train_m.sha256,
            key=args.cache_key,
        )
        val_npz = find_cache(
            args.cache_root, args.backbone, args.res, val_m.name, val_m.sha256, key=args.cache_key
        )
        train_cache = load_cache(train_npz, train_m.sha256)
        val_cache = train_cache if val_npz == train_npz else load_cache(val_npz, val_m.sha256)
    except (FileNotFoundError, CacheMismatchError) as exc:
        raise TrainError("missing_cache", str(exc)) from None
    if train_cache.meta["cache_key"] != val_cache.meta["cache_key"]:
        raise TrainError(
            "bad_value", "train and validation features come from different cache keys"
        )

    def matrix(cache, rows):
        where = {str(i): k for k, i in enumerate(cache.image_id)}
        index = np.array([where[r["image_id"]] for r in rows])
        y = np.array([classes.index(r["class_km2"]) for r in rows], dtype=np.int64)
        return features(cache, index, args.tokens), y

    x, y = matrix(train_cache, train_m.rows)
    xv, yv = matrix(val_cache, val_rows)

    head_cfg = {"path": repo_relative(Path(cfg_path)), "sha256": sha256_file(Path(cfg_path)), **cfg}
    key = train_cache.meta["cache_key"]
    inputs = {
        "trainer_version": TRAINER_VERSION,
        "backbone_id": args.backbone,
        "res": args.res,
        "cache_key": key,
        "tokens": args.tokens,
        "head_config_sha256": head_cfg["sha256"],
        "seed": args.seed,
        "train": [train_m.sha256, args.split],
        "val": [val_m.sha256, args.val_split],
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    tokens = args.tokens.replace("+", "_")
    run_id = f"{args.backbone}-{args.res}-{args.head}-{tokens}-s{args.seed}-{digest[:8]}"
    folder = Path(args.heads_root) / run_id
    if (folder / "run.json").exists():
        print(f"run exists, nothing to do: {repo_relative(folder)}", flush=True)
        return 0

    model, history, summary = fit(x, y, xv, yv, len(classes), cfg, args.seed)
    wallclock = round(time.perf_counter() - started, 3)
    quotable = role == "train_eval"
    bb = train_cache.meta
    run = {
        "run_json_version": 1,
        "run_id": run_id,
        "head": args.head,
        "tokens": args.tokens,
        "seed": args.seed,
        "classes": classes,
        "input_dim": int(x.shape[1]),
        "model_version": model_version(args.backbone, args.res, args.head, train_m.sha256),
        "quotable": quotable,
        "allow_test_only": bool(args.allow_test_only),
        "backbone": {
            "backbone_id": args.backbone,
            "res": args.res,
            "cache_key": key,
            "weights_sha256": bb["weights_sha256"],
            "preprocess_string": bb["preprocess_string"],
            "compute_dtype": bb["compute_dtype"],
            "hf_id": bb["hf_id"],
            "revision": bb["revision"],
            "research_only_until_c5": bb["research_only_until_c5"],
        },
        "train": _split_side(train_m, args.split, train_npz, classes),
        "val": _split_side(val_m, args.val_split, val_npz, classes),
        "head_config": head_cfg,
        "fit": {**summary, "history": history},
        "inputs_sha256": digest,
        "wallclock_s": wallclock,
        "device": f"CPU ({platform.processor() or platform.machine()})",
        "torch_version": torch.__version__,
        "created_at": _now(),
        "builder": {"module": "ms.heads.train", "version": ms.__version__, "git_sha": git_sha()},
        "notes": None
        if quotable
        else f"trained on a {role} manifest (--allow-test-only): pipeline check, never quoted",
    }

    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save(
        {"head": args.head, "classes": classes, "tokens": args.tokens, "input_dim": int(x.shape[1]),
         "state_dict": model.state_dict()},
        tmp / "head.pt",
    )  # fmt: skip
    (tmp / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, folder)

    compute_log.log_row(
        step="train_head",
        backbone_id=args.backbone,
        res=args.res,
        dataset=train_m.rows[0]["dataset"],
        n_images_or_runs=1,
        wallclock_s=wallclock,
        device=run["device"],
        vram_gb=None,
        notes=f"{run_id}; {len(y)} train rows, {summary['epochs_run']} epochs",
        extra={
            "head": {
                "run_id": run_id,
                "head": args.head,
                "tokens": args.tokens,
                "seed": args.seed,
                "n_train": len(y),
                "n_val": len(yv),
                "epochs_run": summary["epochs_run"],
                "best_epoch": summary["best_epoch"],
            }
        },
        path=args.compute_log,
    )
    print(
        f"{repo_relative(folder)}: {args.head} on {args.tokens}, "
        f"{len(y)} train / {len(yv)} val rows, "
        f"best epoch {summary['best_epoch']} of {summary['epochs_run']}, val macro-F1 "
        f"{summary['val_macro_f1']:.4f}, {wallclock:.1f} s; {run['model_version']}"
        + ("" if quotable else " (not quotable)"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
