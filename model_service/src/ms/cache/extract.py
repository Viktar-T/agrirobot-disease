"""Feature extraction into the cache - spec 002 (`make cache BB=<id> RES=<res> DS=<manifest>`).

    python -m ms.cache.extract --backbone dinov2_l14_reg --res 224 --dataset ibean
    python -m ms.cache.extract --backbone dinov2_l14_reg --res 224 --manifest <file.jsonl>
        [--raw-root data/raw] [--cache-root data/cache]
        [--config-dir model_service/configs/backbones]
        [--device cuda|cpu] [--dtype float16|float32] [--batch-size N] [--shard-size N]
        [--max-shards K] [--workers N] [--compute-log model_service/results/compute_log.jsonl]

Every row of the manifest (every split, read with purpose=extract) is hashed as it is read,
preprocessed by the backbone config's preprocess_string for --res, and run through the
backbone loaded in float32 from its pinned revision (float16 = CUDA autocast). The run
writes shards of --shard-size rows and resumes from them; the final .npz and its sidecar
appear when every shard is done, and each run that computed images adds one compute-log
row (N7).

Exit codes: 0 the cache is complete (computed, resumed to the end, or already there);
1 stopped early (--max-shards), the next run resumes; 2 input error, nothing new written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ms
from ms import compute_log
from ms.cache import (
    BACKBONE_CONFIGS,
    CACHE_ROOT,
    REPO_ROOT,
    STORAGE_DTYPE,
    TOKEN_TYPES,
    CacheMismatchError,
    backbone_config,
    cache_files,
    cache_key,
    load_cache,
    preprocess,
)
from ms.data.manifests import RAW_ROOT, git_sha, load_manifest, manifest_path, repo_relative

#: spec 002 Clarification 8: the working default, to confirm with the W2 timing
SHARD_SIZE = 1024
WORKERS = min(8, os.cpu_count() or 1)
STATE = "state.json"


class InputError(Exception):
    """Exit 2: something about the inputs is wrong; nothing new is written."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- the backbone ------------------------------------------------------------------------------


def hub_cache_dir() -> str | None:
    """The Hub cache as the environment names it now (HF_HUB_CACHE, else HF_HOME/hub).
    huggingface_hub reads these once, at import, which may precede loading .env."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return str(Path(os.environ["HF_HOME"]) / "hub")
    return None


def resolve_weights(cfg: dict[str, Any]) -> Path:
    """The local folder the model is loaded from: hf_id itself when it is a folder (tests),
    else the pinned Hub snapshot, the local copy first (HF_HOME and HF_TOKEN from .env)."""
    local = Path(cfg["hf_id"])
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = {
        "repo_id": cfg["hf_id"],
        "revision": cfg["revision"],
        "allow_patterns": ["*.json", "*.safetensors"],
        "cache_dir": hub_cache_dir(),
    }
    try:
        folder = Path(snapshot_download(**kwargs, local_files_only=True))
        if any(folder.glob("*.safetensors")):
            return folder
    except LocalEntryNotFoundError:
        pass
    return Path(snapshot_download(**kwargs))


def weights_sha256(folder: Path) -> str:
    """FR-003: the sha256 of the weight file; for several files, the sha256 of their sorted
    `<name> <sha256>` lines."""
    files = sorted(folder.glob("*.safetensors"), key=lambda p: p.name)
    if not files:
        raise InputError("bad_config", f"no .safetensors weights in {folder}")
    if len(files) == 1:
        return sha256_file(files[0])
    lines = "".join(f"{p.name} {sha256_file(p)}\n" for p in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


@dataclass
class Backbone:
    model: torch.nn.Module
    device: str
    dtype: str
    embed_dim: int
    n_register: int
    n_patch: int

    def embed(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(cls, meanpatch) in float32 for a batch [B, 3, n, n] (US-1.2): cls is the first
        token of the last hidden state, meanpatch the mean of the last (n/patch)^2 tokens."""
        x = pixels.to(self.device)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.dtype == "float16"
            else contextlib.nullcontext()
        )
        expected = 1 + self.n_register + self.n_patch
        with torch.inference_mode():
            with autocast:
                hidden = self.model(pixel_values=x).last_hidden_state
            hidden = hidden.float()
            if hidden.shape[1] != expected:
                raise InputError(
                    "token_count_mismatch",
                    f"the model returned {hidden.shape[1]} tokens, not 1 + {self.n_register} "
                    f"registers + {self.n_patch} patches = {expected}",
                )
            return hidden[:, 0].cpu(), hidden[:, -self.n_patch :].mean(dim=1).cpu()


def load_backbone(folder: Path, cfg: dict[str, Any], res: int, device: str, dtype: str) -> Backbone:
    """FR-003: AutoModel from the local folder, in float32, with the config's attention."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(folder), dtype=torch.float32, attn_implementation=cfg["attn_implementation"]
    )
    model.eval().to(device)
    patch = int(model.config.patch_size)
    if patch != int(cfg["patch_size"]):
        raise InputError(
            "bad_config", f"patch size {patch} in the weights, {cfg['patch_size']} in the config"
        )
    if int(model.config.hidden_size) != int(cfg["embed_dim"]):
        raise InputError(
            "bad_config",
            f"width {model.config.hidden_size} in the weights, {cfg['embed_dim']} in the config",
        )
    if res % patch:
        raise InputError(
            "bad_config", f"resolution {res} is not a multiple of the patch size {patch}"
        )
    return Backbone(
        model=model,
        device=device,
        dtype=dtype,
        embed_dim=int(model.config.hidden_size),
        n_register=int(getattr(model.config, "num_register_tokens", 0)),
        n_patch=(res // patch) ** 2,
    )


def device_name(device: str) -> str:
    """The CUDA device name, or `CPU (<processor>)` as the compute log's env row names it."""
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device))
    return f"CPU ({platform.processor() or platform.machine()})"


def vram_gb(device: str) -> float | None:
    if device.startswith("cuda"):
        return round(
            torch.cuda.get_device_properties(torch.device(device)).total_memory / 1024**3, 2
        )
    return None


# --- shards (FR-007) ----------------------------------------------------------------------------


def shard_file(folder: Path, i: int) -> Path:
    return folder / f"shard_{i:05d}.npz"


@dataclass
class ShardState:
    """The shards folder's state.json: what the finished shards belong to, and every run."""

    folder: Path
    cache_key: str
    manifest_sha256: str
    shard_size: int
    n_rows: int
    done: set[int] = field(default_factory=set)
    runs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(
        cls, folder: Path, key: str, manifest_sha: str, shard_size: int, n_rows: int
    ) -> ShardState:
        state = cls(folder, key, manifest_sha, shard_size, n_rows)
        path = folder / STATE
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            same = (old["cache_key"], old["manifest_sha256"], old["shard_size"], old["n_rows"]) == (
                key,
                manifest_sha,
                shard_size,
                n_rows,
            )
            if same:
                state.done = {i for i in old["done"] if shard_file(folder, i).exists()}
                state.runs = old["runs"]
                return state
        if folder.exists():  # another key, manifest or shard size: start again
            shutil.rmtree(folder)
        return state

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        doc = {
            "cache_key": self.cache_key,
            "manifest_sha256": self.manifest_sha256,
            "shard_size": self.shard_size,
            "n_rows": self.n_rows,
            "done": sorted(self.done),
            "runs": self.runs,
        }
        _atomic_write(self.folder / STATE, (json.dumps(doc, indent=2) + "\n").encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


# --- one shard --------------------------------------------------------------------------------


def _load(raw_root: Path, n: int, row: dict[str, Any], preprocess_string: str) -> torch.Tensor:
    """Read, check the sha256 (FR-009) and preprocess one manifest row (1-based row n)."""
    path = raw_root / row["path"]
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise InputError(
            "missing_file", f"row {n}: {row['path']} is not under {raw_root}"
        ) from None
    digest = hashlib.sha256(data).hexdigest()
    if digest != row["sha256"]:
        raise InputError(
            "sha256_mismatch",
            f"row {n}: {row['path']} has sha256 {digest}, the manifest says {row['sha256']}",
        )
    try:
        return preprocess(data, preprocess_string)
    except Exception as exc:  # noqa: BLE001 - every row is expected to decode (FR-009)
        raise InputError("decode_error", f"row {n}: {row['path']}: {exc}") from None


def embed_rows(
    backbone: Backbone,
    rows: list[dict[str, Any]],
    first_row: int,
    raw_root: Path,
    preprocess_string: str,
    batch_size: int,
    pool: ThreadPoolExecutor,
) -> tuple[np.ndarray, np.ndarray]:
    """float16 (cls, meanpatch) for consecutive manifest rows; the next batch is read and
    preprocessed by the pool while the current one runs through the backbone."""
    batches = [rows[k : k + batch_size] for k in range(0, len(rows), batch_size)]

    def submit(b: int) -> list[Future]:
        base = first_row + b * batch_size
        return [
            pool.submit(_load, raw_root, base + k + 1, r, preprocess_string)
            for k, r in enumerate(batches[b])
        ]

    cls_parts, mean_parts = [], []
    pending = submit(0)
    for b in range(len(batches)):
        pixels = torch.stack([f.result() for f in pending])
        if b + 1 < len(batches):
            pending = submit(b + 1)
        cls, meanpatch = backbone.embed(pixels)
        cls_parts.append(cls)
        mean_parts.append(meanpatch)
    return (
        torch.cat(cls_parts).numpy().astype(np.float16),
        torch.cat(mean_parts).numpy().astype(np.float16),
    )


# --- the run -------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ms.cache.extract", description=__doc__.split("\n")[0]
    )
    p.add_argument("--backbone", required=True, help="configs/backbones/<backbone_id>.yaml")
    p.add_argument("--res", type=int, required=True)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--dataset", help="manifest id: data/manifests/<id>_v<N>.jsonl")
    which.add_argument("--manifest", type=Path, help="a manifest file")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--config-dir", type=Path, default=BACKBONE_CONFIGS)
    p.add_argument("--device", default=None, help="cuda when present, else cpu")
    p.add_argument("--dtype", choices=("float16", "float32"), default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--workers", type=int, default=WORKERS, help="image-reading threads")
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        return run(args, started)
    except InputError as exc:
        print(f"{exc.reason}: {exc}", flush=True)
        return 2


def _complete(npz: Path, manifest_sha: str) -> bool:
    try:
        load_cache(npz, manifest_sha256=manifest_sha)
    except (CacheMismatchError, OSError, ValueError, KeyError):
        return False
    return True


def run(args: argparse.Namespace, started: float) -> int:
    from ms.backbones.check import load_env

    load_env()  # HF_HOME and HF_TOKEN before huggingface_hub is imported
    try:
        cfg = backbone_config(args.backbone, args.config_dir)
    except FileNotFoundError as exc:
        raise InputError("bad_config", str(exc)) from None
    res = int(args.res)
    if res not in cfg["resolutions"]:
        raise InputError(
            "bad_config", f"{args.backbone} has no resolution {res} ({cfg['resolutions']})"
        )
    preprocess_string = cfg["preprocess_string"][res]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or cfg["dtype"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise InputError("bad_config", "--device cuda, but torch sees no CUDA device")
    if not device.startswith("cuda") and dtype != "float32":
        raise InputError(
            "bad_config", f"{dtype} means CUDA autocast; on {device} pass --dtype float32"
        )
    batch_size = int(args.batch_size or cfg["batch_size"])
    shard_size = int(args.shard_size or SHARD_SIZE)
    raw_root, cache_root = Path(args.raw_root), Path(args.cache_root)

    manifest_file = Path(args.manifest) if args.manifest else manifest_path(args.dataset)
    if not manifest_file.exists():
        raise InputError("missing_file", f"no manifest {manifest_file} (run make manifests)")
    manifest = load_manifest(manifest_file, None, "extract")
    rows = manifest.rows
    if not rows:
        raise InputError("bad_value", f"{manifest_file.name} has no rows")
    dataset = rows[0]["dataset"]
    for n, r in enumerate(rows, 1):
        if not (raw_root / r["path"]).is_file():
            raise InputError("missing_file", f"row {n}: {r['path']} is not under {raw_root}")

    weights = resolve_weights(cfg)
    weights_sha = weights_sha256(weights)
    key = cache_key(args.backbone, weights_sha, res, preprocess_string, dtype)
    npz, side, shards = cache_files(cache_root, args.backbone, res, key, manifest.name)
    if _complete(npz, manifest.sha256):
        print(f"cache complete, nothing to do: {repo_relative(npz)}", flush=True)
        return 0
    if cfg.get("research_only_until_c5") and not cache_root.resolve().is_relative_to(
        (REPO_ROOT / "data").resolve()
    ):
        raise InputError(
            "research_only", f"{args.backbone} is research-only until C5: caches stay under data/"
        )

    n_shards = math.ceil(len(rows) / shard_size)
    state = ShardState.open(shards, key, manifest.sha256, shard_size, len(rows))
    done_before = len(state.done)
    todo = [i for i in range(n_shards) if i not in state.done]
    if args.max_shards is not None:
        todo = todo[: max(0, args.max_shards)]
    print(
        f"{args.backbone} @ {res} ({dtype}, {device}) on {manifest_file.name}: {len(rows)} rows, "
        f"{n_shards} shard(s) of {shard_size}, {done_before} done, {len(todo)} to do; key {key}",
        flush=True,
    )

    backbone = load_backbone(weights, cfg, res, device, dtype)
    # this run's entry in state.runs: added with its first shard, so that a run killed
    # later still counts its time and images (FR-007; the sidecar sums the runs)
    run_record = {"started_at": _now(), "wallclock_s": 0.0, "images": 0}
    images = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i in todo:
            t0 = time.perf_counter()
            part = rows[i * shard_size : (i + 1) * shard_size]
            cls, meanpatch = embed_rows(
                backbone, part, i * shard_size, raw_root, preprocess_string, batch_size, pool
            )
            _atomic_npz(shard_file(shards, i), cls=cls, meanpatch=meanpatch)
            if not images:
                state.runs.append(run_record)
            images += len(part)
            state.done.add(i)
            run_record.update(wallclock_s=round(time.perf_counter() - started, 3), images=images)
            state.save()
            dt = max(time.perf_counter() - t0, 1e-9)
            print(
                f"  shard {i + 1}/{n_shards}: {len(part)} images, {dt:.1f} s "
                f"({len(part) / dt:.1f} images/s)",
                flush=True,
            )

    complete = len(state.done) == n_shards
    if complete:
        cls = np.concatenate([np.load(shard_file(shards, i))["cls"] for i in range(n_shards)])
        meanpatch = np.concatenate(
            [np.load(shard_file(shards, i))["meanpatch"] for i in range(n_shards)]
        )
        _atomic_npz(
            npz,
            cls=cls,
            meanpatch=meanpatch,
            image_id=np.array([r["image_id"] for r in rows]),
            sha256=np.array([r["sha256"] for r in rows]),
        )
    wallclock = round(time.perf_counter() - started, 3)
    if images:
        run_record.update(wallclock_s=wallclock, images=images)
    if complete:
        total = round(sum(r["wallclock_s"] for r in state.runs), 3)
        meta = {
            "meta_json_version": 1,
            "cache_key": key,
            "backbone_id": args.backbone,
            "weights_sha256": weights_sha,
            "resolution": res,
            "preprocess_string": preprocess_string,
            "compute_dtype": dtype,
            "hf_id": cfg["hf_id"],
            "revision": cfg["revision"],
            "model_class": type(backbone.model).__name__,
            "attn_implementation": cfg["attn_implementation"],
            "research_only_until_c5": bool(cfg.get("research_only_until_c5")),
            "token_types": list(TOKEN_TYPES),
            "storage_dtype": STORAGE_DTYPE,
            "embed_dim": backbone.embed_dim,
            "n_images": len(rows),
            "n_patch_tokens": backbone.n_patch,
            "n_register_tokens": backbone.n_register,
            "shapes": {t: [len(rows), backbone.embed_dim] for t in TOKEN_TYPES},
            "manifest": manifest_file.name,
            "manifest_sha256": manifest.sha256,
            "dataset": dataset,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "device": device_name(device),
            "batch_size": batch_size,
            "shard_size": shard_size,
            "wallclock_s": total,
            "images_per_s": round(len(rows) / total, 3) if total else None,
            "runs": len(state.runs),
            "npz_sha256": sha256_file(npz),
            "created_at": _now(),
            "builder": {
                "module": "ms.cache.extract",
                "version": ms.__version__,
                "git_sha": git_sha(),
            },
        }
        _atomic_write(side, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        shutil.rmtree(shards, ignore_errors=True)
    elif images:
        state.save()

    if images:
        compute_log.log_row(
            step="extract",
            backbone_id=args.backbone,
            res=res,
            dataset=dataset,
            n_images_or_runs=images,
            wallclock_s=wallclock,
            device=device_name(device),
            vram_gb=vram_gb(device),
            notes=f"{manifest_file.name}, {dtype}, batch {batch_size}"
            + ("" if complete else f"; stopped after {len(todo)} shard(s), resumable"),
            extra={
                "cache": {
                    "cache_key": key,
                    "manifest": manifest_file.name,
                    "manifest_sha256": manifest.sha256,
                    "compute_dtype": dtype,
                    "batch_size": batch_size,
                    "shard_size": shard_size,
                    "images_per_s": round(images / wallclock, 3) if wallclock else None,
                    "shards_done_before": done_before,
                    "complete": complete,
                }
            },
            path=args.compute_log,
        )
    where = repo_relative(npz) if complete else repo_relative(shards)
    print(
        f"{'complete' if complete else 'stopped, resumable'}: {where}; {images} images computed "
        f"in {wallclock:.1f} s",
        flush=True,
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
