"""The feature cache - spec 002 (model_service/specs/002-cache/spec.md).

    data/cache/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.npz
        cls and meanpatch, float16 [N, D]; image_id and sha256 [N]; manifest row order
    data/cache/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.meta.json
        the sidecar (FR-006)

The key is (backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype) and
sits in the path, so a changed key never overwrites a cache (DECISIONS 22-23). The command
line is `python -m ms.cache.extract` (make cache); consumers use `find_cache` + `load_cache`.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

__all__ = [
    "CacheMismatchError",
    "Cache",
    "cache_key",
    "find_cache",
    "load_cache",
    "preprocess",
]

#: repository root: model_service/src/ms/cache/__init__.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_ROOT = REPO_ROOT / "data" / "cache"
BACKBONE_CONFIGS = REPO_ROOT / "model_service" / "configs" / "backbones"

#: FR-002, in this order
KEY_FIELDS = ("backbone_id", "weights_sha256", "resolution", "preprocess_string", "compute_dtype")
TOKEN_TYPES = ("cls", "meanpatch")
STORAGE_DTYPE = "float16"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class CacheMismatchError(Exception):
    """A cache that does not match its sidecar, its key, or the manifest asked for (US-4.2)."""


def cache_key(
    backbone_id: str,
    weights_sha256: str,
    resolution: int,
    preprocess_string: str,
    compute_dtype: str,
) -> str:
    """FR-002: the first 16 hex digits of sha256(json.dumps([the five fields]))."""
    fields = [backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype]
    return hashlib.sha256(json.dumps(fields).encode("utf-8")).hexdigest()[:16]


# --- preprocessing (FR-004) -------------------------------------------------------------------


def parse_preprocess(string: str) -> list[tuple[str, Any]]:
    """The `;`-separated name=value tokens; ValueError on anything FR-004 does not define."""
    steps: list[tuple[str, Any]] = []
    for token in string.split(";"):
        name, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"preprocess token {token!r} is not name=value")
        if name in ("resize_short", "center_crop"):
            if not value.isdigit() or int(value) < 1:
                raise ValueError(f"{name}={value!r}: a positive integer is required")
            steps.append((name, int(value)))
        elif name == "norm":
            if value != "imagenet":
                raise ValueError(f"norm={value!r}: only norm=imagenet is defined")
            steps.append((name, value))
        else:
            raise ValueError(f"unknown preprocess token {name!r}")
    return steps


def _rgb(image: Image.Image) -> torch.Tensor:
    """EXIF orientation applied, RGB, uint8 [3, H, W]."""
    upright = ImageOps.exif_transpose(image).convert("RGB")
    return torch.from_numpy(np.array(upright, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def preprocess(image: Image.Image | bytes | str | Path, preprocess_string: str) -> torch.Tensor:
    """FR-004: decode, apply the EXIF orientation, RGB, uint8; then the tokens in order.

    `resize_short=n` (shorter side to n, bicubic, antialias), `center_crop=n`,
    `norm=imagenet`. Returns float32 [3, n, n]. The cache and the service call this one
    function, so the string alone decides what the backbone sees.
    """
    steps = parse_preprocess(preprocess_string)
    if isinstance(image, Image.Image):
        x = _rgb(image)
    elif isinstance(image, bytes | bytearray | memoryview):
        with Image.open(io.BytesIO(image)) as im:
            x = _rgb(im)
    else:
        with Image.open(Path(image)) as im:
            x = _rgb(im)
    for name, value in steps:
        if name == "resize_short":
            x = TF.resize(x, [value], interpolation=InterpolationMode.BICUBIC, antialias=True)
        elif name == "center_crop":
            x = TF.center_crop(x, [value, value])
        else:  # norm=imagenet
            x = (x.to(torch.float32) / 255 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[
                :, None, None
            ]
    return x.to(torch.float32)


# --- configs and locations --------------------------------------------------------------------


def backbone_config(backbone_id: str, config_dir: Path = BACKBONE_CONFIGS) -> dict[str, Any]:
    path = Path(config_dir) / f"{backbone_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no backbone config {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cache_files(
    cache_root: Path, backbone_id: str, res: int, key: str, manifest_name: str
) -> tuple[Path, Path, Path]:
    """(npz, sidecar, shards folder) for manifest `<manifest>_v<N>` (FR-001)."""
    folder = Path(cache_root) / backbone_id / str(res) / key
    return (
        folder / f"{manifest_name}.npz",
        folder / f"{manifest_name}.meta.json",
        folder / f"{manifest_name}.shards",
    )


def sidecar_of(npz: Path) -> Path:
    return npz.with_name(npz.stem + ".meta.json")


# --- loading (FR-010) -------------------------------------------------------------------------


@dataclass
class Cache:
    path: Path
    meta: dict[str, Any]
    image_id: np.ndarray
    sha256: np.ndarray
    cls: np.ndarray
    meanpatch: np.ndarray

    def tokens(self, token_type: str) -> np.ndarray:
        """float16 [N, D] for "cls" or "meanpatch"."""
        if token_type not in TOKEN_TYPES:
            raise ValueError(f"token type must be one of {TOKEN_TYPES}, got {token_type!r}")
        return getattr(self, token_type)


def load_cache(npz: Path | str, manifest_sha256: str | None = None) -> Cache:
    """The arrays and the sidecar; CacheMismatchError when the arrays do not match the
    sidecar's shapes or dtype, the sidecar's key fields do not give its cache_key, or the
    cache was built from another manifest than `manifest_sha256`."""
    npz = Path(npz)
    side = sidecar_of(npz)
    if not npz.exists() or not side.exists():
        raise CacheMismatchError(f"{npz} or its sidecar is missing")
    meta = json.loads(side.read_text(encoding="utf-8"))
    try:
        key = cache_key(**{k: meta[k] for k in KEY_FIELDS})
    except KeyError as exc:
        raise CacheMismatchError(f"{side.name} lacks the key field {exc}") from None
    if key != meta.get("cache_key"):
        raise CacheMismatchError(
            f"{side.name}: the key fields give {key}, the sidecar says {meta.get('cache_key')}"
        )
    if manifest_sha256 is not None and manifest_sha256 != meta.get("manifest_sha256"):
        raise CacheMismatchError(
            f"{npz.name} was built from manifest sha256 {meta.get('manifest_sha256')}, "
            f"not {manifest_sha256}"
        )
    with np.load(npz, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    for token in meta.get("token_types", TOKEN_TYPES):
        a = arrays.get(token)
        shape = meta.get("shapes", {}).get(token)
        if a is None or list(a.shape) != list(shape or []):
            raise CacheMismatchError(f"{npz.name}: {token} is not of shape {shape}")
        if a.dtype != np.dtype(meta.get("storage_dtype", STORAGE_DTYPE)):
            raise CacheMismatchError(
                f"{npz.name}: {token} is {a.dtype}, not {meta['storage_dtype']}"
            )
    for name in ("image_id", "sha256"):
        if name not in arrays or len(arrays[name]) != meta.get("n_images"):
            raise CacheMismatchError(f"{npz.name}: {name} does not hold n_images entries")
    return Cache(
        npz, meta, arrays["image_id"], arrays["sha256"], arrays["cls"], arrays["meanpatch"]
    )


def find_cache(
    cache_root: Path | str,
    backbone_id: str,
    res: int,
    manifest_name: str,
    manifest_sha256: str,
    *,
    key: str | None = None,
) -> Path:
    """The one cache of manifest `<manifest>_v<N>` (these bytes) for a backbone and resolution.

    FileNotFoundError when there is none; CacheMismatchError when several keys match (other
    weights, preprocessing or dtype): name one with `key`.
    """
    folder = Path(cache_root) / backbone_id / str(res)
    found = []
    for npz in sorted(folder.glob(f"*/{manifest_name}.npz")):
        if key is not None and npz.parent.name != key:
            continue
        side = sidecar_of(npz)
        if side.exists():
            meta = json.loads(side.read_text(encoding="utf-8"))
            if meta.get("manifest_sha256") == manifest_sha256:
                found.append(npz)
    if not found:
        raise FileNotFoundError(
            f"no cache of {manifest_name} (sha256 {manifest_sha256[:12]}) for {backbone_id} "
            f"at {res} under {folder}: run make cache BB={backbone_id} RES={res}"
        )
    if len(found) > 1:
        keys = ", ".join(p.parent.name for p in found)
        raise CacheMismatchError(f"{len(found)} caches of {manifest_name} match ({keys}); pick one")
    return found[0]
