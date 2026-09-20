"""The model service (S4.6; specs/006-service/spec.md).

    make serve        # uvicorn ms.service.app:app on 127.0.0.1:8000
    uv run python -m ms.service.example --manifest data/manifests/ibean_v1.jsonl > request.json
    curl -s -H "content-type: application/json" -d @request.json http://127.0.0.1:8000/v1/predict

Four endpoints (H8 §6.2): `POST /v1/predict`, `POST /v1/predict_batch` (replay mode),
`GET /v1/model` (the card summary) and `GET /v1/health`.

A request is piece 1's observation_frame record (interface v0) + `request_id` + `options`
(DECISIONS 28), checked against `interface/schema/v0/observation_frame.schema.json` itself.
A contract violation is HTTP 422 whose body still reads as a response — `decision` abstain,
`abstain_reason` `invalid_input`, and the validator's reasons — so one consumer branch reads
both outcomes (spec 006 US-2.2).

The bytes at `frame.uri` are hashed and the hash must equal `frame.sha256` (DECISIONS 32).
A hash the served run's cache holds is answered without the backbone; anything else is
embedded on the spot and rounded to float16 exactly as the cache stores it, so the replay
path and the live path are one model (US-4.2). `decision`, `abstain_reason` and the
temperature-scaled `scores` are `ms.abstain`'s, at the coverage `options.coverage_target`
names; the service fits nothing (US-5.1). Every request and response, 422s included, is one
line of the request log for piece 3 (US-9).

The served model is `model_service/configs/service.yaml`'s, not "the newest run": with
hundreds of runs on disk, newest is not a choice anybody made (US-10).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

import ms
from ms.abstain import AbstainError, Fit, coverage_key, decide, load_fit, probabilities
from ms.abstain import scores as abstain_scores
from ms.cache import BACKBONE_CONFIGS, CACHE_ROOT, Cache, backbone_config, load_cache, preprocess
from ms.cache.extract import load_backbone, resolve_weights
from ms.data.manifests import RAW_ROOT
from ms.heads import (
    HEAD_CONFIGS,
    HEADS_ROOT,
    Run,
    features,
    features_from_tokens,
    list_runs,
    load_run,
)
from ms.service.log import REQUEST_LOG, log_request, row_for

#: repository root: model_service/src/ms/service/app.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = REPO_ROOT / "interface" / "schema" / "v0" / "observation_frame.schema.json"
SERVICE_CONFIG = REPO_ROOT / "model_service" / "configs" / "service.yaml"
CARDS = REPO_ROOT / "model_service" / "cards"
RECORD_KEYS = ("contract_version", "frame", "metadata")
REQUEST_KEYS = ("request_id", "options")
OPTION_KEYS = ("coverage_target", "return_patch_map")
#: FR-003, in the order a response is written
RESPONSE_FIELDS = (
    "request_id",
    "frame_uid",
    "model_version",
    "backbone",
    "head",
    "scores",
    "top1",
    "decision",
    "abstain_reason",
    "uncertainty",
    "conformal_set",
    "localisation",
    "features",
    "timing_ms",
    "warnings",
)
#: what names a run when the config does not give a run_id outright (US-10.1).
#: head_config_sha256 pins the recipe: without it "linear" also matches the runs of a
#: retired recipe, which the N-table marks superseded and nobody may serve (DECISIONS 80)
RUN_KEYS = ("backbone_id", "res", "tokens", "head", "seed", "head_config_sha256")


def _under_repo(path: Path | str | None) -> str | None:
    """A path as the repository sees it, or as it is when it lies outside (a test's tmp)."""
    if path is None:
        return None
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class ServiceError(RuntimeError):
    """A configuration the service refuses to start on (FR-013); `reason` is its vocabulary."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass
class Settings:
    run: str | None = None
    heads_root: Path = HEADS_ROOT
    cache_root: Path = CACHE_ROOT
    data_root: Path = RAW_ROOT
    schema: Path = SCHEMA
    backbone_configs: Path = BACKBONE_CONFIGS
    head_configs: Path = HEAD_CONFIGS
    request_log: Path = REQUEST_LOG
    service: Path | None = None
    device: str = "auto"
    max_batch: int = 32
    default_coverage: float | None = None
    #: the descriptor of US-10.1 when the config names no run_id
    select: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, path: Path | str) -> Settings:
        """FR-012: configs/service.yaml. Paths in it are relative to the repository root."""
        path = Path(path)
        if not path.is_file():
            raise ServiceError("bad_config", f"no service config at {path}")
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ServiceError("bad_config", f"{path}: {exc}") from None
        if not isinstance(cfg, dict):
            raise ServiceError("bad_config", f"{path} is not a mapping")

        def where(key: str, default: Path) -> Path:
            value = cfg.get(key)
            if value is None:
                return default
            p = Path(value)
            return p if p.is_absolute() else REPO_ROOT / p

        select = {k: cfg[k] for k in (*RUN_KEYS, "train_manifests") if cfg.get(k) is not None}
        return cls(
            run=cfg.get("run_id") or None,
            heads_root=where("heads_root", HEADS_ROOT),
            cache_root=where("cache_root", CACHE_ROOT),
            data_root=where("data_root", RAW_ROOT),
            backbone_configs=where("backbone_configs", BACKBONE_CONFIGS),
            head_configs=where("head_configs", HEAD_CONFIGS),
            request_log=where("request_log", REQUEST_LOG),
            service=path,
            device=str(cfg.get("device", "auto")),
            max_batch=int(cfg.get("max_batch", 32)),
            default_coverage=(
                None if cfg.get("default_coverage") is None else float(cfg["default_coverage"])
            ),
            select=select,
        )

    @classmethod
    def from_env(cls) -> Settings:
        """The config file, then the environment over it (FR-012). A missing file is not an
        error here: the start-up checks say what is wrong, with their own vocabulary."""
        env = os.environ
        path = Path(env.get("MS_SERVICE_CONFIG", SERVICE_CONFIG))
        settings = cls.from_config(path) if path.is_file() else cls(service=None)
        for key, name in (
            ("run", "MS_HEAD_RUN"),
            ("heads_root", "MS_HEADS_ROOT"),
            ("cache_root", "MS_CACHE_ROOT"),
            ("data_root", "MS_DATA_ROOT"),
            ("request_log", "MS_REQUEST_LOG"),
        ):
            value = env.get(name)
            if value:
                setattr(settings, key, value if key == "run" else Path(value))
        return settings


class Served:
    """The one model the service answers with (US-10): its run, its fit, the cached features
    of its cache key by image sha256, and its backbone when there is one to load."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.folder = self._folder(settings)
        try:
            self.run: Run = load_run(self.folder)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ServiceError("missing_run", f"{self.folder} does not load: {exc}") from None
        try:
            self.fit: Fit = load_fit(self.folder, cache_root=settings.cache_root)
        except AbstainError as exc:
            reason = "missing_abstain" if exc.reason == "missing_fit" else exc.reason
            raise ServiceError(
                reason,
                f"{self.run.run_id}: {exc}. A service that cannot abstain is not the service "
                "H8 §6.11 item 1 asks for (spec 006 US-5.4)",
            ) from None
        self.coverage = self._coverage(settings)

        bb = self.run.meta["backbone"]
        key_dir = Path(settings.cache_root) / bb["backbone_id"] / str(bb["res"]) / bb["cache_key"]
        self.caches: list[Cache] = [load_cache(npz) for npz in sorted(key_dir.glob("*.npz"))]
        if not self.caches:
            raise ServiceError("missing_cache", f"no cache under {key_dir}")
        self.index: dict[str, tuple[int, int]] = {}
        for i, cache in enumerate(self.caches):
            for j, digest in enumerate(cache.sha256):
                # first cache in name order wins: under one cache_key the features are the
                # same image's, so which one answers cannot change an answer (US-4.6)
                self.index.setdefault(str(digest), (i, j))

        self.backbone_cfg = self._backbone_cfg(settings)
        self._backbone = None
        Path(settings.request_log).parent.mkdir(parents=True, exist_ok=True)

    # --- what the service is configured with --------------------------------------------

    @staticmethod
    def _folder(settings: Settings) -> Path:
        """US-10.1: the run the config names. Never "the newest run" (DECISIONS 37)."""
        if settings.run:
            for folder in (Path(settings.run), Path(settings.heads_root) / settings.run):
                if (folder / "run.json").is_file():
                    return folder
            raise ServiceError(
                "missing_run",
                f"{settings.run}: no run.json there or under {settings.heads_root}",
            )
        if not settings.select:
            raise ServiceError(
                "bad_config",
                "no run is named: configs/service.yaml gives a run_id, or the backbone, "
                "resolution, tokens, head, seed and training manifests that pick one "
                "(spec 006 US-10.1)",
            )
        select = Served._select(settings)
        matches = [p for p in list_runs(settings.heads_root) if Served._matches(p, select)]
        if not matches:
            raise ServiceError(
                "missing_run", f"no run under {settings.heads_root} matches {select}"
            )
        if len(matches) > 1:
            raise ServiceError(
                "ambiguous_run",
                f"{len(matches)} runs match {select}: " + ", ".join(p.name for p in matches),
            )
        return matches[0]

    @staticmethod
    def _select(settings: Settings) -> dict[str, Any]:
        """The descriptor, with the head recipe filled in when the config leaves it out: a
        head's name is the recipe the repository holds now, never a retired one (US-10.2)."""
        select = dict(settings.select)
        head = select.get("head")
        if head and "head_config_sha256" not in select:
            recipe = Path(settings.head_configs) / f"{head}.yaml"
            if recipe.is_file():
                select["head_config_sha256"] = hashlib.sha256(recipe.read_bytes()).hexdigest()
        return select

    @staticmethod
    def _matches(folder: Path, select: dict[str, Any]) -> bool:
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        flat = {
            **meta,
            **{k: meta["backbone"][k] for k in ("backbone_id", "res")},
            "head_config_sha256": meta["head_config"]["sha256"],
        }
        for key in RUN_KEYS:
            if key in select and flat.get(key) != select[key]:
                return False
        wanted = select.get("train_manifests")
        if wanted is not None:
            have = [Path(p).stem for p in meta["train"]["manifest"].split("+")]
            if have != list(wanted):
                return False
        return True

    def _coverage(self, settings: Settings) -> float:
        wanted = settings.default_coverage
        if wanted is None:
            wanted = self.fit.meta["config"].get("default_coverage")
        try:
            self.fit.thresholds(float(wanted))
        except (AbstainError, TypeError, ValueError):
            raise ServiceError(
                "bad_value",
                f"default coverage {wanted!r} was not fitted for {self.run.run_id} "
                f"(have {', '.join(self.fit.coverages_str)})",
            ) from None
        return float(wanted)

    def _backbone_cfg(self, settings: Settings) -> dict[str, Any] | None:
        """The backbone's recipe, when this deployment has one. Without it the service is
        replay-only: it answers cached frames and 501s the rest (US-4.2, Clarification 17)."""
        bb = self.run.meta["backbone"]
        try:
            cfg = backbone_config(bb["backbone_id"], Path(settings.backbone_configs))
        except (FileNotFoundError, yaml.YAMLError):
            return None
        try:
            folder = resolve_weights(cfg)
        except Exception as exc:  # noqa: BLE001 - a Hub or filesystem failure, either way fatal
            raise ServiceError(
                "missing_weights",
                f"{bb['backbone_id']} is configured in {settings.backbone_configs} but its "
                f"weights do not resolve: {exc}",
            ) from None
        return {**cfg, "weights_folder": folder}

    @property
    def can_compute(self) -> bool:
        return self.backbone_cfg is not None

    @property
    def device(self) -> str:
        wanted = self.settings.device
        if wanted != "auto":
            return wanted
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def backbone(self):
        """The backbone, loaded once (H8 §6.9) and only when a frame needs it: a replayed
        mission never touches it."""
        if self._backbone is None:
            cfg, bb = self.backbone_cfg, self.run.meta["backbone"]
            device = self.device
            dtype = bb.get("compute_dtype", "float32") if device != "cpu" else "float32"
            self._backbone = load_backbone(
                cfg["weights_folder"], cfg, int(bb["res"]), device, dtype
            )
        return self._backbone

    # --- features ------------------------------------------------------------------------

    def cached(self, digest: str) -> np.ndarray | None:
        hit = self.index.get(digest)
        if hit is None:
            return None
        cache_i, row = hit
        return features(self.caches[cache_i], np.array([row]), self.run.tokens)

    def preprocess_images(self, images: list[bytes]):
        """Decode and preprocess, by the run's own preprocess_string: the string alone decides
        what the backbone sees, and it is the cache's string (spec 002 FR-004)."""
        import torch

        recipe = self.run.meta["backbone"]["preprocess_string"]
        return torch.stack([preprocess(data, recipe) for data in images])

    def embed(self, pixels) -> np.ndarray:
        """The features of frames no cache holds: the backbone, then the float16 the cache
        stores, so a computed frame reaches the head as a cached one would (US-4.2)."""
        backbone = self.backbone()
        size = int(self.backbone_cfg.get("batch_size", 32))
        parts = []
        for start in range(0, len(pixels), size):
            cls, meanpatch = backbone.embed(pixels[start : start + size])
            parts.append((cls.numpy(), meanpatch.numpy()))
        by_type = {
            "cls": np.concatenate([p[0] for p in parts]).astype(np.float16),
            "meanpatch": np.concatenate([p[1] for p in parts]).astype(np.float16),
        }
        return features_from_tokens(self.run.tokens, **by_type)

    # --- the card summary (US-7.2) ---------------------------------------------------------

    def card_path(self) -> str | None:
        card = CARDS / f"{self.run.meta['model_version']}.md"
        return _under_repo(card) if card.is_file() else None

    def describe(self) -> dict[str, Any]:
        meta, bb = self.run.meta, self.run.meta["backbone"]
        return {
            "model_version": meta["model_version"],
            "run_id": self.run.run_id,
            "backbone": {
                "id": bb["hf_id"],
                "backbone_id": bb["backbone_id"],
                "revision": bb["revision"],
                "weights_sha256": bb["weights_sha256"],
                "input_res": bb["res"],
                "tokens": self.run.tokens,
                "cache_key": bb["cache_key"],
                "preprocess_string": bb["preprocess_string"],
                "research_only_until_c5": bb["research_only_until_c5"],
            },
            "head": {
                "id": meta["head"],
                "seed": meta["seed"],
                "config_sha256": meta["head_config"]["sha256"],
                "input_dim": meta["input_dim"],
            },
            "train": {
                k: meta["train"][k]
                for k in ("manifest", "manifest_sha256", "split", "role", "frozen", "n")
            },
            "classes": self.run.classes,
            "coverage": self.coverage,
            "coverages": list(self.fit.coverages),
            "thresholds": self.fit.thresholds(self.coverage),
            "temperature": self.fit.temperature,
            "quotable": meta["quotable"],
            "features": "cache+computed" if self.can_compute else "cache",
            "cached_frames": len(self.index),
            "caches": [
                {"manifest": c.meta["manifest"], "n_images": c.meta["n_images"]}
                for c in self.caches
            ],
            "card": self.card_path(),
            "service_config": _under_repo(self.settings.service),
            "notes": meta.get("notes"),
        }


class UriError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def resolve_uri(uri: str, data_root: Path) -> Path:
    """frame.uri -> a file under data_root: a relative path (to data_root), an absolute path,
    or file://. Other schemes (mcap://) are open between pieces 1 and 2 (W3): 501."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = Path(url2pathname(unquote(parsed.path)))
    elif len(parsed.scheme) > 1:
        raise UriError(
            501, f"uri scheme {parsed.scheme}:// is not served yet (open with piece 2, W3)"
        )
    else:  # a path; "D:/x" parses as scheme "d"
        path = Path(uri)
    root = Path(data_root).resolve()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise UriError(422, f"uri resolves outside the data root {root}")
    if not path.is_file():
        raise UriError(422, f"no file at uri {uri}")
    return path


def request_errors(
    body: Any, validator: Draft202012Validator, coverages: list[str] | None = None
) -> list[dict[str, str]]:
    """What is wrong with a request: the record against piece 1's schema, then request_id
    and options (DECISIONS 28; US-2). Empty when the request is valid."""
    if not isinstance(body, dict):
        return [{"path": "", "message": "the request is not a JSON object"}]
    record = {k: body[k] for k in RECORD_KEYS if k in body}
    errors = [
        {"path": ".".join(str(p) for p in e.absolute_path) or "(record)", "message": e.message}
        for e in validator.iter_errors(record)
    ]
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        errors.append({"path": "request_id", "message": "a non-empty string is required"})
    for extra in sorted(set(body) - set(RECORD_KEYS) - set(REQUEST_KEYS)):
        errors.append(
            {"path": extra, "message": "not allowed: request = record + request_id + options"}
        )
    options = body.get("options", {})
    if not isinstance(options, dict):
        errors.append({"path": "options", "message": "an object is required"})
    else:
        for extra in sorted(set(options) - set(OPTION_KEYS)):
            errors.append({"path": f"options.{extra}", "message": "unknown option"})
        target = options.get("coverage_target")
        if target is not None:
            bad = isinstance(target, bool) or not isinstance(target, int | float)
            if bad or (coverages is not None and coverage_key(target) not in coverages):
                have = ", ".join(coverages or [])
                errors.append(
                    {
                        "path": "options.coverage_target",
                        "message": f"coverage {target!r} was not fitted; the fitted ones are "
                        f"{have}. A service does not interpolate an operating point it never "
                        "measured (spec 006 US-2.4)",
                    }
                )
        if not isinstance(options.get("return_patch_map", False), bool):
            errors.append({"path": "options.return_patch_map", "message": "a boolean is required"})
    return sorted(errors, key=lambda e: e["path"])


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _entropy(p: np.ndarray) -> float:
    return round(float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()), 6)


@dataclass
class Prepared:
    """One item of a call, after validation and the hash: either an answer already (a 422 or
    a 501) or the features to score."""

    body: Any
    status: int = 200
    response: dict[str, Any] | None = None
    frame_uid: str | None = None
    coverage: float | None = None
    source: str | None = None  # "cache" or "computed"
    x: np.ndarray | None = None
    image: bytes | None = None
    preprocess_ms: float = 0.0
    backbone_ms: float = 0.0


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    schema = json.loads(Path(settings.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    state: dict[str, Served] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["served"] = Served(settings)
        yield

    app = FastAPI(
        title="agrirobot-disease model service (piece 4)",
        version=ms.__version__,
        description="S4.6: the interface of H8 §6.2 over one registered model.",
        lifespan=lifespan,
    )

    # --- one answer, built in one place (FR-014) -----------------------------------------

    def invalid(request_id: Any, errors: list[dict[str, str]], t0: float) -> dict[str, Any]:
        served = state["served"]
        return {
            "request_id": request_id if isinstance(request_id, str) else None,
            "model_version": served.run.meta["model_version"],
            "decision": "abstain",
            "abstain_reason": "invalid_input",
            "errors": errors,
            "timing_ms": {"total": _ms(t0)},
        }

    def prepare(body: Any, t0: float) -> Prepared:
        """Validate, fetch and hash one item; on the cached path, its features too."""
        served = state["served"]
        errors = request_errors(body, validator, served.fit.coverages_str)
        if errors:
            request_id = body.get("request_id") if isinstance(body, dict) else None
            return Prepared(body, 422, invalid(request_id, errors, t0))
        request_id, frame = body["request_id"], body["frame"]
        coverage = float(body.get("options", {}).get("coverage_target", served.coverage))
        try:
            path = resolve_uri(frame["uri"], settings.data_root)
        except UriError as exc:
            if exc.status == 422:
                return Prepared(
                    body, 422, invalid(request_id, [{"path": "frame.uri", "message": str(exc)}], t0)
                )
            return Prepared(
                body,
                exc.status,
                {"request_id": request_id, "frame_uid": frame["frame_uid"], "detail": str(exc)},
                frame_uid=frame["frame_uid"],
            )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != frame["sha256"]:
            message = f"the bytes at uri hash to {digest}, not frame.sha256"
            return Prepared(
                body, 422, invalid(request_id, [{"path": "frame.sha256", "message": message}], t0)
            )
        prepared = Prepared(body, frame_uid=frame["frame_uid"], coverage=coverage)
        x = served.cached(digest)
        if x is not None:
            # the replay path: no decode and no backbone, which is the point of it (H8 §6.9)
            prepared.x, prepared.source = x, "cache"
            return prepared
        if not served.can_compute:
            prepared.status = 501
            prepared.response = {
                "request_id": request_id,
                "frame_uid": frame["frame_uid"],
                "detail": "frame not in the feature cache, and this deployment has no backbone "
                f"configured for {served.run.meta['backbone']['backbone_id']}: it answers "
                "replayed frames only (spec 006 US-4.2)",
            }
            return prepared
        prepared.image, prepared.source = data, "computed"
        return prepared

    def answer(prepared: Prepared, t0: float) -> dict[str, Any]:
        """The response of US-3 for one prepared item."""
        served = state["served"]
        run, fit, bb = served.run, served.fit, served.run.meta["backbone"]
        body, coverage = prepared.body, float(prepared.coverage)
        t1 = time.perf_counter()
        p = probabilities(fit, run, prepared.x)[0]
        row = {k: float(v[0]) for k, v in abstain_scores(fit, run, prepared.x).items()}
        decision, reason = decide(fit, row, coverage)
        head_ms = _ms(t1)
        tau = fit.thresholds(coverage)

        warnings = []
        if body["metadata"]["bbch"] is None:
            warnings.append("metadata.bbch is null")
        if body.get("options", {}).get("return_patch_map"):
            warnings.append("options.return_patch_map: no localisation in v0 (H8 §6.1)")
        if not run.meta["quotable"]:
            warnings.append("the served head is not quotable: its numbers are never reported")
        if bb["research_only_until_c5"]:
            warnings.append(
                f"{bb['backbone_id']} is research-only until H6 C5 is signed (DECISIONS 13)"
            )
        return {
            "request_id": body["request_id"],
            "frame_uid": prepared.frame_uid,
            "model_version": run.meta["model_version"],
            "backbone": {
                "id": bb["hf_id"],
                "backbone_id": bb["backbone_id"],
                "weights_sha256": bb["weights_sha256"],
                "input_res": bb["res"],
                "tokens": run.tokens,
                "cache_key": bb["cache_key"],
            },
            "head": {
                "id": run.meta["head"],
                "run_id": run.run_id,
                "train_manifest": run.meta["train"]["manifest"],
                "train_manifest_sha256": run.meta["train"]["manifest_sha256"],
                "calibration": {"temperature": fit.temperature},
            },
            "scores": {c: round(float(p[k]), 6) for k, c in enumerate(run.classes)},
            "top1": run.classes[int(p.argmax())],
            "decision": decision,
            "abstain_reason": reason,
            "uncertainty": {
                "max_prob": round(float(p.max()), 6),
                "entropy": _entropy(p),
                "ood_knn": round(row["knn"], 6),
                "ood_knn_threshold": round(float(tau["tau_knn"]), 6),
                "ood_maha": round(row["maha"], 6),
            },
            "conformal_set": None,
            "localisation": None,
            "features": prepared.source,
            "timing_ms": {
                "preprocess": prepared.preprocess_ms,
                "backbone": prepared.backbone_ms,
                "head": head_ms,
                "total": _ms(t0),
            },
            "warnings": warnings,
        }

    def embed_pending(items: list[Prepared]) -> None:
        """US-6.4: every frame of a call that needs the backbone goes through it in one call,
        and each one's timing carries its share of it."""
        served = state["served"]
        pending = [p for p in items if p.image is not None and p.response is None]
        if not pending:
            return
        t1 = time.perf_counter()
        try:
            pixels = served.preprocess_images([p.image for p in pending])
            t2 = time.perf_counter()
            x = served.embed(pixels)
        except Exception as exc:  # noqa: BLE001 - a decode or a shape, both the frame's fault
            for p in pending:
                p.status = 422
                p.response = invalid(
                    p.body.get("request_id"),
                    [{"path": "frame.uri", "message": f"the bytes at uri do not embed: {exc}"}],
                    t1,
                )
            return
        # a call embeds its frames together (US-6.4), so each one carries its share of it
        n = len(pending)
        preprocess_share = round((t2 - t1) * 1000 / n, 3)
        backbone_share = round((time.perf_counter() - t2) * 1000 / n, 3)
        for k, p in enumerate(pending):
            p.x = x[k : k + 1]
            p.preprocess_ms = preprocess_share
            p.backbone_ms = backbone_share

    def record(prepared: Prepared, status: int, response: dict[str, Any]) -> list[str]:
        """US-9: one line per request/response pair. A log that cannot be written is a warning
        on the response, never a 500."""
        served = state["served"]
        try:
            log_request(
                settings.request_log,
                row_for(
                    prepared.body,
                    response,
                    status=status,
                    run_id=served.run.run_id,
                    coverage=prepared.coverage,
                ),
            )
        except OSError as exc:
            return [f"the request log could not be written: {exc}"]
        return []

    def handle(body: Any) -> tuple[int, dict[str, Any]]:
        t0 = time.perf_counter()
        prepared = prepare(body, t0)
        embed_pending([prepared])
        status, response = _finish(prepared, t0)
        return status, response

    def _finish(prepared: Prepared, t0: float) -> tuple[int, dict[str, Any]]:
        status = prepared.status
        response = prepared.response if prepared.response is not None else answer(prepared, t0)
        warnings = record(prepared, status, response)
        if warnings and "warnings" in response:
            response["warnings"] = [*response["warnings"], *warnings]
        return status, response

    async def read_body(request: Request) -> tuple[bool, Any]:
        try:
            return True, json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False, None

    # --- the endpoints (FR-001) ------------------------------------------------------------

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        served = state["served"]
        return {
            "status": "ok",
            "model_version": served.run.meta["model_version"],
            "cached_frames": len(served.index),
        }

    @app.get("/v1/model")
    def model() -> dict[str, Any]:
        return state["served"].describe()

    @app.post("/v1/predict")
    async def predict(request: Request) -> JSONResponse:
        t0 = time.perf_counter()
        ok, body = await read_body(request)
        if not ok:
            response = invalid(None, [{"path": "", "message": "the body is not JSON"}], t0)
            record(Prepared(None), 422, response)
            return JSONResponse(status_code=422, content=response)
        status, response = handle(body)
        return JSONResponse(status_code=status, content=response)

    @app.post("/v1/predict_batch")
    async def predict_batch(request: Request) -> JSONResponse:
        t0 = time.perf_counter()
        ok, body = await read_body(request)
        if not ok:
            response = invalid(None, [{"path": "", "message": "the body is not JSON"}], t0)
            record(Prepared(None), 422, response)
            return JSONResponse(status_code=422, content=response)
        problem = _envelope_error(body, settings.max_batch)
        if problem:
            response = invalid(None, [problem], t0)
            record(Prepared(None), 422, response)
            return JSONResponse(status_code=422, content=response)

        starts = [time.perf_counter() for _ in body["requests"]]
        items = [prepare(item, starts[k]) for k, item in enumerate(body["requests"])]
        embed_pending(items)
        answers, n_invalid = [], 0
        for k, prepared in enumerate(items):
            status, response = _finish(prepared, starts[k])
            n_invalid += status != 200
            answers.append(response)
        return JSONResponse(content={"responses": answers, "n_invalid": n_invalid})

    return app


def _envelope_error(body: Any, max_batch: int) -> dict[str, str] | None:
    """US-6.3: what is wrong with a batch's envelope. The items are each their own business."""
    if not isinstance(body, dict):
        return {"path": "", "message": "the request is not a JSON object"}
    for extra in sorted(set(body) - {"requests"}):
        return {"path": extra, "message": 'not allowed: the batch is {"requests": [...]}'}
    requests = body.get("requests")
    if not isinstance(requests, list):
        return {"path": "requests", "message": "a list of /v1/predict bodies is required"}
    if not requests:
        return {"path": "requests", "message": "at least one request is required"}
    if len(requests) > max_batch:
        return {
            "path": "requests",
            "message": f"{len(requests)} requests, more than max_batch = {max_batch}",
        }
    return None


app = create_app()
