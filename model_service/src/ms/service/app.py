"""The model service (S4.6, W2 slice version; spec 006 in W5): POST /v1/predict from the cache.

    make serve        # uvicorn ms.service.app:app on 127.0.0.1:8000
    uv run python -m ms.service.example --manifest data/manifests/ibean_v1.jsonl > request.json
    curl -s -H "content-type: application/json" -d @request.json http://127.0.0.1:8000/v1/predict

Serves one head run: MS_HEAD_RUN (a run_id under MS_HEADS_ROOT, or a run folder), else the
newest run under data/heads. A request is piece 1's observation_frame record (interface v0)
+ request_id + options (DECISIONS 28). The record is checked against
interface/schema/v0/observation_frame.schema.json. A broken one gets HTTP 422 with
abstain_reason invalid_input and the validator's reasons (H8 6.2). The bytes at frame.uri
(a path under MS_DATA_ROOT, default data/raw, or a file:// URI inside it) are hashed, and
the hash must equal frame.sha256 (DECISIONS 32). A hash found in the caches of the head's
cache key is answered from the cached features, without a GPU.

The rest comes in W4 and W5. Features computed on request: an uncached frame gets HTTP 501.
Also predict_batch, the request log, calibration and abstention: until S4.4, scores are an
uncalibrated softmax and decision is always predict.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

import ms
from ms.cache import CACHE_ROOT, Cache, load_cache
from ms.data.manifests import RAW_ROOT
from ms.heads import HEADS_ROOT, Run, features, list_runs, load_run

#: repository root: model_service/src/ms/service/app.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = REPO_ROOT / "interface" / "schema" / "v0" / "observation_frame.schema.json"
RECORD_KEYS = ("contract_version", "frame", "metadata")
REQUEST_KEYS = ("request_id", "options")
OPTION_KEYS = ("coverage_target", "return_patch_map")


@dataclass
class Settings:
    run: str | None = None
    heads_root: Path = HEADS_ROOT
    cache_root: Path = CACHE_ROOT
    data_root: Path = RAW_ROOT
    schema: Path = SCHEMA

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            run=env.get("MS_HEAD_RUN") or None,
            heads_root=Path(env.get("MS_HEADS_ROOT", HEADS_ROOT)),
            cache_root=Path(env.get("MS_CACHE_ROOT", CACHE_ROOT)),
            data_root=Path(env.get("MS_DATA_ROOT", RAW_ROOT)),
        )


class Served:
    """The head that answers, and the cached features of its cache key, by image sha256."""

    def __init__(self, settings: Settings) -> None:
        self.run: Run = load_run(self._folder(settings))
        bb = self.run.meta["backbone"]
        key_dir = Path(settings.cache_root) / bb["backbone_id"] / str(bb["res"]) / bb["cache_key"]
        self.caches: list[Cache] = [load_cache(npz) for npz in sorted(key_dir.glob("*.npz"))]
        self.index: dict[str, tuple[int, int]] = {}
        for i, cache in enumerate(self.caches):
            for j, digest in enumerate(cache.sha256):
                self.index.setdefault(str(digest), (i, j))

    @staticmethod
    def _folder(settings: Settings) -> Path:
        if settings.run:
            for folder in (Path(settings.run), Path(settings.heads_root) / settings.run):
                if (folder / "run.json").is_file():
                    return folder
            raise RuntimeError(
                f"MS_HEAD_RUN={settings.run}: no run.json there or under {settings.heads_root}"
            )
        runs = list_runs(settings.heads_root)
        if not runs:
            raise RuntimeError(
                f"no head run under {settings.heads_root}: train one with ms.heads.train"
            )
        return runs[-1]

    def describe(self) -> dict[str, Any]:
        meta, bb = self.run.meta, self.run.meta["backbone"]
        return {
            "model_version": meta["model_version"],
            "run_id": self.run.run_id,
            "head": meta["head"],
            "classes": self.run.classes,
            "tokens": self.run.tokens,
            "backbone": {
                k: bb[k]
                for k in ("backbone_id", "hf_id", "revision", "res", "cache_key", "weights_sha256")
            },
            "train": {
                k: meta["train"][k]
                for k in ("manifest", "manifest_sha256", "split", "role", "frozen")
            },
            "fit": {k: meta["fit"][k] for k in ("best_epoch", "epochs_run", "val_macro_f1")},
            "quotable": meta["quotable"],
            "notes": meta.get("notes"),
            "caches": [
                {"manifest": c.meta["manifest"], "n_images": c.meta["n_images"]}
                for c in self.caches
            ],
            "research_only_until_c5": bb["research_only_until_c5"],
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


def request_errors(body: Any, validator: Draft202012Validator) -> list[dict[str, str]]:
    """What is wrong with a request: the record against piece 1's schema, then request_id
    and options (DECISIONS 28). Empty when the request is valid."""
    if not isinstance(body, dict):
        return [{"path": "", "message": "the request is not a JSON object"}]
    record = {k: body[k] for k in RECORD_KEYS if k in body}
    errors = [
        {"path": ".".join(str(p) for p in e.absolute_path) or "(record)", "message": e.message}
        for e in sorted(
            validator.iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path]
        )
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
        target = options.get("coverage_target", 0.9)
        if isinstance(target, bool) or not isinstance(target, int | float) or not 0 < target <= 1:
            errors.append(
                {"path": "options.coverage_target", "message": "a number in (0, 1] is required"}
            )
        if not isinstance(options.get("return_patch_map", False), bool):
            errors.append({"path": "options.return_patch_map", "message": "a boolean is required"})
    return errors


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


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
        description="S4.6, W2 slice: POST /v1/predict from cached features.",
        lifespan=lifespan,
    )

    def invalid(request_id: Any, errors: list[dict[str, str]], t0: float) -> JSONResponse:
        served = state["served"]
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id if isinstance(request_id, str) else None,
                "model_version": served.run.meta["model_version"],
                "decision": "abstain",
                "abstain_reason": "invalid_input",
                "errors": errors,
                "timing_ms": {"total": _ms(t0)},
            },
        )

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
        served = state["served"]
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return invalid(None, [{"path": "", "message": "the body is not JSON"}], t0)
        errors = request_errors(body, validator)
        if errors:
            return invalid(body.get("request_id") if isinstance(body, dict) else None, errors, t0)
        request_id, frame = body["request_id"], body["frame"]
        try:
            path = resolve_uri(frame["uri"], settings.data_root)
        except UriError as exc:
            if exc.status == 422:
                return invalid(request_id, [{"path": "frame.uri", "message": str(exc)}], t0)
            return JSONResponse(
                status_code=exc.status,
                content={
                    "request_id": request_id,
                    "frame_uid": frame["frame_uid"],
                    "detail": str(exc),
                },
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != frame["sha256"]:
            message = f"the bytes at uri hash to {digest}, not frame.sha256"
            return invalid(request_id, [{"path": "frame.sha256", "message": message}], t0)
        hit = served.index.get(digest)
        if hit is None:
            return JSONResponse(
                status_code=501,
                content={
                    "request_id": request_id,
                    "frame_uid": frame["frame_uid"],
                    "detail": "frame not in the feature cache; features computed on request come "
                    "with S4.6 (W5)",
                },
            )

        t1 = time.perf_counter()
        cache_i, row = hit
        p = served.run.predict_proba(
            features(served.caches[cache_i], np.array([row]), served.run.tokens)
        )[0]
        head_ms = _ms(t1)
        run, bb = served.run, served.run.meta["backbone"]
        warnings = []
        if body["metadata"]["bbch"] is None:
            warnings.append("metadata.bbch is null")
        if body.get("options", {}).get("return_patch_map"):
            warnings.append("options.return_patch_map: no localisation in v0 (H8 6.1)")
        if not run.meta["quotable"]:
            warnings.append("the head was trained on a test-only manifest (W2 slice): never quoted")
        warnings.append(
            "no calibration or abstention yet (S4.4, W4): uncalibrated softmax, decision predict"
        )
        return JSONResponse(
            content={
                "request_id": request_id,
                "frame_uid": frame["frame_uid"],
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
                    "calibration": None,
                },
                "scores": {c: round(float(p[k]), 6) for k, c in enumerate(run.classes)},
                "top1": run.classes[int(p.argmax())],
                "decision": "predict",
                "abstain_reason": None,
                "uncertainty": {
                    "max_prob": round(float(p.max()), 6),
                    "entropy": round(float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()), 6),
                    "ood_knn": None,
                    "ood_knn_threshold": None,
                    "ood_maha": None,
                },
                "conformal_set": None,
                "localisation": None,
                "features": "cache",
                "timing_ms": {
                    "preprocess": 0.0,
                    "backbone": 0.0,
                    "head": head_ms,
                    "total": _ms(t0),
                },
                "warnings": warnings,
            }
        )

    return app


app = create_app()
