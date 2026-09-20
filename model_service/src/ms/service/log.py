"""The request log (S4.6; specs/006-service/spec.md US-9) — one JSON line per
request/response pair, the 422s included, because a contract violation is the signal piece 3
wants (H8 §6.2).

    data/predictions/requests.jsonl        (git-ignored; MS_REQUEST_LOG, or service.yaml)

It lives under `data/` and not under `results/`: it grows by one line per frame of every
replayed mission. It holds the frame's hash, never its bytes and never its features. Piece 3
moves these rows into the predictions table when it has one, so the field names are spec
006 US-9.1's and do not change here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: repository root: model_service/src/ms/service/log.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
REQUEST_LOG = REPO_ROOT / "data" / "predictions" / "requests.jsonl"

#: US-9.1, in the order a row is written
FIELDS = (
    "ts",
    "request_id",
    "frame_uid",
    "frame_sha256",
    "uri",
    "site_id",
    "plot_id",
    "zone_id",
    "timestamp_utc",
    "platform",
    "bbch",
    "bbch_null_reason",
    "model_version",
    "run_id",
    "coverage",
    "decision",
    "abstain_reason",
    "top1",
    "scores",
    "uncertainty",
    "features",
    "status",
    "timing_ms",
    "errors",
    "warnings",
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_for(
    body: Any,
    response: dict[str, Any],
    *,
    status: int,
    run_id: str | None,
    coverage: float | None,
) -> dict[str, Any]:
    """One log row from the request body and the answer, whichever of the two is partial.

    A 422 carries no frame and no scores; a broken body carries no metadata at all. Every
    field is still written, so that piece 3 reads one shape (US-9.1).
    """
    body = body if isinstance(body, dict) else {}
    frame = body.get("frame") if isinstance(body.get("frame"), dict) else {}
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    row = {
        "ts": _now(),
        "request_id": response.get("request_id"),
        "frame_uid": response.get("frame_uid", frame.get("frame_uid")),
        "frame_sha256": frame.get("sha256"),
        "uri": frame.get("uri"),
        "site_id": metadata.get("site_id"),
        "plot_id": metadata.get("plot_id"),
        "zone_id": metadata.get("zone_id"),
        "timestamp_utc": metadata.get("timestamp_utc"),
        "platform": metadata.get("platform"),
        "bbch": metadata.get("bbch"),
        "bbch_null_reason": metadata.get("bbch_null_reason"),  # DECISIONS 30
        "model_version": response.get("model_version"),
        "run_id": run_id,
        "coverage": coverage,
        "decision": response.get("decision"),
        "abstain_reason": response.get("abstain_reason"),
        "top1": response.get("top1"),
        "scores": response.get("scores"),
        "uncertainty": response.get("uncertainty"),
        "features": response.get("features"),
        "status": status,
        "timing_ms": response.get("timing_ms"),
        "errors": response.get("errors"),
        "warnings": response.get("warnings"),
    }
    return {k: row[k] for k in FIELDS}


def log_request(path: Path | str, row: dict[str, Any]) -> None:
    """Append one row. Raises; the caller turns a failure into a warning on the response,
    never into a 500 (US-9.2): the answer is what the caller asked for."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path | str = REQUEST_LOG) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
