"""N6 — latency (S4.6; specs/006-service/spec.md US-11; H8 §6.7).

    python -m ms.service.bench                       # both backbones, both resolutions
    python -m ms.service.bench --backbone dinov2_l14_reg --res 224 --batch 1

Times the service's own path: the same `Served.cached`, `Served.preprocess_images`,
`Served.embed` and `ms.abstain` calls `POST /v1/predict` makes, on a served model built for
each (backbone, resolution) in turn. No HTTP server is started — the transport is not what
N6 measures, and a number that included it would not be comparable across backbones.

Four metrics per model (spec 006 FR-010): `latency_ms:b1` and `latency_ms:b32` are H8's N6,
the cost of a frame the service has never seen — preprocess, backbone, head; `cached_b1` and
`cached_b32` are the replay path of H8 §6.9, the same frames answered from the feature cache,
reported beside them.

N6 is **reported, never optimised for** (H8 §6.9) and never a tie-breaker (H9 §7 criterion 4).
Measure it with nothing else on the GPU, and say so in `--notes`: the row carries it.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ms import compute_log
from ms.abstain import decide, probabilities
from ms.abstain import scores as abstain_scores
from ms.cache import BACKBONE_CONFIGS, backbone_config
from ms.data.manifests import load_manifest
from ms.eval import BATCH_NAMES, N_TABLE, NTableError, append_rows, read_rows
from ms.heads import resolve_path
from ms.service.app import SERVICE_CONFIG, Served, ServiceError, Settings, resolve_uri

#: what a call is timed at (H8 §6.7: "batch 1 and 32")
BATCHES = (1, 32)
#: frames timed per (model, batch, path) after the warm-up; the spread across calls is the
#: interval, so this has to leave more than a handful of calls at batch 32
FRAMES = 256
WARMUP = 8
PATHS = ("computed", "cache")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class BenchError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


# --- the frames ------------------------------------------------------------------------------


@dataclasses.dataclass
class Frames:
    """The frames every model is timed on: one manifest's test split, the same rows for every
    backbone and resolution, so the eight numbers compare."""

    manifest: str
    sha256: str
    split: str
    split_rule: str
    rows: list[dict[str, Any]]
    data: list[bytes]

    @property
    def n(self) -> int:
        return len(self.rows)


def take_frames(served: Served, settings: Settings, n: int, *, bytes_too: bool) -> Frames:
    """The first `n` rows of the served run's own first training manifest's test split that
    are in the cache — and, when the computed path is to be timed, on disk as well: the
    replay path and the live path have to be timed on the same frames, or the two numbers
    are not comparable."""
    path = served.run.meta["train"]["manifest"].split("+")[0]
    loaded = load_manifest(resolve_path(path), "test", "evaluate")
    rows, data = [], []
    for row in loaded.rows:
        if len(rows) == n:
            break
        if str(row["sha256"]) not in served.index:
            continue
        if bytes_too:
            try:
                where = resolve_uri(row["path"], settings.data_root)
            except Exception:  # noqa: BLE001 - a row whose file is not here is simply not timed
                continue
            data.append(where.read_bytes())
        rows.append(row)
    if len(rows) < n:
        where_from = f" and on disk under {settings.data_root}" if bytes_too else ""
        raise BenchError(
            "too_few_frames",
            f"{len(rows)} of {n} rows of {loaded.name}'s test split are in the cache{where_from}",
        )
    rules = sorted({r["split_rule"] for r in rows})
    return Frames(path, loaded.sha256, "test", "+".join(rules), rows, data)


# --- the measurements ------------------------------------------------------------------------


def time_calls(served: Served, frames: Frames, batch: int, path: str, warmup: int) -> list[float]:
    """The millisecond cost **per frame** of each call, in order. One call is `batch` frames
    through the same functions `POST /v1/predict` uses for that path."""
    if path not in PATHS:
        raise BenchError("bad_value", f"path must be one of {PATHS}, got {path!r}")
    run, fit = served.run, served.fit
    coverage = served.coverage

    def once(rows: list[dict[str, Any]], data: list[bytes]) -> None:
        if path == "cache":
            x = np.concatenate([served.cached(str(r["sha256"])) for r in rows])
        else:
            x = served.embed(served.preprocess_images(data))
        p = probabilities(fit, run, x)
        row_scores = abstain_scores(fit, run, x)
        for i in range(len(rows)):
            decide(fit, {k: float(v[i]) for k, v in row_scores.items()}, coverage)
            p[i].argmax()

    chunks = [
        (frames.rows[i : i + batch], frames.data[i : i + batch])
        for i in range(0, frames.n, batch)
        if len(frames.rows[i : i + batch]) == batch
    ]
    if not chunks:
        raise BenchError("too_few_frames", f"{frames.n} frames make no call of {batch}")
    for rows, data in chunks[: max(1, warmup // batch)]:
        once(rows, data)  # warm up: the first call loads the backbone and the kernels
    per_frame = []
    for rows, data in chunks:
        started = time.perf_counter()
        once(rows, data)
        per_frame.append((time.perf_counter() - started) * 1000 / batch)
    return per_frame


def summarise(per_frame: list[float]) -> dict[str, float]:
    """The median cost per frame, and the 5th and 95th percentiles over the calls (US-11.2).
    With few calls the percentiles are the extremes, which is what they should be."""
    ordered = sorted(per_frame)
    return {
        "value": round(statistics.median(ordered), 4),
        "ci_low": round(np.percentile(ordered, 5).item(), 4),
        "ci_high": round(np.percentile(ordered, 95).item(), 4),
        "calls": len(ordered),
    }


def qualifier(batch: int, path: str) -> str:
    name = f"b{batch}" if path == "computed" else f"cached_b{batch}"
    if name not in BATCH_NAMES:
        raise BenchError(
            "bad_value", f"batch {batch} on the {path} path is not one of {BATCH_NAMES}"
        )
    return name


def row_for(
    served: Served,
    frames: Frames,
    batch: int,
    path: str,
    stats: dict[str, float],
    *,
    device: str,
    warmup: int,
    notes: str | None,
) -> dict[str, Any]:
    meta, bb = served.run.meta, served.run.meta["backbone"]
    said = [
        f"device {device}",
        f"batch {batch}",
        f"{stats['calls']} calls",
        f"{warmup} frames of warm-up",
        "features computed on request" if path == "computed" else "features read from the cache",
        "median per frame; the interval is the 5th and 95th percentile over the calls",
    ]
    if notes:
        said.append(notes)
    return {
        "ts": _now(),
        "number": "N6",
        "metric": f"latency_ms:{qualifier(batch, path)}",
        "value": stats["value"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "n": frames.n,
        "backbone_id": bb["backbone_id"],
        "res": bb["res"],
        "token_type": meta["tokens"],
        "head": meta["head"],
        "seed": meta["seed"],
        "train_manifest": meta["train"]["manifest"],
        "train_manifest_sha256": meta["train"]["manifest_sha256"],
        "train_split": meta["train"]["split"],
        "test_manifest": frames.manifest,
        "test_manifest_sha256": frames.sha256,
        "test_split": frames.split,
        "split_rule": frames.split_rule,
        "coverage": 1.0,
        "classes": meta["classes"],
        "run_id": meta["run_id"],
        "model_version": meta["model_version"],
        "cache_key": bb["cache_key"],
        "quotable": bool(meta["quotable"]),
        "superseded": None,
        "notes": "; ".join(said),
    }


# --- the command -----------------------------------------------------------------------------


def wanted(args: argparse.Namespace, configs: Path, serving: Served) -> list[tuple[str, int]]:
    """The (backbone, resolution) pairs to measure: what `--backbone` and `--res` name, else
    every resolution of every backbone config — H8 §6.7's "both backbones, both resolutions".

    A deployment that has no backbone configs at all times the model it serves and nothing
    else: it can still read its cache, and a replayed frame is all it can answer anyway.
    """
    ids = args.backbone or sorted(p.stem for p in Path(configs).glob("*.yaml"))
    if not ids:
        bb = serving.run.meta["backbone"]
        return [(bb["backbone_id"], int(bb["res"]))]
    out = []
    for backbone_id in ids:
        try:
            cfg = backbone_config(backbone_id, Path(configs))
        except FileNotFoundError:
            raise BenchError("missing_backbone", f"no config for {backbone_id}") from None
        for res in args.res or cfg["resolutions"]:
            if int(res) not in [int(r) for r in cfg["resolutions"]]:
                raise BenchError("bad_value", f"{backbone_id} has no resolution {res}")
            out.append((backbone_id, int(res)))
    return out


def like_the_service(model: Served) -> dict[str, Any]:
    """What every timed model must share with the one the service serves: its head, its
    recipe, its seed, its tokens and its training manifests. Only the backbone and the
    resolution are allowed to differ, which is what N6 is about."""
    meta = model.run.meta
    return {
        "tokens": meta["tokens"],
        "head": meta["head"],
        "seed": meta["seed"],
        "head_config_sha256": meta["head_config"]["sha256"],
        "train_manifests": [Path(p).stem for p in meta["train"]["manifest"].split("+")],
    }


def served_for(
    base: Settings, like: dict[str, Any], backbone_id: str, res: int, device: str | None
) -> tuple[Served, Settings]:
    """The service's own model, for this backbone and this resolution."""
    settings = dataclasses.replace(
        base,
        run=None,
        select={**like, "backbone_id": backbone_id, "res": res},
        device=device or base.device,
    )
    return Served(settings), settings


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ms.service.bench", description=__doc__.split("\n")[0]
    )
    p.add_argument("--service", type=Path, default=SERVICE_CONFIG)
    p.add_argument("--backbone", action="append", default=None)
    p.add_argument("--res", action="append", type=int, default=None)
    p.add_argument("--batch", nargs="+", type=int, default=list(BATCHES))
    p.add_argument("--path", choices=[*PATHS, "both"], default="both")
    p.add_argument("--n", type=int, default=FRAMES, help="frames timed after the warm-up")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--backbone-configs",
        type=Path,
        default=None,
        help=f"default: the service config's, else {BACKBONE_CONFIGS}",
    )
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--compute-log", type=Path, default=compute_log.DEFAULT_LOG_PATH)
    p.add_argument("--notes", default=None, help="the conditions the measurement was taken in")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (BenchError, ServiceError, NTableError) as exc:
        print(exc, file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    paths = list(PATHS) if args.path == "both" else [args.path]
    base = Settings.from_config(args.service)
    configs = args.backbone_configs or base.backbone_configs
    # the model the service serves is what every timed model is held to (US-11.1)
    serving = Served(dataclasses.replace(base, device=args.device or base.device))
    like = like_the_service(serving)
    print(f"held to {serving.run.run_id}: {like['head']} seed {like['seed']} on {like['tokens']}")
    pairs = wanted(args, configs, serving)
    have = {
        (r["run_id"], r["metric"])
        for r in read_rows(args.n_table)
        if r["number"] == "N6" and not r.get("superseded")
    }
    rows, timed = [], 0
    for backbone_id, res in pairs:
        served, settings = served_for(base, like, backbone_id, res, args.device)
        frames = take_frames(served, settings, args.n, bytes_too="computed" in paths)
        device = served.device
        print(
            f"{served.run.run_id} @ {backbone_id}/{res} on {device}: "
            f"{frames.n} frames of {Path(frames.manifest).stem}'s {frames.split} split"
        )
        for path in paths:
            for batch in args.batch:
                metric = f"latency_ms:{qualifier(batch, path)}"
                if (served.run.run_id, metric) in have:
                    print(f"  {metric}: already in the table")
                    continue
                stats = summarise(time_calls(served, frames, batch, path, args.warmup))
                timed += frames.n
                rows.append(
                    row_for(
                        served,
                        frames,
                        batch,
                        path,
                        stats,
                        device=device,
                        warmup=args.warmup,
                        notes=args.notes,
                    )
                )
                print(
                    f"  {metric}: {stats['value']:.3f} ms per frame "
                    f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] over {stats['calls']} calls"
                )
    if not rows:
        print("nothing to do: every number asked for is already in the table")
        return 0
    added = append_rows(rows, args.n_table)
    compute_log.log_row(
        step="serve",
        backbone_id="+".join(sorted({r["backbone_id"] for r in rows})),
        res=sorted({r["res"] for r in rows})[-1],
        dataset=Path(rows[0]["test_manifest"]).stem,
        n_images_or_runs=timed,
        wallclock_s=round(time.perf_counter() - started, 3),
        device=rows[0]["notes"].split(";")[0].removeprefix("device ").strip(),
        notes=f"N6: {len(rows)} row(s) over {len(pairs)} model(s)",
        path=args.compute_log,
    )
    print(f"{added} row(s) added to {args.n_table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
