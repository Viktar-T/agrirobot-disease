"""Write the golden DINOv2 features of the fixture (spec 002 US-6; DECISIONS 11-12).

    uv run python model_service/tests/fixtures/make_golden.py [--work <scratch dir>]

Extracts tests/fixtures/ibean_30 with dinov2_l14_reg at 224 on CUDA twice, into a scratch
cache (never data/cache, never the compute log):

- float16, the config's dtype (what `make cache` computes), and
- float32, the reference.

Writes golden/dinov2_l14_reg_224_cls.npy: the float16 run's CLS, [30, 1024] float16, in
manifest row order. Writes golden/dinov2_l14_reg_224_cls.json: the tolerance and how it was
measured. The tolerance is the fp16/fp32 gap, 1 - min cosine(CLS fp16, CLS fp32) over the 30
images, rounded up to one significant digit. So a relock may move the features by as much as
float16 itself does, and no more. DINOv2 only: nothing derived from DINOv3 enters git
(DECISIONS 13).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import transformers

from ms.cache import extract as cache_extract
from ms.cache import load_cache

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "ibean_30"
GOLDEN = FIXTURE / "golden"
BACKBONE, RES = "dinov2_l14_reg", 224


def _extract(work: Path, dtype: str) -> np.ndarray:
    code = cache_extract.main(
        [
            "--backbone", BACKBONE,
            "--res", str(RES),
            "--manifest", str(FIXTURE / "ibean_v1.jsonl"),
            "--raw-root", str(FIXTURE),
            "--cache-root", str(work / dtype),
            "--compute-log", str(work / "compute_log.jsonl"),
            "--device", "cuda",
            "--dtype", dtype,
        ]
    )  # fmt: skip
    if code != 0:
        sys.exit(f"extraction ({dtype}) failed with exit {code}")
    (npz,) = sorted((work / dtype).rglob("*.npz"))
    return load_cache(npz).cls


def _round_up(x: float) -> float:
    """Up to one significant digit: 3.2e-05 -> 4e-05."""
    if x <= 0:
        return 1e-6
    exponent = math.floor(math.log10(x))
    return math.ceil(x / 10**exponent) * 10**exponent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work", type=Path, default=None, help="scratch folder (default: a temp dir)")
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("needs CUDA: the golden vectors are the float16 (autocast) features")
    work = args.work or Path(tempfile.mkdtemp(prefix="golden-"))
    fp16, fp32 = _extract(work, "float16"), _extract(work, "float32")
    a, b = fp16.astype(np.float64), fp32.astype(np.float64)
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    gap = float(1 - cosine.min())
    tolerance = _round_up(gap)

    GOLDEN.mkdir(exist_ok=True)
    np.save(GOLDEN / f"{BACKBONE}_{RES}_cls.npy", fp16)
    doc = {
        "tolerance": tolerance,
        "rule": "cosine(CLS, golden) >= 1 - tolerance for every fixture image (spec 002 US-6)",
        "measured": {
            "what": "cosine(CLS float16 autocast, CLS float32) on the 30 fixture images",
            "min": round(float(cosine.min()), 9),
            "mean": round(float(cosine.mean()), 9),
            "gap": gap,
        },
        "vectors": f"{BACKBONE}_{RES}_cls.npy: CLS of the float16 run, manifest row order",
        "manifest": "ibean_v1.jsonl",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "model_service/tests/fixtures/make_golden.py",
    }
    text = json.dumps(doc, indent=2) + "\n"
    (GOLDEN / f"{BACKBONE}_{RES}_cls.json").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
