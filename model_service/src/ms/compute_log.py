"""The compute log (N7) - written by code, never by hand.

One JSON line per unit of compute, at "model_service/results/compute_log.jsonl"
(docs/piece4-work-plan.md section 4, "Conventions"):

    {ts, step, backbone_id, res, dataset, n_images_or_runs, wallclock_s,
     device, vram_gb, notes}

Two documented extensions to that row (model_service/DECISIONS.md, 2026-09-17):

* step = "env" in addition to {extract, train_head, eval, serve}, for the first
  entry of the log - the machine the numbers were produced on (W1 "Environment");
* an optional "env" object on that row, so the driver / CUDA / torch versions are
  structured data rather than prose inside "notes".

Usage from code:

    from ms.compute_log import log_row
    log_row(step="extract", backbone_id="dinov2_l14_reg", res=224,
            dataset="ibean", n_images_or_runs=1296, wallclock_s=1834.2,
            device="NVIDIA GeForce RTX 5060 Laptop GPU", vram_gb=7.96)

and from the command line:

    python -m ms.compute_log record-env     # idempotent: same machine -> no new row
    python -m ms.compute_log show
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Steps allowed in the "step" field ("env" is the W1 extension, see module docstring).
STEPS = ("env", "extract", "train_head", "eval", "serve")

#: Row fields, in the order they are written.
FIELDS = (
    "ts",
    "step",
    "backbone_id",
    "res",
    "dataset",
    "n_images_or_runs",
    "wallclock_s",
    "device",
    "vram_gb",
    "notes",
)

#: repository root: model_service/src/ms/compute_log.py -> ../../../
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_LOG_PATH = REPO_ROOT / "model_service" / "results" / "compute_log.jsonl"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    """Run a command, return stripped stdout, or None if it is unavailable or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def log_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_LOG_PATH


def read_rows(path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield the rows of the compute log; an absent log yields nothing."""
    p = log_path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def log_row(
    *,
    step: str,
    backbone_id: str | None = None,
    res: int | None = None,
    dataset: str | None = None,
    n_images_or_runs: int | None = None,
    wallclock_s: float | None = None,
    device: str | None = None,
    vram_gb: float | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one row to the compute log and return it."""
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}, got {step!r}")
    row: dict[str, Any] = {
        "ts": _utc_now(),
        "step": step,
        "backbone_id": backbone_id,
        "res": res,
        "dataset": dataset,
        "n_images_or_runs": n_images_or_runs,
        "wallclock_s": wallclock_s,
        "device": device,
        "vram_gb": vram_gb,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    p = log_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# --- the environment row ---------------------------------------------------


def _nvidia_smi_gpu() -> dict[str, Any]:
    """GPU name / VRAM / driver from nvidia-smi; an empty dict when there is no GPU."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 3:
        return {}
    gpu: dict[str, Any] = {"name": parts[0], "driver_version": parts[2]}
    try:
        gpu["vram_gb"] = round(float(parts[1]) / 1024.0, 2)
    except ValueError:
        gpu["vram_gb"] = None
    if len(parts) > 3:
        gpu["compute_capability"] = parts[3]
    header = _run(["nvidia-smi"])
    if header:
        for line in header.splitlines():
            if "CUDA Version" in line:
                gpu["cuda_driver_version"] = line.split("CUDA Version:", 1)[1].strip(" |").strip()
                break
    return gpu


def _torch_info() -> dict[str, Any]:
    """What torch sees; an empty dict when torch is not installed (before "make env")."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a broken torch install must not break the log
        return {}
    cudnn_available = torch.backends.cudnn.is_available()
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cudnn_available else None,
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_count"] = torch.cuda.device_count()
    return info


def _os_string() -> str:
    """OS name, release and build. platform.release() still says "10" on Windows 11."""
    system, release, version = platform.system(), platform.release(), platform.version()
    if system == "Windows" and release == "10":
        try:
            build = int(version.rsplit(".", 1)[-1])
        except ValueError:
            build = 0
        if build >= 22000:
            release = "11"
    return f"{system} {release} ({version})"


def _module_version(name: str) -> str | None:
    try:
        return __import__(name).__version__
    except Exception:  # noqa: BLE001 - absent or broken package is a valid answer
        return None


def collect_env() -> dict[str, Any]:
    """Describe this machine. Never raises: a CPU-only box is a valid answer."""
    gpu = _nvidia_smi_gpu()
    torch_info = _torch_info()
    return {
        "hostname": platform.node(),
        "os": _os_string(),
        "cpu": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "gpu_name": gpu.get("name") or torch_info.get("gpu_name"),
        "vram_gb": gpu.get("vram_gb") or torch_info.get("gpu_vram_gb"),
        "driver_version": gpu.get("driver_version"),
        "cuda_driver_version": gpu.get("cuda_driver_version"),
        "compute_capability": gpu.get("compute_capability")
        or torch_info.get("gpu_compute_capability"),
        "torch_version": torch_info.get("torch_version"),
        "torch_cuda_version": torch_info.get("torch_cuda_version"),
        "cuda_available": torch_info.get("cuda_available"),
        "cudnn_version": torch_info.get("cudnn_version"),
        "transformers_version": _module_version("transformers"),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
    }


def _env_fingerprint(env: dict[str, Any]) -> tuple:
    """What makes an environment row worth writing again."""
    return tuple(
        env.get(k)
        for k in (
            "hostname",
            "gpu_name",
            "vram_gb",
            "driver_version",
            "torch_version",
            "torch_cuda_version",
            "python_version",
            "transformers_version",
        )
    )


def record_env(
    path: str | Path | None = None, *, force: bool = False, notes: str | None = None
) -> dict[str, Any] | None:
    """Append the environment row (N7). Idempotent: an unchanged machine adds nothing.

    Returns the row written, or None when an identical environment row is already
    the last one in the log.
    """
    env = collect_env()
    if not force:
        previous = [r for r in read_rows(path) if r.get("step") == "env"]
        if previous and _env_fingerprint(previous[-1].get("env", {})) == _env_fingerprint(env):
            return None
    device = env["gpu_name"] or f"CPU ({env['cpu']})"
    if notes is None:
        bits = []
        if env["driver_version"]:
            bits.append(f"driver {env['driver_version']}")
        if env["cuda_driver_version"]:
            bits.append(f"CUDA {env['cuda_driver_version']}")
        if env["torch_version"]:
            bits.append(f"torch {env['torch_version']} (cuda {env['torch_cuda_version']})")
        bits.append(f"python {env['python_version']}")
        notes = "; ".join(bits)
    return log_row(
        step="env",
        device=device,
        vram_gb=env["vram_gb"],
        notes=notes,
        extra={"env": env},
        path=path,
    )


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ms.compute_log",
        description="Read and write the compute log (N7).",
    )
    parser.add_argument("--path", default=None, help=f"compute log (default: {DEFAULT_LOG_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record-env", help="append the environment row (N7); idempotent")
    rec.add_argument("--force", action="store_true", help="write even if the machine is unchanged")
    rec.add_argument("--notes", default=None, help="override the generated notes string")

    sub.add_parser("show", help="print the log")

    args = parser.parse_args(argv)

    if args.command == "record-env":
        row = record_env(args.path, force=args.force, notes=args.notes)
        if row is None:
            print(f"compute log unchanged (same machine): {log_path(args.path)}")
        else:
            print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0

    for row in read_rows(args.path):
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
