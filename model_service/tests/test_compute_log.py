"""The compute log is the only artefact of the W1 "Environment" task, so it has a test.

Runs on CPU, without a GPU and without torch installed (CI).
"""

from __future__ import annotations

import json

import pytest

from ms import compute_log


def _rows(path):
    return list(compute_log.read_rows(path))


def test_log_row_writes_every_field(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    compute_log.log_row(
        step="extract",
        backbone_id="dinov2_l14_reg",
        res=224,
        dataset="ibean",
        n_images_or_runs=1296,
        wallclock_s=12.5,
        device="test device",
        vram_gb=8.0,
        path=path,
    )
    (row,) = _rows(path)
    assert set(compute_log.FIELDS) <= set(row)
    assert row["step"] == "extract"
    assert row["ts"].endswith("Z")


def test_log_row_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    for _ in range(3):
        compute_log.log_row(step="eval", path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["step"] == "eval" for line in lines)


def test_log_row_rejects_an_unknown_step(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    with pytest.raises(ValueError, match="step must be one of"):
        compute_log.log_row(step="benchmark", path=path)
    assert not path.exists()


def test_collect_env_never_raises_and_names_the_machine():
    env = compute_log.collect_env()
    assert env["python_version"]
    assert env["os"]
    # gpu_name is None on a CPU-only machine; the key is always there.
    assert "gpu_name" in env and "vram_gb" in env and "driver_version" in env


def test_record_env_is_idempotent_for_one_machine(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    first = compute_log.record_env(path)
    assert first is not None
    assert first["step"] == "env"
    assert first["device"]
    assert first["env"]["python_version"]

    assert compute_log.record_env(path) is None
    assert len(_rows(path)) == 1

    assert compute_log.record_env(path, force=True) is not None
    assert len(_rows(path)) == 2
