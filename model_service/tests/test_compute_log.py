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


# --- the audit (W6: every extraction and every head run has a row) ---------


def _roots(tmp_path, *, caches=(), runs=(), fits=()):
    """A cache root and a heads root holding the artefacts named, and nothing else."""
    cache_root, heads_root = tmp_path / "cache", tmp_path / "heads"
    for backbone_id, res, key, manifest in caches:
        folder = cache_root / backbone_id / str(res) / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{manifest}.npz").write_bytes(b"")
    for run_id in runs:
        (heads_root / run_id).mkdir(parents=True, exist_ok=True)
        (heads_root / run_id / "run.json").write_text("{}", encoding="utf-8")
    for run_id in fits:
        (heads_root / run_id).mkdir(parents=True, exist_ok=True)
        (heads_root / run_id / "abstain.json").write_text("{}", encoding="utf-8")
    return cache_root, heads_root


def _extract_row(path, backbone_id, res, key, manifest):
    compute_log.log_row(
        step="extract",
        backbone_id=backbone_id,
        res=res,
        extra={"cache": {"cache_key": key, "manifest": f"{manifest}.jsonl"}},
        path=path,
    )


def test_audit_is_ok_when_every_artefact_has_its_row(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    cache_root, heads_root = _roots(
        tmp_path, caches=[("dinov2_l14_reg", 224, "abc123", "ibean_v1")], runs=["r0"], fits=["r0"]
    )
    _extract_row(path, "dinov2_l14_reg", 224, "abc123", "ibean_v1")
    compute_log.log_row(step="train_head", extra={"head": {"run_id": "r0"}}, path=path)
    compute_log.log_row(step="fit_abstain", extra={"abstain": {"run_id": "r0"}}, path=path)

    result = compute_log.audit(path, cache_root=cache_root, heads_root=heads_root)
    assert result.ok
    assert result.counted == {"extraction": (1, 1), "head run": (1, 1), "abstention fit": (1, 1)}
    assert not any(result.unbacked.values())


def test_audit_names_an_extraction_and_a_head_run_with_no_row(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    cache_root, heads_root = _roots(
        tmp_path,
        caches=[("dinov2_l14_reg", 224, "abc123", "ibean_v1")],
        runs=["logged", "unlogged"],
        fits=[],
    )
    compute_log.log_row(step="train_head", extra={"head": {"run_id": "logged"}}, path=path)

    result = compute_log.audit(path, cache_root=cache_root, heads_root=heads_root)
    assert not result.ok
    assert result.missing["extraction"] == [("dinov2_l14_reg", 224, "abc123", "ibean_v1")]
    assert result.missing["head run"] == ["unlogged"]
    assert "INCOMPLETE" in compute_log.format_audit(result)


def test_a_row_whose_artefact_is_gone_is_reported_but_does_not_fail(tmp_path):
    """A superseded run still cost what it cost: the log keeps the row (DECISIONS 102)."""
    path = tmp_path / "compute_log.jsonl"
    cache_root, heads_root = _roots(tmp_path, runs=["kept"], fits=["kept"])
    for run_id in ("kept", "deleted"):
        compute_log.log_row(step="train_head", extra={"head": {"run_id": run_id}}, path=path)
    compute_log.log_row(step="fit_abstain", extra={"abstain": {"run_id": "kept"}}, path=path)

    result = compute_log.audit(path, cache_root=cache_root, heads_root=heads_root)
    assert result.ok
    assert result.unbacked["head run"] == ["deleted"]
    assert "1 with no artefact: deleted" in compute_log.format_audit(result)


def test_check_exits_1_when_something_has_no_row(tmp_path, capsys):
    path = tmp_path / "compute_log.jsonl"
    cache_root, heads_root = _roots(tmp_path, runs=["unlogged"], fits=["unlogged"])
    argv = ["--path", str(path), "check", "--heads-root", str(heads_root)]
    argv += ["--cache-root", str(cache_root)]

    assert compute_log.main(argv) == 1
    assert "unlogged" in capsys.readouterr().out

    compute_log.log_row(step="train_head", extra={"head": {"run_id": "unlogged"}}, path=path)
    compute_log.log_row(step="fit_abstain", extra={"abstain": {"run_id": "unlogged"}}, path=path)
    assert compute_log.main(argv) == 0


def test_totals_split_gpu_from_cpu_seconds(tmp_path):
    path = tmp_path / "compute_log.jsonl"
    compute_log.log_row(step="extract", wallclock_s=3600.0, device="NVIDIA X", path=path)
    compute_log.log_row(step="train_head", wallclock_s=1800.0, device="CPU (some cpu)", path=path)
    compute_log.log_row(step="env", device="NVIDIA X", path=path)  # no wall-clock

    result = compute_log.totals(path)
    assert result["gpu_hours"] == 1.0
    assert result["cpu_hours"] == 0.5
    assert result["wallclock_hours"] == 1.5
    assert result["steps"]["env"] == {"rows": 1, "wallclock_s": 0.0, "gpu_s": 0.0, "cpu_s": 0.0}
    assert list(result["steps"]) == ["env", "extract", "train_head"]  # the order of STEPS
    assert "1.00 GPU-hours" in compute_log.format_totals(result)


@pytest.mark.skipif(
    not (compute_log.CACHE_ROOT.exists() and compute_log.HEADS_ROOT.exists()),
    reason="needs data/cache and data/heads: the campaign's own artefacts, not in git",
)
def test_this_repositorys_log_accounts_for_every_extraction_and_head_run():
    """The W6 task itself, as a test: on a machine that holds the campaign, the log
    has a row for every cache and every head run on disk. Skipped on a fresh clone."""
    result = compute_log.audit()
    assert result.ok, compute_log.format_audit(result)
