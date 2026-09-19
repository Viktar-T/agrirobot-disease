"""The W2 vertical slice, end to end on the CPU fixture (docs/piece4-work-plan.md, W2 Mon-Tue).

manifest (the committed fixture, spec 001) -> cache (spec 002, a tiny random-init backbone
of the real class, as in test_cache.py) -> linear head (ms.heads.train) -> the N1 rows
(ms.eval.run) -> POST /v1/predict with one cached image (ms.service.app). Specs 003, 005 and
006 come in W3-W5; these tests pin what the slice already promises: the role gate, same seed
-> same weights, re-runs that do nothing, rows that carry their split rule and both
manifest hashes (per class since spec 005), and a service that answers the interface v0
record from the cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import transformers
import yaml
from fastapi.testclient import TestClient
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from ms import compute_log
from ms.cache import extract as cache_extract
from ms.eval import FIELDS as N_FIELDS
from ms.eval import run as eval_run
from ms.heads import train as heads_train
from ms.service.app import Settings, create_app
from ms.service.example import request_for

TESTS = Path(__file__).resolve().parent
FIXTURE = TESTS / "fixtures" / "ibean_30"
MANIFEST = FIXTURE / "ibean_v1.jsonl"
BACKBONE = "tiny_dinov2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_rows() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]


def tiny_backbone(root: Path) -> Path:
    """A tiny random-init Dinov2WithRegistersModel (width 32, patch 14, 4 registers) and its
    backbone config, as in test_cache.py; the config folder."""
    transformers.utils.logging.disable_progress_bar()
    torch.manual_seed(0)
    config = Dinov2WithRegistersConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=28,
        num_register_tokens=4,
    )
    weights = root / "weights"
    Dinov2WithRegistersModel(config).save_pretrained(weights)
    configs = root / "backbones"
    configs.mkdir(parents=True, exist_ok=True)
    backbone = {
        "backbone_id": BACKBONE,
        "hf_id": weights.as_posix(),
        "revision": None,
        "licence": "test weights, random init",
        "gated": False,
        "research_only_until_c5": False,
        "patch_size": 14,
        "embed_dim": 32,
        "resolutions": [28],
        "preprocess_string": {28: "resize_short=28;center_crop=28;norm=imagenet"},
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "batch_size": 8,
        "token_types": ["cls", "meanpatch"],
    }
    (configs / f"{BACKBONE}.yaml").write_text(yaml.safe_dump(backbone), encoding="utf-8")
    return configs


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory) -> Path:
    """The fixture's tiny-backbone cache, extracted once for this module."""
    root = tmp_path_factory.mktemp("slice")
    configs = tiny_backbone(root)
    code = cache_extract.main(
        [
            "--backbone", BACKBONE, "--res", "28", "--manifest", str(MANIFEST),
            "--raw-root", str(FIXTURE), "--config-dir", str(configs),
            "--cache-root", str(root / "cache"), "--compute-log", str(root / "log.jsonl"),
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0
    return root / "cache"


def train(capsys, cache_root: Path, heads: Path, *extra) -> tuple[int, str]:
    code = heads_train.main(
        [
            "--backbone", BACKBONE, "--res", "28", "--head", "linear", "--seed", "0",
            "--train-manifest", str(MANIFEST), "--split", "train", "--val-split", "val",
            "--cache-root", str(cache_root), "--heads-root", str(heads),
            "--compute-log", str(heads.parent / "log.jsonl"), *extra,
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def only_run(heads: Path) -> Path:
    (folder,) = [p for p in heads.iterdir() if (p / "run.json").exists()]
    return folder


# --- the head ---------------------------------------------------------------------------------


def test_a_test_only_manifest_trains_only_when_allowed(tmp_path, capsys, cache_root):
    code, text = train(capsys, cache_root, tmp_path / "heads")
    assert code == 2 and "role_not_trainable" in text
    assert not (tmp_path / "heads").exists() or not any((tmp_path / "heads").iterdir())

    code, text = train(capsys, cache_root, tmp_path / "heads", "--allow-test-only")
    assert code == 0, text
    run = json.loads((only_run(tmp_path / "heads") / "run.json").read_text(encoding="utf-8"))
    assert run["quotable"] is False and run["allow_test_only"] is True
    assert run["classes"] == ["healthy", "rust"] and run["tokens"] == "cls"
    assert run["train"]["manifest_sha256"] == sha256(MANIFEST)
    assert run["train"]["split"] == "train" and run["val"]["split"] == "val"
    assert run["train"]["n"] == sum(r["split"] == "train" for r in manifest_rows())
    assert run["backbone"]["cache_key"] == next(cache_root.rglob("*.npz")).parent.name
    assert run["model_version"] == f"msv0.1+{BACKBONE}@28.linear.man-{sha256(MANIFEST)[:6]}"
    assert 1 <= run["fit"]["best_epoch"] <= run["fit"]["epochs_run"] <= 50
    (row,) = compute_log.read_rows(tmp_path / "log.jsonl")
    assert (row["step"], row["n_images_or_runs"], row["head"]["run_id"]) == (
        "train_head",
        1,
        run["run_id"],
    )


def test_the_same_seed_gives_the_same_weights_and_a_rerun_does_nothing(
    tmp_path, capsys, cache_root
):
    for heads in (tmp_path / "a", tmp_path / "b"):
        assert train(capsys, cache_root, heads, "--allow-test-only")[0] == 0
    a, b = only_run(tmp_path / "a"), only_run(tmp_path / "b")
    assert a.name == b.name  # the run_id is derived from the inputs
    wa = torch.load(a / "head.pt", weights_only=True)["state_dict"]
    wb = torch.load(b / "head.pt", weights_only=True)["state_dict"]
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)

    before = (a / "head.pt").read_bytes()
    code, text = train(capsys, cache_root, tmp_path / "a", "--allow-test-only")
    assert code == 0 and "nothing to do" in text
    assert (a / "head.pt").read_bytes() == before

    code, _ = train(capsys, cache_root, tmp_path / "a", "--allow-test-only", "--seed", "1")
    assert code == 0
    assert len([p for p in (tmp_path / "a").iterdir() if (p / "run.json").exists()]) == 2


# --- the N1 row -------------------------------------------------------------------------------


def test_eval_writes_the_n1_rows_with_their_split_rule_and_both_hashes(
    tmp_path, capsys, cache_root
):
    heads = tmp_path / "heads"
    assert train(capsys, cache_root, heads, "--allow-test-only")[0] == 0
    table, md = tmp_path / "n_table.jsonl", tmp_path / "n_table.md"
    argv = [
        "--heads-root", str(heads), "--cache-root", str(cache_root), "--n-table", str(table),
        "--md", str(md), "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    assert eval_run.main(argv) == 0
    rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
    # spec 005 US-3: macro-F1 and one recall per class; one seed, so no aggregate
    assert sorted(r["metric"] for r in rows) == ["macro_f1", "recall:healthy", "recall:rust"]
    (row,) = [r for r in rows if r["metric"] == "macro_f1"]
    assert list(row) == list(N_FIELDS)
    test_rows = [r for r in manifest_rows() if r["split"] == "test"]
    assert (row["number"], row["n"]) == ("N1", len(test_rows))
    for r in rows:
        assert 0.0 <= r["value"] <= 1.0
        assert r["split_rule"] == "unblocked:random_by_phash_group"
        assert r["train_manifest_sha256"] == r["test_manifest_sha256"] == sha256(MANIFEST)
        assert (r["train_split"], r["test_split"]) == ("train", "test")
        assert (r["ci_low"], r["ci_high"], r["coverage"]) == (None, None, 1.0)
        assert r["quotable"] is False and r["run_id"] == only_run(heads).name
    text = md.read_text(encoding="utf-8")
    assert "## N1" in text and "unblocked:random_by_phash_group" in text and "not quotable" in text

    capsys.readouterr()
    assert eval_run.main(argv) == 0  # nothing new: no row, the table unchanged
    assert len(table.read_text(encoding="utf-8").splitlines()) == 3
    assert "0 row(s) added" in capsys.readouterr().out


# --- the service ------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, capsys, cache_root):
    heads = tmp_path / "heads"
    assert train(capsys, cache_root, heads, "--allow-test-only")[0] == 0
    settings = Settings(run=None, heads_root=heads, cache_root=cache_root, data_root=FIXTURE)
    with TestClient(create_app(settings)) as c:
        yield c


def a_request(split: str = "test") -> dict:
    row = next(r for r in manifest_rows() if r["split"] == split)
    return request_for(row)


def test_predict_answers_a_cached_frame(client):
    body = a_request()
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["request_id"] == body["request_id"]
    assert out["frame_uid"] == body["frame"]["frame_uid"]
    assert set(out["scores"]) == {"healthy", "rust"}
    assert sum(out["scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert out["top1"] == max(out["scores"], key=out["scores"].get)
    assert (out["decision"], out["abstain_reason"], out["features"]) == ("predict", None, "cache")
    assert out["model_version"].startswith(f"msv0.1+{BACKBONE}@28.linear.man-")
    assert out["backbone"]["input_res"] == 28 and out["head"]["id"] == "linear"
    assert "metadata.bbch is null" in out["warnings"]
    assert set(out["timing_ms"]) >= {"preprocess", "backbone", "head"}
    assert client.get("/v1/health").json()["cached_frames"] == 30
    assert client.get("/v1/model").json()["run_id"] == out["head"]["run_id"]


def test_a_broken_frame_is_422_with_the_validators_reason(client):
    body = a_request()
    del body["metadata"]["camera"]
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 422
    out = r.json()
    assert (out["decision"], out["abstain_reason"]) == ("abstain", "invalid_input")
    assert any("camera" in e["message"] for e in out["errors"])

    old = copy.deepcopy(a_request())
    old["frame"]["frame_id"] = old["frame"].pop("frame_uid")  # the pre-rename name (DECISIONS 31)
    assert client.post("/v1/predict", json=old).status_code == 422
    assert client.post("/v1/predict", content=b"not json").status_code == 422
    bad_yaw = a_request()
    bad_yaw["metadata"]["pose"]["yaw"] = 91.0  # degrees: invalid under v0 (DECISIONS 29)
    assert client.post("/v1/predict", json=bad_yaw).status_code == 422


def test_the_hash_is_recomputed_from_the_bytes_at_uri(client):
    body = a_request()
    body["frame"]["sha256"] = manifest_rows()[0]["sha256"]  # another image's hash
    if body["frame"]["sha256"] == a_request()["frame"]["sha256"]:
        body["frame"]["sha256"] = manifest_rows()[1]["sha256"]
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 422
    assert r.json()["errors"][0]["path"] == "frame.sha256"

    outside = a_request()
    outside["frame"]["uri"] = "../../test_slice.py"
    assert client.post("/v1/predict", json=outside).status_code == 422


def test_what_the_slice_does_not_serve_yet_is_501(client):
    body = a_request()
    body["frame"]["uri"] = "mcap://bag/camera/1"  # uri forms: open with piece 2 (W3)
    assert client.post("/v1/predict", json=body).status_code == 501

    uncached = a_request()  # a real file under the data root whose hash no cache holds
    uncached["frame"]["uri"] = "LICENSE-MIT"
    uncached["frame"]["sha256"] = sha256(FIXTURE / "LICENSE-MIT")
    r = client.post("/v1/predict", json=uncached)
    assert r.status_code == 501 and "not in the feature cache" in r.json()["detail"]
