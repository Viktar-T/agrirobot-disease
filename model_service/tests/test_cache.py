"""Spec 002 - the feature cache: acceptance tests (model_service/specs/002-cache/spec.md).

Written before the implementation (W1, spec-driven): every test here fails until ms.cache
exists (W2), each with the same one-line message.

CPU only, on the committed fixture tests/fixtures/ibean_30/ (30 iBean images and their
manifest ibean_v1.jsonl) and a tiny randomly initialised backbone of the real class
(transformers' Dinov2WithRegistersModel: 2 layers, width 32, patch 14, 4 register tokens),
saved to tmp_path: no download, no GPU, a fraction of a second per extraction. The real
backbones run only in the opt-in golden test (MS_GOLDEN=1, DECISIONS 12).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers
import yaml
from PIL import Image
from transformers import AutoModel, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from ms import compute_log

TESTS = Path(__file__).resolve().parent
FIXTURE = TESTS / "fixtures" / "ibean_30"
MANIFEST = FIXTURE / "ibean_v1.jsonl"
GOLDEN = FIXTURE / "golden"

BACKBONE = "tiny_dinov2"
PREPROCESS = {
    28: "resize_short=28;center_crop=28;norm=imagenet",
    42: "resize_short=42;center_crop=42;norm=imagenet",
}
KEY_FIELDS = ("backbone_id", "weights_sha256", "resolution", "preprocess_string", "compute_dtype")
# FR-006
META_FIELDS = {
    "meta_json_version",
    "cache_key",
    *KEY_FIELDS,
    "hf_id",
    "revision",
    "model_class",
    "attn_implementation",
    "research_only_until_c5",
    "token_types",
    "storage_dtype",
    "embed_dim",
    "n_images",
    "n_patch_tokens",
    "n_register_tokens",
    "shapes",
    "manifest",
    "manifest_sha256",
    "dataset",
    "transformers_version",
    "torch_version",
    "device",
    "batch_size",
    "shard_size",
    "wallclock_s",
    "images_per_s",
    "runs",
    "npz_sha256",
    "created_at",
    "builder",
}
# FR-008 / US-3.1
LOG_CACHE_FIELDS = {
    "cache_key",
    "manifest",
    "manifest_sha256",
    "compute_dtype",
    "batch_size",
    "shard_size",
    "images_per_s",
    "shards_done_before",
    "complete",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# --- the module under test ------------------------------------------------------------------


def _module(name: str):
    """ms.cache or ms.cache.extract; until they exist every test here fails with one message."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("ms.cache", name):
            raise
    pytest.fail("ms.cache does not exist yet (spec 002, W2)", pytrace=False)


def run(capsys, *argv) -> tuple[int, str]:
    """python -m ms.cache.extract <argv>, in process: (exit code, everything printed)."""
    module = _module("ms.cache.extract")
    try:
        code = module.main([str(a) for a in argv])
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def extract(capsys, tmp: Path, configs: Path, *extra, res=28, manifest=MANIFEST, raw=FIXTURE):
    """Extract the tiny backbone on CPU into tmp/cache, logging to tmp/compute_log.jsonl."""
    return run(
        capsys,
        "--backbone",
        BACKBONE,
        "--res",
        res,
        "--manifest",
        manifest,
        "--raw-root",
        raw,
        "--config-dir",
        configs,
        "--cache-root",
        tmp / "cache",
        "--compute-log",
        tmp / "compute_log.jsonl",
        "--device",
        "cpu",
        *extra,
    )


def tiny_backbone(root: Path, *, seed: int = 0) -> tuple[Path, Path]:
    """A tiny random-init Dinov2WithRegistersModel saved under root/weights/, and a backbone
    config for it in root/backbones/ with the keys of the real ones. (config dir, weights dir)"""
    transformers.utils.logging.disable_progress_bar()
    torch.manual_seed(seed)
    config = Dinov2WithRegistersConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=28,
        num_register_tokens=4,
    )
    weights = root / "weights" / f"{BACKBONE}-{seed}"
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
        "resolutions": [28, 42],
        "preprocess_string": PREPROCESS,
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "batch_size": 8,
        "token_types": ["cls", "meanpatch"],
    }
    (configs / f"{BACKBONE}.yaml").write_text(yaml.safe_dump(backbone), encoding="utf-8")
    return configs, weights


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_rows(path: Path = MANIFEST) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def caches(cache_root: Path) -> list[Path]:
    """The finished .npz files under a cache root (not the shards of a run under way)."""
    if not cache_root.exists():
        return []
    return sorted(p for p in cache_root.rglob("*.npz") if not p.parent.name.endswith(".shards"))


def sidecar(npz: Path) -> dict:
    return json.loads(npz.with_name(npz.stem + ".meta.json").read_text(encoding="utf-8"))


def the_cache(cache_root: Path) -> tuple[Path, dict]:
    (npz,) = caches(cache_root)
    return npz, sidecar(npz)


def log_rows(tmp: Path) -> list[dict]:
    return list(compute_log.read_rows(tmp / "compute_log.jsonl"))


# --- US-1: one cache per key ------------------------------------------------------------------


def test_us1_1_the_cache_holds_both_token_types_in_manifest_order(tmp_path, capsys):
    configs, weights = tiny_backbone(tmp_path)
    code, text = extract(capsys, tmp_path, configs)
    assert code == 0, text
    npz, meta = the_cache(tmp_path / "cache")
    rows = manifest_rows()

    assert npz.relative_to(tmp_path / "cache").parts == (
        BACKBONE,
        "28",
        meta["cache_key"],
        "ibean_v1.npz",
    )
    with np.load(npz, allow_pickle=False) as z:
        assert set(z.files) == {"cls", "meanpatch", "image_id", "sha256"}
        for token in ("cls", "meanpatch"):
            assert z[token].dtype == np.float16
            assert z[token].shape == (30, 32)
            assert list(z[token].shape) == meta["shapes"][token]
            assert np.isfinite(z[token]).all()
        assert list(z["image_id"]) == [r["image_id"] for r in rows]  # manifest order, every split
        assert list(z["sha256"]) == [r["sha256"] for r in rows]

    assert set(meta) >= META_FIELDS, META_FIELDS - set(meta)
    assert meta["token_types"] == ["cls", "meanpatch"] and meta["storage_dtype"] == "float16"
    assert {k: meta[k] for k in KEY_FIELDS} == {
        "backbone_id": BACKBONE,
        "weights_sha256": sha256(weights / "model.safetensors"),
        "resolution": 28,
        "preprocess_string": PREPROCESS[28],
        "compute_dtype": "float32",
    }
    assert (meta["embed_dim"], meta["n_images"]) == (32, 30)
    assert (meta["n_patch_tokens"], meta["n_register_tokens"]) == (4, 4)  # (28/14)^2; 4 registers
    assert meta["model_class"] == "Dinov2WithRegistersModel"
    assert meta["attn_implementation"] == "sdpa" and meta["research_only_until_c5"] is False
    assert (meta["manifest"], meta["dataset"]) == ("ibean_v1.jsonl", "ibean")
    assert meta["manifest_sha256"] == sha256(MANIFEST)
    assert meta["transformers_version"] == transformers.__version__
    assert meta["torch_version"] == torch.__version__
    assert meta["device"].startswith("CPU") and meta["batch_size"] == 8
    assert meta["wallclock_s"] > 0 and meta["runs"] == 1
    assert meta["images_per_s"] == pytest.approx(30 / meta["wallclock_s"], rel=1e-2)
    assert meta["npz_sha256"] == sha256(npz)


def test_us1_2_cls_is_the_first_token_and_meanpatch_the_patch_tokens_only(tmp_path, capsys):
    configs, weights = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs)[0] == 0
    npz, meta = the_cache(tmp_path / "cache")
    cache = _module("ms.cache")

    model = AutoModel.from_pretrained(weights).eval()
    pixels = torch.stack(
        [cache.preprocess(FIXTURE / r["path"], meta["preprocess_string"]) for r in manifest_rows()]
    )
    with torch.inference_mode():
        hidden = model(pixel_values=pixels).last_hidden_state.float()  # [30, 1 + 4 + 4, 32]
    assert hidden.shape[1] == 1 + 4 + 4
    with np.load(npz, allow_pickle=False) as z:
        cls, meanpatch = z["cls"].astype(np.float32), z["meanpatch"].astype(np.float32)
    np.testing.assert_allclose(cls, hidden[:, 0].numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(meanpatch, hidden[:, -4:].mean(1).numpy(), rtol=2e-3, atol=2e-3)
    # the 4 register tokens are left out: averaging every non-CLS token gives another vector
    assert not np.allclose(meanpatch, hidden[:, 1:].mean(1).numpy(), rtol=2e-3, atol=2e-3)


def test_us1_3_a_second_run_with_the_same_key_does_nothing(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs)[0] == 0
    files = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    rows = log_rows(tmp_path)

    code, text = extract(capsys, tmp_path, configs)
    assert code == 0, text
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    assert after == before
    assert sorted(p for p in (tmp_path / "cache").rglob("*") if p.is_file()) == sorted(files)
    assert log_rows(tmp_path) == rows


def test_us1_4_changed_weights_give_a_new_cache_and_keep_the_old_one(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path, seed=0)
    assert extract(capsys, tmp_path, configs)[0] == 0
    old, old_meta = the_cache(tmp_path / "cache")
    old_bytes = old.read_bytes()

    tiny_backbone(tmp_path, seed=1)  # other weights under the same backbone id
    code, text = extract(capsys, tmp_path, configs)
    assert code == 0, text
    both = caches(tmp_path / "cache")
    assert len(both) == 2 and old in both and old.read_bytes() == old_bytes
    (new,) = [p for p in both if p != old]
    new_meta = sidecar(new)
    assert new_meta["weights_sha256"] != old_meta["weights_sha256"]
    assert new_meta["cache_key"] != old_meta["cache_key"]
    assert new.parent.name == new_meta["cache_key"]


def test_us1_4_the_key_is_the_five_fields_and_nothing_else():
    cache = _module("ms.cache")
    assert tuple(inspect.signature(cache.cache_key).parameters) == KEY_FIELDS
    base = {
        "backbone_id": "dinov2_l14_reg",
        "weights_sha256": "a" * 64,
        "resolution": 224,
        "preprocess_string": "resize_short=224;center_crop=224;norm=imagenet",
        "compute_dtype": "float16",
    }
    key = cache.cache_key(**base)
    expected = hashlib.sha256(json.dumps([base[k] for k in KEY_FIELDS]).encode("utf-8"))
    assert key == expected.hexdigest()[:16]
    other = {
        "backbone_id": "dinov3_l16",
        "weights_sha256": "b" * 64,
        "resolution": 518,
        "preprocess_string": "resize_short=256;center_crop=224;norm=imagenet",
        "compute_dtype": "float32",
    }
    for field in KEY_FIELDS:
        assert cache.cache_key(**{**base, field: other[field]}) != key, field


def test_us1_4_versions_and_batch_size_leave_the_key_alone(tmp_path, capsys, monkeypatch):
    configs, _ = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs)[0] == 0
    rows = log_rows(tmp_path)
    monkeypatch.setattr(transformers, "__version__", "5.99.0")  # as after a relock
    code, text = extract(capsys, tmp_path, configs, "--batch-size", 4, "--shard-size", 16)
    assert code == 0, text
    assert len(caches(tmp_path / "cache")) == 1
    assert log_rows(tmp_path) == rows  # nothing computed


def test_us1_4_resolution_is_part_of_the_key(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs, res=28)[0] == 0
    code, text = extract(capsys, tmp_path, configs, res=42)
    assert code == 0, text
    by_res = {sidecar(p)["resolution"]: p for p in caches(tmp_path / "cache")}
    assert set(by_res) == {28, 42}
    meta = sidecar(by_res[42])
    assert meta["preprocess_string"] == PREPROCESS[42]
    assert meta["n_patch_tokens"] == 9  # (42/14)^2
    assert by_res[42].relative_to(tmp_path / "cache").parts[:2] == (BACKBONE, "42")


def test_us1_5_a_changed_manifest_rebuilds_its_cache(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    lines = MANIFEST.read_text(encoding="utf-8").splitlines(keepends=True)
    draft = tmp_path / "manifests" / "ibean_v1.jsonl"
    draft.parent.mkdir()
    draft.write_text("".join(lines[:-1]), encoding="utf-8", newline="\n")
    assert extract(capsys, tmp_path, configs, manifest=draft)[0] == 0
    assert the_cache(tmp_path / "cache")[1]["n_images"] == 29

    draft.write_text("".join(lines), encoding="utf-8", newline="\n")  # rebuilt before its freeze
    code, text = extract(capsys, tmp_path, configs, manifest=draft)
    assert code == 0, text
    npz, meta = the_cache(tmp_path / "cache")
    assert (meta["n_images"], meta["manifest_sha256"]) == (30, sha256(draft))
    with np.load(npz, allow_pickle=False) as z:
        assert z["cls"].shape == (30, 32)
    assert [r["n_images_or_runs"] for r in log_rows(tmp_path)] == [29, 30]


# --- US-2: resumable ----------------------------------------------------------------------------


def test_us2_1_a_stopped_run_resumes_where_it_stopped(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    code, text = extract(capsys, tmp_path, configs, "--shard-size", 8, "--max-shards", 2)
    assert code == 1, text
    assert caches(tmp_path / "cache") == []  # no final .npz yet
    (first,) = log_rows(tmp_path)
    assert first["n_images_or_runs"] == 16
    assert (first["cache"]["complete"], first["cache"]["shards_done_before"]) == (False, 0)

    code, text = extract(capsys, tmp_path, configs, "--shard-size", 8)
    assert code == 0, text
    second = log_rows(tmp_path)[1]
    assert second["n_images_or_runs"] == 14
    assert (second["cache"]["complete"], second["cache"]["shards_done_before"]) == (True, 2)
    npz, meta = the_cache(tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.shards"))  # removed once complete
    assert meta["runs"] == 2
    assert meta["wallclock_s"] == pytest.approx(
        first["wallclock_s"] + second["wallclock_s"], abs=0.1
    )

    straight = tmp_path / "straight"  # the same extraction in one uninterrupted run
    code, text = extract(capsys, straight, configs, "--shard-size", 8)
    assert code == 0, text
    other, _ = the_cache(straight / "cache")
    with np.load(npz, allow_pickle=False) as a, np.load(other, allow_pickle=False) as b:
        for name in ("cls", "meanpatch", "image_id", "sha256"):
            np.testing.assert_array_equal(a[name], b[name])


# --- US-3: the compute-log row --------------------------------------------------------------


def test_us3_1_a_run_that_computes_writes_one_compute_log_row(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs)[0] == 0
    _, meta = the_cache(tmp_path / "cache")
    (row,) = log_rows(tmp_path)
    assert set(compute_log.FIELDS) <= set(row)
    assert (row["step"], row["backbone_id"], row["res"], row["dataset"]) == (
        "extract",
        BACKBONE,
        28,
        "ibean",
    )
    assert row["n_images_or_runs"] == 30 and row["wallclock_s"] > 0
    assert row["device"] == meta["device"] and row["vram_gb"] is None  # CPU
    assert set(row["cache"]) >= LOG_CACHE_FIELDS, LOG_CACHE_FIELDS - set(row["cache"])
    c = row["cache"]
    assert (c["cache_key"], c["manifest"], c["manifest_sha256"]) == (
        meta["cache_key"],
        "ibean_v1.jsonl",
        sha256(MANIFEST),
    )
    assert (c["compute_dtype"], c["batch_size"], c["complete"], c["shards_done_before"]) == (
        "float32",
        8,
        True,
        0,
    )


# --- US-4: integrity ------------------------------------------------------------------------------


def test_us4_1_a_changed_or_missing_image_stops_the_run(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE / "ibean", raw / "ibean")
    rows = manifest_rows()

    changed = raw / rows[3]["path"]
    original = changed.read_bytes()
    changed.write_bytes(original[:-16] + bytes(16))  # other bytes under the same name
    code, text = extract(capsys, tmp_path, configs, raw=raw)
    assert code == 2
    assert "sha256_mismatch" in text and rows[3]["path"] in text

    changed.write_bytes(original)
    (raw / rows[5]["path"]).unlink()
    code, text = extract(capsys, tmp_path, configs, raw=raw)
    assert code == 2
    assert "missing_file" in text and rows[5]["path"] in text

    assert caches(tmp_path / "cache") == []
    assert log_rows(tmp_path) == []


def test_a_dataset_id_without_a_recipe_is_its_one_manifest_on_disk(tmp_path, capsys, monkeypatch):
    """--dataset <id> reads <id>_v<N>.jsonl, N from the recipe. A crops manifest has no recipe
    (DECISIONS 46): its id names the one version on disk. None, or two, is an input error."""
    configs, _ = tiny_backbone(tmp_path)
    manifests = importlib.import_module("ms.data.manifests")
    extract_module = _module("ms.cache.extract")
    root = tmp_path / "manifests"
    root.mkdir()
    shutil.copy(MANIFEST, root / "ibean_crops_v1.jsonl")
    monkeypatch.setattr(extract_module, "manifest_path", lambda m: manifests.manifest_path(m, root))
    argv = [
        "--backbone", BACKBONE, "--res", 28, "--raw-root", FIXTURE, "--config-dir", configs,
        "--cache-root", tmp_path / "cache", "--compute-log", tmp_path / "compute_log.jsonl",
        "--device", "cpu",
    ]  # fmt: skip

    code, text = run(capsys, *argv, "--dataset", "ibean_crops")
    assert code == 0, text
    npz, meta = the_cache(tmp_path / "cache")
    assert npz.name == "ibean_crops_v1.npz" and meta["manifest"] == "ibean_crops_v1.jsonl"

    for dataset in ("nothing", "ibean_crops"):
        shutil.copy(MANIFEST, root / "ibean_crops_v2.jsonl")  # two versions: name the file
        code, text = run(capsys, *argv, "--dataset", dataset)
        assert code == 2 and "missing_file" in text and "Traceback" not in text
    assert len(log_rows(tmp_path)) == 1


def test_us4_2_load_cache_checks_the_manifest_the_key_and_the_shapes(tmp_path, capsys):
    configs, _ = tiny_backbone(tmp_path)
    assert extract(capsys, tmp_path, configs)[0] == 0
    npz, meta = the_cache(tmp_path / "cache")
    cache = _module("ms.cache")

    loaded = cache.load_cache(npz, manifest_sha256=sha256(MANIFEST))
    assert loaded.meta["cache_key"] == meta["cache_key"]
    assert loaded.cls.shape == (30, 32) and loaded.meanpatch.dtype == np.float16
    assert list(loaded.image_id) == [r["image_id"] for r in manifest_rows()]
    assert list(loaded.sha256) == [r["sha256"] for r in manifest_rows()]

    with pytest.raises(cache.CacheMismatchError):
        cache.load_cache(npz, manifest_sha256="0" * 64)
    path = npz.with_name(npz.stem + ".meta.json")
    good = path.read_text(encoding="utf-8")
    for broken in (
        {**meta, "shapes": {"cls": [29, 32], "meanpatch": [30, 32]}},
        {**meta, "resolution": 42},  # a key field that no longer gives cache_key
    ):
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(cache.CacheMismatchError):
            cache.load_cache(npz)
    path.write_text(good, encoding="utf-8")


# --- US-5: preprocessing ----------------------------------------------------------------------


def test_us5_1_preprocess_does_what_the_string_says(tmp_path):
    cache = _module("ms.cache")
    string = PREPROCESS[28]

    flat = Image.new("RGB", (60, 40), (124, 116, 104))  # landscape: the short side goes to 28
    x = cache.preprocess(flat, string)
    assert tuple(x.shape) == (3, 28, 28) and x.dtype == torch.float32
    expected = (torch.tensor([124.0, 116.0, 104.0]) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    assert torch.allclose(x.mean(dim=(1, 2)), expected, atol=2e-2)
    assert torch.allclose(x.std(dim=(1, 2)), torch.zeros(3), atol=2e-2)  # flat stays flat

    # stored left red / right blue, tagged "rotate 90° clockwise to view": red is on top
    halves = Image.new("RGB", (40, 20), (0, 0, 255))
    halves.paste((255, 0, 0), (0, 0, 20, 20))
    exif = Image.Exif()
    exif[274] = 6  # Orientation
    tagged = tmp_path / "tagged.jpg"
    halves.save(tagged, "JPEG", quality=95, exif=exif)
    x = cache.preprocess(tagged, string)
    red, blue = 0, 2
    assert x[red, :10].mean() > x[blue, :10].mean()
    assert x[blue, -10:].mean() > x[red, -10:].mean()

    for bad in (string + ";flip=h", "resize_short=28;center_crop=28;norm=clip"):
        with pytest.raises(ValueError):
            cache.preprocess(flat, bad)


# --- US-6: golden features (opt-in) -----------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MS_GOLDEN"),
    reason="needs the DINOv2 weights; run before any relock with MS_GOLDEN=1 (DECISIONS 12)",
)
def test_us6_1_dinov2_cls_at_224_on_the_fixture_matches_the_golden_vectors(tmp_path, capsys):
    vectors = GOLDEN / "dinov2_l14_reg_224_cls.npy"
    tolerance = GOLDEN / "dinov2_l14_reg_224_cls.json"
    assert vectors.exists() and tolerance.exists(), (
        "W2: write the golden vectors and their tolerance after the fp16/fp32 check (DECISIONS 11)"
    )
    code, text = run(
        capsys,
        "--backbone",
        "dinov2_l14_reg",
        "--res",
        224,
        "--manifest",
        MANIFEST,
        "--raw-root",
        FIXTURE,
        "--cache-root",
        tmp_path / "cache",
        "--compute-log",
        tmp_path / "compute_log.jsonl",
    )
    assert code == 0, text
    npz, _ = the_cache(tmp_path / "cache")
    with np.load(npz, allow_pickle=False) as z:
        got = z["cls"].astype(np.float32)
    want = np.load(vectors).astype(np.float32)
    cosine = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
    tol = json.loads(tolerance.read_text(encoding="utf-8"))["tolerance"]
    assert cosine.min() >= 1 - tol, cosine.min()


# --- the fixture this file reads -------------------------------------------------------------


def test_the_fixture_manifest_matches_the_fixture_images_and_its_sidecar():
    rows = manifest_rows()
    assert len(rows) == 30
    for r in rows:
        assert sha256(FIXTURE / r["path"]) == r["sha256"], r["path"]
    meta = json.loads((FIXTURE / "ibean_v1.meta.json").read_text(encoding="utf-8"))
    assert meta["manifest_sha256"] == sha256(MANIFEST)
