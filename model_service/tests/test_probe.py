"""The site-prediction probe, N4 (ms.eval.probe, `make probe`; work plan W2 Fri).

CPU only, on synthetic manifests and caches written to tmp_path: three small "sets" (alpha,
beta, gamma) for the dataset probe and one with districts (delta) for the district probe,
with features whose signal each test plants. No image, backbone or GPU is needed: the probe
reads nothing but manifests and caches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ms import compute_log
from ms.cache import cache_files, cache_key
from ms.data.manifests import load_manifest
from ms.eval import FIELDS as N_FIELDS
from ms.eval import probe

SETS = ("alpha", "beta", "gamma")
DIM = 8


def image_row(dataset: str, i: int, klass: str, split: str, **keys) -> dict:
    image_id = f"{dataset}_{i:05d}"
    return {
        "image_id": image_id,
        "dataset": dataset,
        "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
        "class_km2": klass,
        "split": split,
        "split_rule": "holdout_unknown" if split == "holdout_unknown" else f"blocked:{dataset}",
        "group_keys": {"district": None, "phash_group": image_id, **keys},
    }


def write_set(tmp: Path, name: str, rows: list[dict], x: np.ndarray, backbones: list[str]) -> Path:
    """<name>_v1.jsonl under tmp/manifests, and its cache under tmp/cache for each backbone,
    with the arrays and the sidecar fields that load_cache checks."""
    path = tmp / "manifests" / f"{name}_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    m = load_manifest(path, None, "extract")
    n, d = x.shape
    for bb in backbones:
        fields = {
            "backbone_id": bb,
            "weights_sha256": hashlib.sha256(bb.encode()).hexdigest(),
            "resolution": 28,
            "preprocess_string": "resize_short=28;center_crop=28;norm=imagenet",
            "compute_dtype": "float32",
        }
        key = cache_key(**fields)
        npz, side, _ = cache_files(tmp / "cache", bb, 28, key, m.name)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz,
            cls=x.astype(np.float16),
            meanpatch=x.astype(np.float16),
            image_id=np.array([r["image_id"] for r in m.rows]),
            sha256=np.array([r["sha256"] for r in m.rows]),
        )
        meta = {
            **fields,
            "cache_key": key,
            "token_types": ["cls", "meanpatch"],
            "storage_dtype": "float16",
            "shapes": {"cls": [n, d], "meanpatch": [n, d]},
            "n_images": n,
            "manifest": path.name,
            "manifest_sha256": m.sha256,
            "research_only_until_c5": False,
        }
        side.write_text(json.dumps(meta), encoding="utf-8")
    return path


def make_sets(
    tmp: Path,
    *,
    site: float,
    mix: dict[str, float] | None = None,
    backbones: tuple[str, ...] = ("bb",),
    n_train: int = 120,
    n_test: int = 60,
) -> None:
    """alpha, beta, gamma: healthy and rust rows in train and test (share of healthy: `mix`),
    plus a class only gamma has (anthracnose) and held-out rows only alpha has. A row's
    features are noise, the class (strong) and `site` times the set's own direction."""
    rng = np.random.default_rng(1)
    mix = mix or {s: 0.5 for s in SETS}
    for k, s in enumerate(SETS):
        rows = []
        for split, n in (("train", n_train), ("test", n_test)):
            healthy = round(n * mix[s])
            rows += [image_row(s, len(rows) + i, "healthy", split) for i in range(healthy)]
            rows += [image_row(s, len(rows) + i, "rust", split) for i in range(n - healthy)]
        if s == "gamma":
            rows += [image_row(s, len(rows) + i, "anthracnose", "test") for i in range(20)]
        if s == "alpha":
            rows += [
                image_row(s, len(rows) + i, "unknown_als", "holdout_unknown") for i in range(20)
            ]
        x = rng.normal(size=(len(rows), DIM))
        x[:, DIM - 1] += np.array([4.0 if r["class_km2"] == "healthy" else -4.0 for r in rows])
        x[:, k] += site
        write_set(tmp, s, rows, x, list(backbones))


def make_districts(tmp: Path, *, backbones: tuple[str, ...] = ("bb",)) -> list[dict]:
    """delta: four districts with rust and held-out ALS rows, near-duplicate pairs (one
    phash group each), and healthy rows that record no district."""
    rng = np.random.default_rng(2)
    rows, signal = [], []
    for d, district in enumerate(("Dorn", "Esk", "Fenn", "Gale")):
        for klass, split in (
            ("rust", "train"),
            ("rust", "test"),
            ("unknown_als", "holdout_unknown"),
        ):
            for _ in range(15):
                group = f"delta_g{len(rows):05d}"
                for _copy in range(2 if len(rows) % 3 == 0 else 1):
                    rows.append(
                        image_row(
                            "delta", len(rows), klass, split, district=district, phash_group=group
                        )
                    )
                    signal.append(d)
    rows += [image_row("delta", len(rows) + i, "healthy", "train") for i in range(30)]
    signal += [-1] * 30
    x = rng.normal(size=(len(rows), DIM))
    for i, d in enumerate(signal):
        if d >= 0:
            x[i, d] += 3.0
    write_set(tmp, "delta", rows, x, list(backbones))
    return rows


def run(capsys, tmp: Path, *extra, backbones=("bb",)) -> tuple[int, str]:
    argv = [
        "--res", "28", "--datasets", *SETS, "--within", "delta",
        "--manifest-root", tmp / "manifests", "--cache-root", tmp / "cache",
        "--n-table", tmp / "n_table.jsonl", "--md", tmp / "n_table.md",
        "--compute-log", tmp / "compute_log.jsonl",
        *[a for bb in backbones for a in ("--backbone", bb)], *extra,
    ]  # fmt: skip
    code = probe.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def table(tmp: Path) -> list[dict]:
    path = tmp / "n_table.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the dataset probe ----------------------------------------------------------------------------


def test_the_dataset_probe_reads_a_planted_site_and_its_row_says_how(tmp_path, capsys):
    make_sets(tmp_path, site=3.0)
    code, text = run(capsys, tmp_path, "--target", "dataset", "--no-mlflow")
    assert code == 0, text
    (row,) = table(tmp_path)
    assert list(row) == list(N_FIELDS)
    assert (row["number"], row["metric"], row["head"], row["token_type"]) == (
        "N4", "balanced_accuracy:dataset", "logreg", "cls",
    )  # fmt: skip
    assert row["value"] > 0.95 and row["ci_low"] <= row["value"] <= row["ci_high"]
    assert row["classes"] == list(SETS) and row["quotable"] is True
    # the shared classes only (not gamma's anthracnose), train fits and test scores
    assert row["n"] == 3 * 60 and row["train_split"] == "train" and row["test_split"] == "test"
    files = [tmp_path / "manifests" / f"{s}_v1.jsonl" for s in SETS]
    assert row["test_manifest_sha256"] == "+".join(sha256(f) for f in files)
    assert row["train_manifest"].split("+")[0].endswith("alpha_v1.jsonl")
    assert row["split_rule"] == "blocked:alpha+blocked:beta+blocked:gamma"
    assert row["run_id"].startswith("n4-dataset-bb-28-cls-")
    md = (tmp_path / "n_table.md").read_text(encoding="utf-8")
    assert "## N4 — site-prediction probe" in md and "(chance 0.333)" in md
    assert "alpha_v1+beta_v1+gamma_v1 (train)" in md


def test_a_class_mix_that_differs_by_set_does_not_pass_for_the_site(tmp_path, capsys):
    """Features that carry the class and nothing of the set, in sets whose class mix differs:
    a probe that weighs every (set, class) cell the same stays at chance."""
    mix = {"alpha": 0.9, "beta": 0.1, "gamma": 0.5}
    make_sets(tmp_path, site=0.0, mix=mix, n_train=300, n_test=150)
    code, text = run(capsys, tmp_path, "--target", "dataset", "--no-mlflow")
    assert code == 0, text
    (row,) = table(tmp_path)
    assert row["value"] < 0.45 and row["ci_low"] < 1 / 3 < row["ci_high"] + 0.05

    # the confound is real: unweighted, the class alone passes for the set
    x, y = [], []
    for k, s in enumerate(SETS):
        m = load_manifest(tmp_path / "manifests" / f"{s}_v1.jsonl", None, "evaluate")
        npz = next((tmp_path / "cache").rglob(f"{s}_v1.npz"))
        cls = np.load(npz)["cls"].astype(np.float32)
        keep = [i for i, r in enumerate(m.rows) if r["class_km2"] in ("healthy", "rust")]
        x.append(cls[keep])
        y += [k] * len(keep)
    naive = LogisticRegression(max_iter=1000).fit(np.concatenate(x), y)
    assert (naive.predict(np.concatenate(x)) == np.array(y)).mean() > 0.55


def test_cell_weights_give_every_target_and_its_classes_the_same_weight():
    target = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    klass = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0])
    w = probe.cell_weights(target, klass)
    assert w.mean() == pytest.approx(1.0)
    per_target = [w[target == t].sum() for t in range(3)]
    assert per_target == pytest.approx([per_target[0]] * 3)
    assert w[(target == 0) & (klass == 0)].sum() == pytest.approx(
        w[(target == 0) & (klass == 1)].sum()
    )


def test_the_score_is_the_mean_class_averaged_recall_with_a_bootstrap_interval():
    target = np.array([0] * 10 + [1] * 10)
    klass = np.array([0] * 8 + [1] * 2 + [0] * 5 + [1] * 5)
    pred = np.array([0] * 8 + [1] * 2 + [1] * 5 + [1, 1, 1, 0, 0])
    s = probe.score(target, klass, pred, seed=0)
    # target 0: healthy 8/8, rust 0/2 -> 0.5; target 1: 5/5 and 3/5 -> 0.8
    assert s["value"] == pytest.approx(0.65)
    assert s["recall"] == {0: pytest.approx(0.5), 1: pytest.approx(0.8)}
    assert s["ci_low"] <= s["value"] <= s["ci_high"]
    assert probe.score(target, klass, pred, seed=0) == s  # seeded


# --- the district probe ---------------------------------------------------------------------------


def test_the_district_probe_uses_the_rows_that_record_a_district(tmp_path, capsys):
    rows = make_districts(tmp_path)
    code, text = run(capsys, tmp_path, "--target", "district", "--no-mlflow")
    assert code == 0, text
    (row,) = table(tmp_path)
    with_district = [r for r in rows if r["group_keys"]["district"]]
    assert row["n"] == len(with_district) and row["classes"] == ["Dorn", "Esk", "Fenn", "Gale"]
    assert row["metric"] == "balanced_accuracy:district" and row["value"] > 0.9
    assert (row["train_split"], row["test_split"]) == ("cv5", "cv5")
    assert row["split_rule"] == "unblocked:5fold_by_phash_group"
    assert row["test_manifest_sha256"] == sha256(tmp_path / "manifests" / "delta_v1.jsonl")
    assert "(chance 0.250)" in (tmp_path / "n_table.md").read_text(encoding="utf-8")


def test_the_folds_never_split_a_phash_group():
    groups = np.array([f"g{i // 3}" for i in range(300)])  # groups of three
    target = np.array([(i // 3) % 4 for i in range(300)])
    fold = probe.folds(target, groups, 5, seed=0)
    assert set(fold) == set(range(5))
    for g in set(groups):
        assert len(set(fold[groups == g])) == 1


# --- the run --------------------------------------------------------------------------------------


def test_one_row_per_backbone_and_target_and_a_second_run_does_nothing(tmp_path, capsys):
    backbones = ("bb", "bb2")
    make_sets(tmp_path, site=3.0, backbones=backbones)
    make_districts(tmp_path, backbones=backbones)
    code, text = run(capsys, tmp_path, "--no-mlflow", backbones=backbones)
    assert code == 0, text
    rows = table(tmp_path)
    assert sorted((r["backbone_id"], r["metric"]) for r in rows) == [
        (bb, f"balanced_accuracy:{t}") for bb in backbones for t in ("dataset", "district")
    ]
    log = list(compute_log.read_rows(tmp_path / "compute_log.jsonl"))
    assert [r["step"] for r in log] == ["eval"] * 4
    assert {r["probe"]["probe_id"] for r in log} == {r["run_id"] for r in rows}
    assert all(r["probe"]["mlflow_run_id"] is None for r in log)

    before = (tmp_path / "n_table.jsonl").read_bytes(), (tmp_path / "n_table.md").read_bytes()
    code, text = run(capsys, tmp_path, "--no-mlflow", backbones=backbones)
    assert code == 0 and text.count("nothing to do") == 4
    after = (tmp_path / "n_table.jsonl").read_bytes(), (tmp_path / "n_table.md").read_bytes()
    assert after == before
    assert len(list(compute_log.read_rows(tmp_path / "compute_log.jsonl"))) == 4


def test_a_missing_cache_stops_the_run_before_anything_is_written(tmp_path, capsys):
    make_sets(tmp_path, site=3.0, backbones=("bb",))
    make_districts(tmp_path, backbones=("bb",))
    code, text = run(capsys, tmp_path, "--no-mlflow", backbones=("bb", "absent"))
    assert code == 2 and "missing_cache" in text
    assert not (tmp_path / "n_table.jsonl").exists()
    assert not (tmp_path / "compute_log.jsonl").exists()


def test_every_probe_is_one_mlflow_run_and_a_second_run_adds_none(tmp_path, capsys):
    from mlflow.tracking import MlflowClient

    make_sets(tmp_path, site=3.0)
    make_districts(tmp_path)
    uri = "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix()
    (tmp_path / "mlruns").mkdir()
    code, text = run(capsys, tmp_path, "--mlflow-uri", uri)
    assert code == 0, text
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(probe.EXPERIMENT)
    assert exp.artifact_location.startswith((tmp_path / "mlruns" / "artifacts").as_uri())
    runs = client.search_runs([exp.experiment_id])
    rows = {r["run_id"]: r for r in table(tmp_path)}
    assert sorted(r.data.tags["probe_id"] for r in runs) == sorted(rows)
    for r in runs:
        row = rows[r.data.tags["probe_id"]]
        assert r.data.metrics["balanced_accuracy"] == pytest.approx(row["value"], abs=1e-4)
        assert r.data.params["manifest_sha256"] == row["test_manifest_sha256"]
        assert r.data.tags["number"] == "N4" and r.info.status == "FINISHED"
    log = list(compute_log.read_rows(tmp_path / "compute_log.jsonl"))
    assert {r["probe"]["mlflow_run_id"] for r in log} == {r.info.run_id for r in runs}

    code, text = run(capsys, tmp_path, "--mlflow-uri", uri)
    assert code == 0 and text.count("nothing to do") == 2
    assert len(client.search_runs([exp.experiment_id])) == 2
