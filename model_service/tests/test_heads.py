"""Spec 003, heads (ms.heads.train; work plan S4.3 and the W3 Heads task).

CPU only, on the synthetic manifests and caches of synthetic.py: "alpha", a train_eval
manifest with healthy, rust and anthracnose in train, val and test plus held-out rows, and
features with a planted class signal. No image, backbone or GPU is needed. The runs use the
committed head configs (configs/heads/<head>.yaml), H8's recipe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from synthetic import BACKBONE, COUNTS, DIM, RES, make_manifest, rows_of, sha256, write_cache
from threadpoolctl import threadpool_limits

import ms.heads as heads
from ms import compute_log
from ms.heads import train as heads_train

HEADS = ("linear", "proto", "mix")
CONFIGS = Path(heads.HEAD_CONFIGS)
#: run.json v2 (FR-004), in order
RUN_KEYS = [
    "run_json_version", "run_id", "head", "tokens", "seed", "classes", "input_dim",
    "model_version", "quotable", "allow_test_only", "backbone", "train", "val", "head_config",
    "fit", "components", "inputs_sha256", "wallclock_s", "device", "torch_version",
    "created_at", "builder", "notes",
]  # fmt: skip
BACKBONE_KEYS = [
    "backbone_id", "res", "cache_key", "weights_sha256", "preprocess_string", "compute_dtype",
    "hf_id", "revision", "research_only_until_c5",
]  # fmt: skip
SIDE_KEYS = [
    "manifest", "manifest_sha256", "frozen", "role", "split", "split_rules", "n", "class_counts",
    "cache",
]  # fmt: skip
FIT_KEYS = {
    "select", "epochs_run", "best_epoch", "stopped_early", "val_macro_f1", "val_loss",
    "val_balanced_loss", "history",
}  # fmt: skip


def train(
    capsys,
    root: Path,
    *args: str,
    heads_root: Path | None = None,
    manifest: str | tuple[str, ...] = "alpha",
) -> tuple[int, str]:
    names = (manifest,) if isinstance(manifest, str) else manifest
    argv = [
        "--backbone", BACKBONE, "--res", str(RES),
        "--train-manifest", *(str(root / "manifests" / f"{m}_v1.jsonl") for m in names),
        "--cache-root", str(root / "cache"), "--heads-root", str(heads_root or root / "heads"),
        "--compute-log", str(root / "log.jsonl"), *args,
    ]  # fmt: skip
    code = heads_train.main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def runs(heads_root: Path) -> list[dict]:
    """Every run.json under heads_root, ordered by head, then seed."""
    if not heads_root.exists():
        return []
    found = [
        json.loads((p / "run.json").read_text(encoding="utf-8"))
        for p in heads_root.iterdir()
        if (p / "run.json").is_file()
    ]
    return sorted(found, key=lambda r: (HEADS.index(r["head"]), r["seed"]))


def saved(heads_root: Path, run: dict) -> dict:
    return torch.load(heads_root / run["run_id"] / "head.pt", weights_only=True)


def same(a, b) -> bool:
    """Equal, tensors bit for bit, through nested dicts and lists."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return type(a) is type(b) and len(a) == len(b) and all(map(same, a, b))
    return a == b


def log(root: Path) -> list[dict]:
    return list(compute_log.read_rows(root / "log.jsonl"))


def nothing_written(root: Path) -> bool:
    return not runs(root / "heads") and not (root / "log.jsonl").exists()


def unit_features(n: int, seed: int = 5) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- the command (US-1) ---------------------------------------------------------------------


def test_every_head_and_seeds_0_to_4_by_default(tmp_path, capsys):
    """US-1.4: no --head and no --seed is the W3 protocol."""
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path)
    assert code == 0, text
    found = runs(tmp_path / "heads")
    assert [(r["head"], r["seed"]) for r in found] == [(h, s) for h in HEADS for s in range(5)]
    assert sorted(r["head"]["run_id"] for r in log(tmp_path)) == sorted(r["run_id"] for r in found)


def test_several_heads_and_seeds_in_one_call(tmp_path, capsys):
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "linear", "proto", "mix", "--seed", "0", "1")
    assert code == 0, text
    by = {(r["head"], r["seed"]): r for r in runs(tmp_path / "heads")}
    assert sorted(by) == sorted((h, s) for h in HEADS for s in (0, 1))
    for s in (0, 1):
        assert by[("mix", s)]["components"] == [
            {"head": "linear", "run_id": by[("linear", s)]["run_id"]},
            {"head": "proto", "run_id": by[("proto", s)]["run_id"]},
        ]


def test_a_mix_trains_its_missing_components_first(tmp_path, capsys):
    """US-2.3: each component is its own run, with its own compute-log row."""
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "mix", "--seed", "3")
    assert code == 0, text
    found = runs(tmp_path / "heads")
    assert [(r["head"], r["seed"]) for r in found] == [("linear", 3), ("proto", 3), ("mix", 3)]
    assert sorted(r["head"]["head"] for r in log(tmp_path)) == ["linear", "mix", "proto"]


@pytest.mark.parametrize("head", HEADS)
def test_the_same_seed_gives_the_same_weights_and_a_rerun_does_nothing(tmp_path, capsys, head):
    """US-1.2 and US-1.3."""
    make_manifest(tmp_path)
    for name in ("a", "b"):
        code, text = train(
            capsys, tmp_path, "--head", head, "--seed", "0", heads_root=tmp_path / name
        )
        assert code == 0, text
    a, b = tmp_path / "a", tmp_path / "b"
    (run_a,) = [r for r in runs(a) if r["head"] == head]
    (run_b,) = [r for r in runs(b) if r["head"] == head]
    assert run_a["run_id"] == run_b["run_id"]
    assert same(saved(a, run_a), saved(b, run_b))

    rows_before = len(log(tmp_path))
    before = (a / run_a["run_id"] / "head.pt").read_bytes()
    code, text = train(capsys, tmp_path, "--head", head, "--seed", "0", heads_root=a)
    assert code == 0 and "nothing to do" in text
    assert (a / run_a["run_id"] / "head.pt").read_bytes() == before
    assert len(log(tmp_path)) == rows_before

    code, text = train(capsys, tmp_path, "--head", head, "--seed", "1", heads_root=a)
    assert code == 0, text
    (run_1,) = [r for r in runs(a) if r["head"] == head and r["seed"] == 1]
    assert run_1["run_id"] != run_a["run_id"]
    assert not same(saved(a, run_1), saved(a, run_a))


# --- run.json and the configs (FR-003, FR-004) -----------------------------------------------


def test_run_json_holds_the_contract(tmp_path, capsys):
    manifest = make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "linear", "proto", "mix", "--seed", "0")
    assert code == 0, text
    by = {r["head"]: r for r in runs(tmp_path / "heads")}
    digest = sha256(manifest)
    for head, run in by.items():
        assert list(run) == RUN_KEYS
        assert run["run_json_version"] == 2
        assert run["run_id"].startswith(f"{BACKBONE}-{RES}-{head}-cls-s0-")
        assert run["model_version"] == f"msv0.1+{BACKBONE}@{RES}.{head}.man-{digest[:6]}"
        assert run["classes"] == ["healthy", "rust", "anthracnose"]
        assert (run["quotable"], run["allow_test_only"], run["input_dim"]) == (True, False, DIM)
        assert list(run["backbone"]) == BACKBONE_KEYS
        assert list(run["train"]) == SIDE_KEYS and list(run["val"]) == SIDE_KEYS
        assert (run["train"]["split"], run["val"]["split"]) == ("train", "val")
        assert run["train"]["manifest_sha256"] == run["val"]["manifest_sha256"] == digest
        assert run["train"]["frozen"] is True and run["train"]["role"] == "train_eval"
        assert run["head_config"]["head"] == head
        assert run["head_config"]["sha256"] == sha256(CONFIGS / f"{head}.yaml")
        assert set(run["fit"]) >= FIT_KEYS
        assert isinstance(run["wallclock_s"], float) and run["wallclock_s"] >= 0
    lin, proto, mix = by["linear"], by["proto"], by["mix"]
    for trained in (lin, proto):
        assert trained["components"] is None
        assert trained["head_config"]["sampling"] == "class_balanced"
        assert 1 <= trained["fit"]["best_epoch"] <= trained["fit"]["epochs_run"] <= 50
    assert proto["head_config"]["prototypes_per_class"] == 4
    assert proto["head_config"]["tau_init"] == pytest.approx(0.07)
    assert proto["fit"]["tau"] > 0 and proto["fit"]["tau"] != pytest.approx(0.07, abs=1e-6)
    assert mix["head_config"]["weights"] == [0.5, 0.5]
    assert mix["components"] == [
        {"head": "linear", "run_id": lin["run_id"]},
        {"head": "proto", "run_id": proto["run_id"]},
    ]
    assert (mix["fit"]["epochs_run"], mix["fit"]["best_epoch"], mix["fit"]["history"]) == (
        0,
        None,
        [],
    )


def test_the_committed_configs_are_the_h8_recipe():
    """FR-003: one file per head, with H8 6.5's values."""
    cfg = {h: yaml.safe_load((CONFIGS / f"{h}.yaml").read_text(encoding="utf-8")) for h in HEADS}
    for head in ("linear", "proto"):
        c = cfg[head]
        assert c["head"] == head and c["optimizer"] == "adamw" and 5e-4 <= c["lr"] <= 1e-3
        assert (c["weight_decay"], c["batch_size"], c["max_epochs"]) == (1e-4, 256, 50)
        assert (c["loss"], c["focal_gamma"], c["label_smoothing"]) == ("focal", 2, 0.1)
        assert (c["sampling"], c["select"]) == ("class_balanced", "balanced_val_loss")
    assert (cfg["proto"]["prototypes_per_class"], cfg["proto"]["tau_init"]) == (4, 0.07)
    assert cfg["mix"]["head"] == "mix"
    assert (cfg["mix"]["components"], cfg["mix"]["weights"]) == (["linear", "proto"], [0.5, 0.5])


# --- the heads (US-2) --------------------------------------------------------------------------


def test_the_prototype_logit_is_the_nearest_prototype_over_tau():
    head = heads.build_head("proto", 3, 2, prototypes_per_class=2, tau_init=0.5)
    state = head.state_dict()
    assert tuple(state["prototypes"].shape) == (2, 2, 3) and tuple(state["log_tau"].shape) == ()
    assert math.exp(float(state["log_tau"])) == pytest.approx(0.5)
    assert "log_tau" in {n for n, p in head.named_parameters() if p.requires_grad}
    prototypes = torch.tensor([[[2.0, 0, 0], [0, 1, 0]], [[0, 0, 3.0], [1, 1, 0]]])
    head.load_state_dict({"prototypes": prototypes, "log_tau": torch.tensor(math.log(0.5))})
    with torch.no_grad():
        logits = head(torch.tensor([[1.0, 1.0, 0.0]]))
    # class 0: cosines 0.707 and 0.707; class 1: 0 and 1; each over tau = 0.5
    assert torch.allclose(logits, torch.tensor([[math.sqrt(0.5) / 0.5, 1 / 0.5]]), atol=1e-5)


def test_prototypes_start_at_the_per_class_k_means_centroids():
    """US-2.2: K centroids per class of its L2-normalised features, normalised, seeded."""
    rng = np.random.default_rng(0)
    d = 8
    x, y = [], []
    for c in range(2):
        for k in range(4):  # four tight clusters per class, along the basis vectors
            x.append(np.eye(d)[4 * c + k] + rng.normal(scale=0.05, size=(25, d)))
            y += [c] * 25
    x, y = np.concatenate(x).astype(np.float32), np.array(y)
    got = heads_train.init_prototypes(x, y, 2, 4, seed=0)
    p = np.asarray(torch.as_tensor(got).detach(), dtype=np.float64)
    assert p.shape == (2, 4, d)
    assert np.allclose(np.linalg.norm(p, axis=2), 1.0, atol=1e-5)
    for c in range(2):
        for k in range(4):
            assert (p[c] @ np.eye(d)[4 * c + k]).max() > 0.95
    again = np.asarray(torch.as_tensor(heads_train.init_prototypes(x, y, 2, 4, seed=0)).detach())
    assert np.array_equal(p, again.astype(np.float64))


@pytest.mark.parametrize("threads", [1, 4, None])
def test_the_prototype_start_does_not_depend_on_the_thread_count(threads):
    """US-1.2: the same seed, the same start, on any number of threads (on several, k-means
    adds its threads' partial sums in the order they finish)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12000, DIM)).astype(np.float32)
    y = np.repeat([0, 1, 2], 4000)
    with threadpool_limits(limits=1):
        reference = np.asarray(heads_train.init_prototypes(x, y, 3, 4, seed=0))
    for _ in range(2):
        with threadpool_limits(limits=threads):
            start = np.asarray(heads_train.init_prototypes(x, y, 3, 4, seed=0))
        assert np.array_equal(start, reference)


def test_the_mix_is_the_fixed_half_half_mixture_of_its_components(tmp_path, capsys):
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "mix", "--seed", "0")
    assert code == 0, text
    root = tmp_path / "heads"
    by = {r["head"]: heads.load_run(root / r["run_id"]) for r in runs(root)}
    x = unit_features(20)
    p = by["mix"].predict_proba(x)
    expected = 0.5 * by["linear"].predict_proba(x) + 0.5 * by["proto"].predict_proba(x)
    assert np.allclose(p, expected, atol=1e-6)
    z = by["mix"].logits(x)
    assert np.allclose(np.exp(z) / np.exp(z).sum(axis=1, keepdims=True), p, atol=1e-6)


# --- what steers training (US-3, US-4) -------------------------------------------------------


def test_class_balanced_sampling_draws_every_class_equally_often():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    order = np.asarray(
        heads_train.epoch_order(y, "class_balanced", torch.Generator().manual_seed(0))
    )
    assert len(order) == len(y)
    assert np.allclose(np.bincount(y[order], minlength=3) / len(order), 1 / 3, atol=0.05)
    again = heads_train.epoch_order(y, "class_balanced", torch.Generator().manual_seed(0))
    assert np.array_equal(order, np.asarray(again))
    uniform = np.asarray(heads_train.epoch_order(y, "uniform", torch.Generator().manual_seed(0)))
    assert sorted(uniform.tolist()) == list(range(len(y)))  # a permutation


def test_sampling_is_class_balanced_when_the_config_does_not_say(tmp_path, capsys):
    """US-4.1: on by default; run.json records the effective recipe."""
    make_manifest(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = yaml.safe_load((CONFIGS / "linear.yaml").read_text(encoding="utf-8"))
    del cfg["sampling"]
    (configs / "linear.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code, text = train(
        capsys, tmp_path, "--head", "linear", "--seed", "0", "--config-dir", str(configs)
    )
    assert code == 0, text
    (run,) = runs(tmp_path / "heads")
    assert run["head_config"]["sampling"] == "class_balanced"
    assert run["head_config"]["sha256"] == sha256(configs / "linear.yaml")


@pytest.mark.parametrize(
    "args",
    [("--val-split", "test"), ("--split", "test"), ("--val-split", "holdout_unknown")],
)
def test_a_test_or_held_out_split_neither_trains_nor_selects(tmp_path, capsys, args):
    """US-3.1: an exit with its reason, not a traceback, and nothing written."""
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "linear", "--seed", "0", *args)
    assert code == 2 and "test_split_access" in text
    assert nothing_written(tmp_path)


def test_early_stopping_reads_only_the_in_domain_validation_split(tmp_path, capsys):
    """US-3.2: the validation manifest is the training manifest."""
    alpha = make_manifest(tmp_path, "alpha")
    beta = make_manifest(tmp_path, "beta", seed=1)
    code, text = train(
        capsys, tmp_path, "--head", "linear", "--seed", "0", "--val-manifest", str(beta)
    )
    assert code == 2 and "val_not_in_domain" in text
    assert nothing_written(tmp_path)
    code, text = train(
        capsys, tmp_path, "--head", "linear", "--seed", "0", "--val-manifest", str(alpha)
    )
    assert code == 0, text


@pytest.mark.parametrize("head", HEADS)
def test_the_features_of_test_and_held_out_rows_do_not_change_the_weights(tmp_path, capsys, head):
    """US-3.4: the same cache key and manifest, other features on every test and held-out row."""
    manifest = make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", head, "--seed", "0", heads_root=tmp_path / "a")
    assert code == 0, text
    npz = next((tmp_path / "cache").rglob("alpha_v1.npz"))
    with np.load(npz) as z:
        x = z["cls"].astype(np.float64)
    hidden = [r["split"] in ("test", "holdout_unknown") for r in rows_of(manifest)]
    x[hidden] = np.random.default_rng(9).normal(size=(sum(hidden), DIM))
    write_cache(tmp_path / "cache", manifest, x)
    code, text = train(capsys, tmp_path, "--head", head, "--seed", "0", heads_root=tmp_path / "b")
    assert code == 0, text
    (run_a,) = [r for r in runs(tmp_path / "a") if r["head"] == head]
    (run_b,) = [r for r in runs(tmp_path / "b") if r["head"] == head]
    assert same(saved(tmp_path / "a", run_a), saved(tmp_path / "b", run_b))


# --- roles and input errors (US-5, FR-008) ---------------------------------------------------


def test_the_role_gate_holds_for_every_head(tmp_path, capsys):
    make_manifest(tmp_path, role="test_only")
    code, text = train(capsys, tmp_path, "--head", "proto", "--seed", "0")
    assert code == 2 and "role_not_trainable" in text
    assert nothing_written(tmp_path)
    code, text = train(
        capsys, tmp_path, "--head", "proto", "mix", "--seed", "0", "--allow-test-only"
    )
    assert code == 0, text
    assert [r["quotable"] for r in runs(tmp_path / "heads")] == [False, False, False]


def test_a_proto_class_needs_k_train_rows_and_the_error_comes_before_any_run(tmp_path, capsys):
    """US-5.2 and FR-008: every input is checked before the first run trains."""
    counts = {**COUNTS, "train": {**COUNTS["train"], "anthracnose": 3}}
    make_manifest(tmp_path, counts=counts)
    code, text = train(capsys, tmp_path, "--head", "linear", "proto", "--seed", "0")
    assert code == 2 and "too_few_rows" in text
    assert nothing_written(tmp_path)  # not even the linear run, which alone could train
    code, text = train(capsys, tmp_path, "--head", "linear", "--seed", "0")
    assert code == 0, text


def test_a_further_training_manifest_may_be_test_only_and_the_run_stays_quotable(tmp_path, capsys):
    """US-5.1 and FR-004 (N2's Makerere + iBean; the owner's decision of 2026-09-19): the first
    manifest's role decides; each manifest trains on its train split and validates on its own
    validation split; the sides and the model version name both manifests."""
    alpha = make_manifest(tmp_path, "alpha")
    beta = make_manifest(tmp_path, "beta", role="test_only", seed=1)
    gamma = make_manifest(tmp_path, "gamma", seed=2)
    make_manifest(
        tmp_path, "delta", role="holdout_unknown", counts={"holdout_unknown": {"unknown_als": 8}}
    )
    code, text = train(
        capsys, tmp_path, "--head", "linear", "--seed", "0", manifest=("alpha", "beta")
    )
    assert code == 0, text
    (run,) = runs(tmp_path / "heads")
    both = f"{sha256(alpha)}+{sha256(beta)}"
    assert run["quotable"] is True and run["train"]["role"] == "train_eval+test_only"
    assert run["train"]["manifest_sha256"] == run["val"]["manifest_sha256"] == both
    assert run["train"]["n"] == 2 * sum(COUNTS["train"].values())
    assert run["val"]["n"] == 2 * sum(COUNTS["val"].values())
    assert run["model_version"].endswith(f".man-{sha256(alpha)[:6]}-{sha256(beta)[:6]}")
    (row,) = log(tmp_path)
    assert row["dataset"] == "alpha+beta"

    code, text = train(
        capsys,
        tmp_path,
        "--head",
        "linear",
        "--seed",
        "0",
        "--val-manifest",
        str(beta),
        manifest=("alpha", "beta"),
    )
    assert code == 0, text
    only_beta = [r for r in runs(tmp_path / "heads") if r["val"]["manifest_sha256"] == sha256(beta)]
    assert len(only_beta) == 1 and only_beta[0]["train"]["manifest_sha256"] == both
    code, text = train(
        capsys,
        tmp_path,
        "--head",
        "linear",
        "--seed",
        "0",
        "--val-manifest",
        str(gamma),
        manifest=("alpha", "beta"),
    )
    assert code == 2 and "val_not_in_domain" in text
    for names in (("beta", "alpha"), ("alpha", "delta")):  # test_only first; a held-out set
        code, text = train(capsys, tmp_path, "--head", "linear", "--seed", "0", manifest=names)
        assert code == 2 and "role_not_trainable" in text, names


# --- the compute log (US-6) ------------------------------------------------------------------


def test_every_run_writes_its_seconds_to_the_compute_log(tmp_path, capsys):
    make_manifest(tmp_path)
    code, text = train(capsys, tmp_path, "--head", "linear", "proto", "mix", "--seed", "0")
    assert code == 0, text
    by_id = {r["run_id"]: r for r in runs(tmp_path / "heads")}
    rows = log(tmp_path)
    assert len(rows) == 3 and {r["head"]["run_id"] for r in rows} == set(by_id)
    for row in rows:
        run = by_id[row["head"]["run_id"]]
        assert (row["step"], row["n_images_or_runs"]) == ("train_head", 1)
        assert (row["backbone_id"], row["res"], row["dataset"]) == (BACKBONE, RES, "alpha")
        assert row["wallclock_s"] == run["wallclock_s"]
        assert {"run_id", "head", "tokens", "seed", "n_train", "n_val", "epochs_run"} <= set(
            row["head"]
        )
        assert (row["head"]["head"], row["head"]["tokens"], row["head"]["seed"]) == (
            run["head"],
            "cls",
            0,
        )
        assert "best_epoch" in row["head"]
    (mix,) = [r for r in rows if r["head"]["head"] == "mix"]
    components = by_id[mix["head"]["run_id"]]["components"]
    assert mix["head"]["components"] == [c["run_id"] for c in components]
