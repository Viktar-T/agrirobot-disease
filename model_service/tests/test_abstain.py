"""Spec 004, abstention and calibration (ms.abstain; work plan S4.4 and the W4 tasks).

Written before the implementation (spec-driven): every test that needs `ms.abstain` fails
with the same one-line message until the W4 "Scores" task, and the rows of US-6 to US-9 stay
red until the W4 "N3" task.

CPU only, on the synthetic manifests and caches of synthetic.py: "alpha", a train_eval
manifest with healthy, rust and anthracnose in train, val and test plus held-out angular
leaf spot, and "wm", a holdout_unknown manifest of white mould. No image, backbone or GPU is
needed, and the heads are the committed configs (configs/heads/<head>.yaml).
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from synthetic import BACKBONE, COUNTS, RES, direction, make_manifest, write_cache

import ms.eval as n_table
from ms import compute_log
from ms.data.manifests import UNKNOWN, load_manifest
from ms.eval import run as eval_run
from ms.heads import load_run
from ms.heads import train as heads_train

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "model_service" / "configs" / "abstain.yaml"
COVERAGES = (0.80, 0.90, 0.95)
SCORES = ("conf", "knn", "maha", "energy")
DECISION_SCORES = ("conf", "knn")
REASONS = ("low_confidence", "far_from_training")
#: abstain.json (FR-002), in order
FIT_KEYS = [
    "abstain_json_version", "run_id", "head", "tokens", "seed", "classes", "input_dim",
    "model_version", "quotable", "fitted_on", "scores", "thresholds", "temperature", "ece_bins",
    "config", "cache", "inputs_sha256", "wallclock_s", "device", "created_at", "builder", "notes",
]  # fmt: skip
FITTED_ON_KEYS = [
    "manifest", "manifest_sha256", "frozen", "train_split", "n_train", "val_split", "n_val",
    "class_counts",
]  # fmt: skip
#: US-3.3 refuses a slice with fewer than ceil(1 / (1 - c)) rows, 20 at c = 0.95, and the
#: shared fixture's validation slice is exactly that. Guard it here: synthetic.py belongs to
#: specs 003 and 005 too, and one row fewer would turn every fit test red with `too_few_rows`.
MIN_SLICE = math.ceil(round(1 / (1 - max(COVERAGES)), 9))
assert sum(COUNTS["val"].values()) >= MIN_SLICE, (
    f"synthetic COUNTS['val'] has {sum(COUNTS['val'].values())} rows; "
    f"coverage {max(COVERAGES)} needs {MIN_SLICE}"
)
#: a manifest of white mould only, the second held-out set (spec 001 FR-007)
WM_COUNTS = {"holdout_unknown": {"unknown_wm": 24}}
EPSILON = 1e-6


# --- the module under test -------------------------------------------------------------------


def _module(name: str):
    """ms.abstain or ms.abstain.fit; until they exist every test here fails with one message."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("ms.abstain", name):
            raise
    pytest.fail("ms.abstain does not exist yet (spec 004, W4)", pytrace=False)


def config(tmp: Path, **changes) -> Path:
    """An abstain.yaml in tmp with the defaults of FR-012 and `changes` applied."""
    cfg = {
        "k": 5,
        "coverages": list(COVERAGES),
        "default_coverage": 0.90,
        "ece_bins": 15,
        "epsilon": EPSILON,
        "n3": [
            {"manifest": "alpha_v1", "classes": ["unknown_als"]},
            {"manifest": "wm_v1", "classes": ["unknown_wm"]},
        ],
    }
    cfg.update(changes)
    path = tmp / "abstain.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def fit(capsys, root: Path, *args, heads_root: Path | None = None, cfg: Path | None = None):
    """python -m ms.abstain.fit <argv>, in process: (exit code, everything printed)."""
    module = _module("ms.abstain.fit")
    argv = [
        "--heads-root", str(heads_root or root / "heads"), "--cache-root", str(root / "cache"),
        "--config", str(cfg or config(root)), "--compute-log", str(root / "log.jsonl"),
        *(str(a) for a in args),
    ]  # fmt: skip
    try:
        code = module.main(argv)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def evaluate(capsys, root: Path, *args, cfg: Path | None = None):
    """make eval over the synthetic runs, with the abstention config of FR-012."""
    argv = [
        "--heads-root", str(root / "heads"), "--cache-root", str(root / "cache"),
        "--manifest-root", str(root / "manifests"), "--config", str(root / "eval.yaml"),
        "--abstain-config", str(cfg or config(root)), "--n-table", str(root / "n_table.jsonl"),
        "--md", str(root / "n_table.md"), "--compute-log", str(root / "log.jsonl"),
        *(str(a) for a in args),
    ]  # fmt: skip
    try:
        code = eval_run.main(argv)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# --- the synthetic world ---------------------------------------------------------------------


def world(root: Path, *, counts: dict | None = None, n2: bool = False) -> Path:
    """The manifests "alpha" (train_eval) and "wm" (holdout_unknown) with their caches, and an
    eval.yaml. With n2, a second train_eval manifest "beta" is the target of one direction, so
    that make eval has an out-of-domain target too (US-7.4)."""
    alpha = make_manifest(root, "alpha", counts=counts)
    make_manifest(root, "wm", counts=WM_COUNTS, role="holdout_unknown")
    directions = []
    if n2:
        make_manifest(root, "beta", counts=counts, seed=1)
        directions = [
            {
                "train": ["alpha_v1"],
                "test": "beta_v1",
                "split": "test",
                "classes": ["healthy", "rust"],
            }
        ]
    (root / "eval.yaml").write_text(
        yaml.safe_dump({"n2": directions}, sort_keys=False), encoding="utf-8"
    )
    return alpha


def move_far(root: Path, manifest: Path, chosen) -> None:
    """Rewrite a manifest's cache, moving every row for which `chosen(row)` is true onto a
    direction no trained class points in, so those rows are far from the training bank at
    every coverage."""
    m = load_manifest(manifest, None, "extract")
    npz = np.load(next((root / "cache").rglob(f"*{m.name}*.npz")), allow_pickle=False)
    x = npz["cls"].astype(np.float32)
    away = -sum(direction(c) for c in ("healthy", "rust", "anthracnose"))
    away = away / np.linalg.norm(away)
    for i, r in enumerate(m.rows):
        if chosen(r):
            x[i] = 10.0 * away
    write_cache(root / "cache", manifest, x)


def train(root: Path, head: str = "linear", seed: int = 0, *manifests: str) -> dict:
    """One head run on the synthetic manifests; its run.json."""
    names = manifests or ("alpha",)
    argv = [
        "--backbone", BACKBONE, "--res", str(RES),
        "--train-manifest", *(str(root / "manifests" / f"{m}_v1.jsonl") for m in names),
        "--head", head, "--seed", str(seed), "--cache-root", str(root / "cache"),
        "--heads-root", str(root / "heads"), "--compute-log", str(root / "log.jsonl"),
    ]  # fmt: skip
    assert heads_train.main(argv) == 0
    found = [
        json.loads((p / "run.json").read_text(encoding="utf-8"))
        for p in (root / "heads").iterdir()
        if (p / "run.json").is_file()
    ]
    # a mix run trains its two components first, so three folders exist (spec 003 US-2.3)
    return next(r for r in found if r["head"] == head and r["seed"] == seed)


def fitted(root: Path, run_id: str) -> dict:
    return json.loads((root / "heads" / run_id / "abstain.json").read_text(encoding="utf-8"))


def split_features(root: Path, run_meta: dict, manifest: str, split: str | None) -> np.ndarray:
    """The features a run reads for one split of one manifest, as ms.heads.features gives
    them: the reference the scores are checked against."""
    run = load_run(root / "heads" / run_meta["run_id"])
    m = load_manifest(root / "manifests" / f"{manifest}_v1.jsonl", split, "evaluate")
    return eval_run.Store(root / "cache").features(run, m, m.rows)


def rows(root: Path) -> list[dict]:
    return list(n_table.read_rows(root / "n_table.jsonl"))


def log(root: Path) -> list[dict]:
    return list(compute_log.read_rows(root / "log.jsonl"))


# --- reference scores (FR-004), so the tests do not trust the implementation -------------------


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def reference_knn(bank: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(unit(x)[:, None, :] - unit(bank)[None, :, :], axis=2)
    return np.sort(d, axis=1)[:, k - 1]


def reference_maha(bank: np.ndarray, y: np.ndarray, x: np.ndarray, n_classes: int) -> np.ndarray:
    b, z = unit(bank), unit(x)

    def gaussian(rows_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = rows_.mean(axis=0)
        centred = rows_ - mu
        cov = centred.T @ centred / max(len(rows_) - 1, 1)
        cov = cov + EPSILON * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return mu, np.linalg.inv(cov)

    means = [b[y == c].mean(axis=0) for c in range(n_classes)]
    centred = np.concatenate([b[y == c] - means[c] for c in range(n_classes)])
    shared = centred.T @ centred / max(len(b) - n_classes, 1)
    shared = shared + EPSILON * np.trace(shared) / shared.shape[0] * np.eye(shared.shape[0])
    precision = np.linalg.inv(shared)
    mu0, precision0 = gaussian(b)

    def quad(v: np.ndarray, mu: np.ndarray, p: np.ndarray) -> np.ndarray:
        d = v - mu
        return np.einsum("nd,de,ne->n", d, p, d)

    background = quad(z, mu0, precision0)
    return np.min([quad(z, mu, precision) - background for mu in means], axis=0)


def average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    for value in np.unique(v):
        tie = v == value
        ranks[tie] = ranks[tie].mean()
    return ranks


def reference_auroc(strange_known: np.ndarray, strange_unknown: np.ndarray) -> float:
    """Mann-Whitney over the strangeness orientation, unknown the positive class, average
    ranks for ties (US-6.2)."""
    n0, n1 = len(strange_known), len(strange_unknown)
    ranks = average_ranks(np.concatenate([strange_known, strange_unknown]))
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reference_ece(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        take = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if take.any():
            total += take.mean() * abs((pred[take] == y[take]).mean() - conf[take].mean())
    return float(total)


# --- US-1: the scores -------------------------------------------------------------------------


def test_the_module_names_its_scores_and_reasons():
    """US-1, US-4.1, FR-013: four scores, two of them decide, two reasons."""
    abstain = _module("ms.abstain")
    assert abstain.SCORES == SCORES
    assert abstain.DECISION_SCORES == DECISION_SCORES
    assert abstain.REASONS == REASONS
    assert tuple(abstain.COVERAGES) == COVERAGES


@pytest.mark.parametrize("head", ["linear", "proto", "mix"])
def test_conf_is_the_head_s_largest_logit_and_energy_is_minus_logsumexp(tmp_path, capsys, head):
    """US-1.1: both are read off the logits the head already has, for all three heads."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path, head=head)
    assert fit(capsys, tmp_path)[0] == 0
    run = load_run(tmp_path / "heads" / meta["run_id"])
    x = split_features(tmp_path, meta, "alpha", "test")
    got = abstain.scores(abstain.load_fit(tmp_path / "heads" / meta["run_id"]), run, x)
    z = run.logits(x)
    assert np.allclose(got["conf"], z.max(axis=1), atol=1e-6)
    assert np.allclose(got["energy"], -np.log(np.exp(z).sum(axis=1)), atol=1e-5)
    assert all(np.isfinite(got[s]).all() and got[s].shape == (len(x),) for s in SCORES)


def test_knn_is_the_distance_to_the_kth_nearest_training_feature(tmp_path, capsys):
    """US-1.1 and FR-004: the bank is the training rows, L2-normalised, k from the config."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path, cfg=config(tmp_path, k=7))[0] == 0
    run = load_run(tmp_path / "heads" / meta["run_id"])
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    bank = split_features(tmp_path, meta, "alpha", "train")
    x = split_features(tmp_path, meta, "alpha", "test")
    assert np.allclose(abstain.scores(loaded, run, x)["knn"], reference_knn(bank, x, 7), atol=1e-5)


def test_maha_is_the_relative_mahalanobis_distance(tmp_path, capsys):
    """US-1.1 and FR-004: the class distance minus the background Gaussian's."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    run = load_run(tmp_path / "heads" / meta["run_id"])
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    bank = split_features(tmp_path, meta, "alpha", "train")
    m = load_manifest(tmp_path / "manifests" / "alpha_v1.jsonl", "train", "evaluate")
    y = np.array([run.classes.index(r["class_km2"]) for r in m.rows])
    x = split_features(tmp_path, meta, "alpha", "test")
    want = reference_maha(bank, y, x, len(run.classes))
    assert np.allclose(abstain.scores(loaded, run, x)["maha"], want, rtol=1e-4, atol=1e-4)


def test_the_held_out_rows_are_stranger_than_the_test_rows(tmp_path, capsys):
    """US-1.2: the strangeness orientation is -conf, +knn, +maha, +energy.

    The angular-leaf-spot rows are strange in all four. The white-mould rows of the fixture
    are strange only in the two distance scores: their planted direction leans on a trained
    class, so the head is confident about them. That is the case abstention exists for, and
    why the decision does not rest on confidence alone (H8 §6.6)."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    run = load_run(tmp_path / "heads" / meta["run_id"])
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    known = abstain.scores(loaded, run, split_features(tmp_path, meta, "alpha", "test"))
    als = abstain.scores(loaded, run, split_features(tmp_path, meta, "alpha", "holdout_unknown"))
    wm = abstain.scores(loaded, run, split_features(tmp_path, meta, "wm", None))
    assert als["conf"].mean() < known["conf"].mean()
    for score in ("knn", "maha", "energy"):
        assert als[score].mean() > known[score].mean()
    for score in ("knn", "maha"):
        assert wm[score].mean() > known[score].mean()


def test_a_joined_token_type_is_renormalised_before_a_distance(tmp_path, capsys):
    """US-1.3: cls+meanpatch joins two unit vectors; the distance sees a unit vector again."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    argv = [
        "--backbone", BACKBONE, "--res", str(RES), "--tokens", "cls+meanpatch",
        "--train-manifest", str(tmp_path / "manifests" / "alpha_v1.jsonl"),
        "--head", "linear", "--seed", "0", "--cache-root", str(tmp_path / "cache"),
        "--heads-root", str(tmp_path / "heads"), "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    assert heads_train.main(argv) == 0
    folder = next(p for p in (tmp_path / "heads").iterdir() if (p / "run.json").is_file())
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert meta["tokens"] == "cls+meanpatch"
    assert fit(capsys, tmp_path, cfg=config(tmp_path, k=5))[0] == 0
    run = load_run(folder)
    bank = split_features(tmp_path, meta, "alpha", "train")
    x = split_features(tmp_path, meta, "alpha", "test")
    assert np.allclose(np.linalg.norm(bank, axis=1), math.sqrt(2), atol=1e-3)
    got = abstain.scores(abstain.load_fit(folder), run, x)["knn"]
    assert np.allclose(got, reference_knn(bank, x, 5), atol=1e-5)


# --- US-2: fitted on the in-domain validation slice only ---------------------------------------


def test_the_fit_reads_only_train_and_val_rows(tmp_path, capsys, monkeypatch):
    """US-2.1 and SC-2: the leakage rule. Nothing is read with purpose train or select but the
    run's own train and val splits, and no test or held-out split is read at all."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    seen: list[tuple] = []
    import ms.data.manifests as manifests

    real = manifests.load_manifest
    targets = ["ms.data.manifests", "ms.abstain", "ms.abstain.fit"]
    for module in (importlib.import_module(name) for name in targets):
        monkeypatch.setattr(
            module,
            "load_manifest",
            lambda path, split, purpose: (
                seen.append((Path(path).stem, split, purpose)),
                real(path, split, purpose),
            )[1],
            raising=False,
        )
    assert fit(capsys, tmp_path)[0] == 0
    assert seen, "the fit read no manifest"
    assert {purpose for _, _, purpose in seen} <= {"train", "select"}
    assert {split for _, split, _ in seen} <= {"train", "val"}


def test_a_run_that_records_a_test_split_is_refused(tmp_path, capsys):
    """US-2.2 and FR-011: test_split_access, and nothing is written."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    folder = tmp_path / "heads" / meta["run_id"]
    meta["train"]["split"] = "test"
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    code, text = fit(capsys, tmp_path)
    assert code == 2 and "test_split_access" in text
    assert not (folder / "abstain.json").exists()
    assert not [r for r in log(tmp_path) if r["step"] == "fit_abstain"]


def test_several_training_manifests_join_their_bank_and_their_slice(tmp_path, capsys):
    """US-2.3: the N2 direction that trains on two manifests (spec 003 US-5.1)."""
    _module("ms.abstain")
    world(tmp_path)
    make_manifest(tmp_path, "beta")
    meta = train(tmp_path, "linear", 0, "alpha", "beta")
    assert fit(capsys, tmp_path)[0] == 0
    on = fitted(tmp_path, meta["run_id"])["fitted_on"]
    assert list(on) == FITTED_ON_KEYS
    assert on["manifest"].count("+") == 1 and on["manifest_sha256"].count("+") == 1
    assert on["n_train"] == 2 * sum(COUNTS["train"].values())
    assert on["n_val"] == 2 * sum(COUNTS["val"].values())


# --- US-3: a declared coverage ------------------------------------------------------------------


def test_each_threshold_keeps_its_declared_share_of_the_slice(tmp_path, capsys):
    """US-3.1: tau_conf is the (1 - c) quantile of conf, tau_knn the c quantile of knn."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    slice_scores = abstain.scores(loaded, run, split_features(tmp_path, meta, "alpha", "val"))
    n = len(slice_scores["conf"])
    for c in COVERAGES:
        tau = fitted(tmp_path, meta["run_id"])["thresholds"][f"{c:.2f}"]
        assert abs((slice_scores["conf"] >= tau["tau_conf"]).sum() / n - c) <= 1 / n
        assert abs((slice_scores["knn"] <= tau["tau_knn"]).sum() / n - c) <= 1 / n
        assert abs((slice_scores["maha"] <= tau["tau_maha"]).sum() / n - c) <= 1 / n


def test_coverage_achieved_is_the_joint_rule_and_never_above_the_declared_one(tmp_path, capsys):
    """US-3.2: two rejection sets are united, so the joint rule keeps at most c."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    scores = abstain.scores(loaded, run, split_features(tmp_path, meta, "alpha", "val"))
    for c in COVERAGES:
        tau = fitted(tmp_path, meta["run_id"])["thresholds"][f"{c:.2f}"]
        kept = (scores["conf"] >= tau["tau_conf"]) & (scores["knn"] <= tau["tau_knn"])
        assert tau["coverage_achieved"] == pytest.approx(kept.mean(), abs=1e-6)
        assert 0 < tau["coverage_achieved"] <= c + 1e-9


def test_a_slice_too_small_for_the_strictest_coverage_is_refused(tmp_path, capsys):
    """US-3.3: 0.95 needs 20 validation rows; 8 are not a coverage."""
    _module("ms.abstain")
    counts = dict(COUNTS, val={"healthy": 4, "rust": 2, "anthracnose": 2})
    world(tmp_path, counts=counts)
    meta = train(tmp_path)
    code, text = fit(capsys, tmp_path)
    assert code == 2 and "too_few_rows" in text
    assert not (tmp_path / "heads" / meta["run_id"] / "abstain.json").exists()


def test_a_constant_score_is_refused(tmp_path, capsys):
    """US-3.3: a threshold on a score that never moves is not a coverage."""
    _module("ms.abstain")
    alpha = world(tmp_path)
    meta = train(tmp_path)
    m = load_manifest(alpha, None, "extract")
    write_cache(tmp_path / "cache", alpha, np.ones((len(m.rows), 16), dtype=np.float32))
    code, text = fit(capsys, tmp_path, "--run", meta["run_id"], "--force")
    assert code == 2 and "degenerate_score" in text


# --- US-4: the decision -------------------------------------------------------------------------


def test_either_score_crossing_abstains_and_names_its_reason(tmp_path, capsys):
    """US-4.1 and US-4.2: low_confidence when conf crossed, far_from_training when knn did,
    and low_confidence when both do."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    tau = fitted(tmp_path, meta["run_id"])["thresholds"]["0.90"]
    lo, hi = tau["tau_conf"], tau["tau_knn"]
    cases = {
        (lo + 1.0, hi - 1.0): ("predict", None),
        (lo - 1.0, hi - 1.0): ("abstain", "low_confidence"),
        (lo + 1.0, hi + 1.0): ("abstain", "far_from_training"),
        (lo - 1.0, hi + 1.0): ("abstain", "low_confidence"),
    }
    for (conf, knn), want in cases.items():
        got = abstain.decide(loaded, {"conf": conf, "knn": knn, "maha": 0.0, "energy": 0.0}, 0.90)
        assert got == want


def test_a_row_exactly_at_a_threshold_is_predicted(tmp_path, capsys):
    """Edge case: the comparisons are strict."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    tau = fitted(tmp_path, meta["run_id"])["thresholds"]["0.90"]
    at = {"conf": tau["tau_conf"], "knn": tau["tau_knn"], "maha": 0.0, "energy": 0.0}
    assert abstain.decide(loaded, at, 0.90) == ("predict", None)


def test_the_second_opinion_never_decides(tmp_path, capsys):
    """US-4.3: maha and energy are recorded, not consulted."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    loaded = abstain.load_fit(tmp_path / "heads" / meta["run_id"])
    tau = fitted(tmp_path, meta["run_id"])["thresholds"]["0.90"]
    inside = {"conf": tau["tau_conf"] + 1.0, "knn": tau["tau_knn"] - 1.0}
    for extreme in (1e6, -1e6):
        got = abstain.decide(loaded, {**inside, "maha": extreme, "energy": extreme}, 0.90)
        assert got == ("predict", None)


def test_abstain_is_reachable_on_the_held_out_unknowns(tmp_path, capsys):
    """US-4.5 and the work plan's S4.4 row: the decision is reachable, or the number is empty."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    scores = abstain.scores(loaded, run, split_features(tmp_path, meta, "wm", None))
    decisions = [
        abstain.decide(loaded, {s: scores[s][i] for s in SCORES}, 0.90)[0]
        for i in range(len(scores["conf"]))
    ]
    assert decisions.count("abstain") > 0


# --- US-5: temperature scaling -------------------------------------------------------------------


def test_the_temperature_is_fitted_on_the_slice_and_improves_its_likelihood(tmp_path, capsys):
    """US-5.1 and FR-007."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    t = fitted(tmp_path, meta["run_id"])["temperature"]
    assert set(t) == {"T", "nll_before", "nll_after", "iterations"}
    assert t["T"] > 0 and 0.01 <= t["T"] <= 100
    assert t["nll_after"] <= t["nll_before"] + 1e-9


def test_temperature_scales_the_probabilities_and_changes_no_decision(tmp_path, capsys):
    """US-5.2 and US-5.3, SC-3: W3's N1 and N2 are the same numbers after S4.4."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    x = split_features(tmp_path, meta, "alpha", "test")
    p = abstain.probabilities(loaded, run, x)
    assert p.shape == (len(x), len(run.classes))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p.argmax(axis=1) == run.predict_proba(x).argmax(axis=1)).all()
    if abs(loaded.temperature - 1.0) > 1e-3:
        assert not np.allclose(p, run.predict_proba(x), atol=1e-4)


# --- US-6: N3, unknown recall and AUROC -----------------------------------------------------------


def n3_rows(root: Path) -> list[dict]:
    return [r for r in rows(root) if r["number"] == "N3"]


def test_n3_has_a_row_set_per_held_out_class(tmp_path, capsys):
    """US-6.1 and spec 005 US-4.2: angular leaf spot and white mould, never pooled."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    code, text = evaluate(capsys, tmp_path)
    assert code == 0, text
    found = n3_rows(tmp_path)
    assert found, "make eval wrote no N3 row"
    assert {tuple(r["classes"]) for r in found} == {("unknown_als",), ("unknown_wm",)}
    assert {r["test_split"] for r in found} == {"holdout_unknown"}
    assert all(set(r["classes"]) <= set(UNKNOWN) for r in found)


def test_unknown_recall_has_one_row_per_declared_coverage(tmp_path, capsys):
    """US-6.1: the fraction that abstains at the operating point, for each declared coverage."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    for klass in ("unknown_als", "unknown_wm"):
        found = [
            r
            for r in n3_rows(tmp_path)
            if r["metric"] == "unknown_recall" and klass in r["classes"]
        ]
        assert sorted(r["coverage"] for r in found) == sorted(COVERAGES)
        assert all(0.0 <= r["value"] <= 1.0 and r["ci_low"] is None for r in found)
        assert all(r["test_manifest"].count("+") == 0 for r in found)


def test_auroc_rows_carry_coverage_one_and_both_manifests(tmp_path, capsys):
    """US-6.1 and US-6.2: an AUROC uses no threshold, and the verdict reads coverage 1.0 rows
    (spec 005 US-6.1). The row names the knowns and the unknowns."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    found = [r for r in n3_rows(tmp_path) if r["metric"].startswith("auroc:")]
    assert {r["metric"] for r in found} == {f"auroc:{s}" for s in SCORES}
    for r in found:
        assert r["coverage"] == 1.0
        assert 0.0 <= r["value"] <= 1.0
        # the knowns' manifest and the unknowns', here one each: this run trained on one
        # manifest, so its knowns are the rows N1 scores. A run trained on two names three.
        assert r["test_manifest"].count("+") == 1
        assert r["test_manifest_sha256"].count("+") == 1
        assert r["n"] > sum(COUNTS["holdout_unknown"].values())


def test_auroc_is_the_mann_whitney_statistic_on_the_strangeness_orientation(tmp_path, capsys):
    """US-6.2: unknown is the positive class, ties get average ranks."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    known = abstain.scores(loaded, run, split_features(tmp_path, meta, "alpha", "test"))
    unknown = abstain.scores(loaded, run, split_features(tmp_path, meta, "wm", None))
    for score in SCORES:
        sign = -1.0 if score == "conf" else 1.0
        want = reference_auroc(sign * known[score], sign * unknown[score])
        (got,) = [
            r["value"]
            for r in n3_rows(tmp_path)
            if r["metric"] == f"auroc:{score}" and r["classes"] == ["unknown_wm"]
        ]
        assert got == pytest.approx(round(want, 4), abs=1e-4)


def test_every_row_the_abstention_writes_validates(tmp_path, capsys):
    """FR-010 and spec 005 SC-1: the rows go through append_rows, so they pass validate_row."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    for r in rows(tmp_path):
        assert n_table.validate_row(r) == []


def test_a_source_with_no_rows_is_reported_and_costs_no_other_row(tmp_path, capsys):
    """US-6.5: a held-out source that the target has no row of is a printed reason, not an
    exit; the other sources' rows are written as usual."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    empty = config(
        tmp_path,
        n3=[
            {"manifest": "wm_v1", "classes": ["unknown_als"]},  # white mould holds no ALS row
            {"manifest": "wm_v1", "classes": ["unknown_wm"]},
        ],
    )
    code, text = evaluate(capsys, tmp_path, cfg=empty)
    assert code == 0, text
    assert "unknown_als" in text
    assert {tuple(r["classes"]) for r in n3_rows(tmp_path)} == {("unknown_wm",)}


def test_a_mix_run_gets_no_auroc_energy_row(tmp_path, capsys):
    """US-6.1 and US-1.1: mix's logits are log p, so its energy is zero on every row; an AUROC
    over float rounding is not a number, and the reason is printed instead."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path, head="mix")
    assert fit(capsys, tmp_path)[0] == 0
    code, text = evaluate(capsys, tmp_path)
    assert code == 0, text
    mix = [r for r in n3_rows(tmp_path) if r["run_id"] == meta["run_id"]]
    assert mix, "the mix run got no N3 row at all"
    assert {r["metric"] for r in mix if r["metric"].startswith("auroc:")} == {
        f"auroc:{s}" for s in ("conf", "knn", "maha")
    }
    assert "constant_energy" in text


# --- US-7 and US-8: the curve, and who abstains ------------------------------------------------


def test_every_target_has_rows_at_the_declared_coverages(tmp_path, capsys):
    """US-7.1, US-7.2 and SC-5: four points per curve, the declared coverages and 1.0."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    n1 = [r for r in rows(tmp_path) if r["number"] == "N1" and r["seed"] is not None]
    assert {r["coverage"] for r in n1} == {*COVERAGES, 1.0}
    for c in COVERAGES:
        at = {r["metric"] for r in n1 if r["coverage"] == c}
        assert "selective_risk" in at and "abstention_rate" in at and "macro_f1" in at
    curve = {r["coverage"] for r in n1 if r["metric"] == "selective_risk"}
    assert curve == {*COVERAGES, 1.0}, "the curve needs its no-abstention end (US-7.2)"
    assert {r["coverage"] for r in n1 if r["metric"] == "abstention_rate"} == set(COVERAGES)


def test_per_class_abstention_rides_on_the_head_s_own_classes(tmp_path, capsys):
    """US-8.1: an N1 row, because an N3 row's classes are the held-out unknowns."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    found = [r for r in rows(tmp_path) if r["metric"].startswith("abstention_rate:")]
    assert {r["metric"] for r in found} == {f"abstention_rate:{c}" for c in meta["classes"]}
    assert {r["number"] for r in found} == {"N1"}
    assert all(0.0 <= r["value"] <= 1.0 for r in found)


def test_the_coverage_one_numbers_do_not_move(tmp_path, capsys):
    """SC-3: fitting an operating point changes no number already in the table."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert evaluate(capsys, tmp_path)[0] == 0
    before = {
        (r["run_id"], r["metric"]): r["value"] for r in rows(tmp_path) if r["coverage"] == 1.0
    }
    assert before
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    after = {(r["run_id"], r["metric"]): r["value"] for r in rows(tmp_path) if r["coverage"] == 1.0}
    assert {k: after[k] for k in before} == before
    # ...and the rows of the operating point really were written. On the real table every run
    # already has its coverage 1.0 rows from W3, and ms.eval.run skips a part whose identities
    # it has seen (run.py identities(), coverage 1.0): a fit must not be skipped with them.
    assert {r["coverage"] for r in rows(tmp_path) if r["number"] == "N1"} == {*COVERAGES, 1.0}
    assert n3_rows(tmp_path)


def test_an_out_of_domain_target_gets_the_curve_and_its_calibration(tmp_path, capsys):
    """US-7.4 and US-9.2: the N2 target is the out-of-domain half, and it is the half the
    numbers exist for. Calibration is what breaks first under shift (H8 §6.6)."""
    _module("ms.abstain")
    world(tmp_path, n2=True)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    code, text = evaluate(capsys, tmp_path)
    assert code == 0, text
    n2 = [r for r in rows(tmp_path) if r["number"] == "N2" and r["seed"] is not None]
    assert n2, "make eval scored no N2 direction"
    assert {r["coverage"] for r in n2 if r["metric"] == "selective_risk"} == {*COVERAGES, 1.0}
    assert {r["coverage"] for r in n2 if r["metric"] == "abstention_rate"} == set(COVERAGES)
    assert [r for r in n2 if r["metric"] == "ece" and r["coverage"] == 1.0]
    assert "beta_v1" in {Path(r["test_manifest"]).stem for r in n2}


def test_a_class_with_no_accepted_row_is_reported_and_the_eval_carries_on(tmp_path, capsys):
    """US-7.3: a sparse class at one coverage is not an input error. On the real data
    anthracnose is the class this will happen to first."""
    _module("ms.abstain")
    alpha = world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    move_far(tmp_path, alpha, lambda r: r["split"] == "test" and r["class_km2"] == "anthracnose")
    code, text = evaluate(capsys, tmp_path)
    assert code == 0, text
    n1 = [r for r in rows(tmp_path) if r["number"] == "N1"]
    for c in COVERAGES:
        at = {r["metric"] for r in n1 if r["coverage"] == c}
        assert "macro_f1" not in at and "recall:anthracnose" not in at
        assert "selective_risk" in at and "abstention_rate" in at
    assert "anthracnose" in text
    assert {r["metric"] for r in n1 if r["coverage"] == 1.0} >= {"macro_f1", "recall:anthracnose"}


def test_the_five_seeds_of_a_coverage_row_make_one_aggregate(tmp_path, capsys):
    """US-7.2 and FR-010: `n` counts the scored rows, not the accepted ones. An `n` that
    followed the accepted rows would differ by seed, and spec 005's aggregate — which groups
    on every field outside its per-seed list, `n` included — would silently emit nothing."""
    _module("ms.abstain")
    world(tmp_path)
    for seed in range(5):
        train(tmp_path, "linear", seed)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    for metric in ("selective_risk", "macro_f1", "abstention_rate"):
        agg = [
            r
            for r in rows(tmp_path)
            if r["seed"] is None and r["metric"] == metric and r["coverage"] == 0.90
        ]
        assert len(agg) == 1, f"{metric} at coverage 0.90 has {len(agg)} aggregate rows, want 1"
        assert agg[0]["ci_low"] is not None and agg[0]["ci_low"] <= agg[0]["ci_high"]
        assert agg[0]["run_id"].count("+") == 4


def test_a_second_make_eval_appends_nothing(tmp_path, capsys):
    """SC-6 and spec 005 SC-3: the abstention rows are appended once, like every other row."""
    _module("ms.abstain")
    world(tmp_path)
    train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    before = rows(tmp_path)
    assert before
    code, text = evaluate(capsys, tmp_path)
    assert code == 0 and "0 row(s) added" in text
    assert rows(tmp_path) == before


# --- US-9: calibration error ---------------------------------------------------------------------


def test_ece_is_written_in_domain_with_the_declared_bins(tmp_path, capsys):
    """US-9.1 and US-9.2: 15 equal-width bins over the largest probability, coverage 1.0."""
    abstain = _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    assert evaluate(capsys, tmp_path)[0] == 0
    found = [r for r in rows(tmp_path) if r["metric"] == "ece"]
    assert found and all(r["coverage"] == 1.0 for r in found)
    assert not [r for r in found if r["test_split"] == "holdout_unknown"]
    folder = tmp_path / "heads" / meta["run_id"]
    loaded, run = abstain.load_fit(folder), load_run(folder)
    m = load_manifest(tmp_path / "manifests" / "alpha_v1.jsonl", "test", "evaluate")
    x = split_features(tmp_path, meta, "alpha", "test")
    y = np.array([run.classes.index(r["class_km2"]) for r in m.rows])
    want = reference_ece(abstain.probabilities(loaded, run, x), y, 15)
    (got,) = [r["value"] for r in found if r["test_split"] == "test"]
    assert got == pytest.approx(round(want, 4), abs=1e-4)


# --- US-10: reproducible and idempotent -----------------------------------------------------------


def test_the_same_inputs_give_the_same_file_and_a_second_fit_does_nothing(tmp_path, capsys):
    """US-10.1 and US-10.2: identical but for the two fields that record when it ran."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    path = tmp_path / "heads" / meta["run_id"] / "abstain.json"
    first, rows_before = path.read_bytes(), len(log(tmp_path))
    code, text = fit(capsys, tmp_path)
    assert code == 0 and "nothing to do" in text
    assert path.read_bytes() == first and len(log(tmp_path)) == rows_before
    code, text = fit(capsys, tmp_path, "--force")
    assert code == 0, text
    timing = ("wallclock_s", "created_at")
    again = json.loads(path.read_text(encoding="utf-8"))
    was = json.loads(first.decode("utf-8"))
    assert {k: v for k, v in again.items() if k not in timing} == {
        k: v for k, v in was.items() if k not in timing
    }


def test_a_changed_config_is_a_new_fit(tmp_path, capsys):
    """US-10.3 and FR-003: the config's sha256 is an input."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path, cfg=config(tmp_path, k=5))[0] == 0
    before = fitted(tmp_path, meta["run_id"])
    assert fit(capsys, tmp_path, cfg=config(tmp_path, k=9))[0] == 0
    after = fitted(tmp_path, meta["run_id"])
    assert after["inputs_sha256"] != before["inputs_sha256"]
    assert after["scores"]["knn"]["k"] == 9


# --- the artefact, the config and the checks (FR-002, FR-011, FR-012, FR-014) ---------------------


def test_abstain_json_holds_the_fields_of_fr_002_in_order(tmp_path, capsys):
    """FR-001 and FR-002: one file, beside the run, with these keys in this order."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    folder = tmp_path / "heads" / meta["run_id"]
    assert {p.name for p in folder.iterdir()} == {"head.pt", "run.json", "abstain.json"}
    got = fitted(tmp_path, meta["run_id"])
    assert list(got) == FIT_KEYS
    assert got["abstain_json_version"] == 1
    assert got["run_id"] == meta["run_id"] and got["classes"] == meta["classes"]
    assert sorted(got["thresholds"]) == [f"{c:.2f}" for c in COVERAGES]
    assert sorted(got["scores"]) == sorted(SCORES)
    assert got["scores"]["conf"]["higher_is_stranger"] is False
    assert all(got["scores"][s]["higher_is_stranger"] is True for s in ("knn", "maha", "energy"))
    assert got["scores"]["knn"]["k"] == 5 and got["scores"]["maha"]["epsilon"] == EPSILON
    assert got["ece_bins"] == 15


def test_the_committed_config_parses_with_the_keys_of_fr_012():
    """FR-012: one file, the shape configs/eval.yaml uses for its directions."""
    assert CONFIG.exists(), f"{CONFIG} does not exist yet (spec 004, W4)"
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert set(cfg) == {"k", "coverages", "default_coverage", "ece_bins", "epsilon", "n3"}
    assert 10 <= cfg["k"] <= 50
    assert [float(c) for c in cfg["coverages"]] == list(COVERAGES)
    assert cfg["default_coverage"] == 0.90 and cfg["ece_bins"] == 15
    sources = {tuple(d["classes"]): d["manifest"] for d in cfg["n3"]}
    assert sources == {("unknown_als",): "makerere_v1", ("unknown_wm",): "swm_v1"}


def test_an_input_error_is_found_before_anything_is_written(tmp_path, capsys):
    """FR-011: every input is checked before the first fit."""
    _module("ms.abstain")
    world(tmp_path)
    meta = train(tmp_path)
    code, text = fit(capsys, tmp_path, cfg=config(tmp_path, coverages=[0.9, 1.5]))
    assert code == 2 and "bad_config" in text
    assert not (tmp_path / "heads" / meta["run_id"] / "abstain.json").exists()
    assert not [r for r in log(tmp_path) if r["step"] == "fit_abstain"]
    code, text = fit(capsys, tmp_path, "--run", "no-such-run")
    assert code == 2 and "missing_run" in text


def test_a_fit_writes_one_compute_log_row(tmp_path, capsys):
    """FR-014 and N7: step fit_abstain, one row per fit, none for a fit already there."""
    _module("ms.abstain")
    assert "fit_abstain" in compute_log.STEPS
    world(tmp_path)
    meta = train(tmp_path)
    assert fit(capsys, tmp_path)[0] == 0
    fits = [r for r in log(tmp_path) if r["step"] == "fit_abstain"]
    assert len(fits) == 1
    (row,) = fits
    assert row["backbone_id"] == BACKBONE and row["res"] == RES
    assert row["n_images_or_runs"] == 1 and row["wallclock_s"] >= 0
    assert set(row["abstain"]) == {"run_id", "k", "n_train", "n_val", "T"}
    assert row["abstain"]["run_id"] == meta["run_id"]


# --- FR-009: the metric names the N-table gains ---------------------------------------------------


def n3_row(**changes) -> dict:
    base = {
        "ts": "2026-10-06T10:00:00Z", "number": "N3", "metric": "unknown_recall", "value": 0.7,
        "ci_low": None, "ci_high": None, "n": 40, "backbone_id": "dinov2_l14_reg", "res": 224,
        "token_type": "cls", "head": "linear", "seed": 0,
        "train_manifest": "data/manifests/tanzania_v1.jsonl", "train_manifest_sha256": "a" * 64,
        "train_split": "train", "test_manifest": "data/manifests/swm_v1.jsonl",
        "test_manifest_sha256": "b" * 64, "test_split": "holdout_unknown",
        "split_rule": "holdout_unknown", "coverage": 0.9, "classes": ["unknown_wm"],
        "run_id": "dinov2_l14_reg-224-linear-cls-s0-0123abcd", "model_version": None,
        "cache_key": "2b017210105b0b1f", "quotable": True, "superseded": None, "notes": None,
    }  # fmt: skip
    base.update(changes)
    return {k: base[k] for k in n_table.FIELDS}


@pytest.mark.parametrize(
    "metric",
    ["unknown_recall", "auroc:conf", "auroc:knn", "auroc:maha", "auroc:energy", "selective_risk"],
)
def test_the_new_metric_names_validate(metric):
    """FR-009: spec 005 FR-005's vocabulary gains S4.4's names."""
    assert n_table.validate_row(n3_row(metric=metric)) == []


@pytest.mark.parametrize("metric", ["auroc", "auroc:entropy", "unknown_recall:conf", "ece:15"])
def test_a_malformed_new_metric_is_rejected(metric):
    """FR-009: the qualifier of auroc is a score name, and the others take none."""
    problems = n_table.validate_row(n3_row(metric=metric))
    assert [p.split(":")[0] for p in problems] == ["bad_metric"]


def test_abstention_rate_takes_a_class_of_the_row_or_none():
    """FR-009: the qualifier kind is 'class?'."""
    known = {"number": "N1", "test_split": "test", "classes": ["healthy", "rust"]}
    assert n_table.validate_row(n3_row(metric="abstention_rate", **known)) == []
    assert n_table.validate_row(n3_row(metric="abstention_rate:rust", **known)) == []
    problems = n_table.validate_row(n3_row(metric="abstention_rate:unknown_als", **known))
    assert [p.split(":")[0] for p in problems] == ["bad_metric"]


def test_ece_belongs_to_a_scored_split_at_full_coverage():
    """US-9: an ece row is written at coverage 1.0, and N3 keeps its held-out classes."""
    known = {"number": "N1", "test_split": "test", "classes": ["healthy", "rust"]}
    assert n_table.validate_row(n3_row(metric="ece", coverage=1.0, **known)) == []
    problems = n_table.validate_row(n3_row(metric="ece", classes=["healthy"], coverage=1.0))
    assert [p.split(":")[0] for p in problems] == ["n3_not_held_out"]
