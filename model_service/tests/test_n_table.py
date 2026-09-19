"""Spec 005, the N-table (ms.eval; work plan S4.5): the row schema and its checks, the
five-seed aggregate, N1's rows, the table's invariants and its rendering.

CPU only. The row checks need no data. The N1 test trains linear heads on a synthetic
manifest and cache (synthetic.py) and scores them with ms.eval.run. The verdict's tests
(spec 005 US-6) come with the W4 task.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from synthetic import BACKBONE, COUNTS, RES, make_manifest, rows_of, sha256

import ms.eval as n_table
from ms.eval import run as eval_run
from ms.heads import train as heads_train

A, B = "a" * 64, "b" * 64
MAKERERE, TANZANIA = "data/manifests/makerere_v1.jsonl", "data/manifests/tanzania_v1.jsonl"
T_975_4 = 2.776445  # Student t, 0.975 quantile, 4 degrees of freedom


def row(**changes) -> dict:
    """A valid per-seed N1 row, in FIELDS order, with `changes` applied."""
    base = {
        "ts": "2026-09-28T10:00:00Z", "number": "N1", "metric": "macro_f1", "value": 0.8,
        "ci_low": None, "ci_high": None, "n": 100, "backbone_id": "dinov2_l14_reg", "res": 224,
        "token_type": "cls", "head": "linear", "seed": 0, "train_manifest": MAKERERE,
        "train_manifest_sha256": A, "train_split": "train", "test_manifest": MAKERERE,
        "test_manifest_sha256": A, "test_split": "test", "split_rule": "blocked:district",
        "coverage": 1.0, "classes": ["healthy", "rust"],
        "run_id": "dinov2_l14_reg-224-linear-cls-s0-0123abcd",
        "model_version": "msv0.1+dinov2_l14_reg@224.linear.man-aaaaaa",
        "cache_key": "2b017210105b0b1f", "quotable": True, "notes": None,
    }  # fmt: skip
    base.update(changes)
    return {k: base[k] for k in n_table.FIELDS}


def five(values=(0.80, 0.82, 0.78, 0.81, 0.79), **changes) -> list[dict]:
    """The per-seed rows of one number, seeds 0-4."""
    return [
        row(seed=s, value=v, run_id=f"run-s{s}", ts=f"2026-09-28T10:00:0{s}Z", **changes)
        for s, v in enumerate(values)
    ]


def reasons(problems: list[str]) -> set[str]:
    return {p.split(":")[0] for p in problems}


# --- the row (US-1, FR-001 to FR-005) --------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"split_rule": "blocked:date+blocked:district+unblocked:random_by_phash_group"},
        {"split_rule": "unblocked:5fold_by_phash_group", "train_split": "cv5", "test_split": "cv5"},
        {"train_manifest": f"{MAKERERE}+{TANZANIA}", "train_manifest_sha256": f"{A}+{B}"},
        {"seed": None, "ci_low": 0.75, "ci_high": 0.85, "run_id": "r0+r1+r2+r3+r4"},
        {"metric": "recall:rust", "n": 40},
        {"metric": "balanced_accuracy:dataset", "classes": ["ibean", "makerere", "tanzania"]},
        {"model_version": None, "notes": "one seed"},
        {"number": "N2", "test_split": "all", "split_rule": "unblocked:random_by_phash_group"},
        {
            "number": "N3",
            "metric": "recall:unknown_als",
            "test_split": "holdout_unknown",
            "split_rule": "holdout_unknown",
            "classes": ["unknown_als", "unknown_wm"],
        },
    ],
)
def test_a_complete_row_passes(changes):
    assert n_table.validate_row(row(**changes)) == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"split_rule": ""}, "bad_split_rule"),
        ({"split_rule": "unblocked"}, "bad_split_rule"),
        ({"split_rule": "blocked:"}, "bad_split_rule"),
        ({"split_rule": "random"}, "bad_split_rule"),
        ({"split_rule": "blocked:district+blocked:date"}, "bad_split_rule"),  # not sorted
        ({"train_manifest_sha256": "abc"}, "bad_hash"),
        ({"test_manifest_sha256": None}, "bad_hash"),
        ({"test_manifest_sha256": f"{A}+{B}"}, "bad_hash"),  # two hashes, one manifest
        ({"number": "N8"}, "bad_number"),
        ({"metric": "top5"}, "bad_metric"),
        ({"metric": "recall"}, "bad_metric"),  # recall names its class
        ({"metric": "recall:anthracnose"}, "bad_metric"),  # not one of the row's classes
        ({"metric": "macro_f1:rust"}, "bad_metric"),
        ({"value": math.nan}, "bad_value"),
        ({"value": "0.8"}, "bad_value"),
        ({"coverage": 0.0}, "bad_value"),
        ({"coverage": 1.5}, "bad_value"),
        ({"ci_low": 0.7}, "bad_value"),  # without ci_high
        ({"ci_low": 0.9, "ci_high": 0.7}, "bad_value"),
        ({"n": 0}, "bad_value"),
        ({"test_split": "val"}, "bad_value"),  # no number is scored on a validation split
        ({"test_split": "train"}, "bad_value"),
        ({"seed": "0"}, "bad_value"),
        ({"classes": []}, "bad_value"),
        ({"quotable": "yes"}, "bad_value"),
        ({"ts": "yesterday"}, "bad_value"),
    ],
)
def test_a_row_that_breaks_the_schema_is_named(changes, reason):
    assert reason in reasons(n_table.validate_row(row(**changes)))


def test_a_missing_or_moved_field_is_named():
    r = row()
    del r["split_rule"]
    assert "missing_field" in reasons(n_table.validate_row(r))
    r = row()
    moved = {k: r[k] for k in r if k != "value"} | {"value": r["value"]}
    assert "bad_field_order" in reasons(n_table.validate_row(moved))


def test_n3_rows_use_only_held_out_classes():
    """US-4.2."""
    n3 = {"number": "N3", "metric": "recall:unknown_als", "split_rule": "holdout_unknown"}
    trained = row(**n3, test_split="holdout_unknown", classes=["unknown_als", "rust"])
    assert "n3_not_held_out" in reasons(n_table.validate_row(trained))
    on_test = row(**n3, test_split="test", classes=["unknown_als"])
    assert "n3_not_held_out" in reasons(n_table.validate_row(on_test))


def test_a_quotable_row_references_frozen_manifests_only():
    """US-4.3 (spec 001 SC-5): checked when the frozen list is given."""
    other = {"test_manifest": TANZANIA, "test_manifest_sha256": B}
    assert n_table.validate_row(row(), frozen={A}) == []
    problems = n_table.validate_row(row(**other), frozen={A})
    assert "manifest_not_frozen" in reasons(problems)
    assert n_table.validate_row(row(**other, quotable=False), frozen={A}) == []
    assert n_table.validate_row(row(**other)) == []  # no frozen list, no check


def test_append_refuses_an_invalid_row_and_appends_nothing(tmp_path):
    """US-1.2."""
    path = tmp_path / "n_table.jsonl"
    assert n_table.append_rows([row()], path) == 1
    before = path.read_bytes()
    with pytest.raises(n_table.NTableError, match="bad_split_rule"):
        n_table.append_rows([row(run_id="valid"), row(run_id="broken", split_rule="")], path)
    assert path.read_bytes() == before  # not even the valid row


def test_the_committed_table_passes():
    """US-1.3 and SC-1, against FROZEN.jsonl when the manifests are on disk."""
    rows = list(n_table.read_rows(n_table.N_TABLE))
    assert rows
    frozen_list = n_table.REPO_ROOT / "data" / "manifests" / "FROZEN.jsonl"
    frozen = None
    if frozen_list.exists():
        lines = frozen_list.read_text(encoding="utf-8").splitlines()
        frozen = {json.loads(line)["sha256"] for line in lines if line.strip()}
    for r in rows:
        assert n_table.validate_row(r, frozen=frozen) == [], r["run_id"]


# --- five seeds (US-2) -----------------------------------------------------------------------


def test_five_seeds_make_the_mean_and_a_t_interval():
    rows = five()
    (agg,) = n_table.aggregate(rows)
    assert list(agg) == list(n_table.FIELDS) and n_table.validate_row(agg) == []
    v = np.array([0.80, 0.82, 0.78, 0.81, 0.79])
    half = T_975_4 * v.std(ddof=1) / math.sqrt(5)
    assert agg["seed"] is None
    assert agg["value"] == pytest.approx(v.mean(), abs=1e-4)
    assert agg["ci_low"] == pytest.approx(v.mean() - half, abs=1e-4)
    assert agg["ci_high"] == pytest.approx(v.mean() + half, abs=1e-4)
    assert agg["run_id"] == "run-s0+run-s1+run-s2+run-s3+run-s4"
    assert agg["ts"] == "2026-09-28T10:00:04Z"
    assert (agg["number"], agg["metric"], agg["n"], agg["quotable"]) == (
        "N1",
        "macro_f1",
        100,
        True,
    )
    assert n_table.aggregate(rows[::-1]) == [agg]  # the order the rows come in does not matter


def test_no_aggregate_without_exactly_seeds_0_to_4():
    rows = five()
    assert n_table.aggregate(rows[:4]) == []
    assert n_table.aggregate([dict(r, seed=r["seed"] + 1) for r in rows]) == []


def test_one_aggregate_per_number_quotable_only_when_every_seed_is():
    rows = five() + five(metric="recall:rust", n=40)
    aggs = n_table.aggregate(rows)
    assert sorted(a["metric"] for a in aggs) == ["macro_f1", "recall:rust"]
    assert len(n_table.aggregate(rows + aggs)) == 2  # aggregates are not aggregated again
    mixed = five()
    mixed[2]["quotable"] = False
    (agg,) = n_table.aggregate(mixed)
    assert agg["quotable"] is False


# --- N1 and N2 pairs (US-4.1) ----------------------------------------------------------------


def test_n1_and_n2_come_in_pairs():
    n1 = n_table.aggregate(five())
    assert n_table.unpaired(n1) == [("dinov2_l14_reg", 224, "cls", "linear", "N2")]
    to_tanzania = {"number": "N2", "test_manifest": TANZANIA, "test_manifest_sha256": B}
    n2 = n_table.aggregate(five(**to_tanzania))
    assert n_table.unpaired(n1 + n2) == []
    proto_n2 = n_table.aggregate(five(**to_tanzania, head="proto"))
    assert n_table.unpaired(n1 + n2 + proto_n2) == [("dinov2_l14_reg", 224, "cls", "proto", "N1")]
    assert n_table.unpaired(five()) == []  # per-seed rows are not reported numbers
    assert n_table.unpaired([dict(r, quotable=False) for r in n1]) == []


# --- N1 from head runs (US-3) ----------------------------------------------------------------


def test_eval_writes_macro_f1_and_every_class_recall_per_seed_and_their_aggregates(
    tmp_path, capsys
):
    manifest = make_manifest(tmp_path)
    heads_root = tmp_path / "heads"
    for seed in range(5):
        code = heads_train.main(
            [
                "--backbone", BACKBONE, "--res", str(RES), "--head", "linear", "--seed", str(seed),
                "--train-manifest", str(manifest), "--cache-root", str(tmp_path / "cache"),
                "--heads-root", str(heads_root), "--compute-log", str(tmp_path / "log.jsonl"),
            ]
        )  # fmt: skip
        assert code == 0, capsys.readouterr().out
    table, md = tmp_path / "n_table.jsonl", tmp_path / "n_table.md"
    argv = [
        "--heads-root", str(heads_root), "--cache-root", str(tmp_path / "cache"),
        "--n-table", str(table), "--md", str(md), "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    assert eval_run.main(argv) == 0, capsys.readouterr().out
    rows = list(n_table.read_rows(table))
    per_seed = [r for r in rows if r["seed"] is not None]
    aggs = [r for r in rows if r["seed"] is None]
    metrics = ["macro_f1", "recall:anthracnose", "recall:healthy", "recall:rust"]
    assert sorted((r["seed"], r["metric"]) for r in per_seed) == [
        (s, m) for s in range(5) for m in metrics
    ]
    assert sorted(a["metric"] for a in aggs) == metrics
    test_rows = [r for r in rows_of(manifest) if r["split"] == "test"]
    for r in rows:
        assert n_table.validate_row(r) == [], r
        assert (r["number"], r["test_split"], r["coverage"]) == ("N1", "test", 1.0)
        assert r["split_rule"] == "blocked:date" and r["quotable"] is True
        assert r["train_manifest_sha256"] == r["test_manifest_sha256"] == sha256(manifest)
        klass = r["metric"].partition(":")[2]
        n = sum(t["class_km2"] == klass for t in test_rows) if klass else len(test_rows)
        assert r["n"] == n, r["metric"]
    assert all(r["ci_low"] is None and r["ci_high"] is None for r in per_seed)
    run_ids = {r["seed"]: r["run_id"] for r in per_seed}
    for a in aggs:
        assert a["ci_low"] <= a["value"] <= a["ci_high"]
        assert a["run_id"] == "+".join(run_ids[s] for s in range(5))
    assert md.read_text(encoding="utf-8").count("| 0–4 |") == 4  # the aggregates only

    before = table.read_bytes(), md.read_bytes()
    assert eval_run.main(argv) == 0
    assert (table.read_bytes(), md.read_bytes()) == before


# --- N2 from head runs (US-7) -----------------------------------------------------------------


def test_the_decision_is_among_the_directions_classes_only():
    """US-7.3: a 3-class head scored on a 2-class target decides between those two."""
    proba = np.array([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3], [0.2, 0.1, 0.7]])
    head = ["healthy", "rust", "anthracnose"]
    assert eval_run.decide(proba, head, ["healthy", "rust"]).tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="anthracnose"):
        eval_run.decide(proba[:, :2], ["healthy", "rust"], ["healthy", "anthracnose"])


def test_eval_scores_every_n2_direction_of_the_config(tmp_path, capsys):
    """US-7: alpha (three classes) -> beta (its test split), -> gamma (rust only: recall), ->
    delta (test-only: every row); beta + delta (two classes) -> alpha, without anthracnose."""
    two = {s: {c: n for c, n in per.items() if c != "anthracnose"} for s, per in COUNTS.items()}
    alpha = make_manifest(tmp_path, "alpha")
    beta = make_manifest(tmp_path, "beta", seed=1, counts=two)
    gamma = make_manifest(tmp_path, "gamma", seed=2, counts={"test": {"rust": 12}})
    delta = make_manifest(tmp_path, "delta", role="test_only", seed=3, counts=two)
    heads_root = tmp_path / "heads"
    for manifests in ((alpha,), (beta, delta)):
        code = heads_train.main(
            [
                "--backbone", BACKBONE, "--res", str(RES), "--head", "linear",
                "--train-manifest", *map(str, manifests), "--cache-root", str(tmp_path / "cache"),
                "--heads-root", str(heads_root), "--compute-log", str(tmp_path / "log.jsonl"),
            ]
        )  # fmt: skip
        assert code == 0, capsys.readouterr().out
    config = tmp_path / "eval.yaml"
    shared = ["healthy", "rust"]
    config.write_text(
        yaml.safe_dump(
            {
                "n2": [
                    {"train": ["alpha_v1"], "test": "beta_v1", "split": "test", "classes": shared},
                    {"train": ["alpha_v1"], "test": "gamma_v1", "split": "test", "classes": shared,
                     "metrics": ["recall:rust"]},
                    {"train": ["alpha_v1"], "test": "delta_v1", "split": "all", "classes": shared},
                    {"train": ["beta_v1", "delta_v1"], "test": "alpha_v1", "split": "test",
                     "classes": shared},
                ]
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    table = tmp_path / "n_table.jsonl"
    argv = [
        "--heads-root", str(heads_root), "--cache-root", str(tmp_path / "cache"),
        "--manifest-root", str(tmp_path / "manifests"), "--config", str(config),
        "--n-table", str(table), "--md", str(tmp_path / "n_table.md"),
        "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    assert eval_run.main(argv) == 0, capsys.readouterr().out
    rows = list(n_table.read_rows(table))
    assert all(n_table.validate_row(r) == [] for r in rows)
    n2 = [r for r in rows if r["number"] == "N2" and r["seed"] is not None]
    by_target = {}
    for r in n2:
        by_target.setdefault(Path(r["test_manifest"]).stem, []).append(r)
    metrics = lambda target: sorted({r["metric"] for r in by_target[target]})  # noqa: E731
    assert metrics("beta_v1") == metrics("delta_v1") == metrics("alpha_v1") == [
        "macro_f1", "recall:healthy", "recall:rust",
    ]  # fmt: skip
    assert metrics("gamma_v1") == ["recall:rust"]
    assert all(r["classes"] == shared and r["quotable"] for r in n2)

    def n_of(manifest, split, klass=None):
        return sum(
            (split == "all" or r["split"] == split)
            and r["class_km2"] in ([klass] if klass else shared)
            for r in rows_of(manifest)
        )

    for r in n2:
        target = {"beta_v1": beta, "gamma_v1": gamma, "delta_v1": delta, "alpha_v1": alpha}
        manifest = target[Path(r["test_manifest"]).stem]
        split = "all" if manifest == delta else "test"
        assert r["test_split"] == split
        assert r["n"] == n_of(manifest, split, r["metric"].partition(":")[2] or None)
    assert {r["train_manifest_sha256"] for r in by_target["alpha_v1"]} == {
        f"{sha256(beta)}+{sha256(delta)}"
    }
    aggs = [r for r in rows if r["number"] == "N2" and r["seed"] is None]
    assert len(aggs) == 3 + 1 + 3 + 3  # beta, gamma, delta, alpha
    assert n_table.unpaired(rows) == []  # alpha's N1 and N2 pair up
    before = table.read_bytes()
    assert eval_run.main(argv) == 0
    assert table.read_bytes() == before


# --- the rendered table (US-5) ---------------------------------------------------------------


def test_the_md_shows_the_aggregates_and_the_rows_no_aggregate_covers():
    seeds = five()
    (agg,) = n_table.aggregate(seeds)
    n4 = row(
        number="N4",
        metric="balanced_accuracy:dataset",
        head="logreg",
        classes=["ibean", "makerere", "tanzania"],
        value=0.98,
        ci_low=0.97,
        ci_high=0.99,
        run_id="n4-dataset-dinov2_l14_reg-224-cls-5c360651",
        model_version=None,
    )
    md = n_table.render([*seeds, agg, n4])
    n1 = md.split("## N1")[1].split("\n## ")[0]
    assert n1.count("| 0–4 |") == 1 and n1.count("| linear |") == 1
    assert "## N4" in md and "balanced_accuracy:dataset" in md
