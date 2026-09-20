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
from ms.eval import supersede, verdict
from ms.heads import HEAD_CONFIGS
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
        "cache_key": "2b017210105b0b1f", "quotable": True, "superseded": None, "notes": None,
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
        {"superseded": "DECISIONS 80: max_epochs 50 -> 300"},
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
        ({"superseded": ""}, "bad_value"),  # a mark names its reason
        ({"superseded": True}, "bad_value"),
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


# --- superseded rows (US-8) ---------------------------------------------------------------------


def test_superseded_rows_leave_the_pairs_and_keep_their_own_aggregates():
    """US-8.3: a mark is part of what makes seeds one number, and pairs read current rows."""
    old = [dict(r, superseded="old recipe") for r in five(values=(0.5, 0.5, 0.5, 0.5, 0.5))]
    new = [dict(r, run_id=f"new-s{r['seed']}") for r in five()]
    aggs = n_table.aggregate(old + new)
    assert sorted((a["superseded"] or "", a["value"]) for a in aggs) == [
        ("", pytest.approx(0.8)),
        ("old recipe", pytest.approx(0.5)),
    ]
    to_tanzania = {"number": "N2", "test_manifest": TANZANIA, "test_manifest_sha256": B}
    n2_old = [dict(a, superseded="old recipe") for a in n_table.aggregate(five(**to_tanzania))]
    current_n1 = [a for a in aggs if not a["superseded"]]
    assert n_table.unpaired(current_n1 + n2_old) == [("dinov2_l14_reg", 224, "cls", "linear", "N2")]


def test_supersede_marks_the_rows_of_a_replaced_recipe_and_keeps_them(tmp_path, capsys):
    """US-8: after a recipe change, the old runs' rows stay, marked; the new runs make their
    own rows and aggregates; the md and the pairs read the new ones."""
    manifest = make_manifest(tmp_path)
    configs, heads_root = tmp_path / "configs", tmp_path / "heads"
    configs.mkdir()
    table, md = tmp_path / "n_table.jsonl", tmp_path / "n_table.md"
    recipe = yaml.safe_load((HEAD_CONFIGS / "linear.yaml").read_text(encoding="utf-8"))

    def train(max_epochs: int) -> str:
        path = configs / "linear.yaml"
        path.write_text(yaml.safe_dump({**recipe, "max_epochs": max_epochs}), encoding="utf-8")
        code = heads_train.main(
            [
                "--backbone", BACKBONE, "--res", str(RES), "--head", "linear",
                "--train-manifest", str(manifest), "--config-dir", str(configs),
                "--cache-root", str(tmp_path / "cache"), "--heads-root", str(heads_root),
                "--compute-log", str(tmp_path / "log.jsonl"),
            ]
        )  # fmt: skip
        assert code == 0, capsys.readouterr().out
        return sha256(path)

    evaluate = [
        "--heads-root", str(heads_root), "--cache-root", str(tmp_path / "cache"),
        "--manifest-root", str(tmp_path / "manifests"), "--config", str(tmp_path / "none.yaml"),
        "--n-table", str(table), "--md", str(md), "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    old_sha = train(5)
    assert eval_run.main(evaluate) == 0
    first = list(n_table.read_rows(table))
    train(6)
    reason = "test: max_epochs 5 -> 6"
    supersede_argv = [
        "--config-sha256", old_sha, "--reason", reason, "--heads-root", str(heads_root),
        "--n-table", str(table), "--md", str(md),
    ]  # fmt: skip
    assert supersede.main(supersede_argv) == 0
    marked = list(n_table.read_rows(table))
    assert [dict(r, superseded=None) for r in marked] == first  # nothing else changed
    assert {r["superseded"] for r in marked} == {reason}
    assert eval_run.main(evaluate) == 0

    rows = list(n_table.read_rows(table))
    assert all(n_table.validate_row(r) == [] for r in rows)
    old = [r for r in rows if r["superseded"]]
    new = [r for r in rows if not r["superseded"]]
    assert old == marked and len(new) == len(first)
    old_runs = {i for r in old for i in r["run_id"].split("+")}
    assert not old_runs & {i for r in new for i in r["run_id"].split("+")}
    assert sum(r["seed"] is None for r in new) == 4  # macro-F1 and three recalls
    text = md.read_text(encoding="utf-8")
    assert text.split("## N1")[1].split("\n## ")[0].count("| 0–4 |") == 4
    assert f"- {reason}: N1, 4 aggregate row(s) and 20 per-seed row(s)" in text

    capsys.readouterr()
    assert supersede.main(supersede_argv) == 0 and "0 row(s) marked" in capsys.readouterr().out
    assert list(n_table.read_rows(table)) == rows
    assert supersede.main([*supersede_argv[:1], "0" * 64, *supersede_argv[2:]]) == 2


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


# --- the verdict (US-6) ------------------------------------------------------------------------

CHAMP, CHALL = verdict.CHAMPION, verdict.CHALLENGER
#: the three frame directions of US-6.3, and the crop-level one that is reported, not counted
DIRECTIONS = [
    {"train": ["tanzania_v1"], "test": "makerere_v1", "split": "test",
     "classes": ["healthy", "rust"]},
    {"train": ["tanzania_v1"], "test": "makerere_crops_v1", "split": "test",
     "classes": ["healthy", "rust"], "metrics": ["recall:rust"]},
    {"train": ["tanzania_v1"], "test": "ibean_v1", "split": "all", "classes": ["healthy", "rust"]},
    {"train": ["makerere_v1", "ibean_v1"], "test": "tanzania_v1", "split": "test",
     "classes": ["healthy", "rust"]},
]  # fmt: skip
N2_PAIRS = [("tanzania_v1", "makerere_v1"), ("tanzania_v1", "ibean_v1"),
            ("makerere_v1+ibean_v1", "tanzania_v1")]  # fmt: skip
TRAIN_SETS = ("makerere_v1", "tanzania_v1", "makerere_v1+ibean_v1")


def man(stems: str) -> tuple[str, str]:
    """('data/manifests/a_v1.jsonl+...', 'aaa...+...'): a row's manifest and hash fields."""
    parts = stems.split("+")
    return (
        "+".join(f"data/manifests/{s}.jsonl" for s in parts),
        "+".join(chr(ord("a") + i) * 64 for i, _ in enumerate(parts)),
    )


def n2_row(backbone, head, train, test, value, half=0.005, **changes):
    train_m, train_h = man(train)
    test_m, test_h = man(test)
    changes.setdefault("seed", None)
    changes.setdefault("ci_low", round(value - half, 4))
    changes.setdefault("ci_high", round(value + half, 4))
    return row(
        number="N2", metric="macro_f1", backbone_id=backbone, head=head, value=value,
        train_manifest=train_m, train_manifest_sha256=train_h,
        test_manifest=test_m, test_manifest_sha256=test_h,
        test_split="test", classes=["healthy", "rust"],
        run_id=f"{backbone}-224-{head}-cls-s0-1+{backbone}-224-{head}-cls-s4-2", **changes,
    )  # fmt: skip


def n3_row(backbone, head, held, score, train, value, **changes):
    train_m, train_h = man(train)
    changes.setdefault("seed", None)
    changes.setdefault("ci_low", round(value - 0.001, 4))
    changes.setdefault("ci_high", round(value + 0.001, 4))
    return row(
        number="N3", metric=f"auroc:{score}", backbone_id=backbone, head=head, value=value,
        train_manifest=train_m, train_manifest_sha256=train_h,
        test_manifest="data/manifests/swm_v1.jsonl", test_manifest_sha256="f" * 64,
        test_split="holdout_unknown", split_rule="holdout_unknown", classes=[held],
        run_id=f"{backbone}-224-{head}-cls-s0-1+{backbone}-224-{head}-cls-s4-2", **changes,
    )  # fmt: skip


def table_for(n2_value, n3_value):
    """A whole table: `n2_value(backbone, head, direction)` gives (value, half-interval) and
    `n3_value(backbone, head, held-out, score, training set)` gives an AUROC."""
    rows = []
    for backbone in (CHAMP, CHALL):
        for head in verdict.HEADS:
            for where in N2_PAIRS:
                value, half = n2_value(backbone, head, where)
                rows.append(n2_row(backbone, head, where[0], where[1], value, half))
            for held in verdict.HELD_OUT:
                for score in ("conf", "knn"):
                    for train in TRAIN_SETS:
                        rows.append(
                            n3_row(
                                backbone, head, held, score, train,
                                n3_value(backbone, head, held, score, train),
                            )
                        )  # fmt: skip
    return rows


def flat(champion, challenger, half=0.005):
    """Every N2 number at one value per backbone."""
    return lambda b, h, d: (champion if b == CHAMP else challenger, half)


def verdict_of(rows):
    return verdict.decide(rows, DIRECTIONS)


def test_neither_wins_a_criterion_so_the_champion_keeps_it():
    """US-6.6: anything else, no wins at all included, goes to the champion."""
    v = verdict_of(table_for(flat(0.50, 0.50), lambda *a: 0.80))
    assert v["c1"][CHALL]["won"] is False and v["c1"][CHAMP]["won"] is False
    assert v["c2"][CHALL]["won"] is False and v["c2"][CHAMP]["won"] is False
    assert v["winner"] == CHAMP
    assert "neither" in v["why"]


def test_the_challenger_wins_criterion_1_and_takes_the_verdict():
    """US-6.3: 2 pp on the mean, and two directions ahead by 2 pp with separated intervals."""
    v = verdict_of(table_for(flat(0.50, 0.55), lambda *a: 0.80))
    assert v["c1"][CHALL]["won"] is True and v["c1"][CHAMP]["won"] is False
    assert v["winner"] == CHALL and "challenger" in v["why"]


def test_a_lead_with_overlapping_intervals_does_not_win_criterion_1():
    """US-6.3: the five-seed intervals have to separate in two of the three directions."""
    v = verdict_of(table_for(flat(0.50, 0.55, half=0.05), lambda *a: 0.80))
    assert v["c1"][CHALL]["won"] is False
    assert all(h["clear"] == 0 for h in v["c1"][CHALL]["heads"].values())
    assert v["winner"] == CHAMP


def test_a_lead_on_the_mean_alone_does_not_win_criterion_1():
    """US-6.3: ahead on the mean because of one direction, and level in the other two."""

    def n2(backbone, head, direction):
        if backbone == CHAMP:
            return 0.50, 0.005
        return (0.62, 0.005) if direction == N2_PAIRS[0] else (0.50, 0.005)

    v = verdict_of(table_for(n2, lambda *a: 0.80))
    assert v["c1"][CHALL]["heads"]["linear"]["ahead"] >= 0.02
    assert v["c1"][CHALL]["heads"]["linear"]["clear"] == 1
    assert v["c1"][CHALL]["won"] is False and v["winner"] == CHAMP


def test_the_challenger_wins_criterion_2_only_with_every_training_set():
    """US-6.4 and the owner's reading of 2026-09-20: each head, each held-out set, both
    decision scores and each of the three training sets."""

    def ahead(backbone, head, held, score, train):
        return 0.85 if backbone == CHALL else 0.80

    v = verdict_of(table_for(flat(0.50, 0.50), ahead))
    assert v["c2"][CHALL]["won"] is True and v["winner"] == CHALL

    def all_but_one(backbone, head, held, score, train):
        if backbone == CHAMP:
            return 0.80
        one = (head, held, score, train) == ("mix", "unknown_wm", "knn", "tanzania_v1")
        return 0.805 if one else 0.85

    v = verdict_of(table_for(flat(0.50, 0.50), all_but_one))
    assert v["c2"][CHALL]["won"] is False
    assert v["c2"][CHALL]["cleared"] == v["c2"][CHALL]["of"] - 1
    assert v["winner"] == CHAMP


def test_criterion_2_needs_both_decision_scores_and_ignores_the_others():
    """US-6.4: confidence and kNN distance. maha and energy are reported, not counted."""

    def one_score(backbone, head, held, score, train):
        if backbone == CHAMP:
            return 0.80
        return 0.85 if score == "knn" else 0.80

    rows = table_for(flat(0.50, 0.50), one_score)
    rows += [
        n3_row(CHALL, "linear", "unknown_wm", "maha", "tanzania_v1", 0.99),
        n3_row(CHAMP, "linear", "unknown_wm", "maha", "tanzania_v1", 0.10),
    ]
    v = verdict_of(rows)
    assert v["c2"][CHALL]["won"] is False
    assert all(c["key"][2] in verdict.DECISION_SCORES for c in v["c2"][CHALL]["comparisons"])


def probe_rows(backbone, dataset, district):
    return [
        row(number="N4", metric=f"balanced_accuracy:{t}", backbone_id=backbone, head="logreg",
            seed=None, value=v, ci_low=round(v - 0.01, 4), ci_high=round(v + 0.01, 4),
            test_split="cv5", split_rule="unblocked:5fold_by_phash_group",
            classes=["a", "b", "c"], run_id=f"n4-{t}-{backbone}", model_version=None)
        for t, v in (("dataset", dataset), ("district", district))
    ]  # fmt: skip


def test_a_split_goes_to_n4_and_a_tie_to_the_champion():
    """US-6.6: each backbone wins a criterion, so the probe decides — and only when one is
    2 pp lower on both targets."""
    rows = table_for(flat(0.50, 0.55), lambda b, *a: 0.85 if b == CHAMP else 0.80)
    assert verdict_of(rows)["c1"][CHALL]["won"] and verdict_of(rows)["c2"][CHAMP]["won"]
    # the challenger gives the site away less on both targets: it takes the split
    v = verdict_of(rows + probe_rows(CHALL, 0.70, 0.60) + probe_rows(CHAMP, 0.80, 0.70))
    assert v["winner"] == CHALL and "N4 broke the split" in v["why"]
    # lower on one target only: the split stands, and a tie goes to the champion
    v = verdict_of(rows + probe_rows(CHALL, 0.70, 0.70) + probe_rows(CHAMP, 0.80, 0.70))
    assert v["winner"] == CHAMP and "did not break the split" in v["why"]
    # no probe rows at all: the same
    assert verdict_of(rows)["winner"] == CHAMP


def test_the_verdict_reads_only_224_cls_coverage_one_quotable_current_rows():
    """US-6.1. Every other row is invisible to it, whatever it says."""
    rows = table_for(flat(0.50, 0.50), lambda *a: 0.80)
    loud = [
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, res=518),
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, token_type="cls+meanpatch"),
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, coverage=0.9),
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, quotable=False),
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, superseded="a new recipe"),
        n2_row(CHALL, "linear", "tanzania_v1", "makerere_v1", 0.99, seed=0, ci_low=None,
               ci_high=None),
    ]  # fmt: skip
    assert verdict_of(rows + loud)["winner"] == CHAMP
    assert verdict_of(rows + loud)["rows_read"] == verdict_of(rows)["rows_read"]


def test_the_crop_direction_is_reported_and_not_counted():
    """US-6.3: it measures one class's recall, not the macro-F1 H9 names."""
    assert verdict.counted(DIRECTIONS) == N2_PAIRS
    rows = table_for(flat(0.50, 0.50), lambda *a: 0.80)
    rows += [n2_row(CHALL, h, "tanzania_v1", "makerere_crops_v1", 0.99) for h in verdict.HEADS]
    v = verdict_of(rows)
    assert v["c1"][CHALL]["won"] is False
    assert all(len(h["directions"]) == 3 for h in v["c1"][CHALL]["heads"].values())


def test_a_missing_number_never_wins_a_criterion():
    """US-6.3 and US-6.4: a win needs every head, and every comparison it names."""

    def n3(backbone, *rest):
        return 0.85 if backbone == CHALL else 0.80

    rows = [
        r
        for r in table_for(flat(0.50, 0.55), n3)
        if not (r["head"] == "proto" and r["number"] == "N2")
    ]
    v = verdict_of(rows)
    assert v["c1"][CHALL]["heads"]["proto"]["complete"] is False
    assert v["c1"][CHALL]["won"] is False and v["c2"][CHALL]["won"] is True
    assert v["winner"] == CHALL  # criterion 2 alone still wins it

    rows = [
        r
        for r in table_for(flat(0.50, 0.50), n3)
        if not (r["head"] == "proto" and r["number"] == "N3")
    ]
    v = verdict_of(rows)
    assert v["c2"][CHALL]["complete"] is False and v["c2"][CHALL]["won"] is False
    assert v["winner"] == CHAMP


def test_verdict_md_is_written_by_code_and_only_when_it_changes(tmp_path):
    """US-6.1: `make eval` writes it, nobody edits it, and it is rewritten only when the
    numbers move."""
    path = tmp_path / "verdict.md"
    rows = table_for(flat(0.50, 0.50), lambda *a: 0.80)
    written, v = verdict.write_verdict(rows, DIRECTIONS, path)
    assert written is True and v["winner"] == CHAMP
    text = path.read_text(encoding="utf-8")
    assert "dinov2_l14_reg wins" in text.split("\n")[2]
    assert "do not edit" in text and "H9 §7" in text
    assert "## Criterion 1" in text and "## Criterion 2" in text and "## Outcome" in text
    assert "research-only until H6 C5" in text
    assert verdict.write_verdict(rows, DIRECTIONS, path) == (False, v)
    assert path.read_text(encoding="utf-8") == text

    moved = table_for(flat(0.50, 0.55), lambda *a: 0.80)
    written, v = verdict.write_verdict(moved, DIRECTIONS, path)
    assert written is True and v["winner"] == CHALL
    assert "dinov3_l16 wins" in path.read_text(encoding="utf-8").split("\n")[2]


def test_make_eval_refuses_the_verdict_while_a_number_is_unpaired(tmp_path, capsys):
    """US-6.1 and US-4.1: an unpaired number means the table is half-written."""
    argv = [
        "--heads-root", str(tmp_path / "heads"), "--cache-root", str(tmp_path / "cache"),
        "--manifest-root", str(tmp_path / "manifests"), "--config", str(tmp_path / "eval.yaml"),
        "--n-table", str(tmp_path / "n_table.jsonl"), "--md", str(tmp_path / "n_table.md"),
        "--verdict", str(tmp_path / "verdict.md"), "--compute-log", str(tmp_path / "log.jsonl"),
    ]  # fmt: skip
    (tmp_path / "eval.yaml").write_text("n2: []\n", encoding="utf-8")
    # an N1 aggregate with no N2 beside it: unpaired, so no verdict is written
    n_table.append_rows(n_table.aggregate(five()), tmp_path / "n_table.jsonl")
    assert eval_run.main(argv) == 0
    text = capsys.readouterr().out
    assert "unpaired:" in text and "verdict: refused" in text
    assert not (tmp_path / "verdict.md").exists()
