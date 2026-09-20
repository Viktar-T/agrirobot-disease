"""The registry and the model card (ms.registry; work plan S4.6, the W5 "Registry + card").

H8 §6.8: one MLflow experiment per backbone, one run per head run, and a model **registered
only when it has a card**. The card is rendered from the artefacts — the run, its fit, the
frozen manifests and the N-table — so these tests check that it quotes them and nothing else,
and that a card which no longer matches them blocks the registration.

CPU only, on the synthetic manifests and caches of synthetic.py, with a temporary MLflow
SQLite store for the two tests that need one (as test_probe.py does). No network and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from synthetic import BACKBONE, RES, make_manifest

from ms import registry
from ms.abstain import COVERAGES
from ms.abstain import fit as abstain_fit
from ms.eval import read_rows
from ms.eval import run as eval_run
from ms.heads import train as heads_train

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "model_service" / "cards" / "TEMPLATE.md"
DEFAULT_COVERAGE = 0.90


# --- the synthetic world ---------------------------------------------------------------------


def abstain_config(root: Path) -> Path:
    path = root / "abstain.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "k": 5,
                "coverages": list(COVERAGES),
                "default_coverage": DEFAULT_COVERAGE,
                "distance_tpr": 0.95,
                "ece_bins": 15,
                "epsilon": 1e-6,
                "n3": [{"manifest": "alpha_v1", "classes": ["unknown_als"]}],
            }
        ),
        encoding="utf-8",
    )
    return path


def train(root: Path, manifest: Path, *, seed: int = 0, allow_test_only: bool = False) -> dict:
    argv = [
        "--backbone", BACKBONE, "--res", str(RES), "--head", "linear", "--seed", str(seed),
        "--train-manifest", str(manifest), "--split", "train", "--val-split", "val",
        "--cache-root", str(root / "cache"), "--heads-root", str(root / "heads"),
        "--compute-log", str(root / "log.jsonl"),
        *(["--allow-test-only"] if allow_test_only else []),
    ]  # fmt: skip
    assert heads_train.main(argv) == 0
    runs = [
        json.loads((p / "run.json").read_text(encoding="utf-8"))
        for p in (root / "heads").iterdir()
        if (p / "run.json").is_file()
    ]
    return next(r for r in runs if r["seed"] == seed and r["head"] == "linear")


def fit(root: Path, run_id: str) -> dict:
    assert (
        abstain_fit.main(
            [
                "--run",
                run_id,
                "--heads-root",
                str(root / "heads"),
                "--cache-root",
                str(root / "cache"),
                "--config",
                str(abstain_config(root)),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )
    return json.loads((root / "heads" / run_id / "abstain.json").read_text(encoding="utf-8"))


def evaluate(root: Path) -> None:
    (root / "eval.yaml").write_text(yaml.safe_dump({"n2": []}), encoding="utf-8")
    assert (
        eval_run.main(
            [
                "--heads-root",
                str(root / "heads"),
                "--cache-root",
                str(root / "cache"),
                "--manifest-root",
                str(root / "manifests"),
                "--config",
                str(root / "eval.yaml"),
                "--abstain-config",
                str(abstain_config(root)),
                "--n-table",
                str(root / "n_table.jsonl"),
                "--md",
                str(root / "n_table.md"),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )


def command(root: Path, action: str, *extra) -> list[str]:
    return [
        action, "--heads-root", str(root / "heads"), "--cache-root", str(root / "cache"),
        "--n-table", str(root / "n_table.jsonl"), "--cards", str(root / "cards"),
        "--template", str(TEMPLATE), "--backbone-configs", str(root / "backbones"),
        *(str(a) for a in extra),
    ]  # fmt: skip


def backbone_config(root: Path, **changes) -> Path:
    """A config for the synthetic backbone, so the card can quote a licence (H8 §6.8)."""
    folder = root / "backbones"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = {
        "backbone_id": BACKBONE,
        "hf_id": f"test/{BACKBONE}",
        "revision": None,
        "licence": "Apache-2.0",
        "gated": False,
        "research_only_until_c5": False,
        "patch_size": 14,
        "embed_dim": 16,
        "resolutions": [RES],
        "preprocess_string": {RES: f"resize_short={RES};center_crop={RES};norm=imagenet"},
        "dtype": "float32",
        "attn_implementation": "sdpa",
        "batch_size": 8,
        "token_types": ["cls", "meanpatch"],
    }
    cfg.update(changes)
    path = folder / f"{BACKBONE}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """alpha + its cache + a linear head + its abstain.json + the N-table rows it earned."""
    root = tmp_path_factory.mktemp("registry")
    manifest = make_manifest(root, "alpha")
    # five seeds, because a reported number is their mean with an interval (spec 005 US-2)
    runs = [train(root, manifest, seed=seed) for seed in range(5)]
    for r in runs:
        fit(root, r["run_id"])
    evaluate(root)
    backbone_config(root)
    run = runs[0]
    fitted = json.loads(
        (root / "heads" / run["run_id"] / "abstain.json").read_text(encoding="utf-8")
    )
    return {"root": root, "manifest": manifest, "run": run, "runs": runs, "fit": fitted}


@pytest.fixture(autouse=True)
def _no_cached_reads():
    """The module's caches are keyed by path, and every test has its own tmp table."""
    registry.manifest_facts.cache_clear()
    registry._by_run.cache_clear()
    yield


def move_a_number(table: Path) -> None:
    """Halve one reported macro-F1 in place, so the card that quotes it has to change. Rows
    are append-only in the real table (spec 005); a test's copy is its own."""
    rows = list(read_rows(table))
    moved = False
    for row in rows:
        reported = row["seed"] is None and row["coverage"] == 1.0
        if not moved and reported and row["metric"] == "macro_f1":
            row["value"] = round(row["value"] / 2, 4)
            moved = True
    assert moved, "no reported macro-F1 to move"
    lines = "".join(json.dumps(r) + "\n" for r in rows)
    table.write_text(lines, encoding="utf-8", newline="\n")
    registry._by_run.cache_clear()


def card_of(world: dict) -> Path:
    return registry.card_path(world["run"]["model_version"], world["root"] / "cards")


def write_card(world: dict, *extra) -> tuple[int, Path]:
    code = registry.main(command(world["root"], "card", "--run", world["run"]["run_id"], *extra))
    return code, card_of(world)


# --- the card ----------------------------------------------------------------------------------


def test_a_card_is_rendered_from_the_artefacts(world, capsys):
    code, path = write_card(world, "--no-mlflow")
    assert code == 0, capsys.readouterr().out
    text = path.read_text(encoding="utf-8")
    assert "{{" not in text, "a placeholder the renderer left behind"
    run, fitted = world["run"], world["fit"]
    # what H8 §6.8 asks a card to carry, each read from the artefact that holds it
    assert run["model_version"] in text and run["run_id"] in text
    assert run["backbone"]["weights_sha256"] in text  # the model
    assert run["train"]["manifest_sha256"] in text  # the data, by the hash it was frozen at
    assert run["head_config"]["sha256"] in text
    assert "Apache-2.0" in text  # the backbone's licence, from its config
    assert f"{fitted['temperature']['T']:.6f}" in text  # the operating point
    for coverage in fitted["thresholds"]:
        assert f"{fitted['thresholds'][coverage]['tau_conf']:.6f}" in text
    assert "Intended use" in text and "Not validated for" in text  # the template's prose


def test_a_card_quotes_the_tables_numbers_and_nothing_else(world):
    write_card(world, "--no-mlflow")
    text = card_of(world).read_text(encoding="utf-8")
    rows = [
        r
        for r in read_rows(world["root"] / "n_table.jsonl")
        if r["seed"] is None and world["run"]["run_id"] in str(r["run_id"]).split("+")
    ]
    assert rows, "the synthetic world produced no aggregate to quote"
    quoted, skipped = 0, []
    for row in rows:
        # the card quotes a number at coverage 1.0 in its own table, and one at an operating
        # point in the selective-risk table; the rest are abstention rates it does not print
        printed = row["coverage"] == 1.0 or row["metric"] == "selective_risk"
        if printed and f"{row['value']:.4f}" in text:
            quoted += 1
        elif printed:
            skipped.append((row["number"], row["metric"], row["coverage"], row["value"]))
    assert not skipped, skipped
    assert quoted >= 5, f"the card quoted {quoted} of {len(rows)} reported rows"
    assert str(world["root"]) not in text  # no absolute paths from this machine


def test_a_second_card_writes_nothing_and_a_moved_number_makes_it_stale(world, capsys):
    write_card(world, "--no-mlflow")
    before = card_of(world).read_bytes()
    code, _ = write_card(world, "--no-mlflow")
    assert code == 0 and "already what the artefacts render" in capsys.readouterr().out
    assert card_of(world).read_bytes() == before

    model = registry.load_model(
        world["root"] / "heads" / world["run"]["run_id"],
        world["root"] / "cache",
        world["root"] / "n_table.jsonl",
        world["root"] / "backbones",
    )
    current, why = registry.card_is_current(
        model, world["root"] / "cards", TEMPLATE, "not registered"
    )
    assert current and why == ""

    table = world["root"] / "n_table.jsonl"
    before = table.read_bytes()
    try:
        move_a_number(table)
        model = registry.load_model(
            world["root"] / "heads" / world["run"]["run_id"],
            world["root"] / "cache",
            table,
            world["root"] / "backbones",
        )
        current, why = registry.card_is_current(
            model, world["root"] / "cards", TEMPLATE, "not registered"
        )
        assert not current and "is not what the artefacts render now" in why
    finally:
        table.write_bytes(before)
        registry._by_run.cache_clear()


def test_the_card_states_a_gated_backbones_terms(world, tmp_path):
    """H8 §6.8: for the challenger, the date the terms were accepted and research-only."""
    backbone_config(
        world["root"],
        licence="DINOv3 License (Meta, 2025-08-14), gated",
        gated=True,
        research_only_until_c5=True,
        terms_accepted={"date": "2026-09-18", "account": "an-account", "scope": "research only"},
    )
    run = dict(world["run"])
    run["backbone"] = {**run["backbone"], "research_only_until_c5": True}
    folder = world["root"] / "heads" / run["run_id"]
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    try:
        registry.manifest_facts.cache_clear()
        model = registry.load_model(
            folder, world["root"] / "cache", world["root"] / "n_table.jsonl",
            world["root"] / "backbones",
        )  # fmt: skip
        text = registry.render_card(model, status="not registered", template=TEMPLATE)
        assert "Research only until H6 C5" in text
        assert "2026-09-18" in text and "an-account" in text
    finally:
        (folder / "run.json").write_text(json.dumps(world["run"]), encoding="utf-8")
        backbone_config(world["root"])


# --- the gate ----------------------------------------------------------------------------------


def test_register_refuses_without_a_card(world, tmp_path, capsys):
    code = registry.main(
        command(world["root"], "register", "--run", world["run"]["run_id"], "--cards", tmp_path)
    )
    assert code == 2
    text = capsys.readouterr().err
    assert text.startswith("missing_card: ") and "only with a card" in text


def test_register_refuses_a_card_that_is_no_longer_what_the_artefacts_render(world, capsys):
    write_card(world, "--no-mlflow")
    path = card_of(world)
    path.write_text(
        path.read_text(encoding="utf-8").replace("Intended use", "Intended use (edited by hand)"),
        encoding="utf-8",
    )
    code = registry.main(command(world["root"], "register", "--run", world["run"]["run_id"]))
    assert code == 2 and "missing_card" in capsys.readouterr().err
    write_card(world, "--no-mlflow", "--force")  # put it back for the tests that follow


def test_register_refuses_a_run_whose_numbers_are_never_reported(world, tmp_path, capsys):
    """A run trained on a test-only manifest is not a model to register (spec 003 US-5.1)."""
    root = tmp_path / "test_only"
    manifest = make_manifest(root, "solo", role="test_only")
    run = train(root, manifest, allow_test_only=True)
    fit(root, run["run_id"])
    backbone_config(root)
    code = registry.main(command(root, "register", "--run", run["run_id"]))
    assert code == 2
    assert capsys.readouterr().err.startswith("not_quotable: ")


def test_a_run_without_an_abstain_json_has_no_card(world, tmp_path, capsys):
    root = tmp_path / "unfitted"
    manifest = make_manifest(root, "beta")
    run = train(root, manifest)
    backbone_config(root)
    code = registry.main(command(root, "card", "--run", run["run_id"], "--no-mlflow"))
    assert code == 2
    assert capsys.readouterr().err.startswith("missing_abstain: ")


# --- MLflow ------------------------------------------------------------------------------------


def test_every_head_run_is_one_mlflow_run_and_a_second_log_adds_none(world, capsys):
    from mlflow.tracking import MlflowClient

    root = world["root"]
    uri = "sqlite:///" + (root / "mlruns" / "mlflow.db").as_posix()
    (root / "mlruns").mkdir(exist_ok=True)
    assert registry.main(command(root, "log", "--mlflow-uri", uri)) == 0
    assert "5 run(s) logged" in capsys.readouterr().out

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(f"piece4-heads-{BACKBONE}")
    assert experiment is not None  # H8 §6.8: one experiment per backbone
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 5  # one per head run
    logged = next(r for r in runs if r.data.tags["head_run_id"] == world["run"]["run_id"])
    assert logged.data.tags["head_run_id"] == world["run"]["run_id"]
    assert logged.data.tags["model_version"] == world["run"]["model_version"]
    # the things H8 §6.8 asks every run to log
    assert logged.data.params["train_manifest_sha256"] == world["run"]["train"]["manifest_sha256"]
    assert logged.data.params["preprocess_string"] == world["run"]["backbone"]["preprocess_string"]
    assert logged.data.params["weights_sha256"] == world["run"]["backbone"]["weights_sha256"]
    assert int(logged.data.params["k"]) == world["fit"]["scores"]["knn"]["k"]
    assert logged.data.metrics["temperature"] == pytest.approx(
        world["fit"]["temperature"]["T"], abs=1e-6
    )
    assert logged.data.tags["seed"] == str(world["run"]["seed"])
    numbers = {k.split(".", 1)[0] for k in logged.data.metrics if k[0] == "N"}
    # H8 §6.8 asks each run to log N1-N3; the synthetic world earns N1 and N3
    assert {"N1", "N3"} <= numbers, sorted(logged.data.metrics)
    assert any(k.startswith("tau_conf.cov") for k in logged.data.metrics), "the thresholds"

    assert registry.main(command(root, "log", "--mlflow-uri", uri)) == 0
    assert "0 run(s) logged" in capsys.readouterr().out
    assert len(client.search_runs([experiment.experiment_id])) == 5


def test_registering_lands_one_version_that_names_its_card(world, capsys):
    from mlflow.tracking import MlflowClient

    root = world["root"]
    uri = "sqlite:///" + (root / "mlruns2" / "mlflow.db").as_posix()
    (root / "mlruns2").mkdir(exist_ok=True)
    write_card(world, "--no-mlflow", "--force")
    argv = command(root, "register", "--run", world["run"]["run_id"], "--mlflow-uri", uri)
    assert registry.main(argv) == 0
    assert "registered as" in capsys.readouterr().out

    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions(f"name = '{registry.REGISTERED_MODEL}'")
    assert len(versions) == 1
    tags = versions[0].tags
    assert tags["model_version"] == world["run"]["model_version"]
    assert tags["head_run_id"] == world["run"]["run_id"]
    assert tags["card"].endswith(".md")
    import hashlib

    assert tags["card_sha256"] == hashlib.sha256(card_of(world).read_bytes()).hexdigest()

    assert tags["card_content_sha256"] == registry.content_sha256(card_of(world))

    assert registry.main(argv) == 0  # again: the same version, nothing new
    out = capsys.readouterr().out
    assert "already version 1" in out and "has changed since" not in out
    assert len(client.search_model_versions(f"name = '{registry.REGISTERED_MODEL}'")) == 1


def test_a_card_that_moved_on_says_so_instead_of_registering_quietly(world, capsys):
    """A version is tagged with the card's substance. When a number lands and the card is
    written again, the registry would otherwise go on pointing at it in silence.

    A card edited by hand never gets this far: it is not what the artefacts render, and
    `register` refuses it outright (the test above).
    """
    root = world["root"]
    uri = "sqlite:///" + (root / "mlruns3" / "mlflow.db").as_posix()
    (root / "mlruns3").mkdir(exist_ok=True)
    write_card(world, "--no-mlflow", "--force")
    argv = command(root, "register", "--run", world["run"]["run_id"], "--mlflow-uri", uri)
    assert registry.main(argv) == 0
    capsys.readouterr()

    table = root / "n_table.jsonl"
    before = table.read_bytes()
    try:
        move_a_number(table)
        assert write_card(world, "--no-mlflow")[0] == 0  # the card takes the new number
        assert registry.main(argv) == 0
        out = capsys.readouterr().out
        assert "already version 1" in out and "has changed since" in out and "--force" in out
    finally:
        table.write_bytes(before)
        registry._by_run.cache_clear()
        write_card(world, "--no-mlflow", "--force")
