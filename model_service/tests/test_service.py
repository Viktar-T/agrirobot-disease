"""Spec 006, the model service (ms.service; work plan S4.6 and the W5 tasks).

Written before the implementation (spec-driven). The W2 slice already answers
`POST /v1/predict` from the cache (DECISIONS 33, 37), so most tests here are red on
*behaviour*, not on an import: the batch endpoint, the abstention decision, the calibrated
scores, the request log, features computed on request, the card summary, the start-up checks
and N6 are all W5's. `ms.service.bench` does not exist yet and fails with one message.

`test_slice.py` keeps the W2 slice's own service tests; this module is spec 006's.

CPU only. Two worlds, because the service needs both features and bytes:

- the **synthetic** world of synthetic.py (a manifest, its cache, a linear head, its
  `abstain.json`) plus a data root of stand-in files whose bytes are their `image_id`, so
  that `sha256(bytes) == row["sha256"]` and the cached-hash path is exercised without an
  image, and an `odd` cache of hand-built features that force each abstention reason;
- the **fixture** world of `tests/fixtures/ibean_30` with the tiny random-init backbone of
  test_slice.py, the only one with real pixels, for features computed on request (US-4.2).

No network and no GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from synthetic import BACKBONE, COUNTS, RES, direction, make_manifest, write_cache

import ms.eval as n_table
from ms import compute_log
from ms.abstain import COVERAGES, coverage_key, decide, load_fit, probabilities, scores
from ms.cache import extract as cache_extract
from ms.cache import find_cache, load_cache
from ms.data.manifests import load_manifest
from ms.heads import features, load_run
from ms.heads import train as heads_train
from ms.service import log as service_log
from ms.service.app import Settings, create_app
from ms.service.example import request_for

TESTS = Path(__file__).resolve().parent
FIXTURE = TESTS / "fixtures" / "ibean_30"
FIXTURE_MANIFEST = FIXTURE / "ibean_v1.jsonl"
#: the response, in the order FR-003 writes it
RESPONSE_KEYS = [
    "request_id", "frame_uid", "model_version", "backbone", "head", "scores", "top1",
    "decision", "abstain_reason", "uncertainty", "conformal_set", "localisation", "features",
    "timing_ms", "warnings",
]  # fmt: skip
#: the 422 payload (FR-004)
ERROR_KEYS = ["request_id", "model_version", "decision", "abstain_reason", "errors", "timing_ms"]
#: H8 §6.2's uncertainty block, exactly five fields and no more (US-5.3)
UNCERTAINTY_KEYS = ["max_prob", "entropy", "ood_knn", "ood_knn_threshold", "ood_maha"]
TIMING_KEYS = ["preprocess", "backbone", "head", "total"]
#: the request log's fields (US-9.1), in order
LOG_KEYS = [
    "ts", "request_id", "frame_uid", "frame_sha256", "uri", "site_id", "plot_id", "zone_id",
    "timestamp_utc", "platform", "bbch", "bbch_null_reason", "model_version", "run_id",
    "coverage", "decision", "abstain_reason", "top1", "scores", "uncertainty", "features",
    "status", "timing_ms", "errors", "warnings",
]  # fmt: skip
#: the two fitted reasons (spec 004 US-4.2); invalid_input is this spec's own (US-5.6)
REASONS = ("low_confidence", "far_from_training")
#: N6's qualifiers (FR-010): the two H8 asks for, and the replay path reported beside them
LATENCY_QUALIFIERS = ("b1", "b32", "cached_b1", "cached_b32")
DEFAULT_COVERAGE = 0.90
#: settings_for's "the world's own service.yaml", so that None can mean "no config at all"
_CONFIG = Path("<the world's service.yaml>")
#: the contract violations of US-2.5, by the name of what each one breaks
VIOLATIONS = [
    "missing_metadata_field", "pre_rename_frame_id", "yaw_in_degrees", "bbch_null_without_reason",
    "unknown_top_level_key", "no_request_id", "unknown_option", "unfitted_coverage",
    "coverage_one_is_not_an_operating_point", "wrong_sha256", "uri_outside_the_data_root",
]  # fmt: skip


# --- the module under test -----------------------------------------------------------------------


def _module(name: str):
    """ms.service.bench and ms.service.log; until they exist every test fails with one line."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    pytest.fail(f"{name} does not exist yet (spec 006, W5)", pytrace=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the synthetic world -------------------------------------------------------------------------


def abstain_config(root: Path, **changes) -> Path:
    cfg = {
        "k": 5,
        "coverages": list(COVERAGES),
        "default_coverage": DEFAULT_COVERAGE,
        "distance_tpr": 0.95,
        "ece_bins": 15,
        "epsilon": 1e-6,
        "n3": [{"manifest": "alpha_v1", "classes": ["unknown_als"]}],
    }
    cfg.update(changes)
    path = root / f"abstain{len(list(root.glob('abstain*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def service_config(root: Path, run_id: str, **changes) -> Path:
    """configs/service.yaml as FR-012 describes it, pointing at one run; `changes` overrides
    a key, for the start-up refusals of US-10.2."""
    cfg = {
        "run_id": run_id,
        "default_coverage": DEFAULT_COVERAGE,
        "max_batch": 32,
        "device": "cpu",
        "heads_root": str(root / "heads"),
        "cache_root": str(root / "cache"),
        "data_root": str(root / "raw"),
        "backbone_configs": str(root / "backbones"),
        "request_log": str(root / "predictions" / "requests.jsonl"),
    }
    cfg.update(changes)
    path = root / f"service{len(list(root.glob('service*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def stand_ins(root: Path, manifest: Path) -> Path:
    """A data root with one file per manifest row whose bytes are its image_id, so the file's
    sha256 is the row's (synthetic.image_row). The cached-hash path hashes bytes; it never
    decodes them, so a stand-in is enough for everything but US-4.2."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for row in load_manifest(manifest, None, "extract").rows:
        (raw / f"{row['image_id']}.img").write_bytes(row["image_id"].encode())
    return raw


def odd_row(i: int) -> dict:
    image_id = f"odd_{i:05d}"
    return {
        "image_id": image_id,
        "dataset": "odd",
        "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
        "class_km2": "unknown_wm",
        "split": "holdout_unknown",
        "split_rule": "holdout_unknown",
        "group_keys": {"date": None, "phash_group": image_id},
    }


def unit16(x: np.ndarray) -> np.ndarray:
    """A direction, rounded the way a cache stores it (spec 002 FR-005) and normalised the
    way ms.heads.features hands it to a head, so the sweep scores exactly what the service
    will score for that cached row."""
    v = np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def candidates(bank: np.ndarray, classes: list[str]) -> list[np.ndarray]:
    """Directions to try, in order: away from every trained class towards a held-out one
    (a strange frame), then between two training features of different classes (a frame the
    head cannot tell apart, and which is not strange at all)."""
    away = direction("unknown_wm")
    out = [unit16(alpha * direction(classes[0]) + (1 - alpha) * 3.0 * away) for alpha in
           np.linspace(1.0, 0.0, 41)]  # fmt: skip
    step = max(1, len(bank) // 8)
    for a in bank[::step]:
        for b in bank[step // 2 + 1 :: step]:
            out += [unit16(beta * a + (1 - beta) * b) for beta in np.linspace(0.5, 0.35, 4)]
    return out


def forcing_rows(root: Path, run_meta: dict) -> dict[str, dict]:
    """An `odd_v1` cache under the run's key holding one hand-built feature per abstention
    reason, and the manifest row of each, by reason.

    The vectors are searched for, not guessed: `candidates` offers directions away from the
    training features and directions between two of them, and the first that makes
    `ms.abstain.decide` give each reason at the default coverage is kept. The search uses the
    spec's own scores (spec 004 US-4.1), so the tests assert that the service agrees with the
    fit, never with a number written here.
    """
    folder = root / "heads" / run_meta["run_id"]
    run, fit = load_run(folder), load_fit(folder, cache_root=root / "cache")
    found: dict[str, np.ndarray] = {}
    for x in candidates(fit.bank, run_meta["classes"]):
        row = {k: float(v[0]) for k, v in scores(fit, run, x[None, :]).items()}
        decision, reason = decide(fit, row, DEFAULT_COVERAGE)
        if decision == "abstain":
            found.setdefault(reason, x)
        if len(found) == len(REASONS):
            break
    missing = [r for r in REASONS if r not in found]
    assert not missing, f"the search forced no {missing}: the synthetic world cannot test them"

    rows = [odd_row(i) for i, _ in enumerate(REASONS)]
    path = root / "manifests" / "odd_v1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    path.with_name("odd_v1.meta.json").write_text(
        json.dumps({"manifest": "odd", "version": 1, "role": "holdout_unknown"}), encoding="utf-8"
    )
    write_cache(root / "cache", path, np.stack([found[r] for r in REASONS]), backbone=BACKBONE)
    for row in rows:
        (root / "raw" / f"{row['image_id']}.img").write_bytes(row["image_id"].encode())
    return dict(zip(REASONS, rows, strict=True))


def train_synthetic(root: Path, manifest: Path, seed: int = 0) -> dict:
    assert (
        heads_train.main(
            [
                "--backbone",
                BACKBONE,
                "--res",
                str(RES),
                "--head",
                "linear",
                "--seed",
                str(seed),
                "--train-manifest",
                str(manifest),
                "--split",
                "train",
                "--val-split",
                "val",
                "--cache-root",
                str(root / "cache"),
                "--heads-root",
                str(root / "heads"),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )
    runs = [
        json.loads((p / "run.json").read_text(encoding="utf-8"))
        for p in (root / "heads").iterdir()
        if (p / "run.json").is_file()
    ]
    return next(r for r in runs if r["seed"] == seed and r["head"] == "linear")


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """alpha + its cache + a linear head at seed 0 + its abstain.json + the odd features +
    the stand-in data root + a service.yaml, built once for this module."""
    root = tmp_path_factory.mktemp("service")
    manifest = make_manifest(root, "alpha")
    run_meta = train_synthetic(root, manifest)
    assert (
        importlib.import_module("ms.abstain.fit").main(
            [
                "--run",
                run_meta["run_id"],
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
    folder = root / "heads" / run_meta["run_id"]
    stand_ins(root, manifest)
    return {
        "root": root,
        "manifest": manifest,
        "run": run_meta,
        "fit": json.loads((folder / "abstain.json").read_text(encoding="utf-8")),
        "odd": forcing_rows(root, run_meta),
        "config": service_config(root, run_meta["run_id"]),
    }


def settings_for(world: dict, *, service: Path | None = _CONFIG, **changes) -> Settings:
    """The Settings the W5 service is built with. FR-012 makes the config file the source and
    the environment an override, so a test names only the pieces it changes; `service=None`
    builds a Settings that names no run at all, which is what US-10.2 refuses."""
    root = world["root"]
    if service is None:
        base = Settings(
            heads_root=root / "heads",
            cache_root=root / "cache",
            data_root=root / "raw",
            request_log=root / "predictions" / "requests.jsonl",
        )
    else:
        base = Settings.from_config(world["config"] if service is _CONFIG else service)
    return dataclasses.replace(base, **changes)


@pytest.fixture(scope="module", autouse=True)
def the_repositorys_request_log_is_never_written():
    """A test writes its rows into its own tmp root, never into `data/predictions/` (US-9,
    FR-012). The suite learnt this the hard way on `results/verdict.md` (DECISIONS 111)."""
    before = service_log.REQUEST_LOG.read_bytes() if service_log.REQUEST_LOG.exists() else None
    yield
    after = service_log.REQUEST_LOG.read_bytes() if service_log.REQUEST_LOG.exists() else None
    assert after == before, f"the tests wrote to {service_log.REQUEST_LOG}"


@pytest.fixture
def client(world):
    with TestClient(create_app(settings_for(world))) as c:
        yield c


def scored_rows(world: dict) -> list[dict]:
    rows = load_manifest(world["manifest"], "test", "evaluate").rows
    return [r for r in rows if r["class_km2"] in world["run"]["classes"]]


def a_request(world: dict, row: dict | None = None) -> dict:
    """A /v1/predict body for one synthetic row: the stand-in file is its uri, its bytes hash
    to its sha256, and the metadata is ms.service.example's (DECISIONS 30)."""
    row = row or scored_rows(world)[0]
    return request_for({**row, "path": f"{row['image_id']}.img", "width": 64, "height": 48})


def broken(world: dict, case: str) -> dict:
    """One body per contract violation of US-2.5."""
    body = copy.deepcopy(a_request(world))
    if case == "missing_metadata_field":
        del body["metadata"]["camera"]
    elif case == "pre_rename_frame_id":
        body["frame"]["frame_id"] = body["frame"].pop("frame_uid")  # DECISIONS 31
    elif case == "yaw_in_degrees":
        body["metadata"]["pose"]["yaw"] = 91.0  # H8's example; invalid in v0 (DECISIONS 29)
    elif case == "bbch_null_without_reason":
        del body["metadata"]["bbch_null_reason"]  # DECISIONS 30
    elif case == "unknown_top_level_key":
        body["extra"] = 1
    elif case == "no_request_id":
        del body["request_id"]
    elif case == "unknown_option":
        body["options"]["patch_map"] = True
    elif case == "unfitted_coverage":
        body["options"]["coverage_target"] = 0.85
    elif case == "coverage_one_is_not_an_operating_point":
        body["options"]["coverage_target"] = 1.0
    elif case == "wrong_sha256":
        body["frame"]["sha256"] = next(
            r["sha256"] for r in scored_rows(world) if r["sha256"] != body["frame"]["sha256"]
        )
    elif case == "uri_outside_the_data_root":
        body["frame"]["uri"] = "../../test_service.py"
    else:  # pragma: no cover - a typo in VIOLATIONS
        raise AssertionError(f"no such violation: {case}")
    return body


def log_rows(world: dict) -> list[dict]:
    path = world["root"] / "predictions" / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- US-2: the request, and every 422 ------------------------------------------------------------


def test_a_valid_request_is_answered(client, world):
    r = client.post("/v1/predict", json=a_request(world))
    assert r.status_code == 200, r.text
    assert list(r.json()) == RESPONSE_KEYS


@pytest.mark.parametrize("case", VIOLATIONS)
def test_every_contract_violation_is_422_with_the_validators_reason(client, world, case):
    r = client.post("/v1/predict", json=broken(world, case))
    assert r.status_code == 422, f"{case}: {r.status_code} {r.text}"
    out = r.json()
    assert list(out) == ERROR_KEYS
    assert (out["decision"], out["abstain_reason"]) == ("abstain", "invalid_input")
    assert out["errors"] and all(set(e) == {"path", "message"} for e in out["errors"])
    assert out["errors"] == sorted(out["errors"], key=lambda e: e["path"])


def test_a_body_that_is_not_json_is_422(client):
    r = client.post("/v1/predict", content=b"not json")
    assert r.status_code == 422 and r.json()["abstain_reason"] == "invalid_input"


def test_the_422_names_the_fitted_coverages(client, world):
    """US-2.4: the service may not interpolate an operating point it never measured."""
    out = client.post("/v1/predict", json=broken(world, "unfitted_coverage")).json()
    message = " ".join(e["message"] for e in out["errors"])
    assert all(coverage_key(c) in message for c in COVERAGES)


def test_a_fitted_coverage_is_matched_to_two_decimals(client, world):
    body = a_request(world)
    body["options"]["coverage_target"] = 0.9  # the same operating point as 0.90
    assert client.post("/v1/predict", json=body).status_code == 200


def test_an_mcap_uri_is_still_501(client, world):
    body = a_request(world)
    body["frame"]["uri"] = "mcap://bag/camera/1"  # open with piece 2 (DECISIONS 28, 37)
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 501 and set(r.json()) == {"request_id", "frame_uid", "detail"}


# --- US-3: the response --------------------------------------------------------------------------


def test_the_response_carries_h8_6_2s_fields(client, world):
    out = client.post("/v1/predict", json=a_request(world)).json()
    assert list(out) == RESPONSE_KEYS
    assert list(out["uncertainty"]) == UNCERTAINTY_KEYS  # five fields: no energy, no conf
    assert list(out["timing_ms"]) == TIMING_KEYS
    assert set(out["scores"]) == set(world["run"]["classes"])
    assert sum(out["scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert out["top1"] == max(out["scores"], key=out["scores"].get)
    assert out["decision"] in ("predict", "abstain")
    assert (out["conformal_set"], out["localisation"]) == (None, None)
    assert out["model_version"] == world["run"]["model_version"]
    assert out["backbone"]["input_res"] == RES and out["backbone"]["tokens"] == "cls"
    assert out["head"]["calibration"] == {"temperature": world["fit"]["temperature"]["T"]}


def cached_features(world: dict, row: dict) -> np.ndarray:
    run = load_run(world["root"] / "heads" / world["run"]["run_id"])
    m = load_manifest(world["manifest"], None, "extract")
    npz = find_cache(
        world["root"] / "cache", BACKBONE, RES, m.name, m.sha256,
        key=run.meta["backbone"]["cache_key"],
    )  # fmt: skip
    i = [r["image_id"] for r in m.rows].index(row["image_id"])
    return features(load_cache(npz), np.array([i]), run.tokens)


def test_scores_are_temperature_scaled(client, world):
    """US-3.2: the response's probabilities are ms.abstain's, not Run.predict_proba's
    (spec 004 US-5.2 and US-5.4)."""
    folder = world["root"] / "heads" / world["run"]["run_id"]
    run, fit = load_run(folder), load_fit(folder, cache_root=world["root"] / "cache")
    row = scored_rows(world)[0]
    expected = probabilities(fit, run, cached_features(world, row))[0]
    out = client.post("/v1/predict", json=a_request(world, row)).json()
    got = np.array([out["scores"][c] for c in run.classes])
    assert got == pytest.approx(expected, abs=1e-6)
    assert out["uncertainty"]["max_prob"] == pytest.approx(float(expected.max()), abs=1e-6)
    entropy = float(-(expected * np.log(np.clip(expected, 1e-12, 1.0))).sum())
    assert out["uncertainty"]["entropy"] == pytest.approx(entropy, abs=1e-6)  # nats
    assert got != pytest.approx(run.predict_proba(cached_features(world, row))[0], abs=1e-6)


def test_a_warning_never_changes_the_decision(client, world):
    """H8 §6.2: warnings never change the decision."""
    plain = client.post("/v1/predict", json=a_request(world)).json()
    body = a_request(world)
    body["options"]["return_patch_map"] = True  # a warning, never a 422 (US-2.3)
    noisy = client.post("/v1/predict", json=body).json()
    assert any("return_patch_map" in w for w in noisy["warnings"])
    assert noisy["decision"] == plain["decision"]
    assert noisy["abstain_reason"] == plain["abstain_reason"]
    assert noisy["scores"] == plain["scores"] and noisy["localisation"] is None


def test_the_slices_placeholder_warning_is_gone(client, world):
    """The W2 slice warned on every answer that there was no calibration or abstention yet
    (DECISIONS 37). From W5 there is, and the warning would be a lie."""
    out = client.post("/v1/predict", json=a_request(world)).json()
    assert not any("no calibration or abstention yet" in w for w in out["warnings"])


# --- US-4: the cached-hash path ------------------------------------------------------------------


def test_a_cached_hash_is_answered_without_the_backbone(client, world):
    out = client.post("/v1/predict", json=a_request(world)).json()
    assert out["features"] == "cache"
    assert out["timing_ms"]["backbone"] == 0
    assert client.get("/v1/health").json()["cached_frames"] >= sum(COUNTS["test"].values())


def test_the_hash_is_recomputed_from_the_bytes_at_uri(client, world):
    """DECISIONS 32: a wrong hash in a request cannot select another image's features."""
    out = client.post("/v1/predict", json=broken(world, "wrong_sha256"))
    assert out.status_code == 422 and out.json()["errors"][0]["path"] == "frame.sha256"


# --- US-5: abstention ----------------------------------------------------------------------------


@pytest.mark.parametrize("reason", REASONS)
def test_abstain_is_reachable_with_both_reasons(client, world, reason):
    """H8 §6.11 item 1. The rows were walked away from the training features until
    ms.abstain.decide gave each reason; the service has to agree with the fit."""
    out = client.post("/v1/predict", json=a_request(world, world["odd"][reason])).json()
    assert out["decision"] == "abstain"
    assert out["abstain_reason"] == reason
    assert out["scores"] and out["top1"] in world["run"]["classes"]  # US-3.3, Clarification 12


def test_the_uncertainty_block_carries_the_distance_and_its_threshold(client, world):
    out = client.post("/v1/predict", json=a_request(world)).json()
    tau = world["fit"]["thresholds"][coverage_key(DEFAULT_COVERAGE)]
    assert out["uncertainty"]["ood_knn_threshold"] == pytest.approx(tau["tau_knn"])
    assert isinstance(out["uncertainty"]["ood_knn"], float)
    assert isinstance(out["uncertainty"]["ood_maha"], float)  # a second opinion, never decisive


def test_only_the_confidence_gate_moves_with_the_coverage(client, world):
    """US-2.4 with spec 004 US-3.1: tau_knn is the same value at every declared coverage, so
    the threshold in the response does not move; the confidence gate does."""
    seen = set()
    for coverage in COVERAGES:
        body = a_request(world, world["odd"]["low_confidence"])
        body["options"]["coverage_target"] = coverage
        out = client.post("/v1/predict", json=body).json()
        assert out["uncertainty"]["ood_knn_threshold"] == pytest.approx(
            world["fit"]["thresholds"][coverage_key(coverage)]["tau_knn"]
        )
        seen.add((out["decision"], out["abstain_reason"]))
    assert seen  # the operating point is read per request, not pinned at start-up


# --- US-6: the batch -----------------------------------------------------------------------------


def test_predict_batch_answers_every_item_in_order(client, world):
    bodies = [a_request(world, r) for r in scored_rows(world)[:4]]
    r = client.post("/v1/predict_batch", json={"requests": bodies})
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(out) == {"responses", "n_invalid"} and out["n_invalid"] == 0
    assert len(out["responses"]) == len(bodies)
    for body, response in zip(bodies, out["responses"], strict=True):
        assert response["request_id"] == body["request_id"]
        assert response["frame_uid"] == body["frame"]["frame_uid"]
        assert list(response) == RESPONSE_KEYS
    alone = client.post("/v1/predict", json=bodies[0]).json()
    for key in ("scores", "decision", "abstain_reason", "top1", "model_version"):
        assert out["responses"][0][key] == alone[key]  # one function builds both (FR-014)


def test_one_invalid_item_costs_only_that_item(client, world):
    """US-6.2: a replayed mission must not lose its good frames to one broken record."""
    bodies = [broken(world, "missing_metadata_field"), a_request(world)]
    out = client.post("/v1/predict_batch", json={"requests": bodies}).json()
    assert out["n_invalid"] == 1
    assert out["responses"][0]["abstain_reason"] == "invalid_input"
    assert out["responses"][0]["errors"]
    assert out["responses"][1]["decision"] in ("predict", "abstain")


@pytest.mark.parametrize(
    "envelope", [[], {"frames": []}, {"requests": {}}, {"requests": []}, {"requests": None}]
)
def test_a_broken_envelope_is_422(client, envelope):
    r = client.post("/v1/predict_batch", json=envelope)
    assert r.status_code == 422
    assert r.json()["abstain_reason"] == "invalid_input"


def test_a_batch_longer_than_max_batch_is_422(client, world):
    r = client.post("/v1/predict_batch", json={"requests": [a_request(world)] * 33})
    assert r.status_code == 422 and "max_batch" in json.dumps(r.json())


# --- US-7 and US-8: the card summary, and model_version ------------------------------------------


def test_health_costs_nothing_and_names_the_model(client, world):
    out = client.get("/v1/health").json()
    assert out["status"] == "ok" and out["model_version"] == world["run"]["model_version"]


def test_the_card_summary_reproduces_the_run(client, world):
    """H8 §6.11 item 2: everything the card needs to rebuild the run is in /v1/model."""
    out = client.get("/v1/model").json()
    run = world["run"]
    assert out["model_version"] == run["model_version"] and out["run_id"] == run["run_id"]
    assert out["backbone"]["weights_sha256"] == run["backbone"]["weights_sha256"]
    assert out["backbone"]["cache_key"] == run["backbone"]["cache_key"]
    assert out["backbone"]["revision"] == run["backbone"]["revision"]
    assert out["head"]["id"] == run["head"] and out["head"]["seed"] == run["seed"]
    assert out["head"]["config_sha256"] == run["head_config"]["sha256"]
    assert out["train"]["manifest_sha256"] == run["train"]["manifest_sha256"]
    assert out["train"]["split"] == run["train"]["split"]
    assert out["classes"] == run["classes"] and out["quotable"] == run["quotable"]
    assert [float(c) for c in out["coverages"]] == list(COVERAGES)
    assert out["coverage"] == DEFAULT_COVERAGE
    assert out["temperature"] == world["fit"]["temperature"]["T"]
    assert "card" in out  # cards/<model_version>.md; null until the W5 Registry task


def test_model_version_is_the_same_string_everywhere(client, world):
    version = world["run"]["model_version"]
    assert world["fit"]["model_version"] == version
    assert client.get("/v1/model").json()["model_version"] == version
    assert client.get("/v1/health").json()["model_version"] == version
    assert client.post("/v1/predict", json=a_request(world)).json()["model_version"] == version
    assert version.startswith(f"msv0.1+{BACKBONE}@{RES}.linear.man-")


def test_model_version_names_the_model_and_the_run_id_names_the_fit(world):
    """US-8.3: two seeds share the string and not the run id, so nobody reads model_version
    as a unique key."""
    other = train_synthetic(world["root"], world["manifest"], seed=1)
    assert other["model_version"] == world["run"]["model_version"]
    assert other["run_id"] != world["run"]["run_id"]


# --- US-9: the request log -----------------------------------------------------------------------


def test_every_request_and_response_is_one_line_of_the_log(client, world):
    before = len(log_rows(world))
    body = a_request(world)
    out = client.post("/v1/predict", json=body).json()
    rows = log_rows(world)
    assert len(rows) == before + 1
    row = rows[-1]
    assert list(row) == LOG_KEYS
    assert row["request_id"] == body["request_id"]
    assert row["frame_uid"] == body["frame"]["frame_uid"]
    assert row["frame_sha256"] == body["frame"]["sha256"]
    assert row["bbch"] is None and row["bbch_null_reason"] == "not_recorded"  # DECISIONS 30
    assert row["status"] == 200 and row["decision"] == out["decision"]
    assert row["scores"] == out["scores"] and row["model_version"] == out["model_version"]
    assert row["run_id"] == world["run"]["run_id"] and row["coverage"] == DEFAULT_COVERAGE
    assert row["features"] == out["features"]


def test_a_422_is_logged_too(client, world):
    """H8 §6.2: contract violations are logged — they are the signal piece 3 wants."""
    before = len(log_rows(world))
    client.post("/v1/predict", json=broken(world, "yaw_in_degrees"))
    rows = log_rows(world)
    assert len(rows) == before + 1
    assert rows[-1]["status"] == 422 and rows[-1]["abstain_reason"] == "invalid_input"
    assert rows[-1]["errors"]


def test_a_batch_is_logged_one_line_per_frame(client, world):
    """US-6.5: piece 3's predictions table is per frame."""
    before = len(log_rows(world))
    bodies = [a_request(world, r) for r in scored_rows(world)[:3]]
    client.post("/v1/predict_batch", json={"requests": bodies})
    assert len(log_rows(world)) == before + 3


# --- US-10: which model is served ----------------------------------------------------------------


def test_the_served_run_comes_from_the_config(world):
    with TestClient(create_app(Settings.from_config(world["config"]))) as c:
        assert c.get("/v1/model").json()["run_id"] == world["run"]["run_id"]


def test_the_service_refuses_to_start_without_an_abstain_json(world, tmp_path):
    """US-5.4: a service that cannot abstain is not the service H8 §6.11 item 1 asks for."""
    source = world["root"] / "heads" / world["run"]["run_id"]
    folder = tmp_path / "heads" / world["run"]["run_id"]
    folder.mkdir(parents=True)
    for name in ("run.json", "head.pt"):
        (folder / name).write_bytes((source / name).read_bytes())
    with (
        pytest.raises(Exception, match="missing_abstain"),
        TestClient(create_app(settings_for(world, heads_root=tmp_path / "heads"))),
    ):
        pass


def test_the_service_refuses_when_no_run_is_named(world):
    """US-10.2: "the newest run under data/heads" was the slice's rule (DECISIONS 37) and
    stops here — with hundreds of runs on disk, newest is not a choice anybody made."""
    with (
        pytest.raises(Exception, match="ambiguous_run|bad_config"),
        TestClient(create_app(settings_for(world, run=None, service=None))),
    ):
        pass


def test_the_service_refuses_a_run_it_cannot_find(world):
    config = service_config(world["root"], "no-such-run")
    with (
        pytest.raises(Exception, match="missing_run"),
        TestClient(create_app(Settings.from_config(config))),
    ):
        pass


# --- US-11: N6 -----------------------------------------------------------------------------------


def an_n6_row(world: dict, **changes) -> dict:
    run = world["run"]
    row = {
        "ts": "2026-10-15T09:00:00Z", "number": "N6", "metric": "latency_ms:b1", "value": 12.5,
        "ci_low": 11.0, "ci_high": 18.0, "n": 64, "backbone_id": BACKBONE, "res": RES,
        "token_type": "cls", "head": "linear", "seed": 0,
        "train_manifest": str(world["manifest"]),
        "train_manifest_sha256": run["train"]["manifest_sha256"], "train_split": "train",
        "test_manifest": str(world["manifest"]),
        "test_manifest_sha256": run["train"]["manifest_sha256"], "test_split": "test",
        "split_rule": "blocked:date", "coverage": 1.0, "classes": run["classes"],
        "run_id": run["run_id"], "model_version": run["model_version"],
        "cache_key": run["backbone"]["cache_key"], "quotable": False, "superseded": None,
        "notes": "device cpu; 8 warm-up; batch 1",
    }  # fmt: skip
    row.update(changes)
    return row


def test_the_n_table_accepts_a_latency_row_and_rejects_a_bare_one(world):
    """FR-010: spec 005 FR-005's vocabulary gains latency_ms with the qualifier kind batch."""
    assert "latency_ms" in n_table.METRICS
    for qualifier in LATENCY_QUALIFIERS:
        assert n_table.validate_row(an_n6_row(world, metric=f"latency_ms:{qualifier}")) == []
    for bad in ("latency_ms", "latency_ms:b7", "latency_ms:cuda", "latency_ms:healthy"):
        problems = n_table.validate_row(an_n6_row(world, metric=bad))
        assert any(p.startswith("bad_metric") for p in problems), bad


def test_bench_writes_the_n6_rows_and_a_compute_log_row(world, capsys):
    bench = _module("ms.service.bench")
    root = world["root"]
    code = bench.main(
        [
            "--service",
            str(world["config"]),
            "--batch",
            "1",
            "--n",
            "4",
            "--warmup",
            "1",
            "--device",
            "cpu",
            "--path",
            "cache",
            "--n-table",
            str(root / "n6.jsonl"),
            "--compute-log",
            str(root / "log.jsonl"),
        ]  # fmt: skip
    )
    assert code == 0, capsys.readouterr().out
    rows = list(n_table.read_rows(root / "n6.jsonl"))
    assert rows and all(r["number"] == "N6" for r in rows)
    assert {r["metric"] for r in rows} <= {f"latency_ms:{q}" for q in LATENCY_QUALIFIERS}
    for row in rows:
        assert n_table.validate_row(row) == []
        assert row["value"] > 0 and row["ci_low"] <= row["value"] <= row["ci_high"]
        assert (row["coverage"], row["seed"]) == (1.0, 0)  # one seed, a percentile interval
        assert "cpu" in (row["notes"] or "")
    assert any(r["step"] == "serve" for r in compute_log.read_rows(root / "log.jsonl"))
    before = len(rows)
    assert (
        bench.main(
            [
                "--service",
                str(world["config"]),
                "--batch",
                "1",
                "--n",
                "4",
                "--warmup",
                "1",
                "--device",
                "cpu",
                "--path",
                "cache",
                "--n-table",
                str(root / "n6.jsonl"),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )
    assert len(list(n_table.read_rows(root / "n6.jsonl"))) == before  # spec 005 FR-004


# --- US-4.2: features computed on request (the fixture world) ------------------------------------


@pytest.fixture(scope="module")
def fixture_world(tmp_path_factory) -> dict:
    """ibean_30 with test_slice.py's tiny random-init backbone: the only world with real
    pixels, so the only one where an uncached frame can be embedded.

    Its validation slice is two rows (1 healthy, 1 rust), so the only coverage it can carry
    is 0.50: spec 004 US-3.3 needs ceil(1 / (1 - c)) rows. The operating point is not what
    these two tests are about.
    """
    slice_tests = importlib.import_module("test_slice")
    root = tmp_path_factory.mktemp("service_fixture")
    configs = slice_tests.tiny_backbone(root)
    assert (
        cache_extract.main(
            [
                "--backbone",
                slice_tests.BACKBONE,
                "--res",
                "28",
                "--manifest",
                str(FIXTURE_MANIFEST),
                "--raw-root",
                str(FIXTURE),
                "--config-dir",
                str(configs),
                "--cache-root",
                str(root / "cache"),
                "--compute-log",
                str(root / "log.jsonl"),
                "--device",
                "cpu",
            ]  # fmt: skip
        )
        == 0
    )
    assert (
        heads_train.main(
            [
                "--backbone",
                slice_tests.BACKBONE,
                "--res",
                "28",
                "--head",
                "linear",
                "--seed",
                "0",
                "--train-manifest",
                str(FIXTURE_MANIFEST),
                "--split",
                "train",
                "--val-split",
                "val",
                "--allow-test-only",
                "--cache-root",
                str(root / "cache"),
                "--heads-root",
                str(root / "heads"),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )
    (folder,) = [p for p in (root / "heads").iterdir() if (p / "run.json").is_file()]
    run_meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    cfg = abstain_config(root, coverages=[0.50], default_coverage=0.50, k=3, n3=[])
    assert (
        importlib.import_module("ms.abstain.fit").main(
            [
                "--run",
                run_meta["run_id"],
                "--heads-root",
                str(root / "heads"),
                "--cache-root",
                str(root / "cache"),
                "--config",
                str(cfg),
                "--compute-log",
                str(root / "log.jsonl"),
            ]  # fmt: skip
        )
        == 0
    )
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in FIXTURE_MANIFEST.read_text(encoding="utf-8").splitlines()]
    row = next(r for r in rows if r["split"] == "test")
    with Image.open(FIXTURE / row["path"]) as im:
        # the same pixels, different bytes: a lossless re-container, with the EXIF rotation
        # baked in as ms.cache.preprocess would apply it. A hash no cache holds.
        ImageOps.exif_transpose(im).convert("RGB").save(raw / "unseen.png")
    return {
        "root": root,
        "configs": configs,
        "run": run_meta,
        "cached_row": row,
        "unseen": raw / "unseen.png",
        "config": service_config(
            root,
            run_meta["run_id"],
            default_coverage=0.50,
            data_root=str(raw),
            backbone_configs=str(configs),
        ),
    }


@pytest.fixture
def fixture_client(fixture_world):
    raw = fixture_world["root"] / "raw"
    for row in [
        json.loads(line) for line in FIXTURE_MANIFEST.read_text(encoding="utf-8").splitlines()
    ]:
        target = raw / Path(row["path"]).name
        if not target.exists():
            target.write_bytes((FIXTURE / row["path"]).read_bytes())
    with TestClient(create_app(Settings.from_config(fixture_world["config"]))) as c:
        yield c


def fixture_request(fixture_world: dict, *, unseen: bool) -> dict:
    row = fixture_world["cached_row"]
    if unseen:
        path = fixture_world["unseen"]
        row = {**row, "image_id": "unseen", "path": path.name, "sha256": sha256(path)}
    else:
        row = {**row, "path": Path(row["path"]).name}
    body = request_for(row)
    body["options"]["coverage_target"] = 0.50  # the only one this world can carry
    return body


def test_an_uncached_frame_is_computed_on_request(fixture_client, fixture_world):
    """US-4.2: the 501 the W2 slice answered (DECISIONS 37) becomes an answer in W5."""
    r = fixture_client.post("/v1/predict", json=fixture_request(fixture_world, unseen=True))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["features"] == "computed"
    assert out["timing_ms"]["backbone"] > 0 and out["timing_ms"]["preprocess"] > 0
    assert set(out["scores"]) == set(fixture_world["run"]["classes"])


def test_the_cached_and_the_computed_path_agree(fixture_client, fixture_world):
    """US-4.4 and SC-3: one device, one compute dtype and the float16 rounding of US-4.2, so
    the demo's cheap path and the real path are one model, not two within a tolerance."""
    cached = fixture_client.post(
        "/v1/predict", json=fixture_request(fixture_world, unseen=False)
    ).json()
    computed = fixture_client.post(
        "/v1/predict", json=fixture_request(fixture_world, unseen=True)
    ).json()
    assert (cached["features"], computed["features"]) == ("cache", "computed")
    assert computed["decision"] == cached["decision"] and computed["top1"] == cached["top1"]
    for klass, value in cached["scores"].items():
        assert computed["scores"][klass] == pytest.approx(value, abs=1e-6)
