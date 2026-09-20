"""The registry and the model card (S4.6's W5 "Registry + card"; H8 §6.8).

    python -m ms.registry card     --run <run_id>            # cards/<model_version>.md
    python -m ms.registry log      [--run <run_id> ...]      # one MLflow run per head run
    python -m ms.registry register --run <run_id>            # only with a card that is current

One MLflow **experiment per backbone** (`piece4-heads-<backbone_id>`), one run per head run,
logging the manifest hashes, the preprocessing string, the seed, N1-N3 and the calibration
temperature (H8 §6.8). A model is **registered only when it has a card** — and the card has
to be the one the artefacts render now, so a model whose numbers moved cannot be registered
until its card is written again. That is the whole point of the gate: the card is what the
champion/challenger rule reads (H8 D-7).

The card itself is `cards/<model_version>.md`, rendered from `cards/TEMPLATE.md`: the prose
is the template's and the same for every v0 model, and every number, hash and licence line
comes from the run, its `abstain.json`, the frozen manifests and `results/n_table.jsonl`.
Nobody edits a rendered card by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ms.abstain import AbstainError, Fit, coverage_key, load_fit
from ms.cache import BACKBONE_CONFIGS, backbone_config
from ms.data.manifests import git_sha, load_manifest
from ms.eval import N_TABLE, read_rows
from ms.heads import HEADS_ROOT, list_runs, resolve_path

#: repository root: model_service/src/ms/registry.py -> ../../../
REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS = REPO_ROOT / "model_service" / "cards"
TEMPLATE = CARDS / "TEMPLATE.md"
LOCAL_MLFLOW = REPO_ROOT / "mlruns"
#: H8 §6.8: one experiment per backbone
EXPERIMENT = "piece4-heads-{backbone_id}"
#: D-7's "model set behind the service interface": one name, one version per model that joins
REGISTERED_MODEL = "piece4-bean-disease-v0"
#: the blocks the template asks for
BLOCKS = (
    "classes_sentence",
    "n_classes",
    "model_block",
    "data_block",
    "evaluation_block",
    "operating_point_block",
    "failure_modes_block",
    "licences_block",
    "reproduce_block",
)
CARD_VERSION = "1"
#: why an N2 row's n is smaller than the target's test split (spec 005 US-7.3)
N2_NOTE = (
    "\n\nA cross-dataset row decides among the classes **both** sides have (spec 005 US-7.3), "
    "so rows of a class this model does not carry are not scored at all and `n` is smaller "
    "than the target's test split by exactly those rows. They are not errors and not "
    "abstentions; they were never put to it."
)
#: why an N3 row names several manifests: its knowns are the run's own in-domain test rows
#: and its unknowns the held-out set, so the row carries both sides (spec 004 Clarification 6)
N3_NOTE = (
    "\n\nAn N3 row is scored on two sets at once: the run's own in-domain test rows, which "
    'are the knowns, and the held-out set, which are the unknowns. "Tested on" therefore '
    "names the training manifests and the held-out source together, and a manifest that is "
    "both appears twice.\n\nA held-out class can be a new **disease** without being a new "
    "**dataset**: angular leaf spot here is held out of training but comes from a set this "
    "model trained the rest of, while white mould is a set it has never seen at all (spec 004 "
    "Clarification 7). The two AUROCs are not measuring the same kind of strangeness."
)


class RegistryError(Exception):
    """A refusal, with the vocabulary the command prints: `<reason>: <what>`."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _n(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _interval(row: dict[str, Any]) -> str:
    if row.get("ci_low") is None:
        return "—"
    return f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"


def under_repo(path: Path) -> str:
    """A path as the repository sees it, or as it is when it lies outside (a test's tmp)."""
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def stem(joined: str | None) -> str:
    """A manifest field as a name: several manifests join their paths with `+`, and so do
    their names (spec 005 FR-001 edge cases)."""
    if not joined:
        return "—"
    return "+".join(Path(p).stem for p in str(joined).split("+"))


def table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(rows_) + " |" for rows_ in (r for r in rows)]
    return "\n".join(lines)


# --- what a card is rendered from ----------------------------------------------------------------


@dataclass
class Model:
    """One head run, everything the card and the registry read about it."""

    folder: Path
    meta: dict[str, Any]
    fit: Fit
    manifests: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    config: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def model_version(self) -> str:
        return self.meta["model_version"]

    @property
    def backbone(self) -> dict[str, Any]:
        return self.meta["backbone"]


def find_run(run: str, heads_root: Path) -> Path:
    for folder in (Path(run), Path(heads_root) / run):
        if (folder / "run.json").is_file():
            return folder
    raise RegistryError("missing_run", f"{run}: no run.json there or under {heads_root}")


@functools.cache
def manifest_facts(path: str) -> dict[str, Any]:
    """One training manifest: its sidecar, and the distinct licence and attribution lines its
    rows carry (spec 001 FR-003). A manifest over two source records has two of them."""
    resolved = resolve_path(path)
    loaded = load_manifest(resolved, None, "serve")
    side = json.loads(resolved.with_name(f"{resolved.stem}.meta.json").read_text(encoding="utf-8"))
    sources: dict[tuple[str, str, str], int] = {}
    for row in loaded.rows:
        # spec 001 FR-003 puts both on every row; a manifest without them is one the card
        # has to say it cannot vouch for rather than crash on
        key = (
            row["dataset"],
            row.get("licence") or "— (not in the manifest)",
            row.get("attribution") or "— (not in the manifest)",
        )
        sources[key] = sources.get(key, 0) + 1
    return {
        "path": path,
        "name": resolved.stem,
        "sha256": loaded.sha256,
        "role": side.get("role"),
        "rule": side.get("rule"),
        "counts": side.get("counts", {}),
        "splits": side.get("splits", {}),
        "sources": [
            {"dataset": d, "licence": lic, "attribution": att, "n": n}
            for (d, lic, att), n in sorted(sources.items())
        ],
    }


def load_model(
    folder: Path,
    cache_root: Path | None,
    n_table: Path,
    backbone_configs: Path = BACKBONE_CONFIGS,
) -> Model:
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    try:
        fit = load_fit(folder, cache_root=cache_root, rebuild=False)
    except AbstainError as exc:
        raise RegistryError(
            "missing_abstain",
            f"{meta['run_id']}: {exc}. A model without an operating point has no card",
        ) from None
    manifests = [manifest_facts(p) for p in meta["train"]["manifest"].split("+")]
    rows = rows_of(meta["run_id"], n_table)
    try:
        config = backbone_config(meta["backbone"]["backbone_id"], Path(backbone_configs))
    except FileNotFoundError:
        config = {}
    return Model(folder, meta, fit, manifests, rows, config)


@functools.lru_cache(maxsize=4)
def _by_run(n_table: str) -> dict[str, list[dict[str, Any]]]:
    """The table once, indexed by the runs each row stands on. A card for every run would
    otherwise read 25,000 rows as many times as there are runs."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in read_rows(Path(n_table)):
        if row.get("superseded"):
            continue
        for rid in str(row.get("run_id") or "").split("+"):
            out.setdefault(rid, []).append(row)
    return out


def rows_of(run_id: str, n_table: Path) -> list[dict[str, Any]]:
    """The N-table rows this run stands behind: its own, and the five-seed aggregates it is
    one of. Superseded rows are left out — a retired number is not a model's evidence."""
    return list(_by_run(str(n_table)).get(run_id, []))


# --- the blocks ----------------------------------------------------------------------------------


def model_block(model: Model) -> str:
    bb, meta = model.backbone, model.meta
    return table(
        ["Part", "What"],
        [
            ["`model_version`", f"`{model.model_version}`"],
            ["Backbone", f"`{bb['hf_id']}` (`{bb['backbone_id']}`)"],
            ["Backbone revision", f"`{bb['revision']}`"],
            ["Weights sha256", f"`{bb['weights_sha256']}`"],
            ["Input resolution", f"{bb['res']} px"],
            ["Features", f"`{meta['tokens']}`, {meta['input_dim']}-d, frozen (never fine-tuned)"],
            ["Preprocessing", f"`{bb['preprocess_string']}`"],
            ["Compute dtype", f"`{bb['compute_dtype']}`, stored float16"],
            ["Cache key", f"`{bb['cache_key']}`"],
            ["Head", f"`{meta['head']}`, recipe `{meta['head_config']['path']}`"],
            ["Head recipe sha256", f"`{meta['head_config']['sha256']}`"],
            ["Seed", f"{meta['seed']} (the numbers below are over seeds 0–4)"],
            ["Classes", ", ".join(f"`{c}`" for c in meta["classes"])],
            ["Training epochs", f"{meta['fit']['epochs_run']}, best {meta['fit']['best_epoch']}"],
            ["Quotable", "yes" if meta["quotable"] else "**no** — its numbers are never reported"],
        ],
    )


def data_block(model: Model) -> str:
    parts = []
    for m in model.manifests:
        counts = m["counts"].get("split", {})
        classes = m["counts"].get("class_km2", {})
        parts.append(
            f"### `{m['name']}` — {m['rule']}, role `{m['role']}`\n\n"
            + table(
                ["Field", "Value"],
                [
                    ["Manifest", f"`{m['path']}`"],
                    ["sha256", f"`{m['sha256']}`"],
                    ["Rows", str(m["counts"].get("rows", "—"))],
                    ["Splits", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "—"],
                    ["Classes", ", ".join(f"{k} {v}" for k, v in sorted(classes.items())) or "—"],
                    ["Split rule", f"`{m['rule']}`"],
                ],
            )
            + "\n\nSources, licences and the attribution each one requires:\n\n"
            + table(
                ["Dataset", "Rows", "Licence", "Attribution"],
                [
                    [s["dataset"], str(s["n"]), f"`{s['licence']}`", s["attribution"]]
                    for s in m["sources"]
                ],
            )
        )
    return "\n\n".join(parts)


#: numbers a card quotes per seed, because they have no five-seed aggregate: N6 is one
#: measurement with a percentile interval, as N4 is (spec 006 US-11.3; DECISIONS 54)
PER_SEED_NUMBERS = ("N4", "N6")


def _aggregates(model: Model, number: str, coverage: float | None = 1.0) -> list[dict[str, Any]]:
    """The rows a card may quote for one number: the five-seed aggregates, or the rows
    themselves for a number that never has one."""
    reported = number in PER_SEED_NUMBERS
    return [
        r
        for r in model.rows
        if r["number"] == number
        and (reported or r["seed"] is None)
        and (coverage is None or r["coverage"] == coverage)
    ]


def evaluation_block(model: Model) -> str:
    parts = []
    for number, title in (
        ("N1", "N1 — within-dataset, blocked"),
        ("N2", "N2 — cross-dataset"),
        ("N3", "N3 — the unknowns it was never shown"),
        ("N6", "N6 — latency"),
    ):
        rows = _aggregates(model, number)
        body = table(
            ["Metric", "Value", "95 % interval", "Tested on", "Split rule", "n"],
            [
                [
                    f"`{r['metric']}`",
                    _n(r["value"]),
                    _interval(r),
                    f"`{stem(r['test_manifest'])}` / {r['test_split']}",
                    f"`{r['split_rule']}`",
                    str(r["n"]),
                ]
                for r in sorted(rows, key=lambda r: (str(r["test_manifest"]), r["metric"]))
            ],
        )
        note = {"N2": N2_NOTE, "N3": N3_NOTE}.get(number, "") if rows else ""
        parts.append(f"### {title}\n\n{body}{note}")
    selective = _aggregates(model, "N2", coverage=None) + _aggregates(model, "N1", coverage=None)
    at_coverage = [r for r in selective if r["coverage"] != 1.0 and r["metric"] == "selective_risk"]
    if at_coverage:
        parts.append(
            "### Selective risk, by declared coverage\n\n"
            + table(
                ["Coverage", "Selective risk", "95 % interval", "Tested on"],
                [
                    [
                        f"{r['coverage']:.2f}",
                        _n(r["value"]),
                        _interval(r),
                        f"`{stem(r['test_manifest'])}` / {r['test_split']}",
                    ]
                    for r in sorted(
                        at_coverage, key=lambda r: (str(r["test_manifest"]), r["coverage"])
                    )
                ],
            )
        )
    missing = [n for n in ("N1", "N2", "N3", "N6") if not _aggregates(model, n)]
    if missing:
        parts.append(
            "**Not measured for this model**: "
            + ", ".join(missing)
            + ". An honest blank, not a zero — "
            + "see `docs/piece4-work-plan.md` and `model_service/DECISIONS.md` for why."
        )
    return "\n\n".join(parts)


def operating_point_block(model: Model) -> str:
    fit, meta = model.fit, model.fit.meta
    default = meta["config"].get("default_coverage")
    rows = []
    for coverage in fit.coverages_str:
        t = meta["thresholds"][coverage]
        rows.append(
            [
                f"{coverage}{' (default)' if float(coverage) == float(default) else ''}",
                f"{t['tau_conf']:.6f}",
                f"{t['tau_knn']:.6f}",
                f"{t['tau_maha']:.6f}",
                f"{t['coverage_achieved']:.4f}",
            ]
        )
    fitted_on = meta["fitted_on"]
    return (
        table(
            [
                "Declared coverage",
                "`tau_conf`",
                "`tau_knn`",
                "`tau_maha` (reported)",
                "Achieved on the slice",
            ],
            rows,
        )
        + "\n\n"
        + table(
            ["Field", "Value"],
            [
                ["Temperature", f"{fit.temperature:.6f}"],
                ["k (nearest neighbours)", str(fit.k)],
                [
                    "Fitted on",
                    f"`{fitted_on['val_split']}` of `{stem(fitted_on['manifest'])}`, "
                    f"{fitted_on['n_val']} rows — the in-domain validation slice only",
                ],
                ["Bank", f"{fitted_on['n_train']} training features"],
                [
                    "Decision",
                    "abstain when `conf < tau_conf` (`low_confidence`) or `knn > tau_knn` "
                    "(`far_from_training`); the Mahalanobis distance is reported, never "
                    "consulted",
                ],
            ],
        )
    )


def failure_modes_block(model: Model) -> str:
    """What the N3 rows say about the two diseases the model was never shown."""
    rows = []
    for r in sorted(_aggregates(model, "N3"), key=lambda r: (str(r["classes"]), r["metric"])):
        if not r["metric"].startswith("auroc:"):
            continue
        rows.append(
            [
                ", ".join(r["classes"]),
                f"`{r['metric'].split(':', 1)[1]}`",
                _n(r["value"]),
                _interval(r),
                f"`{stem(r['test_manifest'])}`",
            ]
        )
    unknown = [
        [
            ", ".join(r["classes"]),
            f"{r['coverage']:.2f}",
            _n(r["value"]),
            f"`{stem(r['test_manifest'])}`",
        ]
        for r in sorted(
            [r for r in _aggregates(model, "N3", coverage=None) if r["metric"] == "unknown_recall"],
            key=lambda r: (str(r["classes"]), r["coverage"]),
        )
    ]
    return (
        "How well the strangeness scores tell an unseen disease from a known one "
        "(AUROC, 0.5 is a coin toss):\n\n"
        + table(["Held-out", "Score", "AUROC", "95 % interval", "Knowns + held-out source"], rows)
        + "\n\nAnd the share of those images the model actually refuses, at each declared "
        "coverage:\n\n"
        + table(["Held-out", "Coverage", "Unknown recall", "Knowns + held-out source"], unknown)
    )


def licences_block(model: Model) -> str:
    """The backbone's licence comes from its config, the data's from the manifest rows. For a
    gated backbone, H8 §6.8 also wants the date its terms were accepted."""
    bb, cfg = model.backbone, model.config
    terms = "No restriction beyond its own licence"
    if bb["research_only_until_c5"]:
        accepted = cfg.get("terms_accepted") or {}
        when = accepted.get("date")
        terms = (
            "**Research only until H6 C5 is signed** (DECISIONS 13); nothing built with it "
            "leaves the university machines"
            + (
                f". Terms accepted {when} on the account `{accepted.get('account')}`, "
                f"{accepted.get('scope', 'research evaluation only')}"
                if when
                else ". The date the terms were accepted is not recorded in the backbone "
                "config, and H8 §6.8 asks for it"
            )
        )
    rows = [
        [
            "Backbone",
            f"`{bb['hf_id']}`",
            str(cfg.get("licence") or "— (not in the backbone config)"),
            terms,
        ]
    ]
    seen = set()
    for what, facts in (
        *(("Trained on", m) for m in model.manifests),
        *(("Evaluated on", m) for m in evaluation_manifests(model)),
    ):
        for s in facts["sources"]:
            if (s["dataset"], what) in seen:
                continue
            seen.add((s["dataset"], what))
            rows.append([what, f"`{s['dataset']}`", s["licence"], _terms(s)])
    return table(["What", "Which", "Licence", "Terms"], rows) + "\n\n" + attributions(model)


def _terms(source: dict[str, Any]) -> str:
    """What the licence actually asks of us — not what it is polite to do anyway."""
    licence = str(source["licence"]).upper()
    if licence.startswith("CC0"):
        return (
            "No attribution required. The line below is given anyway, because the authors "
            "made the dataset and a paper that used it without saying so would be poorer"
        )
    if licence.startswith(("CC-BY", "CC BY")):
        return "**Attribution required**: the line below must travel with any use"
    if licence == "MIT":
        return "The licence text and its copyright notice must travel with any redistribution"
    return "Read the licence before any use beyond this evaluation"


def evaluation_manifests(model: Model) -> list[dict[str, Any]]:
    """The manifests this model's numbers were measured on, and not trained on. Their images
    are in no training set, and their licences still reach the card: a CC BY 4.0 set that
    produced a number in the table has to be attributed wherever that number goes."""
    trained = {m["name"] for m in model.manifests}
    names: dict[str, str] = {}
    for row in model.rows:
        for path in str(row.get("test_manifest") or "").split("+"):
            if path and Path(path).stem not in trained:
                names.setdefault(Path(path).stem, path)
    return [manifest_facts(path) for _, path in sorted(names.items())]


def attributions(model: Model) -> str:
    """Every attribution line the card owes, once each, ready to be copied."""
    lines, seen = [], set()
    for facts in [*model.manifests, *evaluation_manifests(model)]:
        for s in facts["sources"]:
            if s["attribution"] in seen:
                continue
            seen.add(s["attribution"])
            lines.append(f"- **{s['dataset']}** — {s['attribution']}")
    return "The attribution lines, in full:\n\n" + "\n".join(lines)


def reproduce_block(model: Model) -> str:
    meta = model.meta
    bb = model.backbone
    manifests = " ".join(m["path"] for m in model.manifests)
    return (
        "```bash\n"
        f"make cache BB={bb['backbone_id']} RES={bb['res']} DS=<each dataset>\n"
        f"python -m ms.heads.train --backbone {bb['backbone_id']} --res {bb['res']} \\\n"
        f"    --head {meta['head']} --seed {meta['seed']} \\\n"
        f"    --train-manifest {manifests} \\\n"
        f"    --split {meta['train']['split']} --val-split {meta['val']['split']}\n"
        'make abstain ARGS="--run ' + meta["run_id"] + '"\n'
        "make eval\n"
        "```\n\n"
        "The run id is derived from those inputs, so the same command gives the same folder "
        "and the same weights, bit for bit (spec 003). What pins it:\n\n"
        + table(
            ["Input", "Hash"],
            [
                [
                    "Training manifests",
                    " + ".join(f"`{m['sha256'][:12]}…`" for m in model.manifests),
                ],
                ["Backbone weights", f"`{bb['weights_sha256'][:12]}…`"],
                ["Feature cache key", f"`{bb['cache_key']}`"],
                ["Head recipe", f"`{meta['head_config']['sha256'][:12]}…`"],
                ["Abstention recipe", f"`{model.fit.meta['config']['sha256'][:12]}…`"],
            ],
        )
    )


# --- the card ------------------------------------------------------------------------------------


#: a count in words, for the prose the template writes about a model's own label space
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def classes_sentence(classes: list[str]) -> str:
    """ "*healthy* or *rust*", "*healthy*, *rust* or *anthracnose*": the classes THIS model
    answers with. The template cannot name them, because a model trained on a set without
    anthracnose does not have it (DECISIONS 130)."""
    marked = [f"*{c}*" for c in classes]
    if len(marked) == 1:
        return marked[0]
    return ", ".join(marked[:-1]) + " or " + marked[-1]


def rendered_from() -> str:
    """The repository the card was rendered from. A card can never carry the sha of the
    commit that will contain it, so the sha is provenance and never the thing that pins the
    model — the hashes in "Reproducing this model" are."""
    sha = git_sha() or "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return sha
    return f"{sha} + uncommitted changes" if dirty else sha


def render_card(model: Model, *, status: str, template: Path = TEMPLATE) -> str:
    text = Path(template).read_text(encoding="utf-8")
    values = {
        "model_version": model.model_version,
        "run_id": model.run_id,
        "classes_sentence": classes_sentence(model.meta["classes"]),
        "n_classes": _WORDS.get(len(model.meta["classes"]), str(len(model.meta["classes"]))),
        "status": status,
        "generated_at": _now(),
        "git_sha": rendered_from(),
        "model_block": model_block(model),
        "data_block": data_block(model),
        "evaluation_block": evaluation_block(model),
        "operating_point_block": operating_point_block(model),
        "failure_modes_block": failure_modes_block(model),
        "licences_block": licences_block(model),
        "reproduce_block": reproduce_block(model),
    }
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    left = [b for b in BLOCKS if "{{" + b + "}}" in text]
    if left:
        raise RegistryError("bad_template", f"{template} has placeholders nothing fills: {left}")
    return text


def card_path(model_version: str, cards: Path = CARDS) -> Path:
    return Path(cards) / f"{model_version}.md"


def volatile(text: str) -> str:
    """The card without the lines that change on every render: a card is current when the
    artefacts render the same thing, not when it was written at the same second."""
    return "\n".join(
        line for line in text.splitlines() if not line.startswith("**Status**: ")
    ).strip()


def content_sha256(card: Path) -> str:
    """The card's substance, hashed: what `volatile` leaves after the status and the
    timestamp are taken out."""
    return hashlib.sha256(volatile(card.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()


def card_is_current(model: Model, cards: Path, template: Path, status: str) -> tuple[bool, str]:
    path = card_path(model.model_version, cards)
    if not path.is_file():
        return False, f"there is no card at {path}"
    rendered = render_card(model, status=status, template=template)
    if volatile(path.read_text(encoding="utf-8")) != volatile(rendered):
        return False, f"{path} is not what the artefacts render now: write it again"
    return True, ""


# --- MLflow --------------------------------------------------------------------------------------


class Tracker:
    """One MLflow run per head run, in the experiment of its backbone (H8 §6.8). The store is
    --mlflow-uri, else MLFLOW_TRACKING_URI, else mlruns/mlflow.db (git-ignored) until piece 3
    runs the server, as the N4 probe does (DECISIONS 55)."""

    def __init__(self, uri: str | None) -> None:
        from mlflow.tracking import MlflowClient

        uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            LOCAL_MLFLOW.mkdir(parents=True, exist_ok=True)
            uri = "sqlite:///" + (LOCAL_MLFLOW / "mlflow.db").as_posix()
        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)

    def experiment_of(self, backbone_id: str) -> str:
        name = EXPERIMENT.format(backbone_id=backbone_id)
        exp = self.client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        artifacts = None
        if self.uri.startswith("sqlite:///"):
            artifacts = (Path(self.uri[len("sqlite:///") :]).parent / "artifacts").as_uri()
        return self.client.create_experiment(name, artifact_location=artifacts)

    def run_of(self, run_id: str, backbone_id: str) -> str | None:
        runs = self.client.search_runs(
            [self.experiment_of(backbone_id)],
            filter_string=f"tags.head_run_id = '{run_id}'",
            max_results=1,
        )
        return runs[0].info.run_id if runs else None

    def log(self, model: Model) -> str:
        from mlflow.entities import Metric, Param

        meta, bb, fit = model.meta, model.backbone, model.fit
        tags = {
            "head_run_id": model.run_id,
            "model_version": model.model_version,
            "backbone_id": bb["backbone_id"],
            "res": str(bb["res"]),
            "tokens": meta["tokens"],
            "head": meta["head"],
            "seed": str(meta["seed"]),
            "cache_key": bb["cache_key"],
            "quotable": str(meta["quotable"]).lower(),
            "research_only_until_c5": str(bb["research_only_until_c5"]).lower(),
            "n_table": "model_service/results/n_table.jsonl",
            "git_sha": git_sha() or "",
        }
        params = {
            "hf_id": bb["hf_id"],
            "revision": bb["revision"],
            "weights_sha256": bb["weights_sha256"],
            "preprocess_string": bb["preprocess_string"],
            "compute_dtype": bb["compute_dtype"],
            "train_manifest": meta["train"]["manifest"],
            "train_manifest_sha256": meta["train"]["manifest_sha256"],
            "train_split": meta["train"]["split"],
            "val_split": meta["val"]["split"],
            "n_train": meta["train"]["n"],
            "n_val": meta["val"]["n"],
            "classes": ",".join(meta["classes"]),
            "head_config_sha256": meta["head_config"]["sha256"],
            "abstain_config_sha256": fit.meta["config"]["sha256"],
            "k": fit.k,
            "coverages": ",".join(fit.coverages_str),
            "default_coverage": fit.meta["config"].get("default_coverage"),
        }
        metrics = {"temperature": fit.temperature, "epochs_run": meta["fit"]["epochs_run"]}
        for coverage in fit.coverages_str:
            suffix = _coverage_suffix(float(coverage))
            for name, value in fit.meta["thresholds"][coverage].items():
                metrics[f"{name}.{suffix}"] = value
        for row in model.rows:
            if row["seed"] is None:
                metrics[_metric_name(row)] = row["value"]
        run = self.client.create_run(
            self.experiment_of(bb["backbone_id"]), run_name=model.run_id, tags=tags
        )
        rid = run.info.run_id
        now = int(time.time() * 1000)
        self.client.log_batch(
            rid,
            metrics=[Metric(k, float(v), now, 0) for k, v in metrics.items() if v is not None],
            params=[Param(k, str(v)) for k, v in params.items()],
        )
        self.client.log_dict(rid, meta, "model/run.json")
        self.client.log_dict(rid, fit.meta, "model/abstain.json")
        self.client.log_artifact(rid, str(model.folder / "head.pt"), "model")
        self.client.set_terminated(rid)
        return rid

    def register(self, model: Model, mlflow_run_id: str, card: Path) -> Any:
        """A version of the one registered model (D-7's model set), tagged with the card it
        may not be registered without."""
        from mlflow.exceptions import MlflowException

        with contextlib.suppress(MlflowException):  # it exists already
            self.client.create_registered_model(
                REGISTERED_MODEL,
                description="Piece 4 model service v0: the models that answer the interface "
                "of H8 §6.2. A winning challenger joins; it never evicts the champion (D-7).",
            )
        return self.client.create_model_version(
            REGISTERED_MODEL,
            f"runs:/{mlflow_run_id}/model",
            mlflow_run_id,
            tags={
                "model_version": model.model_version,
                "head_run_id": model.run_id,
                "backbone_id": model.backbone["backbone_id"],
                "card": under_repo(card),
                "card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
                # the card without the lines that change on every render, so that a cosmetic
                # re-render is not drift and a number that moved is
                "card_content_sha256": content_sha256(card),
                "quotable": str(model.meta["quotable"]).lower(),
                "research_only_until_c5": str(model.backbone["research_only_until_c5"]).lower(),
            },
            description=f"{model.model_version} — see the card at {under_repo(card)}",
        )

    def version_of(self, model_version: str) -> Any | None:
        from mlflow.exceptions import MlflowException

        try:
            found = self.client.search_model_versions(
                f"name = '{REGISTERED_MODEL}' and tags.model_version = '{model_version}'"
            )
        except MlflowException:
            return None
        return found[0] if found else None


def _metric_name(row: dict[str, Any]) -> str:
    """An N-table row as one MLflow metric name: the number, the metric, what it was scored
    on and at which coverage, because a run has many of each."""
    target = stem(row["test_manifest"]) or "none"
    metric = row["metric"].replace(":", "_")
    name = f"{row['number']}.{metric}.{target}"
    if row["coverage"] != 1.0:
        name += f".{_coverage_suffix(row['coverage'])}"
    return _mlflow_name(name)


def _coverage_suffix(coverage: float) -> str:
    return "cov" + coverage_key(coverage).replace(".", "_")


def _mlflow_name(name: str) -> str:
    """MLflow allows alphanumerics, underscores, dashes, periods, spaces and slashes in a
    metric name, and nothing else."""
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in name)


# --- the command ---------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ms.registry", description=__doc__.split("\n")[0])
    p.add_argument("action", choices=("card", "log", "register"))
    p.add_argument("--run", action="append", default=None, help="a run id; repeatable")
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--cards", type=Path, default=CARDS)
    p.add_argument("--template", type=Path, default=TEMPLATE)
    p.add_argument("--backbone-configs", type=Path, default=BACKBONE_CONFIGS)
    p.add_argument("--mlflow-uri", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--force", action="store_true", help="write a card again, log a run again")
    return p.parse_args(argv)


def _models(args: argparse.Namespace, *, skip_unfitted: bool = False) -> list[Model]:
    folders = (
        [find_run(r, args.heads_root) for r in args.run] if args.run else list_runs(args.heads_root)
    )
    if not folders:
        raise RegistryError("missing_run", f"no run under {args.heads_root}")
    models, unfitted = [], 0
    for folder in folders:
        try:
            models.append(load_model(folder, args.cache_root, args.n_table, args.backbone_configs))
        except RegistryError as exc:
            # a run nobody has fitted has no operating point, so it has nothing to log and no
            # card to write; logging says how many it passed over, as make eval does
            if not (skip_unfitted and exc.reason == "missing_abstain"):
                raise
            unfitted += 1
    if unfitted:
        print(f"{unfitted} run(s) without an abstain.json were skipped: run make abstain first")
    if not models:
        raise RegistryError("missing_abstain", f"no fitted run under {args.heads_root}")
    return models


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except RegistryError as exc:
        print(exc, file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    if args.action == "card":
        models = _models(args)
        for model in models:
            path = card_path(model.model_version, args.cards)
            status = "registered" if _registered(args, model) else "not registered"
            text = render_card(model, status=status, template=args.template)
            current, why = card_is_current(model, args.cards, args.template, status)
            if current and not args.force:
                print(f"{path}: already what the artefacts render ({why or 'unchanged'})")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"{path}: written for {model.model_version}")
        return 0

    if args.action == "log":
        if args.no_mlflow:
            raise RegistryError("bad_value", "log needs MLflow; drop --no-mlflow")
        tracker = Tracker(args.mlflow_uri)
        logged = 0
        for model in _models(args, skip_unfitted=True):
            if tracker.run_of(model.run_id, model.backbone["backbone_id"]) and not args.force:
                continue
            tracker.log(model)
            logged += 1
        print(f"{logged} run(s) logged to {tracker.uri}")
        return 0

    models = _models(args)
    if len(models) != 1:
        raise RegistryError(
            "bad_value", f"register takes one --run, not {len(models)}: a card is one model's"
        )
    model = models[0]
    if not model.meta["quotable"]:
        raise RegistryError(
            "not_quotable",
            f"{model.run_id} trained on a test-only manifest: its numbers are never reported, "
            "so it is not a model to register",
        )
    current, why = card_is_current(model, args.cards, args.template, "registered")
    if not current:
        raise RegistryError(
            "missing_card",
            f"{model.model_version}: {why}. H8 §6.8 registers a model only with a card "
            "(python -m ms.registry card --run " + model.run_id + ")",
        )
    if args.no_mlflow:
        raise RegistryError("bad_value", "register needs MLflow; drop --no-mlflow")
    tracker = Tracker(args.mlflow_uri)
    existing = tracker.version_of(model.model_version)
    if existing is not None and not args.force:
        print(
            f"{model.model_version} is already version {existing.version} of "
            f"{REGISTERED_MODEL}; nothing to do"
        )
        # the version was tagged with the card as it stood then. If the card has moved since —
        # a number landed, the table grew — say so: the registry would otherwise point at a
        # card that no longer reads the way it did when the model was registered.
        card = card_path(model.model_version, args.cards)
        digest = content_sha256(card) if card.is_file() else None
        if digest != existing.tags.get("card_content_sha256"):
            print(
                f"  note: {under_repo(card)} has changed since version {existing.version} was "
                "registered against it. Register again with --force to record the card it "
                "reads now."
            )
        return 0
    mlflow_run_id = tracker.run_of(model.run_id, model.backbone["backbone_id"]) or tracker.log(
        model
    )
    # the card said "not registered" a moment ago; it does not any more. It is written before
    # the version is created, so that `card_sha256` is the hash of the card as it now stands
    card = card_path(model.model_version, args.cards)
    card.write_text(
        render_card(model, status="registered", template=args.template),
        encoding="utf-8",
        newline="\n",
    )
    version = tracker.register(model, mlflow_run_id, card)
    print(
        f"{model.model_version} registered as {REGISTERED_MODEL} version {version.version} "
        f"(MLflow run {mlflow_run_id}) in {tracker.uri}"
    )
    return 0


def _registered(args: argparse.Namespace, model: Model) -> bool:
    if args.no_mlflow:
        return False
    try:
        return Tracker(args.mlflow_uri).version_of(model.model_version) is not None
    except Exception:  # noqa: BLE001 - no store yet is simply "not registered"
        return False


if __name__ == "__main__":
    sys.exit(main())
