"""Abstention and calibration (S4.4; specs/004-abstention/spec.md).

Four post-hoc scores over a trained head (spec 003) and its cached features:

- `conf`   the head's largest logit                       higher = more confident
- `knn`    distance to the k-th nearest training feature  higher = stranger
- `maha`   the relative Mahalanobis distance              higher = stranger
- `energy` -logsumexp of the logits, logged only          higher = stranger

A row abstains when `conf < tau_conf` or `knn > tau_knn`, and the reason names which
(`low_confidence` first when both cross). `maha` is the second opinion: fitted, recorded,
reported, never consulted. Every threshold, and the temperature `T` of the calibrated
probabilities, is fitted on the **in-domain validation slice** and never on a test manifest
(US-2): a threshold set on the rows it is then measured on would make every N3 number a
self-report. One fit is `data/heads/<run_id>/abstain.json`, written by `ms.abstain.fit`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors

from ms.cache import CACHE_ROOT, find_cache, load_cache
from ms.data.manifests import REPO_ROOT, load_manifest
from ms.heads import Run, features, resolve_path

#: the four scores (FR-013), and the two that decide (US-4.1)
SCORES = ("conf", "knn", "maha", "energy")
DECISION_SCORES = ("conf", "knn")
#: US-4.2, in the order H8 §6.2 lists them: the first one that crossed names the abstention
REASONS = ("low_confidence", "far_from_training")
#: whether a larger value means a stranger row (US-1.2); every AUROC uses this orientation
HIGHER_IS_STRANGER = {"conf": False, "knn": True, "maha": True, "energy": True}
#: the declared coverages (US-3.1); the config may name others
COVERAGES = (0.80, 0.90, 0.95)
DEFAULT_COVERAGE = 0.90
ABSTAIN_JSON_VERSION = 1
#: an input of inputs_sha256 (FR-003): a change here is a new fit
FITTER_VERSION = 1
ABSTAIN_CONFIG = REPO_ROOT / "model_service" / "configs" / "abstain.yaml"
CONFIG_KEYS = (
    "k",
    "coverages",
    "default_coverage",
    "distance_tpr",
    "ece_bins",
    "epsilon",
    "n3",
)
#: H8 §6.6 sets the distance threshold once, "at TPR 95 % on in-domain validation"
DISTANCE_TPR = 0.95
#: US-3.3: a score whose range over the slice is this small has no threshold to give
FLAT = 1e-9


class AbstainError(ValueError):
    """A refused fit or an unusable file; `reason` is FR-011's."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# --- the config (FR-012) -------------------------------------------------------------------------


def load_abstain_config(path: Path | str = ABSTAIN_CONFIG) -> dict[str, Any]:
    """configs/abstain.yaml, checked; AbstainError('bad_config') on anything malformed."""
    path = Path(path)
    if not path.exists():
        raise AbstainError("bad_config", f"{path} does not exist")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AbstainError("bad_config", f"{path}: {exc}") from None
    if not isinstance(cfg, dict) or set(cfg) != set(CONFIG_KEYS):
        raise AbstainError("bad_config", f"{path}: the keys must be {', '.join(CONFIG_KEYS)}")
    k, bins = cfg["k"], cfg["ece_bins"]
    if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1):
        raise AbstainError("bad_config", f"{path}: k must be an integer >= 1, got {k!r}")
    if not (isinstance(bins, int) and not isinstance(bins, bool) and bins >= 2):
        raise AbstainError("bad_config", f"{path}: ece_bins must be an integer >= 2, got {bins!r}")
    coverages = cfg["coverages"]
    if not (isinstance(coverages, list) and coverages):
        raise AbstainError("bad_config", f"{path}: coverages must be a non-empty list")
    for c in coverages:
        if not (isinstance(c, int | float) and not isinstance(c, bool) and 0 < c < 1):
            raise AbstainError("bad_config", f"{path}: a coverage must be in (0, 1), got {c!r}")
    tpr = cfg["distance_tpr"]
    if not (isinstance(tpr, int | float) and not isinstance(tpr, bool) and 0 < tpr < 1):
        raise AbstainError("bad_config", f"{path}: distance_tpr must be in (0, 1), got {tpr!r}")
    if cfg["default_coverage"] not in coverages:
        raise AbstainError(
            "bad_config", f"{path}: default_coverage {cfg['default_coverage']!r} is not declared"
        )
    eps = cfg["epsilon"]
    if not (isinstance(eps, int | float) and not isinstance(eps, bool) and eps > 0):
        raise AbstainError("bad_config", f"{path}: epsilon must be > 0, got {eps!r}")
    if not isinstance(cfg["n3"], list):
        raise AbstainError("bad_config", f"{path}: n3 must be a list of held-out sources")
    for source in cfg["n3"]:
        ok = (
            isinstance(source, dict)
            and isinstance(source.get("manifest"), str)
            and isinstance(source.get("classes"), list)
            and source["classes"]
            and all(isinstance(c, str) for c in source["classes"])
        )
        if not ok:
            raise AbstainError("bad_config", f"{path}: a malformed n3 source {source!r}")
    return cfg


def coverage_key(coverage: float) -> str:
    """How a coverage keys `thresholds` in abstain.json: '0.80', '0.90', '0.95'."""
    return f"{float(coverage):.2f}"


# --- the scores (FR-004) -------------------------------------------------------------------------


def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1, so that a distance does not depend on how many token types
    were joined (US-1.3)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def logsumexp(z: np.ndarray) -> np.ndarray:
    top = z.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(z - top).sum(axis=1, keepdims=True))).ravel()


def knn_distance(bank: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """The distance from each row of `x` to its k-th nearest neighbour in `bank`; both are
    L2-normalised first (H8 §6.6, `[G93]`)."""
    if not 1 <= k <= len(bank):
        raise AbstainError("bad_value", f"k = {k} needs a bank of at least k rows ({len(bank)})")
    index = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    index.fit(unit(bank))
    distances, _ = index.kneighbors(unit(x), n_neighbors=k, return_distance=True)
    return distances[:, -1].astype(np.float64)


@dataclass
class Gaussians:
    """The class-conditional Gaussians with one shared covariance, and the background one the
    relative Mahalanobis distance subtracts (`[G83]`, and Clarification 12)."""

    means: np.ndarray
    precision: np.ndarray
    background_mean: np.ndarray
    background_precision: np.ndarray

    def distance(self, x: np.ndarray) -> np.ndarray:
        z = unit(x).astype(np.float64)
        background = _quadratic(z, self.background_mean, self.background_precision)
        per_class = np.stack([_quadratic(z, mu, self.precision) for mu in self.means])
        return per_class.min(axis=0) - background


def _quadratic(z: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    d = z - mean
    return np.einsum("nd,de,ne->n", d, precision, d)


def _ridged(scatter: np.ndarray, denominator: int, epsilon: float) -> np.ndarray:
    cov = scatter / max(denominator, 1)
    trace = float(np.trace(cov))
    if not (trace > 0):
        raise AbstainError(
            "degenerate_score", "the training features have no spread: no Gaussian to fit"
        )
    return cov + epsilon * trace / cov.shape[0] * np.eye(cov.shape[0])


def gaussians(bank: np.ndarray, y: np.ndarray, n_classes: int, epsilon: float) -> Gaussians:
    """Fitted on the training features alone (US-2.1). The shared covariance is the pooled
    within-class scatter over N - C; the background one is the whole bank's over N - 1."""
    z = unit(bank).astype(np.float64)
    means, centred = [], []
    for c in range(n_classes):
        rows = z[y == c]
        if not len(rows):
            raise AbstainError("bad_value", f"class {c} has no training row to fit a Gaussian on")
        means.append(rows.mean(axis=0))
        centred.append(rows - means[-1])
    spread = np.concatenate(centred)
    shared = _ridged(spread.T @ spread, len(z) - n_classes, epsilon)
    background_mean = z.mean(axis=0)
    around = z - background_mean
    background = _ridged(around.T @ around, len(z) - 1, epsilon)
    return Gaussians(
        np.stack(means), np.linalg.inv(shared), background_mean, np.linalg.inv(background)
    )


# --- a fitted run (FR-002, FR-013) ------------------------------------------------------------


@dataclass
class Fit:
    """A loaded abstain.json, with the feature-space objects rebuilt from the cache its
    provenance names. Nothing in it is fitted on the fly."""

    folder: Path
    meta: dict[str, Any]
    bank: np.ndarray | None = field(default=None, repr=False)
    gaussians: Gaussians | None = field(default=None, repr=False)

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def temperature(self) -> float:
        return float(self.meta["temperature"]["T"])

    @property
    def k(self) -> int:
        return int(self.meta["scores"]["knn"]["k"])

    @property
    def coverages(self) -> tuple[float, ...]:
        return tuple(float(c) for c in sorted(self.meta["thresholds"]))

    def thresholds(self, coverage: float) -> dict[str, float]:
        key = coverage_key(coverage)
        if key not in self.meta["thresholds"]:
            raise AbstainError(
                "bad_value", f"coverage {key} was not fitted (have {', '.join(self.coverages_str)})"
            )
        return self.meta["thresholds"][key]

    @property
    def coverages_str(self) -> list[str]:
        return sorted(self.meta["thresholds"])

    def rebuild(self, cache_root: Path | str | None = None) -> Fit:
        """Read the training features again — the bank of the kNN score and the Gaussians —
        from the manifests and caches `fitted_on` and `cache` name. purpose = train (US-2.1)."""
        if self.bank is not None:
            return self
        on, cache = self.meta["fitted_on"], self.meta["cache"]
        npzs = cache["npz"].split("+")
        xs, ys = [], []
        for path, recorded in zip(on["manifest"].split("+"), npzs, strict=True):
            m = load_manifest(resolve_path(path), on["train_split"], "train")
            npz = resolve_path(recorded)
            if cache_root is not None or not npz.exists():
                npz = find_cache(
                    Path(cache_root or CACHE_ROOT),
                    cache["backbone_id"],
                    cache["res"],
                    m.name,
                    m.sha256,
                    key=cache["cache_key"],
                )
            loaded = load_cache(npz, m.sha256)
            where = {str(i): k for k, i in enumerate(loaded.image_id)}
            rows = [r for r in m.rows if r["class_km2"] in self.meta["classes"]]
            index = np.array([where[r["image_id"]] for r in rows])
            xs.append(features(loaded, index, self.meta["tokens"]))
            ys.append(
                np.array([self.meta["classes"].index(r["class_km2"]) for r in rows], dtype=np.int64)
            )
        self.bank = np.concatenate(xs)
        self.gaussians = gaussians(
            self.bank,
            np.concatenate(ys),
            len(self.meta["classes"]),
            float(self.meta["scores"]["maha"]["epsilon"]),
        )
        return self


def load_fit(folder: Path | str, cache_root: Path | str | None = None, *, rebuild: bool = True):
    """The fit beside a run. Its inputs_sha256 has to match what the file records, because a
    fit is written by code and never by hand (FR-013)."""
    folder = Path(folder)
    path = folder / "abstain.json"
    if not path.exists():
        raise AbstainError("missing_fit", f"{path} does not exist: run `make abstain` first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("inputs_sha256") != inputs_sha256(meta):
        raise AbstainError("bad_value", f"{path}: inputs_sha256 does not match its own inputs")
    fit = Fit(folder, meta)
    return fit.rebuild(cache_root) if rebuild else fit


def inputs_sha256(meta: dict[str, Any]) -> str:
    """FR-003: what a fit is a function of. The run id already carries the seed and head."""
    on, cache = meta["fitted_on"], meta["cache"]
    inputs = {
        "fitter_version": FITTER_VERSION,
        "run_id": meta["run_id"],
        "cache_key": cache["cache_key"],
        "tokens": meta["tokens"],
        "config_sha256": meta["config"]["sha256"],
        "train": [on["manifest_sha256"], on["train_split"]],
        "val": [on["manifest_sha256"], on["val_split"]],
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()


# --- scoring, deciding, calibrating (US-1, US-4, US-5) ------------------------------------------


def logit_scores(run: Run, x: np.ndarray) -> dict[str, np.ndarray]:
    """The two scores that come from the head's logits alone: they cost nothing, and they are
    the only two that change from one run of a group to the next."""
    z = run.logits(x).astype(np.float64)
    return {"conf": z.max(axis=1), "energy": -logsumexp(z)}


def distance_scores(fit: Fit, x: np.ndarray) -> dict[str, np.ndarray]:
    """The two scores that come from the training features: the same for every run whose bank
    is the same, which is every head and seed of one (backbone, resolution, tokens,
    manifests). `bank_key` is what makes two fits share them."""
    if fit.bank is None or fit.gaussians is None:
        raise AbstainError("bad_value", f"{fit.run_id}: the fit was loaded without its bank")
    return {"knn": knn_distance(fit.bank, x, fit.k), "maha": fit.gaussians.distance(x)}


def bank_key(fit: Fit) -> tuple:
    """What two fits share when their bank and their Gaussians are the same features."""
    on, cache = fit.meta["fitted_on"], fit.meta["cache"]
    return (
        on["manifest_sha256"],
        on["train_split"],
        cache["cache_key"],
        fit.meta["tokens"],
        fit.k,
        float(fit.meta["scores"]["maha"]["epsilon"]),
        tuple(fit.meta["classes"]),
    )


def scores(
    fit: Fit, run: Run, x: np.ndarray, distances: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """The four scores of US-1.1, one finite value per row of `x`. `distances` are the two
    feature-space scores when a caller has them already."""
    return {**logit_scores(run, x), **(distances or distance_scores(fit, x))}


def decide(fit: Fit, row: dict[str, float], coverage: float) -> tuple[str, str | None]:
    """predict or abstain, and the reason (US-4). Either decision score crossing abstains;
    the comparisons are strict, so a row exactly at a threshold is predicted."""
    tau = fit.thresholds(coverage)
    if float(row["conf"]) < tau["tau_conf"]:
        return "abstain", REASONS[0]
    if float(row["knn"]) > tau["tau_knn"]:
        return "abstain", REASONS[1]
    return "predict", None


def decisions(fit: Fit, row_scores: dict[str, np.ndarray], coverage: float) -> np.ndarray:
    """A boolean per row: true where it abstains. The vectorised form of `decide`."""
    tau = fit.thresholds(coverage)
    return (np.asarray(row_scores["conf"]) < tau["tau_conf"]) | (
        np.asarray(row_scores["knn"]) > tau["tau_knn"]
    )


def probabilities(fit: Fit, run: Run, x: np.ndarray) -> np.ndarray:
    """softmax(logits / T) over the known classes only (US-5.2). T > 0 leaves every argmax
    where it was, so no N1 or N2 number moves."""
    return softmax(run.logits(x).astype(np.float64) / fit.temperature)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """The mean negative log-likelihood of the true classes at one temperature."""
    scaled = z / temperature
    return float((logsumexp(scaled) - scaled[np.arange(len(y)), y]).mean())


def fit_temperature(z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """FR-007: the T that minimises the slice's mean NLL, by a golden-section search over
    log T in [log 0.01, log 100] to a tolerance of 1e-6. No random start, so it repeats."""
    lo, hi = math.log(0.01), math.log(100.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(z, y, math.exp(c)), nll(z, y, math.exp(d))
    iterations = 0
    while b - a > 1e-6:
        iterations += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(z, y, math.exp(d))
    temperature = math.exp((a + b) / 2)
    before, after = nll(z, y, 1.0), nll(z, y, temperature)
    if after > before + 1e-9:
        raise AbstainError(
            "degenerate_temperature",
            f"the fitted T = {temperature:.4f} scores worse on the slice than T = 1 "
            f"({after:.6f} > {before:.6f})",
        )
    return {
        "T": round(temperature, 6),
        "nll_before": round(before, 6),
        "nll_after": round(after, 6),
        "iterations": iterations,
    }


# --- the two statistics the table reports (US-6.2, US-9.1) --------------------------------------


def auroc(strange_known: np.ndarray, strange_unknown: np.ndarray) -> float:
    """The Mann-Whitney statistic with the unknowns as the positive class, on the strangeness
    orientation of US-1.2, with average ranks for ties."""
    known = np.asarray(strange_known, dtype=np.float64)
    unknown = np.asarray(strange_unknown, dtype=np.float64)
    if not len(known) or not len(unknown):
        raise AbstainError("bad_value", "an AUROC needs both a known and an unknown side")
    both = np.concatenate([known, unknown])
    order = np.argsort(both, kind="stable")
    ranks = np.empty(len(both), dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1, dtype=np.float64)
    values, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.zeros(len(values), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n1, n0 = len(unknown), len(known)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strangeness(score: str, values: np.ndarray) -> np.ndarray:
    """A score turned so that a larger value means a stranger row (US-1.2)."""
    values = np.asarray(values, dtype=np.float64)
    return values if HIGHER_IS_STRANGER[score] else -values


def ece(proba: np.ndarray, y: np.ndarray, bins: int) -> float:
    """US-9.1: equal-width bins over the largest probability, the weighted mean over the bins
    of |accuracy - mean confidence|."""
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y)
    if not len(y):
        raise AbstainError("bad_value", "an ECE needs at least one row")
    confidence, predicted = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        take = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        if take.any():
            accuracy = float((predicted[take] == y[take]).mean())
            total += float(take.mean()) * abs(accuracy - float(confidence[take].mean()))
    return total
