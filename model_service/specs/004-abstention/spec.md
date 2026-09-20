# Feature specification: 004 — Abstention and calibration

**Branch**: `004-abstention` · **Created**: 2026-09-20 · **Status**: Specified 2026-09-20 with failing tests in `model_service/tests/test_abstain.py` (DECISIONS 86–93); US-3.1 amended the same day, before any row was written, to put the distance threshold at TPR 95 % as H8 §6.6 says (DECISIONS 94). `ms.abstain` implemented 2026-09-20 with the W4 "Scores" task; the rows of US-6 to US-9 come with the "N3" task
**Input**: H8 §6.6 (abstention and calibration, copied verbatim in US-1), §6.2 (the response's `decision`, `abstain_reason` and `uncertainty` fields), §6.7 (N1's abstention rate at the declared coverage; N3), `docs/piece4-work-plan.md` S4.4 and the W4 tasks, H9 §7 criterion 2 with the owner's readings (DECISIONS 79), specs 001 (roles, purposes, the held-out unknowns), 002 (caches), 003 (runs, features) and 005 (rows, aggregates, metric names). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

A head always answers: its softmax puts 1.0 across the three trained classes whatever it is shown, a white-mould crop included. Abstention is what turns that into a usable answer, and H8 D-10 asks for it to be built first, not added later.

Two things have to be true for an abstention number to mean anything. It must be **post-hoc** — computed from what the head already has, so that no number in the table comes from a different model than the one that produced N1 and N2. And it must be **fitted on the in-domain validation slice only** (spec 003 US-3): a threshold set on the rows it is then measured on turns every N3 number into a self-report, and the exam was frozen before the first head trained (spec 001 US-5) precisely so that cannot happen.

The names are fixed here, before any N3 number exists, because the verdict reads them. H9 §7 criterion 2 counts an AUROC advantage on the decision scores, and the owner's reading (DECISIONS 79) is "both decision scores, confidence and kNN distance". Those two rows have to exist under those two names for the verdict to be computable by code.

## Commands

`python -m ms.abstain.fit` (`make abstain`) fits one run's thresholds and temperature; `python -m ms.eval.run` (`make eval`) then writes the rows at the declared coverages and the N3 rows (spec 005 FR-009 keeps the writers).

    python -m ms.abstain.fit [--run <run_id> ...] [--heads-root data/heads]
        [--cache-root data/cache] [--config model_service/configs/abstain.yaml]
        [--compute-log model_service/results/compute_log.jsonl] [--force]

- `--run` defaults to every run under `--heads-root`. One run is one fit, and a run whose `abstain.json` already records the same inputs is left alone; `--force` refits it.
- `python -m ms.eval.run` gains `--abstain-config <file>` (the same default), and writes the rows of US-6 to US-9 for every run that has an `abstain.json`. Its other flags are spec 005's, unchanged.
- Exit codes: 0 when every run is fitted or already there. 2 on an input error: every input is checked before the first fit (FR-011), so an input error writes no `abstain.json` and no compute-log row.

## User scenarios and testing

### US-1: Three scores, and a fourth that is only logged (priority P1)

H8 §6.6, the design this spec implements, verbatim:

> ### 6.6 Abstention and calibration
>
> Built first, not added later (D-10). Two scores, both post-hoc, both computed from what the head already has:
>
> 1. **Confidence score** — max logit / max prototype cosine (`[H47]`: a good closed-set classifier is most of open-set recognition; `[H48]`: feature-magnitude-sensitive scores such as max-logit hold up at scale). Threshold τ_conf set on the in-domain validation slice for a **declared coverage** (default 90 %; also report 80 % and 95 %) — the selective-classification framing of `[G71]`.
> 2. **Feature-space distance** — k-nearest-neighbour distance to the training features (k = 10–50, L2-normalised; `[G93]`: no distributional assumption, 24.77 % lower FPR@TPR95 than the Mahalanobis baseline SSD+ on ImageNet) with the **relative Mahalanobis** distance `[G83]` as the second opinion; threshold at TPR 95 % on in-domain validation. Energy scores `[G84]` are logged but not used for the decision in v0.
>
> Decision: `abstain` if *either* score crosses its threshold; `abstain_reason` names which. **Calibration:** temperature scaling `[G44]` fitted on the in-domain validation slice; expected calibration error reported in-domain *and* out-of-domain, because calibration is what breaks first under shift `[G49][C50]`. **Conformal sets** (RAPS, `[G74]`; `[G47]`): optional in v0 — implement only if W4 has slack; on a 3-class label space they add little beyond the coverage guarantee, which matters later (D-1) rather than now.
>
> What abstention is measured on. *Unknown recall*: the fraction of held-out ALS and white-mould images that receive `abstain` at the operating point; *AUROC* of each score for known-vs-unknown; *selective risk vs coverage* curves in- and out-of-domain (`[C50]` reports 0.520 selective risk at 80 % coverage as the best result under shift — the bracket to beat, not to assume). And the caution of `[G73]` applies: check that abstention does not fall disproportionately on the rarest class.

1. **Given** a run (spec 003 FR-005) and features `x` as `ms.heads.features` returns them, **then** `ms.abstain.scores(fit, run, x)` returns one finite value per row for each of four scores:
   - `conf` — the head's largest logit. For `linear` that is the largest weighted sum; for `proto` it is the largest prototype cosine over τ, which is H8's "max prototype cosine" up to the positive factor 1/τ; for `mix` the logits are log p, so it is the log of the largest mixed probability (spec 003 US-2). **Higher means more confident.**
   - `knn` — the Euclidean distance from the L2-normalised feature to its k-th nearest neighbour among the L2-normalised **training** features of the same run (FR-004). **Higher means stranger.**
   - `maha` — the relative Mahalanobis distance of `[G83]`: the smallest, over the trained classes, of the class Mahalanobis distance minus the distance to a single background Gaussian fitted on all the training features (FR-004). **Higher means stranger.**
   - `energy` — −logsumexp over the head's logits, at temperature 1 (`[G84]`). **Higher means stranger.** For `mix` the logits are already normalised log-probabilities (spec 003 US-2.3), so their logsumexp is log 1 and `energy` is identically zero, up to float32 rounding: `[G84]`'s score is defined on unnormalised logits, and `mix` has none. The formula stays the same for all three heads; what follows from it is US-3.3 and US-6.1.
2. **Given** any of the four, **then** its **strangeness orientation** is fixed once and for all: `−conf`, `+knn`, `+maha`, `+energy`. Every AUROC (US-6) is computed on the strangeness orientation, so that a score which separates unknowns perfectly scores 1.0 whichever way it points.
3. **Given** `token_type = cls+meanpatch`, **then** `knn` and `maha` L2-normalise the joined vector again before measuring, so that a score does not depend on how many token types were joined. The head reads the joined vector as spec 003 FR-007 gives it, unchanged.
4. **energy is logged, never decided on.** "Logged" is the N-table (an `auroc:energy` row) and `abstain.json`, not the response: H8 §6.2's `uncertainty` block has no energy field, and nothing is added to it here. No threshold of energy ever changes a decision in v0. `maha` is the second opinion (US-4.3): fitted, recorded, reported, and never consulted by `decide`.

### US-2: Everything is fitted on the in-domain validation slice (P1)

1. **Given** a run, **when** `ms.abstain.fit` fits it, **then** it reads exactly two sets of rows through spec 001's loader:
   - the run's **training** rows — every training manifest's `--split` rows as `run.json` records them — with `purpose = train`, for the kNN bank and the Gaussians;
   - the run's **validation** rows — every training manifest's own `--val-split` rows — with `purpose = select`, for the thresholds, the temperature and nothing else.
2. **Given** any attempt to fit on a `test` or `holdout_unknown` split, **then** the loader raises (spec 001 FR-011) and the fit exits 2 with `test_split_access`, writing nothing. This holds for a `--run` whose `run.json` records such a split as its own.
3. **Given** a run trained on several manifests (spec 003 US-5.1), **then** the bank is their training rows joined in the order `run.json` names them, and the slice is their validation rows joined the same way. The manifests, their sha256s and the row counts go into `abstain.json`.
4. **Given** a target of N2 or N3, **then** no threshold, temperature or Gaussian is ever refitted on it. The operating point that scores a cross-dataset target is the one fitted in domain, which is the whole point of the number.

### US-3: A declared coverage (P1)

1. **Given** the validation slice's scores and a declared coverage `c` ∈ {0.80, 0.90, 0.95} (FR-012 names them; 0.90 is the default operating point), **then** the thresholds are empirical quantiles of the slice, each keeping a share of it by **that score alone**:
   - `tau_conf` is the (1 − c) quantile of `conf`, so the confidence gate keeps `c` of the slice; a row abstains when `conf < tau_conf`;
   - `tau_knn` is the `distance_tpr` quantile of `knn` — 0.95, H8 §6.6's "threshold at TPR 95 % on in-domain validation" — and a row abstains when `knn > tau_knn`. It is **the same value at every declared coverage**;
   - `tau_maha` is the `distance_tpr` quantile of `maha`, recorded and reported, never part of the decision (US-4.3).

   Quantiles are linear-interpolated (`numpy.quantile`'s default), so a threshold need not be one of the observed values. The coverage knob therefore moves how much confidence the head must have, and leaves the "is this photo unlike anything I trained on" gate where H8 puts it.
2. **Given** the three thresholds of one coverage, **then** `abstain.json` also records `coverage_achieved`: the share of the validation slice that the **joint** rule of US-4 keeps. It is at most `c`, because two rejection sets are united — about `0.95 · c` when the two scores disagree freely — and it is the honest number to read beside the declared one. Measured on the first real fit (DECISIONS 94): 0.76, 0.86 and 0.90 at the three declared coverages.
3. **Given** a slice with fewer than ⌈1 / (1 − c)⌉ rows for a declared coverage — 20 rows at 0.95 — or one of the three **thresholded** scores (`conf`, `knn`, `maha`) constant over the slice, meaning its range is at most 1e-9, **then** the fit exits 2 with `too_few_rows` or `degenerate_score`, and nothing is written: a threshold from four rows, or from one value, is not a coverage. `energy` carries no threshold, so its spread never refuses a fit — which matters, because `mix`'s energy is constant by construction (US-1.1).
4. **Given** the N-table, **then** a row's `coverage` field is the **declared** coverage, and what actually happened on the scored set is its own row, `abstention_rate` (US-8). A declared coverage is a promise about the in-domain validation slice, one score at a time. It is not a guarantee about any test set, and the two differ by exactly the amount the shift costs.

### US-4: The decision (P1)

1. **Given** a row's scores and a coverage, **then** `ms.abstain.decide(fit, scores, coverage)` returns `predict` or `abstain`, and abstains when **either** decision score crosses: `conf < tau_conf` or `knn > tau_knn`.
2. **Given** an abstention, **then** `abstain_reason` is `low_confidence` when `conf` crossed and `far_from_training` when `knn` did. When both cross, the reason is `low_confidence`, the first of H8 §6.2's list. The second crossing is not lost: the response's `uncertainty` block carries `ood_knn` beside `ood_knn_threshold` (H8 §6.2; FR-015), so the distance crossing the reason does not name is readable from the response itself, and `tau_conf` is in `abstain.json` beside it.
3. **`maha` and `energy` never decide.** They are recorded beside the decision. H8 calls the relative Mahalanobis a second opinion and puts energy under "logged but not used", and H8 §6.2's response has an `ood_knn_threshold` field and no Mahalanobis threshold beside `ood_maha`.
4. `invalid_input` is the third `abstain_reason` of H8 §6.2. It belongs to the service (006): a frame that fails the contract never reaches a score. Nothing here produces it.
5. **Given** the held-out unknowns at the default operating point, **then** `abstain` is reachable: SC-4 asks for a non-zero unknown recall on both held-out sets before S4.4 counts as green.

### US-5: Temperature scaling, which changes no decision (P1)

1. **Given** the validation slice, **then** `T > 0` is the temperature that minimises the negative log-likelihood of the slice's true classes under `softmax(logits / T)`, found by a scalar minimisation over `log T` (FR-007), and recorded in `abstain.json`.
2. **Given** `T`, **then** `ms.abstain.probabilities(fit, run, x)` returns `softmax(logits / T)`: the scores of the service's response, over the known classes only. "Unknown" stays a decision, never a class with a probability (H8 §6.2).
3. **Given** any `T > 0`, **then** the argmax over a fixed set of classes is unchanged, so **every N1 and N2 number at coverage 1.0 is the same number after S4.4 as before it**. A test checks this on a fitted run, because a calibration step that quietly moved the headline numbers would invalidate W3's table.
4. `Run.predict_proba` (spec 003 FR-005) stays uncalibrated and is what `ms.eval.run` uses for the decision. Temperature enters the table only through `ece` rows and the service.

### US-6: N3 — unknown recall and AUROC (P1)

1. **Given** a fitted run and the held-out sources of `configs/abstain.yaml` (FR-012), **when** `make eval` runs, **then** it appends, for each held-out set separately — angular leaf spot and white mould, never pooled (DECISIONS 79) — rows with `number = "N3"`, `test_split = "holdout_unknown"` and `classes` the held-out class alone (spec 005 US-4.2):
   - `unknown_recall`, the share of that set's rows that get `abstain`, one row per declared coverage;
   - `auroc:conf`, `auroc:knn`, `auroc:maha` and `auroc:energy`, at `coverage = 1.0`, because an AUROC uses no threshold. The verdict reads quotable aggregates at coverage 1.0 (spec 005 US-6.1), so the two decision scores' rows are the ones H9 §7 criterion 2 needs.

   No `auroc:energy` row is written for a `mix` run: its energy is zero on every row (US-1.1), so the statistic would be an ordering of float32 rounding, and a five-seed aggregate of it would put 0.5 in the table as if it had been measured. `constant_energy: <run_id>` is printed instead, as US-6.5 prints for a source with no rows.
2. **Given** an `auroc:<score>` row, **then** the unknowns are the positive class, the knowns are the run's own in-domain test rows, and the value is the Mann–Whitney statistic over the strangeness orientation of US-1.2, with average ranks for ties. `n` is the number of rows it was computed over, knowns and unknowns together, and `notes` names both sides with their counts.
   - The knowns are **the test split of every training manifest of the run**, read with `purpose = evaluate`, in the order `run.json` names them, and rows of the held-out classes are not among them. For a run trained on one manifest they are exactly the rows N1 scores, so the two numbers stand on one population, and `test_manifest` names two manifests. For the run trained on Makerere and iBean they are Makerere's test split and, iBean being test-only, every iBean row (spec 001 FR-008), so `test_manifest` names three. The joined form is spec 005's, and the count of hashes always matches the count of manifests.
   - `test_manifest` and `test_manifest_sha256` join the knowns' manifest and the unknowns' manifest with `+`, in that order (spec 005's "several manifests in one row"). An `unknown_recall` row names the unknowns' manifest alone, because no known row enters it.
3. **Given** a held-out source, **then** it is a manifest and a class, and v0 uses `makerere_v1` for `unknown_als` (5,098 rows) and `swm_v1` for `unknown_wm`. Angular leaf spot is held out in three frozen manifests, and the other two are left out of v0: iBean's rows because iBean is a training source in one N2 direction (spec 003 US-5.1), and `makerere_crops_v1`'s because the crops carry no healthy class, so their known side is not the population the other numbers are read against (DECISIONS 46, and the crop-level N2 rows are reported and not counted for the same reason, DECISIONS 79). One source per held-out set keeps the verdict's "each held-out set" a single pair of numbers. Adding a line to `configs/abstain.yaml` adds rows; it never changes the ones already written.
4. **Whether the unknowns are also a new dataset depends on the run, and the row says so.** H8 §6.7 reports open-set numbers as cross-domain, and for a Tanzania-trained run the ALS rows of `makerere_v1` are both a new disease and a new dataset. For a run that trained on Makerere they are a new disease in a dataset it knows. Both are worth having and neither may be read as the other, so `notes` name the unknowns' dataset and state which of the two cases the row is. The source is the same for both backbones, so the verdict compares like with like either way.
5. **Given** a source with no rows, or a run with no in-domain test rows, **then** that row is not written and the reason is printed; the other rows are written as usual.
6. **Given** runs at another resolution or with another token type (the W4 ablations), **then** they get N3 rows too, because a fitted run is scored wherever it exists. The verdict reads 224 and CLS (spec 005 US-6.1); the other rows are the ablation's.

### US-7: Selective risk against coverage, in and out of domain (P1)

1. **Given** a fitted run, **then** `make eval` appends, for every (run, target) pair it already scores — N1 on the run's own test split, N2 on each direction of `configs/eval.yaml` — the same metrics it writes at coverage 1.0 (`macro_f1`, `recall:<class>`) again at each declared coverage, over the **accepted** rows only, plus:
   - `selective_risk`, the share of accepted rows whose decision is wrong;
   - `abstention_rate`, the share of scored rows that abstained.

   `n` on these rows counts the **scored** rows, never the accepted ones: the target's scored rows for `macro_f1`, `selective_risk` and `abstention_rate`, and that class's scored rows for `recall:<class>` and `abstention_rate:<class>` — the same counts the metric carries at coverage 1.0. `n` is not one of spec 005's per-seed fields (US-2.1), so an `n` that followed the accepted rows would differ from seed to seed, `aggregate` would group the five seeds into five, and the number would never be reported at all. The accepted count goes in `notes`, which is per-seed, and the share itself is the `abstention_rate` row beside it.
2. **Given** those rows, **then** the curve is the table: four points per (run, target, metric), the declared coverages and 1.0, each with its five-seed aggregate (spec 005 US-2). `selective_risk` is therefore written at coverage 1.0 as well — the curve's no-abstention end, where it is one minus the accuracy over every scored row. `abstention_rate` is not: it is zero there by construction. Nothing is plotted here; piece 5 draws from the rows.
3. **Given** a coverage at which a class has no accepted row, **then** `macro_f1` and that class's `recall` are not written for that coverage and the reason is printed (spec 005's macro-F1 needs a row of every class). `selective_risk` and `abstention_rate` are still written, and the eval carries on: a sparse class at one coverage is not an input error and never exits 2. Should a coverage accept no row at all, `selective_risk` is not written either — it would be 0/0 over `n = 0` rows, which spec 005 FR-001 rejects — the reason is printed, and only `abstention_rate`, which is then 1.0, records what happened.
4. In domain is the run's own test split; out of domain is every N2 target. The pair is the point of the number, as N1 against N2 is.

### US-8: Per-class abstention (P1)

1. **Given** an in-domain test split at a declared coverage, **then** `abstention_rate:<class>` is written for each of the run's classes: the share of that class's rows that abstained. These are **N1** rows, whose `classes` are the head's: spec 005 US-4.2 keeps an N3 row's classes to the held-out unknowns, so a trained class's abstention rate has no other place.
2. The caution of `[G73]`: abstention that falls on the rarest class hides the disease you most need to see. The rows make it visible — anthracnose is the rarest trained class (spec 003 US-4) — and the W4 task reads them. No rule here refuses a run for it; it is reported, like N4.

### US-9: Calibration error, in and out of domain (P1)

1. **Given** a fitted run and a target, **then** `ece` is the expected calibration error of `softmax(logits / T)` over every scored row of that target, at `coverage = 1.0`: 15 equal-width bins over the largest probability, and the weighted mean over the bins of |accuracy − mean confidence|.
2. It is written for the in-domain test split (in domain) and for every N2 target (out of domain), because calibration is what breaks first under shift (H8 §6.6). `notes` carry the bin count. The held-out unknowns carry no `ece` row: they have no trained-class label, so there is nothing for a probability to be right or wrong about.
3. `ece` is reported, never a gate, and never a verdict criterion: H9 §7 does not name it.

### US-10: A fit is reproducible and idempotent (P1)

1. **Given** the same run, cache, config and manifests, **then** `abstain.json` is identical but for `wallclock_s` and `created_at`, which record when it ran, and `inputs_sha256` is the sha256 of the same inputs object. Nothing here is random: the bank, the Gaussians, the quantiles and the temperature are deterministic functions of the rows.
2. **Given** a run already fitted with the same `inputs_sha256`, **then** the fit prints "nothing to do", writes nothing and adds no compute-log row. `--force` refits and replaces the file.
3. **Given** a changed config (a different `k`, another coverage), **then** `inputs_sha256` changes and the file is rewritten. **The rows do not follow.** A row's identity (spec 005 FR-004) is its run, number, metric, test manifest, split and coverage, and none of those changes when the abstention recipe does, so `append_rows` drops the new rows as already known. The abstention recipe is therefore settled before the first row is written (Clarification 11).

### Edge cases

- **The bank is the training rows, not the cache.** A cache holds every split of its manifest (spec 002); the bank takes the training rows' features only. A row of the test split entering the bank would leak the exam into the distance score, so the fit counts the bank's rows and records the count.
- **k against the bank.** `k` must be at least 1 and below the bank's size; H8's range is 10–50 and the config's default is 20. A bank smaller than `k + 1` rows is `too_few_rows`.
- **The Gaussians.** One covariance shared over the trained classes (the pooled within-class scatter over `N − C`), plus a background Gaussian over all training rows with the class ignored (its scatter over `N − 1`), both with `epsilon · trace(Σ) / D` added to the diagonal, `epsilon` the config's 1e-6, so they invert. A class with fewer than `D + 1` training rows is still fitted, because the covariance is shared; a class with no training rows is `bad_value`. A bank with no variance at all makes the ridge zero too, so FR-011 refuses it before any inversion.
- **Ties at the threshold.** The comparisons are strict (`conf < tau_conf`, `knn > tau_knn`), so a row exactly at the threshold is predicted, not abstained.
- **A run without `abstain.json`.** `make eval` writes its coverage 1.0 rows as it always did and prints `not_fitted: <run_id>` for the rest. S4.4 never breaks the W3 table.
- **float16 caches.** Features are read as float32 through `ms.heads.features` (spec 002 FR-005), so a distance never depends on the storage dtype.

## Requirements

### Functional

- **FR-001 Location.** A fit is `data/heads/<run_id>/abstain.json`, beside `head.pt` and `run.json`, and is git-ignored with the run (spec 003 FR-001). It is written under a temporary name and renamed when complete. Nothing else in the repository is written by a fit.
- **FR-002 `abstain.json`** (`abstain_json_version = 1`). Its keys, in this order:
  - `abstain_json_version`, `run_id`, `head`, `tokens`, `seed`, `classes`, `input_dim`, `model_version`, `quotable`;
  - `fitted_on`: `manifest`, `manifest_sha256`, `frozen`, `train_split`, `n_train` (the bank), `val_split`, `n_val` (the slice), `class_counts` of the slice. Several manifests join their `manifest` and `manifest_sha256` with `+`, as the N-table does;
  - `scores`: for each of `conf`, `knn`, `maha`, `energy`, its orientation as `higher_is_stranger` (false for `conf`, true for the other three; US-1.2), and for `knn` also `k`, for `maha` the `epsilon` used;
  - `thresholds`: one object per declared coverage, keyed by the coverage as a string (`"0.80"`, `"0.90"`, `"0.95"`), each with `tau_conf`, `tau_knn`, `tau_maha` and `coverage_achieved`;
  - `temperature`: `T`, `nll_before`, `nll_after`, `iterations`;
  - `ece_bins`;
  - `config`: `path` and `sha256` of `configs/abstain.yaml`, then the effective recipe with defaults filled in;
  - `cache`: `backbone_id`, `res`, `cache_key`;
  - `inputs_sha256`, `wallclock_s`, `device`, `created_at`, `builder` (module, version, git sha), `notes`.
- **FR-003 `inputs_sha256`.** The first 64 hex digits of the sha256 of `json.dumps(inputs, sort_keys=True)`, where `inputs` holds `fitter_version`, `run_id`, `cache_key`, `tokens`, `config_sha256`, and `train` and `val` as `[manifest sha256, split]` with several manifests' sha256s joined by `+`. The run id already carries the seed and the head (spec 003 FR-002).
- **FR-004 The scores**, as US-1 defines them, all computed from the run's own logits and the cached features:
  - `conf(x) = max_c logits(x)_c`;
  - `knn(x) = ‖x̂ − b_(k)‖₂`, where `x̂` is the L2-normalised feature, the bank holds the L2-normalised training features and `b_(k)` is the k-th nearest of them;
  - `maha(x̂) = min_c [ (x̂ − μ_c)ᵀ Σ⁻¹ (x̂ − μ_c) − (x̂ − μ_0)ᵀ Σ_0⁻¹ (x̂ − μ_0) ]` on the same L2-normalised feature. μ_c are the class means over the bank and Σ is one covariance shared over the trained classes: the pooled within-class scatter over `N − C` (N the bank's rows, C the trained classes). (μ_0, Σ_0) is a single background Gaussian over the whole bank with the class ignored, its scatter over `N − 1`. Both get the ridge of FR-011 before they are inverted;
  - `energy(x) = −logsumexp_c logits(x)_c`.
- **FR-005 Thresholds.** As US-3: `tau_conf` per declared coverage, `tau_knn` and `tau_maha` at `distance_tpr` whatever the coverage, and `coverage_achieved` for the joint rule. The declared coverages are the config's, `{0.80, 0.90, 0.95}` is the default set, and every block of `thresholds` carries all four numbers so that one operating point is read from one place.
- **FR-006 The decision.** `decide` returns `(decision, abstain_reason)` with `decision ∈ {predict, abstain}` and `abstain_reason ∈ {low_confidence, far_from_training}` or null, by US-4. `invalid_input` is spec 006's.
- **FR-007 Temperature.** `T` minimises the validation slice's mean negative log-likelihood under `softmax(z / T)`, over `log T ∈ [log 0.01, log 100]`, by a bounded scalar minimisation to a tolerance of 1e-6. `nll_before` is at `T = 1`. A fit whose `nll_after` is worse than `nll_before` is `degenerate_temperature` and exits 2.
- **FR-008 Calibration error.** `ece` as US-9.1, with the config's `ece_bins` (default 15).
- **FR-009 Metric names.** Spec 005 FR-005's vocabulary gains, with these tests:
  - `unknown_recall`, with no qualifier;
  - `auroc:<score>`, the qualifier one of `conf`, `knn`, `maha`, `energy`;
  - `selective_risk` and `ece`, with no qualifier;
  - `abstention_rate`, with no qualifier or with one of the row's classes.

  `ms.eval.METRICS` gains them with a qualifier kind each: `score` for `auroc` (the qualifier is one of `ms.abstain.SCORES`) and `class?` for `abstention_rate` (no qualifier, or one of the row's classes) beside the existing `None`, `class` and `optional`. `validate_row` rejects `auroc:entropy` and a bare `auroc` as `bad_metric`, and `abstention_rate:unknown_als` on a row whose classes are the head's.
- **FR-010 The rows.** Every row is appended through `ms.eval.append_rows` (spec 005 FR-009) and carries what spec 005 FR-001 asks of it. In particular:
  - N3 rows have `test_split = "holdout_unknown"` and held-out `classes` only;
  - a row at a declared coverage carries that coverage, and its `n` counts the scored rows, not the accepted ones (US-7.1), so that the five seeds of a number share a `number_key` and aggregate; `auroc:*` and `ece` rows carry coverage 1.0;
  - `quotable`, `split_rule`, the hashes and the five-seed aggregates are spec 005's rules unchanged.
- **FR-011 Checks before fitting.** These are checked before the first fit writes anything: the run loads and its manifests are unchanged; the splits are readable for their purposes; the cache exists for the run's backbone, resolution and cache key; the config parses and its coverages are in (0, 1); the bank and the slice are big enough; the bank has some variance, `trace(Σ) > 0`, checked before anything is inverted, because the ridge `ε · trace(Σ) / D` is itself zero when the bank's rows are all the same point; and each of `conf`, `knn` and `maha` moves over the slice (US-3.3). A failure prints `<reason>: …` and exits 2. The reasons are `missing_run`, `missing_cache`, `frozen_manifest_modified`, `test_split_access`, `bad_config`, `bad_value`, `too_few_rows`, `degenerate_score` and `degenerate_temperature`.
- **FR-012 Config.** `model_service/configs/abstain.yaml`, one file, with `k` (20), `coverages` (`[0.80, 0.90, 0.95]`), `default_coverage` (0.90), `distance_tpr` (0.95), `ece_bins` (15), `epsilon` (1e-6) and `n3`: the held-out sources, each with `manifest` and `classes`, in the shape `configs/eval.yaml` uses for `n2`. Its sha256 is an input (FR-003), so a changed file is a new fit.
- **FR-013 API.** `ms.abstain` exports `SCORES` (`conf`, `knn`, `maha`, `energy`), `DECISION_SCORES` (`conf`, `knn`), `REASONS` (`low_confidence`, `far_from_training`), `COVERAGES`, `Fit`, `fit_run`, `load_fit`, `scores`, `decide`, `probabilities`, `ece` and `auroc`. `ms.abstain.fit` exports `main(argv) -> exit code`.
  - `Fit` is the loaded `abstain.json` with what scoring needs: `meta` (the file as it is on disk), `run_id`, `temperature` (the scalar `T`), `k`, `thresholds(coverage)` (the three τ of one declared coverage), `bank` and the Gaussians, rebuilt from the cache the provenance names. Nothing in it is fitted on the fly: `load_fit` refuses a file whose `inputs_sha256` does not match what it reads (`bad_value`).
  - `scores(fit, run, x)` returns a dict of the four arrays; `decide(fit, scores, coverage)` returns `(decision, abstain_reason)`; `probabilities(fit, run, x)` the temperature-scaled probabilities; `ece(proba, y, bins)` and `auroc(strangeness_known, strangeness_unknown)` the two statistics, so that the eval and the service compute them the same way.
- **FR-014 Compute log.** A fit that writes appends one row through `ms.compute_log`: `step = "fit_abstain"`, the backbone, resolution and dataset of the run, `n_images_or_runs = 1`, its own `wallclock_s`, `device`, and an `abstain` object with `run_id`, `k`, `n_train`, `n_val` and `T`. "Its own" is the run's share alone: the bank, the kNN distances of the slice and the Gaussians do not depend on the head, so one call computes them once for every run that shares them, and that work is in no run's seconds — as spec 003 US-6 keeps a call's shared reading out of every run's. The end-to-end cost of a call is therefore larger than the sum of its rows, and the task that runs it records it (DECISIONS 95). `ms.compute_log.STEPS` gains `fit_abstain`, as it gained `env` and `train_head`; every other step keeps its meaning. A fit that was already there writes no row (N7, spec 003 US-6).
- **FR-015 Hand-over to the service (006).** The service reads `abstain.json` beside the run it serves and answers with `decision`, `abstain_reason`, temperature-scaled `scores`, and an `uncertainty` block holding `max_prob`, `entropy`, `ood_knn`, `ood_knn_threshold` and `ood_maha` (H8 §6.2). `options.coverage_target` selects one of the fitted coverages; a coverage that was not fitted is a 422. Nothing here starts the service.

## Success criteria (measurable, technology-agnostic)

- **SC-1** Every quotable head run at 224 has an `abstain.json` whose `fitted_on` names only `train` and `val` splits of its own training manifests (the W4 "Scores" task).
- **SC-2** No test or held-out row is ever read with `purpose ∈ {train, select}` while fitting, and a test proves the refusal exits 2 with `test_split_access`.
- **SC-3** With a fitted run, N1 and N2 at coverage 1.0 keep the values the table already holds: temperature moves probabilities, not decisions (US-5.3).
- **SC-4** N3 exists for both backbones, the three heads and five seeds, for both held-out sets, with `unknown_recall` above zero at the default coverage and an `auroc:conf` and an `auroc:knn` aggregate per (backbone, head, training manifest, held-out set) — three training sets exist at 224 (Makerere, Tanzania, Makerere + iBean), so criterion 2 has three values per (backbone, head, held-out set) to choose between, and **which of them the verdict reads is spec 005 US-6's to settle** before the W4 "Verdict" task runs (the W4 "N3" task).
- **SC-5** Every (run, target) pair that has a coverage 1.0 row has rows at the three declared coverages too, so that the selective-risk curve has four points.
- **SC-6** Re-running `make abstain` fits nothing and adds no compute-log row; re-running `make eval` appends no row.
- **SC-7** The CPU tests use synthetic manifests and caches, run in under one minute, and need neither network nor GPU.

## Assumptions

- The head runs exist as spec 003 says, their caches as spec 002 says, and the manifests are frozen (DECISIONS 51). A fit reads them and trains nothing.
- The held-out manifests hold enough rows to measure on: `swm_v1` for white mould, `makerere_v1`'s angular-leaf-spot rows for ALS (spec 001 FR-007 makes every `unknown_*` row `holdout_unknown`).
- Piece 5 draws the curves from the rows; nothing here plots.

## Clarifications

Closed on 2026-09-20, by this spec, against H8 §6.6 and the work plan's S4.4 row. Where H8 is silent, the reading is the one that keeps the number honest — it never loosens what may be fitted on.

1. **The confidence score is the head's largest logit, for all three heads.** H8 writes "max logit / max prototype cosine" because the two heads express confidence differently; for `proto` the largest logit *is* the largest prototype cosine divided by τ > 0, so one definition covers both, and `mix` (logits = log p) gives the log of its largest probability. One definition also keeps the score comparable across the three heads of one backbone.
2. **The decision uses two scores; the relative Mahalanobis is a second opinion.** H8's decision sentence says "either score" after naming two, its `abstain_reason` vocabulary has exactly two fitted reasons, and its response carries `ood_knn_threshold` but no Mahalanobis threshold. The work plan's S4.4 row still asks for `tau_maha`, so it is fitted, recorded and reported — and never consulted by `decide`. DECISIONS 79 reads H9 §7 criterion 2 as "confidence and kNN distance", the same two.
3. **A declared coverage is the confidence gate's; the distance gate sits at TPR 95 % under it.** H8 §6.6 says exactly that, and this spec follows it literally. The work plan's S4.4 row — "thresholds `{tau_conf, tau_knn, tau_maha}` at declared coverages {0.80, 0.90, 0.95}" — is satisfied by a `thresholds` block per coverage that carries all three; `tau_knn` and `tau_maha` simply hold the same value in each.
   - **Why not move all three with `c`** (the reading this spec took until it had a number). Each threshold at the `c` quantile makes the joint rule keep about `c²`: measured on the first real fit, 0.64 of the slice at a declared 0.80. H8's rule keeps about `0.95 · c` — 0.76 — so what the operating point is called stays much closer to what it does. A distance gate at TPR 95 % is also the operating point the OOD literature H8 cites reports (FPR@TPR95), which keeps N3 readable beside it. DECISIONS 94 records the change and the measurement.
   - Not taken: searching a pair of thresholds whose union rejects exactly 1 − c. It would make each score's threshold depend on the other and leave "TPR 95 %" with no meaning at all. A curve plotted against the achieved coverage needs no such search: `abstention_rate` sits beside `selective_risk` at every operating point (US-7.1).
4. **`k = 20`, the middle of H8's 10–50.** It is a config value, not a constant in code, so the ablation is a config change with a new `inputs_sha256`.
5. **AUROC is computed on one orientation** (US-1.2) so that "≥ 0.02 AUROC advantage" (H9 §7) compares like with like, and `auroc:*` rows carry coverage 1.0 so that the verdict, which reads coverage 1.0 aggregates (spec 005 US-6.1), finds them.
6. **The knowns of an AUROC row are the run's own in-domain test rows.** H8 says "known-vs-unknown" without naming the knowns. The in-domain test split is the set N1 already reports, so the two numbers stand on the same rows, and the row carries both manifests' hashes.
7. **One source per held-out set in v0**: `makerere_v1` for ALS, `swm_v1` for white mould. iBean's and the crops' ALS rows are left out (US-6.3), and DECISIONS 79 wants one pair of numbers per held-out set. The cost is that ALS is a new dataset for a Tanzania-trained run and a known one for a Makerere-trained run; US-6.4 makes the row say which, rather than pretending the two are one number.
8. **Temperature never changes a decision**, so W3's N1 and N2 stay as they are (US-5.3). Calibration enters the table as `ece` rows and the service as scaled probabilities.
9. **`selective_risk` is the 0–1 error over the accepted rows**, the quantity `[C50]` brackets at 0.520 under shift. The class-balanced view of the same operating point is the `macro_f1` and `recall:<class>` rows written at that coverage, so nothing hides behind an average.
10. **The N-table's `coverage` field is the declared coverage, not the achieved one** (US-3.4). What was achieved on the scored set is the `abstention_rate` row beside it.
11. **The abstention recipe is settled before the first row is written.** Changing `k`, ε or the bins later changes no run id, so the new rows would share their identity with the old ones and `append_rows` would drop them (US-10.3): unlike a head recipe, an abstention recipe cannot be superseded row by row. Widening `coverages` is the one safe change, because a coverage that has no rows yet appends cleanly. Making the fit part of a row's identity is spec 005's to decide, and it is not needed if `k` is settled in the W4 "Scores" task, as the owner settled `max_epochs` before the heads were trained again (DECISIONS 80).
12. **The relative Mahalanobis formula is stated here because the cited note is the plain one.** H8 writes "relative Mahalanobis `[G83]`", and `[G83]` is the class-conditional Mahalanobis detector. The relative form — the same distance minus the distance to one background Gaussian — is what FR-004 implements and what the second opinion means; a source for the relative variant is the owner's to add to `00_sources.md`.

## Out of scope

- **Conformal sets (RAPS).** H8 makes them optional in v0 — "implement only if W4 has slack" — and the owner left them out of this week's queue on 2026-09-20. Nothing here writes a `conformal_set`; the service's field stays null until they exist.
- The numbers themselves, and the plots (piece 5).
- Training anything: a fit is post-hoc on a trained run (spec 003).
- The service's transport, its 422s and `invalid_input` (spec 006).
- A second opinion becoming a decision score, and any threshold fitted on anything but the in-domain validation slice.
