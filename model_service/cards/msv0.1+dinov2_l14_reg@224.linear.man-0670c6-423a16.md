# Model card — msv0.1+dinov2_l14_reg@224.linear.man-0670c6-423a16

<!--
The template of H8 §6.8, filled in by `python -m ms.registry card` (`make card`). The prose
below is the same for every piece 4 v0 model; everything between the markers is rendered
from the artefacts — the run, its fit, the manifests and `results/n_table.jsonl` — so a card
cannot drift from the numbers it quotes. Do not edit a rendered card by hand: `ms.registry
register` refuses a card that differs from what the artefacts render now.
-->

**Status**: registered · **Generated**: 2026-09-20T13:57:42Z · **Rendered from**: `b581f37 + uncommitted changes`
**Run**: `dinov2_l14_reg-224-linear-cls-s0-89fe37fe` · **Served by**: `model_service/configs/service.yaml`

*A card cannot carry the sha of the commit that will contain it, so the one above says
which repository rendered it and nothing more. What pins this model is the set of
hashes under "Reproducing this model".*

## Intended use

Organ-level triage of bean-leaf imagery: one photograph of a bean plant in, one of
*healthy* or *rust* out, or *abstain* when the model should not be trusted on it. It is
built for the central-node machine behind the plot map and the future decision layer (H8
§6.9), reached over the interface of H8 §6.2 — never on the robot, and never inside a control
loop.

**This model decides between two classes**, the ones its training manifests hold.
The label space of the task is `healthy`, `rust` and `anthracnose` (H8 §6.3); a class absent
from the training data is a class this model cannot return, and photographs of it land on
whichever of its own classes they look most like — or on `abstain`.

It is a **probe of the pipeline and of the data, not a model choice** (H8 §6.1): frozen
foundation features with a small head, trained on open smartphone photographs of African bean
fields, kept deliberately simple so that what the numbers show is the data and the protocol.

### Not validated for, and not to be used for

- **Polish imagery, robot views, under-panel light, pods, whole plots.** Nothing in the
  training or evaluation data was taken in Poland, from a robot, or under an agrivoltaic
  panel. H8 §6.7's guard-rail, written before any number existed, is that the drop from one
  African set to another is a *lower bound* on the drop to Poland — a floor under the cost,
  never an estimate of it, and this card does not measure that drop at all.
- **Anything outside its two trained classes.** Angular leaf spot and white mould
  were held out on purpose and are never predicted; every other disease, pest, deficiency or
  damage is outside the label space altogether, and so is any class of the task this model
  was not trained on.
- **Severity, counting or localisation.** There are no lesion boxes, no masks and no severity
  score in v0 (H8 §6.1). A response's `localisation` is always null.
- **Nutrient deficiencies.** No image data for them exist (E1 §3).
- **Any decision taken on one frame alone.** The map aggregates; a single answer, abstention
  included, is evidence and not a verdict.

## The model

<!-- model:begin -->
| Part | What |
|---|---|
| `model_version` | `msv0.1+dinov2_l14_reg@224.linear.man-0670c6-423a16` |
| Backbone | `facebook/dinov2-with-registers-large` (`dinov2_l14_reg`) |
| Backbone revision | `e4c89a4e05589de9b3e188688a303d0f3c04d0f3` |
| Weights sha256 | `edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b` |
| Input resolution | 224 px |
| Features | `cls`, 1024-d, frozen (never fine-tuned) |
| Preprocessing | `resize_short=224;center_crop=224;norm=imagenet` |
| Compute dtype | `float16`, stored float16 |
| Cache key | `2b017210105b0b1f` |
| Head | `linear`, recipe `model_service/configs/heads/linear.yaml` |
| Head recipe sha256 | `bc43d483346d74527fae2b849ebea1a478791fef0051f27d5391e97387fb46ad` |
| Seed | 0 (the numbers below are over seeds 0–4) |
| Classes | `healthy`, `rust` |
| Training epochs | 197, best 187 |
| Quotable | yes |
<!-- model:end -->

## Training data

Each row of each manifest carries its own licence and attribution line; the manifests are
frozen by sha256 before any head is trained (spec 001 US-5), and those hashes are what the
`model_version` string above is built from.

<!-- data:begin -->
### `makerere_v1` — blocked:district, role `train_eval`

| Field | Value |
|---|---|
| Manifest | `data/manifests/makerere_v1.jsonl` |
| sha256 | `0670c68b51f5da3cf8be67aebf0d5da1eae2c9db0828dc0018eceadb974b4bde` |
| Rows | 15402 |
| Splits | holdout_unknown 5098, test 2015, train 6818, val 1471 |
| Classes | healthy 5284, rust 5020, unknown_als 5098 |
| Split rule | `blocked:district` |

Sources, licences and the attribution each one requires:

| Dataset | Rows | Licence | Attribution |
|---|---|---|---|
| makerere | 15402 | `CC0-1.0` | Mugalu, Ben-Wycliff; Nakatumba-Nabende, Joyce; Katumba, Andrew; Babirye, Claire; Tusubira, Francis-Jeremy; Mutebi, Chodrine; Nsumba, Solomon; Namanya, Gloria (2022). Makerere University Beans Image Dataset (Version 2.1) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/TCKVEW |

### `ibean_v1` — unblocked:random_by_phash_group, role `test_only`

| Field | Value |
|---|---|
| Manifest | `data/manifests/ibean_v1.jsonl` |
| sha256 | `423a16666a4880ddf9b2bb934a29b614c834a1b9ba7170b4d8b994935bb86060` |
| Rows | 1295 |
| Splits | holdout_unknown 432, test 172, train 604, val 87 |
| Classes | healthy 427, rust 436, unknown_als 432 |
| Split rule | `unblocked:random_by_phash_group` |

Sources, licences and the attribution each one requires:

| Dataset | Rows | Licence | Attribution |
|---|---|---|---|
| ibean | 1295 | `MIT` | Makerere AI Lab (2020). iBean: bean leaf disease images (healthy, angular leaf spot, bean rust) [Data set]. MIT licence. https://github.com/AI-Lab-Makerere/ibean |
<!-- data:end -->

## Evaluation

Every number is read from `model_service/results/n_table.jsonl` with the split rule printed
beside it (H8 D-9); nothing here is computed by the card. N1 to N3 are five-seed means with a
95 % Student *t* interval — an interval of zero width is a score every seed agreed on, which
the two distance scores are by construction, because their bank does not depend on the seed.
N4 and N6 are one measurement with an interval of their own kind, as their rows say.

Unless a row says otherwise its coverage is 1.0: **abstention off**, every frame decided. What
the operating points do to the same numbers is the selective-risk table at the end of this
section, and `selective_risk` at coverage 1.0 is simply the error rate over every frame.

<!-- evaluation:begin -->
### N1 — within-dataset, blocked

_None._

### N2 — cross-dataset

| Metric | Value | 95 % interval | Tested on | Split rule | n |
|---|---|---|---|---|---|
| `ece` | 0.0101 | [0.0072, 0.0129] | `tanzania_v1` / test | `blocked:date` | 21059 |
| `macro_f1` | 0.8894 | [0.8782, 0.9007] | `tanzania_v1` / test | `blocked:date` | 21059 |
| `recall:healthy` | 0.9599 | [0.9544, 0.9654] | `tanzania_v1` / test | `blocked:date` | 19410 |
| `recall:rust` | 0.9813 | [0.9784, 0.9843] | `tanzania_v1` / test | `blocked:date` | 1649 |
| `selective_risk` | 0.0385 | [0.0336, 0.0433] | `tanzania_v1` / test | `blocked:date` | 21059 |

A cross-dataset row decides among the classes **both** sides have (spec 005 US-7.3), so rows of a class this model does not carry are not scored at all and `n` is smaller than the target's test split by exactly those rows. They are not errors and not abstentions; they were never put to it.

### N3 — the unknowns it was never shown

| Metric | Value | 95 % interval | Tested on | Split rule | n |
|---|---|---|---|---|---|
| `auroc:conf` | 0.7007 | [0.6787, 0.7227] | `makerere_v1+ibean_v1+makerere_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 7285 |
| `auroc:energy` | 0.7022 | [0.6815, 0.7228] | `makerere_v1+ibean_v1+makerere_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 7285 |
| `auroc:knn` | 0.5927 | [0.5927, 0.5927] | `makerere_v1+ibean_v1+makerere_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 7285 |
| `auroc:maha` | 0.6910 | [0.6910, 0.6910] | `makerere_v1+ibean_v1+makerere_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 7285 |
| `auroc:conf` | 0.6980 | [0.6665, 0.7295] | `makerere_v1+ibean_v1+swm_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 2871 |
| `auroc:energy` | 0.6795 | [0.5963, 0.7627] | `makerere_v1+ibean_v1+swm_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 2871 |
| `auroc:knn` | 0.9960 | [0.9960, 0.9960] | `makerere_v1+ibean_v1+swm_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 2871 |
| `auroc:maha` | 0.5819 | [0.5819, 0.5819] | `makerere_v1+ibean_v1+swm_v1` / holdout_unknown | `blocked:district+holdout_unknown+unblocked:random_by_phash_group` | 2871 |

An N3 row is scored on two sets at once: the run's own in-domain test rows, which are the knowns, and the held-out set, which are the unknowns. "Tested on" therefore names the training manifests and the held-out source together, and a manifest that is both appears twice.

A held-out class can be a new **disease** without being a new **dataset**: angular leaf spot here is held out of training but comes from a set this model trained the rest of, while white mould is a set it has never seen at all (spec 004 Clarification 7). The two AUROCs are not measuring the same kind of strangeness.

### N6 — latency

| Metric | Value | 95 % interval | Tested on | Split rule | n |
|---|---|---|---|---|---|
| `latency_ms:b1` | 115.8060 | [81.5600, 149.0183] | `makerere_v1` / test | `blocked:district` | 256 |
| `latency_ms:b32` | 31.1242 | [28.0285, 33.8090] | `makerere_v1` / test | `blocked:district` | 256 |
| `latency_ms:cached_b1` | 43.3488 | [42.1960, 53.0056] | `makerere_v1` / test | `blocked:district` | 256 |
| `latency_ms:cached_b32` | 3.8346 | [3.7804, 4.4367] | `makerere_v1` / test | `blocked:district` | 256 |

### Selective risk, by declared coverage

| Coverage | Selective risk | 95 % interval | Tested on |
|---|---|---|---|
| 0.80 | 0.0102 | [0.0077, 0.0126] | `tanzania_v1` / test |
| 0.90 | 0.0202 | [0.0160, 0.0245] | `tanzania_v1` / test |
| 0.95 | 0.0346 | [0.0290, 0.0402] | `tanzania_v1` / test |

**Not measured for this model**: N1. An honest blank, not a zero — see `docs/piece4-work-plan.md` and `model_service/DECISIONS.md` for why.
<!-- evaluation:end -->

## Operating point

Abstention is two gates (H8 §6.6): the head's confidence, whose threshold moves with the
declared coverage, and the distance to the training features, whose threshold sits at TPR
95 % whatever the coverage. A request may ask for any fitted coverage; one that was not
fitted is refused, because a service does not interpolate an operating point it never
measured.

<!-- operating-point:begin -->
| Declared coverage | `tau_conf` | `tau_knn` | `tau_maha` (reported) | Achieved on the slice |
|---|---|---|---|---|
| 0.80 | 0.223169 | 0.485453 | 7.531394 | 0.7619 |
| 0.90 (default) | 0.132776 | 0.485453 | 7.531394 | 0.8556 |
| 0.95 | 0.074734 | 0.485453 | 7.531394 | 0.9031 |

| Field | Value |
|---|---|
| Temperature | 0.126285 |
| k (nearest neighbours) | 20 |
| Fitted on | `val` of `makerere_v1+ibean_v1`, 1558 rows — the in-domain validation slice only |
| Bank | 7422 training features |
| Decision | abstain when `conf < tau_conf` (`low_confidence`) or `knn > tau_knn` (`far_from_training`); the Mahalanobis distance is reported, never consulted |
<!-- operating-point:end -->

## Known failure modes

<!-- failure-modes:begin -->
How well the strangeness scores tell an unseen disease from a known one (AUROC, 0.5 is a coin toss):

| Held-out | Score | AUROC | 95 % interval | Knowns + held-out source |
|---|---|---|---|---|
| unknown_als | `conf` | 0.7007 | [0.6787, 0.7227] | `makerere_v1+ibean_v1+makerere_v1` |
| unknown_als | `energy` | 0.7022 | [0.6815, 0.7228] | `makerere_v1+ibean_v1+makerere_v1` |
| unknown_als | `knn` | 0.5927 | [0.5927, 0.5927] | `makerere_v1+ibean_v1+makerere_v1` |
| unknown_als | `maha` | 0.6910 | [0.6910, 0.6910] | `makerere_v1+ibean_v1+makerere_v1` |
| unknown_wm | `conf` | 0.6980 | [0.6665, 0.7295] | `makerere_v1+ibean_v1+swm_v1` |
| unknown_wm | `energy` | 0.6795 | [0.5963, 0.7627] | `makerere_v1+ibean_v1+swm_v1` |
| unknown_wm | `knn` | 0.9960 | [0.9960, 0.9960] | `makerere_v1+ibean_v1+swm_v1` |
| unknown_wm | `maha` | 0.5819 | [0.5819, 0.5819] | `makerere_v1+ibean_v1+swm_v1` |

And the share of those images the model actually refuses, at each declared coverage:

| Held-out | Coverage | Unknown recall | Knowns + held-out source |
|---|---|---|---|
| unknown_als | 0.80 | 0.3635 | `makerere_v1` |
| unknown_als | 0.90 | 0.2292 | `makerere_v1` |
| unknown_als | 0.95 | 0.1419 | `makerere_v1` |
| unknown_wm | 0.80 | 0.9985 | `swm_v1` |
| unknown_wm | 0.90 | 0.9930 | `swm_v1` |
| unknown_wm | 0.95 | 0.9854 | `swm_v1` |
<!-- failure-modes:end -->

Beyond what the numbers show:

- **The acquisition site is in the features.** A probe reads the source dataset from the
  frozen features at far above chance, and the district within Makerere too (N4). A head can
  therefore learn *where* instead of *what*. That is why the splits that decide a number are
  blocked — Makerere by district, Tanzania by capture date — and why the cross-dataset number
  is the one to read. iBean carries no metadata to block on, so it is test-only and
  `unblocked:random_by_phash_group`, and no number of its own is ever quoted.
- **Calibration is fitted in domain, and shift is what breaks it first.** Over the whole
  campaign the expected calibration error runs about 0.005 in domain against 0.367 out of it
  (DECISIONS 103). This model's own `ece` rows are in the table above: read those, not the
  median — a small one says this transfer direction was kind, not that calibration holds.
  The abstention gates, not the probabilities, are what the operating point rests on.
- **A class-trip confound in Makerere.** Its healthy photographs all come from one April trip
  and most of its rust from May, so an in-domain number on that set partly measures the trip.
- **Compression, blur, rain, night.** Nothing in the evaluation isolates them.

## Licences and terms

<!-- licences:begin -->
| What | Which | Licence | Terms |
|---|---|---|---|
| Backbone | `facebook/dinov2-with-registers-large` | Apache-2.0 | No restriction beyond its own licence |
| Trained on | `makerere` | CC0-1.0 | No attribution required. The line below is given anyway, because the authors made the dataset and a paper that used it without saying so would be poorer |
| Trained on | `ibean` | MIT | The licence text and its copyright notice must travel with any redistribution |
| Evaluated on | `swm` | CC-BY-4.0 | **Attribution required**: the line below must travel with any use |
| Evaluated on | `tanzania` | CC-BY-4.0 | **Attribution required**: the line below must travel with any use |

The attribution lines, in full:

- **makerere** — Mugalu, Ben-Wycliff; Nakatumba-Nabende, Joyce; Katumba, Andrew; Babirye, Claire; Tusubira, Francis-Jeremy; Mutebi, Chodrine; Nsumba, Solomon; Namanya, Gloria (2022). Makerere University Beans Image Dataset (Version 2.1) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/TCKVEW
- **ibean** — Makerere AI Lab (2020). iBean: bean leaf disease images (healthy, angular leaf spot, bean rust) [Data set]. MIT licence. https://github.com/AI-Lab-Makerere/ibean
- **swm** — Pereira, Rubens; Torres, Ricardo; Neto, Justino; Borges, Dibio; Lobo Junior, Murillo; Pedrini, Helio (2026). Sclerotinia and White Mold (SWM) dataset (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/jxvmr8rchx.1
- **tanzania** — Laizer, Hudson; Mduma, Neema; Machuve, Dina; Lyimo, Tumaini; Babirye,  Claire; Swai, Jenifa; Siwingwa, Adam (2023). Common Beans Imagery Dataset for Early Detection of Crop Diseases (Version 01) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8286126
- **tanzania** — Mduma, Neema; Laizer, Hudson; Kiriba, Deodatus (2025). A Labeled Dataset of Healthy and Diseased Common Beans from Tanzania [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15685315
<!-- licences:end -->

## Reproducing this model

<!-- reproduce:begin -->
```bash
make cache BB=dinov2_l14_reg RES=224 DS=<each dataset>
python -m ms.heads.train --backbone dinov2_l14_reg --res 224 \
    --head linear --seed 0 \
    --train-manifest data/manifests/makerere_v1.jsonl data/manifests/ibean_v1.jsonl \
    --split train --val-split val
make abstain ARGS="--run dinov2_l14_reg-224-linear-cls-s0-89fe37fe"
make eval
```

The run id is derived from those inputs, so the same command gives the same folder and the same weights, bit for bit (spec 003). What pins it:

| Input | Hash |
|---|---|
| Training manifests | `0670c68b51f5…` + `423a16666a48…` |
| Backbone weights | `edccedab2c4e…` |
| Feature cache key | `2b017210105b0b1f` |
| Head recipe | `bc43d483346d…` |
| Abstention recipe | `59c1f756d182…` |
<!-- reproduce:end -->

## Contact and provenance

Piece 4 of the AgriRobot bean-disease system (H8 §6), owner the ML role. The plan is
`87_H8_current-plan-demo1-end-to-end-without-a-robot.md` §6; the execution is
`docs/piece4-work-plan.md`; the dated choices are `model_service/DECISIONS.md`; the contracts
are `model_service/specs/`. The champion/challenger verdict this model sits under is
`model_service/results/verdict.md`, written by code from the table alone.
