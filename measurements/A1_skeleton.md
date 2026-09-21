# A1 (skeleton) — Frozen foundation features for bean disease recognition: a blocked, two-backbone evaluation with abstention across three open field datasets

Piece 4's evaluation note, and the methods-and-results skeleton of article A1 (H8 §6.1).
Executes **H8 §6.11 items 5 and 6**. Owner M. Written 2026-09-21 (W6).

**Every number here is read out of an artefact, not typed.** Sources: `model_service/results/n_table.jsonl`
(sha256 `5195ec3d560260c3…`), `model_service/results/compute_log.jsonl` (`e0802c7e2afacbb2…`),
`model_service/results/verdict.md`, and the frozen manifests' sidecars. Repository at `839a769`.
Regenerate the inputs with `make eval`, `make bench`, `make compute-log-summary`; a number that
moves here without one of those moving is a bug.

**Where a number does not exist, this note prints a blank and says why.** A blank is never
rounded up into a claim. The list of blanks is §6.

---

## 0. What this note is

The article A1 can be written without a single Polish image, because its claims are about the
open bean sets and about a workflow, not about 'Złota Saxa' (H8 §6.1). This note is that
article's methods and results, in the order they will be written, so that W11 ends with a
draft rather than with a plan to write one.

What it is **not**: a report on our own plot (piece 5 owns the measurements report), a
recommendation of a model (the service is a probe of the pipeline and of the data — H7
statement 3), or a demonstration that the system works end to end (§6, blank 1).

---

## 1. Method

### 1.1 Data

Five frozen manifests over four public sources. Row counts are manifest rows after exact
(sha256) de-duplication and the exclusions each sidecar lists.

| manifest | source | licence (as the record states) | rows | classes (KM2) | role |
|---|---|---|---|---|---|
| `tanzania_v1` | Zenodo 15685315 + Zenodo 8286126 | CC BY 4.0 | 112,176 | healthy 97,048 · rust 8,243 · anthracnose 6,885 | train + eval |
| `makerere_v1` | Harvard Dataverse doi:10.7910/DVN/TCKVEW v2.1 | CC0-1.0 | 15,402 | healthy 5,284 · rust 5,020 · unknown\_als 5,098 | train + eval |
| `makerere_crops_v1` | the same, cut to its 36,640 lesion boxes + 10 % margin | CC0-1.0 | 27,019 | rust 13,816 · unknown\_als 13,203 | eval (crops) |
| `ibean_v1` | Hugging Face `AI-Lab-Makerere/beans` @ `27aa014` | MIT | 1,295 | healthy 427 · rust 436 · unknown\_als 432 | test-only |
| `swm_v1` | Mendeley Data jxvmr8rchx v1 (R-SWM originals) | CC BY 4.0 | 684 | unknown\_wm 684 | held-out unknown |

Attribution lines per licence are in each `data/raw/<id>/DOWNLOAD.json` and are reproduced in
the model card. Two corrections the work made to H8 §6.1's own table: **Makerere is CC0-1.0**,
not CC BY 4.0, per the record (the datasheet disagrees with the record; the record governs);
and the Tanzanian sets overlap — 36,683 of tz59k's 36,684 distinct images are byte-identical
to tz155k images, so pooling them adds one image, and the two are **one** manifest.

### 1.2 The class map and the held-out unknowns

`configs/class_map_v1.yaml` (v1, sha256 `0af56275…`) maps every source's raw label to five KM2
classes. Three are trained: **healthy, rust, anthracnose**. Two are never trained and exist to
test "I don't know": **`unknown_als`** (angular leaf spot) and **`unknown_wm`** (white mould).
`unknown` is a decision the service makes, not a class the head has.

Anthracnose exists in Tanzania only. A head trained on Makerere or on Makerere + iBean has
**two** classes, and cannot say `anthracnose` — including the model the demo service serves.

### 1.3 Splits and the blocking rule

Blocked splits everywhere the data allow it (D-9), and the rule is printed under every number
in the N-table.

- `makerere_v1` — **`blocked:district`**. Train: Bugiri+Mayuge, Kayunga, Kiboga, Mbale,
  Serere, Sironko. Validation: Mubende, Pallisa. Test: Hoima, Lyantonde, Ntungamo. No district
  is on two sides.
- `tanzania_v1` — **`blocked:date`**. The archives carry no region or session; every image but
  one has an EXIF capture date. Train 78,454 · validation 11,354 · test 22,368.
- `makerere_crops_v1` — inherits its parent's district blocks.
- `ibean_v1` — **`unblocked:random_by_phash_group`**, and test-only. Its own N1 is never
  quoted; it appears as an N2 *target* and, in one direction, as part of an N2 *source*, and
  the unblocked rule is stated wherever it does.
- Near-duplicates are grouped by perceptual hash (threshold 2) and a group never straddles a
  split.

Every manifest was frozen before the first head was trained, and its sha256 written to
`data/manifests/FROZEN.jsonl` and to `DECISIONS.md`:

| manifest | sha256 | frozen |
|---|---|---|
| `ibean_v1` | `423a1666…` | 2026-09-19T13:18:26Z |
| `makerere_v1` | `0670c68b…` | 2026-09-19T13:18:27Z |
| `makerere_crops_v1` | `efab8bb0…` | 2026-09-19T13:23:20Z |
| `swm_v1` | `601234ce…` | 2026-09-19T13:23:21Z |
| `tanzania_v1` | `bf50c32a…` | 2026-09-19T13:23:23Z |

### 1.4 Backbones and the feature cache

Two frozen backbones, no fine-tuning and no adapters of any kind (F6 §5).

| | champion | challenger |
|---|---|---|
| id | `dinov2_l14_reg` | `dinov3_l16` |
| weights | `facebook/dinov2-with-registers-large` @ `e4c89a4e` | `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc286` |
| licence | Apache-2.0 | DINOv3 License (Meta, 2025-08-14), **gated, research-only until H6 C5 is signed** |
| terms accepted | — | 2026-09-18, on the ML role's own account |
| patch | 14 | 16 |
| dim | 1024 | 1024 |
| resolutions | 224, 518 | 224, 512 |

Features are extracted once per (backbone, resolution, manifest) and cached as float16: `cls`
(the first token) and `meanpatch` (the mean over patch tokens, registers excluded). The cache
key is `(backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype)` and sits
in the cache path, so a changed key never overwrites a cache. Preprocessing is
`resize_short=R;center_crop=R;norm=imagenet`, done in torchvision from that string, not by an
image processor.

Nothing containing DINOv3 features or weights leaves the project's machines until C5 is
signed; the release with the paper is code, manifests and **DINOv2** features only.

### 1.5 Heads

Three small heads on the frozen features, all sharing one recipe:

- **linear** — one affine map; multinomial logistic regression.
- **proto** — 4 prototypes per class, k-means-initialised on the run's seed; a class scores the
  cosine of its nearest prototype over a learnable temperature.
- **mix** — the fixed 0.5/0.5 mixture of the two heads' probabilities.

Recipe: AdamW, lr 1e-3, weight decay 1e-4, batch 256, focal loss (γ = 2), label smoothing 0.1,
class-balanced sampling, early stopping on the in-domain validation split's per-class-averaged
cross-entropy with patience 10, `max_epochs` 300. H8 §6.5 says 50; the owner raised it on
2026-09-19 so that early stopping, not the cap, ends every run. **No run reached 300**; the
longest stopped after 208 epochs.

k-means is pinned to one thread: on several threads scikit-learn sums partial clusters in
completion order, and the same seed gave different prototypes from call to call.

### 1.6 Abstention and calibration

Fitted on the in-domain **validation** split only; the test splits are never touched
(`configs/abstain.yaml`, sha256 an input of every fit).

- **Three decision scores.** *Confidence* (the head's top probability), *kNN distance* to the
  k = 20 nearest training features, and a *relative Mahalanobis* distance to each class's
  training cloud. *Energy* is recorded and not used.
- **The coverage knob.** `τ_conf` is the quantile of the confidence score that keeps 0.80,
  0.90 or 0.95 of the in-domain validation rows. The two distance gates keep 95 % of that
  slice at every coverage, so the knob moves how sure the head must be and leaves the "this
  photo is unlike anything I trained on" gate where it is.
- **The joint rule abstains when any gate fires**, so the coverage a run achieves is at most
  the declared one — and out of domain it is far below it (§2.2).
- **Temperature scaling** on the same validation slice; ECE over 15 equal-width bins on the
  largest probability.
- k = 20 is the middle of H8 §6.6's 10–50 range, taken by fiat: the validation slice holds no
  unknown row, and choosing k on the held-out unknowns would be fitting on the exam.

### 1.7 Protocol

- **Five seeds** (0–4) for every configuration; every quoted number is the mean with a 95 %
  Student-*t* interval over the five.
- **The grid, as the run folders record it:** 2 backbones × 3 training sets (Tanzania,
  Makerere, Makerere + iBean) × 3 feature settings (224/CLS, 224/CLS⊕mean-patch, 518 or
  512/CLS) × 3 heads × 5 seeds = **270 current head runs**, plus the one iBean vertical-slice
  run, which is never quotable. 54 configurations, five seeds each. 361 run folders exist:
  those 271, and **90** from the retired 50-epoch recipe, whose folders stay and whose table
  rows are marked `superseded`.
- **The N-table** holds 25,677 rows, of which 18,508 are current and 3,070 are current,
  quotable five-seed aggregates. Rows are never deleted and never changed except for the
  `superseded` mark.
- **The numbers** are H8 §6.7's: N1 (within-dataset, blocked), N2 (cross-dataset), N3
  (unknowns), N4 (site probe), N5 (negative control — piece 5), N6 (latency), N7 (compute).
- **The verdict criteria were written before any number existed** (H9 §7, copied into spec 005
  US-6) and the verdict is computed by code from the table. Nobody edits it.
- **Main setting for every table below: 224 px, CLS, declared coverage 1.00** unless the
  column says otherwise. Resolution and token type are ablations (§2.7).

---

## 2. Results

### 2.1 The N-table

macro-F1, five seeds, mean [95 % interval]. `n` is the test rows. The crop direction reports
rust recall, because the crops carry no healthy class.

**Champion — `dinov2_l14_reg`, 224, CLS**

| head | N1 Makerere<br>`blocked:district`<br>n=2,015 | N1 Tanzania<br>`blocked:date`<br>n=22,368 | N2 T→Makerere<br>n=2,015 | N2 T→iBean<br>`unblocked`<br>n=863 | N2 M+iBean→T<br>n=21,059 | N2 T→crops<br>rust recall, n=2,655 |
|---|---|---|---|---|---|---|
| linear | 0.9913 [0.9902, 0.9925] | 0.9941 [0.9938, 0.9945] | **0.3622** [0.3604, 0.3641] | 0.5446 [0.5387, 0.5505] | **0.8894** [0.8782, 0.9007] | 0.4747 [0.4569, 0.4926] |
| proto | 0.9933 [0.9913, 0.9953] | 0.9955 [0.9949, 0.9962] | 0.3655 [0.3409, 0.3902] | 0.5249 [0.4594, 0.5905] | 0.7328 [0.6665, 0.7990] | 0.4173 [0.3654, 0.4693] |
| mix | 0.9941 [0.9915, 0.9967] | 0.9950 [0.9945, 0.9956] | 0.3626 [0.3466, 0.3786] | 0.5266 [0.4799, 0.5732] | 0.8605 [0.8431, 0.8779] | 0.4354 [0.3998, 0.4710] |

**Challenger — `dinov3_l16`, 224, CLS**

| head | N1 Makerere | N1 Tanzania | N2 T→Makerere | N2 T→iBean | N2 M+iBean→T | N2 T→crops (rust recall) |
|---|---|---|---|---|---|---|
| linear | 0.9913 [0.9901, 0.9926] | 0.9915 [0.9901, 0.9929] | 0.3598 [0.3586, 0.3610] | 0.5156 [0.5112, 0.5200] | 0.8756 [0.8673, 0.8838] | 0.4852 [0.4787, 0.4917] |
| proto | 0.9913 [0.9904, 0.9922] | 0.9956 [0.9950, 0.9963] | 0.3458 [0.3440, 0.3476] | 0.5308 [0.5138, 0.5477] | 0.7177 [0.6520, 0.7833] | 0.3189 [0.2835, 0.3542] |
| mix | 0.9914 [0.9904, 0.9925] | 0.9960 [0.9957, 0.9962] | 0.3474 [0.3443, 0.3504] | 0.5205 [0.5103, 0.5307] | 0.8125 [0.7721, 0.8530] | 0.3671 [0.3427, 0.3915] |

*Caption (N4, §2.4): a logistic probe reads the acquisition dataset from the same frozen CLS
features at **0.983** (DINOv2) and **0.992** (DINOv3) balanced accuracy against a chance of
0.333 — which is why every split above is blocked.*

Per-class recall at coverage 1.00, champion linear: Tanzania healthy 0.9999 · rust 0.9766 ·
anthracnose 0.9939; Makerere healthy 0.9952 · rust 0.9871. The rarest trained class is not the
worst-recognised one in domain.

**Reading it.** N1 ≫ N2 is the shape H8 §6.7 predicted, and it is extreme: within a dataset,
across held-out districts and dates, both backbones with all three heads sit at **0.991–0.996**
macro-F1; moved to another dataset, the same heads run from **0.346** to **0.889**. The gap is
not evidence of a good model. It is the measurement the article is about.

The three frame directions do not behave alike, and the difference is not small:

- **Tanzania → Makerere: 0.346–0.366.** On two roughly balanced classes (healthy/rust) a
  constant answer scores about 0.33, so this is two to three points of macro-F1 above saying
  the same word every time. Trained on 78,454 Tanzanian frames, neither backbone transfers to
  Uganda in any useful sense.
- **Tanzania → iBean: 0.516–0.545.** Better, still near the floor.
- **Makerere + iBean → Tanzania: 0.718–0.889.** The same pair of datasets, the other way
  round, works — best with the linear head on the champion, 0.889.

So the *larger* training set is the *worse* teacher here, and transfer between the same two
datasets is strongly asymmetric. The plan's guard-rail was "if N1 is not ≫ N2, suspect
leakage"; the observed asymmetry asks the mirrored question, and §5 says what we have and have
not done to answer it.

### 2.2 The abstention curves

Selective risk (error rate on the rows the model **accepted**) with, in brackets, the share of
rows it **abstained** on. The coverage heading is the *declared* coverage — the quantile fitted
on the in-domain validation slice — not the coverage achieved on the test set.

**Champion — `dinov2_l14_reg`, 224, CLS**

| part | head | declared 1.00 | declared 0.95 | declared 0.90 | declared 0.80 |
|---|---|---|---|---|---|
| N1 Makerere | linear | 0.0086 | 0.0018 (4.9 %) | 0.0009 (6.9 %) | 0.0000 (13.9 %) |
| N1 Makerere | proto | 0.0066 | 0.0058 (3.6 %) | 0.0057 (5.6 %) | 0.0036 (10.8 %) |
| N1 Makerere | mix | 0.0059 | 0.0013 (4.5 %) | 0.0003 (6.2 %) | 0.0000 (11.5 %) |
| N1 Tanzania | linear | 0.0022 | 0.0014 (2.0 %) | 0.0010 (2.6 %) | 0.0008 (5.1 %) |
| N1 Tanzania | proto | 0.0017 | 0.0012 (2.2 %) | 0.0012 (3.1 %) | 0.0012 (5.4 %) |
| N1 Tanzania | mix | 0.0018 | 0.0009 (2.0 %) | 0.0008 (2.6 %) | 0.0007 (4.5 %) |
| N2 T→Makerere | linear | 0.4703 | 0.4479 (5.7 %) | 0.4002 (13.1 %) | 0.2376 (31.5 %) |
| N2 T→Makerere | proto | 0.4687 | 0.4689 (0.1 %) | 0.4690 (0.1 %) | 0.4692 (0.2 %) |
| N2 T→Makerere | mix | 0.4700 | 0.4126 (11.2 %) | 0.3433 (20.5 %) | 0.1900 (35.5 %) |
| N2 T→iBean | linear | 0.3898 | 0.2523 (32.1 %) | 0.1429 (47.4 %) | 0.0280 (69.3 %) |
| N2 T→iBean | proto | 0.4011 | 0.4011 (0.0 %) | 0.4011 (0.0 %) | 0.4013 (0.1 %) |
| N2 T→iBean | mix | 0.4007 | 0.2244 (34.9 %) | 0.1128 (50.1 %) | 0.0192 (70.0 %) |
| N2 M+iBean→T | linear | 0.0385 | 0.0346 (67.1 %) | 0.0202 (69.5 %) | 0.0102 (73.3 %) |
| N2 M+iBean→T | proto | 0.0997 | 0.1208 (66.3 %) | 0.1116 (70.5 %) | 0.0871 (78.5 %) |
| N2 M+iBean→T | mix | 0.0503 | 0.0521 (68.5 %) | 0.0323 (71.7 %) | 0.0158 (76.3 %) |

**Challenger — `dinov3_l16`, 224, CLS**

| part | head | declared 1.00 | declared 0.95 | declared 0.90 | declared 0.80 |
|---|---|---|---|---|---|
| N1 Makerere | linear | 0.0086 | 0.0023 (3.8 %) | 0.0005 (5.2 %) | 0.0003 (9.1 %) |
| N1 Tanzania | linear | 0.0033 | 0.0009 (2.5 %) | 0.0007 (4.0 %) | 0.0006 (7.8 %) |
| N2 T→Makerere | linear | 0.4714 | 0.4113 (11.6 %) | 0.3272 (22.7 %) | 0.1720 (37.4 %) |
| N2 T→Makerere | mix | 0.4768 | 0.3622 (18.4 %) | 0.2476 (30.8 %) | 0.1007 (42.7 %) |
| N2 T→iBean | linear | 0.4079 | 0.1541 (49.5 %) | 0.0728 (66.6 %) | 0.0062 (85.2 %) |
| N2 T→iBean | mix | 0.4049 | 0.1549 (52.5 %) | 0.0658 (68.3 %) | 0.0045 (87.8 %) |
| N2 M+iBean→T | linear | 0.0449 | 0.0267 (45.9 %) | 0.0188 (47.9 %) | 0.0098 (52.2 %) |
| N2 M+iBean→T | mix | 0.0780 | 0.0574 (47.8 %) | 0.0407 (51.2 %) | 0.0219 (57.6 %) |

**Reading them.** Three things.

1. **Abstention buys most where the model is worst, and it buys a lot.** Champion linear at
   declared coverage 0.90: **+7.01 pp** of selective accuracy on T→Makerere, **+24.69 pp** on
   T→iBean, **+1.83 pp** on M+iBean→T; mean **+11.18 pp** over the three directions. In domain
   it buys almost nothing, because there is almost nothing to buy (+0.77 and +0.12 pp).
2. **The declared coverage is an in-domain promise and nothing more.** At "0.90" the champion's
   linear head abstains on 6.9 % of Makerere frames in domain — and on 69.5 % of Tanzanian
   frames in its best cross-dataset direction. The distance gates, fitted at TPR 95 % on the
   training set's own validation rows, see most of the other dataset as unlike anything they
   were trained on. They are right, and the label "90 % coverage" is still misleading. The
   article must quote the **achieved** abstention rate beside every selective number.
3. **The prototype head's confidence is unusable as an abstention score out of domain.** On
   T→Makerere and T→iBean the champion's proto head abstains on 0.0–0.2 % of rows at every
   declared coverage and its selective risk does not move (−0.03 pp, +0.00 pp). It is confident
   and wrong. `mix`, whose probabilities are half the linear head's, keeps a usable score and
   abstains on 11–36 % of the same rows.

**Does abstention fall on the rarest class?** H8 §6.6 asks. In Tanzania at declared 0.90 it
falls hardest on **rust**, not on the rarest class: champion linear abstains on 10.7 % of rust
rows against 2.5 % of anthracnose and 2.0 % of healthy (challenger linear: 15.1 % / 6.1 % /
2.9 %). So the answer is "not the rarest, but a minority class" — the disease the demo most
needs to see is the one the model most often refuses to name.

### 2.3 N3 — the unknowns

Two held-out classes the heads never saw: angular leaf spot (`unknown_als`, 5,098 Makerere
rows) and white mould (`unknown_wm`, 684 SWM rows). AUROC needs no threshold and is quoted at
coverage 1.00; unknown recall is the share flagged at each declared coverage. Linear head,
224/CLS.

| trained on | held-out | backbone | AUROC conf | AUROC kNN | recall @0.95 | @0.90 | @0.80 |
|---|---|---|---|---|---|---|---|
| Tanzania | `unknown_als` | DINOv2 | 0.9377 | 0.4444 | 0.186 | 0.313 | 0.578 |
| Tanzania | `unknown_als` | DINOv3 | 0.9171 | 0.6901 | 0.330 | 0.485 | 0.683 |
| Tanzania | `unknown_wm` | DINOv2 | 0.9965 | 0.9816 | 0.904 | 0.963 | 0.996 |
| Tanzania | `unknown_wm` | DINOv3 | 0.9933 | 0.9990 | 0.987 | 0.993 | 0.998 |
| Makerere | `unknown_als` | DINOv2 | 0.7019 | 0.6227 | 0.157 | 0.248 | 0.395 |
| Makerere | `unknown_als` | DINOv3 | 0.6134 | 0.6565 | 0.137 | 0.186 | 0.281 |
| Makerere | `unknown_wm` | DINOv2 | 0.6203 | 0.9972 | 0.992 | 0.994 | 0.998 |
| Makerere | `unknown_wm` | DINOv3 | 0.8024 | 0.9996 | 1.000 | 1.000 | 1.000 |
| Makerere+iBean | `unknown_als` | DINOv2 | 0.7007 | 0.5927 | 0.142 | 0.229 | 0.364 |
| Makerere+iBean | `unknown_als` | DINOv3 | 0.6253 | 0.6408 | 0.148 | 0.195 | 0.291 |
| Makerere+iBean | `unknown_wm` | DINOv2 | 0.6980 | 0.9960 | 0.985 | 0.993 | 0.999 |
| Makerere+iBean | `unknown_wm` | DINOv3 | 0.8654 | 0.9996 | 1.000 | 1.000 | 1.000 |

**Reading it.** The two unknowns are not one problem.

- **White mould is caught**: 0.96–1.00 recall at declared 0.90 with the linear head, AUROC
  0.98–1.00 on the kNN distance. It is another crop, another organ, another continent's photographs; the distance
  score sees it immediately.
- **Angular leaf spot is mostly missed**: 0.19–0.49 recall at declared 0.90. It is a bean leaf
  disease photographed exactly like the trained ones, and a Makerere-trained head is looking at
  *its own dataset's* held-out class — the distance score has nothing to go on (AUROC 0.59–0.66)
  and the confidence score is barely better (0.61–0.70).
- The one place ALS is caught reasonably (AUROC 0.92–0.94, recall 0.31–0.48 at 0.90) is the
  Tanzania-trained head, where ALS arrives as part of a whole other dataset. That is the
  acquisition site being detected, not the disease — the same signal N4 measures.
- **The kNN score is a dataset detector, not an unknown detector.** Its AUROC is identical
  across all three heads of a backbone (it does not use the head), and on
  Tanzania→`unknown_als` for DINOv2 it is **0.4444** — worse than chance.

### 2.4 N4 — the site-prediction probe

Logistic regression (C = 1) on standardised L2-normalised CLS, 224 px, intervals bootstrapped
over rows (1,000 draws), not over seeds.

| probe | DINOv2 | DINOv3 | chance |
|---|---|---|---|
| dataset (3-way), fitted on train splits (81,125 rows), scored on test (23,246) | **0.9826** [0.9797, 0.9857] | **0.9924** [0.9903, 0.9943] | 0.333 |
| Makerere district (12-way), 10,118 rows with a district, 5-fold CV grouped by pHash | **0.7675** [0.7490, 0.7870] | **0.7701** [0.7457, 0.7948] | 0.083 |

Per-dataset recall (DINOv2): iBean 1.000, Makerere 0.954, Tanzania 0.994. Per-district recall
runs from 0.42 (Mbale) to 0.94 (Hoima).

A high N4 is the expected result (medical models score 88–98 % on the acquisition site) and is
the *reason* the splits are blocked, not a failure of the pipeline. What it establishes for the
article: the frozen features carry "which dataset, which district, which camera" at near-ceiling
accuracy, **nine times the chance rate**, and any unblocked number on these sets is measuring
that.

### 2.5 N6 — latency

`ms.service.bench` times the service's own functions (no HTTP), on 256 Makerere test frames,
GPU idle, same frames for every model. Median per frame, [5th, 95th] percentile over calls.

| model | batch 1, live | batch 32, live | batch 1, cached | batch 32, cached |
|---|---|---|---|---|
| DINOv2 @ 224 | 115.8 ms | 31.1 ms | 43.3 ms | 3.83 ms |
| DINOv2 @ 518 | 137.5 ms | 81.9 ms | 45.7 ms | 3.74 ms |
| DINOv3 @ 224 | 134.0 ms | 31.4 ms | 47.6 ms | 3.91 ms |
| DINOv3 @ 512 | 136.1 ms | 76.9 ms | 44.9 ms | 4.18 ms |

**At batch 1 the abstention scoring is most of the cost, not the backbone**: the replay path,
with no backbone at all, costs 43–48 ms, so a fixed ≈ 45 ms of kNN and Mahalanobis sits under
every number and the four models land within 20 ms of each other although their resolutions
differ fivefold. At batch 32 that fixed cost amortises to 3.7–4.2 ms and the backbone shows
through. A replayed frame costs eight times less than a live one and needs no GPU, which is why
the demo does not depend on one.

These are **the service's** milliseconds — preprocessing, backbone, head and both abstention
scores — and are not comparable with a published forward-pass time.

### 2.6 N7 — compute

From `make compute-log-summary`, over the log's own rows.

| step | rows | wall-clock | where |
|---|---|---|---|
| `extract` (feature caches) | 20 | 20,314.6 s (5.64 h) | GPU |
| `serve` (the N6 benchmark) | 1 | 275.0 s | GPU |
| `train_head` | 422 | 1,598.8 s | CPU |
| `fit_abstain` | 362 | 22.7 s | CPU |
| `eval` (scoring the table) | 12 | 4,066.5 s | CPU |

**7.30 h of logged compute: 5.72 GPU-hours and 1.58 CPU-hours**, on one NVIDIA RTX 5060 Laptop
(8 GB, driver 592.15, CUDA 13.1, torch 2.11.0+cu128, Python 3.11.14) — the machine the log's
`env` row describes. 626,304 images passed through a backbone (156,576 manifest rows × 2
backbones × 2 resolutions), at 1,015.7 s per cache on average.

`make compute-log-check` confirms the ledger is complete: every cache on disk and every head
run has its row (20/20 extractions, 361 head runs in 422 rows, 361 abstention fits in 362).

### 2.7 Two ablations

Each changes one thing and keeps the rest. Mean change in macro-F1 over the quotable five-seed
aggregates, in percentage points.

| ablation | N1 | N2 |
|---|---|---|
| CLS ⊕ mean-patch instead of CLS, at 224 | +0.03 pp (−0.26 … +0.36), n=12 | **−3.53 pp** (−14.00 … +5.04), n=18 |
| 518 / 512 px instead of 224, on CLS | +0.22 pp (−0.03 … +0.65), n=12 | +0.79 pp (−10.75 … +8.91), n=18 |

Neither helps the number that matters. The richer fingerprint **hurts** cross-dataset transfer
on average, and its losses are all on Tanzania → iBean (−11.7 to −14.0 pp for the proto and mix
heads of both backbones) while its gains are all on Makerere + iBean → Tanzania (up to
+5.04 pp) — it appears to help when the training set already spans two datasets and to hurt
when it is one. Five times the pixels and about five times the GPU time move N1 by a fifth of a
point and N2 by less than one, with a range that crosses zero in both directions.

### 2.8 The verdict

Computed by `ms.eval.verdict` from the table, by criteria fixed before any number existed
(H9 §7 → spec 005 US-6), over the 306 quotable, current five-seed aggregates at 224/CLS,
coverage 1.00. Not edited by hand.

> **`dinov2_l14_reg` wins**: neither backbone won a criterion, and anything else goes to the
> champion.

- **Criterion 1 (N2, ≥ 2 pp with each head and non-overlapping intervals in two of three
  directions):** the challenger is behind by 1.51 pp (linear), 0.96 pp (proto) and 2.31 pp
  (mix) on the three-direction mean, and **no** direction has non-overlapping intervals.
  Neither wins.
- **Criterion 2 (N3, ≥ 0.02 AUROC everywhere):** the challenger clears the bar in 14 of 36
  comparisons and the champion in 6 of 36; a win needs all 36. Neither wins.
- **N6 is reported, not counted** (H9 §7 criterion 4), and the measurement bears that out:
  0.3 ms apart at 224.
- **Licence is a hard gate** regardless of the numbers: DINOv3 is research-only until H6 C5 is
  signed.
- A winning challenger would **join** the model set behind the interface, never evict the
  champion (D-7).

**The honest reading is a tie.** Neither backbone won anything; the champion keeps the seat by
a pre-declared default, not by a measured margin. That is a result, and the article reports it
as one.

---

## 3. The four claims of H8 §6.1

### Claim A — *frozen features + a small head under dataset shift, and what abstention buys*

> Under dataset shift between the open bean sets, frozen features + a small head give
> **[N1 → N2]** macro-F1, and abstention buys **[x] pp** of selective accuracy at **[y] %**
> coverage — the first blocked cross-dataset bean numbers.

**Measured.** Champion (DINOv2 ViT-L/14-reg, 224, CLS, linear head, five seeds):

- **N1 → N2: 0.991–0.994 → 0.362 / 0.545 / 0.889** macro-F1, the three frame directions being
  Tanzania → Makerere, Tanzania → iBean and Makerere + iBean → Tanzania. Across all three heads
  and both backbones: N1 0.991–0.996, N2 0.346–0.889.
- **Abstention buys +11.18 pp** of selective accuracy on average over the three directions at a
  **declared** coverage of 0.90 (+7.01, +24.69, +1.83 pp), while abstaining on 13.1 %, 47.4 %
  and 69.5 % of rows respectively. At declared 0.80 the mean gain is larger still and the
  abstention rates reach 31–73 %.

**The qualification that belongs in the sentence**: "at 90 % coverage" is the coverage declared
on the in-domain validation slice. The achieved coverage out of domain is 30–87 %. The article
quotes both numbers or neither.

**Blank inside claim A**: the served demo model has no N1 at all, because the harness writes N1
only for a run trained on one manifest and it trains on two. That is a gap in the harness, not
a number that came out badly.

**"The first" is H8's word, not a measurement.** Nothing in this note establishes that these
are the first blocked cross-dataset bean numbers; that is a literature claim and belongs to
A1's related-work section, checked there.

### Claim B — *how much of the signal is the acquisition site*

> A large share of the open sets' signal is the acquisition site: a probe predicts
> dataset/district from frozen features at **[z] %** (chance 33 %), and colour normalisation
> moves it by **[w] pp** and the cross-dataset number by **[v] pp**.

**Half measured.**

- **[z] = 98.3 %** (DINOv2) and **99.2 %** (DINOv3) balanced accuracy on the 3-way dataset
  probe, against a chance of 33.3 %; **76.8 %** and **77.0 %** on the 12-way Makerere district
  probe, against a chance of 8.3 %.
- **[w] — blank.** The colour-normalisation ablation is **piece 5's** (H8 §6.7, N4 row) and has
  not been run. No number exists.
- **[v] — blank**, for the same reason.

The first half of claim B is measured and strong. The second half — that the site signal is
partly colour and can be partly removed — is unmeasured, and the article may not imply it.

### Claim C — *the two backbones, head to head*

> DINOv2 and DINOv3 compared head-to-head on bean, blocked, five seeds, criteria declared
> before the numbers, licences read from the primary texts.

**Measured, and the answer is a tie.** 54 configurations × 5 seeds, both backbones at 224 and
at 518/512, three heads, three training sets, blocked splits throughout, criteria written into
spec 005 before the first head was trained, verdict computed by code (§2.8). Neither backbone
won either criterion; the champion keeps the seat by the pre-declared default.

Licences, from the primary texts: DINOv2 **Apache-2.0**; DINOv3 **DINOv3 License (Meta,
2025-08-14), gated**, accepted 2026-09-18 on the ML role's own account, research-only until
H6 C5 is signed.

**Partly open, and it is a bookkeeping gap, not a measurement one**: H8 §6.11 item 4 asks for
the licence status of each backbone *in the card*. Only the champion has a card, because
nothing registers a model that did not win. The challenger's licence status is stated here and
in `configs/backbones/dinov3_l16.yaml`; whether it also needs a card of its own is the owner's
call.

### Claim D — *what the whole study cost*

> The whole study — two backbones × three heads × two resolutions × five seeds × **[n]**
> transfer directions — costs **[h]** GPU-hours on one consumer card: extraction once, every
> head in seconds.

**Measured.**

- **[n] = 4** transfer directions over three ordered dataset pairs (Tanzania → Makerere frames,
  Tanzania → Makerere crops, Tanzania → iBean, Makerere + iBean → Tanzania).
- **[h] = 5.72 GPU-hours** (7.30 h of logged compute in all), on one **NVIDIA RTX 5060 Laptop,
  8 GB**.
- The grid is larger than the claim's sketch: 2 backbones × 3 training sets × 3 feature settings
  × 3 heads × 5 seeds = **270 head runs**, 626,304 images through a backbone.
- **Extraction is the entire GPU cost** — 20,314.6 s of the 20,589.6 GPU seconds, 98.7 % —
  and **every head afterwards trains in seconds**: 3.79 s per run over 422 runs, on the CPU.

**One correction the claim needs.** The plan's sentence has two costs, extraction and heads.
There is a third: **scoring the table costs 4,066.5 s**, more than twice the head training,
because every run's kNN score queries a bank of up to 78k training features. The article's
sentence should be *extraction is the GPU cost, heads are free, and scoring is what you pay for
abstention*.

**Blank inside claim D**: the optional laptop-CPU head run for a "runs on a laptop" sentence
has not been timed as a separate, declared measurement — although every head run in this study
already ran on a CPU, which may make the measurement unnecessary rather than missing.

---

## 4. Three sentences of what surprised us

1. **Transfer between the same two datasets is strongly asymmetric, and the bigger training set
   is the worse teacher**: 78,454 Tanzanian frames give 0.362 macro-F1 on Makerere, while 7,422
   Makerere + iBean frames give 0.889 on Tanzania.
2. **A declared coverage is an in-domain promise that does not survive the shift**: at "90 %
   coverage" the champion's linear head abstains on 6.9 % of in-domain frames and on 69.5 % of
   the frames in its best cross-dataset direction — the distance gates, fitted at TPR 95 % on
   the training set's own validation rows, correctly see most of another dataset as unfamiliar,
   and the label becomes meaningless.
3. **Temperature scaling fixes calibration exactly where it was already fine and does nothing
   where it matters**: in-domain ECE is 0.0008–0.0118, and on the two hard cross-dataset
   directions it is 0.327–0.472, so an honest "80 % sure" in domain is a meaningless "80 %
   sure" on another farm.

---

## 5. What the numbers do not show

- **Poland.** No Polish field, no Polish cultivar, no 'Złota Saxa', no Polish imagery of any
  kind. Every number is Africa → Africa. The Africa → Africa N2 drop is a *lower bound* on the
  Africa → Poland drop, not an estimate of it.
- **The robot's view.** Every image is a hand-held close-up of a leaf. Nothing here measures the
  camera height, angle, motion blur, exposure or field of view a robot produces, and the
  distance gates that fire on 70 % of another *dataset* have never seen a robot frame.
- **Panels and pods.** Organ-level leaf classification only. No whole-plant panels, no pods, no
  stems.
- **Deficiencies.** No deficiency classes: no image data exist for them (E1 §3).
- **Localisation and severity.** No lesion boxes, no masks, no severity. The Makerere crop
  direction uses the dataset's own boxes as *inputs*; nothing predicts a box.
- **Anthracnose outside Tanzania.** Anthracnose exists in one source, so no cross-dataset number
  covers it, and the model the demo serves has two classes and cannot say the word.
- **Whether the transfer asymmetry is leakage.** Duplicates were removed by sha256 and grouped
  by pHash *within* each manifest; a systematic near-duplicate search *between* Makerere/iBean
  and Tanzania has not been run. A 0.889 in the direction that works deserves that check before
  it is quoted as a headline.
- **The negative control (N5) and the colour-normalisation ablation.** Both are piece 5's and
  neither has been run, so "frozen features beat a fine-tuned ImageNet CNN" is not a claim this
  note supports either way.
- **One machine.** N6 and N7 are one RTX 5060 Laptop on one evening with the GPU idle. No
  multi-machine, multi-run or thermal variation is in those intervals.
- **N4's intervals are not seed intervals.** They bootstrap over rows within cells, with one
  seed; every other interval here is over five seeds.
- **iBean's own accuracy.** iBean is test-only and unblocked (`unblocked:random_by_phash_group`);
  its within-dataset number exists in the table and is never quoted.
- **An end-to-end mission.** See blank 1 below.

---

## 6. The blanks, in one list

Each of these is a place where a number is expected and does not exist. None of them is
reported above as a claim.

1. **The end-to-end virtual mission — H8 §6.11 item 1 — is NOT done.** The service answers a
   frame from the cache and from its bytes, `abstain` is reachable with both reasons, and a
   broken frame gets `invalid_input` as a 422. But **no mission has been scored end to end**,
   and a real bag frame has never reached the service: `mcap://` returns 501. The reason is
   outside piece 4 — **piece 2 does not exist yet** (`emulator/` holds a README and nothing
   else), so there is no mission, no frame topic name, no ROS 2 message type, and the MCAP
   `uri` forms are still open in piece 1. The prediction log
   (`data/predictions/requests.jsonl`) therefore has **zero rows**. This is a blank, not a
   partial success, and the article may not say the system was demonstrated end to end.
2. **N5, the negative control** (ImageNet CNN fine-tuned on Tanzania, tested as N2) — piece 5.
3. **The colour-normalisation ablation** and its effect on N4 and on N2 — piece 5. Claim B's
   second half depends on it.
4. **N1 for any head trained on more than one manifest**, which includes the model the service
   serves. A harness gap in spec 005, the owner's to settle.
5. **A card for the challenger**, or another home for DINOv3's licence status, so that §6.11
   item 4 reads as written rather than as interpreted.
6. **A between-dataset near-duplicate search** (§5), which the asymmetry of §2.1 makes worth
   doing before the headline number is quoted.
7. **A declared laptop-CPU head-run timing**, if the article wants the "it runs on a laptop"
   sentence as a measurement rather than as an inference from the CPU runs already logged.
8. **ACRE false-positive rate on European robot-view healthy frames** and the **CLIP-fusion
   arm** — H8 stretch items, deliberately not attempted in W6.

---

## 7. Where the artefacts are

| what | where |
|---|---|
| the N-table | `model_service/results/n_table.jsonl`, rendered `n_table.md` |
| the verdict | `model_service/results/verdict.md` (written by code) |
| the compute log | `model_service/results/compute_log.jsonl` (`make compute-log-check`, `make compute-log-summary`) |
| the model card | `model_service/cards/msv0.1+dinov2_l14_reg@224.linear.man-0670c6-423a16.md` |
| the frozen manifests | `data/manifests/*.jsonl` + `FROZEN.jsonl` (DVC) |
| the prediction log | `data/predictions/requests.jsonl` (schema exists; no rows yet — blank 1) |
| the decisions behind every choice | `model_service/DECISIONS.md` |
