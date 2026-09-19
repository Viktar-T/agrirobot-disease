# Piece 4 — model service v0: actionable work plan (W1–W6)

Executes **H8 §6.10** ("Work plan — the coming month, week by week") and is accepted by **H8 §6.11**. The plan is H8; this file is the execution: tasks, commands, tests, files. Owner **M** (ML role), with **P** for data in W1–W2. Last updated 2026-09-16 (Wed, mid-W1). Update the checkboxes weekly; when a task changes meaning, change H8 first.

Method: spec-driven — each contract below gets a one-page spec (`model_service/specs/NNN-<name>/spec.md`) and a failing test before implementation; a *spec* is a contract + acceptance tests + a protocol, never an expected number. Build a **vertical slice through iBean first** (1,296 images, MIT), then widen along one axis at a time (dataset → backbone → resolution → head → abstention → probe). Every artefact is keyed by hashes so that re-running is a no-op and adding a backbone is a config entry.

---

## 0. Where things live

```
model_service/
  specs/            one folder per spec: 001-manifests/, 002-cache/, 003-heads/, 004-abstention/, 005-results-table/, 006-service/ — each with spec.md (+ plan.md, tasks.md when written)
  src/ms/           package: data/, cache/, heads/, abstain/, eval/, service/
  configs/          backbones/*.yaml, datasets/*.yaml (downloads), manifests/*.yaml (manifest recipes), class_map_v1.yaml, heads/*.yaml, eval.yaml
  tests/            pytest; fixtures/ibean_30/ (30 images, MIT, and their manifest ibean_v1.jsonl — attribution in its README)
  results/          n_table.jsonl, compute_log.jsonl, verdict.md  (small; tracked in git)
  cards/            model cards (markdown), one per registered model
  DECISIONS.md      dated choices, in the style of interface/README.md "choices the draft makes"
data/                        (git-ignored; DVC from W2 — piece 3)
  raw/<dataset>/             as downloaded, untouched
  manifests/<manifest>_v<N>.jsonl + .meta.json; FROZEN.jsonl      (spec 001)
  cache/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.npz + .meta.json   (spec 002)
  bags/                      piece 2
mlruns/                      local MLflow (git-ignored) until piece 3 runs the server
```

Make targets (all idempotent): `make env`, `make download DS=<name>`, `make manifests DS=<name>`, `make cache BB=<backbone_id> RES=224|518|512 DS=<name>`, `make heads`, `make eval`, `make probe`, `make serve`, `make test`.

---

## 1. The six contracts (specs)

| Spec | Artefact | Invariants (tests fail until true) | H8 | Week |
|---|---|---|---|---|
| **S4.1 Manifests** | `data/manifests/<dataset>_<ver>.jsonl` — one row per image: `image_id, sha256, path, dataset, source_record, class_raw, class_km2, group_keys{district, subcounty, date, region, session, phash_group}, split, split_rule, licence, attribution` | every row has a sha256 that matches the file; duplicates (exact or phash) share a `phash_group` and never straddle splits; `class_km2 ∈ {healthy, rust, anthracnose, unknown_als, unknown_wm}` and the two `unknown_*` classes never appear in a train split; `split_rule` is a non-empty string; the test manifest hash is recorded before any head is trained | §6.3 | W1–W2 |
| **S4.2 Feature cache** | `data/cache/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.npz` (`cls[N,D]` and `meanpatch[N,D]` float16, `meanpatch` averaging the patch tokens only; `image_id[N]`, `sha256[N]`) + `<manifest>_v<N>.meta.json` (the key fields and `cache_key`, `token_types`, `manifest_sha256`, `transformers_version`, `torch_version`, `device`, `batch_size`, `wallclock_s`, `images_per_s`; full list in spec 002 FR-006) | cache key = (backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype), part of the path; a changed key yields a new cache and leaves the old one; shapes match the sidecar; re-running with the same key and manifest does nothing; runs resume from shards; every run writes its `wallclock_s` to `results/compute_log.jsonl` (N7) | §6.4 | W2–W3 |
| **S4.3 Heads** | `ms.heads.train --backbone --res --head {linear,proto,mix} --seed --train-manifest --val-manifest` → `heads/<run_id>/head.pt + run.json` | same seed → identical weights; class-balanced sampling on by default; early stopping reads the in-domain validation slice only; seconds per run logged to the compute log | §6.5 | W3 |
| **S4.4 Abstention + calibration** | `ms.abstain.fit` → thresholds `{tau_conf, tau_knn, tau_maha}` at declared coverages {0.80, 0.90, 0.95}; temperature `T` | thresholds and `T` are fitted on the in-domain validation slice and never touch a test manifest; `decision == "abstain"` is reachable on held-out unknowns; per-class abstention rate is reported | §6.6 | W4 |
| **S4.5 Results table** | `results/n_table.jsonl` — one row per `(backbone_id, res, token_type, head, seed, train_manifest, test_manifest, number ∈ {N1,N2,N3,N4,N5,N6,N7}, metric, value, ci_low, ci_high, split_rule, coverage)` | every row carries `split_rule` and the two manifest hashes; N1 and N2 exist as a pair for every (backbone, head, res); N3 rows use only held-out classes; the champion/challenger verdict is computed from this file by code, not by hand | §6.7 | W3–W4 |
| **S4.6 Service** | OpenAPI generated from the request/response schemas; endpoints `POST /v1/predict`, `POST /v1/predict_batch`, `GET /v1/model`, `GET /v1/health`; request = `observation_frame` record (interface v0) + `request_id` + `options` | broken frame → HTTP 422 with the validator's reason; `abstain` response with `abstain_reason`; batch endpoint; a frame whose sha256 is in the cache is answered without a GPU; `model_version` string reproduces the run | §6.2, §6.9 | W5 |

The model card (§6.8) is a template (`cards/TEMPLATE.md`), not a spec. The ROS 2 client (§6.9) is a thin node outside the service; it is built on the Linux/WSL side where the emulator runs (piece 2), not on the Windows dev box.

---

## 2. Week by week

### W1 — Wed 16 · Thu 17 · Fri 18 Sep (three days left)

Goal: environment, data on disk, the first two specs, the fixture. Nothing trained yet.

- [x] **Environment** (M, 2 h): `pyproject.toml` (Python 3.11; torch, torchvision, `transformers>=4.56`, numpy, pandas, pyarrow, scikit-learn, faiss-cpu or sklearn NearestNeighbors, mlflow, dvc, fastapi, uvicorn, pydantic, jsonschema, pytest, ruff); `make env`; `.env` with `HF_TOKEN` (git-ignored). Record GPU model / VRAM / driver in `results/compute_log.jsonl` as the first entry (N7).
- [ ] **DINOv3 terms** (M, 15 min): accept on the ML role's own Hugging Face account; write the date and the account into `DECISIONS.md` and later into the card. Research-only until H6 C5 is signed; nothing containing DINOv3 leaves the university machines.
- [ ] **Downloads** (P, background): iBean and Makerere first (small; today), Tanzania #1 (Zenodo 15685315) tonight, Tanzania #2 (Zenodo 8286126) and #3 (Zenodo 16751336, 47 GB — only if hashes show it adds images), SWM white mould (E1 #8). Into `data/raw/<dataset>/` untouched. `make download DS=...` writes `data/raw/<dataset>/DOWNLOAD.json` (URL, date, archive sha256, licence text).
- [x] **Spec S4.1** (M, 3 h) + `tests/test_manifests.py` failing: field list above; the class map (`class_raw → class_km2`); grouping keys per dataset (Makerere: district → sub-county → date; Tanzania: region/session folders or dates if present, else `phash_group` + `split_rule = "unblocked"`; iBean: test-only, `unblocked`). — *Done 2026-09-19: spec 001 clarified against the data on disk (EXIF and XML scans); `configs/class_map_v1.yaml`; recipes in `configs/manifests/`. Tanzania is one manifest over tz155k and tz59k, `blocked:date` from EXIF. Makerere is `blocked:district`, with healthy images placed by the file-name date. iBean is test-only and unblocked. SWM is the R-SWM originals. `tests/test_manifests.py`: 35 red until W2, 2 green. Choices are in DECISIONS 14–19.*
- [x] **Spec S4.2** (M, 2 h) + `tests/test_cache.py` failing: cache key, `meta.json` fields, the two token types, the compute-log row. — *Done 2026-09-19: `specs/002-cache/spec.md`. The key is the plan's four fields + `compute_dtype`, and it sits in the cache path. `cls` = first token; `meanpatch` = patch tokens only, leaving out the 4 registers. The sidecar fields are listed in FR-006, and each extraction run writes a compute-log row with a `cache` object. Runs resume from shards. `tests/test_cache.py`: 13 red until W2, 1 green, and the golden test is opt-in. They run on CPU on the fixture with a tiny random-init backbone. Choices are in DECISIONS 22–27.*
- [x] **Fixture** (M, 1 h): `tests/fixtures/ibean_30/` — 30 iBean images (10 per class) + `LICENSE-MIT` + `manifest.jsonl`; CI runs on CPU on this fixture only. — *Done 2026-09-19. Laid out like `data/raw/` for spec 001 US-1.1: `ibean/{DOWNLOAD.json, SHA256SUMS, extracted/<split>/<class>/}`, with 6/2/2 images per class from the source's train/validation/test, byte-for-byte copies, 4.0 MB. Attribution is in `README.md`. The manifest is `ibean_v1.jsonl` + `.meta.json`, the names spec 001 gives the builder's output. `tests/fixtures/make_ibean_30.py` writes it until the W2 builder does, and a test checks the two against each other. CPU-only and fixture-only are properties of the tests; the repository has no CI workflow yet (DECISIONS 20).*
- [x] **Backbone configs** (M, 30 min): `configs/backbones/dinov2_l14_reg.yaml` (`facebook/dinov2-with-registers-large`, Apache-2.0, patch 14, resolutions {224, 518}, ImageNet normalisation) and `dinov3_l16.yaml` (`facebook/dinov3-vitl16-pretrain-lvd1689m`, DINOv3 licence, gated, patch 16, resolutions {224, 512}). — *Done 2026-09-18, with pinned Hub revisions, fp16 + SDPA, batch 32, the two token types, and `make hub-check` (DECISIONS 11–13; `test_backbones.py`). Checked 2026-09-19 against the pinned snapshots' `config.json`: patch 14 / 16 and width 1024 both match, both have 4 register tokens, and each ships one `model.safetensors`.*
- [x] **Interface hand-shake** (M + P, 30 min): confirm with piece 1 that the service request = `observation_frame` + `request_id` + `options`; adopt their answers to the open questions that touch us (`frame_uid` rename; `bbch_null_reason` enum; radians for yaw). Note in `DECISIONS.md`. — *Done 2026-09-19, confirmed by Viktar Taustyka acting for piece 1 (P). Confirmed as drafted: the request shape, radians for yaw and the `bbch_null_reason` enum. The image id is renamed to `frame.frame_uid` in piece 1's schema, example and README. The frame `sha256` is aligned with our content hash. See DECISIONS 28–32. E and the robotics side still have to review piece 1's draft (piece 1's own "done when").*

Done when: `make test` runs (red for S4.1/S4.2); iBean and Makerere on disk with sha256 lists; the one-page data note (`data/README.md` extended: what is on disk, sizes, licences, what is missing).

### W2 — 21–25 Sep

Goal: the **iBean vertical slice** end to end, then manifests and 224-px caches for all sets; the first number (N4).

- [x] **Mon–Tue — the slice on iBean**: `make manifests` (iBean only) → `make cache BB=dinov2_l14_reg RES=224` (CPU is acceptable here: ⚠ ~30–40 min for 1,296 images) → linear probe (`ms.heads.train --head linear`, one seed) → one N1 row (`split_rule = "unblocked"`) → `results/n_table.jsonl` → `make serve` and `POST /v1/predict` with one cached image. This is the whole pipeline once, on the smallest data; everything after is widening. — *Done 2026-09-19, ahead of W2, on the GPU. `ibean_v1`: 1,295 rows (train 604, val 87, test 172, 432 held out), not frozen yet (Clarification 4). Cache DINOv2 @ 224, fp16: 1,295 images in 19.6 s. The linear head needs `--allow-test-only`, because iBean is test-only. N1 macro-F1 0.977, `unblocked:random_by_phash_group`, never quoted. `make serve` answered `POST /v1/predict` for a cached test image from the cache. The heads, eval and service code is ahead of specs 003, 005 and 006, and its choices are DECISIONS 33–41. Two findings for spec 003: macro-F1 on 87 validation images was too coarse for early stopping, and the linear recipe underfits iBean (3 steps per epoch). The fp16/fp32 check set the golden tolerance at 2e-6.*
- [x] **Wed — Makerere**: manifest with real grouping keys (district/sub-county/date from the Dataverse metadata; check completeness — H6 E4 lead); box-level crops (10 % margin) as a second manifest; the first *blocked* split; S4.1 tests green. — *Done 2026-09-19, ahead of W2; not frozen yet (that comes with the Tanzania task). `makerere_v1`: 15,402 rows in 11 district blocks (Bugiri+Mayuge merged), split by an exact search over whole blocks. Completeness, for P (H6 E4): district, date, variety and age are on every rust and ALS row; 30 Kiboga rows lack the sub-county; healthy rows carry only the file-name date. The class ↔ trip confound is structural (healthy only in April), and N1 must name it. `makerere_crops_v1`: 27,019 crops (10 % margin) in `data/derived/`, none of them healthy. 9,618 boxes on EXIF-turned images are left out, because the annotations mix frames there. S4.1: 36/36 green, and the crop tests too. See DECISIONS 42–46.*
- [x] **Wed–Thu — Tanzania**: exact-hash de-duplication across the three records; phash groups; the class map; manifests frozen; test manifests hashed and listed in `DECISIONS.md` ("frozen before any head"). — *Done 2026-09-19, ahead of W2. The tz3 test (`RUST_6.zip`, 0.91 GB) shows that tz3 adds images: 321 new rust photos, 309 of them undated. By the owner's decision, `tanzania_v1` is tz155k + tz59k, and the rest of tz3 (about 46 GB) is a later call. `tanzania_v1`: 112,176 rows, blocked by capture date, 0.70/0.10/0.20 per class. The pHash threshold is 2 bits, not 6: at 6, false matches chained 20 dates and left anthracnose 1 validation row (Clarification 4). Frozen and listed in DECISIONS 51: `ibean_v1`, `makerere_v1`, `makerere_crops_v1`, `swm_v1`, `tanzania_v1`, before any W3 head. See DECISIONS 47–51.*
- [x] **Thu–Fri — caches at 224** for both backbones on all sets (GPU); compute log rows for every extraction (N7). — *Done 2026-09-19, ahead of W2, on the GPU. All five frozen manifests have a 224 cache for both backbones: 156,576 images each, in 30 min (DINOv2, key `2b017210105b0b1f`) and 35 min (DINOv3, key `718a3399a7ad5b68`), 1.4 GB in all. Every extraction wrote its compute-log row, each cache's rows add up to its images, and a rerun does nothing (spec 002 SC-1 to SC-3). The GPU is the limit at about 90 images/s, and decoding is the limit on the large Makerere and SWM pictures. After 40 minutes the laptop lowered the GPU's power cap, so a night run should be planned at about 80 images/s. The shard and batch sizes stay (Clarification 8). `make cache DS=makerere_crops` now finds the crops manifest, which has no recipe. See DECISIONS 52–53.*
- [x] **Fri — site-prediction probe (N4)**: logistic regression on cached CLS features predicting the dataset (3-way; balanced accuracy vs chance 33 %) and, within Makerere, the district. One row per backbone in the N-table; logged in MLflow (`make probe`). — *Done 2026-09-19, ahead of W2. `ms.eval.probe` fits multinomial logistic regression on the standardised, L2-normalised CLS. Every (target, class) cell weighs the same, so a set's class mix cannot pass for the set. The dataset probe is fitted on the train splits' healthy and rust rows and scored on the test splits'. The district probe uses the 10,118 Makerere rows that record a district, with 5-fold CV grouped by phash_group, because the districts are the split's blocks. Balanced accuracy for the dataset: DINOv2 0.983, DINOv3 0.992 (chance 0.333). For the district: 0.767 and 0.770 (chance 0.083), with the misses between districts visited on the same days. The features carry the site, largely as the capture session. There is one N-table row per backbone and target (4 rows) and one MLflow run for each, in a local SQLite store (`mlruns/mlflow.db`), because MLflow 3.16 refuses the plain file store. See DECISIONS 54–56.*
- [x] Hand the manifest format to piece 3 for DVC (`dvc add data/manifests data/cache` — P). — *Done 2026-09-19, ahead of W2. DVC is initialised at the repository root with analytics off. `data/manifests.dvc` (the five frozen manifests, their sidecars and `FROZEN.jsonl`) and `data/cache.dvc` (the ten 224 caches) are in git. The DVC cache is local, and there is no remote yet: choosing one is piece 3's decision, and DINOv3 features may only go to a remote on the project's machines. The root `.gitignore` now ignores `/data/?*`, because DVC reads `/data/*` as ignoring `data/` itself and then finds no pointer file. The hand-over, with both formats and what stays open for piece 3, is a section of `store/README.md`. See DECISIONS 57.*

Done when: manifests for all sets in DVC; 224 caches for both backbones; N4 in MLflow and in `n_table.jsonl`; first N7 entries; S4.1 and S4.2 green.

### W3 — 28 Sep–2 Oct

Goal: heads and the first N1/N2 pairs at 224; the high-resolution caches running overnight.

- [x] **Spec S4.3 + S4.5** (Mon): head CLI, run.json, the N-table row schema; failing tests. — *Done 2026-09-19, ahead of W3. Spec 003 (`specs/003-heads/spec.md`): `ms.heads.train` takes several heads and seeds, and its defaults are the W3 protocol, linear, proto and mix for seeds 0–4. Every input is checked before the first run trains, and the validation split must be the training manifest's own. The mix is the fixed 0.5/0.5 mixture of the linear and proto runs of the same seed, not a third trained model. `run.json` v2 keeps every v1 key and adds `components` and the effective recipe. Spec 005 (`specs/005-results-table/spec.md`): rows are checked before they are appended. Five seeds give an aggregate row (`seed` null, the mean and a 95 % Student t interval). N1 is macro-F1 plus one recall per class, and code checks the N1/N2 pairs and N3's held-out classes. The verdict's rules copy this plan's summary of H9 §7, because H9 is not in the repository; what H9 §7 still has to settle before N2 is spec 005 Clarification 6. `tests/test_heads.py`: 21 red, 2 green. `tests/test_n_table.py`: 46 red. The slice's N1 test now expects spec 005's rows. Choices are in DECISIONS 58–66.*
- [x] **Heads** (Mon–Wed): linear probe (logistic regression on L2-normalised CLS), prototype head (K = 4 per class, cosine, τ init 0.07 learnable, per-class k-means init), fixed 0.5/0.5 mixture; AdamW lr 5e-4–1e-3, wd 1e-4, batch 256, focal γ = 2, label smoothing 0.1, 50 epochs, early stop on in-domain val; class-balanced sampling; **five seeds**; seconds per run to the compute log. — *Done 2026-09-19, ahead of W3. `ms.heads.train` trains linear, proto and mix for seeds 0–4 by default (spec 003). Spec 005's code came with it (row checks, five-seed aggregates, per-class N1 rows), so `test_heads.py` and `test_n_table.py` are green. 60 runs at 224, both backbones on `makerere_v1` and `tanzania_v1`, took 238 s of CPU in all, each run with its compute-log row, and a rerun does nothing. Running them showed that multi-threaded k-means gave proto another start from call to call. It now runs on one thread, and a second training in another process gives the same weights bit for bit. On Makerere (27 steps per epoch) the linear head is still improving when the 50 epochs end. H8's recipe stays, and a logit scale for that head is the owner's call. See DECISIONS 67–70. Later on 2026-09-19 the owner raised `max_epochs` to 300 for every head, with lr, patience and early stopping unchanged. All 90 runs at 224 were trained again in 495 s of CPU. Early stopping ended every one, the longest after 208 epochs. The 50-epoch rows stay in the N-table, marked superseded (DECISIONS 80).*
- [x] **N1** (Wed): within-dataset blocked (Makerere by district, `blocked:district`; Tanzania by capture date, `blocked:date`, as frozen in `tanzania_v1`); per-class recall incl. the rarest; abstention rate reported later (W4). — *Done 2026-09-19, ahead of W3. Both backbones at 224, linear, proto and mix, five seeds: 210 per-seed rows and 42 aggregates (mean and 95 % t interval) in `n_table.jsonl`, all quotable. Macro-F1 is 0.988–0.994 on Makerere and 0.992–0.996 on Tanzania. proto and mix are above linear everywhere, and the two backbones are within 0.004. The rarest class is not the weakest: Tanzania's anthracnose is recalled at 0.992–0.995 and its rust at 0.977–0.985. Makerere's test split carries the class ↔ trip confound: all its healthy rows come from April's Hoima, nearly all its rust rows from May. The 6 rust rows from Hoima itself are recalled 4–6 of 6. So N1 is high but not yet evidence across sites; that is N2's job. See DECISIONS 71–73. Scored again after the 300-epoch retrain: the linear heads on Makerere rose to 0.991 for both backbones, and Tanzania did not move (DECISIONS 81).*
- [x] **N2** (Thu): cross-dataset — Tanzania → Makerere (frame-level and crop-level; healthy/rust), Tanzania → iBean (healthy/rust), Makerere + iBean → Tanzania (healthy/rust); anthracnose only where both sides have it. Table rendered from `n_table.jsonl` by `make eval` (`results/n_table.md`). — *Done 2026-09-19, ahead of W3, with the owner's four decisions of the same day. Makerere + iBean train together and are quotable. The crop level reports rust recall only, because there are no healthy crops. Tanzania's three-class heads decide between healthy and rust. The targets are the frozen test splits, and every row of iBean. The directions are in `configs/eval.yaml`. Both backbones at 224, three heads, five seeds: 300 per-seed rows and 60 aggregates, all quotable. Macro-F1 is 0.35–0.37 for Tanzania → Makerere, where the Tanzania heads call Makerere's rust frames healthy (rust recall ≤ 0.02), and 0.52–0.55 for Tanzania → iBean. Makerere + iBean → Tanzania scores 0.72–0.89, and the rust crops are recalled at 0.32–0.49. See DECISIONS 74–76. Scored again after the 300-epoch retrain: only Makerere + iBean → Tanzania moved, most for DINOv3's linear head (0.802 → 0.876) and mix (0.782 → 0.813). The Tanzania → X directions are unchanged (DECISIONS 81).*
- [x] **High-res caches** (nightly from Tue): `RES=518` (DINOv2) / `RES=512` (DINOv3); ⚠ of the order of a GPU-day per backbone on a 3070-class card; log N7. — *Done 2026-09-19, ahead of W3, in one evening on the GPU instead of a night per backbone. The go/no-go of DECISIONS 11 was taken first, on the iBean timing at 518 (17.5 images/s, about 2.5 h per backbone projected), so every frozen manifest got its high-res cache, not a sample. Both backbones on all five: 156,576 images each, in 2 h 28 min (DINOv2 @ 518, key `2f7e632db9fe0d1f`, 17.6 images/s) and 2 h 6 min (DINOv3 @ 512, key `9f7ededb57be5fc9`, 20.8 images/s), one backbone after the other; 1.4 GB more in `data/cache/`, with its DVC pointer updated. Every extraction wrote its compute-log row (N7), each cache's rows add up to its images, every cache loads against its frozen manifest in row order, and a rerun does nothing (spec 002 SC-1 to SC-3). The GPU held about 77 W for the whole 4.6 h, so the rates never fell, and batch 32 with shards of 1,024 still fits in 4.2 GB of VRAM. See DECISIONS 83–84.*
- [x] Guard-rail check: if N1 ≫ N2 does not appear, look for leakage (duplicates across splits) before anything else. — *Done 2026-09-19, ahead of W3. N1 ≫ N2 appears for every backbone and head. Against the N1 of their own set, the N2 directions lose 0.10–0.65 macro-F1 (Tanzania → Makerere 0.63–0.65, Tanzania → iBean 0.45–0.48, Makerere + iBean → Tanzania 0.10–0.27). Every N1 interval lies above every N2 interval, so no leakage hunt was triggered. Checked anyway: no duplicate straddles the splits of any frozen manifest, and `overlap` finds no exact or near duplicate across the sets. See DECISIONS 77–78. Run again after the 300-epoch retrain: N1 ≫ N2 still holds everywhere, with gaps of 0.10–0.65 and every N1 interval above its N2 interval (DECISIONS 82).*

Done when: N1/N2 for both backbones at 224, five seeds, intervals; 518/512 caches complete; S4.3 and S4.5 green.

### W4 — 5–9 Oct

Goal: abstention, calibration, unknown recall; the full N-table; the champion/challenger verdict.

- [ ] **Spec S4.4** (Mon) + failing tests.
- [ ] **Scores** (Mon–Tue): max-logit / max-cosine (confidence); kNN distance to training features (k = 10–50, L2-normalised); relative Mahalanobis as second opinion; energy logged only. Thresholds at coverage 0.80 / 0.90 / 0.95 on in-domain validation; temperature scaling on the same slice; ECE in- and out-of-domain.
- [ ] **N3** (Tue–Wed): unknown recall and AUROC on held-out ALS and white mould; selective-risk-vs-coverage curves in- and out-of-domain; per-class abstention (does it fall on the rarest class?).
- [ ] **Repeat N1/N2 at 518/512 and with CLS ⊕ mean-patch** (Wed–Thu): the two ablations, five seeds.
- [ ] **Verdict** (Fri): `make eval` writes `results/verdict.md` from the N-table by the pre-declared rules (H9 §7: ≥ 2 pp on N2 with non-overlapping five-seed intervals on two of three directions; ≥ 0.02 AUROC on N3; N4/N6 tie-breakers; licence = hard gate). No hand edits to the verdict.
- [ ] Optional if the week has slack: conformal sets (RAPS) at α = 0.10.

Done when: the full N-table (N1–N4, N7) for both backbones; `verdict.md`; S4.4 green.

### W5 — 12–16 Oct

Goal: the service, the registered model, the card; one virtual mission scored end to end.

- [ ] **Spec S4.6** (Mon) + failing tests (422, abstain, batch, cached-hash path, `model_version`).
- [ ] **Service** (Mon–Wed): FastAPI; request = interface v0 frame record + `request_id` + `options`; response per H8 §6.2; `model_version = "msv0.1+<backbone_id>@<res>.<head_id>.man-<train_manifest_hash[:6]>"`; JSONL log of every request/response (piece 3 will move it into the predictions table).
- [ ] **Registry + card** (Wed): MLflow run per head; register only with a card (`cards/<model_version>.md` from `cards/TEMPLATE.md`: intended use, data + licences + attribution lines, the N-table, operating point, known failure modes, backbone licence and — for DINOv3 — the acceptance date and "research-only until C5").
- [ ] **N6** (Thu): latency at batch 1 and 32, both backbones, both resolutions.
- [ ] **ROS 2 client** (Thu–Fri, with E): thin rclpy node subscribing to the frame topic, calling the service, publishing `/apv/disease/scores`; one virtual mission from the emulator scored end to end.

Done when: a registered model with a card answers the interface; N6 in the N-table; S4.6 green; H8 §6.11 items 1, 2, 4 met.

### W6 — 19–23 Oct

Goal: buffer, the evaluation note as the article skeleton, hand-over.

- [ ] Fix what W5 broke; close the open items of the compute log (every extraction and every head run has a row).
- [ ] **Evaluation note** = `measurements/A1_skeleton.md`: methods (data, splits, backbones, heads, abstention, protocol), the N-table, the abstention curves, claims A–D of H8 §6.1 each with a measured number or an honest blank, three sentences of what surprised us, the list of what the numbers do not show.
- [ ] **Hand-over**: piece 5 reads `n_table.jsonl` (negative control N5 and the colour-normalisation ablation are theirs); piece 6 reads the prediction log / predictions table.
- [ ] Stretch, only if everything above is done: ACRE false-positive rate on European robot-view healthy frames; CLIP-fusion arm.

Done when: H8 §6.11 items 3, 5, 6 met; the note filed; piece 6 can build the map.

---

## 3. The vertical slice (W2, Monday–Tuesday), as commands

```
make download DS=ibean
make manifests DS=ibean                     # S4.1: data/manifests/ibean_v1.jsonl, split_rule=unblocked
make cache BB=dinov2_l14_reg RES=224 DS=ibean   # S4.2: cache + meta.json + compute_log row
python -m ms.heads.train --backbone dinov2_l14_reg --res 224 --head linear --seed 0 \
       --train-manifest data/manifests/ibean_v1.jsonl --split train --val-split val \
       --allow-test-only                     # iBean's role is test_only (DECISIONS 35)
make eval                                    # one N1 row -> results/n_table.jsonl + n_table.md
make serve                                   # then: POST /v1/predict with one cached image
                                             # (body: python -m ms.service.example)
```

Expected: an afternoon if the GPU is there, a day on CPU. What it proves: the contracts hold end to end on real bean images before any scale is added. What it does not prove: anything about accuracy — iBean is unblocked and tiny; its number is never quoted.

---

## 4. Conventions

- **Backbone ids**: `dinov2_l14_reg`, `dinov3_l16`, optional `dinov3_b16`; a backbone is a YAML file, nothing else changes.
- **Cache key**: `(backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype)`, and it is part of the cache path (spec 002; DECISIONS 11, 22–23); `preprocess_string` = e.g. `resize_short=518;center_crop=518;norm=imagenet`.
- **Resolutions**: 224 for the fast pass (DINOv2 16×16 patches, DINOv3 14×14); 518 (DINOv2, 37×37) and 512 (DINOv3, 32×32) for the high-res pass. Sizes are multiples of the patch size.
- **Tokens stored**: `cls` and `meanpatch`, float16; full patch tokens only for a 2,000-image subset (Makerere crops + SWM) in a separate file.
- **Seeds**: 0–4; every reported number is mean ± 95 % interval over the five.
- **Class map**: `healthy, rust, anthracnose` are the KM2 classes with data; `unknown_als` and `unknown_wm` are held-out unknowns (never trained on); `unknown` is a decision in the service, not a class.
- **Compute log** (`results/compute_log.jsonl`): `{ts, step ∈ {extract, train_head, eval, serve}, backbone_id, res, dataset, n_images_or_runs, wallclock_s, device, vram_gb, notes}` — written by code, never by hand; one optional row from a laptop CPU head run if a "laptop" sentence is ever wanted for the article.
- **Nothing in specs states an expected number.** Criteria that decide something (the verdict) are written before the numbers exist, in `model_service/specs/005-results-table/spec.md`, copied from H9 §7.

## 5. Definition of done (H8 §6.11, as a checklist)

- [ ] 1 · service answers the interface for a virtual-mission frame and a real bag frame; `abstain` reachable; `invalid_input` on a broken frame (S4.6 tests).
- [ ] 2 · registered model with a card; `model_version` reproduces the run (S4.6 + card).
- [ ] 3 · N-table (N1–N3, N6) for both backbones, five seeds, 224 and 518/512, split rule under every number (S4.5 tests); N4/N5 with piece 5.
- [ ] 4 · verdict stated by the pre-declared criteria; licence status per backbone in the card.
- [ ] 5 · evaluation note filed (method, N-table, curves, surprises, what the numbers do not show).
- [ ] 6 · compute log (N7) with our own numbers; the note is the A1 skeleton (claims A–D answered or blank).

## 6. Dependencies and risks

| On | What piece 4 needs | When |
|---|---|---|
| Piece 1 (interface) | the frame record schema and the answers to the open questions that touch the request (`frame_uid`, `bbch_null_reason`, yaw unit) | W1 hand-shake; v0.1 in W10 must not break S4.6 |
| Piece 3 (store) | DVC for manifests/caches (W2); MLflow server (until then local `mlruns/`); the predictions table (W5; until then the JSONL log) | W2, W5 |
| Piece 2 (emulator) | one virtual mission on the frame topic for the W5 end-to-end test; the ROS 2 side runs on Linux/WSL | W5 |
| Hardware | a GPU for the Tanzania caches and the high-res pass; the iBean slice runs on CPU | W2 onwards |

Risks and responses: H8 §6.12 (downloads, near-duplicate leakage, DINOv3 gating, no GPU, abstention on the rarest class, numbers that look too good). Two more that are specific to this file: the vertical slice slipping past Tuesday of W2 (then cut the Makerere crop-level manifest, not the slice), and specs growing beyond one page (then they are describing the implementation, not the contract — cut).
