# Piece 4 — model service v0: actionable work plan (W1–W6)

Executes **H8 §6.10** ("Work plan — the coming month, week by week") and is accepted by **H8 §6.11**. The plan is H8; this file is the execution: tasks, commands, tests, files. Owner **M** (ML role), with **P** for data in W1–W2. Last updated 2026-09-16 (Wed, mid-W1). Update the checkboxes weekly; when a task changes meaning, change H8 first.

Method: spec-driven — each contract below gets a one-page spec (`model_service/specs/NNN-<name>/spec.md`) and a failing test before implementation; a *spec* is a contract + acceptance tests + a protocol, never an expected number. Build a **vertical slice through iBean first** (1,296 images, MIT), then widen along one axis at a time (dataset → backbone → resolution → head → abstention → probe). Every artefact is keyed by hashes so that re-running is a no-op and adding a backbone is a config entry.

---

## 0. Where things live

```
model_service/
  specs/            one folder per spec: 001-manifests/, 002-cache/, 003-heads/, 004-abstention/, 005-results-table/, 006-service/ — each with spec.md (+ plan.md, tasks.md when written)
  src/ms/           package: data/, cache/, heads/, abstain/, eval/, service/
  configs/          backbones/*.yaml, datasets/*.yaml, heads/*.yaml, eval.yaml
  tests/            pytest; fixtures/ibean_30/ (30 images, MIT — attribution file next to them)
  results/          n_table.jsonl, compute_log.jsonl, verdict.md  (small; tracked in git)
  cards/            model cards (markdown), one per registered model
  DECISIONS.md      dated choices, in the style of interface/README.md "choices the draft makes"
data/                        (git-ignored; DVC from W2 — piece 3)
  raw/<dataset>/             as downloaded, untouched
  manifests/<dataset>_<ver>.jsonl
  cache/<backbone_id>/<res>/<dataset>.npz + meta.json
  bags/                      piece 2
mlruns/                      local MLflow (git-ignored) until piece 3 runs the server
```

Make targets (all idempotent): `make env`, `make download DS=<name>`, `make manifests`, `make cache BB=<backbone_id> RES=224|518`, `make heads`, `make eval`, `make probe`, `make serve`, `make test`.

---

## 1. The six contracts (specs)

| Spec | Artefact | Invariants (tests fail until true) | H8 | Week |
|---|---|---|---|---|
| **S4.1 Manifests** | `data/manifests/<dataset>_<ver>.jsonl` — one row per image: `image_id, sha256, path, dataset, source_record, class_raw, class_km2, group_keys{district, subcounty, date, region, session, phash_group}, split, split_rule, licence, attribution` | every row has a sha256 that matches the file; duplicates (exact or phash) share a `phash_group` and never straddle splits; `class_km2 ∈ {healthy, rust, anthracnose, unknown_als, unknown_wm}` and the two `unknown_*` classes never appear in a train split; `split_rule` is a non-empty string; the test manifest hash is recorded before any head is trained | §6.3 | W1–W2 |
| **S4.2 Feature cache** | `data/cache/<backbone_id>/<res>/<dataset>.npz` (`cls[N,D]`, `meanpatch[N,D]` float16, `image_id[N]`) + `meta.json` (`backbone_id, weights_sha256, resolution, token_types, preprocess_string, transformers_version, torch_version, device, wallclock_s, images_per_s`) | cache key = (backbone_id, weights_sha256, resolution, preprocess_string); a changed weights hash yields a new cache; shapes match `meta.json`; re-running with the same key does nothing; `wallclock_s` is written to `results/compute_log.jsonl` (N7) | §6.4 | W2–W3 |
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
- [ ] **Spec S4.2** (M, 2 h) + `tests/test_cache.py` failing: cache key, `meta.json` fields, the two token types, the compute-log row.
- [x] **Fixture** (M, 1 h): `tests/fixtures/ibean_30/` — 30 iBean images (10 per class) + `LICENSE-MIT` + `manifest.jsonl`; CI runs on CPU on this fixture only. — *Done 2026-09-19, except `manifest.jsonl`: the golden manifest comes from the W2 builder. Laid out like `data/raw/` for spec 001 US-1.1: `ibean/{DOWNLOAD.json, SHA256SUMS, extracted/<split>/<class>/}`, with 6/2/2 images per class from the source's train/validation/test, byte-for-byte copies, 4.0 MB. Attribution is in `README.md`.*
- [ ] **Backbone configs** (M, 30 min): `configs/backbones/dinov2_l14_reg.yaml` (`facebook/dinov2-with-registers-large`, Apache-2.0, patch 14, resolutions {224, 518}, ImageNet normalisation) and `dinov3_l16.yaml` (`facebook/dinov3-vitl16-pretrain-lvd1689m`, DINOv3 licence, gated, patch 16, resolutions {224, 512}).
- [ ] **Interface hand-shake** (M + P, 30 min): confirm with piece 1 that the service request = `observation_frame` + `request_id` + `options`; adopt their answers to the open questions that touch us (`frame_uid` rename; `bbch_null_reason` enum; radians for yaw). Note in `DECISIONS.md`.

Done when: `make test` runs (red for S4.1/S4.2); iBean and Makerere on disk with sha256 lists; the one-page data note (`data/README.md` extended: what is on disk, sizes, licences, what is missing).

### W2 — 21–25 Sep

Goal: the **iBean vertical slice** end to end, then manifests and 224-px caches for all sets; the first number (N4).

- [ ] **Mon–Tue — the slice on iBean**: `make manifests` (iBean only) → `make cache BB=dinov2_l14_reg RES=224` (CPU is acceptable here: ⚠ ~30–40 min for 1,296 images) → linear probe (`ms.heads.train --head linear`, one seed) → one N1 row (`split_rule = "unblocked"`) → `results/n_table.jsonl` → `make serve` and `POST /v1/predict` with one cached image. This is the whole pipeline once, on the smallest data; everything after is widening.
- [ ] **Wed — Makerere**: manifest with real grouping keys (district/sub-county/date from the Dataverse metadata; check completeness — H6 E4 lead); box-level crops (10 % margin) as a second manifest; the first *blocked* split; S4.1 tests green.
- [ ] **Wed–Thu — Tanzania**: exact-hash de-duplication across the three records; phash groups; the class map; manifests frozen; test manifests hashed and listed in `DECISIONS.md` ("frozen before any head").
- [ ] **Thu–Fri — caches at 224** for both backbones on all sets (GPU); compute log rows for every extraction (N7).
- [ ] **Fri — site-prediction probe (N4)**: logistic regression on cached CLS features predicting the dataset (3-way; balanced accuracy vs chance 33 %) and, within Makerere, the district. One row per backbone in the N-table; logged in MLflow (`make probe`).
- [ ] Hand the manifest format to piece 3 for DVC (`dvc add data/manifests data/cache` — P).

Done when: manifests for all sets in DVC; 224 caches for both backbones; N4 in MLflow and in `n_table.jsonl`; first N7 entries; S4.1 and S4.2 green.

### W3 — 28 Sep–2 Oct

Goal: heads and the first N1/N2 pairs at 224; the high-resolution caches running overnight.

- [ ] **Spec S4.3 + S4.5** (Mon): head CLI, run.json, the N-table row schema; failing tests.
- [ ] **Heads** (Mon–Wed): linear probe (logistic regression on L2-normalised CLS), prototype head (K = 4 per class, cosine, τ init 0.07 learnable, per-class k-means init), fixed 0.5/0.5 mixture; AdamW lr 5e-4–1e-3, wd 1e-4, batch 256, focal γ = 2, label smoothing 0.1, 50 epochs, early stop on in-domain val; class-balanced sampling; **five seeds**; seconds per run to the compute log.
- [ ] **N1** (Wed): within-dataset blocked (Makerere by district; Tanzania by region/session or `unblocked`); per-class recall incl. the rarest; abstention rate reported later (W4).
- [ ] **N2** (Thu): cross-dataset — Tanzania → Makerere (frame-level and crop-level; healthy/rust), Tanzania → iBean (healthy/rust), Makerere + iBean → Tanzania (healthy/rust); anthracnose only where both sides have it. Table rendered from `n_table.jsonl` by `make eval` (`results/n_table.md`).
- [ ] **High-res caches** (nightly from Tue): `RES=518` (DINOv2) / `RES=512` (DINOv3); ⚠ of the order of a GPU-day per backbone on a 3070-class card; log N7.
- [ ] Guard-rail check: if N1 ≫ N2 does not appear, look for leakage (duplicates across splits) before anything else.

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
       --train-manifest data/manifests/ibean_v1.jsonl --split train --val-split val
make eval                                    # one N1 row -> results/n_table.jsonl + n_table.md
make serve                                   # then: POST /v1/predict with one cached image
```

Expected: an afternoon if the GPU is there, a day on CPU. What it proves: the contracts hold end to end on real bean images before any scale is added. What it does not prove: anything about accuracy — iBean is unblocked and tiny; its number is never quoted.

---

## 4. Conventions

- **Backbone ids**: `dinov2_l14_reg`, `dinov3_l16`, optional `dinov3_b16`; a backbone is a YAML file, nothing else changes.
- **Cache key**: `(backbone_id, weights_sha256, resolution, preprocess_string)`; `preprocess_string` = e.g. `resize_short=518;center_crop=518;norm=imagenet`.
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
