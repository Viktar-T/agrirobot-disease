# model_service — decisions

Dated choices for piece 4, in the style of `interface/README.md` ("choices the draft makes").
One entry per choice: what was decided, why, and what would reverse it. The plan is H8;
the execution is `docs/piece4-work-plan.md`. Frozen manifest hashes (S4.1, US-5) are
appended here as they are made.

## 2026-09-17 — W1 · environment (`pyproject.toml`, `make env`)

1. **One Python environment for the repository, at the root.** `pyproject.toml`, `Makefile`
   and `.venv/` live at the repository root and the package is `model_service/src/ms`
   (installed as `ms`, editable). The work plan's commands (`make cache …`,
   `python -m ms.heads.train …`) are written to be run from the root, and pieces 2 and 3
   build on Linux with their own tooling (colcon, DVC), so a second environment under
   `model_service/` would only add a `cd`. Reversed if piece 3 needs conflicting pins on
   the same box — then this file moves to `model_service/pyproject.toml`.
2. **Python 3.11** (`requires-python = ">=3.11,<3.12"`), as H8 §6 asks. The box's system
   Python is 3.14; `uv` fetches and pins 3.11.14 for the venv, so nothing on the machine
   has to change.
3. **`uv` as the environment manager** (lock file committed: `uv.lock`). It resolves the
   PyTorch CUDA index and the PyPI packages in one lock, and `uv sync` is the idempotent
   `make env` the plan asks for. `pip install -e .` still works for anyone without `uv`.
4. **torch from the `cu128` index, not PyPI.** The dev GPU is an **NVIDIA GeForce RTX 5060
   Laptop (8 GB, Blackwell, compute capability 12.0, driver 592.15, CUDA 13.1)**; wheels
   built for CUDA ≤ 12.6 have no `sm_120` kernels. Locked: `torch 2.11.0+cu128`,
   `torchvision 0.26.0+cu128`. Reversed when a machine with a different card is used —
   change the one index URL in `pyproject.toml`.
5. **`scikit-learn` `NearestNeighbors`, not `faiss-cpu`** for the kNN score of S4.4. The
   plan allows either; the training sets here are ≤ 2·10⁵ vectors of ≤ 1024 dimensions,
   where exact search is fast enough, and it is one dependency fewer with Windows wheels
   that are certain. Reversed if W4 timings say otherwise.
6. **Three dependencies beyond the plan's list**: `pillow` (image decode for S4.1 hashes,
   pHash, crops — it arrives with torchvision anyway, so it is declared rather than
   implied), `pyyaml` (the plan's own `configs/*.yaml` and `class_map_v1.yaml`),
   `python-dotenv` (reads `.env`, so `HF_TOKEN` is not typed into a shell).
7. **`transformers` 5.17** — the plan asks for `>= 4.56` (the version that first carried
   DINOv3); the current major is 5.x and it carries both `Dinov2WithRegistersModel` and
   `DINOv3ViTModel` (checked on install, 2026-09-17). If a 5.x API change bites during
   S4.2, pin `>=4.56,<5` and relock; the exact version ends up in every cache `meta.json`
   either way. → Settled in 12 (2026-09-18).
8. **Compute log (N7) — two extensions to the row of the work plan's §4.** `step = "env"`
   is allowed in addition to {extract, train_head, eval, serve}, because the first entry
   the plan asks for (GPU model / VRAM / driver) is not any of those four; and that row
   may carry an `env` object with the structured machine description (driver, CUDA, torch,
   cuDNN, transformers, python, OS, git sha), so the versions are data rather than prose
   inside `notes`. The log is written only by `ms.compute_log`, never by hand;
   `record-env` is idempotent — an unchanged machine adds no row.
9. **`make.ps1` beside the `Makefile`.** The dev box is Windows without `make`; the shim
   runs the same commands so `.\make.ps1 env` and `make env` mean the same thing. The
   `Makefile` stays the canonical list (it is what the Linux/WSL side of pieces 2 and 3
   will use).
10. **Secrets**: `.env` (git-ignored) from the tracked `.env.example`; `HF_TOKEN` only,
    with an optional `HF_HOME` to keep the ~1.2 GB checkpoints off the system drive.
    DINOv3 acceptance date and account go in the next entry, with the W1 "DINOv3 terms"
    task.

### Installed on 2026-09-17 (`make env` on the dev box)

`python 3.11.14`, `torch 2.11.0+cu128` (CUDA runtime 12.8, cuDNN 9.19), `torchvision
0.26.0+cu128`, `transformers 5.17.0`, `mlflow 3.16.1`, `dvc 3.67.1`; 198 packages resolved.
GPU: **RTX 5060 Laptop, 7.96 GB, driver 592.15, CUDA 13.1, compute capability 12.0**;
a CUDA matmul on `sm_120` runs. First compute-log row written:
`model_service/results/compute_log.jsonl`. `make test` green (5 tests), `make lint` clean.
8 GB of VRAM is not the binding constraint for extraction; time is (see 11).

## 2026-09-18 — W1 · where extraction runs, and the transformers line (approved)

11. **All feature extraction runs on the dev laptop: 224 in W2, 518/512 overnight in W3.**
    Estimates for ViT-L without gradients, about 7 GB usable (the display holds about
    0.6 GB): weights 0.6 GB in fp16; activations per image at 518 about 25 MB with SDPA
    and about 120 MB with eager attention (DINOv3 at 512: about 20 / 70 MB); about
    1.0 TFLOP per image at 518 (DINOv3 at 512: about 0.7), about 6× the 224 pass. Batch 32
    at 518 fits even with eager attention, so memory does not limit inference. The open
    question is time, and running unattended on a laptop (sleep, thermal throttling at
    80 W). Rejected: a university GPU server (none arranged); high-res on a subset only
    (kept as the fallback below).
    - **Precision**: inference in fp16 autocast with SDPA, batch 32. Before the first real
      cache, compare fp16 with fp32 once on the 30-image fixture (cosine similarity of the
      CLS vectors); the measured gap sets the tolerance of the golden test in 12.
    - **Cache key (S4.2)**: `compute_dtype` joins `(backbone_id, weights_sha256,
      resolution, preprocess_string)`, because fp16 and fp32 features differ and the
      plan's key would not see it. Batch size goes in `meta.json` only.
    - **Resumable**: extraction writes shards and skips finished ones, so a laptop that
      sleeps loses one shard rather than a night; one compute-log (N7) row per run.
    - **Go/no-go, fixed before the number exists**: Monday of W2, time iBean (1,296
      images) at 224 and at 518 and project the high-res pass for both backbones over all
      manifests. More than three nights → high-res only on the evaluation manifests plus
      a sample of the training data, recorded here. A server only if one becomes
      available.
    - **Before the DINOv3 weights are downloaded**: confirm that this laptop counts as a
      university machine under the DINOv3 terms (H6 C5).
12. **`transformers>=5.17,<6`** (lock: 5.17.0), and the library version kept out of the
    features. No cache exists yet, so staying on 5.x costs nothing now; a later switch
    would cost a GPU-night per backbone. Rejected: pinning `>=4.56,<5` (matches the plan
    text, but it is the older line and a later move means re-extracting); leaving
    `>=4.56` open (a relock could jump to 6.x silently).
    - **Preprocessing** in torchvision from `preprocess_string`, not `AutoImageProcessor`,
      so processor defaults cannot change the cache.
    - **Explicit loading**: each `configs/backbones/*.yaml` sets dtype, attention
      implementation and a pinned Hub `revision` (commit sha); defaults are what change
      between majors, and the pin stops the Hub repo changing underneath.
    - **Golden-feature test**: DINOv2 CLS at 224 on the fixture (30 × 1024 fp16, about
      60 KB) committed to `tests/fixtures/`; tolerance from the fp16 check in 11; skipped
      in CI (needs the weights) and run before any relock. A failure means a new cache
      version, not silent drift. DINOv2 only: features derived from DINOv3 stay out of
      git under its terms.
    - **Cache key**: library versions stay out of it (every relock would otherwise
      invalidate GPU-nights of caches); the golden test guards drift, and
      `transformers_version` / `torch_version` stay in `meta.json` as planned.
    - **Fallback** to `>=4.56,<5` only before the first real cache (Monday of W2), and only
      if either backbone will not load or run as S4.2 needs under 5.x, or the golden test
      cannot be made stable.

## 2026-09-18 — W1 · DINOv3 terms (accepted)

13. **DINOv3 licence accepted for research evaluation.** Request "DINOv3 — Gating Group
    Collection" on Hugging Face, submitted and accepted 2026-09-18 on the ML role's own account
    `Viktar-Taustyka`; one acceptance covers the collection (vitl16 and, if used, vitb16).
    Licence: "DINOv3 License" (Meta, 14 Aug 2025) — custom, not OSI; gated weights. Token
    `DINOv3-for-agrirobot-disease`, fine-grained, read-only ("read repository contents", "read
    contents of gated repos"), stored as `HF_TOKEN` in `.env` (git-ignored, untracked; see 10).
    - Scope: challenger only (H8 §6.4; H9 §6). Research-only until licence sheet H6 C5 is signed
      with 4TECH; nothing containing DINOv3 — weights, features, heads trained on them, golden
      vectors — leaves the project's machines or enters git (see 12).
    - The dev laptop counts as a project machine for that rule (closes the open point in 11):
      the rule is ours (no distribution before C5), not Meta's; the licence permits use on any
      machine and binds *distribution* (attribution + licence copy), which we are not doing.
    - Article A1 acknowledges DINOv3 use if its numbers are reported (licence clause on
      publications).
    - Pinned Hub revisions (configs/backbones): dinov3_l16 `ea8dc2863c51be0a264bab82070e3e8836b02d51`, dinov2_l14_reg `e4c89a4e05589de9b3e188688a303d0f3c04d0f3`;
      `hub-check` is the command that re-verifies access and prints the current shas.
    Reversed by: a signed C5 (lifts research-only) or a licence change by Meta.

## 2026-09-19 — W1 · spec 001 (manifests), clarified against the data

The spec's Clarifications (`specs/001-manifests/spec.md`) closed on the downloaded data (the W1
data note in `data/README.md`, plus EXIF and XML scans of every Tanzanian and Makerere image).
Tests: `tests/test_manifests.py`, red until W2. A throwaway reference builder, never committed,
passed them all (apart from US-1.1, which needs the fixture) to show they can be satisfied.

14. **Tanzania is one manifest, `tanzania`, built from tz155k and tz59k**, not one per record.
    tz59k sits inside tz155k (finding 1). Separate manifests would split the same pictures twice,
    with nothing to keep a picture in the same split both times. Now the overlap is a number in
    the sidecar (SC-4), and N2 and N4 treat Tanzania as one source anyway. Reversed by: a use for
    tz59k on its own (none is planned).
15. **One row per distinct image (sha256).** Copies go into `dup_paths`. An image filed under two
    labels is excluded (`label_conflict`, finding 3). In the Tanzanian records, 28–38 % of the
    files are extra copies (finding 2), so one row per file would weight rust about 2.5× in
    training and cache the same features several times over.
16. **Split rules.**
    - Makerere: `blocked:district`. Healthy images have no XML, so each is placed by its capture
      date, in the district(s) that day's annotated images name. Images of every class count,
      including held-out angular leaf spot: 26 Apr has angular leaf spot only, and without it
      that day's 870 healthy images would have no block. Bugiri and Mayuge share 25–26 Apr
      and therefore form one block.
    - Tanzania: `blocked:date`, from EXIF DateTimeOriginal, because the source records no region
      or session. The one picture without EXIF falls back to its phash group, and its
      `split_rule` says so.
    - iBean: `unblocked:random_by_phash_group`. It is test-only in the protocol; its split serves
      the W2 slice only.
    - Targets are 70/10/20 of the rows of each class. Only whole groups move, and every trained
      class must reach every split, or the build fails.

    Reversed by finer metadata, for example GPS clusters of farms for Tanzania (present in
    18–69 % of images, depending on class) in a v2.
17. **Capture date: one source per dataset, the one that agrees with the acquisition.**
    - Makerere: the XML `datetime`. For images without XML, the file name, read as Unix
      milliseconds in UTC+03:00. It falls on the XML's day for 10,116 of 10,118 annotated
      images, and on the trip days for every healthy image.
    - Makerere's EXIF was rewritten after the trip. IFD0 DateTime reads 2021-08-07, and
      DateTimeOriginal is missing in 361 healthy images and dated after the trip (2021-06-10,
      2021-08-11) in 306.
    - Tanzania: EXIF DateTimeOriginal. Every image has it apart from one picture held by both
      records. The file names are upload times: they disagree with DateTimeOriginal for 55 % of
      tz155k's healthy images.
    - IFD0 DateTime, the file-change time, is never used.
18. **SWM rows are the R-SWM originals.** These are frames, comparable with the ALS frames. A
    picture is `unknown_wm` if and only if one of its boxes is White Mold. Pictures showing only
    apothecia or sclerotia are `excluded` by the class map. The 300-px crops become `swm_crops`
    (P2). Reversed by: N3 needing crop-level unknowns first.
19. **Manifest recipes live in `configs/manifests/<manifest>.yaml`**, next to
    `configs/class_map_v1.yaml`. The download configs in `configs/datasets/` stay one per record,
    and `DatasetConfig` rejects keys it does not know.

## 2026-09-19 — W1 · the fixture manifest; two amendments to spec 001

20. **The fixture's manifest is `tests/fixtures/ibean_30/ibean_v1.jsonl` plus its sidecar**, not
    `manifest.jsonl` as the work plan names it. These are the names spec 001 gives the builder's
    output, so the builder can write the manifest in place and the loader finds the sidecar.
    - Until the builder exists (W2), `tests/fixtures/make_ibean_30.py` writes it: 30 rows,
      healthy and rust 7/1/2 into train/val/test, angular leaf spot held out. The same script
      also rebuilds the images.
    - A red test in `test_manifests.py` requires a fresh build to match these rows in every
      field except `split`, which is the builder's own seeded draw. An independent
      implementation already reproduces every other field.
    - "CI runs on CPU on this fixture only" holds as a property of the tests: they run on CPU
      and read no data outside `tests/` and `tmp_path`. The cache tests use a tiny
      random-init backbone, so there are no downloads, and the golden test is opt-in. There is
      no CI workflow in the repository yet. On Linux, `uv sync` would pull the cu128 torch
      wheels (about 3 GB), so how CI installs torch is a choice for whoever sets CI up.
21. **Spec 001 amended in two places.**
    - **pHash is pinned bit for bit** in FR-006: greyscale, Lanczos to 32×32, orthonormal
      DCT-II, top-left 8×8 against their median, row-major, first bit most significant.
      Without that, `phash` would depend on the implementation, and the committed fixture
      manifest could not be checked against the builder.
    - **The loader gains `purpose = extract`** (FR-011). Feature extraction (spec 002) has to
      read every split, test and held-out included, and it uses no labels.

## 2026-09-19 — W1 · spec 002 (feature cache)

`specs/002-cache/spec.md`; tests `tests/test_cache.py`, red until W2. A throwaway reference
extractor, never committed, passed all of them to show they can be satisfied. The golden test
is opt-in and was not run.

22. **The key is part of the cache path**: `data/cache/<backbone_id>/<res>/<cache_key>/
    <manifest>_v<N>.npz` with its `.meta.json`. The plan's layout
    (`data/cache/<backbone_id>/<res>/<dataset>.npz`) would overwrite a cache whenever the
    weights, preprocessing or dtype change. With the key in the path, both caches stay, as the
    fp16/fp32 comparison of 11 needs.
23. **`cache_key` is the first 16 hex digits of sha256(`json.dumps([backbone_id,
    weights_sha256, resolution, preprocess_string, compute_dtype])`)**. These are the four
    fields of the plan plus `compute_dtype` (11). Library versions, device, batch size and
    shard size stay out of the key (12) and go to the sidecar.
24. **`weights_sha256` is the sha256 of the weight file the model is loaded from**, not a Hub
    revision: a corrupted or replaced download changes the key. At the pinned revisions both
    backbones ship a single `model.safetensors` of about 1.2 GB.
25. **`meanpatch` is the mean of the patch tokens only.** Both backbones carry 4 register
    tokens (their `config.json` at the pins), so the last hidden state is `[CLS, 4 registers,
    (res/patch)² patches]`. The register tokens are not image regions, so they are left out;
    any other token count stops the run. `cls` is `last_hidden_state[:, 0]`, after the final
    norm.
26. **Preprocessing is the string, and the string is pinned.**
    - The EXIF orientation is applied first, then the image is converted to RGB.
    - `resize_short` resizes the shorter side, bicubic with antialias; `center_crop` follows;
      `norm=imagenet` divides by 255 and applies the ImageNet mean and std.
    - A changed meaning needs a new token. One `preprocess()` serves both the cache and the
      service (006).
    - `float16` means CUDA autocast over float32 weights; on CPU only `float32` runs.
27. **What the cache records, and how it is tested.**
    - `sha256[N]` is stored next to `image_id[N]`, for 006's cached-hash path.
    - Every extraction run writes one compute-log row with a `cache` object, an extension of
      the row like `env` in 8.
    - Runs resume from shards; `--max-shards K` stops a run on purpose.
    - The CPU tests use a tiny random-init model of the real class
      (`Dinov2WithRegistersModel`, width 32, 4 registers, about 180 KB) saved to `tmp_path`,
      so the transformers code path runs with no download.

## 2026-09-19 — W1 · interface hand-shake with piece 1 (their draft v0)

Piece 4 adopts piece 1's W1 draft (`interface/README.md`,
`interface/schema/v0/observation_frame.schema.json`). **Confirmed on 2026-09-19 by Viktar
Taustyka, the ML role, acting for piece 1 as P.** The request shape, yaw in radians and the
`bbch_null_reason` enum are confirmed as drafted, and the image id is renamed to `frame_uid`
(31). E and the robotics side have not reviewed piece 1's draft yet; that review is part of
piece 1's own "done when", not of this hand-shake.

28. **Service request = the `observation_frame` record + `request_id` + `options`.** Piece 1
    shaped the record as `frame` + `metadata` precisely so that this holds (their choice 1).
    - S4.6 validates the record against piece 1's schema (`contract_version = "v0"`,
      `additionalProperties: false`), and against their validator library once it exists,
      never against a copy of our own. A broken frame gives HTTP 422 with the validator's
      reason.
    - The `uri` forms (MCAP, file roots) are still open between pieces 1 and 2 (W3). S4.6
      needs them to fetch the bytes.
29. **Yaw in radians, in [-π, π]** (ROS REP 103; their choice 4; the schema enforces the
    bounds). The H8 §6.2 example `"yaw": 91.0` is invalid under v0, so S4.6's examples and
    tests use radians; the schema example uses 1.588.
30. **`bbch_null_reason` ∈ {`not_recorded`, `not_applicable`, `unknown`}** (their choice 5).
    It is required when `bbch` is null and not allowed otherwise.
    - Piece 4's v0 model does not use `bbch`; the service carries `bbch` and
      `bbch_null_reason` into its request log unchanged.
    - Frames built from the open datasets have `bbch = null` with `not_recorded`.
31. **The image id is `frame.frame_uid`, renamed in v0 on 2026-09-19.** In ROS, `frame_id`
    names a coordinate frame, and v0 had both `frame.frame_id` (the image) and
    `metadata.pose.frame_id` (the TF frame).
    - The rename was made now, while v0 is a draft and no code reads it. After v0.1 (W10) it
      would break every consumer, and the plan requires that v0.1 does not break S4.6.
    - Piece 1's schema, its example and its README carry the change (their choice 11). A
      record that still says `frame.frame_id` is now rejected.
    - `pose.frame_id` keeps its name. S4.6 reads `frame_uid` in one place and echoes it in
      the response.
32. **The frame's `sha256` is piece 4's content hash.** Piece 1 defines it as the hash of the
    image bytes exactly as stored at `uri` (their choice 10), the same definition as spec 001's
    `sha256` and spec 002's `sha256[N]`. S4.6's cached-hash path looks it up in the caches
    after recomputing it from the bytes it fetches, so a wrong hash in a request cannot select
    another image's features.

## 2026-09-19 — W2 · the vertical slice on iBean (ahead of schedule)

The work plan's "Mon–Tue — the slice on iBean", run end to end on the dev laptop's GPU. Its
numbers exercise the pipeline and are never quoted: iBean is test-only in the protocol and
unblocked (H8 6.3).

33. **The slice runs ahead of specs 003, 005 and 006.** `ms.heads.train`, `ms.eval.run` and
    `ms.service.app` are thin first versions with tests on the CPU fixture
    (`tests/test_slice.py`). 34–37 record their choices, and the specs (W3–W5) may change any
    of them with their own failing tests. The manifest builder implements spec 001 with a
    reader for iBean only: Makerere, Tanzania and SWM bring theirs, and blocked splits, with
    their W2 tasks, so the 10 tests of `test_manifests.py` that build them stay red until then.
    The cache implements spec 002 in full.
34. **The linear probe: logistic regression on L2-normalised CLS, with the H8 6.5 recipe.**
    - `configs/heads/linear.yaml`: AdamW, lr 1e-3, wd 1e-4, batch 256, focal loss γ = 2,
      label smoothing 0.1, at most 50 epochs, class-balanced sampling, patience 10. Training
      runs on CPU and is deterministic for a seed.
    - Early stopping and the epoch kept follow the validation split's cross-entropy averaged
      per class (ties: macro-F1). The first slice run used macro-F1 itself: on iBean's 87
      validation images it moves in one-image steps, peaked by chance at epoch 6 and stopped
      the run at epoch 16 of 50 while the validation loss was still falling (0.675 → 0.526).
      That run was deleted; its compute-log row stays.
    - The head is still underfit on iBean. 604 rows make 3 steps per epoch, and at epoch 50 the
      validation loss is 0.42 and falling (max probabilities about 0.65). The large sets take
      far more steps per epoch, and temperature scaling (S4.4) sharpens the probabilities.
      Whether the linear head needs more steps or a logit scale is for spec 003: a finding,
      not a fix.
    - Runs live in `data/heads/<run_id>/` (git-ignored: heads on DINOv3 features are
      research-only, 13), as `head.pt` + `run.json`. The run_id is
      `<backbone>-<res>-<head>-<tokens>-s<seed>-<hash8>`, the hash taken over every input
      (cache key, both manifest hashes and splits, the head config's sha256, seed, trainer
      version). So a rerun does nothing, and a changed config is a new run.
    - `model_version = msv0.1+<backbone_id>@<res>.<head>.man-<train manifest sha256[:6]>`.
35. **A test-only manifest trains only with `--allow-test-only`.** Spec 001 FR-001 leaves the
    role for 003 to enforce, and the slice trains on iBean, whose role is `test_only`. The
    flag is recorded in `run.json`, and the run, its N-table row and every service response
    say "not quotable".
36. **The N-table row (provisional S4.5).** It holds the plan's S4.5 fields, both manifest
    hashes and splits, `n`, `classes`, `run_id`, `model_version`, `cache_key`, `quotable`
    and `notes`.
    - `split_rule` is the evaluated rows' rule, `unblocked:random_by_phash_group`; the plan's
      "unblocked" is shorthand for it.
    - One seed leaves `ci_low` and `ci_high` null (the interval is over five seeds); no
      abstention yet gives `coverage` 1.0.
    - `make eval` appends a row only when its identity (run, number, metric, test manifest
      hash, split, coverage) is new, then renders `n_table.md`. A second run adds nothing.
37. **The service slice answers from the cache only.**
    - The request is checked against piece 1's schema, plus `request_id` and `options`. A
      broken one gets HTTP 422 with `invalid_input` and the validator's reasons.
    - `frame.uri` is a path relative to `data/raw` (`MS_DATA_ROOT`), an absolute path or
      `file://`, and must resolve inside the data root. Other schemes (`mcap://`) get 501
      until the uri forms are fixed with piece 2 (W3).
    - The sha256 is recomputed from the bytes (32); a mismatch is 422. A hash cached under the
      head's cache key, for any manifest, is answered without the backbone; any other gets 501
      until features are computed on request (W5).
    - Scores are an uncalibrated softmax over the head's classes, and `decision` is always
      `predict` until S4.4. The response echoes `frame_uid` (31) and says all this in
      `warnings`. The service serves `MS_HEAD_RUN`, else the newest run under `data/heads`.
38. **Golden features and their tolerance (spec 002 Clarification 7, closed).** On the
    fixture at 224, CLS under fp16 autocast against fp32 (RTX 5060, torch 2.11.0+cu128,
    transformers 5.17.0) has a minimum cosine of 0.999998924 (mean 0.999999222). The gap,
    1.08e-6, rounded up to one significant digit gives the tolerance, 2e-6: a relock may move
    the features by as much as float16 itself does. The golden vectors are the fp16 run's CLS,
    30 × 1024 float16 (60 KB), in `tests/fixtures/ibean_30/golden/` beside the measurement;
    `tests/fixtures/make_golden.py` rebuilds them. Another GPU's fp16 kernels may differ at
    the same scale, so the test belongs to this box, before relocks.
39. **The fixture's manifest is now the builder's.** The builder reproduced every field of the
    committed rows except `split`, its own seeded draw (8 of 30 rows moved), and
    `make_ibean_30.py` now calls it, as 20 foresaw.
40. **Builder choices inside spec 001.**
    - `manifest_version` is the manifest version as a string ("1" for v1).
    - pHash groups come from exact all-pairs Hamming distances in chunks of at most 2^24
      comparisons (numpy `bitwise_count`): Tanzania's 112k rows are about 6·10^9 pairs, a
      minute or so.
    - The split: groups shuffled with the seed, largest first, each into the split that
      brings its classes closest to 70/10/20. Then repair moves fill any split that lacks a
      class, and if none can, the build fails with `class_missing_from_split`.
    - Files are hashed and decoded by 8 threads, and every row of every member is read.
41. **What the slice measured (never quoted).**
    - `ibean_v1.jsonl`: 1,295 rows (healthy 427, rust 436, `unknown_als` 432). The renamed
      `.DS_Store` is excluded (`not_an_image`), and there are no near-duplicates at pHash ≤ 6.
      Split: train 604, val 87, test 172 (0.70/0.10/0.20 per class). sha256 `423a1666…6060`.
      Not frozen: Clarification 4 (the pHash threshold) is still open.
    - Cache `dinov2_l14_reg` @ 224, fp16, key `2b017210105b0b1f`: 1,295 images in 19.6 s,
      model loading and weight hashing included (66 images/s). A rerun does nothing.
    - Head `dinov2_l14_reg-224-linear-cls-s0-402db169`: 50 epochs in 0.3 s; validation
      macro-F1 0.977.
    - N1: macro-F1 0.977 on 172 test rows, `unblocked:random_by_phash_group`, not quotable.
    - `make serve`, then `POST /v1/predict` with a cached test image: HTTP 200 in 1.8 ms,
      from the cache. A broken frame got 422 with three validator reasons.

## 2026-09-19 — W2 · Makerere: the first blocked split, and its crops (ahead of schedule)

The work plan's "Wed — Makerere". The manifest is built and checked but not frozen yet:
freezing comes with the Tanzania task, before any head (SC-6).

42. **Readers for Makerere, Tanzania and SWM; the S4.1 tests are green (36 of 36).** They
    follow the acceptance tests and the W1 scans.
    - Makerere: the XML gives class, district, sub-county, datetime (the date), variety, age
      (`plant_age`) and boxes. A healthy image has its folder and its file-name date.
    - Tanzania: the class folder without its chunk digits, and EXIF DateTimeOriginal, read
      while the image is decoded.
    - SWM: the R-SWM originals and their VOC files.

    The Tanzania and SWM manifests are built on the real data with the next task.
43. **The split search is exact for blocked splits.** FR-008 asks for an assignment close to
    the targets. The first version, greedy, cannot swap whole blocks, and put Makerere's
    healthy at 0.63/0.20/0.17 instead of 0.63/0.17/0.20.
    - Up to 12 groups, every assignment is scored and the closest is taken. Makerere's 11
      blocks give 3^11 = 177,147 assignments.
    - Up to 500 groups (Tanzania's dates), moves and pairwise swaps improve the greedy
      assignment.
    - Beyond that the greedy one stands; iBean has 863 fine-grained groups.
    - The greedy keeps the first version's float arithmetic, so `ibean_v1` (sha256
      `423a1666…6060`, the slice's) and the fixture are unchanged.
44. **`makerere_v1.jsonl`: 15,402 rows** (healthy 5,284, rust 5,020, ALS 5,098 held out).
    Nothing is excluded, there is one near-duplicate pair and no exact copies. sha256
    `8d9d1f06…0542`.
    - Blocks: 11 from 12 districts, with Bugiri and Mayuge merged. Train: Bugiri+Mayuge,
      Kayunga, Kiboga, Mbale, Serere, Sironko. Val: Mubende, Pallisa. Test: Hoima,
      Lyantonde, Ntungamo. Shares: healthy 0.63/0.17/0.20, rust 0.70/0.11/0.19. Healthy val
      cannot go below 0.17, because its smallest block holds 910 of 5,284.
    - Completeness (H6 E4, for P): district, date, variety and plant age are on every rust and
      ALS row; sub-county is missing on 30 rows, all in Kiboga. Healthy rows carry only the
      date (from the file name). Their district is inferred through the day, for the split
      only (FR-013).
    - The class ↔ trip confound is structural. Healthy exists only in the four April blocks,
      so the test split's healthy rows are all Hoima's (1,050), and nearly all its rust rows
      are from the May trip (Lyantonde and Ntungamo; Hoima has 6). N1 on Makerere has to say
      so, and N4 will see it.
    - For N1, N4 and the class map: 25 % of the ALS and 11 % of the rust images are flagged
      `hasOtherSymptoms`, so co-infection is possible. Resolutions differ by class:
      1024×498 is common among healthy images and rare among rust.
45. **Makerere's boxes are not in one frame on EXIF-turned images.** 3,047 of the 10,101
    annotated images carry an EXIF rotation. Their XML `size` is always the stored size, yet
    the boxes follow either frame.
    - Of the 624 turned images whose width and height swap: 262 images fit the upright frame
      only, 50 the stored frame only, 212 mix the two box by box, 98 fit both and 2 neither.
      A visual check of three agrees: in the mixed one, the boxes sit on infected leaves in
      two different frames. An app that redraws the image when the phone turns would do
      this.
    - A colour test cannot decide the square ones: on unturned images it picks a 180° turn
      twice as often as the truth.
    - So crops take each box in the one frame whose bounds hold it. A box that both frames
      hold, or neither, is excluded as `box_frame_unknown` (US-6.4). Unturned images (7,054)
      are unaffected.
    - Recovering the 9,618 unknown boxes needs a content test that passes the unturned
      control, for example a leaf-vs-background probe on the features of the known-frame
      crops. That would be crops v2, if crop-level numbers need them.
46. **`makerere_crops_v1.jsonl` (US-6): 27,019 crops** (rust 13,816, ALS 13,203) of 36,640
    boxes. sha256 `efab8bb0…52ea2`.
    - Excluded: 9,618 boxes as `box_frame_unknown`. Clipped at the image edge: 7,199.
      Cut in the stored pixels: 26,013; in the upright frame: 1,006.
    - Files: `data/derived/makerere_crops/`, 2.0 GB, JPEG quality 95 without chroma
      subsampling; 262 s. The rust crops follow their parents' splits: train 9,378, val
      1,783, test 2,655.
    - There are no healthy crops, because healthy images have no boxes. A crop-level
      healthy/rust number (H8 N2: "frame-level and box-crop-level") needs healthy crops from
      elsewhere: a question for N2 in W3.
    - The crops live in `data/derived/`, not `data/raw/`, which stays as downloaded. The
      sidecar names that root, and `validate` and the cache read the files there. Crops have
      no recipe file: `crops` derives them from the parent manifest and `--margin`.

## 2026-09-19 — W2 · Tanzania, the pHash threshold, and the manifests frozen (ahead of schedule)

The work plan's "Wed–Thu — Tanzania", ending with the freeze. The owner decided the freeze
and tz3's place in it on 2026-09-19, after the tz3 test below.

47. **tz3 adds images; `tanzania_v1` is frozen without it.** The owner allowed the test
    download of `RUST_6.zip` (0.91 GB, Zenodo 16751336, checksum verified).
    - None of its 321 rust photos is an exact copy of a tz155k or tz59k file. None lies within
      8 pHash bits of a `tanzania_v1` picture, under any of four turns.
    - They are full resolution (4160×3120; tz155k is 1000×750). The 12 that carry a date were
      taken on 2024-12-24, after tz155k's last day. The other 309 carry no capture date, so
      `blocked:date` could place almost none of them: they would join the split at random.
    - Decision: freeze `tanzania_v1` from tz155k and tz59k (spec 001 Clarification 3). The
      rest of tz3 (19 archives, about 46 GB) is a later decision. It may serve better as a
      manifest of its own, a later-in-time test, than as a member of `tanzania`. The config's
      `hold` records this.
48. **The pHash threshold is 2 bits, not 6** (spec 001 Clarification 4, closed). Every code
    has 32 one-bits, so distances are even: 2 and 3 are the same rule.
    - Copies re-saved at quality 70 or 50, or halved and re-saved, move at most 2 bits. This
      was measured on 300 real images each from iBean, Makerere and tz155k. The distinct
      images in those samples lie 10 or more bits apart, apart from one byte-identical pair
      in tz155k.
    - At 6 bits, `tanzania` had 26,131 pairs. The 73 that cross capture dates were false
      matches in every one inspected: other leaves, other diseases, down to 2 bits. Close-up
      leaves share one low-frequency layout, a vein across green. These pairs chained 20
      dates into one block and split anthracnose 0.987/0.0001/0.013. A 300-image sample
      could not show this; 6.3·10^9 pairs can.
    - At 2 bits one cross-date link remains. It is false too, and joins two healthy dates.
      Same-date copies at 4–6 bits are real, but under `blocked:date` they share a block
      anyway.
    - The four recipes, the two W1 test lines that pinned the draft value, and FR-006 now say
      2. `ibean_v1` keeps its bytes (no pair at 6). `makerere_v1` changes in one pair's
      group ids only; its blocks and split are those of 44, sha256 `0670c68b…4bde`.
49. **`tanzania_v1.jsonl`: 112,176 rows** (healthy 97,048, rust 8,243, anthracnose 6,885).
    sha256 `bf50c32a…acb1`.
    - Records: tz155k has 155,841 files, 112,219 distinct; tz59k has 59,071 files, 36,684
      distinct, of which 36,683 are also in tz155k. 53,018 rows carry 102,444 exact copies.
    - Excluded: 287 files of images held under two labels (`label_conflict`), and 5 that do
      not decode (`decode_error`), which the W1 note did not list.
    - Near duplicates: 2,394 groups at 2 bits (6,549 rows, the largest 50); 3 groups mix
      classes.
    - Blocks: 143 capture dates (train 67, val 38, test 38), one merged pair, and one row
      without EXIF, split unblocked. Shares: healthy and rust 0.70/0.10/0.20, anthracnose
      0.69/0.12/0.19. Days: healthy 122, rust 38, anthracnose 20, from 2022-10-21 to
      2024-09-25.
50. **`swm_v1.jsonl`: 684 rows, all `unknown_wm`** (White Mold 202, Mature Sclerotium + White
    Mold 482). 594 originals showing only apothecia or sclerotia are excluded by the class
    map, and so are the 1,278 drawn-box renderings. sha256 `601234ce…072c`.
51. **The frozen manifests** (spec 001 US-5; `data/manifests/FROZEN.jsonl`). All were frozen
    on 2026-09-19 at git `40ab86e` plus this commit's threshold of 2, after `validate` with
    file hashes passed. `makerere_crops_v1` was frozen after its parent.

    | manifest | rows | sha256 |
    |---|---|---|
    | `ibean_v1.jsonl` | 1,295 | `423a16666a4880ddf9b2bb934a29b614c834a1b9ba7170b4d8b994935bb86060` |
    | `makerere_v1.jsonl` | 15,402 | `0670c68b51f5da3cf8be67aebf0d5da1eae2c9db0828dc0018eceadb974b4bde` |
    | `makerere_crops_v1.jsonl` | 27,019 | `efab8bb093a6579cc22355931bac2f5de3c60b126835ea423353bff417752ea2` |
    | `swm_v1.jsonl` | 684 | `601234cedaa7e2a73d59520d95b5dbed9a51950ffc29d5c9b0947d4d5b5a072c` |
    | `tanzania_v1.jsonl` | 112,176 | `bf50c32afb273d6bed60aa288b4a0d29bce04d60e0371d1732887f605e36acb1` |

    "Frozen before any head" (SC-6): this list comes before every W3 head run. The one
    earlier head, the W2 slice's, was trained on `ibean_v1` with these same bytes, and it is
    never quoted (35).

## 2026-09-19 — W2 · the 224 caches for both backbones (ahead of schedule)

The work plan's "Thu–Fri — caches at 224", run on the dev laptop's GPU after the freeze (51).

52. **Every frozen manifest has a 224 cache for both backbones** (spec 002 SC-1). That is
    156,576 images per backbone, in fp16 autocast, batch 32, shards of 1,024. There is one
    key per backbone: `dinov2_l14_reg` `2b017210105b0b1f` (the slice's, 41) and `dinov3_l16`
    `718a3399a7ad5b68`.

    | manifest | rows | DINOv2 s | images/s | DINOv3 s | images/s |
    |---|---|---|---|---|---|
    | `ibean_v1` | 1,295 | 19.6 (41) | 66 | 18.1 | 72 |
    | `swm_v1` | 684 | 39.7 | 17 | 44.8 | 15 |
    | `makerere_crops_v1` | 27,019 | 297.6 | 91 | 299.5 | 90 |
    | `makerere_v1` | 15,402 | 207.2 | 74 | 295.7 | 52 |
    | `tanzania_v1` | 112,176 | 1,259.4 | 89 | 1,416.2 | 79 |
    | all five | 156,576 | 1,823 (30 min) | 86 | 2,074 (35 min) | 75 |

    - Each time includes loading the model and hashing its weights. Every cache loads against
      its manifest's frozen sha256, and its features are finite. Every extraction wrote one
      compute-log row, and each cache's rows add up to its `n_images`; no run had to resume
      (SC-3). A second `make cache` does nothing and adds no row (SC-2).
    - DINOv3 on `makerere_v1` shared the CPU with a probe trial and the test suite, so its 52
      images/s is not the machine's rate.
    - The `.npz` files hold 0.70 GB per backbone, 0.50 GB of it Tanzania's, in `data/cache/`
      (git-ignored). DINOv3's stay there (13).
    - The sidecars name git `ab5732a`, whose extraction code made them. This commit changes
      only how `--dataset` finds a manifest (53).
53. **Throughput. Spec 002 Clarification 8 is closed: shards of 1,024 and batches of 32 stay.**
    - At 224 the GPU is the limit on small pictures: about 90 images/s for both backbones on
      the crops and on Tanzania, with the GPU at 93–100 % and held at 55–62 W by its power
      cap, near 80 °C. DINOv3's 201 tokens run no faster than DINOv2's 261.
    - On larger pictures decoding is the limit: 74 images/s on Makerere (1.3 MP on average)
      and 15–17 on SWM (10 MP phone originals).
    - After about 40 minutes of load, the laptop lowered the GPU's power cap to about 40 W
      (1.4 GHz instead of 1.9). That is most likely why DINOv3 ran Tanzania at 79 images/s
      against DINOv2's 89. An overnight run should be planned at that lower rate.
    - At 224 a shard takes 11–15 s, so a laptop that sleeps loses little. A run uses 2.5 GB
      of the 8 GB of VRAM, the display's share included.
    - For W3 (11), a projection, not a measurement. Per image, 518 is about 6.2× the work of
      224 for DINOv2 (1,374 tokens instead of 261), and 512 about 5.8× for DINOv3 (1,029
      instead of 201). If the rate falls in proportion from the lower one above, the
      high-res pass over the five manifests takes about 3–3.5 h per backbone: one night for
      both, not the plan's GPU-day each. The timing at 518 that 11 asks for still has to
      confirm this.
    - `make cache DS=makerere_crops` stopped with a traceback: a crops manifest has no recipe
      (46), and `--dataset` read the version from the recipe. Now an id without a recipe
      names its one version on disk, and no file, or two versions, gives `missing_file`
      (exit 2). A frozen manifest whose bytes changed now stops with
      `frozen_manifest_modified` (exit 2), not a traceback. `test_cache.py` covers the id.

## 2026-09-19 — W2 · N4, the site-prediction probe (ahead of schedule)

The work plan's "Fri — site-prediction probe (N4)", on the 224 caches of 52. `ms.eval.probe`
(`make probe`) is new, with CPU tests on synthetic manifests and caches (`tests/test_probe.py`).

54. **The probe's protocol, fixed before the first number.**
    - There are two targets. `dataset` is iBean, Makerere or Tanzania (chance 1/3). It is
      fitted on each manifest's train split and scored on its test split, so the scored rows
      come from districts and dates the probe never saw (iBean's split is unblocked).
      `district` is the district within Makerere (chance 1/12). It uses the 10,118 rows
      whose XML records a district: rust and angular leaf spot, from every split. Healthy
      rows record none (FR-013). The districts are the split's blocks, so the train split
      holds none of the test's districts. This target is scored by 5-fold cross-validation
      instead, grouped by `phash_group`, stratified by district, seed 0.
    - Only the classes that every target has take part (for `dataset`: healthy and rust),
      and every (target, class) cell weighs the same, in the fit and in the score.
      Otherwise a set's class mix would pass for the set: Tanzania's rows are 92 % healthy,
      and Ntungamo's 87 % angular leaf spot.
    - The probe is multinomial logistic regression (lbfgs, C = 1) on the L2-normalised CLS,
      standardised. Nothing is tuned, so no validation split is spent.
    - The number is balanced accuracy: each target's recall averaged over its classes, then
      over the targets. The interval is a bootstrap over the scored rows within their cells
      (1,000 draws). It ignores that rows cluster by day and field, so it is narrower than
      the uncertainty about sites. The probe has no seeds to vary.
    - The N-table row has `metric = balanced_accuracy:<target>`, `head = logreg`, and a null
      `model_version`. A row over several manifests joins their paths, and their hashes,
      with `+` in both manifest fields. `split_rule` joins the scored rows' rules, or is
      `unblocked:5fold_by_phash_group` for the district. `classes` lists the targets, and
      `n_table.md` prints the chance level beside the metric. That gives one row per
      (backbone, target): the plan's "one row per backbone", once per target.
    - iBean's role is `test_only` (35), but the probe fits no disease model, so it may read
      iBean's train split.
55. **MLflow's local store is SQLite: `mlruns/mlflow.db`, with its artifacts in
    `mlruns/artifacts/`** (git-ignored), until piece 3 runs the server. `MLFLOW_TRACKING_URI`
    overrides it.
    - MLflow 3.16 refuses the plain `mlruns/` file store that the plan names ("maintenance
      mode"), unless `MLFLOW_ALLOW_FILE_STORE=true` forces it. Its default, `mlflow.db` in
      the working directory, would leave one store in every folder a command runs in.
    - N4 logs one run per probe in the experiment `piece4-n4-site-probe`. Each run has tags
      (the probe_id of its N-table row, the backbone, the target, the cache key and
      `research_only_until_c5`), the parameters, the metrics, and `probe.json` (the cells
      and the confusion matrix). A store that holds DINOv3 runs stays on the project's
      machines (13).
56. **What N4 measured at 224.** The numbers are quotable; the rows are in
    `results/n_table.jsonl` and the runs in MLflow.

    | backbone | dataset (chance 0.333) | district (chance 0.083) |
    |---|---|---|
    | `dinov2_l14_reg` | 0.983 (0.980–0.986) | 0.767 (0.749–0.787) |
    | `dinov3_l16` | 0.992 (0.990–0.994) | 0.770 (0.746–0.795) |

    - Both backbones give the set away almost every time. No iBean row is mistaken, and
      Tanzania's are mistaken 0.3–1 % of the time. Makerere is the hardest (recall 0.954 and
      0.980). Its misses go mostly to iBean, the other Ugandan set, also from Makerere
      University.
    - The district can be read too: three times out of four, against one in twelve by
      chance. The misses fall between districts visited on the same days. 40 % of Bugiri's
      rows go to Mayuge (25–26 Apr; both record a "Budaya" sub-county), and 26–28 % of
      Mbale's go to Pallisa (22 May; both record "Namunsi"), and some to Sironko. So what
      the features carry is largely the capture session: day, field, light and phone.
    - The fits converged in 18–89 lbfgs iterations and took 17–41 s each on the CPU.
    - Reading: a head can find the site in the features. A blocked N1 can still reward site
      cues that happen to go with a class, and Makerere's healthy rows, which all come from
      April (44), are such a cue. N4 says what can be read from the features, not what a
      head reads; N2 (W3) is the test of that.
    - Between the backbones, DINOv3 gives the set away slightly more (the intervals do not
      overlap), and the district is a tie. How N4 breaks a tie is for the verdict's rule
      (H9 §7, W4), not for this entry.

## 2026-09-19 — W2 · manifests and caches under DVC, handed to piece 3

The work plan's "Hand the manifest format to piece 3 for DVC". The task is P's; it was done at
the owner's request, and what stays open for piece 3 is listed in `store/README.md`.

57. **`dvc init` at the repository root, then `dvc add data/manifests data/cache`.**
    - The pointers are `data/manifests.dvc` (11 files, 174 MB: the five frozen manifests,
      their sidecars and `FROZEN.jsonl`) and `data/cache.dvc` (20 files, 1.40 GB: the ten
      224 caches of 52). The DVC cache is the local `.dvc/cache/` (1.5 GB of copies; NTFS
      has no reflinks).
    - There is no remote: that is piece 3's decision. A remote that receives
      `data/cache/dinov3_l16/` must be on the project's machines (13).
    - DVC analytics are off (`core.analytics false` in `.dvc/config`), and every DVC command
      here ran with `DVC_NO_ANALYTICS=1`, so nothing was sent.
    - The root `.gitignore` now ignores `/data/?*` instead of `/data/*`, and lets
      `data/*.dvc` through. For git the two patterns are the same. DVC's git layer
      (scmrepo over dulwich) reads `/data/*` as ignoring `data/` itself: `dvc status` found
      no pointer file ("no data tracked"), although `dvc add` had written both.
    - The hand-over is a section of `store/README.md`. It gives both formats in short, with
      pointers to specs 001 and 002, the DVC routine, and what stays open for piece 3: the
      remote, and whether `data/raw/` and `data/derived/` go under DVC.

## 2026-09-19 — W3 · specs 003 (heads) and 005 (the N-table), ahead of schedule

The work plan's "Spec S4.3 + S4.5": `specs/003-heads/spec.md` and
`specs/005-results-table/spec.md`. Their tests run on synthetic manifests and caches
(`tests/synthetic.py`) and are red until the W3 Heads and N1 tasks. `test_heads.py` has 21 red
and 2 green, which the W2 trainer's linear head already passes. `test_n_table.py` has 46 red.
The slice's N1 test in `test_slice.py` now expects spec 005's rows, so it is red too.

58. **The head command runs the W3 protocol by default.**
    - `--head` takes any of linear, proto and mix, and `--seed` several seeds. The defaults
      are all three heads and seeds 0–4 (work plan §4). Each (head, seed) pair is one run.
    - Every input is checked before the first run trains, so an input error writes nothing.
    - Asking for a test or held-out split is an exit with `test_split_access`, not a
      traceback.
    - The validation manifest has to be the training manifest (`val_not_in_domain`). Early
      stopping on another manifest would tune on N2's target.
59. **The mixture mixes two trained heads.** H8's "fixed 0.5/0.5 mixture" is
    p = 0.5 · p_linear + 0.5 · p_proto, from the linear and proto runs with the same inputs and
    seed. Nothing is trained, and its logits are log p.
    - A mixture trained end to end would make its parts differ from the linear and proto
      heads that the table reports.
    - A mix run trains its missing components first and records their run ids.
60. **The prototype head.**
    - It has K = 4 prototypes per class. A class's logit is the cosine of its nearest
      prototype over τ.
    - τ starts at 0.07 and is learned as `log_tau`, without weight decay: decay would pull τ
      towards 1 and flatten the logits.
    - The prototypes start from per-class k-means on the L2-normalised train features,
      seeded by the run's seed. A class with fewer than 4 train rows is `too_few_rows`.
61. **`run.json` v2 keeps every v1 key**, so the eval and the service read it unchanged.
    - It adds `components` and the effective recipe: a config without `sampling` records
      `class_balanced`, the default.
    - The trainer version is an input of the run id, so v2 runs get new ids. The W2 slice's
      run is not quotable anyway.
62. **The N-table holds the per-seed rows and their aggregates.**
    - An aggregate has `seed = null`, the mean over seeds 0–4 and a 95 % Student t interval
      (4 degrees of freedom), and joins the five run ids with `+`. With five values, a
      percentile interval would be just the minimum and the maximum.
    - Pieces 5 and 6 read the `.jsonl`, so the reported number has to be in it. The `.md`
      shows the aggregates and the rows that no aggregate covers.
63. **N1 is macro-F1 plus one recall per class**, so the rarest class is never averaged away.
    `n` is the number of rows each value is computed over.
64. **Rows are checked before they are appended** (`validate_row`; `NTableError`).
    - The fields are in order. The split rule is from spec 001's vocabulary or N4's
      `unblocked:5fold_by_phash_group`, and each manifest has one 64-hex hash.
    - Only known numbers and metrics pass, and no number is scored on a train or validation
      split.
    - N3 rows use held-out classes only. When the frozen list is given, every quotable row's
      manifests are frozen (spec 001 SC-5).
    - The committed rows pass, and a test checks the file.
65. **The verdict's rules are copied from the work plan, not from H9 §7.** H9 is not in the
    repository.
    - Spec 005 US-6 declares what the plan states: ≥ 2 pp on two of three N2 directions with
      non-overlapping intervals, ≥ 0.02 AUROC on N3, N4 then N6 as tie-breakers, and the
      licence gate.
    - Spec 005 Clarification 6 lists what H9 §7 still has to settle before N2's numbers
      exist.
66. **Left open.**
    - For N2: training on two manifests (Makerere + iBean) against iBean's `test_only` role,
      and what happens to Tanzania's anthracnose (spec 003 Clarifications 10–11; spec 005
      Clarification 7).
    - For the Heads task: the linear head's underfitting (34), measured before the five-seed
      runs (spec 003 Clarification 9).

## 2026-09-19 — W3 · the heads, five seeds at 224 (ahead of schedule)

The work plan's "Heads", built to specs 003 and 005 (58–66). The code of spec 005 came with it,
so `test_heads.py` (26 tests) and `test_n_table.py` (46) are green, with the rest of the suite:
202 passed, 1 skipped (the golden test, opt-in).

67. **The heads as spec 003 says, and one fix that running them showed.**
    - proto starts from per-class k-means (scikit-learn, k-means++, one initialisation,
      `random_state` = the seed). Then AdamW trains the prototypes and `log_tau`, with no
      weight decay on `log_tau`.
    - A mix's `head.pt` holds both components' states and params, so it loads on its own.
    - **k-means runs on one thread.** On several threads, scikit-learn adds their partial
      cluster sums in the order they finish. So the same seed gave other centroids from call
      to call: two calls on Makerere's features in one process disagreed. The first 60 runs
      therefore had proto heads that no second run could repeat. On Makerere they even kept
      other epochs (48 instead of 31).
    - On one thread the start is the same in every process. The trainer version went to 3,
      and the 60 runs were deleted and trained again. Seven of them were then trained a
      second time, in another process and under load, and gave the same weights bit for bit.
    - The deleted runs' compute-log rows stay (34), so the log holds 60 rows without a
      folder. `test_heads.py` now checks the start on 1 thread, 4 threads and all of them.
      Without the one-thread limit, k-means on that test's data gives other centroids on 4
      threads and on all of them.
    - `wallclock_s` counts a run's fit (or mixing) and its writing. A call reads its
      manifests and caches once for all its runs, and that reading is in no run's seconds
      (spec 003 US-6, amended).
    - Spec 003 was amended in three more places: FR-005 (how a mix's `head.pt` holds its
      components), FR-008 (`missing_file`) and SC-3 (every folder has one row; a deleted
      run's row stays).
68. **The N-table code of spec 005.**
    - `validate_row` checks every row before `append_rows` writes it. An invalid row raises
      `NTableError`, and nothing is appended.
    - `aggregate` adds the five-seed rows, and `unpaired` lists the N1/N2 pairs that are
      still incomplete.
    - `n_table.md` shows the aggregates (seeds `0–4`) and the rows that no aggregate
      covers, sorted by manifest, backbone, head and metric.
    - `make eval` writes macro-F1 and one recall per class for every run. A row is quotable
      only when its run is quotable and both manifests are frozen.
    - The probe (N4) now checks its row before MLflow and the compute log see it (exit 2), and
      a new test covers that.
    - `test_probe.py`'s synthetic sets are now blocked by real spec 001 keys (`date`,
      `district`, `region`), because the N-table accepts only those.
69. **The linear head on Makerere ends at the 50-epoch cap** (spec 003 Clarification 9,
    closed). Seed 0 was measured first, then all five seeds:
    - On Makerere (6,818 train rows, 27 steps per epoch) the linear head's best epoch is 50
      for four seeds of five on both backbones (48 for the fifth). The validation loss still
      falls by about 0.005 per 10 epochs, so the head is not done when H8's recipe stops it.
    - proto reaches its best by epochs 11–47 there. Its τ is learned down to 0.04–0.06, which
      makes its logits larger from the start.
    - On Tanzania (78,454 train rows, 307 steps per epoch) every head stops early. The linear
      head's selection score, the per-class validation loss, is best at epochs 3–5. Its
      validation macro-F1 still rises a little after that (seed 0, DINOv2: 0.982 at epoch 3,
      0.987 at epoch 13), because the loss grows on the few rows the head gets wrong.
    - H8's recipe stays: 50 epochs, and lr 1e-3 is already the top of its range. A fixed
      logit scale for the linear head (for example 1/0.07, proto's starting τ) or more
      epochs would change the recipe, so it is the owner's call. It would be a config change
      with new runs, a few minutes of CPU.
70. **The W3 runs at 224: 60 runs.** Both backbones × `makerere_v1` and `tanzania_v1` ×
    linear, proto and mix × seeds 0–4 (spec 003 SC-1), 238 s of CPU in all.
    - Rerunning a call does nothing and adds no row (SC-2).
    - Every run folder has one compute-log row (SC-3).
    - Makerere trains healthy and rust (6,818 rows; 1,471 for validation). Tanzania trains
      healthy, rust and anthracnose (78,454; 11,354).

    | backbone | set | head | best epoch, seeds 0–4 | validation macro-F1, mean (min–max) | τ | s per run |
    |---|---|---|---|---|---|---|
    | `dinov2_l14_reg` | Makerere | linear | 50, 50, 50, 50, 48 | 0.956 (0.953–0.958) | — | 1.4 |
    | | | proto | 34, 30, 40, 23, 47 | 0.962 (0.956–0.964) | 0.041–0.051 | 2.8 |
    | | | mix | — | 0.965 (0.963–0.968) | — | 0.1 |
    | | Tanzania | linear | 3, 3, 4, 5, 4 | 0.983 (0.982–0.984) | — | 5.9 |
    | | | proto | 6, 6, 2, 33, 6 | 0.991 (0.991–0.992) | 0.058–0.071 | 13.5 |
    | | | mix | — | 0.991 (0.990–0.992) | — | 0.2 |
    | `dinov3_l16` | Makerere | linear | 50, 50, 50, 50, 48 | 0.948 (0.947–0.949) | — | 1.2 |
    | | | proto | 35, 24, 44, 11, 39 | 0.950 (0.942–0.956) | 0.043–0.061 | 2.1 |
    | | | mix | — | 0.955 (0.951–0.958) | — | 0.1 |
    | | Tanzania | linear | 3, 3, 3, 3, 3 | 0.987 (0.986–0.988) | — | 3.5 |
    | | | proto | 13, 18, 24, 25, 6 | 0.992 (0.991–0.994) | 0.052–0.066 | 16.7 |
    | | | mix | — | 0.992 (0.990–0.993) | — | 0.2 |

    These validation numbers drive early stopping. They are not N1, which reads the test
    splits (the next task).

## 2026-09-19 — W3 · N1 at 224: three heads, five seeds (ahead of schedule)

The work plan's "N1", scored by `make eval` (spec 005 US-3) on the 60 runs of 70. The table
gained 254 rows: 210 per seed and 42 aggregates, all quotable, plus the two per-class rows of
the W2 slice's run, which is not quotable. The rows are in `results/n_table.jsonl` and
`n_table.md`.

71. **N1 follows the frozen manifests' split rules.**
    - Makerere is `blocked:district`: its test districts are Hoima, Lyantonde and Ntungamo
      (44).
    - `tanzania_v1` is `blocked:date`, with 38 capture dates in its test split (49). The task
      line said "region/session or `unblocked`". It was written before the data showed that
      Tanzania has neither region nor session (spec 001 Clarification 2), and it now says
      "by capture date".
72. **What N1 measured at 224**: the mean over seeds 0–4 with its 95 % t interval.

    | set | backbone | head | macro-F1 | recall healthy | recall rust | recall anthracnose |
    |---|---|---|---|---|---|---|
    | Makerere | `dinov2_l14_reg` | linear | 0.990 (0.989–0.991) | 0.995 (0.994–0.996) | 0.985 (0.983–0.987) | — |
    | | | proto | 0.993 (0.991–0.995) | 0.994 (0.990–0.997) | 0.993 (0.988–0.998) | — |
    | | | mix | 0.994 (0.991–0.996) | 0.996 (0.994–0.998) | 0.991 (0.987–0.995) | — |
    | | `dinov3_l16` | linear | 0.988 (0.987–0.989) | 0.993 (0.991–0.994) | 0.983 (0.980–0.987) | — |
    | | | proto | 0.991 (0.990–0.992) | 0.988 (0.985–0.992) | 0.995 (0.993–0.997) | — |
    | | | mix | 0.992 (0.991–0.993) | 0.990 (0.988–0.991) | 0.994 (0.991–0.996) | — |
    | Tanzania | `dinov2_l14_reg` | linear | 0.994 (0.994–0.995) | 1.000 (1.000–1.000) | 0.977 (0.976–0.977) | 0.994 (0.994–0.994) |
    | | | proto | 0.996 (0.995–0.996) | 1.000 (1.000–1.000) | 0.983 (0.981–0.985) | 0.994 (0.994–0.995) |
    | | | mix | 0.995 (0.995–0.996) | 1.000 (1.000–1.000) | 0.981 (0.979–0.983) | 0.994 (0.994–0.994) |
    | | `dinov3_l16` | linear | 0.992 (0.990–0.993) | 0.998 (0.997–0.999) | 0.985 (0.985–0.986) | 0.992 (0.992–0.992) |
    | | | proto | 0.996 (0.995–0.996) | 1.000 (1.000–1.000) | 0.983 (0.980–0.987) | 0.995 (0.994–0.995) |
    | | | mix | 0.996 (0.996–0.996) | 1.000 (1.000–1.000) | 0.985 (0.984–0.987) | 0.994 (0.993–0.995) |

    - Every head scores at least 0.988 macro-F1 on both sets. proto and mix are above linear
      everywhere, by 0.001–0.005. That fits the linear head stopping at the epoch cap on
      Makerere (69). The two backbones are within 0.004 of each other.
    - The test rows are 1,050 healthy and 965 rust on Makerere, and 19,410 healthy, 1,649
      rust and 1,309 anthracnose on Tanzania. "1.000" for Tanzania's healthy rows is 0.9999:
      about 2 misses in 19,410.
    - **The rarest class is not the weakest.** Tanzania's rarest class, anthracnose (4,751
      train rows), is recalled at 0.992–0.995. Its rust (5,770 train rows) is the weakest
      class of the table, at 0.977–0.985.
    - On Makerere the two classes are nearly the same size. Rust is the weaker one for every
      DINOv2 head and for DINOv3's linear head, and healthy for DINOv3's proto and mix.
    - The t interval is not clipped. A recall of 0.9999 with a small spread can show an upper
      bound of 1.0001 in the `.jsonl` (DINOv3's proto and mix on Tanzania's healthy rows).
73. **How to read N1: high, and not yet evidence of skill across sites.**
    - **Makerere.** Its test split carries the class ↔ trip confound (44). All 1,050 healthy
      test rows are Hoima's (the April trip), and 959 of the 965 rust rows are Lyantonde's and
      Ntungamo's (the May trip). N4 showed that the features carry the capture session (56).
    - The 6 rust rows from Hoima were taken on the healthy rows' trip. The DINOv3 heads
      recall all 6. The DINOv2 heads recall 4 of 6 (linear), 5 on average (mix) and 5.4 on
      average (proto). That is weak evidence, on 6 rows, that the heads see rust and not
      only the trip.
    - **Tanzania.** Blocking by capture date keeps each day's copies on one side. A farm
      revisited on the next day can still sit on the other side (spec 001 edge cases).
    - So N1 bounds what the heads can do within a set. N2, across sets, tests whether the
      features carry the disease rather than the site, and the guard-rail check compares
      the two.
    - Abstention rates come with S4.4 (W4).

## 2026-09-19 — W3 · N2 at 224: three heads, five seeds (ahead of schedule)

The work plan's "N2". Four questions the plan left open were put to the owner, who decided
them on 2026-09-19 (74). `make eval` wrote 300 per-seed rows and 60 aggregates, all quotable.
Every N1 aggregate now has its N2 pair (spec 005 US-4.1).

74. **The owner's four decisions, and how N2 carries them out.**
    - **Makerere + iBean train together, and the runs are quotable.** Spec 001 gives iBean
      the role `test_only`, and it stays so. As a training source it now joins Makerere in
      this one direction (spec 001 Clarification 10, amended).
    - `ms.heads.train` takes several `--train-manifest`s: the first one's role decides, and a
      further one may be `test_only`. Each trains on its train split and stops early on its own
      validation split (spec 003 US-5.1, closing Clarification 10).
    - The 30 runs (two backbones × three heads × five seeds) train on 7,422 rows (Makerere's
      6,818 and iBean's 604) and validate on 1,558. The linear head again ends at the epoch
      cap (69).
    - A run over two manifests joins their paths and hashes with `+` in `run.json`, as the
      N-table does. Its model version names both: `man-0670c6-423a16`. A single-manifest run
      keeps its run id, so the 60 runs of 70 stayed as they were.
    - **The crop level reports rust recall only.** `makerere_crops_v1` holds no healthy crop
      (46), so `Tanzania → Makerere rust crops` has one row per head and seed: `recall:rust`.
    - **Tanzania's three-class heads decide between healthy and rust**, the classes both sides
      have. Rows of other classes are not scored: "anthracnose only where both sides have it"
      (spec 003 Clarification 11, closed).
    - **The targets are the frozen test splits, and every row of iBean.** Makerere's and
      Tanzania's test splits are the rows N1 reads. iBean is a test-only set, so all 863 of
      its healthy and rust rows are scored, with `test_split = all` (spec 005 FR-001).
    - The directions live in `configs/eval.yaml` (spec 005 US-7, FR-010). `make eval`
      scores a run on every direction whose training manifests are its own. N1 is scored only
      for runs trained on one manifest.
    - H9 §7 is not in the repository, so the verdict's rules still copy the work plan's
      summary. N2's numbers now exist before that text was checked; spec 005 Clarification 6
      stays open.
75. **What N2 measured at 224**: the mean over seeds 0–4 with its 95 % t interval.

    | direction | backbone | head | macro-F1 | recall healthy | recall rust |
    |---|---|---|---|---|---|
    | Tanzania → Makerere | `dinov2_l14_reg` | linear | 0.362 (0.360–0.364) | 1.000 (1.000–1.000) | 0.018 (0.016–0.020) |
    | | | proto | 0.365 (0.341–0.390) | 1.000 (1.000–1.000) | 0.021 (-0.002–0.045) |
    | | | mix | 0.363 (0.347–0.379) | 1.000 (1.000–1.000) | 0.018 (0.003–0.034) |
    | | `dinov3_l16` | linear | 0.360 (0.359–0.361) | 1.000 (1.000–1.000) | 0.016 (0.015–0.017) |
    | | | proto | 0.346 (0.344–0.348) | 1.000 (1.000–1.000) | 0.003 (0.001–0.005) |
    | | | mix | 0.347 (0.344–0.350) | 1.000 (1.000–1.000) | 0.004 (0.002–0.007) |
    | Tanzania → Makerere rust crops | `dinov2_l14_reg` | linear | — | — | 0.475 (0.457–0.493) |
    | | | proto | — | — | 0.417 (0.365–0.469) |
    | | | mix | — | — | 0.435 (0.400–0.471) |
    | | `dinov3_l16` | linear | — | — | 0.485 (0.479–0.492) |
    | | | proto | — | — | 0.319 (0.283–0.354) |
    | | | mix | — | — | 0.367 (0.343–0.392) |
    | Tanzania → iBean (all) | `dinov2_l14_reg` | linear | 0.545 (0.539–0.550) | 1.000 (1.000–1.000) | 0.228 (0.221–0.236) |
    | | | proto | 0.525 (0.459–0.591) | 1.000 (0.998–1.001) | 0.206 (0.122–0.291) |
    | | | mix | 0.527 (0.480–0.573) | 1.000 (0.998–1.001) | 0.207 (0.147–0.267) |
    | | `dinov3_l16` | linear | 0.516 (0.511–0.520) | 1.000 (1.000–1.000) | 0.193 (0.187–0.198) |
    | | | proto | 0.531 (0.514–0.548) | 1.000 (0.998–1.001) | 0.211 (0.191–0.232) |
    | | | mix | 0.520 (0.510–0.531) | 1.000 (1.000–1.000) | 0.199 (0.186–0.211) |
    | Makerere + iBean → Tanzania | `dinov2_l14_reg` | linear | 0.889 (0.881–0.897) | 0.960 (0.956–0.964) | 0.980 (0.979–0.981) |
    | | | proto | 0.727 (0.651–0.803) | 0.922 (0.904–0.939) | 0.657 (0.469–0.845) |
    | | | mix | 0.861 (0.843–0.879) | 0.949 (0.941–0.957) | 0.958 (0.935–0.981) |
    | | `dinov3_l16` | linear | 0.802 (0.794–0.810) | 0.910 (0.905–0.915) | 0.993 (0.993–0.993) |
    | | | proto | 0.718 (0.652–0.783) | 0.857 (0.803–0.910) | 0.917 (0.838–0.996) |
    | | | mix | 0.782 (0.736–0.829) | 0.895 (0.862–0.929) | 0.992 (0.991–0.993) |

    The scored rows: Makerere's test split, 1,050 healthy and 965 rust; its rust crops, 2,655;
    iBean, 427 healthy and 436 rust; Tanzania's test split, 19,410 healthy and 1,649 rust.
76. **How to read N2.**
    - **The Tanzania heads call foreign rust healthy.** On Makerere's frames they find 0–2 %
      of the rust, on iBean 19–23 %, and on Makerere's rust crops 32–49 %. Their own
      three-class argmax picks anthracnose for at most 1.6 % of these rows. So the choice
      between healthy and rust (74) does not cause this; it is how the heads read the
      pictures.
    - Tanzania's pictures are close-ups of single leaves (1000×750). Makerere's crops are
      nearer to that than its whole frames, and they fare best of the three.
    - **Makerere + iBean → Tanzania holds up far better, at 0.72–0.89.** Its healthy recall
      of 0.86–0.96 on 19,410 healthy rows still means about 780–2,780 false rust calls, and
      that is what pulls macro-F1 below the rust recall.
    - The proto heads vary a lot across seeds in this direction. DINOv2's rust recall
      interval is 0.47–0.85.
    - The t interval is not clipped (72), so it can dip below 0 as well (DINOv2's proto on
      Makerere's rust, −0.002).
    - **Between the backbones**, DINOv3 is ahead in two cells only, inside the intervals:
      proto on iBean (+0.006) and linear on the crops (+0.010). DINOv2 is ahead elsewhere, by
      up to 0.087 (linear, Makerere + iBean → Tanzania). The verdict is W4's, by its rules.
    - N1 is 0.988–0.996 and N2 is 0.35–0.89, so N1 ≫ N2. What that does and does not show
      is the guard-rail check's, the next task.

## 2026-09-19 — W3 · the guard-rail check: N1 against N2

The work plan's "Guard-rail check": if N1 ≫ N2 does not appear, look for leakage before
anything else. It compares the quotable aggregates of 72 and 75: macro-F1, and rust recall for
the crops.

77. **N1 ≫ N2 appears for every backbone and head, so no leakage hunt was triggered.**

    | backbone | head | N1 Makerere | N1 Tanzania | N2 T → M | N2 T → M rust crops (recall) | N2 T → iBean | N2 M + iBean → T |
    |---|---|---|---|---|---|---|---|
    | `dinov2_l14_reg` | linear | 0.990 | 0.994 | 0.362 | 0.475 | 0.545 | 0.889 |
    | | proto | 0.993 | 0.996 | 0.365 | 0.417 | 0.525 | 0.727 |
    | | mix | 0.994 | 0.995 | 0.363 | 0.435 | 0.527 | 0.861 |
    | `dinov3_l16` | linear | 0.988 | 0.992 | 0.360 | 0.485 | 0.516 | 0.802 |
    | | proto | 0.991 | 0.996 | 0.346 | 0.319 | 0.531 | 0.718 |
    | | mix | 0.992 | 0.996 | 0.347 | 0.367 | 0.520 | 0.782 |

    - Against the N1 of the set its heads trained on, each N2 direction loses macro-F1:
      - Tanzania → Makerere: 0.63–0.65;
      - Tanzania → iBean: 0.45–0.48;
      - Makerere + iBean → Tanzania: 0.10–0.27, measured against Makerere's N1.
    - The rust crops are recalled at 0.32–0.49, against 0.98–0.99 within the sets.
    - For every head, every N1 interval lies above every N2 interval. The smallest gap is
      DINOv2's linear head on Makerere + iBean → Tanzania: 0.990 against 0.889.
78. **Two cheap checks, for the record.** The rule did not call for them, but an N1 near 0.99
    is the kind of number H8's risks call "too good".
    - Within each frozen manifest (Makerere, Tanzania, iBean and the Makerere crops), no
      pHash group, block or image straddles train, val and test, and no parent's crops do.
      This was recomputed from the rows; spec 001 already checks it at build.
    - Across the sets, `overlap` finds no exact and no near duplicate (pHash within 2 bits)
      for Tanzania and Makerere, Tanzania and iBean, Makerere and iBean, or Tanzania and the
      Makerere crops. pHash at 2 bits finds re-encoded and resized copies, not new crops or
      re-framings of a scene.
    - So N1's height is not duplicates. It does include cues that belong to a set, which the
      features carry (N4, 56), and on Makerere the class ↔ trip confound (73).
    - The gap is the size of the shift between sets. N1 is the within-set ceiling, and N2 is
      the number to read for a new farm.

## 2026-09-19 — W3 · the verdict's rules, from H9 §7 (the owner's decision)

The owner pointed to H9 §7 and settled how to read it on 2026-09-19. Spec 005 US-6 now copies
§7 verbatim, with its file and date, and states the rules the W4 verdict will compute. The
verdict itself belongs to W4 and was not run.

79. **The verdict follows H9 §7, read the way that keeps the champion.**
    - **Source.** `D:\Life-OS\FUND-GRANT\30_projects\AgriRobot\05_sourses\
      10.03_dr_choroby-fasoli-sparag\88_H9_backbone-decision-dinov2-vs-dinov3.md`, §7, last
      changed 2026-09-14 14:00 (+02:00). The N-table's first number is from 2026-09-19, so
      the criteria were written before any number, as the work plan asks.
    - **The owner's readings:**
      - **N2**: macro-F1 on the shared classes. The challenger must be ≥ 2 pp ahead on the
        mean over the three directions. It must also be ≥ 2 pp ahead, with non-overlapping
        five-seed intervals, in at least two of them. H9's sentence can be read as one
        condition or as two; this reading takes both.
      - **N3**: ≥ 0.02 higher AUROC on both decision scores, confidence and kNN distance.
        H9 says "the abstention score", in the singular.
      - **Combining.** The challenger wins if it wins N2 or N3 and the champion wins neither;
        the champion wins a criterion by the same rule, with the roles swapped. A split goes
        to N4: the backbone whose probe accuracy is ≥ 2 pp lower wins it. Anything else, no
        wins at all included, goes to the champion. H9 does not say how its criteria combine.
      - **N6** is reported only (H9 §7 criterion 4). The work plan's W4 line and H8 §6.7's
        summary call it a tie-breaker, but H9 governs. The W4 line was left as it is.
      - **The rows**: 224 px, CLS, coverage 1.0, and the challenger must win with all three
        heads.
      - **Outcome.** A winning challenger joins the model set and never evicts the champion
        this season. The licence is a hard gate for anything that ships. The report states
        both backbones' numbers whatever the result.
    - **Where H9 is silent or ambiguous, the owner chose the reading that keeps the
      champion** (H9: "a tie goes to the champion"). Four more points needed a reading
      before the rules can be computed, and they take the same side:
      - N4 has two targets, the dataset and the district. The tie-breaker needs the probe
        to be ≥ 2 pp lower on both.
      - N3 is counted for each held-out set, angular leaf spot and white mould, not pooled.
      - The three directions are the frame-level ones. The crop-level rows measure rust
        recall, one class, not the macro-F1 on shared classes that H9 names, so they are
        reported and not counted.
      - "Averaged over transfer directions" is the plain mean of the three directions'
        five-seed means. "Do not overlap" means the leader's `ci_low` lies above the other's
        `ci_high` (spec 005 US-2's 95 % t intervals).
    - Nothing computes the verdict yet. `verdict.md` and its tests come with the W4 task.

## 2026-09-19 — W3 · max_epochs 300: the heads again, N1 and N2 again (the owner's decision)

The owner's answer to 69: raise the cap, so that early stopping and not the cap ends every
run. The heads at 224 were trained again, N1 and N2 scored again, and the guard-rail check run
again. The 50-epoch rows stay in the N-table, marked superseded. The verdict (W4) reads the
new runs.

80. **`max_epochs` 300 for every head. The 50-epoch rows are kept and marked superseded.**
    - `linear.yaml` and `proto.yaml` now say `max_epochs: 300`. H8 §6.5 says 50; H8 is the
      owner's to update.
    - lr 1e-3, patience 10 and early stopping on the in-domain validation split are
      unchanged. `mix.yaml` has no epochs: a mix takes its components' weights.
    - A changed config is a new run id (spec 003 FR-002). So 90 new runs were trained:
      both backbones × Makerere, Tanzania and Makerere + iBean × three heads × five seeds.
      That took 495 s of CPU, each run with its compute-log row. The old runs' folders stay
      under `data/heads/`.
    - **No run reached 300.** All 60 fitted runs ended by early stopping, the longest after
      208 epochs.
    - On Makerere and on Makerere + iBean, the linear heads now keep epochs 108–198 (the
      longest being DINOv2's on Makerere + iBean). Their validation macro-F1 rose by about
      0.007.
    - The Tanzania heads and most proto runs stopped before 50 anyway, so they keep the same
      best epoch and the same weights. Their numbers did not move.
    - **Superseded, not deleted.** N-table rows gained a `superseded` field (spec 005 FR-001,
      US-8). `python -m ms.eval.supersede` marked every row that names a run trained with the
      50-epoch configs (linear `427816d6…`, proto `c20fb209…`) or a mix of one: 612 rows,
      252 of N1 and 360 of N2, all with the same reason text. No other field changed; this
      was checked row by row. The W2 slice's rows and the four N4 rows are not marked.
    - The mark is part of what makes seeds one number, so old and new seeds never share an
      aggregate. `unpaired`, `n_table.md`'s sections and the verdict read the current rows
      only. The md counts the superseded ones at its end.
    - The table's rule is now "never deleted, and never changed except for this mark"
      (spec 005 edge cases).
81. **What moved, and what did not.** `make eval` scored the 90 new runs: 612 new rows, all
    quotable, and every N1/N2 pair complete.
    - 18 of the 102 current aggregates differ from their superseded twins by 0.0005 or more.
      All are in cells whose runs had been stopped at the cap: the linear heads, their
      mixes, and DINOv2's proto on Makerere + iBean, whose seed 3 now keeps epoch 70.

    | number | direction | backbone | head | macro-F1, 50 epochs | macro-F1, 300 cap |
    |---|---|---|---|---|---|
    | N1 | Makerere | `dinov2_l14_reg` | linear | 0.990 (0.989–0.991) | 0.991 (0.990–0.993) |
    | N1 | Makerere | `dinov2_l14_reg` | mix | 0.994 (0.991–0.996) | 0.994 (0.992–0.997) |
    | N1 | Makerere | `dinov3_l16` | linear | 0.988 (0.987–0.989) | 0.991 (0.990–0.993) |
    | N2 | Makerere + iBean → Tanzania | `dinov2_l14_reg` | proto | 0.727 (0.651–0.803) | 0.733 (0.666–0.799) |
    | N2 | Makerere + iBean → Tanzania | `dinov3_l16` | linear | 0.802 (0.794–0.810) | 0.876 (0.867–0.884) |
    | N2 | Makerere + iBean → Tanzania | `dinov3_l16` | mix | 0.782 (0.736–0.829) | 0.813 (0.772–0.853) |

    - The other 12 moves are recall rows, most of them in the same cells. The biggest is
      DINOv3's linear head on Tanzania's healthy rows: 0.910 → 0.952. That means far fewer false rust calls,
      which is where its N2 gain comes from.
    - DINOv2's linear head on Makerere + iBean → Tanzania stays at 0.889. Longer training
      moved its rust recall (0.980 → 0.981) but not its macro-F1.
    - Unchanged: N1 on Tanzania and every Tanzania → X direction, because the Tanzania heads
      stopped early before. Tables 72 and 75 hold for them.
    - Between the backbones, the gap on Makerere + iBean → Tanzania narrowed. Linear is now
      0.889 against 0.876, where it was 0.889 against 0.802. The verdict is W4's, by 79's
      rules.
82. **The guard-rail check again: N1 ≫ N2 still appears for every backbone and head.**

    | backbone | head | N1 Makerere | N1 Tanzania | N2 T → M | N2 T → M rust crops (recall) | N2 T → iBean | N2 M + iBean → T |
    |---|---|---|---|---|---|---|---|
    | `dinov2_l14_reg` | linear | 0.991 | 0.994 | 0.362 | 0.475 | 0.545 | 0.889 |
    | | proto | 0.993 | 0.996 | 0.365 | 0.417 | 0.525 | 0.733 |
    | | mix | 0.994 | 0.995 | 0.363 | 0.435 | 0.527 | 0.861 |
    | `dinov3_l16` | linear | 0.991 | 0.992 | 0.360 | 0.485 | 0.516 | 0.876 |
    | | proto | 0.991 | 0.996 | 0.346 | 0.319 | 0.531 | 0.718 |
    | | mix | 0.991 | 0.996 | 0.347 | 0.367 | 0.520 | 0.812 |

    - Against the N1 of their own set, the N2 directions lose 0.10–0.65 macro-F1, and every
      N1 interval lies above every N2 interval. No leakage hunt was triggered.
    - The smallest gap is still DINOv2's linear head on Makerere + iBean → Tanzania, 0.102.
      DINOv3's linear gap there fell from 0.186 to 0.116.
    - The duplicate checks of 78 depend only on the frozen manifests, which have not changed,
      so they hold as they were.
