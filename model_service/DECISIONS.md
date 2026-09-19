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
