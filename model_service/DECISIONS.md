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
`interface/schema/v0/observation_frame.schema.json`), which is marked "not yet reviewed by
P + E or the robotics side". The confirmation itself is still pending (M + P): if piece 1's
review changes an answer, the entry below changes with it.

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
31. **`frame_uid` is still open on piece 1's side, and piece 4 supports the rename.** In ROS,
    `frame_id` names a coordinate frame, and v0 has both `frame.frame_id` (the image) and
    `metadata.pose.frame_id` (the TF frame). Until piece 1 answers, `frame.frame_id` is the
    image id. S4.6 reads it in one place and echoes it in the response, so a rename in v0.1
    (W10) is a one-line change there.
32. **The frame's `sha256` is piece 4's content hash.** Piece 1 defines it as the hash of the
    image bytes exactly as stored at `uri` (their choice 10), the same definition as spec 001's
    `sha256` and spec 002's `sha256[N]`. S4.6's cached-hash path looks it up in the caches
    after recomputing it from the bytes it fetches, so a wrong hash in a request cannot select
    another image's features.
