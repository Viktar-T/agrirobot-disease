# Feature specification: 002 — Feature cache

**Branch**: `002-cache` · **Created**: 2026-09-19 · **Status**: Implemented 2026-09-19 (`ms.cache`, `ms.cache.extract`, the W2 slice); acceptance tests in `model_service/tests/test_cache.py` green, the golden test opt-in and green; SC-1 met on 2026-09-19 for both backbones and every frozen manifest, at 224 (DECISIONS 52) and at 518/512 (DECISIONS 83)
**Input**: H8 §6.4 (backbones, features), `docs/piece4-work-plan.md` S4.2 and §4 (conventions), DECISIONS 11–13 (extraction on the dev laptop, fp16 + SDPA, the key, transformers 5.x, DINOv3 terms), the backbone configs (`configs/backbones/*.yaml`), spec 001 (manifests). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Every number in piece 4 comes from frozen-backbone features, and computing them is the only expensive step: a GPU-night per backbone at high resolution (DECISIONS 11). The cache makes each feature computed once for a given model, input and precision, and makes it impossible to mix features from two different models without noticing. The model is its weights; the input is the resolution and the preprocessing; the precision is the compute dtype. It also leaves the trail N7 needs: time and throughput per extraction, in the compute log.

## Commands

`python -m ms.cache.extract`; `make cache BB=<backbone_id> RES=<res> DS=<manifest>` runs it.

    --backbone <id> --res <n> (--dataset <manifest> | --manifest <file.jsonl>)
    [--raw-root data/raw] [--cache-root data/cache] [--config-dir model_service/configs/backbones]
    [--device cuda|cpu] [--dtype float16|float32] [--batch-size N] [--shard-size N] [--max-shards K]
    [--compute-log model_service/results/compute_log.jsonl]

- `--dataset <id>` reads `data/manifests/<id>_v<N>.jsonl`, where `N` is the version in the manifest's recipe. A crops manifest has no recipe (spec 001 FR-012): its id names its one version on disk. No file, or several versions, is an input error (`missing_file`, exit 2).
- `--dtype` defaults to the backbone config's `dtype`, `--batch-size` to its `batch_size`, and `--device` to `cuda` when present, else `cpu`.
- Exit codes: 0 when the cache is complete (computed, resumed to the end, or already there); 1 when the run stopped early (`--max-shards`) and the next run resumes; 2 on an input error, with nothing new written.

## User scenarios and testing

### US-1: One cache per key (priority P1)

As the ML role, I extract features for a manifest once per model and preprocessing, and I can always tell which model, input and precision produced a feature file.

1. **Given** a backbone config and the fixture manifest, **when** `extract` runs, **then** `<cache-root>/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.npz` exists.
   - It holds `cls` and `meanpatch` as float16 `[N, D]`, and `image_id` and `sha256` as `[N]`, all in manifest row order (every split).
   - The sidecar `<manifest>_v<N>.meta.json` holds every field of FR-006, and its `shapes` match the arrays.
2. **Given** that cache, **then** the two token types are as follows. `cls` is the first token of the model's last hidden state (after its final norm). `meanpatch` is the mean of the last (res / patch)² tokens: the patch tokens only, with CLS and the register tokens left out.
3. **Given** a complete cache for the same key and the same manifest bytes, **when** `extract` runs again, **then** it exits 0 and computes nothing, rewrites nothing, and adds no compute-log row.
4. **Given** a change in any key field (FR-002: backbone id, weights, resolution, preprocessing, compute dtype), **then** the key changes and the new cache goes into a new folder, leaving the old cache as it was. Library versions, batch size, shard size and device do not change the key.
5. **Given** a manifest whose sha256 differs from the one in the cache's sidecar (a draft rebuilt before its freeze), **then** its cache is recomputed.

### US-2: Resumable (P1)

As the ML role, I run a GPU-night on a laptop that may sleep, and I lose at most one shard.

1. **Given** `--shard-size 8 --max-shards 2`, **then** the run computes two shards and exits 1, and no final `.npz` exists yet.
   - **When** the run is repeated, **then** it computes only the missing shards and exits 0.
   - The final arrays equal those of one uninterrupted run.
   - Each run adds its own compute-log row (US-3).
   - The sidecar's `wallclock_s` is the sum over the runs, and `runs` counts them.
   - A run killed mid-shard (Ctrl-C, sleep) keeps every completed shard as well.

### US-3: The compute-log row (P1)

1. **Given** a run that computed images, **then** it appends one `step = "extract"` row to the compute log through `ms.compute_log`, with these fields:
   - `backbone_id`, `res`, `dataset` (the manifest id);
   - `n_images_or_runs`: the images computed in this run, not those taken from earlier shards;
   - `wallclock_s`, `device`, `vram_gb`, `notes`;
   - a `cache` object: `cache_key`, `manifest`, `manifest_sha256`, `compute_dtype`, `batch_size`, `shard_size`, `images_per_s`, `shards_done_before`, `complete`.

   A run that computed nothing (US-1.3) or failed on input (US-4.1) writes no row.

### US-4: Integrity (P1)

1. **Given** an image whose bytes differ from its manifest row's `sha256`, **then** the run stops with exit 2 and `sha256_mismatch`, naming the row and the path. A missing file does the same with `missing_file`. The shard it belongs to is not written, and neither is a compute-log row.
2. **Given** a cache, **when** a consumer calls `load_cache(npz, manifest_sha256=…)`, **then** it gets the arrays and the sidecar. It gets `CacheMismatchError` in three cases:
   - the arrays do not match the sidecar's `shapes` or dtype;
   - the sidecar's `cache_key` does not match its own key fields;
   - the given `manifest_sha256` is not the one the cache was built from.

### US-5: Preprocessing is the string (P1)

1. **Given** an image and a `preprocess_string`, **when** `preprocess(image, preprocess_string)` runs, **then** it returns a float32 tensor `[3, n, n]` made as FR-004 describes. The EXIF orientation is applied first. An unknown token or value raises `ValueError`. The cache and the service (006) call the same function.

### US-6: Golden features (P2, opt-in)

1. **Given** the DINOv2 weights on the machine and `MS_GOLDEN=1`, **when** the fixture is extracted with `dinov2_l14_reg` at 224, **then** each CLS vector has cosine similarity ≥ 1 − `tolerance` with the committed `tests/fixtures/ibean_30/golden/dinov2_l14_reg_224_cls.npy`. The tolerance is committed beside the vectors and comes from the fp16/fp32 comparison on the fixture (DECISIONS 11). This test is skipped in CI and run before any relock of torch or transformers; a failure means a new cache version (DECISIONS 12). The golden vectors are DINOv2 only: nothing derived from DINOv3 enters git (DECISIONS 13).

### Edge cases

- **Tokens.** Both pinned backbones have 4 register tokens, so the last hidden state is `[CLS, 4 registers, (res/patch)² patches]`: 261 tokens for DINOv2 at 224, 1374 at 518; 201 for DINOv3 at 224, 1029 at 512. Any other length stops the run, because a model that pads or drops tokens would make `meanpatch` wrong without anyone noticing.
- **fp16 and CPU.** `float16` means autocast on CUDA. On CPU, only `float32` is accepted (exit 2).
- **Large sets.** The final `.npz` for Tanzania at D = 1024 holds about 112k × 2 × 2 KB ≈ 0.5 GB. Shards keep peak memory at one shard.
- **DINOv3.** Its caches stay in `data/cache/`, which is git-ignored, on the project's machines; the sidecar records `research_only_until_c5`.

## Requirements

### Functional

- **FR-001 Location.** `data/cache/<backbone_id>/<resolution>/<cache_key>/<manifest>_v<N>.npz` plus `<manifest>_v<N>.meta.json`. While a run is under way, the shards and their state live in `<manifest>_v<N>.shards/` beside them; that folder is removed once the `.npz` is complete. Because the key is in the path, a changed key never overwrites a cache.
- **FR-002 Key.**
  - The key has five fields, in this order: `backbone_id`, `weights_sha256`, `resolution`, `preprocess_string`, `compute_dtype` (DECISIONS 11).
  - `cache_key` is the first 16 hex digits of the sha256 of `json.dumps([those five])` in UTF-8, with `json` defaults. `cache_key(backbone_id, weights_sha256, resolution, preprocess_string, compute_dtype)` computes it and takes nothing else.
  - Library versions, device, batch size and shard size stay out of the key (DECISIONS 12); they go to the sidecar.
- **FR-003 Weights.** The model is `AutoModel.from_pretrained(hf_id, revision=<pinned>, attn_implementation=<config>)` in float32. `compute_dtype = float16` runs it under `torch.autocast("cuda", float16)`; `float32` runs it without autocast. `weights_sha256` is the sha256 of the weight file it is loaded from (`model.safetensors`, about 1.2 GB at both pins). With several weight files, it is the sha256 of their sorted `<file name> <sha256>` lines.
- **FR-004 Preprocessing.** The tokens are `;`-separated `name=value` pairs, applied in order:
  - First, always: decode the image, apply its EXIF orientation, convert it to RGB and make a uint8 tensor.
  - `resize_short=n`: the shorter side becomes `n` and the aspect ratio is kept (torchvision `resize`, bicubic, antialias on).
  - `center_crop=n`: an `n × n` centre crop.
  - `norm=imagenet`: divide by 255, subtract (0.485, 0.456, 0.406) and divide by (0.229, 0.224, 0.225).

  Any other name or value is an error. Changing what a token does needs a new token, so the string, and with it the key, changes too.
- **FR-005 Arrays.** The `.npz` contains:
  - `cls` and `meanpatch`: float16 `[N, D]`. Both are computed in float32 from the last hidden state (US-1.2) and then cast.
  - `image_id` and `sha256`: fixed-width unicode `[N]`. `sha256` lets 006 answer a frame whose hash is cached without a GPU.

  The rows are the manifest's rows in its order, every split included. It is read through the spec 001 loader with `purpose = extract`. The file loads with `numpy.load(..., allow_pickle=False)`.
- **FR-006 Sidecar.** The sidecar holds:
  - `meta_json_version`, `cache_key` and the five key fields;
  - `hf_id`, `revision`, `model_class`, `attn_implementation`, `research_only_until_c5`;
  - `token_types` (`["cls", "meanpatch"]`), `storage_dtype` (`float16`), `embed_dim`, `n_images`, `n_patch_tokens`, `n_register_tokens`, `shapes` (per token type, `[N, D]`);
  - `manifest` (file name), `manifest_sha256`, `dataset`;
  - `transformers_version`, `torch_version`, `device`, `batch_size`, `shard_size`;
  - `wallclock_s` (summed over runs), `images_per_s` (`n_images / wallclock_s`), `runs`;
  - `npz_sha256`, `created_at` and the builder (module, version, git sha).

  `device` is the CUDA device name, or `CPU (<processor>)` as the compute log's environment row names it.
- **FR-007 Shards.** Shards are consecutive runs of `--shard-size` rows in manifest order (the default is set in the plan). Each finished shard is written before the next one starts. The state file records the key, the manifest sha256, the shard size, the finished shards and each run's wall-clock time. A run that finds state for another key, manifest or shard size discards it and starts again. `--max-shards K` computes at most K new shards, then stops with exit 1.
- **FR-008 Compute log.** Rows are written as US-3 describes, through `ms.compute_log.log_row(step="extract", …, extra={"cache": {…}})`. The `cache` object is an extension of the row, like the `env` object on the environment row (DECISIONS 8). `vram_gb` is the GPU's total memory, or null on CPU.
- **FR-009 Integrity.** Every image is hashed as it is read, and a failure stops the run as US-4.1 describes. The rows come from the manifest, so every file is expected to decode; a decode error stops the run with `decode_error`.
- **FR-010 Loader.** `load_cache(npz, manifest_sha256=None)` returns `.meta`, `.image_id`, `.sha256`, `.cls` and `.meanpatch`. The checks of US-4.2 raise `CacheMismatchError`.
- **FR-011 API.** `ms.cache` exports `preprocess`, `cache_key`, `load_cache` and `CacheMismatchError`. The command line is `ms.cache.extract` (`main(argv) -> exit code`).
- **FR-012 Research-only.** The sidecar copies `research_only_until_c5` from the backbone config. Caches of a research-only backbone are never written outside `data/`, and no test fixture holds DINOv3 features (DECISIONS 13).

## Success criteria (measurable, technology-agnostic)

- **SC-1** 224-px caches exist and load (`load_cache`) for both backbones and every frozen manifest (W2). The 518/512 caches follow in W3.
- **SC-2** Re-running any `make cache` with an unchanged key and manifest does no work and adds no row.
- **SC-3** For every cache, the compute-log rows of its runs add up to its `n_images`, and every extraction run has a row (N7).
- **SC-4** The CPU tests on the fixture run in under one minute and need neither network nor GPU.
- **SC-5** Git holds no array derived from DINOv3.

## Assumptions

- Manifests exist and load as spec 001 says. Raw files are where the manifest's `path` says, under `--raw-root`.
- The backbone configs pin a Hub revision, dtype, attention implementation, preprocess string per resolution, batch size and token types (DECISIONS 12; `test_backbones.py`).
- Throughput and night-length numbers come from the W2 Monday timing (DECISIONS 11), not from this spec.

## Clarifications

Closed on 2026-09-19:

1. **Key fields.** `compute_dtype` joins the key of the work plan (DECISIONS 11). fp16 and fp32 features differ, and the plan's four fields would not see it.
2. **The key is in the path.** The plan's layout (`data/cache/<backbone_id>/<res>/<dataset>.npz`) would overwrite a cache when the weights change. With the key in the path, both caches stay, for example the fp16/fp32 comparison.
3. **`meanpatch` is patch tokens only.** Both backbones carry 4 register tokens (their `config.json` at the pinned revisions); averaging them in would mix non-image tokens into the image descriptor.
4. **`weights_sha256`** is the sha256 of the loaded weight file, not of the Hub revision. A corrupted or replaced download changes the key.
5. **`sha256[N]`** is stored next to `image_id[N]`, for 006's cached-hash path.
6. **Every split is extracted.** Evaluation needs test and held-out features too, so the spec 001 loader gains `purpose = extract`, which reads every split and uses no labels.
7. **Golden tolerance (closed on 2026-09-19).** 2e-6. The fp16/fp32 comparison on the fixture gave a minimum CLS cosine of 0.999998924; the gap, 1.08e-6, is rounded up to one significant digit (DECISIONS 38).
8. **Shard size and batch size (closed on 2026-09-19).** 1024 and 32 stay. At 224 a shard takes 11–15 s, and a run uses 2.5 GB of the 8 GB of VRAM. The GPU is the limit on small pictures (about 90 images/s for both backbones), and decoding on large ones (DECISIONS 53). They hold at 518/512 as well: a shard takes 48–60 s, a run uses 4.2 GB, and only SWM stays decode-bound (DECISIONS 84).

Open:

9. **Full patch tokens** for a 2,000-image subset (work plan §4): a separate file, specified when W3 needs it.

## Out of scope

Heads (003). Abstention (004). Any augmentation or test-time augmentation. Multi-crop. Serving-time image handling beyond reusing `preprocess` (006). Moving caches into DVC (piece 3).
