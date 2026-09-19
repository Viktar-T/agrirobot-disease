# store/ — piece 3: ingest and observation store

H8 §4 piece 3 · H6 E14 · owner P · W2–W5 (the W1 item was this repository's skeleton)

Validator at ingest → MCAP extractor → PostGIS tables (sites, plots, plants, frames, predictions, plot_visits; station rows as a stub) → DVC manifests for every dataset and split → MLflow tracking server. No annotation tooling in Demo 1.

**Done when** (H8): one cart-equivalent pass is queryable by plot and time, a manifest hash identifies every image in every split, and every model run has an MLflow run ID.

## Hand-over from piece 4 (W2, 2026-09-19): manifests and feature caches under DVC

Piece 4 hands its two data artefacts to piece 3. Both are made by code and never edited by hand. The data stays git-ignored, and git holds the DVC pointer files.

| Folder | Pointer in git | Holds | Made by | Contract |
|---|---|---|---|---|
| `data/manifests/` | `data/manifests.dvc` (11 files, 174 MB) | one manifest per (id, version) with its sidecar, and `FROZEN.jsonl` | `make manifests DS=<id>`; crops: `python -m ms.data.manifests crops` | spec 001 |
| `data/cache/` | `data/cache.dvc` (40 files, 2.79 GB) | frozen-backbone features per backbone, resolution, cache key and manifest | `make cache BB=<backbone> RES=<res> DS=<id>` | spec 002 |

### The manifest format (spec 001, `model_service/specs/001-manifests/spec.md`)

- **`<manifest>_v<N>.jsonl`** is JSON Lines in UTF-8 with LF: one row per distinct image (by sha256), sorted by `image_id`. The row fields are those of FR-003:
  - identity: `manifest_version`, `image_id` (`<manifest>_<sha256[:16]>`), `dataset`, `source_record`, `source_version`;
  - the file: `path` (relative to `data/raw/`; for crops, relative to the root their sidecar names, `data/derived/`), `dup_paths`, `sha256` (of the stored bytes), `phash`, `width`, `height`, `format`;
  - the labels: `class_raw` (verbatim), `class_km2` (`healthy`, `rust`, `anthracnose`, `unknown_als`, `unknown_wm` or `none`), `boxes`;
  - the split: `group_keys` (`district`, `subcounty`, `date`, `region`, `session`, `variety`, `plant_age`, `phash_group`; published values only, no GPS), `split_group`, `split` (`train`, `val`, `test` or `holdout_unknown`), `split_rule`;
  - `licence`, `attribution` and `notes`. Crops rows add `parent_image_id`, `box_index` and `crop`.
- **`<manifest>_v<N>.meta.json`** is the sidecar (FR-005). It holds the recipe and the class map with their hashes, the counts per class and split, the blocks, the excluded files with their reasons, `manifest_sha256` and the builder.
- **`FROZEN.jsonl`** has one line `{manifest, sha256, frozen_at, git_sha}` per frozen manifest. A frozen manifest never changes; a change becomes `_v<N+1>`. The loader refuses a frozen file whose bytes differ. Only evaluation, serving and feature extraction may read `test` and `holdout_unknown` rows (FR-011).
- A manifest is identified by the sha256 of its file, and every N-table row carries the hashes of its manifests. An image's `sha256` is also the frame `sha256` of interface v0 (`model_service/DECISIONS.md` 32).
- `python -m ms.data.manifests validate <file.jsonl>` checks a file against the contract, including every row's file hash.
- Frozen on 2026-09-19 (`model_service/DECISIONS.md` 51): `ibean_v1`, `makerere_v1`, `makerere_crops_v1`, `swm_v1` and `tanzania_v1`.

### The cache format (spec 002, `model_service/specs/002-cache/spec.md`)

`data/cache/<backbone_id>/<res>/<cache_key>/<manifest>_v<N>.npz` holds `cls` and `meanpatch` (float16 `[N, 1024]`) and `image_id` and `sha256` (`[N]`), in the manifest's row order. Beside it, `.meta.json` holds the key fields, the manifest's sha256, the library versions and the timing. The key (backbone, weights sha256, resolution, preprocessing, compute dtype) is part of the path, so a new key never overwrites a cache. The 224 caches for both backbones and all five manifests were made on 2026-09-19 (`model_service/DECISIONS.md` 52), and the high-res ones (DINOv2 at 518, DINOv3 at 512) the same evening (83).

### For piece 3 (P)

- DVC is initialised at the repository root, with analytics off in `.dvc/config`, and `dvc add data/manifests data/cache` has run. The DVC cache is the local `.dvc/cache/` (git-ignored). A new manifest version or a new cache needs another `dvc add` of its folder, and the changed `.dvc` file goes into the commit that made it. `dvc status` shows drift.
- The root `.gitignore` ignores `/data/?*` and lets `data/*.dvc` through. Keep `?*`: with `/data/*`, DVC's git layer counts `data/` itself as ignored and never finds the pointer files.
- **There is no remote yet**, and choosing one is piece 3's decision. `data/cache/dinov3_l16/` holds DINOv3 features, which are research-only until H6 C5 (`model_service/DECISIONS.md` 13). A remote that receives them must be on the project's machines; otherwise that folder has to stay out of what is pushed.
- Two folders are not under DVC. `data/raw/` (over 90 GB) is kept as downloaded and checked by each `DOWNLOAD.json` and `SHA256SUMS`. `data/derived/` holds the 2 GB of Makerere crops, rebuilt byte for byte from `makerere_v1`. Whether DVC should hold them is piece 3's decision.
- MLflow runs go to a local store, `mlruns/mlflow.db` (git-ignored), until piece 3 runs the server (`model_service/DECISIONS.md` 55). `MLFLOW_TRACKING_URI` points the code at a server.
