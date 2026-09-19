# Feature specification: 001 — Dataset manifests

**Branch**: `001-manifests` · **Created**: 2026-09-17 · **Clarified**: 2026-09-19, against the data on disk · **Status**: Implemented on 2026-09-19 (`ms.data.manifests`): the 36 acceptance tests in `model_service/tests/test_manifests.py` green; US-6 planned on 2026-09-19 against the Makerere boxes, its tests in `model_service/tests/test_crops.py` green
**Input**: H8 §6.3 (data preparation), `docs/piece4-work-plan.md` S4.1, the E1 dataset audit (`50_E1_audit-of-existing-datasets.md`), the W1 data note (`data/README.md`, findings 1–10). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Everything in piece 4 — feature caches (002), heads (003), abstention (004), the results table (005) and the service's cached-hash path (006) — reads images through **one manifest per dataset version** instead of through folder names. The manifest fixes three things that cannot be repaired later: the identity of every image (a content hash), the class map from each source's labels to the KM2 classes, and the **split with its rule**, frozen before any head is trained (H8 §2 D-9). Without it, the first accuracy number leaks through duplicates, and the "two numbers" of D-1 cannot be printed with their split rule.

## Commands

`python -m ms.data.manifests <verb>`; `make manifests DS=<manifest>` runs `build`. Defaults come from the recipe `configs/manifests/<manifest>.yaml` (FR-001).

- `build --dataset <manifest> [--raw-root data/raw] [--out data/manifests] [--version N] [--seed N] [--rule R] [--phash-threshold N] [--allow-unknown-licence]` exits 0 when the manifest is written or unchanged, 2 on a build error (nothing written), and 3 when the build would change a frozen manifest.
- `validate <file.jsonl> [--raw-root data/raw] [--frozen-list <dir>/FROZEN.jsonl]` exits 0, 2 on a content violation, and 3 on a freeze violation.
- `freeze <file.jsonl>` exits 0 or 2. `overlap <a.jsonl> <b.jsonl>` prints one JSON object with the `exact` and `near` counts and exits 0. `crops …` is P2 (US-6).

## User scenarios and testing

### US-1: Build a manifest from the raw folders (priority P1)

As the ML role, I run one command over `data/raw/<member>/` and get a manifest whose rows identify every distinct image by content and carry its provenance and licence, its raw and KM2 class, its grouping keys and its split.

1. **Given** the fixture `tests/fixtures/ibean_30/`, laid out like `data/raw/` (`ibean/DOWNLOAD.json` and `ibean/extracted/<split>/<class>/`, 10 images per source class, next to `LICENSE-MIT`), **when** `build --dataset ibean --raw-root tests/fixtures/ibean_30 --out <tmp>` runs, **then** the manifest has 30 rows and every `sha256` matches the bytes at `path`. `class_km2` is `healthy` ×10, `rust` ×10 and `unknown_als` ×10. The `unknown_als` rows are `holdout_unknown`; the others fall in train, val and test with `split_rule = unblocked:random_by_phash_group`. The sidecar reports the same counts and `role = test_only`.
2. **Given** the same raw folders built twice, **then** the two `.jsonl` files are byte-identical and the second build rewrites nothing. This follows from stable ordering, content-derived ids and the seed in the recipe.
3. **Given** a member holding a file that is not an image (iBean holds a renamed `.DS_Store`, `train/healthy/healthy_train.120tore`, finding 9) and a truncated JPEG, **then** neither gets a row. Both are in `meta.json.excluded`, with `not_an_image` and `decode_error`, and the build exits 0.
4. **Given** any build, **then** every file under the folders the reader scans ends up in exactly one place: a row's `path`, a row's `dup_paths`, an annotation file the reader consumed, or `meta.json.excluded` with a reason. Nothing is dropped silently.

### US-2: Duplicates never straddle a split (P1)

1. **Given** one file under two names with the same label, **then** it gets one row: `path` is the first copy (member order, then path order) and the other name goes into `dup_paths`. For scale, tz155k holds 43,622 extra copies (finding 2).
2. **Given** one file under two labels, **then** it gets no row, and every copy is excluded with `label_conflict`. tz155k holds 41 such images (finding 3).
3. **Given** an image and a copy re-encoded as JPEG quality 70, **then** the two rows share one `phash_group` and one split. Their Hamming distance is at most the `phash_threshold` in `meta.json`.
4. **Given** `tz155k` and `tz59k`, where tz59k is inside tz155k (finding 1), **when** `build --dataset tanzania` runs, **then** there is one row per distinct image across both records. `path` lies in the first member of the recipe (tz155k) and the copies in the other record are in `dup_paths`. `meta.json.members` gives, per record, the files scanned, the distinct images among them, and the distinct images shared with each other record.
5. **Given** a manifest in which one `phash_group` or `split_group` has rows in more than one of {train, val, test}, **when** it is validated, **then** it exits 2 and the first reason printed is `group_straddles_split`, with the group id and the row numbers.
6. **Given** manifests of two sources, **when** `overlap a.jsonl b.jsonl` runs, **then** the shared images (same `sha256`, `exact`) and the near pairs (different bytes, pHash within the threshold, `near`) are printed. Duplicates across manifests are reported, never merged. Finding 4 found no exact overlap between Tanzania, Makerere and iBean.

### US-3: Blocked splits with a printed rule (P1)

As the ML role, I need the split to follow the acquisition structure the source publishes, and every downstream table to be able to print the rule that produced it.

1. **Given** Makerere, **when** it is built (`blocked:district`), **then** no district has rows in more than one of {train, val, test}, and every trained-class row carries `split_rule = blocked:district`. `meta.json.splits` lists, per split, the blocks and their row counts.
2. **Given** rows that lack the blocking key, **then** each joins the block of the key values that rows of any class carry on the same capture `date`. Held-out rows count here: they record where the team was that day, and they are still never trained on. On 26 Apr only angular leaf spot was annotated. A date with several districts merges them into one block: on 25–26 Apr, Bugiri and Mayuge form `district:Bugiri+Mayuge`. The key stays null in `group_keys`, because it is not invented (FR-013). Makerere healthy images have no XML, so this is how they are placed. A row that still cannot be placed is split by its `phash_group` with `split_rule = unblocked:random_by_phash_group`; `meta.json` counts rows per rule and warns `partly_unblocked`.
3. **Given** Tanzania, **when** it is built (`blocked:date`), **then** no capture date has rows in more than one split.
4. **Given** iBean, which has no grouping metadata, **then** `split_rule = unblocked:random_by_phash_group`, groups are atomic, and `meta.json.warnings` contains `unblocked`.
5. **Given** `--rule blocked:district` on a manifest where no row has a district, **then** the build exits 2 with `rule_not_applicable` and writes nothing.
6. **Given** any within-dataset split, **then** every trained class present in the manifest has rows in each of train, val and test. If no assignment of whole groups can achieve that, the build exits 2 with `class_missing_from_split` and writes nothing.
7. **Given** a manifest, **when** any consumer (cache, heads, eval, service) prints a number derived from it, **then** the number carries `split_rule` and the manifest hash. Spec 005 enforces this; the fields exist here.

### US-4: The class map and the held-out unknowns (P1)

1. **Given** a `class_raw` that is not in `configs/class_map_v1.yaml`, **then** the build fails with `unmapped_class`, naming the value, and nothing is written.
2. **Given** rows with `class_km2 ∈ {unknown_als, unknown_wm}`, **then** `split` and `split_rule` are both `holdout_unknown`. If one of them is found in train or val, validation fails with `unknown_in_train`.
3. **Given** a label that the map sends to `excluded` (SWM pictures with only apothecia or sclerotia), **then** it gets no row, and the image is in `meta.json.excluded` with `excluded_by_class_map`.
4. **Given** rows with `class_km2 = none` (ACRE, a stretch goal), **then** `split = test` and `split_rule = test_only`. If one of them is found in train or val, validation fails with `unlabelled_in_train`.

### US-5: Freeze (P1)

As the ML role, I declare a manifest version frozen before the first head is trained. From then on it is immutable, and its test split is unreadable to training and selection code.

1. **Given** `freeze data/manifests/makerere_v1.jsonl`, **then** the file's sha256 is appended to `data/manifests/FROZEN.jsonl` as `{manifest, sha256, frozen_at, git_sha}` and the line for `DECISIONS.md` is printed. Freezing it again adds nothing.
2. **Given** a frozen manifest whose bytes later change, **then** validation reports `frozen_manifest_modified` and exits 3, and the loader refuses the file.
3. **Given** a loader call for `test` or `holdout_unknown` with `purpose = train` or `select`, **then** it raises `TestSplitAccessError`. Only `evaluate`, `serve` and `extract` (feature extraction, spec 002) can read them.
4. **Given** a frozen version, **when** a build would change it, **then** the build exits 3 and leaves the file untouched. The change becomes `<manifest>_v2.jsonl`, and v1 stays as it was, frozen.
5. **Given** a manifest with `licence = unknown`, **then** it can only be built with `--allow-unknown-licence`, which the sidecar records, and `freeze` refuses it with `licence_unknown` (exit 2).

### US-6: Crops manifests (P2; planned on 2026-09-19 against the Makerere boxes, tests in `tests/test_crops.py`)

1. **Given** `makerere_v1.jsonl` with `boxes`, **when** `crops makerere_v1.jsonl --margin 0.10` runs, **then** `makerere_crops_v1.jsonl` has one row per box. Each row carries `parent_image_id`, `box_index`, `crop` (the rectangle cut: `x`, `y`, `w`, `h` and its `frame`) and its own `sha256` (of the written crop). It inherits the parent's `group_keys`, `split_group`, `split` and `split_rule` unchanged, and takes `class_km2` from the box label.
2. **Given** a crops manifest, **when** it is validated, **then** every `parent_image_id` exists in the parent manifest, and no parent's crops straddle splits.
3. `swm_crops` holds the source's 300-px SWM crops, each linked to its R-SWM original by file name. (Not built yet.)
4. **Given** a parent that its EXIF orientation turns, **then** each box is cut in the frame that holds it: the stored pixels, or the upright image. Makerere's annotations use either, image by image and even box by box (the W2 scan of its 624 turned images whose width and height swap). A crop cut from the stored pixels is then turned upright, as whole frames are. A box that both frames hold, or neither, gets no row: it is in `meta.json.excluded` as `box_frame_unknown`. A parent that is not turned has one frame.
5. **Given** a box that reaches past the image, **then** its crop is clipped to the image. A box with nothing inside the image gets no row (`box_outside_image`).

### Edge cases

- **Capture date** is the local calendar day of the capture, taken from one source per dataset.
  - **Makerere**: the XML `datetime`. For images without XML, the date comes from the file name, read as Unix milliseconds in UTC+03:00 (the offset of the XML datetimes). The file name falls on the XML's day for 10,116 of 10,118 annotated images. EXIF is not used, because it was rewritten after the trip: IFD0 `DateTime` reads 2021-08-07, and `DateTimeOriginal` is missing in 361 healthy images and dated after the trip (2021-06-10, 2021-08-11) in 306.
  - **Tanzania**: EXIF `DateTimeOriginal`, never IFD0 `DateTime`. The file names are not capture times; they disagree with `DateTimeOriginal` for over half of tz155k's healthy images.
- **EXIF orientation**: the hash is taken over the stored bytes. `width` and `height` are orientation-applied, and boxes stay in the source's coordinates.
- **Makerere class ↔ trip confound** (finding 6): every healthy image comes from the April trip, so only the four April blocks (Mubende, Hoima, Kiboga, Bugiri + Mayuge) can carry healthy into val and test. The manifest shows this rather than hiding it; N1 and N4 have to name it.
- **Adjacent days** (Tanzania): blocking by day removes same-session duplicates, but not the likeness of a farm revisited the next day. The N1 reading has to name this.
- **Near-duplicate chains**: pHash components can link blocks and merge them. Merged blocks show in `split_group` and `meta.json.splits`.
- **tz3**: if its test shows new images, it joins the `tanzania` members and the result is `tanzania_v2`.

## Requirements

### Functional

- **FR-001** One manifest per (manifest id, version): `data/manifests/<manifest>_v<N>.jsonl` plus `<manifest>_v<N>.meta.json`. The `.jsonl` is JSON Lines in UTF-8 with LF, one object per line, keys in FR-003 order and rows sorted by `image_id`. Each manifest id has a recipe `configs/manifests/<manifest>.yaml` holding:
  - `members`: the raw datasets under `data/raw/`, in precedence order;
  - `role`: `train_eval`, `test_only` or `holdout_unknown`, i.e. what the protocol may use the manifest for (003 and 005 enforce it);
  - `rule`, `group_keys`, `seed`, `phash_threshold` and `version`.
- **FR-002** The raw datasets are `configs/datasets/` (downloads, W1). The manifests built from them are listed in the table below. `tz3` (Zenodo 16751336) joins `tanzania` only if its test shows new images. `acre` is a stretch goal.
- **FR-003** Row fields, all present and nullable only where marked:
  - `manifest_version`: `"1"`.
  - `image_id`: `<manifest>_<sha256[:16]>`, unique, one row per distinct image.
  - `dataset`: the manifest id.
  - `source_record`: `https://doi.org/<doi>`, else the record URL. `source_version`: the record version, else its publication date. Both come from the member's `DOWNLOAD.json`.
  - `path`: relative to `data/raw/`, POSIX, starting with the member id. `dup_paths`: the other files with the same bytes, sorted, or `[]`.
  - `sha256`: 64 hex digits of the stored bytes. `phash`: 16 hex digits, 64-bit.
  - `width`, `height` (int) and `format` (`jpeg`/`png`).
  - `class_raw`: the source's label, verbatim. `class_km2` ∈ {`healthy`, `rust`, `anthracnose`, `unknown_als`, `unknown_wm`, `none`}.
  - `boxes`: a list of `{x, y, w, h, class_raw}` in pixels, or null.
  - `group_keys`: {`district`, `subcounty`, `date` (ISO), `region`, `session`, `variety`, `plant_age` (int), `phash_group`}. Each is null unless the source publishes it, except `phash_group`, which is always set.
  - `split_group`: the unit the row was split with: its block for a trained-class row placed by `blocked:*`, otherwise its `phash_group` (held-out rows, unblocked rows, rows that fell back under US-3.2).
  - `split` ∈ {`train`, `val`, `test`, `holdout_unknown`} and `split_rule` (FR-004).
  - `licence` ∈ {`CC-BY-4.0`, `CC0-1.0`, `MIT`, `unknown`} and `attribution` (verbatim), both from `DOWNLOAD.json`.
  - `notes`: a string or null.
- **FR-004** The `split_rule` vocabulary is `blocked:<key>` (a `group_keys` name), `unblocked:random_by_phash_group`, `test_only` and `holdout_unknown`. Anything else fails validation with `bad_split_rule`.
- **FR-005** The sidecar holds:
  - `manifest`, `version`, `role`, `rule`, `seed`, `phash_threshold`;
  - `class_map` (path, version, sha256) and `recipe_sha256`;
  - `members` (per record: `files`, the files scanned apart from annotation files; `distinct`, their distinct sha256s; `shared_with`, the distinct sha256s shared with each other record);
  - `counts` (`rows`, then per `class_raw`, `class_km2`, `split`, `split_rule`, and split × `class_km2`) and the achieved fractions;
  - `splits` (blocked rules only: split → block → trained rows);
  - `excluded` (`path`, `reason`) and `warnings` (`unblocked`, `partly_unblocked`, `mixed_class_phash_group`);
  - `licence_source`, `allow_unknown_licence`, `manifest_sha256`, `built_at` and the builder (module, version, git sha).

  The sidecar is rewritten only when something other than `built_at` and the builder has changed.
- **FR-006** Duplicates:
  - **Exact** duplicates share a `sha256`, across all members of a manifest. They get one row with `dup_paths`; if the copies disagree on the label, the image is excluded with `label_conflict`.
  - **Near** duplicates: pHash is pinned so that every implementation writes the same bits. Take the orientation-applied image as greyscale (PIL `L`), resize it to 32×32 with Lanczos and apply an orthonormal 2-D DCT-II. Keep the top-left 8×8 coefficients, DC included. Each bit is "coefficient > the median of the 64", read row by row with the first bit most significant, and the result is written as 16 hex digits. Connected components at Hamming ≤ `phash_threshold` (2, recorded; Clarification 4) form the `phash_group`, whose id is the smallest `image_id` in the component.
  - **Blocks** are the connected components of the trained-class rows under three links: same key value, same `phash_group`, and, for a row without the key, the key values that rows of any class (held-out ones included) carry on the same `date` (US-3.2). A block's id is `<key>:<values sorted, joined by +>`. Blocks and phash groups are atomic for splitting.
  - Duplicates **across manifests** are reported by `overlap`, never merged.
- **FR-007** The class map is `configs/class_map_v1.yaml`: per manifest, `class_raw` → KM2 class or `excluded`. It is additive only: changing an entry means `class_map_v2`. `unknown_*` rows are always `holdout_unknown` and `none` rows are always `test`.
- **FR-008** Splits for trained-class rows target train 0.70, val 0.10 and test 0.20 of the rows of each trained class. Only whole groups move, and the seed is the recipe's `seed`. Among the assignments that put every trained class present into every split, the builder takes one close to the targets (by the sum over splits and classes of the squared gap between achieved and target shares): with up to 12 groups (a blocked split) it scores every assignment and takes the closest, and beyond that it improves a seeded greedy one by moves and swaps. If there is none, it fails with `class_missing_from_split`. The achieved fractions go to the sidecar. In `test_only` manifests every row is `test`; in `holdout_unknown` manifests every row is `holdout_unknown`.
- **FR-009** Reasons are printed as `<reason>: …`, with the 1-based row numbers and the group id where there is one.
  - **Validation**: `missing_file`, `sha256_mismatch`, `duplicate_image_id`, `rows_not_sorted`, `missing_field`, `bad_value`, `bad_split_rule`, `group_straddles_split`, `class_missing_from_split`, `unknown_in_train`, `unlabelled_in_train`, `licence_unknown`, `frozen_manifest_modified`; for crops manifests also `parent_missing`.
  - **Build**, which also exits non-zero: `unmapped_class`, `rule_not_applicable`, `class_missing_from_split`, `licence_unknown`, `frozen_manifest_modified`, and `sha256_mismatch` for a file that differs from its member's `SHA256SUMS`.
  - **Exclusion**, recorded in `meta.json.excluded`: `not_an_image`, `decode_error`, `label_conflict`, `excluded_by_class_map`, `drawn_boxes`, `no_label`; for crops, `box_frame_unknown` and `box_outside_image`, each named `<parent path>#box<index>`.
- **FR-010** Licence gate: `licence = unknown` fails the build and validation unless the manifest was built with `--allow-unknown-licence`. `freeze` refuses it in every case.
- **FR-011** The loader, used by 002–006, is `load_manifest(path, split, purpose)` with `purpose ∈ {train, select, evaluate, serve, extract}`. `extract` is feature extraction (002): it reads every split and uses no labels. The loader returns `.rows`, `.sha256` (of the file) and `.frozen` (listed in `FROZEN.jsonl` beside it).
  - `test` and `holdout_unknown` are readable only for `evaluate`, `serve` and `extract`; any other purpose raises `TestSplitAccessError`.
  - A frozen manifest whose bytes changed raises `FrozenManifestModified`.
- **FR-012** Crops manifests (US-6) are derived and never hand-edited: `crops <parent.jsonl> [--margin 0.10] [--out <dir>] [--crop-root data/derived]` writes `<manifest>_crops_v<N>.jsonl` beside its parent, where N is the parent's version.
  - The margin is a share of the box side, on each side, taken outward to whole pixels in exact decimal arithmetic.
  - The crops are JPEG files (quality 95, no chroma subsampling) in `data/derived/<manifest>_crops/`, named `<parent_image_id>_b<box_index>.jpg`. `data/raw/` stays as downloaded. `path` is relative to that root, which the sidecar names as `root`, so `validate` and the cache (002) find the files there.
  - A row has FR-003's fields, then `parent_image_id`, `box_index` and `crop`. It inherits `group_keys`, `split_group`, `split` and `split_rule`, except that a crop whose label maps to a held-out unknown is `holdout_unknown` whatever its parent.
  - The sidecar records `kind: crops`, the margin, the encoding, the frame rule, the parent manifest and its sha256, and per-reason counts of the boxes without a row. A rebuild with the same inputs rewrites no file, and a frozen crops manifest is never changed (exit 3).
- **FR-013** No fields beyond what the source publishes. `group_keys` hold published values only. Derived values, such as the district of a healthy image, appear only in `split_group`. The manifest holds no GPS.
- **FR-014** Building again with the same raw files, recipe, class map and builder version produces a byte-identical `.jsonl` and leaves the sidecar untouched. A frozen version is never overwritten (exit 3).

### Datasets: labels, grouping keys, rules

| Manifest ← members | Role | Rows from (under `data/raw/<member>/`) | `class_raw` → KM2 (`class_map_v1`) | `group_keys` the source publishes | `split_rule` |
|---|---|---|---|---|---|
| `ibean` ← ibean | test-only: an N2 target. Its within-dataset split serves the W2 slice, and those numbers are never quoted. It also trains, with Makerere, in N2's Makerere + iBean → Tanzania (the owner's decision of 2026-09-19; spec 003 US-5.1) | `extracted/<split>/<class>/` | folder: `healthy`; `bean_rust` → rust; `angular_leaf_spot` → unknown_als | none (the source's own split stays visible in `path`, unused) | `unblocked:random_by_phash_group` |
| `makerere` ← makerere | train and evaluate | `extracted/<archive>/<archive>/*.jpg`, with the XML beside each image | XML `class`: `ALS` → unknown_als, `Bean Rust` → rust; without XML: `healthy` (the folder) | XML: `district`, `subcounty`, `date` (`datetime`), `variety`, `plant_age` (`age`), and boxes. Healthy images: `date` only, from the file name (see Edge cases) | `blocked:district`; healthy images join through their date (US-3.2) |
| `tanzania` ← tz155k, tz59k | train and evaluate | `extracted/<archive>/<class folder>/*.jpg` | folder without its chunk digits: `healthy`, `rust`, `anthra` → anthracnose | `date` (EXIF `DateTimeOriginal`). No region or session: the folders are flat class folders split into chunks, and the tz59k record states one region (Mbeya) | `blocked:date` |
| `swm` ← swm | held-out unknown | `extracted/r-swm-dataset/r-swm-dataset/original-images/<split>/images/`, with the VOC XML in `…/pascal-voc/<split>/`. The `bbox/` renderings are excluded (`drawn_boxes`), and the SWM crops are `swm_crops` (US-6) | VOC object names, distinct, sorted, joined ` + `: anything with `White Mold` → unknown_wm; apothecia or sclerotia only → excluded | `date` and `session` (the `ds-<date>-<place>` prefix) from the file name | `holdout_unknown` |
| `acre` ← acre (stretch) | test-only | to be decided with its recipe | → none | none | `test_only` |

### Key entities

- **Manifest**: the rows of one manifest version, identified by its file sha256. It is a draft until frozen.
- **Recipe**: how a manifest is built from its members: role, rule, keys, seed and threshold.
- **Row**: one distinct image. It carries identity (`sha256`, `image_id`, `path`, `dup_paths`), provenance (record, licence, attribution), labels (`class_raw`, `class_km2`, boxes), structure (`group_keys`, `split_group`) and assignment (`split`, `split_rule`).
- **Block**: the atomic unit of a blocked split. **Phash group**: the atomic unit of near-duplicates.
- **Class map**: the versioned table from source labels to KM2 classes, the two held-out unknowns, `none` and `excluded`.
- **Freeze record**: one `FROZEN.jsonl` line.

## Success criteria (measurable, technology-agnostic)

- **SC-1** Manifests exist and validate (exit 0) for `ibean`, `makerere`, `tanzania` and `swm` on the project machine. `tz3` and `acre` are either built or recorded as not built, with the reason.
- **SC-2** Building any manifest twice produces byte-identical files.
- **SC-3** 100 % of rows carry `sha256`, `class_km2`, `split`, `split_rule`, `licence` and `attribution`, and no frozen manifest has a row with `licence = unknown`.
- **SC-4** The Tanzanian overlap (tz155k ↔ tz59k, and tz3 if built) is a number in the `tanzania` sidecar, not an assumption.
- **SC-5** Every downstream artefact of piece 4 (cache `meta.json`, head `run.json`, `n_table.jsonl` rows, `model_version`) references a manifest sha256 that appears in `FROZEN.jsonl`.
- **SC-6** The frozen manifests for the W3 head runs are listed in `DECISIONS.md` with their hashes before the timestamp of the first head run.
- **SC-7** The fixture build (US-1.1) runs in CI on CPU in under one minute.
- **SC-8** Every file under the scanned folders is accounted for (US-1.4).

## Assumptions

- The raw datasets are downloaded and extracted as in W1 (`make download`, `make extract`). Each `data/raw/<member>/` has a `DOWNLOAD.json` (record, version, licence, attribution), an `extracted/` folder and a `SHA256SUMS`. iBean's `extracted/` predates `ms.data.extract` and is laid out as `<split>/<class>/` (data note).
- Counts and fields come from the W1 data note. They are expectations for a sanity check, not requirements.
- pHash is as described in FR-006. The threshold is a recorded parameter, not a constant in code.

## Clarifications

Items 1–7 keep the draft's numbers, because other files cite them; items 8–11 are new. Closed items were settled on 2026-09-19 against the data (the W1 note, plus EXIF and XML scans of every Tanzanian and Makerere image).

1. **Makerere licence (closed)**: CC0-1.0, as the Dataverse record states (`DOWNLOAD.json`). The datasheet says CC BY (finding 10).
2. **Tanzania grouping (closed)**: there is no region, session or index in the archives (flat class folders split into chunks), and the tz59k record states one region, Mbeya. Every image has EXIF `DateTimeOriginal` except one rust picture held by both records, so the rule is `blocked:date`, with that one row unblocked (US-3.2). Days: healthy 122, rust 48, anthracnose 30, over 2022-10 to 2024-09. GPS is present in 18–69 % of images by class and is not used in v1.
3. **tz3 (tested and decided on 2026-09-19)**: `RUST_6.zip` (0.91 GB) adds images. None of its 321 rust photos is an exact copy of a tz155k or tz59k file, and none lies within 8 pHash bits of a `tanzania_v1` picture under any turn. They are full resolution (4160×3120). The 12 that carry a date were taken on 2024-12-24, after tz155k ends, and 309 carry no capture date, so `blocked:date` could place almost none of them. The owner's decision: freeze `tanzania_v1` from tz155k and tz59k. The full download (47.1 GB) is a later decision, perhaps as a manifest of its own, a later-in-time test, rather than a member of `tanzania` (DECISIONS 47).
4. **pHash threshold (closed on 2026-09-19)**: 2 of 64 bits. Every code has 32 one-bits, so distances are even, and 2 and 3 are the same rule. Copies re-saved at quality 70 or 50, or halved and re-saved, move at most 2 bits (300 real images each from iBean, Makerere and tz155k). At the draft default of 6, `tanzania` had 73 links across capture dates. Every one inspected was a false match: other leaves, other diseases. They chained 20 dates into one block and left 1 anthracnose row in val. At 2 bits one cross-date link remains, also false, and it joins two healthy dates. Same-date copies at 4–6 bits are real, but under `blocked:date` they share a block anyway (DECISIONS 48).
5. **Makerere blocking key (closed)**: `district` alone. A capture date holds one or two districts, so a date adds nothing inside a block, and sub-counties (1–11 per district) are too fine. Healthy images join through their capture date, which comes from the file name because their EXIF is unreliable (Edge cases). Bugiri and Mayuge share 25–26 Apr.
6. **Crop margin (closed on 2026-09-19)**: 10 % of the box side on each side (H8 §6.4). Makerere has 36,640 boxes on 10,101 images, whole infected leaves more than lesions: short side median 250 px, 5th percentile 70 px, minimum 14 px. A 10 % margin keeps a leaf's edge in the crop without taking in its neighbours.
7. **ACRE (open)**: a stretch goal (test-only, `none`); decide at the end of W2.
8. **Tanzania is one manifest (closed)**: `tanzania`, over tz155k and tz59k, not one manifest per record. tz59k adds one image (finding 1), and N2 and N4 treat Tanzania as one source.
9. **One row per distinct image (closed)**: copies go to `dup_paths` (finding 2), and an image held under two labels is excluded (finding 3).
10. **iBean (closed)**: test-only in the protocol, with an unblocked within-dataset split for the W2 slice. Angular leaf spot is `unknown_als`. Amended on 2026-09-19 by the owner: iBean also trains, beside Makerere, in N2's Makerere + iBean → Tanzania, where it is a training source and not a target (spec 003 US-5.1). Its numbers as a target stay N2's Tanzania → iBean, over all its healthy and rust rows (spec 005 US-7).
11. **SWM rows are the R-SWM originals (closed)**: the class map decides which labels are `unknown_wm`, and the 300-px crops become `swm_crops` (P2).

## Out of scope

Annotation, and any labels beyond the sources' own. Image preprocessing and resizing (002). DVC mechanics: piece 3 tracks the manifest with DVC, and this spec does not say how. Feature extraction. Any Polish or Saxa data. The KM2 test set: a future manifest under the same contract, once Polish imagery exists.
