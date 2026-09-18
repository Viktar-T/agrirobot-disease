# Feature specification: 001 — Dataset manifests

**Branch**: `001-manifests` · **Created**: 2026-09-17 · **Status**: Draft (for `/clarify`, then `/plan`)
**Input**: H8 §6.3 (data preparation), `docs/piece4-work-plan.md` S4.1, the E1 dataset audit (`50_E1_audit-of-existing-datasets.md`). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Everything in piece 4 — feature caches (002), heads (003), abstention (004), the results table (005) and the service's cached-hash path (006) — reads images through **one manifest per dataset version** instead of through folder names. The manifest fixes three things that cannot be repaired later: the identity of every image (a content hash), the class map from each source's labels to the KM2 classes, and the **split with its rule**, frozen before any head is trained (H8 §2 D-9). Without it, the first accuracy number leaks through duplicates and the "two numbers" of D-1 cannot be printed with their split rule.

## User scenarios and testing

### US-1 — Build a manifest from a raw dataset folder (priority P1)

As the ML role, I run one command over `data/raw/<dataset>/` and get a manifest whose rows identify every image by content, carry its provenance and licence, its raw and KM2 class, its grouping keys and its split.

Acceptance scenarios:

1. **Given** the fixture `tests/fixtures/ibean_30/` (30 iBean images, 10 per source class), **when** `ms manifests build --dataset ibean --raw-root tests/fixtures --out <tmp>`, **then** the manifest has 30 rows, every `sha256` matches the file bytes, `class_km2` is `healthy` ×10, `rust` ×10, `unknown_als` ×10, every row has `split = test` and `split_rule = test_only`, and the sidecar `meta.json` reports the same counts.
2. **Given** the same raw folder built twice, **when** the two outputs are compared, **then** they are byte-identical (stable ordering, ids derived from content, fixed seed recorded in the sidecar).
3. **Given** a raw folder containing one non-image file and one corrupt JPEG, **when** built, **then** neither gets a row; both appear in `meta.json.excluded` with a reason (`not_an_image`, `decode_error`), and the build exits 0.

### US-2 — Duplicates never straddle a split (P1)

As the ML role, I need exact and near-duplicate images to be detected and kept together, so that within-dataset numbers are not inflated and the Tanzanian records' overlap is measured rather than guessed (H6 B2).

Acceptance scenarios:

1. **Given** a raw folder where one file exists under two names, **when** built, **then** both rows share `sha256`, the second carries `exact_dup_of = <image_id of the first>`, and both share one `phash_group`.
2. **Given** an image and a copy re-encoded as JPEG at quality 70, **when** built, **then** the two rows share one `phash_group` (Hamming distance on the 64-bit pHash ≤ the threshold recorded in `meta.json`).
3. **Given** manifests for `tz155k` and `tz59k`, **when** `ms manifests overlap tz155k_v1.jsonl tz59k_v1.jsonl`, **then** the number of shared `sha256` values is printed and written to both sidecars as `overlap.<other dataset>`; rows shared across the two carry `exact_dup_of` pointing at the `tz155k` row.
4. **Given** any manifest in which one `phash_group` has rows in more than one of {train, val, test}, **when** `ms manifests validate`, **then** the exit code is 2 and the first reason printed is `group_straddles_split` with the group id and the row numbers.

### US-3 — Blocked splits with a printed rule (P1)

As the ML role, I need the split to follow the acquisition structure the source provides (district, sub-county, date, region, session), and every downstream table to be able to print the rule that produced it.

Acceptance scenarios:

1. **Given** Makerere rows with `group.district`, `group.subcounty` and `group.capture_date`, **when** built with `--rule blocked:district`, **then** no district value appears in more than one split, `split_rule = blocked:district` on every row, and `meta.json.splits` lists the districts per split with their row counts.
2. **Given** a dataset with no usable grouping metadata, **when** built with `--rule unblocked`, **then** the split is random over `phash_group` (groups atomic), `split_rule = unblocked:random_by_phash_group`, and the sidecar carries `warning: unblocked`.
3. **Given** `--rule blocked:district` and a dataset whose rows lack `group.district`, **when** built, **then** the build fails with `rule_not_applicable` and no manifest is written.
4. **Given** a manifest, **when** any consumer (cache, heads, eval, service) prints a number derived from it, **then** the number carries `split_rule` and the manifest hash — enforced by 005, but the fields must exist here.

### US-4 — The class map and the held-out unknowns (P1)

As the ML role, I need each source's label mapped to the KM2 classes by a fixed, versioned table, with angular leaf spot and white mould kept as **held-out unknowns** that are never trained on (H8 §6.3).

Acceptance scenarios:

1. **Given** the class map `configs/class_map_v1.yaml`, **when** a row's `class_raw` is not in the map, **then** the build fails with `unmapped_class` naming the value; nothing is silently dropped.
2. **Given** any row with `class_km2 ∈ {unknown_als, unknown_wm}`, **when** validated, **then** its `split` is `holdout_unknown`; a row of these classes in `train` or `val` fails validation with `unknown_in_train`.
3. **Given** ACRE rows (no disease labels), **when** built, **then** `class_km2 = none`, `split = test`, `split_rule = test_only`; a `none` row in `train` or `val` fails validation with `unlabelled_in_train`.

### US-5 — Freeze (P1)

As the ML role, I declare a manifest version frozen before the first head is trained; from then on it is immutable and its test split is unreadable to training and selection code.

Acceptance scenarios:

1. **Given** `ms manifests freeze data/manifests/makerere_v1.jsonl`, **when** it runs, **then** the file's sha256 is appended to `data/manifests/FROZEN.jsonl` (`{manifest, sha256, frozen_at, git_sha}`), and the same line is proposed for `model_service/DECISIONS.md`.
2. **Given** a frozen manifest whose bytes later change, **when** validated, **then** `frozen_manifest_modified` is reported and the exit code is 3.
3. **Given** a frozen manifest, **when** a loader is asked for `split = test` with `purpose = train` or `purpose = select`, **then** it raises `TestSplitAccessError`; `purpose = evaluate` is the only way to read the test split.
4. **Given** a change is needed after the freeze (a corrected label, a new duplicate rule), **when** it is made, **then** it produces `<dataset>_v2.jsonl`; `v1` stays as it was and stays frozen.

### US-6 — Crops manifest for Makerere (P2)

As the ML role, I need a second manifest derived from the Makerere bounding boxes, so that the organ-level run (H8 §6.7 N2 "crop-level") is possible without changing the readers.

Acceptance scenarios:

1. **Given** `makerere_v1.jsonl` with `boxes`, **when** `ms manifests crops makerere_v1.jsonl --margin 0.10`, **then** `makerere_crops_v1.jsonl` has one row per box with `parent_image_id`, `box_index`, its own `sha256` (of the written crop file), the parent's `group`, `split` and `split_rule` inherited unchanged, and `class_km2` from the box label.
2. **Given** the crops manifest, **when** validated, **then** every `parent_image_id` exists in the parent manifest and no parent's crops straddle splits.

### Edge cases

- Two source records that are supersets of each other (Tanzania #1 / #2 / #3, relation unstated `[C98]`): built as separate manifests; overlap measured (US-2.3); the pooled training set for heads is defined in 003 as "tz155k plus tz59k rows not marked `exact_dup_of`".
- Images with EXIF orientation: the hash is of the stored bytes (never of a re-oriented decode); width/height are the decoded, orientation-applied values.
- Makerere licence differs between the datasheet (CC BY) and a mirror (CC0): the manifest records what the Dataverse record itself says; until it is read, `licence = unknown` and the build needs `--allow-unknown-licence` (see Clarifications).
- A dataset with region metadata only in folder names (possible for Tanzania): the folder structure is parsed into `group.session` and reported; if nothing is there, the rule is `unblocked` and says so.

## Requirements

### Functional

- **FR-001** One manifest per (dataset, version): `data/manifests/<dataset>_v<N>.jsonl` (JSON Lines, UTF-8, LF, one object per line, rows sorted by `image_id`) plus a sidecar `data/manifests/<dataset>_v<N>.meta.json`.
- **FR-002** Datasets in scope and their ids: `tz155k` (Tanzania 2025, Zenodo 15685315 `[C01][C02]`), `tz59k` (Tanzania 2023, Zenodo 8286126 `[C03]`), `tz3` (Tanzania 2025-11, Zenodo 16751336 `[C98]`, optional — only if US-2.3 shows new images), `makerere` (Lacuna beans, Harvard Dataverse doi:10.7910/DVN/TCKVEW `[C99]`), `ibean` (Makerere AI Lab GitHub original, MIT `[C05]`), `swm` (white mould crops `[C18]`), `acre` (stretch, `[C16]`).
- **FR-003** Row fields (all present; nullable where marked): `manifest_version` ("1"); `image_id` = `<dataset>_<sha256[:16]>`, charset `[A-Za-z0-9_.-]`, ≤ 128 chars, unique in the file; `dataset`; `source_record` (DOI or record URL), `source_version` (record version or download date); `path` (relative to `data/raw/`, POSIX separators); `sha256` (64 hex, file bytes as stored); `phash` (16 hex, 64-bit pHash of the decoded image), `phash_group` (string); `exact_dup_of` (image_id or null); `width`, `height` (int), `format` (`jpeg`/`png`); `class_raw` (the source's label, verbatim); `class_km2` ∈ {`healthy`, `rust`, `anthracnose`, `unknown_als`, `unknown_wm`, `none`}; `boxes` (list of `{x, y, w, h, class_raw}` in pixels, or null); `group` (object: `district`, `subcounty`, `region`, `session`, `capture_date` (ISO date), `device`, `variety`, `plant_age` — each nullable) and `group_key` (the composed key actually used for blocking, or null); `split` ∈ {`train`, `val`, `test`, `holdout_unknown`, `unassigned`}; `split_rule` (non-empty string from the fixed vocabulary below); `licence` ∈ {`CC-BY-4.0`, `CC0-1.0`, `MIT`, `unknown`}; `attribution` (the attribution line, verbatim); `notes` (string or null).
- **FR-004** `split_rule` vocabulary: `blocked:<key>[+<key>…]` (e.g. `blocked:district`, `blocked:district+capture_date`), `unblocked:random_by_phash_group`, `test_only`, `holdout_unknown`. Anything else fails validation.
- **FR-005** Sidecar `meta.json`: dataset, version, build timestamp, `git_sha` of the builder, `builder_version`, `seed`, `phash_threshold`, `rule`, counts per `class_raw`, per `class_km2` and per `split`, `splits` (group values per split when blocked), `excluded` (path + reason), `overlap` (per other dataset, once computed), `licence_source` (what the licence field was read from), and `manifest_sha256` (of the `.jsonl` file).
- **FR-006** Duplicate handling: exact duplicates by `sha256` within a dataset and across the Tanzanian family; near-duplicates by pHash connected components at Hamming ≤ `phash_threshold` (default 6, recorded); groups are atomic for splitting; cross-family duplicates (Tanzania ↔ Makerere/iBean) are reported in `overlap`, never merged.
- **FR-007** Class map: `configs/class_map_v1.yaml`, one table `dataset → class_raw → class_km2`, versioned; `unknown_*` rows are always `holdout_unknown`; `none` rows are never in `train`/`val`.
- **FR-008** Split proportions within a dataset when a within-dataset split is made: train 70 % / val 10 % / test 20 % **of groups**, seeded; iBean is `test_only`; `swm` is `holdout_unknown`; `acre` is `test_only` with `class_km2 = none`.
- **FR-009** Validation CLI `ms manifests validate <file> [--frozen-list data/manifests/FROZEN.jsonl]`: exit 0 on success; 2 on a content violation; 3 on a freeze violation; every failure names the reason from the fixed list (`sha256_mismatch`, `duplicate_image_id`, `group_straddles_split`, `unknown_in_train`, `unlabelled_in_train`, `unmapped_class`, `bad_split_rule`, `missing_field`, `licence_unknown`, `frozen_manifest_modified`) and the row number.
- **FR-010** Licence gate: a row with `licence = unknown` fails validation unless the manifest was built with `--allow-unknown-licence`, which is recorded in the sidecar; frozen manifests may not contain `unknown` licences.
- **FR-011** Loader API (used by 002–006): `load_manifest(path, split, purpose)` with `purpose ∈ {train, select, evaluate, serve}`; `test` of a frozen manifest is readable only with `purpose = evaluate`; `holdout_unknown` only with `evaluate`; the loader returns the manifest sha256 alongside the rows so that every consumer can record it.
- **FR-012** Crops manifest (US-6): derived, never hand-edited; margin recorded; parent linkage; inherited group/split/rule.
- **FR-013** No fields beyond what the source publishes; GPS and dates from Makerere are kept as published under the source's licence and are not re-published by this project except through the source record.

### Key entities

- **Manifest** — a frozen or draft list of rows for one dataset version, identified by its file sha256.
- **Row** — one image: identity (sha256, image_id), provenance (dataset, record, licence, attribution), labels (class_raw, class_km2, boxes), structure (group, phash_group), assignment (split, split_rule).
- **Group** — the atomic unit of splitting: a pHash component, or a blocking-key value (district, …) when a blocked rule is used.
- **Class map** — versioned table from source labels to KM2 classes plus the two held-out unknowns and `none`.
- **Freeze record** — `FROZEN.jsonl` line: manifest, sha256, timestamp, git sha.

## Success criteria (measurable, technology-agnostic)

- **SC-1** Manifests exist and validate (exit 0) for `ibean`, `makerere`, `tz155k` and `tz59k` (and `swm`) on the project machine; `tz3` and `acre` are either built or explicitly excluded in `meta.json`.
- **SC-2** Building any manifest twice produces byte-identical files.
- **SC-3** 100 % of rows carry `sha256`, `class_km2`, `split`, `split_rule`, `licence`, `attribution`; 0 rows with `licence = unknown` in any frozen manifest.
- **SC-4** The Tanzanian overlap (`tz155k` ↔ `tz59k`, and `tz3` if built) is a number in the sidecars, not an assumption.
- **SC-5** Every downstream artefact of piece 4 (cache `meta.json`, head `run.json`, `n_table.jsonl` rows, `model_version`) references a manifest sha256 that appears in `FROZEN.jsonl`.
- **SC-6** The frozen manifests for the W3 head runs are listed in `DECISIONS.md` with their hashes before the first head run's timestamp.
- **SC-7** The fixture build (US-1.1) runs in CI on CPU in under one minute.

## Assumptions

- Raw datasets are downloaded untouched into `data/raw/<dataset>/` with a `DOWNLOAD.json` (URL, date, archive sha256, licence text) — `make download` from the work plan.
- Dataset facts (counts, classes, licences, metadata fields) come from the E1 audit and are re-checked against the records on download; the audit's counts are expectations for a sanity check, not requirements.
- pHash is computed on the decoded image resized to 32×32 greyscale (the standard pHash); the threshold is a recorded parameter, not a constant in code.
- Makerere per-image metadata (variety, plant age, district, sub-county, GPS, date `[C99]`) is complete enough to block by district; if not, US-3.3 fires and the rule falls back to `unblocked` with a warning (H6 E4 lead).

## Clarifications (to settle in `/clarify` before `/plan`)

1. **Makerere licence** — CC BY (datasheet) or CC0 (mirror): read on the Dataverse record; until then `--allow-unknown-licence`.
2. **Tanzania grouping** — do the Zenodo archives carry region/session/date in folder names or an index file? Decides `blocked:*` vs `unblocked` for `tz155k`/`tz59k`.
3. **`tz3`** — build it only if US-2.3 shows images not in `tz155k`/`tz59k`; otherwise exclude with reason.
4. **pHash threshold** — default 6 of 64 bits; confirm on the fixture duplicates (US-2.2) before freezing anything.
5. **Blocking key for Makerere** — `district` alone, or `district+capture_date`? (`capture_date` spans April–May 2021 only.)
6. **Crop margin** — 10 % of the box side (H8 §6.4); confirm against the Makerere box statistics once downloaded.
7. **ACRE** — in scope as a stretch (test-only, `none`); decide at W2 end whether to build it.

## Out of scope

Annotation and any labels beyond the sources' own; image preprocessing and resizing (002); DVC mechanics (piece 3 — the manifest is what DVC tracks, this spec does not say how); feature extraction; any Polish or Saxa data; the KM2 test set (a future manifest under the same contract, when Polish imagery exists).
