# Feature specification: 003 — Heads

**Branch**: `003-heads` · **Created**: 2026-09-19 · **Status**: Implemented 2026-09-19 (`ms.heads`, `ms.heads.train`, `configs/heads/`); acceptance tests in `model_service/tests/test_heads.py` green; SC-1 met at 224 on 2026-09-19 for both backbones on `makerere_v1` and `tanzania_v1` (DECISIONS 67–70)
**Input**: H8 §6.5 (heads), `docs/piece4-work-plan.md` S4.3, the W3 "Heads" task and §4 (conventions: seeds 0–4), DECISIONS 33–35 (the slice's trainer, its recipe and the role gate), spec 001 (roles, the loader's purposes), spec 002 (caches). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Every N1–N3 number comes from a head trained on cached features, so a head run has to be something anyone can repeat and trace. The same inputs and seed must give the same weights, the run must record exactly what it read, and no test row may steer it, not even through early stopping. The rare classes need protection, because an average can hide a class the head never predicts. Every run also leaves its seconds in the compute log (N7).

## Commands

`python -m ms.heads.train`; `make heads BB=<backbone_id> RES=<res> ARGS="--train-manifest <file>"` runs it.

    --backbone <id> --res <n> --train-manifest <file.jsonl> [<file.jsonl> ...] [--split train]
    [--head linear|proto|mix ...] [--seed N ...] [--val-manifest <file.jsonl>] [--val-split val]
    [--tokens cls|meanpatch|cls+meanpatch] [--cache-key K] [--allow-test-only]
    [--cache-root data/cache] [--heads-root data/heads] [--config-dir model_service/configs/heads]
    [--compute-log model_service/results/compute_log.jsonl]

- `--head` defaults to `linear proto mix`, and `--seed` to `0 1 2 3 4` (work plan §4), so `make heads` runs the W3 protocol. Each (head, seed) pair is one run.
- `--tokens` defaults to `cls`; `cls+meanpatch` is the W4 ablation.
- Exit codes: 0 when every run is trained or already there. 2 on an input error: every input is checked before the first run trains (FR-008), so an input error writes no run folder and no compute-log row.

## User scenarios and testing

### US-1: One run per (head, seed), reproducible (priority P1)

1. **Given** a `train_eval` manifest and its cache, **when** `train --head linear --seed 0` runs, **then** `<heads-root>/<run_id>/` holds `head.pt` and `run.json` (FR-004, FR-005).
2. **Given** the same inputs and seed, **then** the weights are identical bit for bit and the `run_id` is the same. A second call prints "nothing to do", writes nothing and adds no compute-log row.
3. **Given** another seed, or a change in any other input (cache key, tokens, manifest bytes, split, head config), **then** the `run_id` changes and a new folder is written beside the old one.
4. **Given** no `--head` and no `--seed`, **then** 15 runs are written: linear, proto and mix, each for seeds 0–4.

### US-2: The three heads (P1)

1. **linear**: one affine map on the features, then a softmax over the classes (multinomial logistic regression), trained with the recipe (FR-006).
2. **proto**: K prototypes per class (`prototypes_per_class`, 4). A class's logit is the largest cosine similarity between the features and its K prototypes, divided by τ. τ is learnable, starts at 0.07 and is stored as `log_tau`. Each class's prototypes start as the k-means centroids (K clusters, seeded by the run's seed) of that class's L2-normalised train features, themselves L2-normalised. Then the head is trained with the recipe (FR-006).
3. **mix**: p = 0.5 · p_linear + 0.5 · p_proto, the softmax probabilities of the linear and proto runs with the same inputs and seed. The weights are fixed: nothing is trained. Its logits are log p, so a softmax gives p back.
   - A missing component is trained first, as its own run with its own compute-log row.
   - `run.json` names both components' run ids. `head.pt` holds both state dicts, so a mix run loads on its own.

### US-3: Only the in-domain validation split steers training (P1)

1. **Given** any run, **then** the train rows are read with `purpose = train` and the validation rows with `purpose = select` (spec 001 FR-011). `--split` or `--val-split` naming `test` or `holdout_unknown` exits 2 with `test_split_access`, and nothing is written.
2. **Given** `--val-manifest` naming a manifest that is not a training manifest (by sha256), **then** the run exits 2 with `val_not_in_domain`. The default is every training manifest's own validation split.
3. **Given** a run, **then** early stopping and the epoch kept read only the validation split's rows of the head's classes. The score is the cross-entropy averaged per class, ties broken by macro-F1, with `patience` epochs (DECISIONS 34).
4. **Given** a cache whose test and held-out rows carry other features, **then** every head trains the same weights as before.

### US-4: Class balance is on by default (P1)

1. **Given** a head config without `sampling`, **then** the run samples `class_balanced`, and `run.json` records it. Each epoch draws n rows with replacement, with a row's probability ∝ 1 / its class's count, so every class is drawn equally often in expectation. `uniform` (a seeded permutation of the rows) has to be named.
2. The loss is the focal loss (γ = 2) against label-smoothed targets (0.1).

### US-5: Roles (P1)

1. A `train_eval` manifest trains, and the run is quotable. A `test_only` manifest trains only with `--allow-test-only`, and the run is marked not quotable (DECISIONS 35). A `holdout_unknown` manifest never trains. The refusals exit 2 with `role_not_trainable`.
   - **Several training manifests** (N2's Makerere + iBean → Tanzania; the owner's decision of 2026-09-19). The first manifest's role decides, as above. A further one may be `train_eval` or `test_only`, and never `holdout_unknown`.
   - Each manifest trains on its `--split` rows and validates on its own `--val-split` rows.
2. The head's classes are the trained classes (`healthy`, `rust`, `anthracnose`, in that order) present in the train split (of every training manifest). Fewer than two classes, or a class with no validation rows, is `bad_value`. For proto, a class with fewer than K train rows is `too_few_rows`.

### US-6: Seconds per run (P1, N7)

1. **Given** a run that trains or mixes, **then** it appends one compute-log row through `ms.compute_log` with these fields:
   - `step = "train_head"`, `backbone_id`, `res`, `dataset` (the training manifest's), `n_images_or_runs = 1`;
   - `wallclock_s`: the run's own seconds, from the start of its fit (or its mixing) to writing its folder. The runs of one call read their shared inputs (manifests, caches) once, and that reading is in no run's seconds;
   - `device`, `vram_gb` (null on CPU), `notes`;
   - a `head` object: `run_id`, `head`, `tokens`, `seed`, `n_train`, `n_val`, `epochs_run`, `best_epoch`, and for mix, `components` (the two run ids).

   A run that was already there writes no row.

### Edge cases

- **Determinism.** Heads train on CPU in float32. "Identical" means on one machine with the same library versions. The seed drives `torch.manual_seed`, the sampler's generator and k-means.
- **Steps per epoch.** iBean's 604 train rows make 3 steps per epoch at batch 256 (DECISIONS 34). Makerere's make about 27, and Tanzania's about 300.
- **The mixture's components** share every input and the seed. A run trained on other inputs is never mixed in.

## Requirements

### Functional

- **FR-001 Location.** A run is `data/heads/<run_id>/` with `head.pt` and `run.json`. It is git-ignored, because heads on DINOv3 features are research-only (DECISIONS 13, 34). The folder is written under a temporary name and renamed when complete.
- **FR-002 Run id.** `<backbone_id>-<res>-<head>-<tokens, + as _>-s<seed>-<hash8>`. `hash8` is the first 8 hex digits of the sha256 of `json.dumps(inputs, sort_keys=True)`, where `inputs` holds:
  - `trainer_version`, `backbone_id`, `res`, `cache_key`, `tokens`, `head_config_sha256`, `seed`;
  - `train` and `val`, each as `[manifest sha256, split]`, with several manifests' sha256s joined by `+`;
  - for mix, the components' run ids.
- **FR-003 Configs.** One file per head in `configs/heads/` (`linear.yaml`, `proto.yaml`, `mix.yaml`).
  - The recipe keys are `head`, `optimizer`, `lr`, `weight_decay`, `batch_size`, `max_epochs`, `patience`, `loss`, `focal_gamma`, `label_smoothing`, `sampling` and `select`.
  - proto adds `prototypes_per_class` and `tau_init`. mix holds `components: [linear, proto]` and `weights: [0.5, 0.5]`.
  - The values are H8 §6.5's. A changed file is a new run, because its sha256 is an input.
- **FR-004 `run.json`** (`run_json_version = 2`). Its keys, in this order:
  - `run_json_version`, `run_id`, `head`, `tokens`, `seed`, `classes`, `input_dim`, `model_version`, `quotable`, `allow_test_only`;
  - `backbone`: `backbone_id`, `res`, `cache_key`, `weights_sha256`, `preprocess_string`, `compute_dtype`, `hf_id`, `revision`, `research_only_until_c5`;
  - `train` and `val`: `manifest`, `manifest_sha256`, `frozen`, `role`, `split`, `split_rules`, `n`, `class_counts`, `cache`. With several manifests, `manifest`, `manifest_sha256`, `role` and `cache` join theirs with `+` (the N-table's form). `frozen` is true only when all are frozen, and `n` and `class_counts` are totals;
  - `head_config`: `path` and `sha256` of the file, then the effective recipe, with defaults filled in;
  - `fit`: `select`, `epochs_run`, `best_epoch`, `stopped_early`, `val_macro_f1`, `val_loss`, `val_balanced_loss`, `history`. proto adds `tau`, the learned τ of the kept epoch. For mix, `epochs_run = 0`, `best_epoch = null`, `history = []`, and the validation scores are the mixture's;
  - `components`: for mix, `[{head, run_id}]` for linear and proto; otherwise null;
  - `inputs_sha256`, `wallclock_s`, `device`, `torch_version`, `created_at`, `builder` (module, version, git sha), `notes`.

  Every v1 key keeps its meaning, so the eval (005) and the service (006) read v2 runs unchanged.
- **FR-005 `head.pt`.**
  - It holds `{head, classes, tokens, input_dim, params, state_dict}` and loads with `torch.load(..., weights_only=True)`. `params` are the head's constructor arguments.
  - linear's state is `linear.weight` `[C, D]` and `linear.bias` `[C]`. proto's is `prototypes` `[C, K, D]` and `log_tau` `[]`.
  - mix's `params` hold `weights` and each component's params (`linear`, `proto`), and its `state_dict` holds both components' states (`linear.*`, `proto.*`). A `components` entry names their runs.
  - `ms.heads.load_run(folder)` returns a `Run` whose `logits(x)` and `predict_proba(x)` score features.
- **FR-006 Recipe.**
  - AdamW with the config's `lr` (H8 range 5e-4 to 1e-3) and weight decay 1e-4. Weight decay does not apply to `log_tau`.
  - Batch 256, at most `max_epochs` epochs, early stopping after `patience` (10) epochs without a better validation score.
  - `max_epochs` is 300, so that early stopping, not the cap, ends every run (Clarification 9). H8 §6.5 said 50 and was amended to 300 on 2026-09-20, after the owner's decision of 2026-09-19.
  - The focal loss with γ = 2 against targets smoothed by 0.1; class-balanced sampling.
  - The epoch kept is the best by `select`.
- **FR-007 Features.** Each token type (spec 002) is L2-normalised; `cls+meanpatch` joins the two normalised vectors (DECISIONS 34). The same `ms.heads.features` feeds training, the eval (005) and the service (006).
- **FR-008 Checks before training.** These are checked before any run trains:
  - The manifests load: roles, split access and frozen bytes.
  - The validation manifest is in domain, and the classes are valid (US-5.2).
  - One cache key holds both manifests' caches, and every head config parses.

  A failure prints `<reason>: …` and exits 2. The reasons are `role_not_trainable`, `test_split_access`, `val_not_in_domain`, `frozen_manifest_modified`, `missing_file`, `missing_cache`, `bad_config`, `bad_value` and `too_few_rows`.
- **FR-009 Compute log.** Each run writes one row, as US-6 describes.
- **FR-010 API.**
  - `ms.heads` exports `HEADS` (`linear`, `proto`, `mix`), `build_head(head, dim, n_classes, **params)`, `features`, `model_version`, `load_run`, `list_runs` and `Run`.
  - `ms.heads.train` exports `main(argv) -> exit code`, `focal_loss`, `epoch_order(y, sampling, generator)` (one epoch's row indices) and `init_prototypes(x, y, n_classes, k, seed)` (`[C, K, D]`, unit rows).
- **FR-011 Model version.** `msv0.1+<backbone_id>@<res>.<head>.man-<train manifest sha256[:6]>` (H8 §6.2; work plan W5). Several training manifests give their `[:6]` joined with `-`. It is the same for the five seeds, whose runs differ by `run_id`.

## Success criteria (measurable, technology-agnostic)

- **SC-1** Both backbones at 224 have the three heads × five seeds on `makerere_v1`, on `tanzania_v1` and on `makerere_v1` + `ibean_v1` (N2): every run has its folder and its compute-log row (the W3 Heads and N2 tasks).
- **SC-2** Re-running `make heads` with unchanged inputs does nothing and adds no row.
- **SC-3** Every run folder has exactly one `train_head` row in the compute log. A deleted run's row stays (DECISIONS 34), so the reverse need not hold.
- **SC-4** The CPU tests use synthetic manifests and caches, run in under one minute, and need neither network nor GPU.
- **SC-5** No run reads a test or held-out row with `purpose = train` or `select` (US-3).

## Assumptions

- The caches exist as spec 002 says, and the manifests are frozen (DECISIONS 51).
- Heads train on CPU. How long a run takes is measured by the Heads task, not stated here.

## Clarifications

Closed on 2026-09-19:

1. **mix mixes the two trained heads; it is not a third trained model.** "Fixed 0.5/0.5" means the weights are not learned, and each component is the linear or proto run with the same inputs and seed. A model trained through the mixture would make its parts differ from the linear and proto heads the table reports.
2. **A prototype class's logit is its nearest prototype**: the largest cosine over K, divided by τ.
3. **τ is learned as `log_tau` and is not decayed.** Decay on `log_tau` would pull τ towards 1 and flatten the logits.
4. **Prototypes start from per-class k-means**, K = 4, seeded by the run's seed, on the L2-normalised train features, with normalised centroids. A class with fewer than K train rows cannot fill K prototypes (`too_few_rows`).
5. **The validation split is the training manifest's own.** Early stopping on another manifest would tune on N2's target.
6. **The selection score is the per-class-averaged validation cross-entropy for every trained head** (DECISIONS 34).
7. **The defaults are the W3 protocol**: every head, seeds 0–4.
8. **`run_json_version` 2 keeps every v1 key**, and adds `components` and the effective recipe. `trainer_version` 2 is an input, so v2 runs get new run ids; the slice's run is not quotable anyway.

9. **The linear head's steps (closed on 2026-09-19, by measurement; DECISIONS 69).** On Makerere (27 steps per epoch) the linear head is still improving when the 50 epochs end: its best epoch is 50 for four seeds of five on both backbones. On Tanzania (307 steps per epoch) it stops by epoch 15. H8's recipe stays: 50 epochs, and lr 1e-3 is already the top of its range. A logit scale or more epochs would be a config change with new run ids, not a contract change, and it is the owner's call. **The owner's decision of 2026-09-19:** `max_epochs` 300 for every head, with lr 1e-3, patience 10 and early stopping on the in-domain validation split unchanged, so that early stopping, not the cap, ends every run. The heads at 224 were trained again. The 50-epoch runs' rows stay in the N-table, marked superseded (spec 005 US-8; DECISIONS 80).

10. **Training on several manifests (closed on 2026-09-19 by the owner).** N2's "Makerere + iBean → Tanzania" trains on Makerere's and iBean's train splits and stops early on their validation splits. The runs are quotable although iBean's role is `test_only` (US-5.1; spec 001 Clarification 10).
11. **Anthracnose in N2 (closed on 2026-09-19 by the owner).** The Tanzania heads keep their three classes. N2 decides among the classes both sides have, healthy and rust, and scores only those rows (spec 005 US-7).

## Out of scope

Abstention and calibration (004). The N-table and the verdict (005). Serving (006). Hyper-parameter search: the recipe is H8's, one config per head.
