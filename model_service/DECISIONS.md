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
