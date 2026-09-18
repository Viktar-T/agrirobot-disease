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
   either way.
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
Note the VRAM: 8 GB decides the extraction batch sizes at 518/512 in W3 and is the reason
the high-res pass is a nightly job.
