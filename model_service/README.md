# model_service/ — piece 4: model service v0

H8 §4 piece 4, detailed in H8 §6 · H6 E15 · owner M · W1–W6

One service behind one interface — `(frame + metadata) → (scores, uncertainty, abstain/unknown, model version)` — backed by cached frozen DINOv2 (champion) / DINOv3 (challenger) features, small heads and an abstention rule trained on the open bean sets; plus the evaluation harness and a thin ROS 2 client.

**Done when**: the acceptance criteria of H8 §6.11 are met.

The week-by-week execution is `docs/piece4-work-plan.md`; dated choices are in `DECISIONS.md`.

## Environment (W1)

One Python 3.11 environment for the whole repository, declared in the root `pyproject.toml`
and locked in `uv.lock`; the package `ms` lives in `src/ms/` and is installed editable.
Run everything from the **repository root**:

```bash
make env          # uv sync + .env from .env.example + the environment row in the compute log
make test         # pytest (model_service/tests)
make lint         # ruff
make env-check    # what torch sees: version, CUDA, device
```

On the Windows dev box there is no `make`; `.\make.ps1 env`, `.\make.ps1 test`, … run the
same commands. `make` (the Makefile) is what the Linux/WSL side uses.

`.env` (git-ignored, from the tracked `.env.example`) holds `HF_TOKEN` — needed only for the
gated DINOv3 weights, after the terms are accepted on the ML role's own account, and an
optional `HF_HOME` so the checkpoints do not land on the system drive.

## Layout

| Path | What | Week |
|---|---|---|
| `specs/` | one folder per contract: `001-manifests/` … `006-service/`, each `spec.md` | W1–W5 |
| `src/ms/` | the package: `compute_log.py`, then `data/`, `cache/`, `heads/`, `abstain/`, `eval/`, `service/` | W1–W5 |
| `configs/` | `backbones/*.yaml`, `datasets/*.yaml`, `heads/*.yaml`, `class_map_v1.yaml`, `eval.yaml` | W1–W3 |
| `tests/` | pytest; `fixtures/ibean_30/` runs on CPU | W1 onwards |
| `results/` | `compute_log.jsonl` (N7), later `n_table.jsonl`, `verdict.md` — small, tracked in git | W1 onwards |
| `cards/` | model cards, one per registered model | W5 |

## The compute log (N7)

`results/compute_log.jsonl` — one JSON line per unit of compute, written by code only
(`ms.compute_log`), never by hand. The first line is the machine itself
(`step = "env"`: GPU, VRAM, driver, CUDA, torch, python); every extraction, head run,
evaluation and service benchmark appends its own row with `wallclock_s`.

```bash
python -m ms.compute_log record-env   # idempotent: an unchanged machine adds no row
python -m ms.compute_log show
```
