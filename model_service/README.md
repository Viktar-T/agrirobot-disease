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
make hub-check    # Hugging Face access: account, gated + current sha per backbone, DINOv3 load
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
| `configs/` | `backbones/*.yaml`, `datasets/*.yaml` (downloads), `manifests/*.yaml` (manifest recipes, spec 001), `class_map_v1.yaml`, `heads/*.yaml`, `eval.yaml` | W1–W3 |
| `tests/` | pytest; `fixtures/ibean_30/` runs on CPU | W1 onwards |
| `results/` | `compute_log.jsonl` (N7), `n_table.jsonl` + `n_table.md` (the numbers), later `verdict.md` — small, tracked in git | W1 onwards |
| `cards/` | model cards, one per registered model | W5 |

## The pipeline (the W2 vertical slice)

The whole chain once, on iBean (`docs/piece4-work-plan.md` §3; DECISIONS 33–41). Its numbers
only exercise the pipeline and are never quoted: iBean is test-only and unblocked.

```bash
make manifests DS=ibean                         # spec 001: data/manifests/ibean_v1.jsonl + sidecar
make cache BB=dinov2_l14_reg RES=224 DS=ibean   # spec 002: data/cache/... + a compute-log row
uv run python -m ms.heads.train --backbone dinov2_l14_reg --res 224 --head linear --seed 0 \
    --train-manifest data/manifests/ibean_v1.jsonl --split train --val-split val --allow-test-only
make eval                                       # one N1 row -> results/n_table.jsonl + .md
make serve                                      # then, from another shell:
uv run python -m ms.service.example > request.json
curl -s -H "content-type: application/json" -d @request.json http://127.0.0.1:8000/v1/predict
```

On Windows, `.\make.ps1 manifests -DS ibean`, `.\make.ps1 cache -BB dinov2_l14_reg -RES 224
-DS ibean`, and so on. Every step is idempotent: run it again and it does nothing. Heads land
in `data/heads/<run_id>/` (git-ignored).

## The heads (S4.3)

Three heads on the cached features, five seeds each (`specs/003-heads/spec.md`): `linear`
(logistic regression), `proto` (4 prototypes per class, cosine over a learned temperature)
and `mix` (their fixed 0.5/0.5 mixture). The recipe is `configs/heads/<head>.yaml`.

```bash
make heads BB=dinov2_l14_reg RES=224 ARGS="--train-manifest data/manifests/makerere_v1.jsonl"
make heads BB=dinov3_l16 RES=224 ARGS="--train-manifest data/manifests/tanzania_v1.jsonl --head proto --seed 0"
```

A call trains linear, proto and mix for seeds 0–4 unless `--head` and `--seed` say otherwise,
and skips every run that already exists. Runs land in `data/heads/<run_id>/` (git-ignored),
and each writes its seconds to the compute log. `make eval` then scores them into the N-table.

## The site probe (N4)

How easily the cached CLS features give away where a picture was taken: which bean set, and
within Makerere which district (DECISIONS 54). It needs the 224 caches of every backbone.

```bash
make probe        # every backbone config at 224, both targets -> N4 rows in results/, MLflow runs
make probe ARGS="--backbone dinov2_l14_reg --target district"      # one of them
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db     # experiment piece4-n4-site-probe
```

The MLflow store `mlruns/` is local and git-ignored until piece 3 runs the server;
`MLFLOW_TRACKING_URI` points the code at a server instead (DECISIONS 55).

## The compute log (N7)

`results/compute_log.jsonl` — one JSON line per unit of compute, written by code only
(`ms.compute_log`), never by hand. The first line is the machine itself
(`step = "env"`: GPU, VRAM, driver, CUDA, torch, python); every extraction, head run,
evaluation and service benchmark appends its own row with `wallclock_s`.

```bash
python -m ms.compute_log record-env   # idempotent: an unchanged machine adds no row
python -m ms.compute_log show
```
