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
| `configs/` | `backbones/*.yaml`, `datasets/*.yaml` (downloads), `manifests/*.yaml` (manifest recipes, spec 001), `class_map_v1.yaml`, `heads/*.yaml`, `eval.yaml`, `abstain.yaml`, `service.yaml` (the model the service answers with) | W1–W5 |
| `tests/` | pytest; `fixtures/ibean_30/` runs on CPU | W1 onwards |
| `results/` | `compute_log.jsonl` (N7), `n_table.jsonl` + `n_table.md` (the numbers), `verdict.md` — small, tracked in git | W1 onwards |
| `cards/` | `TEMPLATE.md` and one rendered card per registered model | W5 |

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
and each writes its seconds to the compute log. `make eval` then scores them into the N-table:
N1 on each run's own test split, and N2 on the cross-dataset directions of
`configs/eval.yaml` (a run over two manifests takes `--train-manifest a.jsonl b.jsonl`).

## Abstention and calibration (S4.4)

What lets the model say "I don't know" (`specs/004-abstention/spec.md`). Four post-hoc scores
over a trained head: the confidence (its largest logit), the distance to the k-th nearest
training feature, the relative Mahalanobis distance as a second opinion, and energy, logged
only. A row abstains when the confidence falls below `tau_conf` or the distance rises above
`tau_knn`, and `abstain_reason` names which. Every threshold, and the temperature that
calibrates the probabilities, is fitted on the run's own **in-domain validation slice** and
never on a test manifest.

```bash
make abstain                                   # every run under data/heads/ -> abstain.json
make abstain ARGS="--run <run_id> --force"     # one of them, fitted again
```

The recipe is `configs/abstain.yaml`: `k`, the declared coverages, the distance scores' TPR,
the ECE bins and N3's held-out sources. Its sha256 is an input of every fit, so a change there
is a new fit — and, because a row's identity carries no fit, it has to be settled before the
first N3 row is written (DECISIONS 91). A fit lands beside its run in
`data/heads/<run_id>/abstain.json` (git-ignored) and writes one compute-log row.

## The service (S4.6)

The one thing other pieces call (`specs/006-service/spec.md`), and the interface H8 §6.2
fixes: `POST /v1/predict`, `POST /v1/predict_batch`, `GET /v1/model` (the card summary) and
`GET /v1/health`.

```bash
make serve                                      # uvicorn on 127.0.0.1:8000
uv run python -m ms.service.example > request.json
curl -s -H "content-type: application/json" -d @request.json http://127.0.0.1:8000/v1/predict
curl -s http://127.0.0.1:8000/v1/model | python -m json.tool
```

A request is piece 1's `observation_frame` record (interface v0) plus `request_id` and
`options`, checked against their schema file itself. A contract violation is HTTP 422 whose
body still reads as a response — `decision` abstain, `abstain_reason` `invalid_input`, and
the validator's reasons — so one consumer branch reads both outcomes, and a broken frame of
a replayed batch costs only itself. The answer holds the temperature-scaled scores over the
known classes, the decision and its reason, the uncertainty block of H8 §6.2, and
`model_version`, which with `head.run_id` leads back to the exact run that answered.

The bytes at `frame.uri` are hashed and the hash must equal `frame.sha256`, so a wrong hash
cannot select another image's features. A hash the cache holds is answered without the
backbone; anything else is embedded on the spot and rounded to float16 exactly as the cache
stores it, so a replayed mission and a live frame are one model and not two.

Which model that is comes from `configs/service.yaml` — the verdict's backbone
(`dinov2_l14_reg` at 224 on CLS) and the owner's head and training manifests — never "the
newest run under `data/heads`". The service refuses to start when the config resolves to no
run, to several, or to a run without its `abstain.json`.

Every request and response, the 422s included, is one JSON line of
`data/predictions/requests.jsonl` (git-ignored; `MS_REQUEST_LOG`). Piece 3 moves those rows
into its predictions table when it has one.

## Latency (N6)

What a frame costs the service, per frame, at batch 1 and 32, for both backbones and both
resolutions (H8 §6.7). It times the service's own functions — no HTTP server, because the
transport is not what N6 measures — on a model built for each (backbone, resolution) that
shares the served model's head, recipe, seed, tokens and training manifests.

```bash
make bench ARGS="--notes 'the GPU was idle and no other session was working'"
make bench ARGS="--backbone dinov2_l14_reg --res 224 --batch 1 --path cache"
```

Four metrics per model: `latency_ms:b1` and `latency_ms:b32` are H8's N6, the cost of a frame
the service has never seen (preprocess + backbone + head + the abstention scores);
`cached_b1` and `cached_b32` are the replay path of H8 §6.9, the same frames answered from
the cache. The value is the median per frame and the interval the 5th and 95th percentiles
over the calls — one measurement with one seed, as N4 is, not a five-seed aggregate.

N6 is **reported, never optimised for** (H8 §6.9) and never a tie-breaker (H9 §7 criterion 4).
Measure it with nothing else on the GPU and say so: `--notes` goes into every row.

## The registry and the model card (H8 §6.8)

Every head run is one MLflow run, in one experiment per backbone
(`piece4-heads-<backbone_id>`), logging the manifest hashes, the preprocessing string, the
seed, N1–N3 and the calibration temperature. A model is **registered only when it has a
card**, and the card has to be the one the artefacts render now — so a model whose numbers
moved cannot be registered until its card is written again.

```bash
make card ARGS="--run <run_id>"       # cards/<model_version>.md, rendered from the artefacts
make register ARGS="--run <run_id>"   # log every run, then register that one
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

The card is `cards/<model_version>.md`, rendered from `cards/TEMPLATE.md`: the prose is the
template's and the same for every v0 model, and every number, hash, licence and attribution
line comes from the run, its `abstain.json`, the frozen manifests and `results/n_table.jsonl`.
Nobody edits a rendered card by hand — `register` refuses one that differs from what the
artefacts render. A number the table does not hold is an honest blank, never a zero.

Registering lands one version of `piece4-bean-disease-v0`, tagged with the card's path and
its sha256: H8 D-7's model set, which a winning challenger joins and never evicts.

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
