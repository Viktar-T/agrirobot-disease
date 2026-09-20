# Feature specification: 006 — Model service v0

**Branch**: `006-service` · **Created**: 2026-09-20 · **Status**: Specified 2026-09-20, ahead of W5, with `model_service/tests/test_service.py`: 30 red until the W5 "Service" and "N6" tasks, 19 green. The W2 slice already serves `POST /v1/predict` from the cache, ahead of this spec (DECISIONS 33, 37), so the request contract, the 422s, the cached-hash path and `model_version` are the green ones; what it leaves for W5 is US-4.2 (features on request), US-5 (abstention and calibration), US-6 (the batch), US-7.2 (the card summary), US-9 (the request log), US-10 (the served model) and US-11 (N6)
**Input**: H8 §6.2 (the interface, copied verbatim in US-1), §6.9 (serving), §6.1 (scope and non-goals), §6.7 (the N6 row), §6.11 items 1 and 2, `docs/piece4-work-plan.md` S4.6 and the W5 tasks, interface v0 (`interface/schema/v0/observation_frame.schema.json` and `interface/README.md`) with the hand-shake of DECISIONS 28–32, the slice's choices (DECISIONS 33, 37) and specs 001 (rows, hashes, roles), 002 (caches, preprocessing), 003 (runs, `model_version`), 004 (the decision, the temperature, FR-015) and 005 (rows, aggregates, metric names). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Everything piece 4 has built so far is files on a disk: manifests, caches, runs, thresholds and a table. The service is the one thing other pieces can call, and it is the only artefact of piece 4 that the demo actually exercises — piece 2 replays a virtual mission into it, piece 3 keeps what it answered, piece 6 draws the map from that. H8 §6.1 puts it plainly: every consumer talks to *a model* without knowing *which* model.

That is why the contract, not the model, is what this spec fixes. The response carries `model_version`, so an answer leads back to the exact run that gave it; it carries `decision` beside `scores`, so "I don't know" is a first-class answer and not a low number someone has to interpret; and a request that breaks the contract is a 422 with the validator's reason rather than a guess, because those 422s are the signal piece 3 wants (H8 §6.2).

The three hard parts are all about not lying. A frame whose features are already cached must get the same answer as one computed on the spot, or the demo's cheap path and the real path are two different models. The abstention thresholds must be the ones fitted in domain (spec 004 US-2), never re-derived from what the service happens to see. And `model_version` must be reproducible: H8 §6.11 item 2 asks that the string reproduce the run, which means the manifest hashes and the weights hash stand behind it.

## Commands

    make serve            # uvicorn ms.service.app:app --host 127.0.0.1 --port 8000
    python -m ms.service.example [--manifest …] [--split test] [--index 0 | --image-id <id>]
    python -m ms.service.bench  [--service model_service/configs/service.yaml]
        [--batch 1 32] [--n 64] [--warmup 8] [--device auto|cpu|cuda] [--path computed|cache]
        [--n-table model_service/results/n_table.jsonl] [--compute-log …]   # N6 (US-11)

`ms.service.app` is the service; `ms.service.example` prints a request for one manifest row (the W2 slice; DECISIONS 33). `ms.service.bench` is the only writer of N6 rows, and it calls the service's own code paths, not a copy of them.

Exit codes for the two modules: 0; 2 on an input error, with nothing written. The service itself refuses to start on a bad configuration (FR-013) rather than answering from a model nobody chose.

## User scenarios and testing

### US-1: The interface, verbatim (priority P1)

H8 §6.2, the contract this spec implements, verbatim:

> ### 6.2 The interface
>
> Transport: HTTP/JSON (FastAPI-style) on the central-node machine, plus a thin ROS 2 client node that subscribes to the contract frame topic, calls the service and republishes the response on `/apv/disease/scores`. gRPC can replace HTTP later without touching the schema. Endpoints: `POST /v1/predict`, `POST /v1/predict_batch` (replay mode), `GET /v1/model` (card summary), `GET /v1/health`.
>
> **Request** (the frame reference and the contract fields of piece 1; the service re-validates and never silently fills a missing field):
>
> ```json
> {
>   "request_id": "uuid",
>   "frame": {"uri": "mcap://bag/topic/seq or file path", "sha256": "…", "width": 4096, "height": 3000},
>   "metadata": {
>     "site_id": "MAK-Luwero", "plot_id": "P07", "zone_id": "open", "timestamp_utc": "2021-04-22T09:14:03Z",
>     "pose": {"x": 1.2, "y": 0.4, "z": 0.6, "yaw": 91.0, "frame_id": "plot", "source": "synthetic"},
>     "platform": "emulator", "illumination_mode": "ambient",
>     "camera": {"model": "Samsung-M13", "exposure_us": null, "gain": null, "flash": false, "white_balance": "auto"},
>     "sowing_batch": "unknown", "bbch": null, "bbch_null_reason": "not recorded"
>   },
>   "options": {"coverage_target": 0.90, "return_patch_map": false}
> }
> ```
>
> **Response:**
>
> ```json
> {
>   "request_id": "uuid",
>   "model_version": "msv0.1+dinov2-l14-reg@518.proto-k4.man-3f9a1c",
>   "backbone": {"id": "facebook/dinov2-with-registers-large", "weights_sha256": "…", "input_res": 518, "tokens": "cls"},
>   "head": {"id": "proto-k4", "train_manifest": "dvc:manifests/tz155k_v1.json", "calibration": {"temperature": 1.37}},
>   "scores": {"healthy": 0.81, "rust": 0.12, "anthracnose": 0.05},
>   "top1": "healthy",
>   "decision": "predict",
>   "abstain_reason": null,
>   "uncertainty": {"max_prob": 0.81, "entropy": 0.62, "ood_knn": 0.42, "ood_knn_threshold": 0.55, "ood_maha": 11.9},
>   "conformal_set": ["healthy"],
>   "localisation": null,
>   "timing_ms": {"preprocess": 9, "backbone": 140, "head": 1},
>   "warnings": ["metadata.bbch is null"]
> }
> ```
>
> Rules. `decision ∈ {predict, abstain}`; `abstain_reason ∈ {low_confidence, far_from_training, invalid_input}`; `scores` are temperature-scaled probabilities over the *known* classes only — "unknown" is a **decision**, not a class with a probability (F3 §12); `model_version` is a string composed of the service version, the backbone id and input resolution, the head id and the **hash of the training manifest** — so two responses with the same string are reproducible from DVC + MLflow; every response carries `timing_ms`; `warnings` never change the decision. Contract violations return HTTP 422 with the validator's reason and are logged — they are the signal piece 3 wants.

Three details of that block are settled elsewhere and this spec follows them, not the example: the record's field names and shape are piece 1's schema, where the image id is `frame.frame_uid` (DECISIONS 31) and `bbch_null_reason` is `not_recorded`, not `"not recorded"` (DECISIONS 30); `"yaw": 91.0` is invalid under v0, which is radians (DECISIONS 29); and `head.train_manifest` names the manifest the run records, a path and its sha256, because `dvc:manifests/tz155k_v1.json` is a form that does not exist here (spec 003).

### US-2: The request is the record plus two fields (P1)

1. **Given** a request body, **then** it is exactly piece 1's `observation_frame` record (`contract_version`, `frame`, `metadata`) plus `request_id` and an optional `options`, and nothing else. The record is validated against `interface/schema/v0/observation_frame.schema.json` itself, never against a copy (DECISIONS 28).
2. **Given** a body that fails any of those checks, **then** the answer is **HTTP 422** whose payload carries `request_id` (or null when it is unusable), `model_version`, `decision = "abstain"`, `abstain_reason = "invalid_input"`, `errors` — one `{path, message}` per problem, sorted, the validator's own wording — and `timing_ms`. Nothing is predicted, and no feature is read.
3. **Given** `options`, **then** its keys are `coverage_target` (a number, default the fit's `default_coverage`) and `return_patch_map` (a boolean, default false). An unknown option is a 422; `return_patch_map: true` is accepted with a warning and changes nothing, because v0 has no localisation (H8 §6.1).
4. **Given** a `coverage_target`, **then** it is matched to a fitted coverage by `ms.abstain.coverage_key`'s two decimals, so `0.9` and `0.90` are the same operating point. A value the served run was **not** fitted at — `0.85`, `1.0`, anything outside the fit's `coverages` — is a 422 naming the fitted ones (spec 004 FR-015). A service is not allowed to interpolate an operating point it never measured, and coverage 1.0, which the N-table uses for "no abstention", is not an operating point the service offers.
5. The 422 cases a test pins: a missing `metadata` field; the pre-rename `frame.frame_id`; a yaw in degrees; `bbch = null` without `bbch_null_reason`; a body that is not JSON; an unknown top-level key; a missing or empty `request_id`; an unknown option; an unfitted `coverage_target`; a `frame.sha256` that the bytes do not hash to; a `uri` that resolves outside the data root.

### US-3: The response (P1)

1. **Given** a valid request, **then** the answer holds every field of US-1's response block, plus `frame_uid` echoed from the request (DECISIONS 31) and `features` (US-4.3). `scores` has one entry per trained class of the served run and no other, and they sum to 1 within float tolerance. The response is a **closed** schema: FastAPI publishes it, `additionalProperties: false`, as a documented superset of H8's example, and a field is added by changing this spec.
2. **Given** the served run's `abstain.json`, **then** `scores` are the **temperature-scaled** probabilities of spec 004 US-5.2, `head.calibration` is `{"temperature": T}`, and `uncertainty.max_prob` and `uncertainty.entropy` are computed on those same probabilities, the entropy in nats.
3. **Given** any answer, **then** `top1` is the argmax of `scores`. It is a label, not a decision: a consumer reads `decision` to know whether the service stands behind it.
4. **Given** any answer, **then** `conformal_set` and `localisation` are null in v0 (spec 004 "out of scope"; H8 §6.1), and `timing_ms` carries `preprocess`, `backbone`, `head` and `total`, in milliseconds.
5. **Given** a condition worth naming — `metadata.bbch` is null, `return_patch_map` was asked for, the served head is not quotable, the backbone is research-only until H6 C5 (DECISIONS 13) — **then** it is a string in `warnings`, and **no warning ever changes `decision`, `scores` or `abstain_reason`** (H8 §6.2). A test asserts that the same frame answered with and without a warning-raising field gives the same decision and the same scores.

### US-4: The cached-hash path and features on request (P1)

1. **Given** `frame.uri`, **then** the bytes are fetched from a path relative to the data root, an absolute path inside it, or `file://` inside it; anything outside is a 422, and a scheme that is still open between pieces 1 and 2 (`mcap://`) is **501** (DECISIONS 37). The bytes are hashed and the hash must equal `frame.sha256` (DECISIONS 32); a mismatch is a 422, so a wrong hash cannot select another image's features.
2. **Given** that hash, **then**:
   - it is looked up in the caches of the served run's `cache_key`, over every manifest cached under it. A hit is answered from the cached features, **without loading the backbone** (H8 §6.9), and `timing_ms.backbone` is 0;
   - a miss computes the features on the spot: `ms.cache.preprocess` with the run's `preprocess_string`, then the backbone at the pinned revision whose `weights_sha256` the run records, then **the same float16 rounding the cache stores** (spec 002 FR-005), then `ms.heads.features` for the run's token spec. Nothing is written to the cache: a cache is keyed to a manifest (spec 002), and a served frame belongs to none.
3. **Given** either path, **then** `features` in the response says which was taken, `"cache"` or `"computed"`.
4. **Given** one frame answered both ways on the same device and compute dtype, **then** the two answers are the **same**, not merely close: the float16 rounding of US-4.2 puts both through one numerical path. Across compute dtypes — a cache extracted in fp16 on the GPU, a frame computed in fp32 on the CPU — they agree on `decision` and `top1` and within spec 002's golden tolerance on the scores (DECISIONS 38). Otherwise the demo's cheap path and the real path are two different models. A test pins this on the fixture.
5. **Given** no GPU, **then** the service runs on the CPU and gives the same answers, more slowly (H8 §6.9). Features are computed in the run's `compute_dtype` where the device supports it and in `float32` where it does not, which is the difference the golden tolerance of US-4.4 already bounds (DECISIONS 26, 38). The device is a setting; it never changes the contract, the resolution or `model_version`.
6. **Given** a hash that sits in more than one cache under the key, **then** the first cache in name order answers and the others are ignored: under one `cache_key` the features are the same image's, so the choice cannot change an answer. `GET /v1/model` lists the caches it indexed.

### US-5: Abstention, calibration and the coverage target (P1)

1. **Given** the served run, **then** the service loads the `abstain.json` beside it (spec 004 FR-015) and never fits anything itself: no threshold, no temperature and no Gaussian is derived from what the service is shown.
2. **Given** a frame's features and the coverage in force, **then** `decision` and `abstain_reason` are exactly `ms.abstain.decide`'s (spec 004 US-4): abstain when `conf < tau_conf` or `knn > tau_knn`, with `low_confidence` first when both cross.
3. **Given** any answer, **then** `uncertainty.ood_knn` is the frame's kNN distance, `ood_knn_threshold` is the `tau_knn` in force, and `ood_maha` is the second opinion — recorded, never consulted (spec 004 US-4.3). The block holds exactly H8 §6.2's five fields and no more.
4. **Given** a run that has no `abstain.json`, **then** the service refuses to start (FR-013) rather than answering `predict` to everything. The W2 slice did answer `predict` to everything, by a warning that said so (DECISIONS 37); from W5 that is a configuration error.
5. **`abstain` is reachable**: a test drives a frame far from the training features and gets `decision = "abstain"` with `far_from_training`, and one with a flat posterior and gets `low_confidence`. This is H8 §6.11 item 1, "`abstain` reachable".
6. `invalid_input` is the third reason of H8 §6.2 and it is this spec's alone (spec 004 US-4.4): it is the 422 of US-2.2, and it never comes from a score.

### US-6: The batch endpoint (P1)

1. **Given** `POST /v1/predict_batch` with `{"requests": [<a /v1/predict body>, …]}`, **then** the answer is `{"responses": [...]}` — one entry per request, **in the order they were sent** — where each entry is exactly what `POST /v1/predict` would have answered for that body, its 422 payload included.
2. **Given** a batch whose envelope is valid, **then** the HTTP status is 200 even when some items are invalid: a replayed mission must not lose 500 good frames to one broken one. An item's own `decision`, `abstain_reason = "invalid_input"` and `errors` carry its failure, and a batch-level `n_invalid` counts them.
3. **Given** a broken envelope — not an object, no `requests`, `requests` not a list, empty, or longer than `max_batch` (FR-012) — **then** the whole call is a 422 in the shape of US-2.2, with no item answered.
4. **Given** several frames whose features must be computed, **then** they are embedded in one backbone call of at most the backbone config's `batch_size`, which is what makes a batch worth having (N6, US-11).
5. **Given** a batch, **then** every item is logged as its own request/response pair (US-9), because piece 3's predictions table is per frame.

### US-7: `GET /v1/model` and `GET /v1/health` (P1)

1. `GET /v1/health` answers `{status, model_version, cached_frames}` without touching a model, so a liveness probe costs nothing.
2. `GET /v1/model` is the **card summary** (H8 §6.2): `model_version`, `run_id`, the backbone (id, `hf_id`, revision, `weights_sha256`, `input_res`, tokens, `cache_key`, `research_only_until_c5`), the head (id, config sha256), the training manifests with their sha256s and splits, the classes, the fitted coverages with the operating point in force, the temperature, `quotable`, and the path of the model card once one exists (W5 "Registry + card").
3. **Given** the card summary, **then** everything H8 §6.11 item 2 needs to reproduce the run is in it: the manifest hashes and the weights hash (US-8).

### US-8: `model_version` reproduces the run (P1)

1. **Given** a served run, **then** `model_version` is `ms.heads.model_version`'s string — `msv0.1+<backbone_id>@<res>.<head_id>.man-<train manifest sha256[:6]>`, several manifests joining their first six digits with `-` — and it is the same string in `run.json`, in `abstain.json`, in every N-table row of that run, in the response and in `GET /v1/model`.
2. **Given** the string and the card summary, **then** the run is reproducible: the backbone's `weights_sha256` and pinned revision, the cache key, the training manifests' sha256s and splits, the head config's sha256 and the seed name the inputs, and `run.json`'s `run_id` is derived from them (spec 003 FR-002).
3. **Given** two runs that differ in anything the string does not carry — the seed above all — **then** their `model_version` is the same string and their `run_id` is not. The string names the *model*, the run id names the *fit*; a response carries both, which is what makes item 2 checkable. A test pins this, so that nobody later reads `model_version` as a unique key.

### US-9: The request log (P1)

1. **Given** any answered call, valid or 422, **then** one JSON line is appended to the request log: `ts`, `request_id`, `frame_uid`, `frame_sha256`, `uri`, the `site_id`, `plot_id`, `zone_id`, `timestamp_utc`, `platform`, `bbch` and `bbch_null_reason` of the record (DECISIONS 30), `model_version`, `run_id`, `coverage`, `decision`, `abstain_reason`, `top1`, `scores`, `uncertainty`, `features`, `status`, `timing_ms`, `errors` and `warnings`.
2. **Given** the log, **then** it is append-only, one line per request/response pair, written after the answer is formed and never in a way that can change it: a log that cannot be written is a `warning` on the response and a message on the service's own log, not a 500.
3. **Given** piece 3, **then** the file is what it reads until the predictions table exists (work plan W5), so the fields above are named once here and not renamed later.
4. The log holds no image bytes and no features — it holds the frame's hash, which is how a row is tied back to the picture.

### US-10: Which model is served (P2)

1. **Given** `model_service/configs/service.yaml`, **then** it names the served run: `backbone_id`, `res`, `tokens`, `head`, `seed` and the training manifests, or a `run_id` outright, plus the default coverage and the roots. The service resolves it to exactly one run under the heads root.
2. **Given** no such run, several matching runs, a run without `abstain.json`, or a run whose manifests have changed since it was trained, **then** the service refuses to start and says which (FR-013). "The newest run under `data/heads`" was the slice's rule (DECISIONS 37) and it stops here: with hundreds of runs on disk, newest is not a choice anybody made.
3. **Given** the verdict (spec 005 US-6), **then** the backbone it names is the one the config names — `dinov2_l14_reg` at 224 on CLS as of 2026-09-20 (DECISIONS 108–109) — and a change of verdict is a change of this file, not of code.
4. **Given** a served run that is not quotable, or a backbone that is research-only until H6 C5, **then** every response says so in `warnings` and `GET /v1/model` says so in the summary.

### US-11: N6 — latency (P2)

1. **Given** `python -m ms.service.bench`, **then** it measures the service's own path end to end and appends N6 rows through `ms.eval.append_rows` (spec 005 FR-009): at batch 1 and batch 32, for both backbones and both resolutions (H8 §6.7; work plan W5). It calls the functions the endpoint calls, on a served model it builds for each (backbone, resolution) in turn, so the eight numbers are comparable although the service itself serves one model (US-10). No HTTP server is started: the transport is not what N6 measures.
2. **Given** a measurement, **then** the row's `metric` is `latency_ms:<qualifier>` with the qualifier one of `b1`, `b32`, `cached_b1`, `cached_b32`: the two `b*` are H8's N6, the cost of a frame the service has never seen (preprocess + backbone + head); the two `cached_*` are the replay path of H8 §6.9, reported beside them. `value` is the median millisecond cost **per frame**, `ci_low` and `ci_high` the 5th and 95th percentiles over the timed frames, and `n` the number of frames timed after the warm-up.
3. **Given** a row, **then** it carries the run's own identity (`backbone_id`, `res`, `token_type`, `head`, `seed`, `run_id`, `model_version`, `cache_key`), the frames' manifest and split as `test_manifest` / `test_split` with their `split_rule`, `coverage` 1.0, and `notes` naming the device, the warm-up, the repetitions and the batch size. There is no aggregate: N6 has one seed and a percentile interval, as N4 does (spec 005 US-2.3, DECISIONS 54).
4. **Given** a run of the bench, **then** it appends one `serve` row to the compute log (N7) with its wall-clock and device, and running it again appends no N-table row it already holds (spec 005 FR-004).
5. **Given** the GPU, **then** the numbers are measured with nothing else running on it, and the row's `notes` say so. N6 is reported, never optimised for (H8 §6.9) and never a tie-breaker (spec 005 US-6.7).

### Edge cases

- **A frame in no cache and no GPU.** It is computed on the CPU (US-4.5). Slow is an answer; 501 is not, once W5 lands.
- **A batch of the same frame.** Answered once per item, logged once per item; the cache lookup is per hash and costs nothing twice.
- **A frame whose bytes decode to a different size than `frame.width` / `frame.height`.** A warning, not an error: the contract's width and height are the sender's claim, and the decision is taken on the pixels.
- **A zero-byte or undecodable file at a valid `uri` with a matching hash.** A 422 with the decoder's reason, not a 500.
- **Two requests with the same `request_id`.** Both answered and both logged; de-duplication is piece 3's, and the service holds no state between calls.
- **`options.coverage_target` on a batch.** Per item, because each item carries its own options.

## Requirements

### Functional

- **FR-001 Endpoints.** `POST /v1/predict`, `POST /v1/predict_batch`, `GET /v1/model`, `GET /v1/health` (H8 §6.2), and the OpenAPI document FastAPI generates from the request and response models (work plan §1, S4.6). No other route exists in v0.
- **FR-002 Request.** As US-2: the record of `interface/schema/v0/observation_frame.schema.json` plus `request_id` (a non-empty string) and `options`. Validation is by that schema file, with `format` checking on, plus the two extra fields; the record is never filled in or repaired.
- **FR-003 Response fields**, in this order: `request_id`, `frame_uid`, `model_version`, `backbone`, `head`, `scores`, `top1`, `decision`, `abstain_reason`, `uncertainty`, `conformal_set`, `localisation`, `features`, `timing_ms`, `warnings`. `decision ∈ {predict, abstain}`; `abstain_reason ∈ {low_confidence, far_from_training, invalid_input}` or null; `uncertainty` holds `max_prob`, `entropy`, `ood_knn`, `ood_knn_threshold`, `ood_maha`; `timing_ms` holds `preprocess`, `backbone`, `head`, `total`.
- **FR-004 Error response.** HTTP 422 with `request_id`, `model_version`, `decision = "abstain"`, `abstain_reason = "invalid_input"`, `errors` (a sorted list of `{path, message}`) and `timing_ms`. HTTP 501 is kept for a `uri` scheme pieces 1 and 2 have not fixed yet, with `request_id`, `frame_uid` and `detail` (DECISIONS 37).
- **FR-005 Features.** Cached lookup by the recomputed sha256 over the served run's cache key, first cache in name order winning (US-4.6); otherwise computed with `ms.cache.preprocess` on the run's `preprocess_string`, then the run's backbone at its pinned revision and `weights_sha256`, in the run's `compute_dtype` where the device supports it and `float32` where it does not, in batches of at most the backbone config's `batch_size`, then rounded to float16 as spec 002 stores it, then `ms.heads.features` for the run's token spec. `features ∈ {cache, computed}` says which. The two paths are identical at one compute dtype and within spec 002's golden tolerance across dtypes (US-4.4; DECISIONS 38). Nothing is written to a cache (FR-015).
- **FR-006 Decision.** `ms.abstain.decide` and `ms.abstain.probabilities` on the loaded `Fit` (spec 004 FR-013), at the coverage in force. The service fits nothing.
- **FR-007 Batch.** `{"requests": [...]}` in, `{"responses": [...], "n_invalid": <int>}` out, order preserved, per-item 422 payloads, envelope errors as FR-004, at most `max_batch` items.
- **FR-008 `model_version`.** `ms.heads.model_version`'s string, unchanged (spec 003). The service composes no string of its own.
- **FR-009 The request log.** JSONL, append-only, at the configured path (FR-012), one line per request/response pair including 422s, with the fields of US-9.1. A failure to write is a warning, never a 500.
- **FR-010 N6.** `ms.service.bench` writes the rows of US-11 through `ms.eval.append_rows`, and one `serve` row to the compute log. `ms.eval.METRICS` gains `latency_ms` with a new qualifier kind, `batch`, whose vocabulary is `b1`, `b32`, `cached_b1`, `cached_b32`; `validate_row` rejects a bare `latency_ms` and any other qualifier as `bad_metric`.
- **FR-011 Timing.** `timing_ms` is measured with `time.perf_counter` around the three stages and the whole call, in milliseconds, rounded to three decimals. It is on every response, the 422s included (H8 §6.2, "every response carries `timing_ms`").
- **FR-012 Config.** `model_service/configs/service.yaml`: the served run (US-10.1), `default_coverage`, `max_batch` (32, the backbone configs' batch size), `device` (`auto`), `heads_root`, `cache_root`, `data_root`, `backbone_configs` (the directory US-4.2 reads the backbone's recipe from) and `request_log`. Every value is overridable by the environment variables the slice already reads (`MS_HEAD_RUN`, `MS_HEADS_ROOT`, `MS_CACHE_ROOT`, `MS_DATA_ROOT`), and `MS_SERVICE_CONFIG` names the file.
- **FR-013 Checks at start-up.** Before the service answers anything: the config parses; it resolves to exactly one run; the run loads and its manifests are unchanged; its `abstain.json` loads and matches the run (spec 004 FR-013's `inputs_sha256` check); the cache of its key exists; the data root exists; the request log's folder is writable; the backbone weights are on disk when the device can load them. A failure stops start-up with `<reason>: …`. The reasons are `bad_config`, `missing_run`, `ambiguous_run`, `missing_abstain`, `missing_cache`, `frozen_manifest_modified`, `missing_weights` and `bad_value`.
- **FR-014 API.** `ms.service.app` exports `Settings`, `Served`, `create_app(settings) -> FastAPI`, `request_errors`, `resolve_uri` and `app`; `ms.service.example` exports `request_for(row)` and `main(argv)`; `ms.service.bench` exports `main(argv)`; `ms.service.log` exports `log_request(path, row)` and `read_rows(path)`. The response is built by one function, so `/v1/predict` and `/v1/predict_batch` cannot drift apart.
  - What W5 adds to the modules it reads: a way to build features from freshly embedded tokens rather than from a `Cache` (`ms.heads`), and `latency_ms` in `ms.eval.METRICS` (FR-010). Nothing is added to `ms.abstain`: a batch takes its per-row reasons by calling `decide` once per row, so spec 004's API list (FR-013) is unchanged.
- **FR-015 What the service never does.** It never writes to a cache, never writes an N-table row (only `ms.service.bench` does), never fits a threshold or a temperature, never trains, and never repairs a request.

## Success criteria (measurable, technology-agnostic)

- **SC-1** The service answers the interface for a virtual-mission frame and for a bag frame — one from the cache and one computed on the spot — with `abstain` reachable and `invalid_input` on a broken frame (H8 §6.11 item 1).
- **SC-2** `GET /v1/model` carries everything the card needs, and `model_version` plus the run id lead back to the run that answered (H8 §6.11 item 2).
- **SC-3** One frame answered from the cache and computed on the spot gives the same decision and scores within the golden tolerance.
- **SC-4** A batch of *n* frames answers *n* responses in order, and one invalid item costs only that item.
- **SC-5** Every request and response, 422s included, is one line of the request log, and piece 3 can read it without asking what a field means.
- **SC-6** N6 exists in the N-table for both backbones, both resolutions and both batch sizes, with the device in `notes`.
- **SC-7** The CPU tests use the committed fixture and synthetic runs, run in under a minute, and need neither network nor GPU.

## Assumptions

- Piece 1's v0 schema is the one in this repository, and v0.1 (W10) will not break it (work plan §6).
- The `uri` forms beyond a file path are open with piece 2 until W3 of the emulator's own plan; until then `mcap://` is a 501 (DECISIONS 28, 37).
- The head runs, their caches and their `abstain.json` exist as specs 002–004 leave them, and the manifests are frozen (DECISIONS 51).
- The ROS 2 client (H8 §6.9) is a separate thin node, built on the Linux/WSL side with piece 2; it calls this service over HTTP and adds nothing to the contract.
- MLflow registration and the model card are the W5 "Registry + card" task, and read `GET /v1/model`; nothing here registers anything.

## Clarifications

Closed on 2026-09-20 by this spec, against H8 §6.2 and §6.9, piece 1's schema and the W5 tasks. Where H8 is silent, the reading is the one that keeps an answer traceable.

1. **A batch is a list of single-frame requests**, `{"requests": [...]}`, not a frame list under one shared `request_id`. One schema then serves both endpoints, each item keeps its own `request_id` and `options` for piece 3's per-frame log, and "replay mode" (H8 §6.2) is literally *n* replayed requests. The cost is a little repetition in the body, which is what makes the log rows self-contained.
2. **A broken item does not fail the batch** (US-6.2). H8's 422 rule is about a request; in a batch the request is the envelope. A replayed mission of thousands of frames would otherwise be lost to one bad record, and the per-item 422 payload keeps the signal piece 3 wants.
3. **The served model is a config entry, not "the newest run"** (US-10). The slice's rule was fine with one run on disk and is wrong with 361. The verdict names the backbone; the head, the seed and the training manifest are the owner's choice at the "Registry + card" task, and the card records them. **Open for the owner**: which head and which training manifest the demo serves — the verdict settles the backbone and nothing else.
4. **`frame_uid` is in the response although H8's example has no such field.** The example predates the rename (DECISIONS 31), a consumer needs the frame's identity to join a response to a frame, and the slice already echoes it (DECISIONS 37).
5. **The `uncertainty` block is exactly H8's five fields.** `conf` and `tau_conf` are not added, although `low_confidence` is then a reason the response does not quantify: widening the contract is a v0.1 question to settle with piece 1, and `abstain.json` holds `tau_conf` beside the response (spec 004 US-4.2). Energy is never in the response (spec 004 US-1.4).
6. **A computed frame is not written to the cache.** A cache belongs to a manifest and is keyed by one (spec 002); a served frame belongs to no manifest, and a service that grew its own cache would make `cached_frames` a number nobody could reproduce.
7. **N6 is measured once, on one machine**, the one the compute log's environment row describes. A row has no device field (spec 005 FR-001), so two devices would share a row identity and the second would be dropped (FR-004). The device is in `notes`; a device axis is a change to spec 005, and v0 does not need one.
8. **`top1` is a label, not a decision** (US-3.3). It is the argmax of the scores whatever `decision` says, so that an abstained frame still shows what the head leaned towards — which is exactly what a human reviewing an abstention wants to see.
9. **A missing `abstain.json` is a configuration error, not a warning** (US-5.4). The slice answered `predict` to everything and said so in `warnings`; from W5, a service that cannot abstain is not the service H8 §6.11 item 1 asks for.
10. **The request log lives under `data/`**, which is git-ignored and DVC's (work plan §0), because it grows without bound and holds one line per frame of every replayed mission. `results/` is for the small tracked artefacts.
11. **A contract violation is an HTTP 422 whose body is still readable as a response.** H8 §6.2 makes `invalid_input` an `abstain_reason` *and* says contract violations return 422; both readings are satisfied by one answer — the status is 422, and the payload carries `decision = "abstain"` and `abstain_reason = "invalid_input"`, so one consumer branch reads both outcomes. This is what the slice already does (DECISIONS 37), and it is what makes US-6.2's per-item payloads work.
12. **An abstained response still carries `scores` and `top1`.** H8 gives no example of one. Abstention is a post-hoc gate on a head that has already run (H8 §6.6), so hiding what it leaned towards would throw away exactly what a human reviewing an abstention needs (US-3.3).
13. **The record is validated against piece 1's schema file** until piece 1 ships the validator library DECISIONS 28 asks for; when it does, the service changes one import and this line. `interface/README.md` still lists the `uri` forms as open, so US-4.1 promotes the slice's rule (DECISIONS 37) to the contract and `mcap://` stays a 501 until pieces 1 and 2 close it.
14. **The data root is `data/raw`, one root.** The derived crops (`data/derived`, DECISIONS 46) are a training artefact and never arrive as a frame; a second root would widen what a `uri` may reach for no case that exists.
15. **A computed feature is rounded to float16 before the head sees it** (US-4.2), because that is how the cache stores it (spec 002 FR-005). The live path then *is* the replay path, bit for bit, instead of being within a tolerance of it — which is the whole reason H8 §6.9 puts the cache in front of the backbone. The precision given up is the precision the whole campaign already gave up when the caches were written. Not taken: keeping the computed path in float32 and bounding the difference, which would leave the demo answering one thing and the live service another, by a margin nobody measures per frame.
16. **"Not quotable" is read from the run's `quotable`, not from `--allow-test-only`.** A run that trained on Makerere + iBean is quotable although iBean is test-only (DECISIONS 74), so it raises no warning; a run of the W2 slice's kind is not quotable and raises one (DECISIONS 35).

## Out of scope

- **The ROS 2 client** (H8 §6.9; work plan W5, Thu–Fri): a separate node on the Linux/WSL side, built with piece 2 and E. This spec ends at HTTP.
- **MLflow registration and the model card** (H8 §6.8; W5 Wed): the card is a template, not a spec (work plan §1). The service exposes what the card needs (US-7.2) and registers nothing.
- **Localisation, severity, patch maps and conformal sets** (H8 §6.1; spec 004 out of scope): the fields stay null in v0.
- **gRPC**, authentication, rate limiting and multi-model serving: H8 §6.9 puts the service on the central-node machine behind the map, not in a control loop.
- **The predictions table** (piece 3). The JSONL log is the hand-over until it exists.
- **Anything that changes a number.** The service reports N6 and nothing else; N1–N3 come from `ms.eval.run` on the same runs.
