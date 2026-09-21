# plot_map/ — piece 6: weekly plot-state map v0

H8 §4 piece 6 · H6 E16 · owner P (+M for the rule) · W6–W9

Per plot-visit aggregation in SQL from the predictions table; state OK / watch / verify by a fixed, documented placeholder rule; abstentions shown as verification load; rendered as a three-colour map.

**Done when** (H8): the map regenerates from the store by one command after a replay and shows the deliberately broken frame absent and the verification load present.

## Hand-over from piece 4 (W6, 2026-09-21): the prediction log

Piece 6 draws the map from predictions. Piece 4 writes one JSON line per request/response
pair; piece 3 moves those lines into the `predictions` table when it has one, and the field
names do not change on the way, so a query written against the file works against the table.

**The one thing to know first: the log has no rows yet.** The writer, the schema and the
service are done and tested; the *frames* are not there, because piece 2 is a placeholder. See
"What is missing" below — it is the reason the map cannot be built today, and it is not
piece 4's to unblock.

### What piece 6 gets now

| Artefact | Where | Made by | Contract |
|---|---|---|---|
| the request log | `data/predictions/requests.jsonl` (git-ignored; `MS_REQUEST_LOG`, or `request_log` in `model_service/configs/service.yaml`) | the service, on every call | spec 006 US-9 |
| the service that writes it | `make serve` → `POST /v1/predict`, `POST /v1/predict_batch`, `GET /v1/model`, `GET /v1/health` | `ms.service.app` | spec 006, H8 §6.2 |
| an example request body | `uv run python -m ms.service.example > request.json` | `ms.service.example` | interface v0 |
| the model behind it | `model_service/cards/msv0.1+dinov2_l14_reg@224.linear.man-0670c6-423a16.md` | `make card` | H8 §6.8 |

### The row (spec 006 US-9.1), in the order it is written

`ts`, `request_id`, `frame_uid`, `frame_sha256`, `uri`, `site_id`, `plot_id`, `zone_id`,
`timestamp_utc`, `platform`, `bbch`, `bbch_null_reason`, `model_version`, `run_id`,
`coverage`, `decision`, `abstain_reason`, `top1`, `scores`, `uncertainty`, `features`,
`status`, `timing_ms`, `errors`, `warnings`.

The log holds the frame's **hash**, never its bytes and never its features.

What the map needs from it: `plot_id` and `zone_id` to aggregate by, `timestamp_utc` to make a
plot visit, `decision` (`predict` or `abstain`) for the three-colour rule, `abstain_reason`
and `status` to tell a refusal from a broken frame, `top1` and `scores` for the state, and
`model_version` + `run_id` so a map can say which model drew it.

### Two real rows

Produced on 2026-09-21 by calling the service twice on iBean test frames — an answered frame
and a deliberately broken one — with `MS_REQUEST_LOG` pointed at a scratch file:

```json
{"ts": "2026-09-21T06:22:22Z", "request_id": "669148ab-…", "frame_uid": "ibean_0365f72a067e356d",
 "frame_sha256": "0365f72a…", "uri": "ibean/extracted/train/bean_rust/bean_rust_train.30.jpg",
 "site_id": "ibean", "plot_id": "unknown", "zone_id": "open", "timestamp_utc": "2026-09-21T06:22:22Z",
 "platform": "emulator", "bbch": null, "bbch_null_reason": "not_recorded",
 "model_version": "msv0.1+dinov2_l14_reg@224.linear.man-0670c6-423a16",
 "run_id": "dinov2_l14_reg-224-linear-cls-s0-89fe37fe", "coverage": 0.9,
 "decision": "predict", "abstain_reason": null, "top1": "rust",
 "scores": {"healthy": 0.064959, "rust": 0.935041},
 "uncertainty": {"max_prob": 0.935041, "entropy": 0.240399, "ood_knn": 0.359918,
                 "ood_knn_threshold": 0.485453, "ood_maha": 1.864594},
 "features": "cache", "status": 200,
 "timing_ms": {"preprocess": 0.0, "backbone": 0.0, "head": 77.418, "total": 82.993},
 "errors": null, "warnings": ["metadata.bbch is null"]}
```

```json
{"ts": "2026-09-21T06:22:22Z", "request_id": "1e8c2ef3-…", "frame_uid": "ibean_041265a271457db5",
 "frame_sha256": "0000…", "uri": "ibean/extracted/validation/healthy/healthy_val.39.jpg",
 "site_id": "ibean", "plot_id": "unknown", "zone_id": "open", …,
 "coverage": null, "decision": "abstain", "abstain_reason": "invalid_input",
 "top1": null, "scores": null, "uncertainty": null, "features": null, "status": 422,
 "timing_ms": {"total": 7.63},
 "errors": [{"path": "frame.sha256", "message": "the bytes at uri hash to 041265a2…, not frame.sha256"}],
 "warnings": null}
```

**Three kinds of row, and the map must tell them apart.**

| `status` | `decision` | `abstain_reason` | what it means for the map |
|---|---|---|---|
| 200 | `predict` | null | a state: use `top1` and `scores` |
| 200 | `abstain` | `low_confidence` or `far_from_training` | **verification load**: the model saw the frame and would not name it |
| 422 | `abstain` | `invalid_input` | a **broken frame**: it must show as *absent* from the plot, not as verification load |

The 422 body still reads as a response, which is what lets one consumer branch read both
outcomes and what makes a broken frame of a replayed batch cost only itself.

`scores` covers only the classes the served model has. **The demo model has two** — `healthy`
and `rust` — because neither of its training manifests holds an anthracnose row; the card says
so, and a map legend must not promise a third colour for a class the model cannot output.

### How to produce rows today

```bash
make serve                                             # uvicorn on 127.0.0.1:8000
uv run python -m ms.service.example > request.json     # one cached iBean frame, as a mission would send it
curl -s -H "content-type: application/json" -d @request.json http://127.0.0.1:8000/v1/predict
tail -1 data/predictions/requests.jsonl
```

`POST /v1/predict_batch` takes `{"requests": [...]}`, at most 32 per call, and is the replay
mode. A frame whose hash the cache already holds is answered without the GPU, at about 4 ms
per frame in a batch of 32 — which is why a replayed mission needs no GPU at all.

### What is missing: the frames

`data/predictions/requests.jsonl` **does not exist yet**: no frame has been served outside the
test suite, so the map has nothing to aggregate. Four things have to exist first, none of them
piece 4's:

1. **Piece 2's emulator.** `emulator/` holds a README and nothing else, so there is no mission
   file, no replay and no frames on a topic.
2. **The frame topic's name.** Nothing in this repository fixes it.
3. **A ROS 2 message type** for piece 1's record and for the response on
   `/apv/disease/scores` — piece 1's contract is a JSON Schema, and whether it travels as a
   `.msg` or as JSON in a string is pieces 1 and 2's to settle.
4. **The `uri` forms for MCAP frames**, still open in `interface/README.md`. Until they are
   closed the service answers `mcap://` with **501**, so a real bag frame cannot reach it, and
   a replayed frame reaches it only if the emulator writes the bytes under the data root and
   puts that path in `frame.uri`.

Two consequences for piece 6 to plan around:

- **`site_id`, `plot_id` and `zone_id` have never carried a real value.** They come from
  piece 1's record, filled by the emulator; the example above fills them synthetically
  (`plot_id: "unknown"`, `zone_id: "open"`). The plot-visit aggregation is written against
  fields that exist in the schema and have never been exercised with mission data.
- **The "deliberately broken frame" of piece 6's own definition of done is reachable today** —
  it is the 422 row above — and so is the verification load, as `decision: "abstain"`. Neither
  needs to wait for the emulator to be *designed against*; both need it to be *demonstrated*.
