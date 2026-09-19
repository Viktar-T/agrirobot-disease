# interface/ — piece 1: interface v0 (the minimal observation contract)

H8 §4 piece 1 · H6 E12 · owner P + E · W1–W2 (v0.1 in W10)

What the robot, or any stand-in for it, hands over per frame, and what it accepts back (trigger / waypoint request stubs). Delivered as a JSON Schema, a validator library used by every consumer, and ROS 2 message definitions.

**Done when** (H8): a synthetic frame validates, a frame missing any mandatory field is rejected with a named reason, and the robotics side has read and acknowledged the document.

## Schema v0 — DRAFT (W1; hand-shake with piece 4 done 2026-09-19 by P, see below; E and the robotics side have not reviewed it yet)

| File | Direction | Status |
|---|---|---|
| `schema/v0/observation_frame.schema.json` | robot or stand-in → disease system, one record per frame | draft |
| `schema/v0/trigger_request.schema.json` | disease system → robot | stub |
| `schema/v0/waypoint_request.schema.json` | disease system → robot | stub |

JSON Schema 2020-12; the files are ASCII-only, so they open with any default encoding. Each schema carries one valid example (`examples`); the frame example is the H8 §6.2 request metadata, adjusted to the choices below.

### Choices the draft makes beyond the H8 field list (to confirm)

Piece 4 (the model service) confirmed choices 1, 4 and 5 and asked for choice 11 in the hand-shake of 2026-09-19 (`model_service/DECISIONS.md` 28–32). The other choices are still to confirm.

1. **Shape = `frame` + `metadata`**, as in the model-service request (H8 §6.2), so that request = this record + `request_id` + `options`.
2. **Added `contract_version`** (`"v0"`), so a validator can tell v0 from v0.1 (W10).
3. **Added `frame.uri`, `frame.width`, `frame.height`** (taken from §6.2): the record must point at the pixels.
4. **`pose` = camera pose at exposure; x/y/z in metres, `yaw` in radians in [-π, π]** (ROS REP 103). The §6.2 example (`"yaw": 91.0`) reads as degrees and is rejected.
5. **`bbch`** = integer 0–99 or `null`; **`bbch_null_reason`** is required when `bbch` is null and not allowed otherwise, one of `not_recorded | not_applicable | unknown` (§6.2 has the free text "not recorded").
6. **Nullable but never missing**: `camera.exposure_us`, `camera.gain` and `bbch` may be `null` (open datasets record none), but the key must be present.
7. **Strict frame record** (`additionalProperties: false` at every level): unknown or misspelt fields are rejected. The two stubs accept extra fields until they are designed.
8. **Identifiers** (`frame_uid`, `site_id`, `plot_id`, `zone_id`, `sowing_batch`, `pose.frame_id`): `[A-Za-z0-9_.-]`, at most 128 characters, no spaces, slashes or colons. Reserved values: `zone_id = open`, `sowing_batch = unknown`.
9. **`timestamp_utc`** = acquisition (exposure) time, not ingest time; must end in `Z`.
10. **`sha256`** = hash of the image bytes exactly as stored at `uri` (the encoded file, or the MCAP message data).
11. **The image id is `frame.frame_uid`** (decided 2026-09-19, hand-shake with piece 4). `frame_id` is left to the coordinate frame (`metadata.pose.frame_id`), which is what the name means in ROS. The rename was made while v0 was still a draft and no code read it.

### Open questions for the review
- **`camera.flash` vs `illumination_mode`** overlap: a record can say flash = true and mode = ambient. Keep both with a consistency rule, or drop one?
- **`camera.gain` unit**: dB (machine-vision camera) or ISO (phone)?
- **`uri` forms** for MCAP frames, and for files (relative to which root?) — with piece 2, W3.
- **Pose of the camera or of the platform** (+ camera extrinsics)?
- **No `plant_id`, no mission / visit id**, although the mission file (piece 2), the `plants` and `plot_visits` tables (piece 3) and the per-visit map (piece 6) use them.
- **Return channel**: the two stubs are identical except for intent; one message with a `kind` field would do.
- **Deferred to the full contract** (G4 §2.3), not in v0: panel-relative position (under / edge / inter-row / open) and distance to the panel edge, days since sowing, sun angles, irradiance, cultivar and seed lot, treatment, EPPO codes, annotation and lab-confirmation fields.
