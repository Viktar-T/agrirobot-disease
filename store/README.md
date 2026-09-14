# store/ — piece 3: ingest and observation store

H8 §4 piece 3 · H6 E14 · owner P · W2–W5 (the W1 item was this repository's skeleton)

Validator at ingest → MCAP extractor → PostGIS tables (sites, plots, plants, frames, predictions, plot_visits; station rows as a stub) → DVC manifests for every dataset and split → MLflow tracking server. No annotation tooling in Demo 1.

**Done when** (H8): one cart-equivalent pass is queryable by plot and time, a manifest hash identifies every image in every split, and every model run has an MLflow run ID.
