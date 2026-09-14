# agrirobot-disease

Code for the AgriRobot disease-detection work on bean ('Złota Saxa'). Current scope: **Demo 1 — "end-to-end without a robot"**, from W1 (14–18 Sep 2026) to the demo week W11 (23–27 Nov 2026).

The plan is not kept here. It lives in the project docs:

- **H8** — `AgriRobot/05_sourses/10.03_dr_choroby-fasoli-sparag/87_H8_current-plan-demo1-end-to-end-without-a-robot.md` (the single plan: pieces, calendar, done-when criteria)
- **H6** — `85_H6_actions-list.md` in the same folder (rows E12–E16 and F16 track the pieces)

## Layout

One directory per Demo 1 piece (H8 §4). Owner roles: **P** = platform/data, **M** = ML, **E** = embedded/imaging.

| Directory | H8 piece | Owner | Weeks | H6 row | State |
|---|---|---|---|---|---|
| `interface/` | 1 — interface v0 (minimal observation contract) | P + E | W1–W2, v0.1 in W10 | E12 | placeholder |
| `emulator/` | 2 — robot emulator | E | W2–W4, W7–W8 | E13 | placeholder |
| `store/` | 3 — ingest and observation store | P | W2–W5 | E14 | placeholder |
| `model_service/` | 4 — model service v0 | M | W1–W6 | E15 | placeholder |
| `measurements/` | 5 — the measurements report | M (+P) | W4–W8 | F16 | placeholder |
| `plot_map/` | 6 — weekly plot-state map v0 | P (+M) | W6–W9 | E16 | placeholder |
| `data/` | local data root (open sets, caches, bags) | P | from W1 | — | git-ignored |

## Conventions

- No data, bags, model weights, MLflow runs or secrets in git (see `.gitignore`); datasets get DVC versioning in W2 (piece 3).
- Text files are LF in every checkout (see `.gitattributes`).
- Team-facing documents are in English (H8 §12).
