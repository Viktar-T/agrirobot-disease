# N-table

Rendered from `n_table.jsonl` by `make eval` (`ms.eval.run`) and `make probe`
(`ms.eval.probe`); do not edit. Every number carries its split rule (D-9). Seeds `0–4`
mark the mean over five seeds with its 95 % Student t interval; the per-seed rows
behind it are in `n_table.jsonl`. A row marked *not quotable* only exercises the
pipeline (for example the W2 slice, trained on a test-only manifest).

## N1 — within-dataset

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | linear | 0 | ibean_v1 (train) → ibean_v1 (test) | macro_f1 | 0.977 | — | 1.00 | 172 | `unblocked:random_by_phash_group` | *not quotable* |
| dinov2_l14_reg | 224 | cls | linear | 0 | ibean_v1 (train) → ibean_v1 (test) | recall:healthy | 0.965 | — | 1.00 | 85 | `unblocked:random_by_phash_group` | *not quotable* |
| dinov2_l14_reg | 224 | cls | linear | 0 | ibean_v1 (train) → ibean_v1 (test) | recall:rust | 0.989 | — | 1.00 | 87 | `unblocked:random_by_phash_group` | *not quotable* |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.991 | 0.990–0.993 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.995 | 0.994–0.996 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.987 | 0.984–0.991 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.994 | 0.992–0.997 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.996 | 0.994–0.998 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.992 | 0.987–0.998 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.993 | 0.991–0.995 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.994 | 0.990–0.997 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.993 | 0.988–0.998 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.991 | 0.990–0.993 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.994 | 0.992–0.996 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.989 | 0.987–0.991 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.991 | 0.990–0.993 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.989 | 0.988–0.991 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.994 | 0.992–0.996 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.991 | 0.990–0.992 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.988 | 0.985–0.992 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.995 | 0.993–0.997 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.994 | 0.994–0.995 | 1.00 | 22368 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.994 | 0.994–0.994 | 1.00 | 1309 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.977 | 0.976–0.977 | 1.00 | 1649 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.995 | 0.995–0.996 | 1.00 | 22368 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.994 | 0.994–0.994 | 1.00 | 1309 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.981 | 0.979–0.983 | 1.00 | 1649 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.996 | 0.995–0.996 | 1.00 | 22368 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.994 | 0.994–0.995 | 1.00 | 1309 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.983 | 0.981–0.985 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.992 | 0.990–0.993 | 1.00 | 22368 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.992 | 0.992–0.992 | 1.00 | 1309 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.998 | 0.997–0.999 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.985 | 0.985–0.986 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.996 | 0.996–0.996 | 1.00 | 22368 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.994 | 0.993–0.995 | 1.00 | 1309 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.985 | 0.984–0.987 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.996 | 0.995–0.996 | 1.00 | 22368 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:anthracnose | 0.995 | 0.994–0.995 | 1.00 | 1309 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → tanzania_v1 (test) | recall:rust | 0.983 | 0.980–0.987 | 1.00 | 1649 | `blocked:date` |  |

## N2 — cross-dataset

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.545 | 0.539–0.550 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.228 | 0.221–0.236 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.527 | 0.480–0.573 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 0.998–1.001 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.207 | 0.147–0.267 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.525 | 0.459–0.591 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 0.998–1.001 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.206 | 0.122–0.291 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.516 | 0.511–0.520 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.193 | 0.187–0.198 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.520 | 0.510–0.531 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.199 | 0.186–0.211 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | macro_f1 | 0.531 | 0.514–0.548 | 1.00 | 863 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:healthy | 1.000 | 0.998–1.001 | 1.00 | 427 | `unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → ibean_v1 (all) | recall:rust | 0.211 | 0.191–0.232 | 1.00 | 436 | `unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.475 | 0.457–0.493 | 1.00 | 2655 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.435 | 0.400–0.471 | 1.00 | 2655 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.417 | 0.365–0.469 | 1.00 | 2655 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.485 | 0.479–0.492 | 1.00 | 2655 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.367 | 0.343–0.392 | 1.00 | 2655 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_crops_v1 (test) | recall:rust | 0.319 | 0.283–0.354 | 1.00 | 2655 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.362 | 0.360–0.364 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.018 | 0.016–0.020 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.363 | 0.347–0.379 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.018 | 0.003–0.034 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.365 | 0.341–0.390 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.021 | -0.002–0.045 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.360 | 0.359–0.361 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.016 | 0.015–0.017 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.347 | 0.344–0.350 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.004 | 0.002–0.007 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | macro_f1 | 0.346 | 0.344–0.348 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:healthy | 1.000 | 1.000–1.000 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | tanzania_v1 (train) → makerere_v1 (test) | recall:rust | 0.003 | 0.001–0.005 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.889 | 0.878–0.901 | 1.00 | 21059 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.960 | 0.954–0.965 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.981 | 0.978–0.984 | 1.00 | 1649 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.861 | 0.843–0.878 | 1.00 | 21059 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.949 | 0.941–0.957 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.957 | 0.942–0.972 | 1.00 | 1649 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.733 | 0.666–0.799 | 1.00 | 21059 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.918 | 0.898–0.939 | 1.00 | 19410 | `blocked:date` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.687 | 0.555–0.818 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.876 | 0.867–0.884 | 1.00 | 21059 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.952 | 0.948–0.956 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.992 | 0.992–0.992 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.812 | 0.772–0.853 | 1.00 | 21059 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.916 | 0.890–0.942 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.992 | 0.991–0.993 | 1.00 | 1649 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | macro_f1 | 0.718 | 0.652–0.783 | 1.00 | 21059 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:healthy | 0.857 | 0.803–0.910 | 1.00 | 19410 | `blocked:date` |  |
| dinov3_l16 | 224 | cls | proto | 0–4 | makerere_v1+ibean_v1 (train) → tanzania_v1 (test) | recall:rust | 0.917 | 0.838–0.996 | 1.00 | 1649 | `blocked:date` |  |

## N4 — site-prediction probe

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.983 | 0.980–0.986 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.992 | 0.990–0.994 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.767 | 0.749–0.787 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.770 | 0.746–0.795 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |

## Superseded

These rows stay in `n_table.jsonl`, marked; they are left out above and out of the
verdict (spec 005 US-8).

- max_epochs 50 -> 300, the owner's decision of 2026-09-19: the 50-epoch runs: N1/N2, 102 aggregate row(s) and 510 per-seed row(s)
