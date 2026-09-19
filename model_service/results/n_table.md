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
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.990 | 0.989–0.991 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.995 | 0.994–0.996 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.985 | 0.983–0.987 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.994 | 0.991–0.996 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.996 | 0.994–0.998 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.991 | 0.987–0.995 | 1.00 | 965 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.993 | 0.991–0.995 | 1.00 | 2015 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.994 | 0.990–0.997 | 1.00 | 1050 | `blocked:district` |  |
| dinov2_l14_reg | 224 | cls | proto | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.993 | 0.988–0.998 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.988 | 0.987–0.989 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.993 | 0.991–0.994 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | linear | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.983 | 0.980–0.987 | 1.00 | 965 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | macro_f1 | 0.992 | 0.991–0.993 | 1.00 | 2015 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:healthy | 0.990 | 0.988–0.991 | 1.00 | 1050 | `blocked:district` |  |
| dinov3_l16 | 224 | cls | mix | 0–4 | makerere_v1 (train) → makerere_v1 (test) | recall:rust | 0.994 | 0.991–0.996 | 1.00 | 965 | `blocked:district` |  |
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

## N4 — site-prediction probe

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.983 | 0.980–0.986 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.992 | 0.990–0.994 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.767 | 0.749–0.787 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.770 | 0.746–0.795 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |
