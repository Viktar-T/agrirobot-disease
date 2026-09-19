# N-table

Rendered from `n_table.jsonl` by `make eval` (`ms.eval.run`) and `make probe`
(`ms.eval.probe`); do not edit. Every number carries its split rule (D-9). A row
marked *not quotable* only exercises the pipeline (for example the W2 slice, trained
on a test-only manifest).

## N1 — within-dataset

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | linear | 0 | ibean_v1 (train) → ibean_v1 (test) | macro_f1 | 0.977 | — | 1.00 | 172 | `unblocked:random_by_phash_group` | *not quotable* |

## N4 — site-prediction probe

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.983 | 0.980–0.986 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov2_l14_reg | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.767 | 0.749–0.787 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | ibean_v1+makerere_v1+tanzania_v1 (train) → ibean_v1+makerere_v1+tanzania_v1 (test) | balanced_accuracy:dataset (chance 0.333) | 0.992 | 0.990–0.994 | 1.00 | 23246 | `blocked:date+blocked:district+unblocked:random_by_phash_group` |  |
| dinov3_l16 | 224 | cls | logreg | 0 | makerere_v1 (cv5) → makerere_v1 (cv5) | balanced_accuracy:district (chance 0.083) | 0.770 | 0.746–0.795 | 1.00 | 10118 | `unblocked:5fold_by_phash_group` |  |
