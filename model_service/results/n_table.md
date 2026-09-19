# N-table

Rendered from `n_table.jsonl` by `python -m ms.eval.run` (`make eval`); do not edit.
Every number carries its split rule (D-9). A row marked *not quotable* only exercises
the pipeline (for example the W2 slice, trained on a test-only manifest).

## N1 — within-dataset

| backbone | res | tokens | head | seed | train → test | metric | value | 95 % CI | coverage | n | split rule | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_l14_reg | 224 | cls | linear | 0 | ibean_v1 (train) → ibean_v1 (test) | macro_f1 | 0.977 | — | 1.00 | 172 | `unblocked:random_by_phash_group` | *not quotable* |
