# Feature specification: 005 — Results table (the N-table)

**Branch**: `005-results-table` · **Created**: 2026-09-19 · **Status**: Implemented 2026-09-19 with the W3 Heads task (`ms.eval`: row checks, aggregates, pairs, rendering; `ms.eval.run`: N1 per class) and the N2 task (US-7, `configs/eval.yaml`); acceptance tests in `model_service/tests/test_n_table.py` green (DECISIONS 68, 74). SC-2 met at 224 on 2026-09-19. The verdict (US-6) and its tests come with the W4 "Verdict" task.
**Input**: H8 §6.7 (results), H9 §7 (the verdict, as the work plan summarises it), `docs/piece4-work-plan.md` S4.5, the W3–W4 tasks and §4 (seeds; criteria written before the numbers), DECISIONS 36 (the provisional row) and 54 (the N4 row), specs 001 (split rules, hashes, freeze) and 003 (runs). Spec = contract + acceptance tests + protocol; no expected numbers on real data.

## Why

Every number piece 4 reports is a line of `results/n_table.jsonl`, and a number without its split rule and its data cannot be reported (H8 D-9). The file ties each number to both manifests (by hash), the run and the model version. It turns five seeds into one number with an interval, so that a difference can be judged. And it is the only input of the champion/challenger verdict, which code computes by rules written here before the numbers exist.

## Commands

`python -m ms.eval.run` (`make eval`) scores head runs into rows (N1, and N2 by the directions of `configs/eval.yaml`), aggregates the seeds and renders `n_table.md`; W4 adds `verdict.md`. `python -m ms.eval.probe` (`make probe`) writes the N4 rows. Both write rows only through `ms.eval.append_rows`.

    [--run <run_id> ...] [--heads-root data/heads] [--cache-root data/cache]
    [--manifest-root data/manifests] [--config model_service/configs/eval.yaml]
    [--n-table model_service/results/n_table.jsonl] [--md model_service/results/n_table.md]
    [--compute-log model_service/results/compute_log.jsonl]

Exit codes: 0; 2 on an input error or an invalid row, with nothing appended.

## User scenarios and testing

### US-1: Every row carries its data and its rule (priority P1)

1. **Given** any row, **then** it has the fields of FR-001 in that order, and `validate_row` finds nothing wrong with it. In particular:
   - `split_rule` is a non-empty rule from FR-003;
   - `train_manifest_sha256` and `test_manifest_sha256` are 64 hex digits each, one per manifest named in `train_manifest` and `test_manifest`, joined with `+` in the same order when there are several.
2. **Given** a row that fails a check of FR-001 to FR-005, **then** `validate_row` returns the problems, each starting with its reason. `append_rows` raises `NTableError` on the first problem and appends nothing.
3. **Given** the rows already in `results/n_table.jsonl` (the W2 slice's N1 row and the four N4 rows), **then** `validate_row` finds nothing wrong with any of them.

### US-2: Five seeds make one reported number (P1)

1. **Given** the per-seed rows of one number for seeds 0–4, **when** `aggregate` runs, **then** it returns one row for them. The rows of one number match in every field except `ts`, `value`, `ci_low`, `ci_high`, `seed`, `run_id`, `quotable` and `notes`. The aggregate row has:
   - `seed = null`; `value` is the mean of the five values;
   - `ci_low` and `ci_high` are the mean ∓ t(0.975, 4) · s / √5, with s the standard deviation (ddof = 1) and t = 2.776445;
   - `run_id` joins the five run ids with `+` in seed order; `ts` is the latest of the five;
   - `quotable` is true only when all five rows are quotable.
2. **Given** fewer than five seeds, or seeds other than 0–4, **then** there is no aggregate: the number is not reported yet.
3. Per-seed rows of N1–N3 have `ci_low = ci_high = null`, because one seed has no interval. N4's rows have one seed and a bootstrap interval, which their `notes` state (DECISIONS 54).

### US-3: N1 (P1)

1. **Given** a head run trained on one manifest (spec 003), **when** `make eval` runs, **then** it scores the test split of that manifest (`purpose = evaluate`) and appends these rows:
   - one with `metric = macro_f1`;
   - one `recall:<class>` for each of the head's classes, so that the rarest class's recall is never averaged away.

   `n` is the number of rows the value is computed over: the scored test rows for macro-F1, and that class's test rows for its recall. `split_rule` joins the scored rows' rules, `test_split` is `test`, and `coverage` is 1.0 (no abstention before W4).
2. **Given** all five seeds of a (backbone, resolution, tokens, head, training manifest and split), **then** `make eval` also appends one aggregate per metric (US-2).
3. **Given** a second `make eval` with nothing new, **then** it appends nothing, and `n_table.md` stays as it is.

### US-4: The table's invariants (P1)

1. **N1 and N2 come in pairs.** `unpaired(rows)` lists every (backbone_id, res, token_type, head) that has a quotable N1 aggregate but no quotable N2 aggregate, or the reverse, with the missing number. `make eval` prints the list, and the verdict (US-6) refuses to run while it is not empty.
2. **N3 uses only held-out classes.** An N3 row's `test_split` is `holdout_unknown`, and its `classes` are held-out unknowns only (`unknown_als`, `unknown_wm`). Otherwise the reason is `n3_not_held_out`.
3. **Quotable means frozen.** Given the frozen list (`FROZEN.jsonl`), every manifest hash of a quotable row appears in it (spec 001 SC-5). Otherwise the reason is `manifest_not_frozen`.

### US-5: The rendered table (P1)

1. `n_table.md` is rendered from `n_table.jsonl` alone, with one section per number.
2. It shows every aggregate row, with its seeds shown as `0–4`. It also shows every per-seed row that no aggregate covers, such as N4 and the W2 slice. The per-seed rows behind an aggregate stay in the `.jsonl`.
3. It is rewritten only when its text changes, and never by hand.

### US-6: The verdict (P2; its tests come with the W4 "Verdict" task)

These rules are declared now, before any N2 or N3 number exists (work plan §4). They copy the work plan's summary of H9 §7; Clarification 6 lists what still has to be checked against H9 §7 itself.

1. `make eval` writes `results/verdict.md` from the quotable aggregate rows alone, by code. No hand edits.
2. The champion is `dinov2_l14_reg` and the challenger `dinov3_l16`.
3. **N2.** The challenger wins a direction when its mean is at least 2 percentage points above the champion's and the two five-seed intervals do not overlap. It wins N2 when it wins at least two of the three directions: Tanzania → Makerere, Tanzania → iBean, Makerere + iBean → Tanzania.
4. **N3.** The challenger wins N3 when its AUROC is at least 0.02 above the champion's.
5. **Ties** are broken by N4, then N6.
6. **Licence is a hard gate.** A backbone whose licence does not allow the use cannot be chosen, whatever its numbers. The verdict states the gate for each backbone; for DINOv3 that is research-only until H6 C5 is signed (DECISIONS 13).

### US-7: N2 (P1; the owner's decisions of 2026-09-19)

1. **Given** the directions of `configs/eval.yaml` (`n2`), **then** a head run whose training manifests are a direction's `train` is scored on the direction's target. The directions are:
   - Tanzania → Makerere, on frames and on the rust crops;
   - Tanzania → iBean;
   - Makerere + iBean → Tanzania.
2. **Given** a direction, **then** it reads its target's frozen test split (`split: test`, the rows N1 reads for that set), or every row of a test-only target (`split: all`: iBean).
3. **Given** a head with a class the target lacks (Tanzania's anthracnose), **then** the decision is the argmax over the direction's classes only, healthy and rust. Rows of the other classes are not scored: the work plan's "anthracnose only where both sides have it".
4. **Given** a direction, **then** its rows are macro-F1 and one recall per class, or the metrics it names. The crop direction reports `recall:rust` only, because the crops hold no healthy class (DECISIONS 46).
5. Validation, aggregates, pairs and rendering are as for N1. Running `make eval` again adds nothing.

### Edge cases

- **Several manifests in one row.** Both manifest fields and both hash fields join their parts with `+` in the same order (N4, DECISIONS 54). `split_rule` joins the scored rows' distinct rules, sorted.
- **Rounding.** `value`, `ci_low` and `ci_high` are rounded to 4 decimals.
- **Append only.** Rows are appended by code and never edited in place. `make eval` never rewrites an existing row.

## Requirements

### Functional

- **FR-001 Fields**, in this order (DECISIONS 36): `ts`, `number`, `metric`, `value`, `ci_low`, `ci_high`, `n`, `backbone_id`, `res`, `token_type`, `head`, `seed`, `train_manifest`, `train_manifest_sha256`, `train_split`, `test_manifest`, `test_manifest_sha256`, `test_split`, `split_rule`, `coverage`, `classes`, `run_id`, `model_version`, `cache_key`, `quotable`, `notes`.
  - `ts` is `YYYY-MM-DDTHH:MM:SSZ`.
  - `value` is a finite number. `ci_low` and `ci_high` are both null or both numbers, with `ci_low ≤ ci_high`.
  - `n` is an integer ≥ 1, and `coverage` a number in (0, 1].
  - `seed` is an integer, or null for an aggregate.
  - `classes` is a non-empty list of strings, `quotable` a boolean, and `model_version` and `notes` strings or null.
  - `test_split` is `test`, `holdout_unknown`, `all` (every row of a test-only target, N2) or `cv<k>`: no number is scored on a train or validation split of a manifest that trains.
  - A failure of these is `missing_field`, `bad_field_order`, `bad_hash` or `bad_value`.
- **FR-002 Numbers.** N1 within-dataset, N2 cross-dataset, N3 unknown recall and AUROC, N4 site-prediction probe, N5 negative control (piece 5), N6 latency, N7 compute cost. Anything else is `bad_number`.
- **FR-003 Split rules.** Spec 001 FR-004's vocabulary (`blocked:<group key>`, `unblocked:random_by_phash_group`, `test_only`, `holdout_unknown`) plus the cross-validation rule `unblocked:<k>fold_by_phash_group` (N4). Several rules are distinct, sorted and joined with `+`. Anything else is `bad_split_rule`.
- **FR-004 Identity.** A row is the same number as another when these match: `run_id`, `number`, `metric`, `test_manifest_sha256`, `test_split` and `coverage`. A row whose identity is in the table is not appended again.
- **FR-005 Metrics.** A metric is `<name>` or `<name>:<qualifier>`. The names so far:
  - `macro_f1`, with no qualifier;
  - `recall:<class>`, where the class is one of the row's `classes`;
  - `balanced_accuracy`, or `balanced_accuracy:<target>` (N4).

  N3, S4.4, N6 and N7 add their names, with their tests, in their tasks. Anything else is `bad_metric`.
- **FR-006 Aggregation.** As US-2 says: over exactly seeds 0–4, with a 95 % Student t interval.
- **FR-007 Quotable.** A row is quotable only when its runs trained on `train_eval` manifests without `--allow-test-only` (spec 003) and its manifests are frozen. `n_table.md` marks the other rows "not quotable". The verdict reads quotable aggregates only.
- **FR-008 API.** `ms.eval` exports `FIELDS`, `NUMBERS`, `IDENTITY`, `NTableError`, `validate_row(row, frozen=None) -> list[str]`, `read_rows`, `append_rows(rows, path, frozen=None)`, `aggregate(rows) -> list[dict]`, `unpaired(rows)`, `render` and `write_md`.
- **FR-009 Writers.** Rows are written only by code: `ms.eval.run` for N1, and later N2 and N3, and `ms.eval.probe` for N4. All of them go through `append_rows`.
- **FR-010 N2 directions.** `configs/eval.yaml` lists them under `n2`. Each has `train` (the training manifests, as `<manifest>_v<N>`), `test` (the target), `split` (`test` or `all`), `classes` (the decision set, two trained classes at least) and optionally `metrics`. `ms.eval.run.decide(proba, classes, among)` is the decision of US-7.3.

## Success criteria (measurable, technology-agnostic)

- **SC-1** Every row in `results/n_table.jsonl` passes `validate_row`, and a test checks the committed file.
- **SC-2** Both backbones at 224 have N1 for the three heads × five seeds on `makerere_v1` and `tanzania_v1`, each number with its aggregate (W3 N1). N2 follows, in pairs with N1 (W3 N2).
- **SC-3** Running `make eval` twice appends nothing the second time and leaves `n_table.md` unchanged.
- **SC-4** `verdict.md` is written by code from the table alone (W4).

## Assumptions

- The head runs exist as spec 003 says, and the manifests are frozen (DECISIONS 51).
- Pieces 5 and 6 read `n_table.jsonl`, not the rendered `.md` (work plan W6).

## Clarifications

Closed on 2026-09-19:

1. **Per-seed rows and aggregate rows both live in the `.jsonl`.** Pieces 5 and 6 read the file, and the reported number, the mean with its interval over five seeds (work plan §4), has to be in it. The per-seed rows keep the spread visible.
2. **The interval over seeds is Student's t with 4 degrees of freedom.** With five values, a percentile interval would be just the minimum and the maximum.
3. **N1's metrics are macro-F1 and one recall per class.** The task asks for per-class recall "incl. the rarest", and one row per class keeps it.
4. **The N4 cross-validation rule joins the split-rule vocabulary**, as its rows already use it (DECISIONS 54).
5. **`seed = null` marks an aggregate**, and its `run_id` joins the five runs with `+`, as the manifest fields do for N4.

Open:

6. **The verdict's full text.** H9 §7 is not in the repository, so US-6 copies the work plan's summary. It has to be checked against H9 §7 before N2's numbers exist (W3 Thu):
   - which metric N2 compares (macro-F1 over the shared classes?);
   - which heads, resolution and coverage the verdict reads;
   - how an N2 win and an N3 win combine;
   - the order and thresholds of the N4 and N6 tie-breakers.
7. **N2's rows (closed on 2026-09-19 by the owner).** The crop-level direction reports rust recall only. Tanzania's heads decide among healthy and rust. The targets are the frozen test splits, plus every row of iBean. Makerere + iBean train together (US-7; spec 003 Clarifications 10–11).

## Out of scope

The numbers themselves. Abstention (004), which adds rows with `coverage` < 1. Plots and curves.
