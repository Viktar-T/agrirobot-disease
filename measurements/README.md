# measurements/ — piece 5: the measurements report

H8 §4 piece 5 · H6 F16 · owner M (+P for data) · W4–W8

The numbers from the piece-4 evaluation harness, written up as one note: in-/out-of-domain pairs with abstention for both backbones, the site-prediction probe, the negative control and the colour-normalisation ablation.

**Done when** (H8): the note is filed and its numbers are the ones on the demo slide.

## Hand-over from piece 4 (W6, 2026-09-21): the N-table

Piece 4's numbers are one file. Piece 5's report is built from it, and piece 5's own two
measurements — the negative control **N5** and the **colour-normalisation** ablation — are
added to the same file, by the same code, under the same contract. This section says what is
in it today, how to read it, and what is not there.

### What piece 5 gets now

| Artefact | Where | Made by | Contract |
|---|---|---|---|
| the N-table | `model_service/results/n_table.jsonl` | `make eval`, `make probe`, `make bench` | spec 005 |
| the same, rendered | `model_service/results/n_table.md` | `ms.eval.write_md` | spec 005 |
| the verdict | `model_service/results/verdict.md` | `make eval` (`ms.eval.verdict`) | spec 005 US-6, from H9 §7 |
| the compute log (N7) | `model_service/results/compute_log.jsonl` | every step, via `ms.compute_log` | work plan §4; `make compute-log-check`, `make compute-log-summary` |
| the evaluation note | `measurements/A1_skeleton.md` | W6 | H8 §6.11 items 5 and 6 |
| the model card | `model_service/cards/<model_version>.md` | `make card` (rendered, never written) | H8 §6.8 |
| the frozen manifests, the feature caches | `data/manifests/`, `data/cache/` (DVC) | specs 001 and 002 | `store/README.md` |

All four `results/` files are small and tracked in git, so a report can name a commit and mean
exactly one set of numbers.

### The row (spec 005 FR-001)

One JSON line per number, UTF-8 with LF, appended and never rewritten. The 27 fields, in three
groups:

- **the number**: `ts`, `number` (`N1`…`N7`), `metric` (`macro_f1`, `recall:<class>`,
  `balanced_accuracy:<target>`, `unknown_recall`, `auroc:<score>`, `ece`, `selective_risk`,
  `abstention_rate[:<class>]`, `latency_ms:<batch>`), `value`, `ci_low`, `ci_high`, `n`;
- **what produced it**: `backbone_id`, `res`, `token_type`, `head`, `seed`, `run_id`,
  `model_version`, `cache_key`;
- **what it was measured on**: `train_manifest`, `train_manifest_sha256`, `train_split`,
  `test_manifest`, `test_manifest_sha256`, `test_split`, `split_rule`, `coverage`, `classes`,
  and then `quotable`, `superseded`, `notes`.

Every row carries the **split rule** and both manifests' **sha256** (D-9), so a number can
never be quoted without the exam it was sat on.

### The four rules that matter when reading it

1. **A row is never deleted and never changed**, except to gain the `superseded` mark
   (`python -m ms.eval.supersede`), which names the owner's recipe change that replaced its
   runs. Filter on `superseded is None` to get the current table.
2. **An aggregate has `seed: null`**: it is the mean of exactly seeds 0–4 with a 95 % Student-*t*
   interval in `ci_low`/`ci_high`. The five per-seed rows stay beside it. Quote aggregates.
3. **`quotable: false` marks a row that exercises the pipeline and means nothing** — today
   the 40 rows of the W2 iBean vertical slice, trained on a test-only manifest.
4. **N4 and N6 are single measurements, not five-seed aggregates** (`seed: 0`, intervals
   bootstrapped over rows for N4 and percentiles over calls for N6). Do not read their
   intervals as seed intervals.

So the rows a report may quote are:

```python
from ms import eval as E
rows = [r for r in E.read_rows()
        if r["superseded"] is None and r["quotable"] and r["seed"] is None]   # 3,070 today
```

### What is in it today (2026-09-21)

25,677 rows, of which 18,508 are current and **3,070 are current, quotable five-seed
aggregates**: N1 1,062 · N2 1,288 · N3 720. Beside them, N4's 4 rows and N6's 16.

Both backbones (`dinov2_l14_reg`, `dinov3_l16`), three heads (`linear`, `proto`, `mix`), three
training sets (Tanzania, Makerere, Makerere + iBean), three feature settings (224/CLS,
224/CLS⊕mean-patch, 518 or 512/CLS), five seeds: **270 current head runs**. The four N2
directions are in `model_service/configs/eval.yaml`. The headline numbers and their reading
are in `A1_skeleton.md`; it is the document to disagree with, not the raw file.

### What is missing, and it is piece 5's to add

| What | Why it is not here | H8 |
|---|---|---|
| **N5**, the negative control: an ImageNet-pretrained CNN fully fine-tuned on Tanzania, tested as N2 | never in piece 4's scope; the harness has no writer for it, but `N5` is already a valid `number` and `append_rows` will take its rows | §6.7, `[H46]` |
| **the colour-normalisation ablation**: what it moves on N4 and on N2 | never run. It is a different `preprocess_string`, so it is a different cache key and needs its own extraction pass; nothing in the code has to change | §6.7; claim B |
| **N4 "after colour normalisation"** | the same ablation seen from the probe's side. The "before" rows exist; the "after" ones do not | §6.7 |

Claim B of H8 §6.1 depends on the middle row and **cannot be written until it is run**. Its
first half — the site probe at 98.3 % and 99.2 % against a chance of 33.3 % — is measured; its
second half is blank in the note and must stay blank in the report.

One gap that is piece 4's, not piece 5's: **no N1 exists for a head trained on more than one
manifest**, which includes the model the service serves. `ms.eval.run` writes N1 only for a
single training manifest. It is a harness gap in spec 005 and the owner's to settle
(`model_service/DECISIONS.md` 129).

### How to add a number without breaking the table

```python
from ms import eval as E

problems = E.validate_row(row)          # [] or the reasons; checks the frozen manifest hashes
E.append_rows([row])                    # validates, skips identities already present, appends
E.append_rows(E.aggregate(per_seed))    # the five-seed row: seed null, mean, 95 % t interval
E.write_md()                            # re-render n_table.md; writes only if it changed
```

`append_rows` raises `NTableError` on the first invalid row and appends nothing, so a bad batch
cannot half-land. Identity is every field but the per-seed ones, so re-running adds nothing.
Do not append rows by hand, and do not edit `n_table.md`: it is rendered.
