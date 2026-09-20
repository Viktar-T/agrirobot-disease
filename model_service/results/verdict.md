# Champion/challenger verdict

**dinov2_l14_reg wins**: neither backbone won a criterion, and anything else goes to the champion.

Written by `make eval` (`ms.eval.verdict`) from `n_table.jsonl`; do not edit. The
rules are H9 §7's, copied into `specs/005-results-table/spec.md` US-6 before any
number existed, with the owner's readings of 2026-09-19 and 2026-09-20. The rows read
are the quotable five-seed aggregates that are not superseded, at 224 px, on
CLS, at coverage 1.0: 0 of them, the newest
written None.

- Champion: `dinov2_l14_reg` · Challenger: `dinov3_l16`

## Criterion 1 — N2, cross-dataset macro-F1

A backbone wins when, with **each** of the three heads, its mean over the three frame
directions is at least 2 pp above the other's, **and** in at least two directions it is
at least 2 pp ahead with the two five-seed intervals not overlapping. The crop-level
rows report rust recall and are not counted.

| head | challenger mean | champion mean | challenger − champion | directions ≥ 2 pp and separated | challenger wins |
|---|---|---|---|---|---|
| linear | — | — | — | — | no (a direction is missing) |
| proto | — | — | — | — | no (a direction is missing) |
| mix | — | — | — | — | no (a direction is missing) |

Per direction (challenger − champion, and whether the intervals separate):


**Criterion 1**: challenger does not win; champion does not win.

## Criterion 2 — N3, the AUROC of the decision scores

A backbone wins when, with **each** head, **each** held-out set, **both** decision
scores and **each** training set, its AUROC is at least 0.02 higher (the owner's
reading of 2026-09-20). The second opinion and energy are reported, not counted.

| head | held-out | score | training set | challenger | champion | challenger − champion |
|---|---|---|---|---|---|---|

The challenger clears the 0.02 bar in 0 of 0 comparisons, the champion in 0 of 0; a win needs all of them.

**Criterion 2**: challenger does not win; champion does not win.

## Outcome

- **dinov2_l14_reg wins**: neither backbone won a criterion, and anything else goes to the champion.
- A winning challenger **joins** the model set behind the service interface, and never
  evicts the champion in the same season (H8 D-7).
- **The licence is a hard gate** for anything that ships, whatever the numbers say:
  DINOv3 is research-only until H6 C5 is signed (DECISIONS 13).
- N6 (latency) is reported, not counted (H9 §7 criterion 4), and comes with W5.
- Whatever the result, the report states both backbones' numbers, the site probe and
  the abstention curves.

