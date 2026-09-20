# Champion/challenger verdict

**dinov2_l14_reg wins**: neither backbone won a criterion, and anything else goes to the champion.

Written by `make eval` (`ms.eval.verdict`) from `n_table.jsonl`; do not edit. The
rules are H9 §7's, copied into `specs/005-results-table/spec.md` US-6 before any
number existed, with the owner's readings of 2026-09-19 and 2026-09-20. The rows read
are the quotable five-seed aggregates that are not superseded, at 224 px, on
CLS, at coverage 1.0: 306 of them, the newest
written 2026-09-20T06:15:53Z.

- Champion: `dinov2_l14_reg` · Challenger: `dinov3_l16`

## Criterion 1 — N2, cross-dataset macro-F1

A backbone wins when, with **each** of the three heads, its mean over the three frame
directions is at least 2 pp above the other's, **and** in at least two directions it is
at least 2 pp ahead with the two five-seed intervals not overlapping. The crop-level
rows report rust recall and are not counted.

| head | challenger mean | champion mean | challenger − champion | directions ≥ 2 pp and separated | challenger wins |
|---|---|---|---|---|---|
| linear | 0.5837 | 0.5987 | -1.51 pp | 0 of 3 | no |
| proto | 0.5314 | 0.5411 | -0.96 pp | 0 of 3 | no |
| mix | 0.5601 | 0.5832 | -2.31 pp | 0 of 3 | no |

Per direction (challenger − champion, and whether the intervals separate):

- linear, tanzania_v1 → makerere_v1: -0.24 pp, intervals overlap
- linear, tanzania_v1 → ibean_v1: -2.90 pp, intervals overlap
- linear, makerere_v1+ibean_v1 → tanzania_v1: -1.38 pp, intervals overlap
- proto, tanzania_v1 → makerere_v1: -1.97 pp, intervals overlap
- proto, tanzania_v1 → ibean_v1: +0.59 pp, intervals overlap
- proto, makerere_v1+ibean_v1 → tanzania_v1: -1.51 pp, intervals overlap
- mix, tanzania_v1 → makerere_v1: -1.52 pp, intervals overlap
- mix, tanzania_v1 → ibean_v1: -0.61 pp, intervals overlap
- mix, makerere_v1+ibean_v1 → tanzania_v1: -4.80 pp, intervals overlap

**Criterion 1**: challenger does not win; champion does not win.

## Criterion 2 — N3, the AUROC of the decision scores

A backbone wins when, with **each** head, **each** held-out set, **both** decision
scores and **each** training set, its AUROC is at least 0.02 higher (the owner's
reading of 2026-09-20). The second opinion and energy are reported, not counted.

| head | held-out | score | training set | challenger | champion | challenger − champion |
|---|---|---|---|---|---|---|
| linear | unknown_als | conf | makerere_v1 | 0.6134 | 0.7019 | -0.0885 |
| linear | unknown_als | conf | makerere_v1+ibean_v1 | 0.6253 | 0.7007 | -0.0754 |
| linear | unknown_als | conf | tanzania_v1 | 0.9171 | 0.9377 | -0.0206 |
| linear | unknown_als | knn | makerere_v1 | 0.6565 | 0.6227 | +0.0338 |
| linear | unknown_als | knn | makerere_v1+ibean_v1 | 0.6408 | 0.5927 | +0.0481 |
| linear | unknown_als | knn | tanzania_v1 | 0.6901 | 0.4444 | +0.2457 |
| linear | unknown_wm | conf | makerere_v1 | 0.8024 | 0.6203 | +0.1821 |
| linear | unknown_wm | conf | makerere_v1+ibean_v1 | 0.8654 | 0.6980 | +0.1674 |
| linear | unknown_wm | conf | tanzania_v1 | 0.9933 | 0.9965 | -0.0032 |
| linear | unknown_wm | knn | makerere_v1 | 0.9996 | 0.9972 | +0.0024 |
| linear | unknown_wm | knn | makerere_v1+ibean_v1 | 0.9996 | 0.9960 | +0.0036 |
| linear | unknown_wm | knn | tanzania_v1 | 0.9990 | 0.9816 | +0.0174 |
| mix | unknown_als | conf | makerere_v1 | 0.6155 | 0.6721 | -0.0566 |
| mix | unknown_als | conf | makerere_v1+ibean_v1 | 0.6351 | 0.6777 | -0.0426 |
| mix | unknown_als | conf | tanzania_v1 | 0.9423 | 0.9467 | -0.0044 |
| mix | unknown_als | knn | makerere_v1 | 0.6565 | 0.6227 | +0.0338 |
| mix | unknown_als | knn | makerere_v1+ibean_v1 | 0.6408 | 0.5927 | +0.0481 |
| mix | unknown_als | knn | tanzania_v1 | 0.6901 | 0.4444 | +0.2457 |
| mix | unknown_wm | conf | makerere_v1 | 0.8568 | 0.7097 | +0.1471 |
| mix | unknown_wm | conf | makerere_v1+ibean_v1 | 0.8883 | 0.8000 | +0.0883 |
| mix | unknown_wm | conf | tanzania_v1 | 0.9965 | 0.9967 | -0.0002 |
| mix | unknown_wm | knn | makerere_v1 | 0.9996 | 0.9972 | +0.0024 |
| mix | unknown_wm | knn | makerere_v1+ibean_v1 | 0.9996 | 0.9960 | +0.0036 |
| mix | unknown_wm | knn | tanzania_v1 | 0.9990 | 0.9816 | +0.0174 |
| proto | unknown_als | conf | makerere_v1 | 0.5943 | 0.6366 | -0.0423 |
| proto | unknown_als | conf | makerere_v1+ibean_v1 | 0.5912 | 0.5798 | +0.0114 |
| proto | unknown_als | conf | tanzania_v1 | 0.8642 | 0.7169 | +0.1473 |
| proto | unknown_als | knn | makerere_v1 | 0.6565 | 0.6227 | +0.0338 |
| proto | unknown_als | knn | makerere_v1+ibean_v1 | 0.6408 | 0.5927 | +0.0481 |
| proto | unknown_als | knn | tanzania_v1 | 0.6901 | 0.4444 | +0.2457 |
| proto | unknown_wm | conf | makerere_v1 | 0.9996 | 0.9990 | +0.0006 |
| proto | unknown_wm | conf | makerere_v1+ibean_v1 | 0.9996 | 0.9979 | +0.0017 |
| proto | unknown_wm | conf | tanzania_v1 | 0.9955 | 0.9790 | +0.0165 |
| proto | unknown_wm | knn | makerere_v1 | 0.9996 | 0.9972 | +0.0024 |
| proto | unknown_wm | knn | makerere_v1+ibean_v1 | 0.9996 | 0.9960 | +0.0036 |
| proto | unknown_wm | knn | tanzania_v1 | 0.9990 | 0.9816 | +0.0174 |

The challenger clears the 0.02 bar in 14 of 36 comparisons, the champion in 6 of 36; a win needs all of them.

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

