# Model card — {{model_version}}

<!--
The template of H8 §6.8, filled in by `python -m ms.registry card` (`make card`). The prose
below is the same for every piece 4 v0 model; everything between the markers is rendered
from the artefacts — the run, its fit, the manifests and `results/n_table.jsonl` — so a card
cannot drift from the numbers it quotes. Do not edit a rendered card by hand: `ms.registry
register` refuses a card that differs from what the artefacts render now.
-->

**Status**: {{status}} · **Generated**: {{generated_at}} · **Rendered from**: `{{git_sha}}`
**Run**: `{{run_id}}` · **Served by**: `model_service/configs/service.yaml`

*A card cannot carry the sha of the commit that will contain it, so the one above says
which repository rendered it and nothing more. What pins this model is the set of
hashes under "Reproducing this model".*

## Intended use

Organ-level triage of bean-leaf imagery: one photograph of a bean plant in, one of
{{classes_sentence}} out, or *abstain* when the model should not be trusted on it. It is
built for the central-node machine behind the plot map and the future decision layer (H8
§6.9), reached over the interface of H8 §6.2 — never on the robot, and never inside a control
loop.

**This model decides between {{n_classes}} classes**, the ones its training manifests hold.
The label space of the task is `healthy`, `rust` and `anthracnose` (H8 §6.3); a class absent
from the training data is a class this model cannot return, and photographs of it land on
whichever of its own classes they look most like — or on `abstain`.

It is a **probe of the pipeline and of the data, not a model choice** (H8 §6.1): frozen
foundation features with a small head, trained on open smartphone photographs of African bean
fields, kept deliberately simple so that what the numbers show is the data and the protocol.

### Not validated for, and not to be used for

- **Polish imagery, robot views, under-panel light, pods, whole plots.** Nothing in the
  training or evaluation data was taken in Poland, from a robot, or under an agrivoltaic
  panel. H8 §6.7's guard-rail, written before any number existed, is that the drop from one
  African set to another is a *lower bound* on the drop to Poland — a floor under the cost,
  never an estimate of it, and this card does not measure that drop at all.
- **Anything outside its {{n_classes}} trained classes.** Angular leaf spot and white mould
  were held out on purpose and are never predicted; every other disease, pest, deficiency or
  damage is outside the label space altogether, and so is any class of the task this model
  was not trained on.
- **Severity, counting or localisation.** There are no lesion boxes, no masks and no severity
  score in v0 (H8 §6.1). A response's `localisation` is always null.
- **Nutrient deficiencies.** No image data for them exist (E1 §3).
- **Any decision taken on one frame alone.** The map aggregates; a single answer, abstention
  included, is evidence and not a verdict.

## The model

<!-- model:begin -->
{{model_block}}
<!-- model:end -->

## Training data

Each row of each manifest carries its own licence and attribution line; the manifests are
frozen by sha256 before any head is trained (spec 001 US-5), and those hashes are what the
`model_version` string above is built from.

<!-- data:begin -->
{{data_block}}
<!-- data:end -->

## Evaluation

Every number is read from `model_service/results/n_table.jsonl` with the split rule printed
beside it (H8 D-9); nothing here is computed by the card. N1 to N3 are five-seed means with a
95 % Student *t* interval — an interval of zero width is a score every seed agreed on, which
the two distance scores are by construction, because their bank does not depend on the seed.
N4 and N6 are one measurement with an interval of their own kind, as their rows say.

Unless a row says otherwise its coverage is 1.0: **abstention off**, every frame decided. What
the operating points do to the same numbers is the selective-risk table at the end of this
section, and `selective_risk` at coverage 1.0 is simply the error rate over every frame.

<!-- evaluation:begin -->
{{evaluation_block}}
<!-- evaluation:end -->

## Operating point

Abstention is two gates (H8 §6.6): the head's confidence, whose threshold moves with the
declared coverage, and the distance to the training features, whose threshold sits at TPR
95 % whatever the coverage. A request may ask for any fitted coverage; one that was not
fitted is refused, because a service does not interpolate an operating point it never
measured.

<!-- operating-point:begin -->
{{operating_point_block}}
<!-- operating-point:end -->

## Known failure modes

<!-- failure-modes:begin -->
{{failure_modes_block}}
<!-- failure-modes:end -->

Beyond what the numbers show:

- **The acquisition site is in the features.** A probe reads the source dataset from the
  frozen features at far above chance, and the district within Makerere too (N4). A head can
  therefore learn *where* instead of *what*. That is why the splits that decide a number are
  blocked — Makerere by district, Tanzania by capture date — and why the cross-dataset number
  is the one to read. iBean carries no metadata to block on, so it is test-only and
  `unblocked:random_by_phash_group`, and no number of its own is ever quoted.
- **Calibration is fitted in domain, and shift is what breaks it first.** Over the whole
  campaign the expected calibration error runs about 0.005 in domain against 0.367 out of it
  (DECISIONS 103). This model's own `ece` rows are in the table above: read those, not the
  median — a small one says this transfer direction was kind, not that calibration holds.
  The abstention gates, not the probabilities, are what the operating point rests on.
- **A class-trip confound in Makerere.** Its healthy photographs all come from one April trip
  and most of its rust from May, so an in-domain number on that set partly measures the trip.
- **Compression, blur, rain, night.** Nothing in the evaluation isolates them.

## Licences and terms

<!-- licences:begin -->
{{licences_block}}
<!-- licences:end -->

## Reproducing this model

<!-- reproduce:begin -->
{{reproduce_block}}
<!-- reproduce:end -->

## Contact and provenance

Piece 4 of the AgriRobot bean-disease system (H8 §6), owner the ML role. The plan is
`87_H8_current-plan-demo1-end-to-end-without-a-robot.md` §6; the execution is
`docs/piece4-work-plan.md`; the dated choices are `model_service/DECISIONS.md`; the contracts
are `model_service/specs/`. The champion/challenger verdict this model sits under is
`model_service/results/verdict.md`, written by code from the table alone.
