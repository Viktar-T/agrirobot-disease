"""The champion/challenger verdict (spec 005 US-6): `results/verdict.md`, written by code
from the N-table alone as part of `make eval`, and never edited by hand.

The rules are H9 §7's, copied into spec 005 US-6 verbatim, with the owner's readings of
2026-09-19 and 2026-09-20. They were written before any number existed, which is the whole
point of them: nothing here may be tuned to the table it reads.

- The rows read are the quotable five-seed aggregates that are not superseded, at 224 px, on
  CLS, at coverage 1.0.
- **Criterion 1 (N2)**: macro-F1 on the shared classes, in the three frame directions. A
  backbone wins it when, with each of the three heads, its mean over the directions is at
  least 2 pp above the other's, and in at least two directions it is at least 2 pp ahead with
  the two five-seed intervals not overlapping.
- **Criterion 2 (N3)**: a backbone wins it when, with each head, each held-out set, each
  decision score (confidence and kNN distance) and each training set, its AUROC is at least
  0.02 higher.
- The challenger wins when it wins a criterion and the champion wins neither. A split goes to
  N4, where the backbone whose probe balanced accuracy is at least 2 pp lower on both targets
  wins. Anything else, no wins at all included, goes to the champion.
- N6 is reported, not counted (H9 §7 criterion 4). The licence is a hard gate for anything
  that ships, whatever the numbers say.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ms.eval import RESULTS, unpaired

VERDICT_MD = RESULTS / "verdict.md"
#: a verdict is written beside the table it was computed from; `make eval` derives the path
#: from --n-table, so a run over another table can never write over this one
VERDICT_NAME = VERDICT_MD.name

CHAMPION = "dinov2_l14_reg"
CHALLENGER = "dinov3_l16"
BACKBONES = (CHAMPION, CHALLENGER)
#: the rows the verdict reads (US-6.1)
RES, TOKENS, COVERAGE = 224, "cls", 1.0
HEADS = ("linear", "proto", "mix")
#: H9 §7: 2 pp on N2, 0.02 AUROC on N3, and 2 pp on the N4 tie-breaker
MARGIN = 0.02
#: a float guard, because a row's value is rounded to 4 decimals
EPS = 1e-9
#: the two decision scores of spec 004; the second opinion and energy are not counted
DECISION_SCORES = ("auroc:conf", "auroc:knn")
HELD_OUT = ("unknown_als", "unknown_wm")
N4_TARGETS = ("balanced_accuracy:dataset", "balanced_accuracy:district")


class VerdictError(ValueError):
    """A verdict that may not be written; `reason` says why."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def beside(n_table: Path | str) -> Path:
    """The verdict of a table lives beside it. `make eval` uses this unless --verdict says
    otherwise, so that a run over a temporary table writes a temporary verdict."""
    return Path(n_table).parent / VERDICT_NAME


def _stem(joined: str) -> str:
    return "+".join(Path(p).stem for p in joined.split("+"))


def counted(directions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The N2 directions criterion 1 counts: the frame-level ones. A direction that names its
    own metrics reports something other than macro-F1 on the shared classes — the crop-level
    one reports rust recall — and is reported, not counted (US-6.3)."""
    return [("+".join(d["train"]), d["test"]) for d in directions if not d.get("metrics")]


def readable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """US-6.1: quotable five-seed aggregates, not superseded, at 224 px, CLS, coverage 1.0."""
    return [
        r
        for r in rows
        if r.get("seed") is None
        and not r.get("superseded")
        and r.get("quotable") is True
        and r.get("res") == RES
        and r.get("token_type") == TOKENS
        and r.get("coverage") == COVERAGE
    ]


def _n2(rows: list[dict[str, Any]], directions: list[tuple[str, str]]) -> dict:
    """{(head, direction): {backbone: (value, ci_low, ci_high)}} for macro-F1."""
    out: dict = {}
    for r in rows:
        if r["number"] != "N2" or r["metric"] != "macro_f1":
            continue
        where = (_stem(r["train_manifest"]), _stem(r["test_manifest"]))
        if where in directions:
            out.setdefault((r["head"], where), {})[r["backbone_id"]] = (
                r["value"],
                r["ci_low"],
                r["ci_high"],
            )
    return out


def _n3(rows: list[dict[str, Any]]) -> dict:
    """{(head, held-out set, score, training set): {backbone: value}}."""
    out: dict = {}
    for r in rows:
        if r["number"] != "N3" or r["metric"] not in DECISION_SCORES:
            continue
        key = (r["head"], r["classes"][0], r["metric"], _stem(r["train_manifest"]))
        out.setdefault(key, {})[r["backbone_id"]] = r["value"]
    return out


def _n4(rows: Iterable[dict[str, Any]]) -> dict:
    """{target: {backbone: value}} for the site probe, whatever its resolution and tokens."""
    out: dict = {}
    for r in rows:
        if r.get("number") == "N4" and r.get("metric") in N4_TARGETS and not r.get("superseded"):
            out.setdefault(r["metric"], {})[r["backbone_id"]] = r["value"]
    return out


def criterion_1(n2: dict, directions: list[tuple[str, str]], a: str, b: str) -> dict:
    """Does `a` win criterion 1 against `b`? US-6.3, with the reasons, so that verdict.md can
    show its working."""
    per_head, wins = {}, []
    for head in HEADS:
        have = [d for d in directions if n2.get((head, d), {}).keys() >= {a, b}]
        if not have:
            per_head[head] = {"complete": False}
            wins.append(False)
            continue
        means = {x: sum(n2[(head, d)][x][0] for d in have) / len(have) for x in (a, b)}
        ahead = means[a] - means[b]
        clear = []
        for d in have:
            va, vb = n2[(head, d)][a], n2[(head, d)][b]
            clear.append(
                {
                    "direction": d,
                    "gap": va[0] - vb[0],
                    "separated": va[1] is not None and vb[2] is not None and va[1] > vb[2],
                }
            )
        counted_ = sum(1 for c in clear if c["gap"] >= MARGIN - EPS and c["separated"])
        won = len(have) == len(directions) and ahead >= MARGIN - EPS and counted_ >= 2
        per_head[head] = {
            "complete": len(have) == len(directions),
            "mean_a": means[a],
            "mean_b": means[b],
            "ahead": ahead,
            "directions": clear,
            "clear": counted_,
            "won": won,
        }
        wins.append(won)
    return {"won": bool(wins) and all(wins), "heads": per_head}


def criterion_2(n3: dict, a: str, b: str) -> dict:
    """Does `a` win criterion 2 against `b`? US-6.4 with the owner's reading of 2026-09-20:
    every head, every held-out set, both decision scores, every training set."""
    comparisons, wins = [], []
    for key in sorted(n3):
        both = n3[key]
        if both.keys() < {a, b}:
            continue
        gap = both[a] - both[b]
        comparisons.append(
            {"key": key, "a": both[a], "b": both[b], "gap": gap, "cleared": gap >= MARGIN - EPS}
        )
        wins.append(gap >= MARGIN - EPS)
    heads = {c["key"][0] for c in comparisons}
    complete = heads >= set(HEADS) and {c["key"][1] for c in comparisons} >= set(HELD_OUT)
    return {
        "won": bool(wins) and all(wins) and complete,
        "complete": complete,
        "comparisons": comparisons,
        "cleared": sum(wins),
        "of": len(comparisons),
    }


def tie_break(n4: dict, a: str, b: str) -> dict:
    """US-6.6: a split goes to N4. The backbone whose probe balanced accuracy is at least
    2 pp lower on **both** targets wins it — a feature that gives the site away less."""
    per_target, wins = {}, []
    for target in N4_TARGETS:
        both = n4.get(target, {})
        if both.keys() < {a, b}:
            per_target[target] = None
            wins.append(False)
            continue
        lower = both[b] - both[a]
        per_target[target] = {"a": both[a], "b": both[b], "lower_by": lower}
        wins.append(lower >= MARGIN - EPS)
    return {"won": bool(wins) and all(wins), "targets": per_target}


def decide(rows: Iterable[dict[str, Any]], directions: list[dict[str, Any]]) -> dict:
    """The verdict, as a structure `render` turns into verdict.md. `unpaired` is checked by
    the caller (US-6.1)."""
    rows = list(rows)
    read = readable(rows)
    if not read:
        raise VerdictError(
            "no_rows",
            "no row to read: the verdict reads the quotable five-seed aggregates that are not "
            f"superseded, at {RES} px, on {TOKENS.upper()}, at coverage {COVERAGE:.1f}, and the "
            "table has none. A verdict with no comparison behind it is not a verdict",
        )
    counted_dirs = counted(directions)
    n2, n3, n4 = _n2(read, counted_dirs), _n3(read), _n4(rows)
    out: dict[str, Any] = {
        "champion": CHAMPION,
        "challenger": CHALLENGER,
        "directions": counted_dirs,
        "rows_read": len(read),
        # the table's own latest timestamp, not the clock: verdict.md then changes when the
        # numbers do and not when it is written again (US-6.1, "by code, never by hand")
        "as_of": max((r["ts"] for r in read), default=None),
        "c1": {
            CHALLENGER: criterion_1(n2, counted_dirs, CHALLENGER, CHAMPION),
            CHAMPION: criterion_1(n2, counted_dirs, CHAMPION, CHALLENGER),
        },
        "c2": {
            CHALLENGER: criterion_2(n3, CHALLENGER, CHAMPION),
            CHAMPION: criterion_2(n3, CHAMPION, CHALLENGER),
        },
        "n4": n4,
    }
    challenger_wins = out["c1"][CHALLENGER]["won"] or out["c2"][CHALLENGER]["won"]
    champion_wins = out["c1"][CHAMPION]["won"] or out["c2"][CHAMPION]["won"]
    if challenger_wins and not champion_wins:
        winner = CHALLENGER
        why = "the challenger won a criterion and the champion neither"
    elif champion_wins and not challenger_wins:
        winner = CHAMPION
        why = "the champion won a criterion and the challenger neither"
    elif challenger_wins and champion_wins:
        out["tie_break"] = {
            CHALLENGER: tie_break(n4, CHALLENGER, CHAMPION),
            CHAMPION: tie_break(n4, CHAMPION, CHALLENGER),
        }
        split = out["tie_break"][CHALLENGER]["won"] and not out["tie_break"][CHAMPION]["won"]
        winner = CHALLENGER if split else CHAMPION
        why = (
            "each backbone won a criterion; N4 broke the split"
            if split
            else "each backbone won a criterion; N4 did not break the split, and a tie goes "
            "to the champion"
        )
    else:
        winner = CHAMPION
        why = "neither backbone won a criterion, and anything else goes to the champion"
    out["winner"], out["why"] = winner, why
    return out


# --- the rendered verdict -------------------------------------------------------------------


def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def render(v: dict) -> str:
    """verdict.md. Every number in it is in the N-table; nothing is computed here that
    `decide` did not compute from a row."""
    champ, chall = v["champion"], v["challenger"]
    out = [
        "# Champion/challenger verdict",
        "",
        f"**{v['winner']} wins**: {v['why']}.",
        "",
        "Written by `make eval` (`ms.eval.verdict`) from `n_table.jsonl`; do not edit. The",
        "rules are H9 §7's, copied into `specs/005-results-table/spec.md` US-6 before any",
        "number existed, with the owner's readings of 2026-09-19 and 2026-09-20. The rows read",
        f"are the quotable five-seed aggregates that are not superseded, at {RES} px, on",
        f"{TOKENS.upper()}, at coverage {COVERAGE:.1f}: {v['rows_read']} of them, the newest",
        f"written {v['as_of']}.",
        "",
        f"- Champion: `{champ}` · Challenger: `{chall}`",
        "",
        "## Criterion 1 — N2, cross-dataset macro-F1",
        "",
        "A backbone wins when, with **each** of the three heads, its mean over the three frame",
        "directions is at least 2 pp above the other's, **and** in at least two directions it is",
        "at least 2 pp ahead with the two five-seed intervals not overlapping. The crop-level",
        "rows report rust recall and are not counted.",
        "",
        "| head | challenger mean | champion mean | challenger − champion "
        "| directions ≥ 2 pp and separated | challenger wins |",
        "|---|---|---|---|---|---|",
    ]
    for head in HEADS:
        a = v["c1"][chall]["heads"].get(head, {})
        if not a.get("complete"):
            out.append(f"| {head} | — | — | — | — | no (a direction is missing) |")
            continue
        out.append(
            f"| {head} | {a['mean_a']:.4f} | {a['mean_b']:.4f} | {_pp(a['ahead'])} "
            f"| {a['clear']} of {len(a['directions'])} | {'yes' if a['won'] else 'no'} |"
        )
    out += ["", "Per direction (challenger − champion, and whether the intervals separate):", ""]
    for head in HEADS:
        a = v["c1"][chall]["heads"].get(head, {})
        for d in a.get("directions", []):
            out.append(
                f"- {head}, {d['direction'][0]} → {d['direction'][1]}: {_pp(d['gap'])}, "
                f"intervals {'do not overlap' if d['separated'] else 'overlap'}"
            )
    out += [
        "",
        f"**Criterion 1**: challenger {'wins' if v['c1'][chall]['won'] else 'does not win'}; "
        f"champion {'wins' if v['c1'][champ]['won'] else 'does not win'}.",
        "",
        "## Criterion 2 — N3, the AUROC of the decision scores",
        "",
        "A backbone wins when, with **each** head, **each** held-out set, **both** decision",
        "scores and **each** training set, its AUROC is at least 0.02 higher (the owner's",
        "reading of 2026-09-20). The second opinion and energy are reported, not counted.",
        "",
        "| head | held-out | score | training set | challenger | champion "
        "| challenger − champion |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in v["c2"][chall]["comparisons"]:
        head, held, score, train = c["key"]
        out.append(
            f"| {head} | {held} | {score.replace('auroc:', '')} | {train} | {c['a']:.4f} "
            f"| {c['b']:.4f} | {c['gap']:+.4f} |"
        )
    out += [
        "",
        f"The challenger clears the 0.02 bar in {v['c2'][chall]['cleared']} of "
        f"{v['c2'][chall]['of']} comparisons, the champion in "
        f"{v['c2'][champ]['cleared']} of {v['c2'][champ]['of']}; a win needs all of them.",
        "",
        f"**Criterion 2**: challenger {'wins' if v['c2'][chall]['won'] else 'does not win'}; "
        f"champion {'wins' if v['c2'][champ]['won'] else 'does not win'}.",
        "",
    ]
    if "tie_break" in v:
        out += [
            "## The split — N4, the site-prediction probe",
            "",
            "Each backbone won a criterion, so N4 decides: the backbone whose probe balanced",
            "accuracy is at least 2 pp **lower** on both targets wins.",
            "",
        ]
        for who in (chall, champ):
            for target, got in v["tie_break"][who]["targets"].items():
                if got:
                    out.append(
                        f"- {who} on {target}: {got['a']:.4f} against {got['b']:.4f} "
                        f"({_pp(got['lower_by'])} lower)"
                    )
        out.append("")
    out += [
        "## Outcome",
        "",
        f"- **{v['winner']} wins**: {v['why']}.",
        "- A winning challenger **joins** the model set behind the service interface, and never",
        "  evicts the champion in the same season (H8 D-7).",
        "- **The licence is a hard gate** for anything that ships, whatever the numbers say:",
        "  DINOv3 is research-only until H6 C5 is signed (DECISIONS 13).",
        "- N6 (latency) is reported, not counted (H9 §7 criterion 4), and comes with W5.",
        "- Whatever the result, the report states both backbones' numbers, the site probe and",
        "  the abstention curves.",
        "",
    ]
    return "\n".join(out) + "\n"


def write_verdict(
    rows: Iterable[dict[str, Any]],
    directions: list[dict[str, Any]],
    path: Path | str = VERDICT_MD,
) -> tuple[bool, dict]:
    """Write verdict.md when its text changes; (written, the verdict)."""
    v = decide(rows, directions)
    text = render(v)
    p = Path(path)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return False, v
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return True, v


def main(argv: list[str] | None = None) -> int:
    """`python -m ms.eval.verdict` prints the verdict as JSON, for a look without writing."""
    import argparse

    from ms.eval import N_TABLE, read_rows
    from ms.eval.run import EVAL_CONFIG, load_directions

    p = argparse.ArgumentParser(
        prog="python -m ms.eval.verdict", description=__doc__.split("\n")[0]
    )
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--config", type=Path, default=EVAL_CONFIG)
    args = p.parse_args(argv)
    rows = list(read_rows(args.n_table))
    missing = unpaired(rows)
    if missing:
        print(f"unpaired: {missing}", flush=True)
        return 2
    try:
        v = decide(rows, load_directions(args.config))
    except VerdictError as exc:
        print(f"{exc.reason}: {exc}", flush=True)
        return 2
    print(json.dumps(v, indent=1, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
