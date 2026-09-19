"""Mark the N-table rows of replaced runs as superseded (spec 005 US-8).

    python -m ms.eval.supersede --config-sha256 <sha256> [<sha256> ...] --reason "<why>"
        [--heads-root data/heads] [--n-table model_service/results/n_table.jsonl]
        [--md model_service/results/n_table.md]

When the owner replaces a head recipe (a changed configs/heads/<head>.yaml), the runs trained
with the old one are not quoted any more, but their rows stay in the table, marked. A run is
replaced when its run.json names one of the given head-config sha256s, or when it mixes such a
run. Every row that names a replaced run gets `superseded = <reason>`; an aggregate names its
five runs. The mark is the only change a row ever gets: every other field stays as it was, a
row already marked keeps its mark, and a row written before the field existed gets null.
Running it again changes nothing.

Exit codes: 0; 2 when no run matches or a row is invalid (nothing written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ms.data.manifests import repo_relative
from ms.eval import FIELDS, N_TABLE, N_TABLE_MD, read_rows, validate_row, write_md
from ms.heads import HEADS_ROOT, list_runs


def replaced_runs(heads_root: Path | str, config_sha256s: set[str]) -> set[str]:
    """The run ids trained with one of the head configs, and the mixes of such runs."""
    metas = [
        json.loads((f / "run.json").read_text(encoding="utf-8")) for f in list_runs(heads_root)
    ]
    runs = {m["run_id"] for m in metas if m["head_config"]["sha256"] in config_sha256s}
    for m in metas:
        if {c["run_id"] for c in m.get("components") or []} & runs:
            runs.add(m["run_id"])
    return runs


def mark(
    rows: list[dict[str, Any]], runs: set[str], reason: str
) -> tuple[list[dict[str, Any]], int]:
    """The rows in FIELDS order, those that name a replaced run marked; how many were marked
    now."""
    out, marked = [], 0
    for r in rows:
        row = {k: r.get(k) for k in FIELDS}
        if not row["superseded"] and set(str(row["run_id"]).split("+")) & runs:
            row["superseded"] = reason
            marked += 1
        out.append(row)
    return out, marked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ms.eval.supersede", description=__doc__.split("\n")[0]
    )
    p.add_argument("--config-sha256", nargs="+", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    p.add_argument("--n-table", type=Path, default=N_TABLE)
    p.add_argument("--md", type=Path, default=N_TABLE_MD)
    args = p.parse_args(argv)
    if not args.reason.strip():
        print("bad_value: --reason is empty", flush=True)
        return 2
    runs = replaced_runs(args.heads_root, set(args.config_sha256))
    if not runs:
        print(f"no_run: no run under {args.heads_root} has these head configs", flush=True)
        return 2
    before = list(read_rows(args.n_table))
    rows, marked = mark(before, runs, args.reason.strip())
    for r in rows:
        problems = validate_row(r)
        if problems:
            print(f"{problems[0]} ({r['run_id']}); nothing written", flush=True)
            return 2
    if rows != before:
        tmp = args.n_table.with_name(args.n_table.name + ".tmp")
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, args.n_table)
    rendered = write_md(args.md, args.n_table)
    print(
        f"{repo_relative(args.n_table)}: {len(runs)} run(s) replaced, {marked} row(s) marked "
        f"superseded; {repo_relative(args.md)} {'rewritten' if rendered else 'unchanged'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
