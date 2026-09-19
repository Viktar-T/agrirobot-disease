"""Spec 001 US-6 - crops manifests: the acceptance tests (planned 2026-09-19, W2 Wed).

`python -m ms.data.manifests crops <parent.jsonl> --margin 0.10` cuts every box of a parent
manifest into data/derived/<parent>_crops/ and writes <parent>_crops_v<N>.jsonl (FR-012).
The raw trees come from test_manifests.py: the Makerere layout with one box per annotated
image, and pictures whose pixels say where a crop was cut.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import ExifTags, Image
from test_manifests import (
    SPLITS,
    built,
    make_makerere,
    meta,
    rows,
    run,
    sha256,
    write_rows,
)

BOX = {"x": 10, "y": 12, "w": 30, "h": 18}  # every annotated picture of make_makerere
CUT = {"x": 7, "y": 10, "w": 36, "h": 22, "frame": "stored"}  # BOX + 10 % a side, outward


def crops(capsys, parent: Path, raw: Path, out: Path, crop_root: Path, *extra) -> tuple[int, str]:
    return run(
        capsys,
        "crops",
        parent,
        "--raw-root",
        raw,
        "--out",
        out,
        "--crop-root",
        crop_root,
        *extra,
    )


def made(capsys, tmp_path: Path, *extra) -> tuple[Path, Path, Path]:
    """(raw root, parent manifest, crops manifest) for the synthetic Makerere tree."""
    raw = tmp_path / "raw"
    if not (raw / "makerere").exists():
        make_makerere(raw)
    parent = built(capsys, raw, tmp_path / "out", "makerere")
    code, text = crops(capsys, parent, raw, tmp_path / "out", tmp_path / "derived", *extra)
    assert code == 0, text
    return raw, parent, tmp_path / "out" / "makerere_crops_v1.jsonl"


def annotated(raw: Path) -> list[Path]:
    return sorted((raw / "makerere" / "extracted").rglob("*.xml"))


def turned_picture(path: Path) -> None:
    """A 96 x 72 stored picture with EXIF orientation 6 (upright it is 72 x 96): red, with
    STORED_ONLY painted blue in stored pixels and UPRIGHT_ONLY painted green in the upright
    frame. make_makerere's own box, BOX, fits both frames."""
    stored = Image.new("RGB", (96, 72), (220, 20, 20))
    x, y, w, h = (STORED_ONLY[k] for k in "xywh")
    stored.paste((20, 20, 220), (x, y, x + w, y + h))
    upright = stored.transpose(Image.Transpose.ROTATE_270)  # what orientation 6 asks for
    x, y, w, h = (UPRIGHT_ONLY[k] for k in "xywh")
    upright.paste((20, 200, 20), (x, y, x + w, y + h))
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    upright.transpose(Image.Transpose.ROTATE_90).save(path, "JPEG", quality=95, exif=exif)


STORED_ONLY = {"x": 60, "y": 10, "w": 30, "h": 20}  # x + w = 90: past the upright width, 72
UPRIGHT_ONLY = {"x": 10, "y": 70, "w": 30, "h": 20}  # y + h = 90: past the stored height, 72


# --- US-6.1: one row per box ----------------------------------------------------------------------


def test_us6_1_one_row_per_box_inheriting_the_parents_keys_and_split(tmp_path, capsys):
    raw, parent, manifest = made(capsys, tmp_path)
    by_id = {r["image_id"]: r for r in rows(parent)}
    rs = rows(manifest)
    assert len(rs) == sum(len(r["boxes"] or []) for r in by_id.values()) == len(annotated(raw))
    derived = tmp_path / "derived"
    for r in rs:
        p = by_id[r["parent_image_id"]]
        assert r["box_index"] == 0 and r["crop"] == CUT
        assert r["path"] == f"makerere_crops/{p['image_id']}_b0.jpg"
        assert r["sha256"] == sha256(derived / r["path"])  # its own hash, of the written crop
        assert r["image_id"] == f"makerere_crops_{r['sha256'][:16]}"
        assert (r["dataset"], r["format"], r["width"], r["height"]) == (
            "makerere_crops",
            "jpeg",
            CUT["w"],
            CUT["h"],
        )
        assert r["class_raw"] == p["boxes"][0]["class_raw"]
        assert r["class_km2"] == {"Bean Rust": "rust", "ALS": "unknown_als"}[r["class_raw"]]
        for field in ("group_keys", "split_group", "split", "split_rule", "licence", "attribution"):
            assert r[field] == p[field], field
    m = meta(manifest)
    assert (m["kind"], m["margin"], m["manifest"], m["version"]) == (
        "crops",
        0.1,
        "makerere_crops",
        1,
    )
    assert m["parent"] == {
        "manifest": "makerere_v1.jsonl",
        "sha256": sha256(parent),
        "frozen": False,
    }
    assert m["counts"]["rows"] == len(rs) and m["excluded"] == []
    code, text = run(capsys, "validate", manifest)  # the files are found under the sidecar's root
    assert code == 0, text


def test_us6_1_crops_are_rebuilt_to_the_same_bytes_and_nothing_is_rewritten(tmp_path, capsys):
    _, _, manifest = made(capsys, tmp_path)
    files = sorted((tmp_path / "derived").rglob("*"))
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files if p.is_file()}
    text = manifest.read_bytes()
    _, _, again = made(capsys, tmp_path)
    assert again.read_bytes() == text
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files if p.is_file()}
    assert after == before


# --- US-6.2: validation ---------------------------------------------------------------------------


def test_us6_2_validate_names_a_missing_parent_and_crops_that_straddle(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    xml = next(p for p in annotated(raw) if "bean_rust" in p.parts[-2])
    text = xml.read_text(encoding="utf-8")
    second = (
        '<object><name name="name">Bean Rust</name><bndbox><xmin name="xmin">50</xmin>'
        '<ymin name="ymin">30</ymin><xmax name="xmax">80</xmax><ymax name="ymax">60</ymax>'
        "</bndbox></object>"
    )
    xml.write_text(text.replace("</annotation>", second + "</annotation>"), encoding="utf-8")
    _, _, manifest = made(capsys, tmp_path)
    rs = rows(manifest)
    (twice,) = [p for p, n in Counter(r["parent_image_id"] for r in rs).items() if n == 2]
    pair = [i for i, r in enumerate(rs) if r["parent_image_id"] == twice]
    assert {rs[i]["box_index"] for i in pair} == {0, 1}

    broken = json.loads(json.dumps(rs))
    other = next(s for s in SPLITS if s != broken[pair[1]]["split"])
    broken[pair[1]]["split"] = other
    write_rows(manifest, broken)
    code, text = run(capsys, "validate", manifest)
    assert code == 2 and "group_straddles_split:" in text

    broken = json.loads(json.dumps(rs))
    broken[0]["parent_image_id"] = "makerere_0000000000000000"
    write_rows(manifest, broken)
    code, text = run(capsys, "validate", manifest)
    assert code == 2 and "parent_missing:" in text


# --- US-6.4: the frame of the boxes -------------------------------------------------------------


def add_boxes(xml: Path, *boxes: dict[str, int]) -> None:
    extra = "".join(
        f'<object><name name="name">Bean Rust</name><bndbox><xmin name="xmin">{b["x"]}</xmin>'
        f'<ymin name="ymin">{b["y"]}</ymin><xmax name="xmax">{b["x"] + b["w"]}</xmax>'
        f'<ymax name="ymax">{b["y"] + b["h"]}</ymax></bndbox></object>'
        for b in boxes
    )
    text = xml.read_text(encoding="utf-8")
    xml.write_text(text.replace("</annotation>", extra + "</annotation>"), encoding="utf-8")


def test_us6_4_on_a_turned_image_each_box_is_cut_in_the_frame_that_holds_it(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    xml = next(p for p in annotated(raw) if "bean_rust" in p.parts[-2])
    turned_picture(xml.with_suffix(".jpg"))
    add_boxes(xml, STORED_ONLY, UPRIGHT_ONLY)  # boxes 1 and 2; box 0 is BOX
    _, parent, manifest = made(capsys, tmp_path)
    (p,) = [r for r in rows(parent) if r["path"].endswith(xml.with_suffix(".jpg").name)]
    assert (p["width"], p["height"]) == (72, 96)  # the parent row is orientation-applied
    mine = {r["box_index"]: r for r in rows(manifest) if r["parent_image_id"] == p["image_id"]}
    assert set(mine) == {1, 2}
    # box 1 lies in the stored pixels: cut there (36 x 24 with its margin), then turned upright
    assert mine[1]["crop"] == {"x": 57, "y": 8, "w": 36, "h": 24, "frame": "stored"}
    assert (mine[1]["width"], mine[1]["height"]) == (24, 36)
    # box 2 lies in the upright image: cut there, and it is upright already
    assert mine[2]["crop"] == {"x": 7, "y": 68, "w": 36, "h": 24, "frame": "upright"}
    assert (mine[2]["width"], mine[2]["height"]) == (36, 24)
    for k, colour in ((1, "blue"), (2, "green")):
        with Image.open(tmp_path / "derived" / mine[k]["path"]) as crop:
            rgb = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
        assert max(range(3), key=lambda c: rgb[c]) == {"green": 1, "blue": 2}[colour], (k, rgb)
    # box 0 fits both frames: which one the annotator saw cannot be told, so it gets no row
    assert {"path": f"{p['path']}#box0", "reason": "box_frame_unknown"} in meta(manifest)[
        "excluded"
    ]


# --- US-6.5: boxes at the edge --------------------------------------------------------------------


def test_us6_5_boxes_are_clipped_and_a_box_outside_the_picture_gets_no_row(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    xml = next(p for p in annotated(raw) if "bean_rust" in p.parts[-2])
    text = xml.read_text(encoding="utf-8")
    extra = "".join(
        f'<object><name name="name">Bean Rust</name><bndbox><xmin name="xmin">{x0}</xmin>'
        f'<ymin name="ymin">{y0}</ymin><xmax name="xmax">{x1}</xmax>'
        f'<ymax name="ymax">{y1}</ymax></bndbox></object>'
        for x0, y0, x1, y1 in ((80, 60, 120, 90), (200, 200, 220, 220))
    )
    xml.write_text(text.replace("</annotation>", extra + "</annotation>"), encoding="utf-8")
    _, parent, manifest = made(capsys, tmp_path)
    (p,) = [r for r in rows(parent) if r["path"].endswith(xml.with_suffix(".jpg").name)]
    mine = {r["box_index"]: r for r in rows(manifest) if r["parent_image_id"] == p["image_id"]}
    assert set(mine) == {0, 1}
    assert mine[1]["crop"] == {"x": 76, "y": 57, "w": 20, "h": 15, "frame": "stored"}  # clipped
    assert meta(manifest)["excluded"] == [
        {"path": f"{p['path']}#box2", "reason": "box_outside_image"}
    ]
