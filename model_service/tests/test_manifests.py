"""Spec 001 - dataset manifests: the acceptance tests (model_service/specs/001-manifests/spec.md).

Written before the implementation (W1, spec-driven). Every test that runs the builder,
the validator, `freeze`, `overlap` or the loader fails until `ms.data.manifests` exists
(W2), each with the same one-line message. The tests of the class map and the recipes
are green from the start.

No real data. Each test lays out a small raw tree in tmp_path the way data/raw/<member>/
looks on the project machine (W1 data note: folder layouts, Makerere's XML, EXIF dates),
with tiny generated JPEGs, and builds it with the real recipes (configs/manifests/) and
the real class map (configs/class_map_v1.yaml). Only US-1.1 reads the committed fixture
tests/fixtures/ibean_30/ (the W1 "Fixture" task): it builds the fixture, and checks the
fixture's own manifest (ibean_v1.jsonl) against that build.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import ExifTags, Image, ImageDraw

TESTS = Path(__file__).resolve().parent
CONFIGS = TESTS.parent / "configs"
FIXTURE = TESTS / "fixtures" / "ibean_30"

# FR-003, in the order the rows are written
FIELDS = [
    "manifest_version",
    "image_id",
    "dataset",
    "source_record",
    "source_version",
    "path",
    "dup_paths",
    "sha256",
    "phash",
    "width",
    "height",
    "format",
    "class_raw",
    "class_km2",
    "boxes",
    "group_keys",
    "split_group",
    "split",
    "split_rule",
    "licence",
    "attribution",
    "notes",
]
GROUP_KEYS = [
    "district",
    "subcounty",
    "date",
    "region",
    "session",
    "variety",
    "plant_age",
    "phash_group",
]
TRAINED = ("healthy", "rust", "anthracnose")
SPLITS = ("train", "val", "test")
ROLES = ("train_eval", "test_only", "holdout_unknown")
RULE = re.compile(
    r"blocked:(district|subcounty|date|region|session|variety|plant_age)"
    r"|unblocked:random_by_phash_group|test_only|holdout_unknown"
)
UNBLOCKED = "unblocked:random_by_phash_group"
# FR-009
VALIDATE_REASONS = (
    "missing_file",
    "sha256_mismatch",
    "duplicate_image_id",
    "rows_not_sorted",
    "missing_field",
    "bad_value",
    "bad_split_rule",
    "group_straddles_split",
    "class_missing_from_split",
    "unknown_in_train",
    "unlabelled_in_train",
    "licence_unknown",
    "frozen_manifest_modified",
)


# --- the module under test ------------------------------------------------------------------


def _manifests():
    """ms.data.manifests; until it exists every test that needs it fails with one message."""
    try:
        return importlib.import_module("ms.data.manifests")
    except ModuleNotFoundError as exc:
        if exc.name != "ms.data.manifests":
            raise
    pytest.fail("ms.data.manifests does not exist yet (spec 001, W2)", pytrace=False)


def run(capsys, *argv) -> tuple[int, str]:
    """python -m ms.data.manifests <argv>, in process: (exit code, everything printed)."""
    module = _manifests()
    try:
        code = module.main([str(a) for a in argv])
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def build(capsys, raw: Path, out: Path, dataset: str, *extra) -> tuple[int, str]:
    return run(capsys, "build", "--dataset", dataset, "--raw-root", raw, "--out", out, *extra)


def built(capsys, raw: Path, out: Path, dataset: str, *extra) -> Path:
    """Build and expect success; the manifest's path."""
    code, text = build(capsys, raw, out, dataset, *extra)
    assert code == 0, text
    return out / f"{dataset}_v1.jsonl"


def assert_valid(capsys, manifest: Path, raw: Path) -> None:
    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 0, text


def rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]


def write_rows(manifest: Path, items: list[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
    manifest.write_text(text, encoding="utf-8", newline="\n")


def meta(manifest: Path) -> dict:
    return json.loads(manifest.with_name(manifest.stem + ".meta.json").read_text(encoding="utf-8"))


def by_path(items: list[dict]) -> dict[str, dict]:
    return {r["path"]: r for r in items}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def excluded(manifest: Path) -> dict[str, str]:
    return {e["path"]: e["reason"] for e in meta(manifest)["excluded"]}


# --- raw trees: pictures ------------------------------------------------------------------------

# Orthonormal DCT-II basis for 32 x 32, the size pHash works at.
_K, _N = np.arange(32)[:, None], np.arange(32)[None, :]
_DCT = np.sqrt(2 / 32) * np.cos(np.pi * (2 * _N + 1) * _K / 64)
_DCT[0] /= np.sqrt(2)

# A source picture whose q70 re-encode keeps every pHash bit, under every usual resampling.
NEAR_DUP_SEED = 9001


def picture(seed: int, size: tuple[int, int] = (96, 72)) -> Image.Image:
    """A low-frequency picture set by random signs on the 8 x 8 DCT block (magnitudes 60-110):
    different seeds lie >= 16 of 64 pHash bits apart, a JPEG re-encode a few bits at most."""
    rng = np.random.default_rng(seed)
    coef = np.zeros((32, 32))
    coef[:8, :8] = rng.choice([-1.0, 1.0], size=(8, 8)) * rng.uniform(60, 110, size=(8, 8))
    coef[0, 0] = 128 * 32
    grey = np.clip(_DCT.T @ coef @ _DCT, 0, 255).astype(np.uint8)
    return Image.fromarray(grey, "L").resize(size, Image.Resampling.BICUBIC).convert("RGB")


def jpeg(
    path: Path,
    seed: int,
    *,
    size: tuple[int, int] = (96, 72),
    taken: str | None = None,
    modified: str | None = None,
    orientation: int | None = None,
) -> Path:
    """A small JPEG of picture(seed). `taken` is EXIF DateTimeOriginal ("YYYY:MM:DD HH:MM:SS");
    `modified` is IFD0 DateTime, the file-change time, which is never the capture date."""
    exif = Image.Exif()
    if taken:
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = taken
    if modified:
        exif[ExifTags.Base.DateTime] = modified
    if orientation:
        exif[ExifTags.Base.Orientation] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"exif": exif} if (taken or modified or orientation) else {}
    picture(seed, size).save(path, "JPEG", quality=90, **extra)
    return path


def copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def download_json(
    raw: Path,
    member: str,
    licence: str,
    *,
    doi: str | None = None,
    version: str | None = "1",
    published: str = "2023-08-26",
) -> Path:
    """The provenance ms.data.download writes (the fields a manifest reads)."""
    folder = raw / member
    folder.mkdir(parents=True, exist_ok=True)
    doc = {
        "download_json_version": 1,
        "dataset": member,
        "title": f"{member} (test record)",
        "source": {
            "kind": "zenodo",
            "record_id": member,
            "record_url": f"https://example.org/records/{member}",
            "doi": doi,
            "version": version,
            "published": published,
            "publisher": "Example",
        },
        "licence": {"id": licence, "as_stated": licence},
        "attribution": f"Example, A. (2023). {member} [Data set]. https://example.org/{member}",
        "complete": True,
        "files": [],
    }
    (folder / "DOWNLOAD.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return folder


def attribution(raw: Path, member: str) -> str:
    return json.loads((raw / member / "DOWNLOAD.json").read_text(encoding="utf-8"))["attribution"]


# --- raw trees: the datasets --------------------------------------------------------------------

IBEAN_CLASSES = ("healthy", "angular_leaf_spot", "bean_rust")
IBEAN_TAG = {"train": "train", "validation": "val", "test": "test"}


def make_ibean(raw: Path, *, per_class: int = 10, licence: str = "MIT") -> Path:
    """data/raw/ibean: extracted/<split>/<class>/<class>_<tag>.<i>.jpg (the source's split)."""
    folder = download_json(raw, "ibean", licence)
    seeds = itertools.count(1000)
    for cls in IBEAN_CLASSES:
        for i in range(per_class):
            split = (
                "train" if i < 0.6 * per_class else "validation" if i < 0.8 * per_class else "test"
            )
            jpeg(
                folder / "extracted" / split / cls / f"{cls}_{IBEAN_TAG[split]}.{i}.jpg",
                next(seeds),
            )
    return folder


EAT = timezone(timedelta(hours=3))
BOTH, ALS_ONLY = ("Bean Rust", "ALS"), ("ALS",)
# capture day -> (districts, classes annotated that day); as on disk: 25-26 Apr are shared
# by Bugiri and Mayuge, and on 26 Apr only angular leaf spot was annotated
APRIL = {
    "2021-04-22": (("Mubende",), BOTH),
    "2021-04-23": (("Hoima",), BOTH),
    "2021-04-24": (("Kiboga",), BOTH),
    "2021-04-25": (("Bugiri", "Mayuge"), BOTH),
    "2021-04-26": (("Bugiri", "Mayuge"), ALS_ONLY),
}
MAY = {
    "2021-05-21": (("Kayunga",), BOTH),
    "2021-05-24": (("Serere",), BOTH),
    "2021-05-25": (("Ntungamo",), BOTH),
}
APRIL_DAYS = tuple(APRIL)
BLOCK_OF_DAY = {
    "2021-04-22": "district:Mubende",
    "2021-04-23": "district:Hoima",
    "2021-04-24": "district:Kiboga",
    "2021-04-25": "district:Bugiri+Mayuge",
    "2021-04-26": "district:Bugiri+Mayuge",
}


def makerere_xml(path: Path, *, cls: str, district: str, when: str, boxes) -> None:
    """A Makerere annotation as the archives hold it: one line, name= on every element."""
    folder = path.parent.name
    objects = "".join(
        f'<object><name name="name">{cls}</name><bndbox><xmin name="xmin">{x0}</xmin>'
        f'<ymin name="ymin">{y0}</ymin><xmax name="xmax">{x1}</xmax>'
        f'<ymax name="ymax">{y1}</ymax></bndbox></object>'
        for x0, y0, x1, y1 in boxes
    )
    path.write_text(
        f'<annotation><folder name="folder">{folder}</folder>'
        f'<filename name="filename">{path.stem}.jpg</filename>'
        f'<path name="path">beans/{folder}/{path.stem}.jpg</path>'
        '<size><width name="width">96</width><height name="height">72</height>'
        '<depth name="depth">3</depth></size>'
        f'<class name="class">{cls}</class>'
        '<hasOtherSymptoms name="has other symptoms">False</hasOtherSymptoms>'
        '<variety name="variety">Nambale short</variety><age name="age">4</age>'
        f'<district name="district">{district}</district>'
        f'<subcounty name="subcounty">{district} Town</subcounty>'
        f'<datetime name="datetime">{when}</datetime>{objects}</annotation>',
        encoding="utf-8",
    )


def make_makerere(raw: Path, *, healthy_days: tuple[str, ...] = APRIL_DAYS) -> Path:
    """data/raw/makerere: ALS and rust images with an XML beside them in als_1/als_1/ and
    bean_rust_1/bean_rust_1/ (two rust and one ALS per district and day, ALS only on
    26 Apr); healthy images without one in healthy_1/healthy_1/. Files are named by their
    capture time in Unix milliseconds, which dates the healthy images; their EXIF was
    rewritten after the trip, as on disk: IFD0 DateTime a later edit, and DateTimeOriginal
    missing, after the trip or right, one each. Two trips, as on disk: April with healthy
    images, May without; three healthy images on each of `healthy_days`."""
    folder = download_json(raw, "makerere", "CC0-1.0", doi="10.7910/DVN/TCKVEW", version="2.1")
    ext = folder / "extracted"
    seeds = itertools.count(2000)
    minutes = itertools.count()

    def capture(day: str) -> tuple[str, datetime]:
        t = datetime.fromisoformat(f"{day}T09:00:00").replace(tzinfo=EAT)
        t += timedelta(minutes=next(minutes))
        return str(int(t.timestamp() * 1000)), t

    for day, (districts, classes) in {**APRIL, **MAY}.items():
        for district in districts:
            for cls, sub, n in (("Bean Rust", "bean_rust_1", 2), ("ALS", "als_1", 1)):
                for _ in range(n if cls in classes else 0):
                    name, t = capture(day)
                    jpeg(ext / sub / sub / f"{name}.jpg", next(seeds))
                    makerere_xml(
                        ext / sub / sub / f"{name}.xml",
                        cls=cls,
                        district=district,
                        when=t.isoformat(timespec="milliseconds"),
                        boxes=[(10, 12, 40, 30)],
                    )
    for day in healthy_days:
        for taken in (None, "2021:06:10 10:00:00", "right"):
            name, t = capture(day)
            jpeg(
                ext / "healthy_1" / "healthy_1" / f"{name}.jpg",
                next(seeds),
                taken=t.strftime("%Y:%m:%d %H:%M:%S") if taken == "right" else taken,
                modified="2021:08:07 17:19:38",
            )
    return folder


TZ_DAYS = {  # class folder -> capture days; each class has days of its own, as on disk
    "healthy": ("2022-11-24", "2022-11-29", "2022-11-30", "2022-12-02"),
    "rust": ("2023-02-02", "2023-02-06", "2023-02-10", "2023-02-17"),
    "anthra": ("2023-03-03", "2023-03-13", "2023-04-06", "2023-04-07"),
}


def make_tanzania(raw: Path) -> None:
    """data/raw/tz155k and data/raw/tz59k. tz155k: healthy1/healthy1/<unix ms>.jpg (the
    name is not the capture time: days later here, as for most of the real ones),
    rust1/rust1/rustN.jpg, anthra/anthra/anthraN.jpg; two pictures per class and day, every
    rust picture held again in rust2/rust2/, and one rust picture without EXIF. tz59k:
    copies of the tz155k rust and anthracnose pictures in rust/rust/ and anthra/anthra/,
    plus one picture of its own on a tz155k rust day."""
    tz155k = {"doi": "10.5281/zenodo.15685315", "version": None, "published": "2025-07-01"}
    download_json(raw, "tz155k", "CC-BY-4.0", **tz155k)
    download_json(raw, "tz59k", "CC-BY-4.0", doi="10.5281/zenodo.8286126", version="01")
    a, b = raw / "tz155k" / "extracted", raw / "tz59k" / "extracted"
    seeds, n = itertools.count(3000), itertools.count()
    for cls, days in TZ_DAYS.items():
        for day in days:
            for _ in range(2):
                i = next(n)
                t = datetime.fromisoformat(f"{day}T10:00:00").replace(tzinfo=EAT)
                t += timedelta(minutes=i)
                taken = t.strftime("%Y:%m:%d %H:%M:%S")
                if cls == "healthy":
                    upload_ms = int((t + timedelta(days=5)).timestamp() * 1000)
                    jpeg(a / "healthy1" / "healthy1" / f"{upload_ms}.jpg", next(seeds), taken=taken)
                    continue
                sub = "rust1" if cls == "rust" else "anthra"
                p = jpeg(a / sub / sub / f"{cls}{i}.jpg", next(seeds), taken=taken)
                if cls == "rust":
                    copy(p, a / "rust2" / "rust2" / f"rust{i + 1000}.jpg")
                copy(p, b / cls / cls / f"{cls}{i}.jpg")
    jpeg(a / "rust1" / "rust1" / "rust999.jpg", next(seeds))  # no EXIF at all
    jpeg(b / "rust" / "rust" / "rust5000.jpg", next(seeds), taken="2023:02:06 11:00:00")


SWM_PICTURES = {  # R-SWM original -> the labels of its boxes, in the XML's order
    "ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-04-IMG_3825": (
        "White Mold",
        "Mature Sclerotium",
        "White Mold",
    ),
    "ds-2023-06-02-vianopolis-go-20230602_135127": ("White Mold",),
    "ds-2023-07-06-ipameri-go-fazenda-jolart-20230706_162233": ("Apothecium",),
    "ds-2023-09-30-turvelandia-go-grupo-santa-fe-02-pivo-02-20230930_120257": (
        "Mature Sclerotium",
        "Mature Sclerotium",
    ),
}
SWM_ROOT = "swm/extracted/r-swm-dataset/r-swm-dataset"


def voc_box(i: int) -> tuple[int, int, int, int]:
    return 5 + 10 * i, 4 + 8 * i, 25 + 10 * i, 20 + 8 * i


def make_swm(raw: Path) -> Path:
    """data/raw/swm, R-SWM part: original-images/<split>/images/<name>.jpg, the same picture
    with its boxes drawn in under original-images/<split>/bbox/<name>-bboxes.jpg, and the
    Pascal VOC XML under pascal-voc/<split>/<name>.xml."""
    download_json(raw, "swm", "CC-BY-4.0", doi="10.17632/jxvmr8rchx.1")
    base = raw / SWM_ROOT
    seeds = itertools.count(4000)
    for name, labels in SWM_PICTURES.items():
        original = jpeg(base / "original-images" / "train" / "images" / f"{name}.jpg", next(seeds))
        drawn = Image.open(original).convert("RGB")
        for i in range(len(labels)):
            ImageDraw.Draw(drawn).rectangle(voc_box(i), outline=(255, 0, 0), width=2)
        bbox = base / "original-images" / "train" / "bbox" / f"{name}-bboxes.jpg"
        bbox.parent.mkdir(parents=True, exist_ok=True)
        drawn.save(bbox, "JPEG", quality=90)
        objects = "".join(
            f"<object><name>{label}</name><pose>Unspecified</pose><truncated>0</truncated>"
            "<difficult/><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax>"
            "<ymax>{}</ymax></bndbox></object>".format(*voc_box(i))
            for i, label in enumerate(labels)
        )
        xml = base / "pascal-voc" / "train" / f"{name}.xml"
        xml.parent.mkdir(parents=True, exist_ok=True)
        xml.write_text(
            f"<annotation><folder/><filename>{name}</filename>"
            "<source><database/><annotation/><image/></source>"
            f"<size><width>96</width><height>72</height><depth>3</depth></size>{objects}"
            "</annotation>",
            encoding="utf-8",
        )
    return raw / "swm"


# --- configs: green from W1 ---------------------------------------------------------------------


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_class_map_v1_maps_every_label_on_disk_to_a_km2_class():
    cm = _yaml(CONFIGS / "class_map_v1.yaml")
    assert cm["version"] == 1
    assert cm["classes"] == {
        "trained": ["healthy", "rust", "anthracnose"],
        "unknown": ["unknown_als", "unknown_wm"],
        "unlabelled": ["none"],
    }
    allowed = {c for group in cm["classes"].values() for c in group} | {"excluded"}
    recipes = {p.stem for p in (CONFIGS / "manifests").glob("*.yaml")}
    assert set(cm["map"]) == recipes
    for manifest, table in cm["map"].items():
        assert set(table.values()) <= allowed, manifest
    # every label the W1 data note found on disk, verbatim
    on_disk = {
        "ibean": {"healthy": "healthy", "bean_rust": "rust", "angular_leaf_spot": "unknown_als"},
        "makerere": {"healthy": "healthy", "Bean Rust": "rust", "ALS": "unknown_als"},
        "tanzania": {"healthy": "healthy", "rust": "rust", "anthra": "anthracnose"},
    }
    for manifest, table in on_disk.items():
        assert table.items() <= cm["map"][manifest].items(), manifest
    # angular leaf spot and white mould are the held-out unknowns, nothing else is
    for table in cm["map"].values():
        for label, km2 in table.items():
            assert (km2 == "unknown_als") == (label in ("ALS", "angular_leaf_spot")), label
            assert (km2 == "unknown_wm") == ("White Mold" in label), label
    # SWM: every picture-level label set in R-SWM, and the crops' label (sic)
    swm = cm["map"]["swm"]
    for label in (
        "White Mold",
        "Mature Sclerotium + White Mold",
        "Imature Sclerotium and White Mold",
        "Apothecium",
        "Mature Sclerotium",
        "Apothecium + Mature Sclerotium",
    ):
        assert label in swm, label
    assert {k for k, v in swm.items() if v == "excluded"} == {
        "Apothecium",
        "Mature Sclerotium",
        "Apothecium + Mature Sclerotium",
    }


def test_each_recipe_names_its_members_role_rule_and_keys():
    recipes = {p.stem: _yaml(p) for p in (CONFIGS / "manifests").glob("*.yaml")}
    assert set(recipes) == {"ibean", "makerere", "tanzania", "swm"}
    raw_ids = {p.stem for p in (CONFIGS / "datasets").glob("*.yaml")}
    for name, r in recipes.items():
        assert set(r) == {
            "manifest",
            "version",
            "members",
            "role",
            "rule",
            "group_keys",
            "seed",
            "phash_threshold",
        }, name
        assert r["manifest"] == name and r["version"] == 1
        assert r["members"] and set(r["members"]) <= raw_ids, name
        assert r["role"] in ROLES, name
        assert RULE.fullmatch(r["rule"]), name
        assert set(r["group_keys"]) <= set(GROUP_KEYS) - {"phash_group"}, name
        if r["rule"].startswith("blocked:"):
            assert r["rule"].removeprefix("blocked:") in r["group_keys"], name
        assert (r["seed"], r["phash_threshold"]) == (0, 2), name  # Clarification 4
    # the dataset table of spec 001
    got = {n: (r["members"], r["role"], r["rule"], r["group_keys"]) for n, r in recipes.items()}
    assert got == {
        "ibean": (["ibean"], "test_only", UNBLOCKED, []),
        "makerere": (
            ["makerere"],
            "train_eval",
            "blocked:district",
            ["district", "subcounty", "date", "variety", "plant_age"],
        ),
        "tanzania": (["tz155k", "tz59k"], "train_eval", "blocked:date", ["date"]),
        "swm": (["swm"], "holdout_unknown", "holdout_unknown", ["date", "session"]),
    }


# --- US-1: build --------------------------------------------------------------------------------


def test_us1_1_the_ibean_fixture_builds_into_30_rows(tmp_path, capsys):
    manifest = built(capsys, FIXTURE, tmp_path, "ibean")
    rs = rows(manifest)
    assert len(rs) == 30
    for r in rs:
        assert sha256(FIXTURE / r["path"]) == r["sha256"], r["path"]
    assert Counter(r["class_km2"] for r in rs) == {"healthy": 10, "rust": 10, "unknown_als": 10}
    unknown = [r for r in rs if r["class_km2"] == "unknown_als"]
    assert {(r["split"], r["split_rule"]) for r in unknown} == {("holdout_unknown",) * 2}
    trained = [r for r in rs if r["class_km2"] != "unknown_als"]
    assert {r["split"] for r in trained} == set(SPLITS)
    assert {r["split_rule"] for r in trained} == {UNBLOCKED}
    m = meta(manifest)
    assert m["counts"]["rows"] == 30 and m["role"] == "test_only"
    assert m["counts"]["class_km2"] == {"healthy": 10, "rust": 10, "unknown_als": 10}
    assert_valid(capsys, manifest, FIXTURE)


def test_us1_1_the_committed_fixture_manifest_is_what_the_builder_writes(tmp_path, capsys):
    """tests/fixtures/make_ibean_30.py writes ibean_v1.jsonl until the builder exists (W2).
    From then on, a build must give the same rows. Only `split` may differ: it is the
    builder's own seeded draw (FR-008). Every other field is content (FR-003, FR-006)."""
    committed = FIXTURE / "ibean_v1.jsonl"
    assert_valid(capsys, committed, FIXTURE)
    fresh = {r["image_id"]: r for r in rows(built(capsys, FIXTURE, tmp_path, "ibean"))}
    ours = {r["image_id"]: r for r in rows(committed)}
    assert fresh.keys() == ours.keys()
    for image_id, row in ours.items():
        without_split = {k: v for k, v in row.items() if k != "split"}
        assert without_split == {k: v for k, v in fresh[image_id].items() if k != "split"}, image_id


def test_us1_rows_carry_every_contract_field(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")

    text = manifest.read_text(encoding="utf-8")
    assert text.endswith("\n") and "\r" not in text and "\n\n" not in text
    rs = rows(manifest)
    assert len(rs) == 30
    assert [r["image_id"] for r in rs] == sorted(r["image_id"] for r in rs)
    for r in rs:
        assert list(r) == FIELDS
        assert list(r["group_keys"]) == GROUP_KEYS
        assert r["manifest_version"] == "1" and r["dataset"] == "ibean"
        assert re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
        assert re.fullmatch(r"[0-9a-f]{16}", r["phash"])
        assert r["image_id"] == f"ibean_{r['sha256'][:16]}"
        assert r["path"].startswith("ibean/extracted/") and "\\" not in r["path"]
        assert r["sha256"] == sha256(raw / r["path"])
        assert r["dup_paths"] == []
        assert (r["width"], r["height"], r["format"]) == (96, 72, "jpeg")
        assert r["class_raw"] == r["path"].split("/")[3]  # the folder, verbatim
        assert r["boxes"] is None
        assert r["source_record"] == "https://example.org/records/ibean"  # no DOI
        assert r["source_version"] == "1"
        assert r["licence"] == "MIT" and r["attribution"] == attribution(raw, "ibean")
        keys = r["group_keys"]
        assert all(keys[k] is None for k in GROUP_KEYS if k != "phash_group")
        assert keys["phash_group"] and r["split_group"] == keys["phash_group"]
        assert r["split"] in (*SPLITS, "holdout_unknown")
    m = meta(manifest)
    assert m["manifest"] == "ibean" and m["version"] == 1
    assert (m["role"], m["rule"], m["seed"], m["phash_threshold"]) == ("test_only", UNBLOCKED, 0, 2)
    assert m["manifest_sha256"] == sha256(manifest)
    assert m["counts"]["rows"] == len(rs)
    assert m["counts"]["class_raw"] == dict(Counter(r["class_raw"] for r in rs))
    assert m["counts"]["class_km2"] == dict(Counter(r["class_km2"] for r in rs))
    assert m["counts"]["split"] == dict(Counter(r["split"] for r in rs))
    assert m["excluded"] == []
    assert m["members"]["ibean"]["files"] == 30 and m["members"]["ibean"]["distinct"] == 30
    assert_valid(capsys, manifest, raw)


def test_us1_width_and_height_are_orientation_applied_and_the_hash_is_of_the_stored_bytes(
    tmp_path, capsys
):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    turned = jpeg(ext / "train" / "healthy" / "healthy_train.40.jpg", 1500, orientation=6)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    row = by_path(rows(manifest))["ibean/extracted/train/healthy/healthy_train.40.jpg"]
    assert (row["width"], row["height"]) == (72, 96)
    assert row["sha256"] == sha256(turned)


def test_us1_2_building_twice_gives_the_same_bytes(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    first = built(capsys, raw, tmp_path / "a", "makerere")
    jsonl, sidecar = first.read_bytes(), first.with_name("makerere_v1.meta.json").read_bytes()

    again = built(capsys, raw, tmp_path / "a", "makerere")
    assert again.read_bytes() == jsonl
    assert again.with_name("makerere_v1.meta.json").read_bytes() == sidecar  # not rewritten

    elsewhere = built(capsys, raw, tmp_path / "b", "makerere")
    assert elsewhere.read_bytes() == jsonl


def test_us1_3_non_images_and_broken_jpegs_are_excluded_with_a_reason(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    # as on disk: a macOS .DS_Store renamed by the source (W1 data note, finding 9)
    (ext / "train" / "healthy" / "healthy_train.120tore").write_bytes(
        b"\x00\x00\x00\x01Bud1" + bytes(2048)
    )
    whole = jpeg(tmp_path / "whole.jpg", 1999).read_bytes()
    scan = whole.index(
        b"\xff\xda"
    )  # cut inside the entropy-coded data: it opens, but will not decode
    (ext / "test" / "bean_rust" / "bean_rust_test.99.jpg").write_bytes(
        whole[: scan + (len(whole) - scan) // 2]
    )
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    assert excluded(manifest) == {
        "ibean/extracted/train/healthy/healthy_train.120tore": "not_an_image",
        "ibean/extracted/test/bean_rust/bean_rust_test.99.jpg": "decode_error",
    }
    assert len(rows(manifest)) == 30


def test_us1_4_every_scanned_file_is_a_row_a_copy_or_excluded(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    (ext / "train" / "healthy" / "healthy_train.120tore").write_bytes(b"\x00\x00\x00\x01Bud1")
    copy(
        ext / "train" / "bean_rust" / "bean_rust_train.0.jpg", ext / "test" / "bean_rust" / "b.jpg"
    )
    copy(ext / "train" / "healthy" / "healthy_train.0.jpg", ext / "test" / "bean_rust" / "c.jpg")
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    listed = (
        [r["path"] for r in rs]
        + [p for r in rs for p in r["dup_paths"]]
        + [e["path"] for e in meta(manifest)["excluded"]]
    )
    on_disk = [p.relative_to(raw).as_posix() for p in ext.rglob("*") if p.is_file()]
    assert sorted(listed) == sorted(on_disk)  # each file exactly once


# --- US-2: duplicates -----------------------------------------------------------------------


def test_us2_1_2_copies_become_one_row_and_conflicting_copies_none(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    same = copy(
        ext / "train" / "bean_rust" / "bean_rust_train.0.jpg",
        ext / "validation" / "bean_rust" / "bean_rust_val.50.jpg",
    )
    clash = copy(
        ext / "train" / "healthy" / "healthy_train.0.jpg",
        ext / "test" / "bean_rust" / "bean_rust_test.50.jpg",
    )
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    assert len(rs) == 29  # 30 pictures, one of them under two labels
    by_sha = {r["sha256"]: r for r in rs}
    row = by_sha[sha256(same)]
    assert row["path"] == "ibean/extracted/train/bean_rust/bean_rust_train.0.jpg"
    assert row["dup_paths"] == ["ibean/extracted/validation/bean_rust/bean_rust_val.50.jpg"]
    assert sha256(clash) not in by_sha
    assert excluded(manifest) == {
        "ibean/extracted/train/healthy/healthy_train.0.jpg": "label_conflict",
        "ibean/extracted/test/bean_rust/bean_rust_test.50.jpg": "label_conflict",
    }
    assert meta(manifest)["members"]["ibean"] == {"files": 32, "distinct": 30, "shared_with": {}}


def test_us2_3_a_reencoded_copy_shares_the_phash_group_and_the_split(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    src = jpeg(ext / "train" / "healthy" / "healthy_train.90.jpg", NEAR_DUP_SEED)
    Image.open(src).save(ext / "test" / "healthy" / "healthy_test.90.jpg", "JPEG", quality=70)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    a = by_path(rs)["ibean/extracted/train/healthy/healthy_train.90.jpg"]
    b = by_path(rs)["ibean/extracted/test/healthy/healthy_test.90.jpg"]
    assert a["sha256"] != b["sha256"]
    assert hamming(a["phash"], b["phash"]) <= meta(manifest)["phash_threshold"]
    assert a["group_keys"]["phash_group"] == b["group_keys"]["phash_group"]
    assert (a["split_group"], a["split"]) == (b["split_group"], b["split"])
    # the phash_group is the smallest image_id of the component; other pictures stay apart
    assert a["group_keys"]["phash_group"] == min(a["image_id"], b["image_id"])
    sizes = Counter(r["group_keys"]["phash_group"] for r in rs)
    assert sorted(sizes.values()) == [1] * (len(rs) - 2) + [2]
    assert_valid(capsys, manifest, raw)


def test_us2_4_tanzania_is_one_manifest_over_both_records(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_tanzania(raw)
    manifest = built(capsys, raw, tmp_path / "out", "tanzania")
    rs = rows(manifest)
    files = {m: list((raw / m / "extracted").rglob("*.jpg")) for m in ("tz155k", "tz59k")}
    distinct = {m: {sha256(p) for p in ps} for m, ps in files.items()}
    assert sorted(r["sha256"] for r in rs) == sorted(distinct["tz155k"] | distinct["tz59k"])
    assert all(r["dataset"] == "tanzania" and r["image_id"].startswith("tanzania_") for r in rs)

    rust = by_path(rs)["tz155k/extracted/rust1/rust1/rust8.jpg"]  # the first rust picture
    assert rust["class_raw"] == "rust" and rust["class_km2"] == "rust"
    assert rust["dup_paths"] == [
        "tz155k/extracted/rust2/rust2/rust1008.jpg",
        "tz59k/extracted/rust/rust/rust8.jpg",
    ]
    assert rust["source_record"] == "https://doi.org/10.5281/zenodo.15685315"
    assert rust["source_version"] == "2025-07-01"  # the record has no version: its date
    assert rust["attribution"] == attribution(raw, "tz155k")

    (own,) = [r for r in rs if r["path"].startswith("tz59k/")]
    assert own["path"] == "tz59k/extracted/rust/rust/rust5000.jpg"
    assert own["source_record"] == "https://doi.org/10.5281/zenodo.8286126"
    assert own["source_version"] == "01" and own["attribution"] == attribution(raw, "tz59k")
    assert {r["class_raw"] for r in rs} == {"healthy", "rust", "anthra"}
    assert {r["class_km2"] for r in rs} == set(TRAINED)

    shared = len(distinct["tz155k"] & distinct["tz59k"])
    members = meta(manifest)["members"]
    for m, other in (("tz155k", "tz59k"), ("tz59k", "tz155k")):
        assert members[m]["files"] == len(files[m])
        assert members[m]["distinct"] == len(distinct[m])
        assert members[m]["shared_with"] == {other: shared}
    assert_valid(capsys, manifest, raw)


def test_us2_5_validate_names_a_group_that_straddles_splits(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    train = next(i for i, r in enumerate(rs) if r["split"] == "train")
    test = next(i for i, r in enumerate(rs) if r["split"] == "test")
    group = rs[train]["group_keys"]["phash_group"]
    rs[test]["group_keys"]["phash_group"] = rs[test]["split_group"] = group
    write_rows(manifest, rs)

    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 2
    reason = re.compile(rf"({'|'.join(VALIDATE_REASONS)}):")
    first = next(line for line in text.splitlines() if reason.match(line))
    assert first.startswith("group_straddles_split:"), text
    assert group in first
    for line_no in (train + 1, test + 1):
        assert re.search(rf"\b{line_no}\b", first), first


def test_us2_6_overlap_reports_shared_images_and_merges_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    make_makerere(raw)
    healthy = sorted((raw / "makerere" / "extracted" / "healthy_1" / "healthy_1").glob("*.jpg"))
    copy(healthy[0], ext / "test" / "healthy" / "healthy_test.60.jpg")
    a = built(capsys, raw, tmp_path / "out", "ibean")
    b = built(capsys, raw, tmp_path / "out", "makerere")

    module = _manifests()
    code = module.main(["overlap", str(a), str(b)])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (report["exact"], report["near"]) == (1, 0)
    shared = sha256(healthy[0])
    assert shared in {r["sha256"] for r in rows(a)} and shared in {r["sha256"] for r in rows(b)}


# --- US-3: splits ---------------------------------------------------------------------------


def test_us3_1_makerere_is_blocked_by_district(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    manifest = built(capsys, raw, tmp_path / "out", "makerere")
    rs = rows(manifest)
    trained = [r for r in rs if r["class_km2"] in TRAINED]
    assert {r["split_rule"] for r in trained} == {"blocked:district"}
    splits_of = defaultdict(set)
    for r in trained:
        if r["group_keys"]["district"]:
            splits_of[r["group_keys"]["district"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_of.values()), splits_of
    assert splits_of["Bugiri"] == splits_of["Mayuge"]  # merged by the healthy images of 25 Apr
    for cls in ("healthy", "rust"):
        assert {r["split"] for r in trained if r["class_km2"] == cls} == set(SPLITS), cls

    m = meta(manifest)
    assert (m["role"], m["rule"]) == ("train_eval", "blocked:district")
    listed = {block: split for split, blocks in m["splits"].items() for block in blocks}
    assert listed == {r["split_group"]: r["split"] for r in trained}
    for split, blocks in m["splits"].items():
        assert sum(blocks.values()) == sum(r["split"] == split for r in trained)
    assert "district:Bugiri+Mayuge" in listed and "district:Mubende" in listed
    assert_valid(capsys, manifest, raw)


def test_us3_2_images_without_a_district_follow_their_capture_date(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    manifest = built(capsys, raw, tmp_path / "out", "makerere")
    rs = rows(manifest)
    healthy = [r for r in rs if r["class_km2"] == "healthy"]
    assert len(healthy) == 15
    assert all(r["group_keys"]["district"] is None for r in healthy)  # not invented (FR-013)
    # the file name (Unix ms, UTC+03:00), not the EXIF: DateTimeOriginal is missing, after
    # the trip (2021-06-10) or right, and IFD0 DateTime is a later edit (2021-08-07)
    assert {r["group_keys"]["date"] for r in healthy} == set(APRIL)
    # 26 Apr has held-out ALS annotations only; they still say where the team was
    assert {r["split_group"] for r in healthy if r["group_keys"]["date"] == "2021-04-26"} == {
        "district:Bugiri+Mayuge"
    }
    split_of_block = {
        r["split_group"]: r["split"]
        for r in rs
        if r["group_keys"]["district"] and r["class_km2"] == "rust"
    }
    for r in healthy:
        assert r["split_group"] == BLOCK_OF_DAY[r["group_keys"]["date"]]
        assert r["split"] == split_of_block[r["split_group"]]
        assert r["split_rule"] == "blocked:district"


def test_us3_2_an_image_the_rule_cannot_place_is_split_unblocked_and_counted(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw, healthy_days=(*APRIL_DAYS, "2021-04-30"))  # nothing else on 30 Apr
    manifest = built(capsys, raw, tmp_path / "out", "makerere")
    rs = rows(manifest)
    stray = [r for r in rs if r["group_keys"]["date"] == "2021-04-30"]
    assert len(stray) == 3
    for r in stray:
        assert r["split_rule"] == UNBLOCKED
        assert r["split_group"] == r["group_keys"]["phash_group"]
        assert r["split"] in SPLITS
    placed = [r for r in rs if r["class_km2"] in TRAINED and r not in stray]
    assert {r["split_rule"] for r in placed} == {"blocked:district"}
    m = meta(manifest)
    assert m["counts"]["split_rule"][UNBLOCKED] == 3
    assert "partly_unblocked" in m["warnings"]
    assert_valid(capsys, manifest, raw)


def test_makerere_keys_boxes_and_provenance_come_from_the_xml_and_the_record(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw)
    manifest = built(capsys, raw, tmp_path / "out", "makerere")
    rs = rows(manifest)
    (rust, _) = [
        r for r in rs if r["class_raw"] == "Bean Rust" and r["group_keys"]["district"] == "Mubende"
    ]
    assert rust["class_km2"] == "rust"
    assert rust["group_keys"] == {
        "district": "Mubende",
        "subcounty": "Mubende Town",
        "date": "2021-04-22",
        "region": None,
        "session": None,
        "variety": "Nambale short",
        "plant_age": 4,
        "phash_group": rust["group_keys"]["phash_group"],
    }
    assert rust["boxes"] == [{"x": 10, "y": 12, "w": 30, "h": 18, "class_raw": "Bean Rust"}]
    assert (rust["licence"], rust["source_record"], rust["source_version"]) == (
        "CC0-1.0",
        "https://doi.org/10.7910/DVN/TCKVEW",
        "2.1",
    )
    assert rust["attribution"] == attribution(raw, "makerere")
    als = [r for r in rs if r["class_raw"] == "ALS"]
    assert len(als) == 10
    for r in als:
        assert (r["class_km2"], r["split"], r["split_rule"]) == (
            "unknown_als",
            *["holdout_unknown"] * 2,
        )
        assert r["split_group"] == r["group_keys"]["phash_group"]
    assert all(r["boxes"] is None for r in rs if r["class_raw"] == "healthy")
    # every picture is a row; the XML files are annotations the reader consumed (US-1.4)
    pictures = (raw / "makerere" / "extracted").rglob("*.jpg")
    assert sorted(r["path"] for r in rs) == sorted(p.relative_to(raw).as_posix() for p in pictures)


def test_us3_3_tanzania_is_blocked_by_capture_date(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_tanzania(raw)
    manifest = built(capsys, raw, tmp_path / "out", "tanzania")
    rs = rows(manifest)
    dated = [r for r in rs if r["group_keys"]["date"]]
    (undated,) = [r for r in rs if not r["group_keys"]["date"]]  # the picture without EXIF
    assert undated["path"] == "tz155k/extracted/rust1/rust1/rust999.jpg"
    assert undated["split_rule"] == UNBLOCKED
    assert undated["split_group"] == undated["group_keys"]["phash_group"]

    # EXIF DateTimeOriginal, not the file name (tz155k healthy names are upload times)
    assert {r["group_keys"]["date"] for r in dated} == {d for ds in TZ_DAYS.values() for d in ds}
    assert {r["split_rule"] for r in dated} == {"blocked:date"}
    splits_of = defaultdict(set)
    for r in dated:
        assert r["split_group"] == f"date:{r['group_keys']['date']}"
        splits_of[r["group_keys"]["date"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_of.values()), splits_of
    for cls in TRAINED:
        assert {r["split"] for r in rs if r["class_km2"] == cls} == set(SPLITS), cls
    for r in rs:
        assert all(
            r["group_keys"][k] is None for k in GROUP_KEYS if k not in ("date", "phash_group")
        )
    assert "partly_unblocked" in meta(manifest)["warnings"]
    assert_valid(capsys, manifest, raw)


def test_us3_4_ibean_is_unblocked_and_says_so(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    trained = [r for r in rows(manifest) if r["class_km2"] in TRAINED]
    assert {r["split_rule"] for r in trained} == {UNBLOCKED}
    for cls in ("healthy", "rust"):
        assert {r["split"] for r in trained if r["class_km2"] == cls} == set(SPLITS), cls
    m = meta(manifest)
    assert "unblocked" in m["warnings"]
    assert "splits" not in m or not m["splits"]


def test_us3_5_a_blocked_rule_without_its_key_fails_and_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    code, text = build(capsys, raw, tmp_path / "out", "ibean", "--rule", "blocked:district")
    assert code == 2
    assert "rule_not_applicable" in text
    assert not (tmp_path / "out" / "ibean_v1.jsonl").exists()


def test_us3_6_a_split_that_cannot_hold_every_class_fails_and_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_makerere(raw, healthy_days=("2021-04-22",))  # healthy in one block only
    code, text = build(capsys, raw, tmp_path / "out", "makerere")
    assert code == 2
    assert "class_missing_from_split" in text and "healthy" in text
    assert not (tmp_path / "out" / "makerere_v1.jsonl").exists()


# --- US-4: the class map ----------------------------------------------------------------------


def test_us4_1_an_unmapped_label_fails_the_build_and_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    jpeg(ext / "test" / "downy_mildew" / "downy_mildew_test.0.jpg", 1998)
    code, text = build(capsys, raw, tmp_path / "out", "ibean")
    assert code == 2
    assert "unmapped_class" in text and "downy_mildew" in text
    assert not (tmp_path / "out" / "ibean_v1.jsonl").exists()


def test_us4_2_the_unknowns_are_held_out_and_never_trained(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    als = [r for r in rs if r["class_raw"] == "angular_leaf_spot"]
    assert len(als) == 10
    for r in als:
        assert (r["class_km2"], r["split"], r["split_rule"]) == (
            "unknown_als",
            *["holdout_unknown"] * 2,
        )

    als[0]["split"], als[0]["split_rule"] = "train", UNBLOCKED
    write_rows(manifest, rs)
    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 2
    assert "unknown_in_train" in text


def test_us4_3_swm_keeps_the_white_mould_pictures_and_excludes_the_rest(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_swm(raw)
    manifest = built(capsys, raw, tmp_path / "out", "swm")
    rs = by_path(rows(manifest))
    images = f"{SWM_ROOT}/original-images/train/images"
    both = rs[
        f"{images}/ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-04-IMG_3825.jpg"
    ]
    only = rs[f"{images}/ds-2023-06-02-vianopolis-go-20230602_135127.jpg"]
    assert len(rs) == 2
    assert both["class_raw"] == "Mature Sclerotium + White Mold"
    assert only["class_raw"] == "White Mold"
    for r in (both, only):
        assert (r["class_km2"], r["split"], r["split_rule"]) == (
            "unknown_wm",
            *["holdout_unknown"] * 2,
        )
    assert [b["class_raw"] for b in both["boxes"]] == [
        "White Mold",
        "Mature Sclerotium",
        "White Mold",
    ]
    assert both["boxes"][0] == {"x": 5, "y": 4, "w": 20, "h": 16, "class_raw": "White Mold"}
    assert both["group_keys"]["date"] == "2023-09-07"
    assert (
        both["group_keys"]["session"]
        == "ds-2023-09-07-santa-helena-de-goias-go-fazenda-sete-ilhas-pivo-04"
    )
    assert only["group_keys"]["session"] == "ds-2023-06-02-vianopolis-go"

    reasons = excluded(manifest)
    assert reasons.pop(f"{images}/ds-2023-07-06-ipameri-go-fazenda-jolart-20230706_162233.jpg") == (
        "excluded_by_class_map"
    )
    assert (
        reasons.pop(
            f"{images}/ds-2023-09-30-turvelandia-go-grupo-santa-fe-02-pivo-02-20230930_120257.jpg"
        )
        == "excluded_by_class_map"
    )
    assert reasons == {
        f"{SWM_ROOT}/original-images/train/bbox/{name}-bboxes.jpg": "drawn_boxes"
        for name in SWM_PICTURES
    }
    assert meta(manifest)["role"] == "holdout_unknown"
    assert_valid(capsys, manifest, raw)


# --- US-5: freeze -------------------------------------------------------------------------------


def test_us5_1_freeze_records_the_hash_once(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    code, text = run(capsys, "freeze", manifest)
    assert code == 0, text
    frozen = tmp_path / "out" / "FROZEN.jsonl"
    (line,) = frozen.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert set(record) == {"manifest", "sha256", "frozen_at", "git_sha"}
    assert (record["manifest"], record["sha256"]) == ("ibean_v1.jsonl", sha256(manifest))
    assert sha256(manifest) in text  # the line for DECISIONS.md

    code, _ = run(capsys, "freeze", manifest)
    assert code == 0
    assert len(frozen.read_text(encoding="utf-8").splitlines()) == 1


def test_us5_2_a_frozen_manifest_that_changed_is_refused(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    assert run(capsys, "freeze", manifest)[0] == 0
    rs = rows(manifest)
    rs[0]["notes"] = "edited after the freeze"
    write_rows(manifest, rs)

    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 3
    assert "frozen_manifest_modified" in text
    module = _manifests()
    with pytest.raises(module.FrozenManifestModified):
        module.load_manifest(manifest, "train", "train")


def test_us5_3_test_and_holdout_rows_are_read_only_for_evaluation(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    module = _manifests()
    for purpose in ("train", "select"):
        for split in ("test", "holdout_unknown"):
            with pytest.raises(module.TestSplitAccessError):
                module.load_manifest(manifest, split, purpose)
        loaded = module.load_manifest(manifest, "train", purpose)
        assert loaded.rows and {r["split"] for r in loaded.rows} == {"train"}
    for purpose in ("evaluate", "serve", "extract"):
        loaded = module.load_manifest(manifest, "test", purpose)
        assert loaded.rows and {r["split"] for r in loaded.rows} == {"test"}
    assert loaded.sha256 == sha256(manifest)
    assert loaded.frozen is False
    assert run(capsys, "freeze", manifest)[0] == 0
    assert module.load_manifest(manifest, "val", "select").frozen is True


def test_us5_4_a_frozen_version_is_never_rebuilt_into_something_else(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    out = tmp_path / "out"
    manifest = built(capsys, raw, out, "ibean")
    assert run(capsys, "freeze", manifest)[0] == 0
    before = manifest.read_bytes()

    assert build(capsys, raw, out, "ibean")[0] == 0  # same inputs: nothing to do
    jpeg(ext / "test" / "healthy" / "healthy_test.70.jpg", 1997)  # the data changed
    code, text = build(capsys, raw, out, "ibean")
    assert code == 3
    assert "frozen_manifest_modified" in text
    assert manifest.read_bytes() == before

    code, text = build(capsys, raw, out, "ibean", "--version", 2)
    assert code == 0, text
    assert len(rows(out / "ibean_v2.jsonl")) == 31
    assert manifest.read_bytes() == before


def test_us5_5_an_unknown_licence_is_built_only_on_request_and_never_frozen(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_ibean(raw, licence="unknown")
    out = tmp_path / "out"
    code, text = build(capsys, raw, out, "ibean")
    assert code == 2
    assert "licence_unknown" in text
    assert not (out / "ibean_v1.jsonl").exists()

    manifest = built(capsys, raw, out, "ibean", "--allow-unknown-licence")
    assert {r["licence"] for r in rows(manifest)} == {"unknown"}
    assert meta(manifest)["allow_unknown_licence"] is True
    assert_valid(capsys, manifest, raw)
    code, text = run(capsys, "freeze", manifest)
    assert code == 2
    assert "licence_unknown" in text
    assert not (out / "FROZEN.jsonl").exists()


# --- FR-009: what validate and build check -------------------------------------------------------


def _drop_licence(rs):
    del rs[0]["licence"]


def _bad_class(rs):
    rs[0]["class_km2"] = "downy_mildew"


def _bad_rule(rs):
    next(r for r in rs if r["split"] == "train")["split_rule"] = "random"


def _unsorted(rs):
    rs[0], rs[1] = rs[1], rs[0]


def _twice(rs):
    rs.insert(1, json.loads(json.dumps(rs[0])))


def _unlabelled_in_train(rs):
    next(r for r in rs if r["split"] == "train")["class_km2"] = "none"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_drop_licence, "missing_field"),
        (_bad_class, "bad_value"),
        (_bad_rule, "bad_split_rule"),
        (_unsorted, "rows_not_sorted"),
        (_twice, "duplicate_image_id"),
        (_unlabelled_in_train, "unlabelled_in_train"),
    ],
)
def test_validate_names_the_violation(tmp_path, capsys, mutate, reason):
    raw = tmp_path / "raw"
    make_ibean(raw)
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    rs = rows(manifest)
    mutate(rs)
    write_rows(manifest, rs)
    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 2
    assert f"{reason}:" in text, text


def test_validate_checks_every_row_against_the_file_it_names(tmp_path, capsys):
    raw = tmp_path / "raw"
    ext = make_ibean(raw) / "extracted"
    manifest = built(capsys, raw, tmp_path / "out", "ibean")
    jpeg(ext / "train" / "healthy" / "healthy_train.0.jpg", 1996)  # other bytes under the name
    (ext / "test" / "bean_rust" / "bean_rust_test.9.jpg").unlink()
    code, text = run(capsys, "validate", manifest, "--raw-root", raw)
    assert code == 2
    assert "sha256_mismatch:" in text and "missing_file:" in text


def test_build_refuses_a_file_that_differs_from_the_members_sha256sums(tmp_path, capsys):
    raw = tmp_path / "raw"
    folder = make_ibean(raw)
    sums = {
        p.relative_to(folder).as_posix(): sha256(p) for p in (folder / "extracted").rglob("*.jpg")
    }
    sums["extracted/train/healthy/healthy_train.0.jpg"] = "0" * 64
    (folder / "SHA256SUMS").write_text(
        "".join(f"{d}  {rel}\n" for rel, d in sorted(sums.items())), encoding="utf-8"
    )
    code, text = build(capsys, raw, tmp_path / "out", "ibean")
    assert code == 2
    assert "sha256_mismatch" in text and "healthy_train.0.jpg" in text
    assert not (tmp_path / "out" / "ibean_v1.jsonl").exists()
