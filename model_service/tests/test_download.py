"""The downloader (W1 "Downloads"): provenance, checksums, resume, idempotence.

No network: a fake source serves bytes from memory, and the record parsers are fed
trimmed copies of the real API responses (Zenodo, Hugging Face, Mendeley, Dataverse).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ms.data import download as dl

# --- helpers ----------------------------------------------------------------------


class FakeSource:
    """Serves bytes by URL, honours a Range start, counts requests."""

    def __init__(self, blobs: dict[str, bytes], *, honour_range: bool = True):
        self.blobs = blobs
        self.honour_range = honour_range
        self.calls: list[tuple[str, int]] = []

    def __call__(self, url: str, start: int):
        self.calls.append((url, start))
        data = self.blobs[url]
        if start and self.honour_range:
            return True, [data[start:]]
        return False, [data]


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _record(files: dict[str, bytes], *, licence="cc-by-4.0", bad: str | None = None):
    remote = [
        dl.RemoteFile(
            name=name,
            size=len(data),
            checksum=f"md5:{'0' * 32 if name == bad else _md5(data)}",
            url=f"https://example.org/{name}",
        )
        for name, data in files.items()
    ]
    return dl.Record(
        kind="zenodo",
        record_id="123",
        record_url="https://zenodo.org/records/123",
        api_url="https://zenodo.org/api/records/123",
        doi="10.5281/zenodo.123",
        version="1",
        published="2025-07-01",
        title="Test beans",
        creators=["Doe, Jane", "Roe, Rick"],
        publisher="Zenodo",
        licence_stated={"id": licence},
        files=remote,
        raw={"id": 123},
    )


def _cfg(**kw):
    base = {"dataset": "demo", "title": "Demo", "source": {"kind": "zenodo", "record": "123"}}
    return dl.DatasetConfig(**{**base, **kw})


FILES = {"a.zip": b"A" * 5000, "b.zip": b"B" * 3000}


def _run(tmp_path, record, fake, **kw):
    return dl.sync_dataset(
        _cfg(**kw.pop("cfg", {})), record, tmp_path, fetch=fake, log=lambda _m: None, **kw
    )


# --- sync ----------------------------------------------------------------------------


def test_fresh_download_writes_archives_and_provenance(tmp_path):
    fake = FakeSource({f"https://example.org/{n}": b for n, b in FILES.items()})
    doc = _run(tmp_path, _record(FILES), fake, cfg={"expect_licence": "CC-BY-4.0"})

    out = tmp_path / "demo"
    assert (out / "a.zip").read_bytes() == FILES["a.zip"]
    assert not list(out.glob("*.part"))
    saved = json.loads((out / "DOWNLOAD.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(doc))
    assert saved["complete"] is True
    assert saved["licence"]["id"] == "CC-BY-4.0" and saved["licence"]["matches_expected"]
    assert saved["source"]["doi"] == "10.5281/zenodo.123"
    assert saved["attribution"].startswith("Doe, Jane; Roe, Rick (2025). Test beans")
    entry = {e["name"]: e for e in saved["files"]}["b.zip"]
    assert entry["status"] == "verified"
    assert entry["sha256"] == _sha256(FILES["b.zip"])
    assert entry["source_checksum"] == f"md5:{_md5(FILES['b.zip'])}"
    assert json.loads((out / "RECORD.json").read_text(encoding="utf-8")) == {"id": 123}


def test_second_run_downloads_nothing_and_leaves_download_json_alone(tmp_path):
    fake = FakeSource({f"https://example.org/{n}": b for n, b in FILES.items()})
    _run(tmp_path, _record(FILES), fake)
    dj = tmp_path / "demo" / "DOWNLOAD.json"
    before = dj.read_bytes()
    fake.calls.clear()

    _run(tmp_path, _record(FILES), fake)
    assert fake.calls == []
    assert dj.read_bytes() == before


def test_files_already_on_disk_are_adopted_not_downloaded(tmp_path):
    out = tmp_path / "demo"
    out.mkdir()
    for name, data in FILES.items():
        (out / name).write_bytes(data)
    fake = FakeSource({})
    doc = _run(tmp_path, _record(FILES), fake)
    assert fake.calls == []
    assert doc["complete"] is True


def test_checksum_mismatch_fails_and_leaves_no_file(tmp_path):
    fake = FakeSource({f"https://example.org/{n}": b for n, b in FILES.items()})
    doc = _run(tmp_path, _record(FILES, bad="a.zip"), fake)
    entry = {e["name"]: e for e in doc["files"]}["a.zip"]
    assert entry["status"] == "failed"
    assert doc["complete"] is False
    assert not (tmp_path / "demo" / "a.zip").exists()
    assert not (tmp_path / "demo" / "a.zip.part").exists()


def test_interrupted_download_resumes_from_the_part_file(tmp_path):
    data = FILES["a.zip"]
    out = tmp_path / "demo"
    out.mkdir()
    (out / "a.zip.part").write_bytes(data[:2000])
    fake = FakeSource({"https://example.org/a.zip": data})
    doc = _run(tmp_path, _record({"a.zip": data}), fake)
    assert fake.calls == [("https://example.org/a.zip", 2000)]
    assert (out / "a.zip").read_bytes() == data
    assert doc["complete"] is True


def test_server_that_ignores_range_restarts_cleanly(tmp_path):
    data = FILES["a.zip"]
    out = tmp_path / "demo"
    out.mkdir()
    (out / "a.zip.part").write_bytes(data[:2000])
    fake = FakeSource({"https://example.org/a.zip": data}, honour_range=False)
    _run(tmp_path, _record({"a.zip": data}), fake)
    assert (out / "a.zip").read_bytes() == data


def test_no_fetch_records_missing_files_and_downloads_nothing(tmp_path):
    out = tmp_path / "demo"
    out.mkdir()
    (out / "a.zip").write_bytes(FILES["a.zip"])
    fake = FakeSource({})
    doc = _run(tmp_path, _record(FILES), fake, no_fetch=True)
    status = {e["name"]: e["status"] for e in doc["files"]}
    assert status == {"a.zip": "verified", "b.zip": "missing"}
    assert fake.calls == []
    assert doc["complete"] is False


def test_only_selected_files_are_fetched(tmp_path):
    fake = FakeSource({f"https://example.org/{n}": b for n, b in FILES.items()})
    doc = _run(tmp_path, _record(FILES), fake, only=["b.zip"])
    status = {e["name"]: e["status"] for e in doc["files"]}
    assert status == {"a.zip": "not_requested", "b.zip": "verified"}


def test_config_file_list_must_exist_in_the_record(tmp_path):
    with pytest.raises(dl.RecordError, match="does not have"):
        _run(tmp_path, _record(FILES), FakeSource({}), cfg={"files": ["c.zip"]})


def test_a_dataset_on_hold_is_refused_without_files(tmp_path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "big.yaml").write_text(
        "dataset: big\ntitle: Big\nsource: {kind: zenodo, record: '1'}\nhold: too big\n",
        encoding="utf-8",
    )
    code = dl.run(
        "big",
        config_dir=cfg_dir,
        raw_root=tmp_path,
        log=lambda _m: None,
        get_json=lambda url: pytest.fail("the record must not be read"),
    )
    assert code == 2


# --- licences ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        ({"id": "cc-by-4.0"}, "CC-BY-4.0"),
        (
            {"short_name": "CC BY 4.0", "url": "http://creativecommons.org/licenses/by/4.0"},
            "CC-BY-4.0",
        ),
        ({"cardData.license": ["mit"]}, "MIT"),
        (
            {
                "license": {
                    "name": "CC0 1.0",
                    "uri": "http://creativecommons.org/publicdomain/zero/1.0",
                }
            },
            "CC0-1.0",
        ),
        ({"id": "cc-by-4.0", "other": "cc0-1.0"}, "unknown"),
        (None, "unknown"),
    ],
)
def test_licence_normalisation(stated, expected):
    assert dl.normalise_licence(stated) == expected


# --- record parsers (trimmed real API shapes) -----------------------------------------------


def _get(responses):
    return lambda url: responses[url]


def test_zenodo_record():
    api = "https://zenodo.org/api/records/8286126"
    rec = dl.read_record(
        _cfg(source={"kind": "zenodo", "record": "8286126"}),
        _get(
            {
                api: {
                    "doi": "10.5281/zenodo.8286126",
                    "links": {"self_html": "https://zenodo.org/records/8286126"},
                    "metadata": {
                        "title": "Common Beans",
                        "version": "01",
                        "publication_date": "2023-08-26",
                        "creators": [{"name": "Laizer, Hudson"}],
                        "license": {"id": "cc-by-4.0"},
                    },
                    "files": [
                        {
                            "key": "rust.zip",
                            "size": 10,
                            "checksum": "md5:abc",
                            "links": {"self": f"{api}/files/rust.zip/content"},
                        }
                    ],
                    "stats": {"views": 1},
                }
            }
        ),
    )
    assert rec.files[0].url.endswith("/files/rust.zip/content")
    assert rec.files[0].checksum == "md5:abc"
    assert "stats" not in rec.raw
    assert rec.attribution() == (
        "Laizer, Hudson (2023). Common Beans (Version 01) [Data set]. Zenodo. "
        "https://doi.org/10.5281/zenodo.8286126"
    )


def test_huggingface_record_uses_the_pinned_revision():
    repo, rev = "AI-Lab-Makerere/beans", "27aa014"
    rec = dl.read_record(
        _cfg(source={"kind": "huggingface", "repo": repo, "revision": rev, "path": "data"}),
        _get(
            {
                f"https://huggingface.co/api/datasets/{repo}/revision/{rev}": {
                    "author": "AI-Lab-Makerere",
                    "lastModified": "2024-01-03T12:06:51.000Z",
                    "cardData": {"license": ["mit"], "pretty_name": "Beans"},
                    "downloads": 5,
                },
                f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/data": [
                    {
                        "type": "file",
                        "path": "data/test.zip",
                        "size": 7,
                        "lfs": {"oid": "ca67", "size": 7},
                    }
                ],
            }
        ),
    )
    assert rec.files[0].name == "test.zip"
    assert rec.files[0].checksum == "sha256:ca67"
    assert rec.files[0].url == f"https://huggingface.co/datasets/{repo}/resolve/{rev}/data/test.zip"
    assert dl.normalise_licence(rec.licence_stated) == "MIT"
    assert "downloads" not in rec.raw["info"]


def test_mendeley_record_and_version_pin():
    api = "https://data.mendeley.com/public-api/datasets/jxvmr8rchx"
    body = {
        "name": "SWM",
        "version": 1,
        "publish_date": "2026-05-12T15:30:18Z",
        "doi": {"id": "10.17632/jxvmr8rchx.1"},
        "data_licence": {"id": "x", "short_name": "CC BY 4.0"},
        "contributors": [{"first_name": "Rubens", "last_name": "Pereira"}],
        "files": [
            {
                "id": "e0c3",
                "filename": "swm-dataset.zip",
                "metrics": {"downloads": 3},
                "content_details": {
                    "sha256_hash": "3295",
                    "size": 9,
                    "download_url": "https://data.mendeley.com/public-files/x",
                    "download_expiry_time": "2126-01-01",
                },
            }
        ],
        "metrics": {"views": 1},
    }
    cfg = _cfg(source={"kind": "mendeley", "id": "jxvmr8rchx", "version": 1})
    rec = dl.read_record(cfg, _get({api: body}))
    assert rec.files[0].checksum == "sha256:3295"
    assert rec.creators == ["Pereira, Rubens"]
    assert rec.doi == "10.17632/jxvmr8rchx.1"
    assert "download_expiry_time" not in rec.raw["files"][0]["content_details"]
    assert "metrics" not in rec.raw

    with pytest.raises(dl.RecordError, match="pins version 1"):
        dl.read_record(cfg, _get({api: {**body, "version": 2}}))


def test_dataverse_record():
    server, doi = "https://dataverse.harvard.edu", "10.7910/DVN/TCKVEW"
    api = f"{server}/api/datasets/:persistentId/?persistentId=doi:{doi}"
    rec = dl.read_record(
        _cfg(source={"kind": "dataverse", "server": server, "doi": doi}),
        _get(
            {
                api: {
                    "status": "OK",
                    "data": {
                        "latestVersion": {
                            "versionNumber": 2,
                            "versionMinorNumber": 1,
                            "releaseTime": "2022-06-01T00:00:00Z",
                            "license": {
                                "name": "CC0 1.0",
                                "uri": "http://creativecommons.org/publicdomain/zero/1.0",
                            },
                            "metadataBlocks": {
                                "citation": {
                                    "fields": [
                                        {
                                            "typeName": "title",
                                            "value": "Makerere University Beans Image Dataset",
                                        },
                                        {
                                            "typeName": "author",
                                            "value": [
                                                {"authorName": {"value": "Mugalu, Ben-Wycliff"}}
                                            ],
                                        },
                                    ]
                                }
                            },
                            "files": [
                                {
                                    "dataFile": {
                                        "id": 42,
                                        "filename": "beans_1.zip",
                                        "filesize": 11,
                                        "checksum": {"type": "MD5", "value": "ff"},
                                    }
                                }
                            ],
                        }
                    },
                }
            }
        ),
    )
    assert rec.version == "2.1"
    assert rec.files[0].url == f"{server}/api/access/datafile/42"
    assert rec.files[0].checksum == "MD5:ff"
    assert dl.normalise_licence(rec.licence_stated) == "CC0-1.0"


def test_an_unreachable_source_is_a_record_error():
    def down(url):
        raise dl.URLError("504 Gateway Time-out")

    with pytest.raises(dl.RecordError, match="could not read"):
        dl.read_record(_cfg(), down)


# --- the real configs -------------------------------------------------------------------------


SPEC_001_DATASETS = {"tz155k", "tz59k", "tz3", "makerere", "ibean", "swm", "acre"}


def test_every_spec_001_dataset_has_a_valid_config():
    names = {p.stem for p in dl.CONFIG_DIR.glob("*.yaml")}
    assert names == SPEC_001_DATASETS
    for name in names:
        cfg = dl.DatasetConfig.load(name)
        assert cfg.source["kind"] in dl.ADAPTERS
