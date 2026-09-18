"""Raw dataset downloads (W1 "Downloads"): data/raw/<dataset>/ plus DOWNLOAD.json.

    python -m ms.data.download --dataset tz155k             # fetch what is missing, verify all
    python -m ms.data.download --dataset makerere --no-fetch  # adopt files placed by hand
    python -m ms.data.download --dataset tz3 --files RUST_6.zip
    python -m ms.data.download --list

One config per dataset, model_service/configs/datasets/<id>.yaml, names the source record
(Zenodo, Hugging Face, Mendeley Data or Dataverse). The record is read live from the
source's API. Every archive is checked against the checksum the source publishes and
hashed with sha256. Archives land in data/raw/<id>/ untouched; next to them go
RECORD.json (the record as read, minus counters that change on every read) and
DOWNLOAD.json (the provenance spec 001 manifests read: source_record, source_version,
licence, attribution).

Idempotent: a file whose size and mtime match its DOWNLOAD.json entry is not hashed
again (--verify forces it), a verified file is never downloaded again, an interrupted
download resumes from its .part file, and an unchanged DOWNLOAD.json is not rewritten.

Exit codes: 0 every requested file verified; 1 something missing or failed;
2 configuration, hold or record error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

import ms

#: repository root: model_service/src/ms/data/download.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "model_service" / "configs" / "datasets"
RAW_ROOT = REPO_ROOT / "data" / "raw"

DOWNLOAD_JSON = "DOWNLOAD.json"
RECORD_JSON = "RECORD.json"
DOWNLOAD_JSON_VERSION = 1
USER_AGENT = f"agrirobot-disease-ms/{ms.__version__} (dataset download for research)"
CHUNK = 1 << 20

#: licence vocabulary of spec 001 FR-003
LICENCES = ("CC-BY-4.0", "CC0-1.0", "MIT", "unknown")

GetJson = Callable[[str], Any]
#: fetch(url, start) -> (resumed, chunks): resumed is False when the server ignored Range
Fetch = Callable[[str, int], tuple[bool, Iterable[bytes]]]


def _print(msg: str) -> None:
    print(msg, flush=True)


class ConfigError(Exception):
    """A dataset config is missing, malformed or on hold."""


class RecordError(Exception):
    """The source record could not be read or does not match the config."""


# --- configs ---------------------------------------------------------------


@dataclass
class DatasetConfig:
    dataset: str
    title: str
    source: dict[str, Any]
    files: list[str] | None = None
    expect_licence: str | None = None
    attribution: str | None = None
    hold: str | None = None
    notes: str | None = None

    @classmethod
    def load(cls, dataset: str, config_dir: Path = CONFIG_DIR) -> DatasetConfig:
        path = config_dir / f"{dataset}.yaml"
        if not path.exists():
            known = ", ".join(sorted(p.stem for p in config_dir.glob("*.yaml")))
            raise ConfigError(f"no config {path} (known datasets: {known})")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"{path.name}: unknown keys {sorted(unknown)}")
        missing = {"dataset", "title", "source"} - set(raw)
        if missing:
            raise ConfigError(f"{path.name}: missing keys {sorted(missing)}")
        if raw["dataset"] != dataset:
            raise ConfigError(f"{path.name}: dataset is {raw['dataset']!r}, expected {dataset!r}")
        if raw["source"].get("kind") not in ADAPTERS:
            raise ConfigError(f"{path.name}: source.kind must be one of {sorted(ADAPTERS)}")
        if raw.get("expect_licence") not in (None, *LICENCES):
            raise ConfigError(f"{path.name}: expect_licence must be one of {LICENCES}")
        return cls(**raw)


# --- source records ----------------------------------------------------------


@dataclass
class RemoteFile:
    name: str
    size: int | None
    checksum: str | None  # "<algo>:<hex>" as the source publishes it, or None
    url: str


@dataclass
class Record:
    kind: str
    record_id: str
    record_url: str
    api_url: str
    doi: str | None
    version: str | None
    published: str | None
    title: str | None
    creators: list[str]
    publisher: str
    licence_stated: Any
    files: list[RemoteFile]
    raw: Any = field(repr=False, default=None)

    def attribution(self) -> str:
        year = (self.published or "")[:4] or "n.d."
        who = "; ".join(self.creators) or self.publisher
        version = f" (Version {self.version})" if self.version else ""
        where = f"https://doi.org/{self.doi}" if self.doi else self.record_url
        return f"{who} ({year}). {self.title}{version} [Data set]. {self.publisher}. {where}"


def _zenodo(cfg: DatasetConfig, get_json: GetJson) -> Record:
    rec = str(cfg.source["record"])
    api = f"https://zenodo.org/api/records/{rec}"
    d = get_json(api)
    m = d["metadata"]
    files = [
        RemoteFile(f["key"], f.get("size"), f.get("checksum"), f["links"]["self"])
        for f in d["files"]
    ]
    return Record(
        kind="zenodo",
        record_id=rec,
        record_url=d.get("links", {}).get("self_html") or f"https://zenodo.org/records/{rec}",
        api_url=api,
        doi=d.get("doi"),
        version=m.get("version"),
        published=m.get("publication_date"),
        title=m.get("title"),
        creators=[c["name"] for c in m.get("creators", [])],
        publisher="Zenodo",
        licence_stated=m.get("license"),
        files=files,
        raw=_without(d, "stats"),
    )


def _huggingface(cfg: DatasetConfig, get_json: GetJson) -> Record:
    repo, rev = cfg.source["repo"], cfg.source["revision"]
    sub = cfg.source.get("path", "").strip("/")
    info_api = f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"
    tree_api = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{sub}".rstrip("/")
    info, tree = get_json(info_api), get_json(tree_api)
    files = [
        RemoteFile(
            name=t["path"].rsplit("/", 1)[-1],
            size=t.get("size"),
            checksum=f"sha256:{t['lfs']['oid']}" if t.get("lfs") else None,
            url=f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{t['path']}",
        )
        for t in tree
        if t.get("type") == "file"
    ]
    card = info.get("cardData") or {}
    return Record(
        kind="huggingface",
        record_id=f"{repo}@{rev}",
        record_url=f"https://huggingface.co/datasets/{repo}/tree/{rev}/{sub}".rstrip("/"),
        api_url=tree_api,
        doi=None,
        version=rev,
        published=info.get("lastModified"),
        title=card.get("pretty_name") or cfg.title,
        creators=[info["author"]] if info.get("author") else [],
        publisher="Hugging Face",
        licence_stated={"cardData.license": card.get("license")},
        files=files,
        raw={"info": _without(info, "downloads", "likes", "usedStorage"), "tree": tree},
    )


def _mendeley(cfg: DatasetConfig, get_json: GetJson) -> Record:
    ds, pinned = cfg.source["id"], int(cfg.source["version"])
    api = f"https://data.mendeley.com/public-api/datasets/{ds}"
    d = get_json(api)
    if int(d.get("version", -1)) != pinned:
        raise RecordError(
            f"Mendeley {ds}: config pins version {pinned}, record serves {d.get('version')}"
        )
    files = []
    for f in d["files"]:
        cd = f.get("content_details") or {}
        files.append(
            RemoteFile(
                name=f["filename"],
                size=cd.get("size") or f.get("size"),
                checksum=f"sha256:{cd['sha256_hash']}" if cd.get("sha256_hash") else None,
                url=cd.get("download_url")
                or f"https://data.mendeley.com/public-files/datasets/{ds}/files/{f['id']}/file_downloaded",
            )
        )
    licence = {k: v for k, v in (d.get("data_licence") or {}).items() if k != "id"}
    return Record(
        kind="mendeley",
        record_id=f"{ds}.{pinned}",
        record_url=f"https://data.mendeley.com/datasets/{ds}/{pinned}",
        api_url=api,
        doi=(d.get("doi") or {}).get("id"),
        version=str(pinned),
        published=d.get("publish_date"),
        title=d.get("name"),
        creators=[
            f"{c.get('last_name', '')}, {c.get('first_name', '')}".strip(", ")
            for c in d.get("contributors", [])
        ],
        publisher="Mendeley Data",
        licence_stated=licence,
        files=files,
        raw=_mendeley_stable(d),
    )


def _dataverse(cfg: DatasetConfig, get_json: GetJson) -> Record:
    server = cfg.source.get("server", "https://dataverse.harvard.edu").rstrip("/")
    doi = cfg.source["doi"]
    api = f"{server}/api/datasets/:persistentId/?persistentId=doi:{doi}"
    d = get_json(api)
    if d.get("status") != "OK":
        raise RecordError(f"Dataverse {doi}: status {d.get('status')}: {d.get('message')}")
    v = d["data"]["latestVersion"]
    fields = {f["typeName"]: f["value"] for f in v["metadataBlocks"]["citation"]["fields"]}
    files = []
    for f in v.get("files", []):
        df = f["dataFile"]
        ck = df.get("checksum") or {}
        if ck.get("type") and ck.get("value"):
            checksum = f"{ck['type']}:{ck['value']}"
        else:
            checksum = f"md5:{df['md5']}" if df.get("md5") else None
        files.append(
            RemoteFile(
                df["filename"],
                df.get("filesize"),
                checksum,
                f"{server}/api/access/datafile/{df['id']}",
            )
        )
    return Record(
        kind="dataverse",
        record_id=f"doi:{doi}",
        record_url=f"{server}/dataset.xhtml?persistentId=doi:{doi}",
        api_url=api,
        doi=doi,
        version=f"{v.get('versionNumber')}.{v.get('versionMinorNumber')}",
        published=v.get("releaseTime") or d["data"].get("publicationDate"),
        title=fields.get("title"),
        creators=[a["authorName"]["value"] for a in fields.get("author", [])],
        publisher="Harvard Dataverse",
        licence_stated={"license": v.get("license"), "termsOfUse": v.get("termsOfUse")},
        files=files,
        raw=d,
    )


ADAPTERS: dict[str, Callable[[DatasetConfig, GetJson], Record]] = {
    "zenodo": _zenodo,
    "huggingface": _huggingface,
    "mendeley": _mendeley,
    "dataverse": _dataverse,
}


def read_record(cfg: DatasetConfig, get_json: GetJson) -> Record:
    try:
        return ADAPTERS[cfg.source["kind"]](cfg, get_json)
    except RecordError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
        raise RecordError(f"{cfg.dataset}: could not read the source record: {exc!r}") from exc


def _without(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    """A copy without counters that change on every read (keeps RECORD.json stable)."""
    return {k: v for k, v in d.items() if k not in keys}


def _mendeley_stable(d: dict[str, Any]) -> dict[str, Any]:
    out = _without(d, "metrics")
    out["files"] = [
        {
            **_without(f, "metrics"),
            "content_details": _without(f.get("content_details") or {}, "download_expiry_time"),
        }
        for f in d.get("files", [])
    ]
    return out


# --- licences ------------------------------------------------------------------

_LICENCE_ALIASES = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc by 4.0": "CC-BY-4.0",
    "creative commons attribution 4.0 international": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
    "cc0 1.0": "CC0-1.0",
    "cc0": "CC0-1.0",
    "creative commons zero v1.0 universal": "CC0-1.0",
    "mit": "MIT",
}
_LICENCE_URLS = {
    "creativecommons.org/licenses/by/4.0": "CC-BY-4.0",
    "creativecommons.org/publicdomain/zero/1.0": "CC0-1.0",
}


def normalise_licence(stated: Any) -> str:
    """Map whatever the record states onto the spec 001 vocabulary; else "unknown"."""
    found: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            s = x.strip().lower()
            if s in _LICENCE_ALIASES:
                found.append(_LICENCE_ALIASES[s])
            for url, lic in _LICENCE_URLS.items():
                if url in s:
                    found.append(lic)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(stated)
    distinct = sorted(set(found))
    return distinct[0] if len(distinct) == 1 else "unknown"


# --- HTTP ------------------------------------------------------------------------


def _get_json(url: str, *, attempts: int = 3, timeout: float = 90.0) -> Any:
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                raise
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(5 * (i + 1))
    assert last is not None
    raise last


def _urllib_fetch(url: str, start: int) -> tuple[bool, Iterator[bytes]]:
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    resp = urlopen(Request(url, headers=headers), timeout=120)
    resumed = start > 0 and resp.status == 206

    def chunks() -> Iterator[bytes]:
        with resp:
            while block := resp.read(CHUNK):
                yield block

    return resumed, chunks()


# --- files -------------------------------------------------------------------------


def _algo(checksum: str | None) -> str | None:
    if not checksum:
        return None
    return checksum.split(":", 1)[0].lower().replace("-", "")


def hash_file(path: Path, algos: Iterable[str]) -> dict[str, str]:
    """Hex digests of one file for several algorithms, in one read."""
    hashers = {a: hashlib.new(a) for a in dict.fromkeys(algos)}
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            for h in hashers.values():
                h.update(block)
    return {a: h.hexdigest() for a, h in hashers.items()}


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check(path: Path, rf: RemoteFile) -> tuple[str, str, bool | None]:
    """(sha256, source-algorithm digest, matches the source checksum or None if none)."""
    algo = _algo(rf.checksum)
    digests = hash_file(path, ["sha256", *([algo] if algo else [])])
    if not algo:
        return digests["sha256"], "", None
    want = rf.checksum.split(":", 1)[1].lower()  # type: ignore[union-attr]
    return digests["sha256"], digests[algo], digests[algo] == want


def _entry(rf: RemoteFile, path: Path, sha256: str, status: str) -> dict[str, Any]:
    st = path.stat()
    return {
        "name": rf.name,
        "size": st.st_size,
        "url": rf.url,
        "source_checksum": rf.checksum,
        "sha256": sha256,
        "status": status,
        "downloaded_at": _mtime_iso(path),
        "local_mtime_ns": st.st_mtime_ns,
    }


def sync_file(
    rf: RemoteFile,
    dest: Path,
    *,
    previous: dict[str, Any] | None,
    fetch: Fetch,
    no_fetch: bool,
    verify: bool,
    attempts: int = 3,
    log: Callable[[str], None] = _print,
) -> dict[str, Any]:
    """Make one archive present and verified; return its DOWNLOAD.json entry."""
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        st = dest.stat()
        if (
            not verify
            and previous
            and previous.get("status") in ("verified", "unverified")
            and previous.get("size") == st.st_size
            and previous.get("local_mtime_ns") == st.st_mtime_ns
            and previous.get("source_checksum") == rf.checksum
        ):
            log(f"  kept      {rf.name} (unchanged since last check)")
            return previous
        sha256, _, ok = _check(dest, rf)
        if ok is not False:
            status = "verified" if ok else "unverified"
            log(f"  {status:9s} {rf.name} ({st.st_size:,} bytes)")
            return _entry(rf, dest, sha256, status)
        if rf.size and st.st_size < rf.size and not no_fetch:
            log(f"  partial   {rf.name}: {st.st_size:,} of {rf.size:,} bytes - resuming")
            dest.replace(part)
        else:
            log(f"  MISMATCH  {rf.name}: {_algo(rf.checksum)} differs from the source")
            return {
                "name": rf.name,
                "size": st.st_size,
                "url": rf.url,
                "source_checksum": rf.checksum,
                "sha256": sha256,
                "status": "failed",
            }

    if no_fetch:
        log(f"  missing   {rf.name}")
        return {
            "name": rf.name,
            "size": rf.size,
            "url": rf.url,
            "source_checksum": rf.checksum,
            "sha256": None,
            "status": "missing",
        }

    for attempt in range(1, attempts + 1):
        start = part.stat().st_size if part.exists() else 0
        if rf.size and start > rf.size:
            part.unlink()
            start = 0
        log(
            f"  fetching  {rf.name} (attempt {attempt}"
            + (f", from byte {start:,})" if start else ")")
        )
        try:
            resumed, chunks = fetch(rf.url, start)
            with part.open("ab" if resumed else "wb") as fh:
                for block in chunks:
                    fh.write(block)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log(f"  retry     {rf.name}: {exc!r}")
            time.sleep(10 * attempt)
            continue
        if rf.size and part.stat().st_size < rf.size:
            log(f"  retry     {rf.name}: stopped at {part.stat().st_size:,} of {rf.size:,} bytes")
            continue
        sha256, _, ok = _check(part, rf)
        if ok is False:
            log(f"  retry     {rf.name}: {_algo(rf.checksum)} differs from the source; restarting")
            part.unlink()
            continue
        part.replace(dest)
        status = "verified" if ok else "unverified"
        log(f"  {status:9s} {rf.name} ({dest.stat().st_size:,} bytes, downloaded)")
        return _entry(rf, dest, sha256, status)

    log(f"  FAILED    {rf.name}")
    return {
        "name": rf.name,
        "size": rf.size,
        "url": rf.url,
        "source_checksum": rf.checksum,
        "sha256": None,
        "status": "failed",
    }


# --- one dataset ------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def _write_json_if_changed(path: Path, doc: Any, *, volatile: tuple[str, ...] = ()) -> bool:
    """Write doc unless the file already holds the same content (ignoring volatile keys)."""
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = None
        if isinstance(old, dict) and isinstance(doc, dict):
            same = {k: v for k, v in old.items() if k not in volatile} == {
                k: v for k, v in doc.items() if k not in volatile
            }
        else:
            same = old == doc
        if same:
            return False
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return True


def sync_dataset(
    cfg: DatasetConfig,
    record: Record,
    raw_root: Path = RAW_ROOT,
    *,
    fetch: Fetch = _urllib_fetch,
    no_fetch: bool = False,
    verify: bool = False,
    only: list[str] | None = None,
    log: Callable[[str], None] = _print,
) -> dict[str, Any]:
    """Bring data/raw/<dataset>/ in line with the record; write RECORD.json and DOWNLOAD.json."""
    out = raw_root / cfg.dataset
    out.mkdir(parents=True, exist_ok=True)

    wanted = cfg.files if cfg.files is not None else [f.name for f in record.files]
    by_name = {f.name: f for f in record.files}
    absent = [n for n in wanted if n not in by_name]
    if absent:
        raise RecordError(f"{cfg.dataset}: config lists files the record does not have: {absent}")
    if only:
        stray = [n for n in only if n not in wanted]
        if stray:
            raise ConfigError(f"{cfg.dataset}: --files not in this dataset: {stray}")

    previous: dict[str, dict[str, Any]] = {}
    dj = out / DOWNLOAD_JSON
    if dj.exists():
        try:
            previous = {e["name"]: e for e in json.loads(dj.read_text(encoding="utf-8"))["files"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            previous = {}

    log(f"{cfg.dataset}: {record.kind} {record.record_id} - {len(wanted)} file(s) -> {out}")
    entries = []
    for name in wanted:
        rf = by_name[name]
        dest = out / name
        if only and name not in only and not dest.exists():
            entries.append(
                {
                    "name": name,
                    "size": rf.size,
                    "url": rf.url,
                    "source_checksum": rf.checksum,
                    "sha256": None,
                    "status": "not_requested",
                }
            )
            continue
        entries.append(
            sync_file(
                rf,
                dest,
                previous=previous.get(name),
                fetch=fetch,
                no_fetch=no_fetch,
                verify=verify,
                log=log,
            )
        )

    licence_id = normalise_licence(record.licence_stated)
    if cfg.expect_licence and licence_id != cfg.expect_licence:
        log(f"  WARNING   licence: record states {licence_id}, config expects {cfg.expect_licence}")

    doc = {
        "download_json_version": DOWNLOAD_JSON_VERSION,
        "dataset": cfg.dataset,
        "title": record.title or cfg.title,
        "source": {
            "kind": record.kind,
            "record_id": record.record_id,
            "record_url": record.record_url,
            "api_url": record.api_url,
            "doi": record.doi,
            "version": record.version,
            "published": record.published,
            "publisher": record.publisher,
        },
        "licence": {
            "id": licence_id,
            "expected": cfg.expect_licence,
            "matches_expected": None
            if not cfg.expect_licence
            else licence_id == cfg.expect_licence,
            "as_stated": record.licence_stated,
        },
        "attribution": cfg.attribution or record.attribution(),
        "creators": record.creators,
        "complete": all(e["status"] in ("verified", "unverified") for e in entries),
        "files": entries,
        "notes": cfg.notes,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": {"module": "ms.data.download", "version": ms.__version__, "git_sha": _git_sha()},
    }
    _write_json_if_changed(out / RECORD_JSON, record.raw)
    changed = _write_json_if_changed(dj, doc, volatile=("checked_at", "builder"))
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    log(
        f"  licence {licence_id}; {', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}; "
        f"{DOWNLOAD_JSON} {'written' if changed else 'unchanged'}"
    )
    return doc


# --- CLI ------------------------------------------------------------------------------------


def run(
    dataset: str,
    *,
    config_dir: Path = CONFIG_DIR,
    raw_root: Path = RAW_ROOT,
    get_json: GetJson = _get_json,
    fetch: Fetch = _urllib_fetch,
    no_fetch: bool = False,
    verify: bool = False,
    only: list[str] | None = None,
    ignore_hold: bool = False,
    log: Callable[[str], None] = _print,
) -> int:
    try:
        cfg = DatasetConfig.load(dataset, config_dir)
        if cfg.hold and not (only or ignore_hold or no_fetch):
            raise ConfigError(f"{dataset} is on hold: {cfg.hold.strip()}")
        record = read_record(cfg, get_json)
        doc = sync_dataset(
            cfg, record, raw_root, fetch=fetch, no_fetch=no_fetch, verify=verify, only=only, log=log
        )
    except (ConfigError, RecordError) as exc:
        log(f"error: {exc}")
        return 2
    requested = [f for f in doc["files"] if f["status"] != "not_requested"]
    return 0 if all(f["status"] in ("verified", "unverified") for f in requested) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ms.data.download", description=__doc__.split("\n")[0]
    )
    p.add_argument("--dataset", help="dataset id (a file in configs/datasets/)")
    p.add_argument("--list", action="store_true", help="list the configured datasets and exit")
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="download nothing; verify and record what is on disk",
    )
    p.add_argument("--verify", action="store_true", help="re-hash every file even if unchanged")
    p.add_argument("--files", nargs="+", metavar="NAME", help="only these archives")
    p.add_argument("--ignore-hold", action="store_true", help="download a dataset that is on hold")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = p.parse_args(argv)

    if args.list:
        for path in sorted(args.config_dir.glob("*.yaml")):
            cfg = DatasetConfig.load(path.stem, args.config_dir)
            state = "on hold" if cfg.hold else "ready"
            have = (
                "DOWNLOAD.json" if (args.raw_root / cfg.dataset / DOWNLOAD_JSON).exists() else "-"
            )
            print(f"{cfg.dataset:10s} {cfg.source['kind']:12s} {state:8s} {have:14s} {cfg.title}")
        return 0
    if not args.dataset:
        p.error("--dataset is required (or --list)")
    return run(
        args.dataset,
        config_dir=args.config_dir,
        raw_root=args.raw_root,
        no_fetch=args.no_fetch,
        verify=args.verify,
        only=args.files,
        ignore_hold=args.ignore_hold,
    )


if __name__ == "__main__":
    sys.exit(main())
