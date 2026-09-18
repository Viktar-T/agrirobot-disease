"""Hugging Face Hub access check for the two backbones (W1, `make hub-check`).

    python -m ms.backbones.check            # account, gated flag + commit sha per repo, DINOv3 load
    python -m ms.backbones.check --no-load  # the same without downloading / loading DINOv3

Reads HF_TOKEN (and the optional HF_HOME) from .env at the repository root before
huggingface_hub is imported, so the cache location follows HF_HOME. The token is never
printed: huggingface_hub reads it from the environment, and every error message is
redacted before it is shown.

Prints the account from whoami(); for each backbone repo the gated flag and the current
commit sha (HfApi().model_info(repo).sha), compared with the revision pinned in
configs/backbones/<backbone_id>.yaml; then loads DINOv3 with
AutoModel.from_pretrained(repo, revision=<sha>) and prints its class and size.

Exit codes: 0 access works; 1 access refused (no token, 401, 403, gated repo not
granted); 2 anything else (network, repo not found, load error).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: repository root: model_service/src/ms/backbones/check.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "model_service" / "configs" / "backbones"

#: backbone_id -> Hub repo (configs/backbones/<backbone_id>.yaml carry the same hf_id)
BACKBONES = {
    "dinov3_l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov2_l14_reg": "facebook/dinov2-with-registers-large",
}
#: the backbones whose weights the check loads (DINOv3: the gated one)
LOAD = ("dinov3_l16",)

Log = Callable[[str], None]
#: loader(repo, revision) -> the model object
Loader = Callable[[str, str], Any]


def _print(msg: str) -> None:
    print(msg, flush=True)


# --- environment -------------------------------------------------------------------


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Put .env into os.environ (existing variables win). Call before importing
    huggingface_hub: it reads HF_HOME once, at import time."""
    from dotenv import load_dotenv

    load_dotenv(path, override=False)


_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")


def redact(text: str) -> str:
    """Remove the token (the value of HF_TOKEN, or anything shaped like one)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if len(token) >= 8:
        text = text.replace(token, "hf_***")
    return _TOKEN_PATTERN.sub("hf_***", text)


# --- errors ------------------------------------------------------------------------


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def access_problem(exc: BaseException) -> str | None:
    """Why the Hub refused access, if exc (or anything it wraps) is a refusal; else None."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError

    for e in _chain(exc):
        if isinstance(e, LocalTokenNotFoundError):
            return "no token: HF_TOKEN is not set (put it in .env, see .env.example)"
        if isinstance(e, GatedRepoError):
            return (
                "gated repo not granted to this account or token: accept the terms on the "
                "model page with the ML role account, and let the fine-grained token read "
                "the contents of gated repos"
            )
        if isinstance(e, HfHubHTTPError):
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 401:
                return "HTTP 401: the token is missing, invalid or revoked (HF_TOKEN in .env)"
            if status == 403:
                return "HTTP 403: the token lacks the permission for this repo"
    return None


# --- the check ---------------------------------------------------------------------------


@dataclass
class RepoStatus:
    backbone_id: str
    repo: str
    gated: Any  # False, or "auto" / "manual" as the Hub states it
    sha: str
    pinned: str | None


def pinned_revision(backbone_id: str, config_dir: Path = CONFIG_DIR) -> str | None:
    path = config_dir / f"{backbone_id}.yaml"
    if not path.exists():
        return None
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("revision")


def format_account(who: dict[str, Any]) -> str:
    token = (who.get("auth") or {}).get("accessToken") or {}
    name, role = token.get("displayName"), token.get("role")
    detail = f' (token "{name}", {role})' if name and role else ""
    return f"  account   {who.get('name', '?')}{detail}"


def format_cache(hub_cache: str, hf_home: str | None) -> str:
    where = f"HF_HOME={hf_home}" if hf_home else "HF_HOME not set: default location"
    return f"  cache     {hub_cache}  ({where})"


def format_repo(s: RepoStatus) -> str:
    if s.pinned is None:
        pin = "no config pin yet"
    elif s.pinned == s.sha:
        pin = "= pinned revision"
    else:
        pin = f"DIFFERS from pinned revision {s.pinned}"
    return f"  {s.backbone_id:15s} {s.repo:41s} gated={s.gated!s:7s} sha={s.sha}  {pin}"


def format_load(backbone_id: str, sha: str, model: Any, seconds: float) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    return (
        f"  load      {backbone_id} @ {sha[:12]}: {type(model).__name__}, "
        f"{n_params / 1e6:.1f} M parameters ({seconds:.1f} s)"
    )


def _load_model(repo: str, revision: str) -> Any:
    from transformers import AutoModel

    return AutoModel.from_pretrained(repo, revision=revision)


def run(
    api: Any,
    *,
    backbones: dict[str, str] = BACKBONES,
    load: tuple[str, ...] = LOAD,
    loader: Loader = _load_model,
    hub_cache: str,
    hf_home: str | None,
    config_dir: Path = CONFIG_DIR,
    log: Log = _print,
) -> int:
    """The check itself; api is an HfApi (or a stand-in with whoami and model_info)."""
    log("Hugging Face Hub access (ms.backbones.check)")
    try:
        log(format_account(api.whoami()))
        log(format_cache(hub_cache, hf_home))
        found: dict[str, RepoStatus] = {}
        for backbone_id, repo in backbones.items():
            info = api.model_info(repo)
            found[backbone_id] = RepoStatus(
                backbone_id, repo, info.gated, info.sha, pinned_revision(backbone_id, config_dir)
            )
            log(format_repo(found[backbone_id]))
        for backbone_id in load:
            s = found[backbone_id]
            t0 = time.perf_counter()
            model = loader(s.repo, s.sha)
            log(format_load(backbone_id, s.sha, model, time.perf_counter() - t0))
    except Exception as exc:  # noqa: BLE001 - every failure becomes one redacted line
        why = access_problem(exc)
        if why:
            log(f"ACCESS REFUSED: {why}")
            log(f"  ({redact(type(exc).__name__ + ': ' + str(exc)).splitlines()[0]})")
            return 1
        log(f"FAILED: {redact(type(exc).__name__ + ': ' + str(exc))}")
        return 2
    moved = [s.backbone_id for s in found.values() if s.pinned and s.pinned != s.sha]
    if moved:
        log(f"ok, but the Hub moved past the pin for {', '.join(moved)} (configs keep the pin)")
    else:
        log("ok")
    return 0


# --- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ms.backbones.check", description=__doc__.split("\n")[0]
    )
    p.add_argument("--no-load", action="store_true", help="skip downloading / loading DINOv3")
    args = p.parse_args(argv)

    load_env()
    from huggingface_hub import HfApi, constants

    return run(
        HfApi(),
        load=() if args.no_load else LOAD,
        hub_cache=str(constants.HF_HUB_CACHE),
        hf_home=os.environ.get("HF_HOME"),
    )


if __name__ == "__main__":
    sys.exit(main())
