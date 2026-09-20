"""Backbone configs (DECISIONS 11-13) and the Hub access check. No network, no weights:
the Hub is a stand-in with whoami() and model_info(), the loader a stand-in model."""

from __future__ import annotations

import re
from types import SimpleNamespace

import httpx
import pytest
import yaml
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from ms.backbones import check

SHA = re.compile(r"^[0-9a-f]{40}$")
KEYS = {
    "backbone_id",
    "hf_id",
    "revision",
    "licence",
    "gated",
    "research_only_until_c5",
    "patch_size",
    "embed_dim",
    "resolutions",
    "preprocess_string",
    "dtype",
    "attn_implementation",
    "batch_size",
    "token_types",
}
#: only a gated backbone carries it: H8 §6.8 asks the model card to state the date its terms
#: were accepted, and a card renders what an artefact holds (DECISIONS 128)
GATED_KEYS = {"terms_accepted"}
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _config(backbone_id):
    path = check.CONFIG_DIR / f"{backbone_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- configs/backbones/*.yaml ---------------------------------------------------------


@pytest.mark.parametrize("backbone_id", sorted(check.BACKBONES))
def test_config_parses_and_pins_a_commit(backbone_id):
    cfg = _config(backbone_id)
    assert set(cfg) - GATED_KEYS == KEYS
    assert cfg["backbone_id"] == backbone_id
    assert cfg["hf_id"] == check.BACKBONES[backbone_id]
    assert SHA.match(cfg["revision"]), "revision must be a commit sha, never a branch"
    assert cfg["embed_dim"] == 1024
    assert cfg["dtype"] == "float16" and cfg["attn_implementation"] == "sdpa"
    assert cfg["token_types"] == ["cls", "meanpatch"]


@pytest.mark.parametrize("backbone_id", sorted(check.BACKBONES))
def test_one_distinct_preprocess_string_per_resolution(backbone_id):
    cfg = _config(backbone_id)
    pre = cfg["preprocess_string"]
    assert sorted(pre) == sorted(cfg["resolutions"])
    assert len(set(pre.values())) == len(pre)
    for res, s in pre.items():
        assert res % cfg["patch_size"] == 0
        assert f"={res};" in s and s.endswith("norm=imagenet")


def test_only_dinov3_is_gated_and_research_only():
    v2, v3 = _config("dinov2_l14_reg"), _config("dinov3_l16")
    assert (v2["gated"], v2["research_only_until_c5"], v2["licence"]) == (
        False,
        False,
        "Apache-2.0",
    )
    assert (v3["gated"], v3["research_only_until_c5"]) == (True, True)
    assert v3["licence"].startswith("DINOv3 License")
    # a gated backbone records when its terms were accepted, because the card states it
    assert "terms_accepted" not in v2
    accepted = v3["terms_accepted"]
    assert set(accepted) == {"date", "account", "scope"}
    assert DATE.fullmatch(str(accepted["date"])), accepted["date"]
    assert accepted["account"] and "research" in accepted["scope"]
    assert (v2["patch_size"], v2["resolutions"]) == (14, [224, 518])
    assert (v3["patch_size"], v3["resolutions"]) == (16, [224, 512])


# --- the check ----------------------------------------------------------------------------

TOKEN = "hf_" + "Q" * 34
V3, V2 = "a" * 40, "b" * 40


class FakeApi:
    def __init__(self, *, whoami_error=None, info_error=None):
        self.whoami_error, self.info_error = whoami_error, info_error

    def whoami(self):
        if self.whoami_error:
            raise self.whoami_error
        return {
            "name": "ml-role",
            "auth": {"accessToken": {"displayName": "DINOv3-token", "role": "fineGrained"}},
        }

    def model_info(self, repo):
        if self.info_error:
            raise self.info_error
        if "dinov3" in repo:
            return SimpleNamespace(sha=V3, gated="manual")
        return SimpleNamespace(sha=V2, gated=False)


class DINOv3ViTModel:
    def parameters(self):
        return [
            SimpleNamespace(numel=lambda: 300_000_000),
            SimpleNamespace(numel=lambda: 3_100_000),
        ]


def _http_error(cls, status, message):
    request = httpx.Request("GET", "https://huggingface.co/api/whoami-v2")
    return cls(message, response=httpx.Response(status, request=request))


def _run(tmp_path, api, loader=None, pins=None):
    for backbone_id, rev in (pins or {}).items():
        (tmp_path / f"{backbone_id}.yaml").write_text(f"revision: '{rev}'\n", encoding="utf-8")
    lines: list[str] = []
    loaded: list[tuple[str, str]] = []

    def default_loader(repo, revision):
        loaded.append((repo, revision))
        return DINOv3ViTModel()

    code = check.run(
        api,
        loader=loader or default_loader,
        hub_cache="D:/hf/hub",
        hf_home="D:/hf",
        config_dir=tmp_path,
        log=lines.append,
    )
    return code, "\n".join(lines), loaded


def test_report_names_account_cache_gated_shas_and_the_loaded_model(tmp_path):
    code, out, loaded = _run(tmp_path, FakeApi(), pins={"dinov3_l16": V3})
    assert code == 0
    assert 'account   ml-role (token "DINOv3-token", fineGrained)' in out
    assert "cache     D:/hf/hub  (HF_HOME=D:/hf)" in out
    assert re.search(rf"dinov3_l16 .* gated=manual +sha={V3}  = pinned revision", out)
    assert re.search(rf"dinov2_l14_reg .* gated=False +sha={V2}  no config pin yet", out)
    assert f"load      dinov3_l16 @ {V3[:12]}: DINOv3ViTModel, 303.1 M parameters" in out
    assert out.endswith("ok")
    # DINOv3 is loaded at the sha the Hub reported, never at "main"
    assert loaded == [(check.BACKBONES["dinov3_l16"], V3)]


def test_a_moved_hub_repo_is_reported_but_not_an_error(tmp_path):
    code, out, _ = _run(tmp_path, FakeApi(), pins={"dinov2_l14_reg": "c" * 40})
    assert code == 0
    assert f"DIFFERS from pinned revision {'c' * 40}" in out
    assert "moved past the pin for dinov2_l14_reg" in out


def test_a_gated_refusal_exits_1_without_echoing_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    def refused(repo, revision):
        gated = _http_error(GatedRepoError, 403, f"Access to {repo} is restricted ({TOKEN})")
        # transformers wraps Hub errors in an OSError
        raise OSError(f"You are trying to access a gated repo. token={TOKEN}") from gated

    code, out, _ = _run(tmp_path, FakeApi(), loader=refused)
    assert code == 1
    assert "ACCESS REFUSED: gated repo not granted" in out
    assert TOKEN not in out and "hf_***" in out


@pytest.mark.parametrize(("status", "why"), [(401, "HTTP 401"), (403, "HTTP 403")])
def test_401_and_403_exit_1(tmp_path, status, why):
    api = FakeApi(whoami_error=_http_error(HfHubHTTPError, status, "Invalid user token."))
    code, out, loaded = _run(tmp_path, api)
    assert code == 1
    assert f"ACCESS REFUSED: {why}" in out
    assert loaded == []


def test_other_failures_exit_2(tmp_path):
    code, out, _ = _run(tmp_path, FakeApi(info_error=ConnectionError("no route to host")))
    assert code == 2
    assert out.splitlines()[-1] == "FAILED: ConnectionError: no route to host"


def test_redact_removes_the_token_and_anything_shaped_like_one(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert check.redact(f"Bearer {TOKEN}") == "Bearer hf_***"
    assert check.redact("header hf_abcdefghijkl123") == "header hf_***"
    assert check.redact("nothing secret") == "nothing secret"
