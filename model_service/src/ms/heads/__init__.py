"""Heads on cached features (S4.3; specs/003-heads/spec.md).

    python -m ms.heads.train --backbone dinov2_l14_reg --res 224 \\
        --train-manifest data/manifests/makerere_v1.jsonl   # linear, proto, mix x seeds 0-4

A run is data/heads/<run_id>/ (git-ignored): head.pt, the weights, and run.json, with the
inputs and their hashes, the fit and the model_version string (H8 6.2). Features are
L2-normalised per token type; "cls+meanpatch" concatenates the two (the W4 ablation).

The heads (spec 003 US-2): `linear`, one affine map (multinomial logistic regression);
`proto`, K prototypes per class, a class scoring the cosine of its nearest prototype over a
learnable temperature; `mix`, the fixed 0.5/0.5 mixture of the linear and proto heads'
probabilities, whose logits are the log of the mixture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ms.cache import Cache

#: repository root: model_service/src/ms/heads/__init__.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
HEADS_ROOT = REPO_ROOT / "data" / "heads"
HEAD_CONFIGS = REPO_ROOT / "model_service" / "configs" / "heads"
TOKEN_SPECS = ("cls", "meanpatch", "cls+meanpatch")
#: the service version at the head of model_version (work plan W5)
SERVICE_VERSION = "msv0.1"


def model_version(backbone_id: str, res: int, head_id: str, train_manifest_sha256: str) -> str:
    """msv0.1+<backbone_id>@<res>.<head_id>.man-<train manifest sha256[:6]> (work plan W5);
    several training manifests (sha256s joined with +) give their [:6] joined with -."""
    man = "-".join(sha[:6] for sha in train_manifest_sha256.split("+"))
    return f"{SERVICE_VERSION}+{backbone_id}@{res}.{head_id}.man-{man}"


def features_from_tokens(tokens: str, **by_type: np.ndarray) -> np.ndarray:
    """float32 [N, D x token types]: each token type L2-normalised, then joined in the order
    `tokens` names them. The service computes a frame's tokens itself (spec 006 US-4.2) and
    a cached row reads them from the .npz; both come here, so one frame answered either way
    reaches the head as the same vector."""
    if tokens not in TOKEN_SPECS:
        raise ValueError(f"tokens must be one of {TOKEN_SPECS}, got {tokens!r}")
    parts = []
    for token in tokens.split("+"):
        if token not in by_type:
            raise ValueError(f"tokens {tokens!r} needs {token}, which was not given")
        x = np.asarray(by_type[token], dtype=np.float32)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        parts.append(x)
    return np.concatenate(parts, axis=1)


def features(cache: Cache, index: np.ndarray, tokens: str) -> np.ndarray:
    """float32 [len(index), D x token types]: each token type L2-normalised, then joined."""
    if tokens not in TOKEN_SPECS:
        raise ValueError(f"tokens must be one of {TOKEN_SPECS}, got {tokens!r}")
    return features_from_tokens(
        tokens, **{token: cache.tokens(token)[index] for token in tokens.split("+")}
    )


class LinearHead(nn.Module):
    """One weighted sum per class: multinomial logistic regression on the features."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProtoHead(nn.Module):
    """K prototypes per class; a class's logit is the largest cosine between the features and
    its prototypes, over tau. tau is learned as log_tau (spec 003 US-2.2)."""

    def __init__(
        self, dim: int, n_classes: int, prototypes_per_class: int = 4, tau_init: float = 0.07
    ) -> None:
        super().__init__()
        if prototypes_per_class < 1 or tau_init <= 0:
            raise ValueError("prototypes_per_class must be >= 1 and tau_init > 0")
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_classes, prototypes_per_class, dim), dim=-1)
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self) -> float:
        return float(self.log_tau.detach().exp())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = torch.einsum(
            "nd,ckd->nck", F.normalize(x, dim=-1), F.normalize(self.prototypes, dim=-1)
        )
        return cos.amax(dim=2) / self.log_tau.exp()


class MixHead(nn.Module):
    """p = w_linear * softmax(linear) + w_proto * softmax(proto), weights fixed; the logits
    are log p, so a softmax gives p back (spec 003 US-2.3)."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        weights: tuple[float, float] | list[float] = (0.5, 0.5),
        linear: dict[str, Any] | None = None,
        proto: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if len(weights) != 2 or min(weights) < 0 or not math.isclose(sum(weights), 1.0):
            raise ValueError(f"mix weights must be two shares summing to 1, got {weights}")
        self.weights = [float(w) for w in weights]
        self.linear = LinearHead(dim, n_classes, **(linear or {}))
        self.proto = ProtoHead(dim, n_classes, **(proto or {}))
        self.register_buffer("log_weights", torch.tensor(self.weights).log(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.stack(
            [F.log_softmax(self.linear(x), dim=1), F.log_softmax(self.proto(x), dim=1)]
        )
        return torch.logsumexp(parts + self.log_weights[:, None, None], dim=0)


HEADS: dict[str, type[nn.Module]] = {"linear": LinearHead, "proto": ProtoHead, "mix": MixHead}


def build_head(head: str, dim: int, n_classes: int, **params: Any) -> nn.Module:
    """A head of its kind; `params` are its constructor arguments (spec 003 FR-005)."""
    if head not in HEADS:
        raise ValueError(f"head must be one of {tuple(HEADS)}, got {head!r}")
    return HEADS[head](dim, n_classes, **params)


@dataclass
class Run:
    """A trained head: its folder, run.json and the model, ready to score features."""

    folder: Path
    meta: dict[str, Any]
    model: nn.Module

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def tokens(self) -> str:
        return self.meta["tokens"]

    def logits(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Softmax over the known classes (no temperature until S4.4, W4)."""
        z = self.logits(x)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def load_run(folder: Path | str) -> Run:
    folder = Path(folder)
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    saved = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
    model = build_head(
        saved["head"], int(saved["input_dim"]), len(saved["classes"]), **saved.get("params", {})
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return Run(folder, meta, model)


def list_runs(heads_root: Path | str = HEADS_ROOT) -> list[Path]:
    """Run folders (with a run.json) under heads_root, oldest first by created_at."""
    root = Path(heads_root)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if (p / "run.json").is_file()]

    def created(p: Path) -> str:
        return json.loads((p / "run.json").read_text(encoding="utf-8")).get("created_at", "")

    return sorted(runs, key=lambda p: (created(p), p.name))


def resolve_path(recorded: str) -> Path:
    """A path as run.json records it: relative to the repository root, or absolute."""
    p = Path(recorded)
    return p if p.is_absolute() else REPO_ROOT / p
