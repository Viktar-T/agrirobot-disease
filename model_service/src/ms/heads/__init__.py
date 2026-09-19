"""Heads on cached features (S4.3; spec 003 comes in W3, this is the W2 slice's linear probe).

    python -m ms.heads.train --backbone dinov2_l14_reg --res 224 --head linear --seed 0 \\
        --train-manifest data/manifests/ibean_v1.jsonl --split train --val-split val

A run is data/heads/<run_id>/ (git-ignored): head.pt, the weights, and run.json, with the
inputs and their hashes, the fit and the model_version string (H8 6.2). Features are
L2-normalised per token type; "cls+meanpatch" concatenates the two (the W4 ablation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ms.cache import Cache

#: repository root: model_service/src/ms/heads/__init__.py -> ../../../../
REPO_ROOT = Path(__file__).resolve().parents[4]
HEADS_ROOT = REPO_ROOT / "data" / "heads"
HEAD_CONFIGS = REPO_ROOT / "model_service" / "configs" / "heads"
TOKEN_SPECS = ("cls", "meanpatch", "cls+meanpatch")
#: the service version at the head of model_version (work plan W5)
SERVICE_VERSION = "msv0.1"


def model_version(backbone_id: str, res: int, head_id: str, train_manifest_sha256: str) -> str:
    """msv0.1+<backbone_id>@<res>.<head_id>.man-<train manifest sha256[:6]> (work plan W5)."""
    return f"{SERVICE_VERSION}+{backbone_id}@{res}.{head_id}.man-{train_manifest_sha256[:6]}"


def features(cache: Cache, index: np.ndarray, tokens: str) -> np.ndarray:
    """float32 [len(index), D x token types]: each token type L2-normalised, then joined."""
    if tokens not in TOKEN_SPECS:
        raise ValueError(f"tokens must be one of {TOKEN_SPECS}, got {tokens!r}")
    parts = []
    for token in tokens.split("+"):
        x = cache.tokens(token)[index].astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        parts.append(x)
    return np.concatenate(parts, axis=1)


class LinearHead(nn.Module):
    """One weighted sum per class: multinomial logistic regression on the features."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


HEADS = {"linear": LinearHead}


def build_head(head: str, dim: int, n_classes: int) -> nn.Module:
    if head not in HEADS:
        raise ValueError(f"head must be one of {tuple(HEADS)} (proto and mix come in W3)")
    return HEADS[head](dim, n_classes)


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
    model = build_head(saved["head"], int(saved["input_dim"]), len(saved["classes"]))
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
