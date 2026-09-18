# Piece 4 (model service v0) - the make targets of docs/piece4-work-plan.md.
# Run from the repository root. Every target is idempotent.
#
# On the Windows dev box there is no make: use .\make.ps1 <target>, which runs the
# same commands. The Makefile is what the Linux/WSL side (pieces 2 and 3) uses.

UV ?= uv
PY := $(UV) run python

# Defaults for the parameterised targets; override on the command line:
#   make cache BB=dinov3_l16 RES=518 DS=makerere
DS  ?= ibean
BB  ?= dinov2_l14_reg
RES ?= 224
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help env env-check hub-check compute-log test lint fmt download manifests cache heads eval probe serve clean

help: ## list the targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# --- W1: environment -------------------------------------------------------

env: ## create/refresh .venv from pyproject.toml + uv.lock, then record the machine (N7)
	$(UV) sync
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example - put HF_TOKEN in it")
	$(MAKE) compute-log

compute-log: ## append the environment row (N7) to model_service/results/compute_log.jsonl
	$(PY) -m ms.compute_log record-env

env-check: ## print what torch sees (device, CUDA, VRAM)
	$(PY) -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

hub-check: ## Hugging Face access: account, gated + commit sha per backbone, DINOv3 load (.env)
	$(PY) -m ms.backbones.check $(ARGS)

test: ## pytest (model_service/tests)
	$(UV) run pytest

lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## ruff format
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# --- W2 onwards: the pipeline ----------------------------------------------
# These modules land with their specs: 001 manifests and 002 cache in W2,
# 003 heads and 005 eval in W3, 004 abstention in W4, 006 service in W5.

download: ## (W1-W2) fetch a raw dataset into data/raw/$(DS) + DOWNLOAD.json
	$(PY) -m ms.data.download --dataset $(DS) $(ARGS)

manifests: ## (W2, S4.1) build data/manifests/$(DS)_v1.jsonl
	$(PY) -m ms.data.manifests build --dataset $(DS) $(ARGS)

cache: ## (W2, S4.2) extract features: BB=<backbone_id> RES=224|518 DS=<dataset>
	$(PY) -m ms.cache.extract --backbone $(BB) --res $(RES) --dataset $(DS) $(ARGS)

heads: ## (W3, S4.3) train the heads (five seeds)
	$(PY) -m ms.heads.train --backbone $(BB) --res $(RES) $(ARGS)

eval: ## (W3-W4, S4.5) results/n_table.jsonl -> n_table.md + verdict.md
	$(PY) -m ms.eval.run $(ARGS)

probe: ## (W2, N4) site-prediction probe on cached features
	$(PY) -m ms.eval.probe $(ARGS)

serve: ## (W5, S4.6) the model service on :8000
	$(UV) run uvicorn ms.service.app:app --host 127.0.0.1 --port 8000

clean: ## remove caches of the tooling (never data, never results)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
