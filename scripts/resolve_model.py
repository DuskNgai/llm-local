#!/usr/bin/env python3
"""Resolve model alias to local path from models.yaml."""

import sys
from pathlib import Path

import yaml

MODELS_YAML = Path(__file__).parent.parent / "models.yaml"
HF_CACHE = Path(__file__).parent.parent / ".cache" / "huggingface"


def _repo_cache_path(repo: str) -> str:
    dirname = repo.replace("/", "--")
    snapshots_dir = HF_CACHE / f"models--{dirname}" / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[-1])
    sys.exit(f"Model '{repo}' not found in HF cache. Run: bash scripts/download-model.sh")


def resolve(alias: str) -> str:
    try:
        models = yaml.safe_load(MODELS_YAML.read_text())
    except Exception:
        sys.exit(f"Failed to parse {MODELS_YAML}")
    if alias not in models:
        sys.exit(f"Model '{alias}' not found in {MODELS_YAML}")
    entry = models[alias]
    if "repo" in entry:
        return _repo_cache_path(entry["repo"])
    if "path" in entry:
        return str(Path(__file__).parent.parent / entry["path"])
    sys.exit(f"Model '{alias}' has neither repo: nor path: in {MODELS_YAML}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <alias>")
    print(resolve(sys.argv[1]))
