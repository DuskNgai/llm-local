#!/usr/bin/env python3
"""Resolve model alias to local path from models.yaml."""

import os
import sys
from pathlib import Path

import yaml
import platform as _plat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = PROJECT_ROOT / "models.yaml"
HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"


def _current_platform() -> str:
    system = _plat.system()
    if system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    sys.exit(f"Unsupported platform: {system}")


def _repo_cache_path(repo: str) -> str:
    dirname = repo.replace("/", "--")
    snapshots_dir = HF_CACHE / f"models--{dirname}" / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(snapshots_dir.iterdir(), key=os.path.getmtime)
        if snapshots:
            return str(snapshots[-1])
    sys.exit(f"Model '{repo}' not found in HF cache. Run: bash scripts/download-model.sh")


def resolve(alias: str) -> str:
    try:
        models = yaml.safe_load(MODELS_YAML.read_text())
    except Exception:
        sys.exit(f"Failed to parse {MODELS_YAML}")
    if not isinstance(models, dict):
        sys.exit(f"{MODELS_YAML} is empty or malformed (expected a mapping)")
    plat = _current_platform()
    if alias not in models:
        sys.exit(f"Model '{alias}' not found in {MODELS_YAML}")
    entry = models[alias]
    if not isinstance(entry, dict):
        sys.exit(f"Model '{alias}' has unexpected format in {MODELS_YAML}")
    if plat not in entry:
        sys.exit(f"Model '{alias}' has no '{plat}' entry in {MODELS_YAML}")
    plat_entry = entry[plat]
    if not isinstance(plat_entry, dict):
        sys.exit(f"Platform entry for '{alias}/{plat}' must be a mapping in {MODELS_YAML}")
    if "repo" in plat_entry:
        return _repo_cache_path(plat_entry["repo"])
    if "path" in plat_entry:
        return str(PROJECT_ROOT / plat_entry["path"])
    sys.exit(f"Model '{alias}/{plat}' has neither repo: nor path: in {MODELS_YAML}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <alias>")
    print(resolve(sys.argv[1]))
