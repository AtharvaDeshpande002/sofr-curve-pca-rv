"""Load config.yaml and resolve every path relative to the repository root.

Why: scripts, tests and the notebook run from different folders. Resolving paths from the repo root
means one config file works everywhere and nothing is hard-coded.
"""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Read the YAML config and turn every entry of `paths` into an absolute pathlib.Path."""
    cfg_path = pathlib.Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
    return cfg
