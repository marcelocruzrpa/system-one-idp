"""Load policy, taxonomy and per-doc-type schemas from `config/`."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SAMPLES_DIR = ROOT / "samples"
CACHE_DIR = ROOT / ".cache"
OUT_DIR = ROOT / "out"

load_dotenv(ROOT / ".env")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def policy() -> dict[str, Any]:
    return _load(CONFIG_DIR / "policy.yaml")


@cache
def taxonomy() -> dict[str, Any]:
    return _load(CONFIG_DIR / "taxonomy.yaml")


@cache
def schema(doc_type: str) -> dict[str, Any] | None:
    """Extraction + validation config for a doc type, or None when the type has no schema (e.g. `other`)."""
    path = CONFIG_DIR / "schemas" / f"{doc_type}.yaml"
    return _load(path) if path.exists() else None


def threshold(name: str) -> float:
    return float(policy()["thresholds"][name])
