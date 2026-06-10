"""Ensure vendored llm-attacks is importable before any llm_attacks import."""

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor"


def ensure_llm_attacks_path() -> None:
    vendor_str = str(_VENDOR_ROOT)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
