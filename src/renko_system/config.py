from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any


INSTRUMENT_CONFIG: Dict[str, Dict[str, float]] = {
    # Metals
    "SIL": {"tick_size": 0.005, "tick_value": 2.5},
    "GC":  {"tick_size": 0.1,   "tick_value": 10},
    "MGC": {"tick_size": 0.1,   "tick_value": 1},

    # Nasdaq
    "NQ":  {"tick_size": 0.25,  "tick_value": 5},
    "MNQ": {"tick_size": 0.25,  "tick_value": 0.5},

    # S&P 500
    "ES":  {"tick_size": 0.25,  "tick_value": 12.5},
    "MES": {"tick_size": 0.25,  "tick_value": 1.25},
}


def detect_instrument_from_filename(filename: str) -> str:
    """Best-effort instrument root detection from filename.

    Matches tokens like ES1!, MNQ1!, MGC, SIL, etc.
    """
    s = filename.upper()

    # Prefer longer / micro roots before full contracts (avoid MES matching ES)
    roots = ["MNQ", "NQ", "MES", "ES", "MGC", "GC", "SIL"]

    for r in roots:
        # token match; optionally digits, optional '!' suffix
        if re.search(rf"(?<![A-Z0-9]){r}(\d+)?!?(?![A-Z0-9])", s):
            return r

    for r in roots:
        if r in s:
            return r

    raise ValueError(f"Could not detect instrument from filename: {filename}")
