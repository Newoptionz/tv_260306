import re

INSTRUMENT_CONFIG = {
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
    s = filename.upper()

    # Prefer micros before full contracts (avoid MES matching ES)
    roots = ["MNQ", "NQ", "MES", "ES", "MGC", "GC", "SIL"]

    for r in roots:
        if re.search(rf"(?<![A-Z0-9]){r}(\d+)?!?(?![A-Z0-9])", s):
            return r

    for r in roots:
        if r in s:
            return r

    raise ValueError(f"Could not detect instrument from filename: {filename}")