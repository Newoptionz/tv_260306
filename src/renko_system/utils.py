from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_bar_i(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bar_i" not in df.columns:
        df["bar_i"] = range(len(df))
    return df


def require_cols(df: pd.DataFrame, cols: list[str], fn_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{fn_name}: missing required columns: {missing}")


def to_bool(x) -> bool:
    """Robust conversion for signal columns.

    Avoids bool(np.nan) == True surprises.
    """
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return x != 0
    if isinstance(x, float) and np.isnan(x):
        return False
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("1", "true", "t", "yes", "y")
    try:
        return bool(x)
    except Exception:
        return False


def iter_signal_exec_pairs(df: pd.DataFrame):
    """Yield (i, ts_sig, row_sig, j, ts_exec, row_exec) where exec is next bar."""
    n = len(df)
    if n < 2:
        return
    idx = df.index
    for i in range(n - 1):
        j = i + 1
        yield i, idx[i], df.iloc[i], j, idx[j], df.iloc[j]


def append_event(df: pd.DataFrame, ts, col: str, event: str) -> None:
    prev = df.at[ts, col]
    if prev is None or (isinstance(prev, float) and np.isnan(prev)) or prev == "":
        df.at[ts, col] = event
    else:
        df.at[ts, col] = (str(prev) + "|" + event).strip("|")
