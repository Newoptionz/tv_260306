from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def list_input_files(data_raw: Path, patterns: tuple[str, ...] = ("*.csv", "*.parquet")) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(data_raw.glob(pat)))
    # unique while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def load_df(path: Path, index_col: Optional[str] = None, parse_dates: bool = True) -> pd.DataFrame:
    """Load a dataframe from CSV or Parquet.

    Notes:
      - If CSV, this assumes the first column is an index if index_col is None.
      - If you want the timestamp to be the index reliably, pass index_col explicitly.
    """
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df

    if path.suffix.lower() == ".csv":
        if index_col is None:
            df = pd.read_csv(path)
            if df.index.name is not None:
                idx_name = str(df.index.name).strip().lower()
                if idx_name in ("time", "datetime", "date"):
                    df = df.reset_index()
        else:
            df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        return df

    raise ValueError(f"Unsupported input file type: {path}")


def save_parquet(df: pd.DataFrame, path: Path, reset_index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df.reset_index() if reset_index else df
    out_df.to_parquet(path, index=not reset_index)
    return path


def save_csv(df: pd.DataFrame, path: Path, float_format: str = "%.5f", reset_index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df.reset_index() if reset_index else df
    out_df.to_csv(path, index=not reset_index, float_format=float_format)
    return path
