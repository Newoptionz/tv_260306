from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


TRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset TEXT,
  stream TEXT,
  trade_id INTEGER,
  side TEXT,
  entry_time TEXT,
  exit_time TEXT,
  entry_bar_i INTEGER,
  exit_bar_i INTEGER,
  entry_px REAL,
  exit_px REAL,
  pnl REAL,
  bars_held INTEGER,
  kern_dist_sig REAL,
  state_sig TEXT
);
"""


def ensure_db(db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(TRADES_TABLE_SQL)
        conn.commit()


def append_trades(db_path: Path, trades: pd.DataFrame, dataset: str) -> int:
    """Append trades dataframe to sqlite.

    Adds a 'dataset' column to track which input file/run produced the trades.
    Returns number of inserted rows.
    """
    if trades is None or trades.empty:
        return 0

    df = trades.copy()
    df.insert(0, "dataset", dataset)

    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("trades", conn, if_exists="append", index=False)
    return len(df)
