from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def parse_time_column(df: pd.DataFrame, tz_chart: str = "Australia/Perth", out_col: str = "dt"):
    """
    Accepts:
      - epoch seconds/ms (numeric) -> dt_utc (UTC) + dt (tz_chart)
      - ISO-8601 strings WITH timezone (Z or ±HH:MM) -> dt + dt_utc
    Rejects:
      - timezone-naive strings (ambiguous)
    """
    cols = {str(c).strip().lower(): c for c in df.columns}
    for key in ("time", "datetime", "date"):
        if key in cols:
            col = cols[key]
            break
    else:
        raise ValueError("No obvious time column found (time / datetime / date).")

    s = df[col]

    # numeric epoch
    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        median = s_num.dropna().astype("int64").median()
        unit = "ms" if median > 1e12 else "s"

        dt_utc = pd.to_datetime(s_num, unit=unit, utc=True, errors="coerce")
        df["dt_utc"] = dt_utc
        df[out_col] = dt_utc.dt.tz_convert(tz_chart)
        return df, col

    # ISO strings (must include timezone)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        raise ValueError(f"Failed to parse timestamps in '{col}'.")

    if getattr(dt.dt, "tz", None) is None:
        ex = s.dropna().astype(str).head(3).tolist()
        raise ValueError(
            f"'{col}' timestamps are timezone-naive (ambiguous). "
            f"Export ISO with timezone, e.g. '2026-01-07T10:13:00+08:00' or '...Z'. "
            f"Examples: {ex}"
        )

    df[out_col] = dt.dt.tz_convert(tz_chart)
    df["dt_utc"] = df[out_col].dt.tz_convert("UTC")
    return df, col


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if present else raise a useful error."""
    if name not in df.columns:
        raise KeyError(f"Missing column '{name}'. Available columns: {list(df.columns)}")
    return df[name]


def preprocess_tv_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a TradingView CSV export into a canonical schema used by the trade markers.

    Canonical outputs:
      - dt, dt_utc
      - ohlc: open/high/low/close (close is hlcc4-ish as per your rule)
      - kern_est
      - open_long, open_short
      - renko_syn_close
      - lor_flip_long, lor_flip_short
      - lor_buy_sig, lor_sell_sig
      - v_buy_sig, v_sell_sig
      - state_code, state_label
      - revision flags r_down/rs_down/r_up/rs_up
    """
    df = df.copy()

    # --- Parse time once ---
    df, _time_col = parse_time_column(df)

    # sort, remove duplicate timestamps
    df = df.sort_values("dt").reset_index(drop=True)
    df = df.drop_duplicates(subset=["dt"], keep="first")

    # sanity (optional)
    assert df["dt"].is_monotonic_increasing
    assert not df["dt"].duplicated().any()

    # --- normalize column names (lowercase/strip) ---
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Deduplicate column labels after normalization (prevents reindex errors)
    if df.columns.duplicated().any():
        new_cols = []
        seen = {}
        for c in df.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
        df.columns = new_cols





    # Renko exit/close column (from indicator)
    if "exit_renko" in df.columns:
        df["close_all"] = df["exit_renko"]

    # --- strict state handling ---
    if "last_state_code" not in df.columns:
        raise KeyError(
            "state_code column not found. Did you export plot(float(last_state_code), title='state_code')?"
        )

    df["state_code"] = (
        pd.to_numeric(df["last_state_code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 3)
    )
    state_map = {0: "NEUTRAL", 1: "IMPULSE", 2: "FAST", 3: "SLOW"}
    df["state_label"] = df["state_code"].map(state_map)

    # --- derive canonical 'close' (your hlcc4-ish rule) ---
    # You wrote: (high+low+close+close)/4
    # Using current df["close"] column as the two closes.
    for req in ("high", "low", "close"):
        _safe_col(df, req)

    df["close"] = (df["high"] + df["low"] + df["close"] + df["close"]) / 4.0

    # --- kernel estimate ---
    # You wrote: df["kern_est"] = df["kernel regression estimate"]
    if "kernel regression estimate" in df.columns:
        df["kern_est"] = df["kernel regression estimate"]
    elif "kernel estimate" in df.columns:
        df["kern_est"] = df["kernel estimate"]
    else:
        raise KeyError(
            "Missing kernel estimate column. Expected 'kernel regression estimate' or 'kernel estimate'."
        )

    # --- required renko/signal columns (lowercase names) ---
    # the columns needed are [open long], [open short], [close1] etc.
    if "open long" in df.columns:
        df["open_long"] = df["open long"].fillna(0).astype(int)
    else:
        df["open_long"] = 0

    if "open short" in df.columns:
        df["open_short"] = df["open short"].fillna(0).astype(int)
    else:
        df["open_short"] = 0

    if "synthetic renko close" in df.columns:
        df["renko_syn_close"] = df["synthetic renko close"]
    else:
        df["renko_syn_close"] = pd.NA

    # flips
    if "regime flip → long" in df.columns:
        df["lor_flip_long"] = pd.to_numeric(df["regime flip → long"], errors="coerce").fillna(0).astype(int)
    else:
        df["lor_flip_long"] = 0

    if "regime flip → short" in df.columns:
        df["lor_flip_short"] = pd.to_numeric(df["regime flip → short"], errors="coerce").fillna(0).astype(int)
    else:
        df["lor_flip_short"] = 0

    # LOR buy/sell signals
    # You wrote: buy/sell are non-null markers
    df["lor_buy_sig"] = df["buy"].notna().astype(int) if "buy" in df.columns else 0
    df["lor_sell_sig"] = df["sell"].notna().astype(int) if "sell" in df.columns else 0

    # VERIFY buy/sell signals (shapes)
    df["v_buy_sig"] = df["shapes.1"].notna().astype(int) if "shapes.1" in df.columns else 0
    df["v_sell_sig"] = df["shapes"].notna().astype(int) if "shapes" in df.columns else 0

    # revisions
    df["r_down"] = df["chars"].notna().astype(int) if "chars" in df.columns else 0
    df["rs_down"] = df["chars.1"].notna().astype(int) if "chars.1" in df.columns else 0
    df["r_up"] = df["chars.2"].notna().astype(int) if "chars.2" in df.columns else 0
    df["rs_up"] = df["chars.3"].notna().astype(int) if "chars.3" in df.columns else 0

    # Final: ensure all signal cols are int 0/1 (avoid NaN=>True bugs)
    for c in ["open_long", "open_short", "lor_flip_long", "lor_flip_short",
              "lor_buy_sig", "lor_sell_sig", "v_buy_sig", "v_sell_sig",
              "r_down", "rs_down", "r_up", "rs_up"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df