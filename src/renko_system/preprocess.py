from __future__ import annotations

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
    if name not in df.columns:
        raise KeyError(f"Missing column '{name}'. Available columns: {list(df.columns)}")
    return df[name]


def preprocess_tv_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a TradingView CSV export into the canonical schema used by trade markers.

    Supports both v09 and v10 indicator exports:
      - v09: last_state_code, Open Long/Short, Regime Flip columns present
      - v10: state_code (0N/1I/2F/3S), Buy/Sell contain price values,
             kernel confident/not confident dots columns

    Canonical outputs
    -----------------
    Timestamps  : dt, dt_utc
    OHLC        : open, high, low, close (close = hlcc4)
    Kernel      : kern_est (smoothed), kern_reg (raw regression)
    Signals     : lor_buy_sig, lor_sell_sig (0/1 flags)
                  lor_buy_px, lor_sell_px   (price values, v10 only)
                  open_long, open_short      (0/1, v09 only; 0 in v10)
                  lor_flip_long, lor_flip_short (0/1, v09 only; 0 in v10)
                  v_buy_sig, v_sell_sig      (verify/shapes signals)
    Kern dots   : kern_confident (non-null = green dot, v10)
                  kern_not_confident (non-null = red dot, v10)
    State       : state_code (int 0-3), state_label
    Revisions   : r_down, rs_down, r_up, rs_up
    Time        : time_bucket (Perth-local session label), bar_i
    """
    df = df.copy()

    # ── parse time ────────────────────────────────────────────────────────────
    df, _time_col = parse_time_column(df)
    df = df.sort_values("dt").reset_index(drop=True)
    df = df.drop_duplicates(subset=["dt"], keep="first")

    # ── normalise column names ────────────────────────────────────────────────
    df.columns = [str(c).strip().lower() for c in df.columns]

    # deduplicate after normalisation
    if df.columns.duplicated().any():
        seen = {}
        new_cols = []
        for c in df.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
        df.columns = new_cols

    # ── detect schema version ────────────────────────────────────────────────
    # v10 exports "state_code (0n/1i/2f/3s)" and buy/sell as price values
    _sc_col = next((c for c in df.columns if c.startswith("state_code")), None)
    _is_v10 = _sc_col is not None and _sc_col != "last_state_code"

    # ── state code ───────────────────────────────────────────────────────────
    if _sc_col is None:
        raise KeyError(
            "state_code column not found. "
            "Expected 'state_code (0N/1I/2F/3S)' (v10) or 'last_state_code' (v09)."
        )
    df["state_code"] = (
        pd.to_numeric(df[_sc_col], errors="coerce")
        .fillna(0).astype(int).clip(0, 3)
    )
    state_map = {0: "NEUTRAL", 1: "IMPULSE", 2: "FAST", 3: "SLOW"}
    df["state_label"] = df["state_code"].map(state_map)

    # ── OHLC ─────────────────────────────────────────────────────────────────
    for req in ("open", "high", "low", "close"):
        _safe_col(df, req)
    df["close"] = (df["high"] + df["low"] + df["close"] + df["close"]) / 4.0

    # ── kernel estimates ──────────────────────────────────────────────────────
    if "kernel estimate" in df.columns:
        df["kern_est"] = df["kernel estimate"]          # smoothed (v10)
    elif "kernel regression estimate" in df.columns:
        df["kern_est"] = df["kernel regression estimate"]
    else:
        raise KeyError("Missing kernel estimate column.")

    if "kernel regression estimate" in df.columns:
        df["kern_reg"] = df["kernel regression estimate"]
    else:
        df["kern_reg"] = df["kern_est"]

    # ── kern confident dots (v10) ─────────────────────────────────────────────
    df["kern_confident"]     = df.get("kernel confident dots",     pd.Series(index=df.index, dtype=float))
    df["kern_not_confident"] = df.get("kernel not confident dots", pd.Series(index=df.index, dtype=float))

    # ── LOR buy/sell signals ──────────────────────────────────────────────────
    # v10: buy/sell columns contain price values; non-null = signal fired
    # v09: buy/sell were non-null markers (same notna() logic works for both)
    if "buy" in df.columns:
        df["lor_buy_sig"] = df["buy"].notna().astype(int)
        df["lor_buy_px"]  = pd.to_numeric(df["buy"], errors="coerce")
    else:
        df["lor_buy_sig"] = 0
        df["lor_buy_px"]  = float("nan")

    if "sell" in df.columns:
        df["lor_sell_sig"] = df["sell"].notna().astype(int)
        df["lor_sell_px"]  = pd.to_numeric(df["sell"], errors="coerce")
    else:
        df["lor_sell_sig"] = 0
        df["lor_sell_px"]  = float("nan")

    # ── open long/short (v09 only; zero in v10) ───────────────────────────────
    df["open_long"]  = (
        df["open long"].fillna(0).astype(int) if "open long" in df.columns else 0
    )
    df["open_short"] = (
        df["open short"].fillna(0).astype(int) if "open short" in df.columns else 0
    )

    # ── regime flips (v09 only; zero in v10) ─────────────────────────────────
    df["lor_flip_long"] = (
        pd.to_numeric(df["regime flip → long"], errors="coerce").fillna(0).astype(int)
        if "regime flip → long" in df.columns else 0
    )
    df["lor_flip_short"] = (
        pd.to_numeric(df["regime flip → short"], errors="coerce").fillna(0).astype(int)
        if "regime flip → short" in df.columns else 0
    )

    # ── verify / shapes signals ───────────────────────────────────────────────
    df["v_buy_sig"]  = df["shapes.1"].notna().astype(int) if "shapes.1" in df.columns else 0
    df["v_sell_sig"] = df["shapes"].notna().astype(int)   if "shapes"   in df.columns else 0

    # ── revision flags ────────────────────────────────────────────────────────
    df["r_down"]  = df["chars"].notna().astype(int)   if "chars"   in df.columns else 0
    df["rs_down"] = df["chars.1"].notna().astype(int) if "chars.1" in df.columns else 0
    df["r_up"]    = df["chars.2"].notna().astype(int) if "chars.2" in df.columns else 0
    df["rs_up"]   = df["chars.3"].notna().astype(int) if "chars.3" in df.columns else 0

    # ── coerce all int signal cols ────────────────────────────────────────────
    int_cols = [
        "open_long", "open_short", "lor_flip_long", "lor_flip_short",
        "lor_buy_sig", "lor_sell_sig", "v_buy_sig", "v_sell_sig",
        "r_down", "rs_down", "r_up", "rs_up",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ── bar index + time bucket ───────────────────────────────────────────────
    df["bar_i"] = range(len(df))
    df["time_bucket"] = df["dt"].apply(_time_bucket)

    return df


def _time_bucket(ts: pd.Timestamp) -> str:
    h = ts.hour
    if    0 <= h <  5: return "a_NIGHT_00-05"
    elif  5 <= h <  7: return "b_PREMARKET_05-07"
    elif  7 <= h < 12: return "c_MORNING_07-12"
    elif 12 <= h < 18: return "d_MIDDAY_12-18"
    elif 18 <= h < 20: return "e_EVENING_18-20"
    elif 20 <= h < 24: return "f_US_OPEN_20-24"
