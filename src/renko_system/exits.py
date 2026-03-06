from __future__ import annotations

"""exits.py
===========
Exit simulation engine, calibration helpers, and trade summary tools.

Public API
----------
simulate_exit(w, side, entry_px, ...)
    Simulate a single exit over a bar window. Returns (exit_px, exit_bar_idx, reason).

simulate_exit_with_pyramiding(w, side, entry_px, ...)
    As above but supports pyramid add-ons. Returns (exit_px, avg_entry_px, reason, qty).

run_exit_strategies(df, trades_df, exit_strategies, ...)
    Run a dict of exit strategies against all trades. Returns (trades_all, summary).

add_mae_mfe(trades, df, ...)
    Annotate a trades DataFrame with MAE/MFE values from bar data.

trade_core_calibration(trades_df, ...)
    Compute median MAE/MFE/bars stats for exit parameter seeding.

build_exit_suite_from_cal(cal, ...)
    Auto-generate a starter exit strategy suite from calibration stats.

summarize_trades_grouped(trades_df, groupby_cols, ...)
    Grouped P&L + excursion summary.

summarize_side_and_time_buckets(trades_df, ...)
    Convenience wrapper: returns by_side / by_time / by_side_time summaries.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Union


# ─────────────────────────────────────────────────────────────────────────────
# MAE / MFE annotation
# ─────────────────────────────────────────────────────────────────────────────

def add_mae_mfe(
    trades: pd.DataFrame,
    df: pd.DataFrame,
    price_entry_col: str = "entry_px",
    entry_i_col: str = "entry_bar_i",
    exit_i_col: str = "exit_bar_i",
    side_col: str = "side",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """Add MAE/MFE columns to a trades DataFrame.

    Convention
    ----------
    LONG  : mae = min(low  - entry_px)  (<= 0),  mfe = max(high - entry_px) (>= 0)
    SHORT : mae = min(entry_px - high)  (<= 0),  mfe = max(entry_px - low)  (>= 0)

    Also adds mae_bar_i / mfe_bar_i (absolute bar index of each extreme).
    """
    trades = trades.copy()

    highs = df[high_col].values
    lows  = df[low_col].values

    mae_list, mfe_list, mae_i_list, mfe_i_list = [], [], [], []

    for _, t in trades.iterrows():
        e = int(t[entry_i_col])
        x = int(t[exit_i_col])

        if x < e:
            mae_list.append(np.nan);  mfe_list.append(np.nan)
            mae_i_list.append(np.nan); mfe_i_list.append(np.nan)
            continue

        entry_px = float(t[price_entry_col])
        side     = str(t[side_col]).upper()

        hi_slice = highs[e : x + 1]
        lo_slice = lows[e  : x + 1]

        if side == "LONG":
            adv = lo_slice - entry_px
            fav = hi_slice - entry_px
        else:
            adv = entry_px - hi_slice
            fav = entry_px - lo_slice

        mae = float(np.min(adv))
        mfe = float(np.max(fav))
        mae_i = e + int(np.argmin(adv))
        mfe_i = e + int(np.argmax(fav))

        mae_list.append(mae);   mfe_list.append(mfe)
        mae_i_list.append(mae_i); mfe_i_list.append(mfe_i)

    trades["mae"]       = mae_list
    trades["mfe"]       = mfe_list
    trades["mae_bar_i"] = mae_i_list
    trades["mfe_bar_i"] = mfe_i_list
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration
# ─────────────────────────────────────────────────────────────────────────────

def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def trade_core_calibration(
    trades_df: pd.DataFrame,
    pnl_col:  Optional[str] = None,
    mae_col:  Optional[str] = None,
    mfe_col:  Optional[str] = None,
    bars_col: Optional[str] = None,
    tick_size: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute core stats for exit calibration.

    Auto-detects common column name variants if not provided explicitly.
    tick_size: if given, also returns tick-equivalent values for MAE/MFE.
    """
    df = trades_df.copy()

    pnl_col  = pnl_col  or _pick_col(df, ["pnl", "pnl_sum", "profit", "pnl_pts", "pnl_points"])
    mae_col  = mae_col  or _pick_col(df, ["mae", "MAE", "mae_pts", "mae_points"])
    mfe_col  = mfe_col  or _pick_col(df, ["mfe", "MFE", "mfe_pts", "mfe_points"])
    bars_col = bars_col or _pick_col(df, ["bars_held", "hold_bars", "bars", "duration_bars"])

    missing = [name for name, col in [("mae", mae_col), ("mfe", mfe_col), ("bars", bars_col)] if col is None]
    if missing:
        raise ValueError(
            f"Could not auto-detect columns: {missing}. "
            f"Pass explicitly. Available: {list(df.columns)}"
        )

    df["mae_abs"] = df[mae_col].abs()

    out: Dict[str, Any] = {
        "trade_count":      int(len(df)),
        "mae_col":          mae_col,
        "mfe_col":          mfe_col,
        "bars_col":         bars_col,
        "pnl_col":          pnl_col,
        "median_mae":       float(df["mae_abs"].median()),
        "median_mfe":       float(df[mfe_col].median()),
        "mfe_75":           float(df[mfe_col].quantile(0.75)),
        "mfe_90":           float(df[mfe_col].quantile(0.90)),
        "median_bars_held": float(df[bars_col].median()),
        "bars_75":          float(df[bars_col].quantile(0.75)),
    }

    if pnl_col and pnl_col in df.columns:
        out.update({
            "pnl_sum":  float(df[pnl_col].sum()),
            "pnl_mean": float(df[pnl_col].mean()),
            "win_rate": float((df[pnl_col] > 0).mean()),
        })

    if tick_size and tick_size > 0:
        out.update({
            "tick_size":          tick_size,
            "median_mae_ticks":   out["median_mae"] / tick_size,
            "median_mfe_ticks":   out["median_mfe"] / tick_size,
            "mfe_75_ticks":       out["mfe_75"]     / tick_size,
            "mfe_90_ticks":       out["mfe_90"]     / tick_size,
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Exit suite builder
# ─────────────────────────────────────────────────────────────────────────────

def round_to_tick(x: float, tick_size: Optional[float]) -> float:
    if tick_size is None or tick_size <= 0:
        return float(x)
    return round(round(x / tick_size) * tick_size, 10)


def build_exit_suite_from_cal(
    cal: Dict[str, Any],
    tick_size: Optional[float] = None,
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Auto-generate a starter exit strategy suite from calibration stats.

    Derived from median MAE/MFE and MFE tail quantiles.
    All distances are in price points; tick-rounded if tick_size is provided.

    Returns a dict of {strategy_name: params_dict}.
    A special "_META_" key contains the source stats (not a strategy).
    """
    mae   = float(cal["median_mae"])
    mfe   = float(cal["median_mfe"])
    mfe75 = float(cal.get("mfe_75", mfe * 1.7))
    mfe90 = float(cal.get("mfe_90", mfe * 2.5))

    sl_tight = mae * 0.9
    sl_base  = mae * 1.2
    sl_wide  = mae * 1.6

    tp_tight = mfe * 1.1
    tp_base  = mfe * 1.4
    tp_wide  = mfe * 1.9

    trail_start_early = max(mfe * 1.05, mfe75 * 0.65)
    trail_start_tail  = mfe75 * 0.90
    trail_dist_base   = min(mae * 0.80, mfe * 0.90)
    trail_dist_wide   = min(mae * 1.10, mfe * 1.20)

    be_at_base = mfe * 1.4
    be_dist    = 0.0

    def R(x: float) -> float:
        return round_to_tick(x, tick_size)

    suite: Dict[str, Dict[str, Any]] = {
        f"{prefix}BASELINE": {},

        f"{prefix}SLT_TPT": dict(stop_loss=R(sl_tight), take_profit=R(tp_tight)),
        f"{prefix}SLB_TPB": dict(stop_loss=R(sl_base),  take_profit=R(tp_base)),
        f"{prefix}SLW_TPW": dict(stop_loss=R(sl_wide),  take_profit=R(tp_wide)),

        f"{prefix}RUN_EARLY": dict(
            stop_loss=R(sl_base),
            trail_start=R(trail_start_early),
            trail_dist=R(trail_dist_base),
        ),
        f"{prefix}RUN_TAIL": dict(
            stop_loss=R(sl_base),
            trail_start=R(trail_start_tail),
            trail_dist=R(trail_dist_base),
        ),
        f"{prefix}RUN_WIDE": dict(
            stop_loss=R(sl_wide),
            trail_start=R(trail_start_tail),
            trail_dist=R(trail_dist_wide),
        ),
        f"{prefix}BE_RUN": dict(
            stop_loss=R(sl_base),
            breakeven_at=R(be_at_base),
            breakeven_dist=R(be_dist),
            trail_start=R(trail_start_tail),
            trail_dist=R(trail_dist_base),
        ),
        f"{prefix}TP90_RUN": dict(
            stop_loss=R(sl_base),
            take_profit=R(mfe90),
            trail_start=R(trail_start_tail),
            trail_dist=R(trail_dist_base),
        ),
    }

    if tick_size and tick_size > 0:
        suite[f"{prefix}BE_RUN"]["breakeven_dist"] = R(tick_size)

    suite["_META_"] = {
        "median_mae": mae,
        "median_mfe": mfe,
        "mfe_75":     mfe75,
        "mfe_90":     mfe90,
        "tick_size":  tick_size,
        "note":       "Auto-generated from calibration. Validate before use.",
    }

    return suite


# ─────────────────────────────────────────────────────────────────────────────
# Single-trade exit simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_exit(
    w: pd.DataFrame,
    side: str,
    entry_px: float,
    stop_loss:      Optional[float] = None,
    take_profit:    Optional[float] = None,
    trail_start:    Optional[float] = None,
    trail_dist:     Optional[float] = None,
    breakeven_at:   Optional[float] = None,
    breakeven_dist: float           = 0.0,
) -> tuple[float, Any, str]:
    """Simulate exit logic over a bar window.

    Parameters
    ----------
    w            : DataFrame slice from entry bar onward (must have high, low, close)
    side         : "LONG" or "SHORT"
    entry_px     : fill price of the entry
    stop_loss    : fixed SL distance in points
    take_profit  : fixed TP distance in points
    trail_start  : profit in points at which trailing stop activates
    trail_dist   : trailing stop distance in points
    breakeven_at : profit in points at which SL moves to breakeven
    breakeven_dist: offset from entry for breakeven SL (0 = true BE, positive = profit lock)

    Returns
    -------
    (exit_px, exit_bar_index, reason)
    reason is one of: "STOP", "TRAIL", "TP", "LOGIC"
    """
    sl       = (entry_px - stop_loss)   if stop_loss   is not None else None
    tp       = (entry_px + take_profit) if take_profit is not None else None
    if side == "SHORT":
        sl = (entry_px + stop_loss)    if stop_loss   is not None else None
        tp = (entry_px - take_profit)  if take_profit is not None else None

    best_px  = entry_px
    trail_sl = None

    for bar_idx, row in w.iterrows():
        hi, lo = float(row["high"]), float(row["low"])

        # update best price and unrealised profit
        if side == "LONG":
            best_px = max(best_px, hi)
            unreal  = best_px - entry_px
        else:
            best_px = min(best_px, lo)
            unreal  = entry_px - best_px

        # activate / update trailing stop
        if trail_start is not None and trail_dist is not None and unreal >= trail_start:
            if side == "LONG":
                trail_sl = max(trail_sl or -np.inf, best_px - trail_dist)
            else:
                trail_sl = min(trail_sl or  np.inf, best_px + trail_dist)

        # move SL to breakeven
        if breakeven_at is not None and unreal >= breakeven_at:
            if side == "LONG":
                be_level = entry_px + breakeven_dist
                sl = max(sl, be_level) if sl is not None else be_level
            else:
                be_level = entry_px - breakeven_dist
                sl = min(sl, be_level) if sl is not None else be_level

        # check exits (STOP before TRAIL before TP, worst-case)
        if side == "LONG":
            if sl       is not None and lo <= sl:       return sl,       bar_idx, "STOP"
            if trail_sl is not None and lo <= trail_sl: return trail_sl, bar_idx, "TRAIL"
            if tp       is not None and hi >= tp:       return tp,       bar_idx, "TP"
        else:
            if sl       is not None and hi >= sl:       return sl,       bar_idx, "STOP"
            if trail_sl is not None and hi >= trail_sl: return trail_sl, bar_idx, "TRAIL"
            if tp       is not None and lo <= tp:       return tp,       bar_idx, "TP"

    # no exit triggered — close at last bar
    last = w.iloc[-1]
    return float(last["close"]), w.index[-1], "LOGIC"


# ─────────────────────────────────────────────────────────────────────────────
# Pyramiding variant
# ─────────────────────────────────────────────────────────────────────────────

_PYRAMID_KEYS = {"pyramid_max_adds", "pyramid_add_at", "pyramid_add_qty"}


def simulate_exit_with_pyramiding(
    w: pd.DataFrame,
    side: str,
    entry_px: float,
    **params,
) -> tuple[float, float, str, float]:
    """simulate_exit with optional pyramid add-ons.

    Extra params (stripped before passing to simulate_exit)
    --------------------------------------------------------
    pyramid_max_adds : int   – max number of add-on entries (default 0)
    pyramid_add_at   : list  – profit thresholds (points) at which to add
    pyramid_add_qty  : float – size of each add (relative to initial 1.0)

    Returns
    -------
    (exit_px, avg_entry_px, reason, total_qty)
    """
    max_adds = int(params.get("pyramid_max_adds", 0) or 0)
    add_at   = params.get("pyramid_add_at", [])
    add_qty  = float(params.get("pyramid_add_qty", 1.0) or 1.0)

    if isinstance(add_at, (int, float)):
        add_at = [float(add_at)]
    else:
        add_at = [float(x) for x in add_at]
    add_at = add_at[:max_adds]

    total_qty    = 1.0
    avg_entry_px = float(entry_px)

    if max_adds > 0 and add_at:
        base = float(entry_px)
        for thr in add_at:
            for i in range(1, len(w)):
                px = float(w["close"].iloc[i])
                hit = (px >= base + thr) if side == "LONG" else (px <= base - thr)
                if hit:
                    avg_entry_px = (avg_entry_px * total_qty + px * add_qty) / (total_qty + add_qty)
                    total_qty   += add_qty
                    break
            else:
                break   # threshold not reached; no further adds

    exit_params = {k: v for k, v in params.items() if k not in _PYRAMID_KEYS}
    exit_px, _, reason = simulate_exit(w=w, side=side, entry_px=avg_entry_px, **exit_params)

    return float(exit_px), float(avg_entry_px), reason, float(total_qty)


# ─────────────────────────────────────────────────────────────────────────────
# Batch exit runner
# ─────────────────────────────────────────────────────────────────────────────

def run_exit_strategies(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    exit_strategies: Dict[str, Dict[str, Any]],
    entry_dt_col:    str  = "entry_dt",
    time_bucket_col: str  = "time_bucket",
    use_pyramiding:  bool = False,
    max_fwd_bars:    int  = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every exit strategy against every trade and return results.

    Parameters
    ----------
    df              : full bar DataFrame (needs high, low, close, open)
    trades_df       : trade candidates with entry_bar_i, entry_px, side,
                      entry_dt, time_bucket
    exit_strategies : dict of {name: params}. Entries starting with "_" are skipped.
    use_pyramiding  : if True, use simulate_exit_with_pyramiding
    max_fwd_bars    : how many bars forward to look from entry

    Returns
    -------
    trades_all : one row per (trade × strategy)
    summary    : grouped by (strategy × time_bucket) with pnl/dd/win_rate
    """
    req = ["entry_bar_i", "entry_px", "side", entry_dt_col, time_bucket_col]
    missing = [c for c in req if c not in trades_df.columns]
    if missing:
        raise KeyError(f"trades_df missing: {missing}")

    tb = trades_df.copy()
    tb["hour"] = pd.to_datetime(tb[entry_dt_col]).dt.hour

    rows = []

    for strat_name, params in exit_strategies.items():
        if strat_name.startswith("_"):
            continue

        exit_params = {k: v for k, v in params.items() if k not in _PYRAMID_KEYS}

        for _, t in tb.iterrows():
            start = int(t["entry_bar_i"])
            end   = min(start + max_fwd_bars, len(df) - 1)
            w     = df.iloc[start : end + 1]

            if use_pyramiding:
                exit_px, avg_entry, reason, qty = simulate_exit_with_pyramiding(
                    w=w, side=t["side"], entry_px=float(t["entry_px"]), **params
                )
            else:
                exit_px, _, reason = simulate_exit(
                    w=w, side=t["side"], entry_px=float(t["entry_px"]), **exit_params
                )
                avg_entry = float(t["entry_px"])
                qty       = 1.0

            pnl_pts = (exit_px - avg_entry) if t["side"] == "LONG" else (avg_entry - exit_px)
            pnl     = pnl_pts * qty

            rows.append({
                "strategy":     strat_name,
                "time_bucket":  t[time_bucket_col],
                "hour":         int(t["hour"]),
                "side":         t["side"],
                "entry_dt":     t[entry_dt_col],
                "entry_bar_i":  int(t["entry_bar_i"]),
                "entry_px":     float(t["entry_px"]),
                "exit_px":      float(exit_px),
                "avg_entry_px": float(avg_entry),
                "qty":          float(qty),
                "pnl_pts":      float(pnl_pts),
                "pnl":          float(pnl),
                "reason":       reason,
            })

    trades_all = pd.DataFrame(rows)

    if trades_all.empty:
        return trades_all, pd.DataFrame()

    trades_all = trades_all.sort_values(["strategy", "time_bucket", "entry_dt"])

    def _max_dd(s: pd.Series) -> float:
        eq = s.cumsum()
        return float((eq - eq.cummax()).min())

    summary = (
        trades_all
        .groupby(["strategy", "time_bucket"], sort=False)
        .agg(
            trades   = ("pnl", "count"),
            pnl_sum  = ("pnl", "sum"),
            pnl_mean = ("pnl", "mean"),
            win_rate = ("pnl", lambda x: float((x >= 0).mean())),
            max_dd   = ("pnl", _max_dd),
        )
        .reset_index()
    )

    summary["pnl_to_dd"] = summary.apply(
        lambda r: (r["pnl_sum"] / -r["max_dd"]) if r["max_dd"] < 0 else np.nan,
        axis=1,
    )

    summary = summary.sort_values(["pnl_to_dd", "pnl_sum"], ascending=[False, False])

    return trades_all, summary


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def tail_stats(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) == 0:
        return pd.Series({"p10": pd.NA, "p50": pd.NA, "p90": pd.NA,
                          "min": pd.NA, "max": pd.NA, "skew": pd.NA})
    return pd.Series({
        "p10":  s.quantile(0.10),
        "p50":  s.quantile(0.50),
        "p90":  s.quantile(0.90),
        "min":  s.min(),
        "max":  s.max(),
        "skew": s.skew(),
    })


def summarize_trades_grouped(
    trades_df: pd.DataFrame,
    groupby_cols: Union[str, Sequence[str]],
    pnl_col:  str = "pnl",
    mae_col:  str = "mae",
    mfe_col:  str = "mfe",
    include_excursions:  bool              = True,
    excursion_quantiles: Sequence[float]   = (0.5, 0.9),
    sort_by:   str  = "pnl_sum",
    ascending: bool = False,
) -> pd.DataFrame:
    """Grouped P&L and excursion summary.

    Returns one row per group with trades, pnl_sum, pnl_mean, win_rate,
    pnl distribution quantiles, and optional MAE/MFE quantiles.
    """
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    df = trades_df.copy()

    missing = [c for c in list(groupby_cols) + [pnl_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    has_mae = mae_col in df.columns
    has_mfe = mfe_col in df.columns

    if include_excursions and has_mae:
        df["mae_abs"] = df[mae_col].abs()

    g = df.groupby(list(groupby_cols), dropna=False)

    base = g.agg(
        trades   = (pnl_col, "count"),
        pnl_sum  = (pnl_col, "sum"),
        pnl_mean = (pnl_col, "mean"),
        win_rate = (pnl_col, lambda s: (s > 0).mean() if len(s) else pd.NA),
    )

    pnl_dist = g[pnl_col].apply(tail_stats).unstack()
    out = base.merge(pnl_dist, left_index=True, right_index=True)

    if include_excursions:
        if has_mae:
            q = g["mae_abs"].quantile(list(excursion_quantiles)).unstack()
            q = q.rename(columns={v: f"mae_abs_q{int(v*100):02d}" for v in q.columns})
            out = out.merge(q, left_index=True, right_index=True)
        if has_mfe:
            q = g[mfe_col].quantile(list(excursion_quantiles)).unstack()
            q = q.rename(columns={v: f"mfe_q{int(v*100):02d}" for v in q.columns})
            out = out.merge(q, left_index=True, right_index=True)

    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)

    return out.reset_index()


def summarize_side_and_time_buckets(
    trades_df: pd.DataFrame,
    pnl_col:         str                = "pnl",
    side_col:        str                = "side",
    time_bucket_col: str                = "time_bucket",
    mae_col:         str                = "mae",
    mfe_col:         str                = "mfe",
    include_excursions:  bool           = True,
    excursion_quantiles: Sequence[float] = (0.5, 0.9),
    sort_by:   str  = "pnl_sum",
    ascending: bool = False,
    export_prefix: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper returning by_side / by_time / by_side_time summaries.

    If export_prefix is given, also saves three CSVs:
        <export_prefix>_by_side.csv
        <export_prefix>_by_time.csv
        <export_prefix>_by_side_time.csv
    """
    for c in [pnl_col, side_col, time_bucket_col]:
        if c not in trades_df.columns:
            raise ValueError(f"Column '{c}' not in trades_df. Available: {list(trades_df.columns)}")

    kw = dict(
        pnl_col=pnl_col, mae_col=mae_col, mfe_col=mfe_col,
        include_excursions=include_excursions,
        excursion_quantiles=excursion_quantiles,
        sort_by=sort_by, ascending=ascending,
    )

    by_side      = summarize_trades_grouped(trades_df, side_col, **kw)
    by_time      = summarize_trades_grouped(trades_df, time_bucket_col, **kw)
    by_side_time = summarize_trades_grouped(trades_df, [side_col, time_bucket_col], **kw)

    if export_prefix:
        by_side.to_csv(f"{export_prefix}_by_side.csv", index=False)
        by_time.to_csv(f"{export_prefix}_by_time.csv", index=False)
        by_side_time.to_csv(f"{export_prefix}_by_side_time.csv", index=False)

    return {"by_side": by_side, "by_time": by_time, "by_side_time": by_side_time}
