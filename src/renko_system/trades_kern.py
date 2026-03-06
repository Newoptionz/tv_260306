from __future__ import annotations

"""trades_kern.py
=================
Entry generator for the Kern-Confident system.

Entry logic
-----------
- Forward-fill the last LOR direction from Buy/Sell columns
- On every bar where kern_confident is non-null (green dot):
    - If kern_est is rising  (today > yesterday) AND last LOR dir == LONG  → candidate LONG entry
    - If kern_est is falling (today < yesterday) AND last LOR dir == SHORT → candidate SHORT entry
- Entry executes at the OPEN of the next bar
- Each candidate is independent — the exit simulation engine decides
  whether to skip, hold, or re-enter based on the exit strategy

Default baseline exit (used when running mark_kern_trades_into_df)
-------------------------------------------------------------------
- Exit LONG on first red dot (kern NOT confident) after entry
- Exit SHORT on first red dot after entry
- This is the simplest baseline; the exit simulator can override it.

Columns expected in df (after C8/C10 normalisation)
----------------------------------------------------
  kern_confident          non-null = green dot
  kern not confident dots non-null = red dot
  buy                     LOR buy signal price (non-null = Buy bar)
  sell                    LOR sell signal price (non-null = Sell bar)
  kern_est                smoothed kernel estimate
  open                    bar open price (used as entry fill price)
  state_code              0-3 integer state
"""

import numpy as np
import pandas as pd

from .utils import ensure_bar_i, require_cols, append_event


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_lz_direction(df: pd.DataFrame,
                        buy_col: str = "buy",
                        sell_col: str = "sell") -> pd.Series:
    """Forward-fill last LOR direction from Buy/Sell price columns.

    Returns a Series of "LONG", "SHORT", or NaN (before first signal).
    """
    direction = pd.Series(np.nan, index=df.index, dtype=object)

    buy_flag  = df[buy_col].notna() & (df[buy_col] != 0)
    sell_flag = df[sell_col].notna() & (df[sell_col] != 0)

    direction[buy_flag]  = "LONG"
    direction[sell_flag] = "SHORT"

    # Where both fire on the same bar, sell wins (conservative)
    direction[buy_flag & sell_flag] = "SHORT"

    return direction.ffill()


def _kern_rising(kern: pd.Series) -> pd.Series:
    """True when kern_est is higher than the previous bar."""
    return kern > kern.shift(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def mark_kern_trades_into_df(
    df: pd.DataFrame,
    buy_col:            str = "buy",
    sell_col:           str = "sell",
    kern_col:           str = "kern_est",
    green_col:          str = "kern confident dots",       # non-null = green
    red_col:            str = "kern not confident dots",   # non-null = red
    open_col:           str = "open",
    state_col:          str = "state_code",
    out_prefix:         str = "kern",
    require_kern_align: bool = True,   # if True, kern slope must match direction
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate kern-confident candidate entries and baseline exits.

    Returns
    -------
    df_marked : pd.DataFrame
        Original df with added event/pos/trade_id columns.
    trades_df : pd.DataFrame
        One row per completed trade with entry/exit metadata.
    """

    df = ensure_bar_i(df.copy())

    required = [buy_col, sell_col, kern_col, green_col, open_col, state_col]
    # red_col is optional for baseline exit; warn rather than crash if missing
    has_red = red_col in df.columns
    require_cols(df, required, "mark_kern_trades_into_df")

    event_col    = f"{out_prefix}_event"
    trade_id_col = f"{out_prefix}_trade_id"
    pos_col      = f"{out_prefix}_pos"
    dir_col      = f"{out_prefix}_lz_dir"

    df[event_col]    = ""
    df[trade_id_col] = np.nan
    df[pos_col]      = 0

    # ── Build helper series ───────────────────────────────────────────────────
    lz_dir   = _build_lz_direction(df, buy_col, sell_col)
    df[dir_col] = lz_dir                         # expose for debugging

    kern     = df[kern_col]
    rising   = _kern_rising(kern)
    green    = df[green_col].notna()
    red      = df[red_col].notna() if has_red else pd.Series(False, index=df.index)
    bar_open = df[open_col]

    # ── State capture at signal bar ───────────────────────────────────────────
    state_vals = df[state_col]

    trades: list[dict] = []
    pos       = 0
    trade_id  = 0
    entry_px  = entry_ts = entry_i = None
    entry_kern_dist = None
    entry_state     = None
    entry_lz_dir    = None

    n   = len(df)
    idx = df.index

    for i in range(n - 1):
        j       = i + 1
        ts_sig  = idx[i]
        ts_exec = idx[j]

        is_green = bool(green.iloc[i])
        is_red   = bool(red.iloc[i])
        direction = lz_dir.iloc[i]           # LONG / SHORT / NaN
        kern_up   = bool(rising.iloc[i])

        exec_px = float(bar_open.iloc[j])    # fill = next bar open

        # ── Determine signal direction ────────────────────────────────────────
        if is_green and pd.notna(direction):
            if direction == "LONG":
                if (not require_kern_align) or kern_up:
                    sig_dir = "LONG"
                else:
                    sig_dir = None
            else:  # SHORT
                if (not require_kern_align) or (not kern_up):
                    sig_dir = "SHORT"
                else:
                    sig_dir = None
        else:
            sig_dir = None

        # ── Entry (only when flat) ────────────────────────────────────────────
        if pos == 0 and sig_dir is not None:
            pos      = 1 if sig_dir == "LONG" else -1
            trade_id += 1

            entry_px    = exec_px
            entry_ts    = ts_exec
            entry_i     = j
            entry_lz_dir = direction

            try:
                entry_kern_dist = abs(float(df[kern_col].iloc[i]) - float(df[open_col].iloc[i]))
            except Exception:
                entry_kern_dist = np.nan

            entry_state = state_vals.iloc[i]

            lbl = "ENTRY_LONG" if pos == 1 else "ENTRY_SHORT"
            append_event(df, ts_exec, event_col, lbl)

        # ── Baseline exit: first red dot after entry ──────────────────────────
        elif pos != 0 and is_red:
            exit_px = exec_px
            exit_ts = ts_exec
            exit_i  = j

            side = "LONG" if pos == 1 else "SHORT"
            pnl  = (exit_px - entry_px) if pos == 1 else (entry_px - exit_px)

            trades.append(dict(
                stream        = "KERN",
                trade_id      = trade_id,
                side          = side,
                entry_time    = entry_ts,
                exit_time     = exit_ts,
                entry_bar_i   = entry_i,
                exit_bar_i    = exit_i,
                entry_px      = entry_px,
                exit_px       = exit_px,
                pnl           = pnl,
                bars_held     = exit_i - entry_i,
                kern_dist_sig = entry_kern_dist,
                state_sig     = entry_state,
                lz_dir        = entry_lz_dir,
            ))

            lbl = "EXIT_LONG" if pos == 1 else "EXIT_SHORT"
            append_event(df, exit_ts, event_col, lbl)

            pos         = 0
            entry_px    = entry_ts = entry_i = None
            entry_kern_dist = entry_state = entry_lz_dir = None

        # ── Mark position on exec bar ─────────────────────────────────────────
        df.at[ts_exec, pos_col] = pos
        if pos != 0:
            df.at[ts_exec, trade_id_col] = trade_id

    return df, pd.DataFrame(trades)
