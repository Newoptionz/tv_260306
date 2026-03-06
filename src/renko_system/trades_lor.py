from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ensure_bar_i, require_cols, to_bool, iter_signal_exec_pairs, append_event


def mark_lor_trades_into_df(
    df: pd.DataFrame,
    price_col: str = "close",
    lor_buy_col: str = "lor_buy_sig",
    lor_sell_col: str = "lor_sell_sig",
    kern_col: str = "kern_est",
    state_col: str = "last_state_code",
    out_prefix: str = "lor",
):
    """Mark LOR flip-cycle trades.

    - Enter on lor_buy/lor_sell (exec next bar)
    - When in LONG, a sell exits long and immediately flips to short (on same exec bar)
    - When in SHORT, a buy exits short and flips to long

    Returns (df_marked, trades_df)
    """

    df = ensure_bar_i(df)

    event_col = f"{out_prefix}_event"
    trade_id_col = f"{out_prefix}_trade_id"
    pos_col = f"{out_prefix}_pos"

    require_cols(
        df,
        [price_col, lor_buy_col, lor_sell_col, kern_col, state_col],
        "mark_lor_trades_into_df",
    )

    df[event_col] = ""
    df[trade_id_col] = np.nan
    df[pos_col] = 0

    trades: list[dict] = []

    pos = 0
    entry_px = entry_ts = entry_i = None
    trade_id = 0
    entry_kern_dist_sig = None
    entry_state_sig = None

    for i, ts_sig, row_sig, j, ts_exec, row_exec in iter_signal_exec_pairs(df):
        # Signal metadata
        try:
            px_sig = float(row_sig[price_col])
            kern_sig = float(row_sig[kern_col])
            kern_dist_sig = abs(px_sig - kern_sig)
        except Exception:
            kern_dist_sig = np.nan

        state_at_sig = row_sig[state_col]

        # Exec price
        try:
            px_exec = float(row_exec[price_col])
        except Exception:
            df.at[ts_exec, pos_col] = pos
            continue

        buy = to_bool(row_sig[lor_buy_col])
        sell = to_bool(row_sig[lor_sell_col])

        if pos == 0:
            if buy and not sell:
                pos = 1
                trade_id += 1
                entry_kern_dist_sig = kern_dist_sig
                entry_state_sig = state_at_sig
                entry_px, entry_ts, entry_i = px_exec, ts_exec, j
                append_event(df, ts_exec, event_col, "ENTRY_LONG")

            elif sell and not buy:
                pos = -1
                trade_id += 1
                entry_kern_dist_sig = kern_dist_sig
                entry_state_sig = state_at_sig
                entry_px, entry_ts, entry_i = px_exec, ts_exec, j
                append_event(df, ts_exec, event_col, "ENTRY_SHORT")

        elif pos == 1:
            if sell:
                exit_px, exit_ts, exit_i = px_exec, ts_exec, j
                pnl = exit_px - entry_px

                trades.append(
                    dict(
                        stream="LOR",
                        trade_id=trade_id,
                        side="LONG",
                        entry_time=entry_ts,
                        exit_time=exit_ts,
                        entry_bar_i=entry_i,
                        exit_bar_i=exit_i,
                        entry_px=entry_px,
                        exit_px=exit_px,
                        pnl=pnl,
                        bars_held=exit_i - entry_i,
                        kern_dist_sig=entry_kern_dist_sig,
                        state_sig=entry_state_sig,
                    )
                )

                append_event(df, exit_ts, event_col, "EXIT_LONG")

                # flip to short
                pos = -1
                trade_id += 1
                entry_kern_dist_sig = kern_dist_sig
                entry_state_sig = state_at_sig
                entry_px, entry_ts, entry_i = exit_px, exit_ts, exit_i
                append_event(df, exit_ts, event_col, "ENTRY_SHORT")

        elif pos == -1:
            if buy:
                exit_px, exit_ts, exit_i = px_exec, ts_exec, j
                pnl = entry_px - exit_px

                trades.append(
                    dict(
                        stream="LOR",
                        trade_id=trade_id,
                        side="SHORT",
                        entry_time=entry_ts,
                        exit_time=exit_ts,
                        entry_bar_i=entry_i,
                        exit_bar_i=exit_i,
                        entry_px=entry_px,
                        exit_px=exit_px,
                        pnl=pnl,
                        bars_held=exit_i - entry_i,
                        kern_dist_sig=entry_kern_dist_sig,
                        state_sig=entry_state_sig,
                    )
                )

                append_event(df, exit_ts, event_col, "EXIT_SHORT")

                # flip to long
                pos = 1
                trade_id += 1
                entry_kern_dist_sig = kern_dist_sig
                entry_state_sig = state_at_sig
                entry_px, entry_ts, entry_i = exit_px, exit_ts, exit_i
                append_event(df, exit_ts, event_col, "ENTRY_LONG")

        df.at[ts_exec, pos_col] = pos
        if pos != 0:
            df.at[ts_exec, trade_id_col] = trade_id

    return df, pd.DataFrame(trades)
