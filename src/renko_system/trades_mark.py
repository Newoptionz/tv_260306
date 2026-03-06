from __future__ import annotations

import pandas as pd

from .trades_lor import mark_lor_trades_into_df
from .trades_verify import mark_verify_trades_into_df
from .trades_kern import mark_kern_trades_into_df


def mark_trades_into_df_split(
    df: pd.DataFrame,
    **kwargs,
):
    """Run LOR, VERIFY, and KERN marking on one df.

    kwargs are forwarded to each function when parameter names match.

    Returns: (df_marked, lor_trades_df, verify_trades_df, kern_trades_df)
    """
    lor_kwargs  = {k: v for k, v in kwargs.items() if k in mark_lor_trades_into_df.__code__.co_varnames}
    v_kwargs    = {k: v for k, v in kwargs.items() if k in mark_verify_trades_into_df.__code__.co_varnames}
    kern_kwargs = {k: v for k, v in kwargs.items() if k in mark_kern_trades_into_df.__code__.co_varnames}

    df1, lor_trades    = mark_lor_trades_into_df(df, **lor_kwargs)
    df2, verify_trades = mark_verify_trades_into_df(df1, **v_kwargs)
    df3, kern_trades   = mark_kern_trades_into_df(df2, **kern_kwargs)

    return df3, lor_trades, verify_trades, kern_trades
