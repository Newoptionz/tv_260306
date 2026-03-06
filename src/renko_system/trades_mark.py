from __future__ import annotations

import pandas as pd

from .trades_lor import mark_lor_trades_into_df
from .trades_verify import mark_verify_trades_into_df


def mark_trades_into_df_split(
    df: pd.DataFrame,
    **kwargs,
):
    """Run LOR then VERIFY marking on one df.

    kwargs are forwarded to functions when parameter names match.

    Returns: (df_marked, lor_trades_df, verify_trades_df)
    """

    lor_kwargs = {k: v for k, v in kwargs.items() if k in mark_lor_trades_into_df.__code__.co_varnames}
    v_kwargs = {k: v for k, v in kwargs.items() if k in mark_verify_trades_into_df.__code__.co_varnames}

    df1, lor_trades = mark_lor_trades_into_df(df, **lor_kwargs)
    df2, verify_trades = mark_verify_trades_into_df(df1, **v_kwargs)

    return df2, lor_trades, verify_trades
