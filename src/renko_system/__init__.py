"""Renko/Lorentzian trade marking and exit simulation toolkit."""

from .instruments import INSTRUMENT_CONFIG, detect_instrument_from_filename
from .preprocess import preprocess_tv_export, parse_time_column
from .trades_mark import mark_trades_into_df_split
from .trades_lor import mark_lor_trades_into_df
from .trades_verify import mark_verify_trades_into_df
from .trades_kern import mark_kern_trades_into_df
from .exits import (
    add_mae_mfe,
    trade_core_calibration,
    build_exit_suite_from_cal,
    simulate_exit,
    simulate_exit_with_pyramiding,
    run_exit_strategies,
    summarize_trades_grouped,
    summarize_side_and_time_buckets,
)
