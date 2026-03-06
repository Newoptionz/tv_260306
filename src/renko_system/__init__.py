"""Renko/Lorentzian + Verify trade marking toolkit.

Keep notebooks thin: import from this package and run scripts in /scripts.
"""

from .config import INSTRUMENT_CONFIG, detect_instrument_from_filename
from .trades_mark import mark_trades_into_df_split
from .trades_lor import mark_lor_trades_into_df
from .trades_verify import mark_verify_trades_into_df
