from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from renko_system.io import list_input_files, load_df, save_parquet
from renko_system.trades_mark import mark_trades_into_df_split
from renko_system.db import append_trades


def run_one(path: Path, out_dir: Path, write_sqlite: bool, db_path: Path | None) -> dict:
    df = load_df(path)
    from renko_system.preprocess import preprocess_tv_export

    df = preprocess_tv_export(df)

    df_marked, lor_trades, verify_trades = mark_trades_into_df_split(
        df,
        price_col="close",
        lor_buy_col="lor_buy_sig",
        lor_sell_col="lor_sell_sig",
        v_buy_col="v_buy_sig",
        v_sell_col="v_sell_sig",
        # adjust if your df uses different names:
        flip_long_col="lor_flip_long",
        flip_short_col="lor_flip_short",
        kern_col="kern_est",
        state_col="last_state_code",
    )

    stem = path.stem

    # Parquet outputs
    out_marked = out_dir / f"{stem}__marked.parquet"
    out_lor = out_dir / f"{stem}__lor_trades.parquet"
    out_ver = out_dir / f"{stem}__verify_trades.parquet"

    save_parquet(df_marked, out_marked, reset_index=False)
    save_parquet(lor_trades, out_lor, reset_index=False)
    save_parquet(verify_trades, out_ver, reset_index=False)

    # CSV outputs (human review)
    out_marked_csv = out_dir / f"{stem}__marked.csv"
    out_lor_csv = out_dir / f"{stem}__lor_trades.csv"
    out_ver_csv = out_dir / f"{stem}__verify_trades.csv"

    # CSV note: for readability, do NOT write the pandas index unless you want it.
    df_marked.to_csv(out_marked_csv, index=False, float_format="%.5f")
    lor_trades.to_csv(out_lor_csv, index=False, float_format="%.5f")
    verify_trades.to_csv(out_ver_csv, index=False, float_format="%.5f")

    inserted = 0
    if write_sqlite and db_path is not None:
        inserted += append_trades(db_path, lor_trades, dataset=stem)
        inserted += append_trades(db_path, verify_trades, dataset=stem)

    return {
        "file": str(path),
        "rows": len(df),
        "marked_out": str(out_marked),
        "lor_trades": len(lor_trades),
        "verify_trades": len(verify_trades),
        "sqlite_inserted": inserted,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/out"))
    ap.add_argument("--sqlite", action="store_true", help="Also append trades to sqlite db")
    ap.add_argument("--db", type=Path, default=Path("data/out/trades.sqlite"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    files = list_input_files(args.data_raw)
    if not files:
        raise SystemExit(f"No input files found in {args.data_raw}")

    summaries = []
    for f in files:
        try:
            summaries.append(run_one(f, args.out, args.sqlite, args.db if args.sqlite else None))
        except Exception as e:
            summaries.append({"file": str(f), "error": repr(e)})

    # Save run summary
    summary_df = pd.DataFrame(summaries)
    summary_path = args.out / "run_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    print(summary_df)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
