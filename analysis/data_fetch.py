#!/usr/bin/env python3
"""
MetaTrader 5 Historical Data Fetcher using centralized config and logger.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_config
from src.logger import get_logger

logger = get_logger("trading_bot.data")

try:
    import MetaTrader5 as mt5
except ImportError:
    logger.error("MetaTrader5 package not found. Install with: pip install MetaTrader5")
    sys.exit(1)


def parse_timeframe(tf: str):
    tf = tf.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe '{tf}'. Use one of: {', '.join(mapping.keys())}")
    return mapping[tf]


def parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    from datetime import datetime as _dt

    try:
        return _dt.strptime(d, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date '{d}'. Use YYYY-MM-DD.") from e


def init_mt5(terminal_path: Optional[str]) -> None:
    ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    if not ok:
        code, msg = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: [{code}] {msg}")


def maybe_login(login: Optional[int], password: Optional[str], server: Optional[str]) -> None:
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 login failed: [{code}] {msg}")


def fetch_rates(symbol: str, timeframe, start: Optional[datetime], end: Optional[datetime], count: Optional[int]):
    mt5.symbol_select(symbol, True)
    if count is not None:
        if count <= 0:
            raise ValueError("count must be > 0")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    else:
        if not start or not end:
            raise ValueError("Either provide --count, or both --start and --end")
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        code, msg = mt5.last_error()
        raise RuntimeError(f"No data returned for {symbol}. MT5 last error: [{code}] {msg}")
    return rates


def rates_to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    df.columns = [str(c).lower() for c in df.columns]
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' from MT5 result.")
    keep = [c for c in ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in df.columns]
    return df[keep]


def save_csv(df: pd.DataFrame, path: Path, append: bool) -> None:
    path = Path(path)
    if append and path.exists():
        try:
            old = pd.read_csv(path)
            if "time" in old.columns:
                try:
                    old["time"] = pd.to_datetime(old["time"])
                except Exception:
                    pass
            merged = pd.concat([old, df], ignore_index=True)
            if "time" in merged.columns:
                merged = merged.drop_duplicates(subset=["time"]).sort_values("time")
            merged.to_csv(path, index=False)
            return
        except Exception:
            pass
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Download historical data from MetaTrader 5 to CSV")
    cfg = get_config()
    default_symbol = cfg.get("data_collection.symbol", "EURUSD")
    default_tf = cfg.get("data_collection.timeframe", "D1")

    parser.add_argument("--symbol", default=default_symbol, help=f"Symbol to download (default: {default_symbol})")
    parser.add_argument("--timeframe", default=default_tf, help=f"Timeframe (default: {default_tf})")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, help="Number of most recent bars to fetch")
    group.add_argument("--start", type=str, help="Start date YYYY-MM-DD for range fetch")

    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD for range fetch")

    parser.add_argument("--csv", default=None, help="Output CSV path (default: {SYMBOL}_{TIMEFRAME}_raw.csv)")
    parser.add_argument("--append", action="store_true", help="Append/merge with existing CSV by time")
    parser.add_argument("--terminal-path", type=str, help="Path to terminal64.exe if MT5 cannot be auto-initialized")
    parser.add_argument("--login", type=int, help="MT5 account login number (optional)")
    parser.add_argument("--password", type=str, help="MT5 account password (optional)")
    parser.add_argument("--server", type=str, help="MT5 trade server name (optional)")

    args = parser.parse_args()

    # Construct CSV filename from symbol/timeframe if not explicitly provided
    if args.csv is None:
        args.csv = f"{args.symbol}_{args.timeframe}_raw.csv"

    try:
        timeframe = parse_timeframe(args.timeframe)
        start_dt = parse_date(args.start) if args.start else None
        end_dt = parse_date(args.end) if args.end else None

        init_mt5(args.terminal_path)
        maybe_login(args.login, args.password, args.server)

        logger.info(f"Fetching {args.symbol} {args.timeframe} data...")
        rates = fetch_rates(args.symbol, timeframe, start_dt, end_dt, args.count)
        df = rates_to_df(rates)

        out_path = Path(args.csv)
        save_csv(df, out_path, append=args.append)
        logger.info(f"Saved {len(df)} rows to {out_path}")
    except Exception as e:
        logger.exception("Data fetch error")
        sys.exit(1)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
