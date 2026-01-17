"""Data loading and basic cleaning utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.data")


class DataLoader:
    """Load raw OHLCV data and apply minimal cleaning."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_csv(self, path: Path) -> pd.DataFrame:
        if not Path(path).exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path, on_bad_lines="skip")
        df.columns = df.columns.str.strip().str.lower()
        return df

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Applying basic cleaning to data")
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df.dropna(subset=price_cols).copy()
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid price data")
        # Normalize/parse time column if present
        time_col_found = None
        for time_col in ["time", "date", "datetime", "timestamp", "price"]:
            if time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                    # Check if conversion was successful
                    if df[time_col].notna().sum() > len(df) * 0.5:
                        time_col_found = time_col
                        break
                except Exception:
                    pass

        if time_col_found:
            # Drop rows where datetime parsing failed
            df = df.dropna(subset=[time_col_found])
            # Rename to standard "time" column if needed
            if time_col_found != "time":
                df = df.rename(columns={time_col_found: "time"})
                logger.debug(f"Renamed '{time_col_found}' column to 'time'")
            df = df.sort_values("time").reset_index(drop=True)

        return df

    def load(self, raw_path: Path) -> pd.DataFrame:
        df = self.load_csv(raw_path)
        df = self.basic_clean(df)
        return df
