"""Dataset validation helpers."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.data")


class DataValidator:
    """Validate raw/processed datasets for required columns and sanity checks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate_ohlc(self, df: pd.DataFrame) -> None:
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLC columns: {missing}")
        if (df["high"] < df["low"]).any():
            raise ValueError("Found rows where high < low")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            logger.warning("Detected non-positive price values")

    def validate_features(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if df[feature_cols].isna().any().any():
            raise ValueError("NaN present in feature columns after engineering")
