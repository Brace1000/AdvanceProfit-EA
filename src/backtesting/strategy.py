"""Backtesting scaffolding: simple next-day return strategy based on classifier predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.backtest")


@dataclass
class BacktestResult:
    trades: int
    win_rate: float
    cum_return: float
    sharpe: float
    max_drawdown: float


class SimpleNextDayStrategy:
    """
    Interprets class predictions: 2 -> long, 0 -> short, 1 -> flat.
    Calculates next-day close-to-close PnL with a fixed trading cost.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_cost = float(config.get("backtesting.trading_cost", 0.0001))

    def run(self, df: pd.DataFrame, preds: np.ndarray) -> BacktestResult:
        df = df.copy().reset_index(drop=True)
        df["pred"] = preds
        df["position"] = df["pred"].map({0: -1, 1: 0, 2: 1})
        # Next-day close return
        df["next_ret"] = df["close"].pct_change(-1) * -1
        df = df.iloc[:-1]
        df["trade_cost"] = (df["position"] != 0).astype(float) * self.trading_cost
        df["pnl"] = df["position"] * df["next_ret"] - df["trade_cost"]

        trades = int((df["position"] != 0).sum())
        wins = int((df["pnl"] > 0).sum())
        losses = int((df["pnl"] < 0).sum())
        win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else 0.0
        cum_return = float((1 + df["pnl"]).prod() - 1)
        avg = float(df["pnl"].mean())
        std = float(df["pnl"].std(ddof=1))
        sharpe = float((avg / std) * np.sqrt(252)) if std and std > 0 else 0.0

        curve = (1 + df["pnl"]).cumprod()
        dd = (curve / curve.cummax()) - 1
        max_dd = float(dd.min()) if len(dd) else 0.0

        return BacktestResult(trades=trades, win_rate=win_rate, cum_return=cum_return, sharpe=sharpe, max_drawdown=max_dd)
