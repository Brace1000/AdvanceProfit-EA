#!/usr/bin/env python3
"""Training pipeline CLI entry point.

Runs the end-to-end pipeline:
  1) Load and clean raw data
  2) Engineer features
  3) Optional HPO with Optuna
  4) Train XGBoost model with proper 3-way split
  5) Evaluate on holdout set (true out-of-sample)

Usage:
  python run_all.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.logger import get_logger
from src.config import get_config
from src.pipelines.training import TrainingPipeline

logger = get_logger(__name__)

LOG_FILE = Path(__file__).parent / "logs" / "run_all.log"
LOG_FILE.parent.mkdir(exist_ok=True)


def main(config_path: Optional[Path] = None) -> None:
    try:
        config = get_config(config_path)
        logger.info("=" * 60)
        logger.info("EUR/USD Trading Model - Training Pipeline")
        logger.info("=" * 60)

        pipeline = TrainingPipeline(config._config)
        results = pipeline.run()

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

        dist = results['class_distribution']
        gap = results['train_accuracy'] - results['holdout_accuracy']

        # Build summary lines
        lines = [
            "=" * 60,
            "TRAINING SUMMARY",
            "=" * 60,
            f"Model: {results['model_path']}",
            f"Features ({len(results['features_used'])}): {results['features_used']}",
            "-" * 60,
            "CLASS DISTRIBUTION (3-class, Sell-only trading):",
            f"  Sell:   {dist['sell']:.1%}",
            f"  Range:  {dist['range']:.1%}",
            f"  Buy:    {dist['buy']:.1%}",
            "-" * 60,
            "ACCURACY:",
            f"  Train:    {results['train_accuracy']:.2%}",
            f"  Holdout:  {results['holdout_accuracy']:.2%}",
            f"  Gap:      {gap:.2%}",
            "-" * 60,
            "BACKTEST (holdout, Sell-only, TP/SL simulation):",
            f"  Win rate:           {results['win_rate']:.2%}",
            f"  Total pips:         {results['total_pips']:+.0f}",
            f"  Sharpe (per-bar):   {results['sharpe_bar']:.2f}",
            f"  Sharpe (per-trade): {results['sharpe_trade']:.2f}",
            f"  Max DD:             {results['max_drawdown']:.2%}",
            "=" * 60,
        ]

        if gap > 0.15:
            lines.append(f"WARNING: Potential overfitting (gap={gap:.2%})")

        # Log to console
        for line in lines:
            logger.info(line)

        # Write clean summary to log file
        LOG_FILE.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Summary saved to {LOG_FILE}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid data or configuration: {e}")
        raise
    except Exception:
        logger.exception("Unexpected error during pipeline execution")
        raise


if __name__ == "__main__":
    main()
