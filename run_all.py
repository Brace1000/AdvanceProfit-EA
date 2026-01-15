#!/usr/bin/env python3
"""Training pipeline CLI entry point.

Runs the end-to-end pipeline:
  1) Load and clean raw data
  2) Engineer features
  3) Optional HPO with Optuna
  4) Train XGBoost model and persist

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
        logger.info(f"Model saved: {results['model_path']}")
        logger.info(f"Train accuracy: {results['train_accuracy']:.2%}")
        logger.info(f"Val accuracy: {results['val_accuracy']:.2%}")
        logger.info(f"Processed dataset: {results['processed_path']}")
        logger.info(f"Features used: {results['features_used']}")
        logger.info("=" * 60)

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
