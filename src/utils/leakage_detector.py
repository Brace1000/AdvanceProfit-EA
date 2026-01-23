"""Leakage detection utilities for time series feature validation."""
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.logger import get_logger

logger = get_logger("trading_bot.leakage")


class LeakageDetector:
    """
    Detects potential data leakage in time series features.

    Uses three key tests:
    1. Correlation Decay: Features should lose predictive power over time
    2. Temporal Consistency: Features should predict forward better than backward
    3. Rolling Statistics Stability: Normalized features shouldn't use global stats
    """

    def __init__(self, df: pd.DataFrame, label_col: str = "label"):
        """
        Initialize detector with feature dataframe.

        Args:
            df: DataFrame with features and labels
            label_col: Name of the label column
        """
        self.df = df.copy()
        self.label_col = label_col

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataframe")

    def _correlation_decay_test(
        self, feature: str, max_lag: int = 5
    ) -> Dict[str, Any]:
        """
        Test if feature correlation decays with increasing lag.

        A leak-free feature should have:
        - Strong correlation with label[t+1]
        - Rapidly decaying correlation with label[t+2], label[t+3], etc.

        Returns:
            dict with correlations at each lag and decay_ratio
        """
        correlations = {}
        for lag in range(1, max_lag + 1):
            future_label = self.df[self.label_col].shift(-lag)
            corr = abs(self.df[feature].corr(future_label))
            correlations[f"lag_{lag}"] = corr

        # Decay ratio: correlation at lag 1 should be much higher than lag 5
        lag1_corr = correlations["lag_1"]
        lag5_corr = correlations[f"lag_{max_lag}"]
        decay_ratio = lag1_corr / (lag5_corr + 1e-10)

        return {
            "correlations": correlations,
            "decay_ratio": decay_ratio,
            "lag1": lag1_corr,
            "lag5": lag5_corr,
        }

    def _temporal_consistency_test(self, feature: str) -> Dict[str, float]:
        """
        Test if feature predicts future better than it describes past.

        A leak-free feature should have:
        - forward_corr > backward_corr

        Returns:
            dict with forward_corr, backward_corr, and temporal_ratio
        """
        forward_corr = abs(self.df[feature].corr(self.df[self.label_col].shift(-1)))
        backward_corr = abs(self.df[feature].corr(self.df[self.label_col].shift(1)))
        temporal_ratio = forward_corr / (backward_corr + 1e-10)

        return {
            "forward_corr": forward_corr,
            "backward_corr": backward_corr,
            "temporal_ratio": temporal_ratio,
        }

    def _rolling_stats_test(
        self, feature: str, window: int = 24
    ) -> Dict[str, float]:
        """
        Test if feature uses rolling (causal) vs global statistics.

        For Z-scored features, the rolling std should vary over time.
        If std_cv (coefficient of variation) is very low, feature likely
        uses global mean/std.

        Returns:
            dict with std_cv (coefficient of variation of rolling std)
        """
        rolling_std = self.df[feature].rolling(window).std()
        std_mean = rolling_std.mean()
        std_std = rolling_std.std()
        std_cv = std_std / (std_mean + 1e-10)

        return {
            "std_cv": std_cv,
            "rolling_std_mean": std_mean,
            "rolling_std_std": std_std,
        }

    @staticmethod
    def _is_calendar_feature(feature: str) -> bool:
        """
        Identify calendar/temporal context features that shouldn't be tested for leakage.

        Calendar features are static identifiers (hour=14 is always 14), not time-series
        data that can leak future information. These provide market structure context
        (e.g., London session has more volatility than Asian session).

        Args:
            feature: Feature name

        Returns:
            True if feature is a calendar feature
        """
        calendar_prefixes = ("session_", "dow_")
        calendar_exact = ("hour",)

        return (
            feature.startswith(calendar_prefixes) or
            feature in calendar_exact
        )

    def run_tests(
        self, feature_cols: List[str], max_lag: int = 5
    ) -> List[str]:
        """
        Run all leakage tests on specified features.

        Calendar features (hour, session_*, dow_*) are automatically skipped as they
        are temporal context identifiers, not time-series features that can leak.

        Args:
            feature_cols: List of feature column names to test
            max_lag: Maximum lag for correlation decay test

        Returns:
            List of warning messages for features that fail tests
        """
        warnings = []

        # Separate calendar from time-series features
        calendar_features = [f for f in feature_cols if self._is_calendar_feature(f)]
        ts_features = [f for f in feature_cols if not self._is_calendar_feature(f)]

        if calendar_features:
            logger.info(f"Skipping {len(calendar_features)} calendar features: {calendar_features}")

        logger.info(f"Running leakage detection on {len(ts_features)} time-series features...")

        for feat in ts_features:
            if feat not in self.df.columns:
                logger.warning(f"Feature '{feat}' not found in dataframe, skipping")
                continue

            # 1. Correlation decay test
            decay_result = self._correlation_decay_test(feat, max_lag)
            decay_ratio = decay_result["decay_ratio"]

            # 2. Temporal consistency test
            temporal_result = self._temporal_consistency_test(feat)
            temporal_ratio = temporal_result["temporal_ratio"]

            # 3. Rolling stats test
            stats_result = self._rolling_stats_test(feat)
            std_cv = stats_result["std_cv"]

            # Log results
            logger.debug(
                f"{feat}: decay_ratio={decay_ratio:.2f}, "
                f"temporal_ratio={temporal_ratio:.2f}, std_cv={std_cv:.3f}"
            )

            # Check thresholds and add warnings
            if decay_ratio < 1.5:
                warnings.append(
                    f"{feat}: Correlation doesn't decay (decay_ratio={decay_ratio:.2f}, "
                    f"expected >1.5). Possible look-ahead bias."
                )

            if temporal_ratio < 1.0:
                warnings.append(
                    f"{feat}: Backward correlation stronger than forward "
                    f"(ratio={temporal_ratio:.2f}). Likely using future data."
                )

            if std_cv < 0.05:
                warnings.append(
                    f"{feat}: Constant rolling std (cv={std_cv:.3f}, expected >0.05). "
                    f"Possible global normalization."
                )

        if warnings:
            logger.warning(f"⚠️  {len(warnings)} potential leakage issues detected")
        else:
            logger.info("✅ All features passed leakage tests")

        return warnings

    def plot_correlation_decay(
        self,
        feature_cols: List[str],
        max_lag: int = 5,
        output_path: str = "correlation_decay_plot.png",
    ) -> None:
        """
        Generate visualization of correlation decay for each feature.

        Calendar features are automatically skipped as they are temporal context
        identifiers that don't exhibit time-series decay patterns.

        A flat line indicates potential leakage.
        A steep downward slope indicates a leak-free feature.

        Args:
            feature_cols: List of features to plot
            max_lag: Maximum lag to plot
            output_path: Path to save plot
        """
        # Only plot time-series features
        ts_features = [f for f in feature_cols if not self._is_calendar_feature(f)]

        if not ts_features:
            logger.warning("No time-series features to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for feat in ts_features:
            if feat not in self.df.columns:
                continue

            decay_result = self._correlation_decay_test(feat, max_lag)
            correlations = decay_result["correlations"]

            lags = list(range(1, max_lag + 1))
            corr_values = [correlations[f"lag_{lag}"] for lag in lags]

            ax.plot(lags, corr_values, marker='o', label=feat)

        ax.set_xlabel("Lag (bars into future)")
        ax.set_ylabel("Absolute Correlation with Label")
        ax.set_title("Correlation Decay Test - Flat Lines Indicate Leakage")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max_lag + 1))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation decay plot saved to {output_path}")
