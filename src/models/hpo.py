"""Optuna-based hyperparameter optimization with two-stage selection."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.logger import get_logger

logger = get_logger("trading_bot.hpo")

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested config value using dot notation."""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Suggest hyperparameters for XGBoost.

    Search space is constrained to prevent overfitting:
    - max_depth: 2-5 (shallow trees generalize better)
    - n_estimators: 50-300 (moderate ensemble size)
    - Strong regularization parameters
    """
    return {
        # Tree structure - kept shallow to prevent overfitting
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),

        # Learning rate - lower values with more trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # Subsampling - helps prevent overfitting
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),

        # Regularization - strong regularization to prevent overfitting
        "gamma": trial.suggest_float("gamma", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),

        # Fixed parameters
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def _compute_sample_weights(y: np.ndarray) -> Optional[np.ndarray]:
    """Compute class-balanced sample weights."""
    try:
        classes = np.array([0, 1, 2], dtype=int)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return class_weights[y]
    except Exception:
        return None


def _backtest_sharpe(preds: np.ndarray, close_val: np.ndarray, commission: float = 0.0001) -> float:
    """
    Compute annualised Sharpe ratio from predictions and close prices.

    Maps: class 2 → long (+1), class 1 → flat (0), class 0 → short (-1).
    """
    if len(preds) < 2 or len(close_val) < 2:
        return 0.0

    position = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    ret = np.diff(close_val) / close_val[:-1]

    n = min(len(position) - 1, len(ret))
    pnl = position[:n] * ret[:n] - (np.abs(position[:n]) > 0).astype(float) * commission

    avg = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    return (avg / std) * np.sqrt(252) if std > 0 else 0.0


def optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    close_val: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Two-stage HPO using separate train and validation sets.

    Stage 1: Optuna maximises F1-macro on validation set.
    Stage 2: Top-N candidates (by F1) are re-evaluated by Sharpe ratio
             on the validation set backtest.  Best Sharpe wins.

    If close_val is None, falls back to Stage-1-only (backward compatible).

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (separate from train)
        y_val: Validation labels
        config: Configuration dict
        close_val: Validation close prices for Stage 2 Sharpe evaluation

    Returns:
        best_params: Best hyperparameters found
        best_value: Best validation F1 score (Stage 1) or Sharpe (Stage 2)
    """
    n_trials = int(_get_nested(config, "hpo.trials", 50))
    study_name = str(_get_nested(config, "hpo.study_name", "xgb_hpo"))
    use_tscv = bool(_get_nested(config, "hpo.use_tscv", False))
    n_splits = int(_get_nested(config, "hpo.n_splits", 3))

    logger.info(f"Starting Optuna study '{study_name}' for {n_trials} trials")
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    sample_weight_train = _compute_sample_weights(y_train)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial)

        if use_tscv:
            # Use TimeSeriesSplit within training data for more robust evaluation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            f1_scores = []

            for tr_idx, te_idx in tscv.split(X_train):
                X_tr, X_te = X_train[tr_idx], X_train[te_idx]
                y_tr, y_te = y_train[tr_idx], y_train[te_idx]

                sw = _compute_sample_weights(y_tr)
                model = xgb.XGBClassifier(**params)

                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_te, y_te)],
                    verbose=False,
                    sample_weight=sw,
                )

                preds = model.predict(X_te)
                f1_scores.append(f1_score(y_te, preds, average="macro", zero_division=0))

            # Also evaluate on the held-out validation set
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False, sample_weight=sample_weight_train)
            val_preds = model.predict(X_val)
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

            # Weight: 50% TSCV score, 50% validation score
            return 0.5 * float(np.mean(f1_scores)) + 0.5 * val_f1

        else:
            # Simple train/val split evaluation
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight=sample_weight_train,
            )

            preds = model.predict(X_val)
            return f1_score(y_val, preds, average="macro", zero_division=0)

    # Create or load study
    storage = _get_nested(config, "hpo.storage", None)
    load_if_exists = bool(_get_nested(config, "hpo.load_if_exists", False))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists if storage else False,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential for reproducibility
    )

    logger.info(f"Stage 1 complete. Best F1 score: {study.best_value:.4f}")
    logger.info(f"Stage 1 best params: {study.best_params}")

    # Log importance of hyperparameters
    try:
        importances = optuna.importance.get_param_importances(study)
        logger.info("Hyperparameter importance:")
        for param, importance in sorted(importances.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {param}: {importance:.3f}")
    except Exception:
        pass

    # ── Stage 2: Re-rank top-N by Sharpe ratio ──────────────────────
    if close_val is not None and len(close_val) > 1:
        top_n = int(_get_nested(config, "hpo.stage2_top_n", 5))
        commission = float(_get_nested(config, "backtesting.commission", 0.0001))

        completed = [t for t in study.trials if t.value is not None]
        top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

        logger.info(f"Stage 2: Evaluating top {len(top_trials)} candidates by Sharpe ratio")

        best_sharpe = -np.inf
        best_params = study.best_params
        best_f1 = study.best_value

        for trial in top_trials:
            params = dict(trial.params)
            params.update({
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
            })

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False, sample_weight=sample_weight_train)
            preds = model.predict(X_val)
            sharpe = _backtest_sharpe(preds, close_val, commission)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)

            logger.info(
                f"  Trial {trial.number}: F1={f1:.4f}, Sharpe={sharpe:.2f}"
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = trial.params
                best_f1 = f1

        logger.info(f"Stage 2 winner: Sharpe={best_sharpe:.2f}, F1={best_f1:.4f}")
        logger.info(f"Selected params: {best_params}")
        return best_params, best_f1

    return study.best_params, float(study.best_value)
