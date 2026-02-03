"""Optuna-based hyperparameter optimization with proper validation strategy."""
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
    Suggest hyperparameters for XGBoost (3-class: Sell/Range/Buy).

    Wide search ranges allow HPO to find the right balance between
    regularization and expressiveness.
    """
    return {
        # Tree structure
        "n_estimators": trial.suggest_int("n_estimators", 30, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 80),

        # Learning rate
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # Subsampling
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),

        # Regularization — wide range so HPO can find the sweet spot
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),

        # Fixed parameters — 3-class: Sell(0) / Range(1) / Buy(2)
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def _compute_sample_weights(y: np.ndarray) -> Optional[np.ndarray]:
    """Compute class-balanced sample weights (3-class)."""
    try:
        classes = np.array([0, 1, 2])
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weight_map = dict(zip(classes, class_weights))
        return np.array([weight_map[label] for label in y])
    except Exception:
        return None


def optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """
    Run HPO using separate train and validation sets.

    This is the proper approach that prevents data leakage:
    - Train on X_train, y_train
    - Evaluate on X_val, y_val
    - Validation set is used ONLY for scoring, not for training

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (separate from train)
        y_val: Validation labels
        config: Configuration dict

    Returns:
        best_params: Best hyperparameters found
        best_value: Best validation F1 score
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

    logger.info(f"HPO complete. Best F1 score: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Log importance of hyperparameters
    try:
        importances = optuna.importance.get_param_importances(study)
        logger.info("Hyperparameter importance:")
        for param, importance in sorted(importances.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {param}: {importance:.3f}")
    except Exception:
        pass

    return study.best_params, float(study.best_value)
