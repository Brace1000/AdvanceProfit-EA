"""Optuna-based hyperparameter optimization scaffold with optional MLflow tracking."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.logger import get_logger

logger = get_logger("trading_bot.hpo")


def suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def optimize(X_train: np.ndarray, y_train: np.ndarray, config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Run HPO on the training segment with TimeSeriesSplit, maximizing macro F1.
    Returns best_params, best_value.
    """
    n_splits = int(config.get("hpo.n_splits", 5))
    n_trials = int(config.get("hpo.trials", 30))
    study_name = str(config.get("hpo.study_name", "xgb_hpo"))

    logger.info(f"Starting Optuna study '{study_name}' for {n_trials} trials with {n_splits} TSCV splits")

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        f1_scores = []
        for tr_idx, te_idx in tscv.split(X_train):
            X_tr, X_te = X_train[tr_idx], X_train[te_idx]
            y_tr, y_te = y_train[tr_idx], y_train[te_idx]
            model = xgb.XGBClassifier(**params)
            # class-balanced sample weights on the fold
            try:
                classes = np.array([0, 1, 2], dtype=int)
                class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
                sw = class_weights[y_tr]
            except Exception:
                sw = None
            # try early stopping if supported
            try:
                model.fit(X_tr, y_tr, verbose=False, eval_set=[(X_te, y_te)], early_stopping_rounds=10, sample_weight=sw)
            except TypeError:
                model.fit(X_tr, y_tr, verbose=False, sample_weight=sw)
            preds = model.predict(X_te)
            f1_scores.append(f1_score(y_te, preds, average="macro"))
        return float(np.mean(f1_scores))

    storage = config.get("hpo.storage")
    load_if_exists = bool(config.get("hpo.load_if_exists", True))
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(load_if_exists) if storage else False,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best score: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params, float(study.best_value)
