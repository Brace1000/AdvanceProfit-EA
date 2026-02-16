import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import get_config
from src.logger import get_logger

app = FastAPI(title="EUR/USD Trading Model API")
logger = get_logger("trading_bot.api")

model = None
expected_n_features: Optional[int] = None
feature_names: Optional[list[str]] = None
feature_contract: Optional[Dict[str, Any]] = None

CONTRACT_PATH = Path("feature_contract.json")


def _load_contract() -> Optional[Dict[str, Any]]:
    """Load feature contract from JSON file."""
    if not CONTRACT_PATH.exists():
        logger.warning(f"Feature contract not found at {CONTRACT_PATH}")
        return None
    try:
        return json.loads(CONTRACT_PATH.read_text())
    except Exception as e:
        logger.error(f"Failed to parse feature contract: {e}")
        return None


@app.on_event("startup")
async def load_model():
    global model, expected_n_features, feature_names, feature_contract

    # Load feature contract (single source of truth)
    feature_contract = _load_contract()
    if feature_contract is not None:
        expected_n_features = feature_contract["feature_count"]
        feature_names = feature_contract["features"]
        logger.info(
            f"Feature contract v{feature_contract['version']} loaded: "
            f"{expected_n_features} features"
        )

    cfg = get_config()
    model_path = Path(cfg.get("model.path", "xgb_eurusd_h1.pkl"))

    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Run run_all.py to train and save the model.")
        return

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Cross-check: model feature count must match contract
    if hasattr(model, "n_features_in_") and feature_contract is not None:
        model_n = int(model.n_features_in_)
        contract_n = feature_contract["feature_count"]
        if model_n != contract_n:
            logger.error(
                f"MISMATCH: model expects {model_n} features but contract specifies {contract_n}"
            )
    elif hasattr(model, "n_features_in_") and feature_contract is None:
        # Fallback: use model's own feature count if no contract
        expected_n_features = int(model.n_features_in_)
        logger.info(f"No contract; using model feature count: {expected_n_features}")


class PredictionRequest(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 11
            }
        }


@app.get("/")
def root():
    return {
        "message": "EUR/USD Trading Model API",
        "model_loaded": model is not None,
        "expected_n_features": expected_n_features,
        "feature_names": feature_names,
        "endpoints": {"/predict": "POST", "/health": "GET", "/contract": "GET"},
    }


@app.get("/health")
def health():
    cfg = get_config()
    model_path = cfg.get("model.path", "xgb_eurusd_h1.pkl")
    staleness_threshold = cfg.get("trading.ml.model_staleness_hours", 168)
    contract_version = feature_contract.get("version", "unknown") if feature_contract else "unknown"

    model_loaded = model is not None
    model_age_hours = 0.0

    if Path(model_path).exists():
        model_age_hours = (time.time() - os.path.getmtime(model_path)) / 3600

    if not model_loaded:
        status = "model_not_loaded"
    elif model_age_hours > staleness_threshold:
        status = "stale"
    else:
        status = "healthy"

    return {
        "status": status,
        "model_loaded": model_loaded,
        "feature_count": expected_n_features if expected_n_features is not None else 0,
        "model_age_hours": round(model_age_hours, 2),
        "model_path": model_path,
        "contract_version": contract_version,
    }


@app.get("/contract")
def contract():
    """Return the feature contract — single source of truth for feature vector."""
    if feature_contract is None:
        raise HTTPException(status_code=503, detail="Feature contract not loaded")
    return {
        "version": feature_contract["version"],
        "feature_count": feature_contract["feature_count"],
        "feature_names": feature_contract["features"],
        "label_mapping": feature_contract.get("label_mapping"),
        "prediction_mapping": feature_contract.get("prediction_mapping"),
        "confidence_threshold": feature_contract.get("confidence_threshold"),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model via run_all.py")

    features = request.features

    # Reject NaN / Inf values — defense-in-depth (EA also validates, but
    # the API must not silently pass invalid numerics to the model).
    invalid = [i for i, v in enumerate(features) if not math.isfinite(v)]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Non-finite feature values at indices {invalid}",
                "hint": "All features must be finite (no NaN, Inf, or -Inf)",
            },
        )

    if expected_n_features is not None and len(features) != expected_n_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Expected {expected_n_features} features, got {len(features)}",
                "expected_features": feature_names,
            },
        )

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        return {
            "sell": float(probs[0]),
            "range": float(probs[1]),
            "buy": float(probs[2]),
            "prediction": ["sell", "range", "buy"][int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run(app, host=cfg.get("api.host", "0.0.0.0"), port=int(cfg.get("api.port", 8000)))
