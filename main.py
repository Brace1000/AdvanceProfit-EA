from pathlib import Path
from typing import Optional

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


@app.on_event("startup")
async def load_model():
    global model, expected_n_features, feature_names

    cfg = get_config()
    model_path = Path(cfg.get("model.path", "xgb_eurusd_h1.pkl"))

    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Run run_all.py to train and save the model.")
        return

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    if hasattr(model, "n_features_in_"):
        expected_n_features = int(model.n_features_in_)
        logger.info(f"Expected number of features: {expected_n_features}")

    names_path = Path("feature_names.json")
    if names_path.exists():
        try:
            import json
            feature_names = json.loads(names_path.read_text())
        except Exception:
            feature_names = None


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
        "endpoints": {"/predict": "POST", "/health": "GET"},
    }


@app.get("/health")
def health():
    cfg = get_config()
    model_path = cfg.get("model.path", "xgb_eurusd_h1.pkl")
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_path": model_path,
        "model_exists": Path(model_path).exists(),
        "expected_n_features": expected_n_features,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model via run_all.py")

    features = request.features
    if expected_n_features is not None and len(features) != expected_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_n_features} features, got {len(features)}."
                   + (f" Names: {feature_names}" if feature_names else ""),
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
