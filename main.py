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

# Sell model (original)
sell_model = None
sell_n_features: Optional[int] = None

# Buy model (new)
buy_model = None
buy_n_features: Optional[int] = None


@app.on_event("startup")
async def load_models():
    global sell_model, sell_n_features, buy_model, buy_n_features

    cfg = get_config()

    # Load Sell model
    sell_path = Path(cfg.get("model.path", "models/xgb_eurusd_h1.pkl"))
    if sell_path.exists():
        sell_model = joblib.load(sell_path)
        logger.info(f"Sell model loaded from {sell_path}")
        if hasattr(sell_model, "n_features_in_"):
            sell_n_features = int(sell_model.n_features_in_)
            logger.info(f"Sell model expects {sell_n_features} features")
    else:
        logger.warning(f"Sell model not found at {sell_path}")

    # Load Buy model
    buy_path = Path(cfg.get("model.buy_path", "models/xgb_eurusd_h1_buy.pkl"))
    if buy_path.exists():
        buy_model = joblib.load(buy_path)
        logger.info(f"Buy model loaded from {buy_path}")
        if hasattr(buy_model, "n_features_in_"):
            buy_n_features = int(buy_model.n_features_in_)
            logger.info(f"Buy model expects {buy_n_features} features")
    else:
        logger.warning(f"Buy model not found at {buy_path}")


class PredictionRequest(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 14
            }
        }


@app.get("/")
def root():
    return {
        "message": "EUR/USD Trading Model API",
        "sell_model_loaded": sell_model is not None,
        "buy_model_loaded": buy_model is not None,
        "endpoints": {
            "/predict": "POST - Sell model predictions",
            "/predict/buy": "POST - Buy model predictions",
            "/health": "GET"
        },
    }


@app.get("/health")
def health():
    cfg = get_config()
    return {
        "status": "healthy" if (sell_model is not None or buy_model is not None) else "no_models_loaded",
        "sell_model": {
            "loaded": sell_model is not None,
            "path": cfg.get("model.path", "models/xgb_eurusd_h1.pkl"),
            "n_features": sell_n_features
        },
        "buy_model": {
            "loaded": buy_model is not None,
            "path": cfg.get("model.buy_path", "models/xgb_eurusd_h1_buy.pkl"),
            "n_features": buy_n_features
        }
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Sell model prediction (original endpoint)."""
    if sell_model is None:
        raise HTTPException(status_code=503, detail="Sell model not loaded")

    features = request.features
    if sell_n_features is not None and len(features) != sell_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {sell_n_features} features, got {len(features)}."
        )

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        probs = sell_model.predict_proba(X)[0]
        return {
            "sell": float(probs[0]),
            "range": float(probs[1]),
            "buy": float(probs[2]),
            "prediction": ["sell", "range", "buy"][int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
        }
    except Exception as e:
        logger.exception("Sell prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/buy")
def predict_buy(request: PredictionRequest):
    """Buy model prediction."""
    if buy_model is None:
        raise HTTPException(status_code=503, detail="Buy model not loaded")

    features = request.features
    if buy_n_features is not None and len(features) != buy_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {buy_n_features} features, got {len(features)}."
        )

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        probs = buy_model.predict_proba(X)[0]
        # Buy model: label 0 = buy wins, label 1 = range, label 2 = buy loses
        return {
            "buy": float(probs[0]),      # Probability buy wins
            "range": float(probs[1]),
            "sell": float(probs[2]),     # Probability buy loses (sell wins)
            "prediction": ["buy", "range", "sell"][int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
        }
    except Exception as e:
        logger.exception("Buy prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run(app, host=cfg.get("api.host", "0.0.0.0"), port=int(cfg.get("api.port", 8000)))
