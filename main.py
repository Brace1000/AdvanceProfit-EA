from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import os

app = FastAPI(title="EUR/USD Trading Model API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_eurusd_h1.pkl"

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"⚠️  Warning: Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first to create the model.")
    else:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")

# Define request model for better validation
class PredictionRequest(BaseModel):
    features: list[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.08, 0.002, 50.5, 0.1, 0.0005, 25.3, 0.0001, 0.0003, 0, 0, 0.0002]
            }
        }

@app.get("/")
def root():
    return {
        "message": "EUR/USD Trading Model API",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Make predictions with 11 features",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first by running train_model.py"
        )
    
    features = request.features
    
    if len(features) != 11:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 11 features, got {len(features)}. Required features: "
                   "close_ema50, ema50_ema200, rsi, rsi_slope, atr_ratio, adx, "
                   "body, range, hour, session, prev_return"
        )
    
    try:
        X = np.array(features).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        
        return {
            "sell": float(probs[0]),
            "range": float(probs[1]),
            "buy": float(probs[2]),
            "prediction": ["sell", "range", "buy"][np.argmax(probs)],
            "confidence": float(np.max(probs))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)