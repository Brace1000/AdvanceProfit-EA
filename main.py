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
expected_n_features = None
feature_names = None

@app.on_event("startup")
async def load_model():
    global model, expected_n_features, feature_names
    if not MODEL_PATH.exists():
        print(f"⚠️  Warning: Model file not found at {MODEL_PATH}")
        print("Please run run_all.py first to create the model.")
    else:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        # Infer expected feature length
        if hasattr(model, "n_features_in_"):
            expected_n_features = int(model.n_features_in_)
            print(f"Expected number of features: {expected_n_features}")
        # Try to load feature_names_ from a companion file written by the trainer in the future
        names_path = BASE_DIR / "feature_names.json"
        if names_path.exists():
            try:
                import json
                feature_names = json.loads(names_path.read_text())
            except Exception:
                feature_names = None

# Define request model for better validation
class PredictionRequest(BaseModel):
    features: list[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    # Example for 20-feature H1+H4 model (order printed by run_all.py)
                    1.080123, 0.001234, 52.3, 0.2401, 0.00012345, 18.2, 0.00010, 0.00030, 14, 1, 0.00020,
                    1.079876, 0.001100, 48.7, -0.2102, 0.00015000, 16.5, 0.00008, 0.00025, 0.00010
                ]
            }
        }

@app.get("/")
def root():
    return {
        "message": "EUR/USD Trading Model API",
        "model_loaded": model is not None,
        "expected_n_features": expected_n_features,
        "feature_names": feature_names,
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "expected_n_features": expected_n_features,
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first by running run_all.py"
        )
    
    features = request.features

    # Validate feature length dynamically against the model
    if expected_n_features is not None and len(features) != expected_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_n_features} features, got {len(features)}."
                    + (f" Names: {feature_names}" if feature_names else "")
        )
    
    try:
        X = np.array(features).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        return {
            "sell": float(probs[0]),
            "range": float(probs[1]),
            "buy": float(probs[2]),
            "prediction": ["sell", "range", "buy"][int(np.argmax(probs))],
            "confidence": float(np.max(probs))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
