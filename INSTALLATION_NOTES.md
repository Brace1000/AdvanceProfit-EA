# Installation Notes - Linux Setup

## Issue Fixed
The project had a dependency issue with `MetaTrader5` package, which is Windows-only and cannot be installed on Linux systems.

## Solution Applied
1. **Modified pyproject.toml**: Commented out the `MetaTrader5` dependency since it's not used in the actual source code and is Windows-specific.
   
2. **Installed all dependencies**: Used Poetry to install all remaining dependencies successfully.

3. **Verified functionality**: 
   - ✅ Training pipeline works (`poetry run python run_all.py`)
   - ✅ API server works (`poetry run uvicorn main:app --host 0.0.0.0 --port 8000`)

## How to Use

### 1. Train the Model
```bash
poetry run python run_all.py
```

This will:
- Load and clean the raw EURUSD data
- Engineer features
- Run hyperparameter optimization with Optuna
- Train the XGBoost model
- Evaluate on holdout set
- Save the model to `xgb_eurusd_h1.pkl`

### 2. Start the API Server
```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:
```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get API Info:**
```bash
curl http://localhost:8000/
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.05, 50, 0.2, 1.5, 25, 0.3, 0.4, 10, 1, 0.01, 0.08, 0.03, 48, 0.15, 1.6, 22, 0.28, 0.38, 0.008]}'
```

## Alternative: Use Python Directly
If you prefer not to use Poetry's environment prefix:

```bash
# Activate the virtual environment
poetry shell

# Then run commands directly
python run_all.py
uvicorn main:app --reload
```

## Model Performance (from latest training)
- **Train Accuracy**: 76.63%
- **Holdout Accuracy**: 57.04%
- **Win Rate**: 64.53%
- **Sharpe Ratio**: 2.92
- **Max Drawdown**: -2.44%

## Note for Windows Users
If you need to use MetaTrader5 on Windows, uncomment the line in [pyproject.toml](./pyproject.toml):
```toml
MetaTrader5 = "^5.0.4563"
```

Then run `poetry install` again.
