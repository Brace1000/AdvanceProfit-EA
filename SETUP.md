# AdvanceProfit-EA Setup Guide

Complete guide to setting up your ML-powered trading bot development environment.

## Prerequisites

- **Windows 10/11** (you're using Windows with WSL)
- **Python 3.10+** installed on Windows
- **Poetry** (package manager) - Already installed ✓
- **MetaTrader 5** (for forex trading)
- **Git** (version control)

## Quick Start (Windows)

### 1. Navigate to Project

Open PowerShell:

```powershell
cd C:\Users\MASTERTECH\Desktop\AdvanceProfit-EA
```

### 2. Install Dependencies

```powershell
# Install all packages with Poetry
poetry install

# This will:
# - Create a virtual environment automatically
# - Install all dependencies from pyproject.toml
# - Set up dev tools (pytest, black, etc.)
```

### 3. Verify Installation

```powershell
# Run the verification script
poetry run python verify_setup.py
```

This will check:
- ✓ Python version
- ✓ All packages installed
- ✓ Directory structure
- ✓ Configuration files
- ✓ Data and model files

### 4. Configure Settings (Optional)

```powershell
# Copy default config (if you want custom settings)
copy config.yaml config.local.yaml

# Edit config.local.yaml with your settings
# (config.local.yaml is gitignored and won't be committed)
```

### 5. Generate Data and Train Model

```powershell
# Run the complete training pipeline
poetry run python run_all.py
```

This will:
- Load EURUSD data
- Engineer features
- Train XGBoost model
- Save model to `xgb_eurusd_h1.pkl`

### 6. Run Tests

```powershell
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=html

# View coverage report
start htmlcov/index.html
```

### 7. Start API Server

```powershell
# Start FastAPI server for predictions
poetry run uvicorn main:app --reload
```

API will be available at:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs (interactive documentation)

## Project Structure

```
AdvanceProfit-EA/
├── config.yaml                 # Configuration (tracked in git)
├── config.local.yaml          # Your custom config (gitignored)
├── pyproject.toml             # Poetry dependencies
├── run_all.py                 # Training pipeline
├── main.py                    # FastAPI server
├── data_fetch.py              # Data collection from MT5
├── AdvanceEA.py               # MQL5 Expert Advisor (rename to .mq5)
├── verify_setup.py            # Setup verification
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration loader
│   ├── logger.py             # Logging utilities
│   ├── data_collection/      # Data fetching modules
│   ├── features/             # Feature engineering
│   ├── models/               # Model training/inference
│   ├── backtesting/          # Strategy backtesting
│   └── api/                  # FastAPI endpoints
│
├── tests/                     # Test suite
│   ├── conftest.py           # Shared fixtures
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
│
├── data/                      # Data storage
│   ├── raw/                  # Raw OHLCV data
│   └── processed/            # Processed features
│
├── models/                    # Saved ML models
├── logs/                      # Application logs
├── notebooks/                 # Jupyter notebooks
├── experiments/               # MLflow experiments
└── docs/                      # Documentation
    └── LOGGING.md            # Logging guide
```

## Using Poetry

### Common Commands

```powershell
# Install all dependencies
poetry install

# Install with dev dependencies
poetry install --with dev

# Add a new package
poetry add package-name

# Add a dev package
poetry add --group dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show

# Run a command in virtual environment
poetry run python script.py

# Activate virtual environment
poetry shell

# Exit virtual environment
exit
```

## Using Configuration

### Python Code

```python
from src.config import get_config

# Get config instance
config = get_config()

# Access values with dot notation
model_path = config.get("model.path")
risk_percent = config.get("trading.risk.risk_percent")

# Get entire section
trading_config = config.trading
api_config = config.api

# Get with default value
max_loss = config.get("trading.risk.max_daily_loss", 3.0)
```

### Configuration Priority

1. `config.local.yaml` (if exists - your custom settings)
2. `config.yaml` (default settings)

**Best Practice**: Keep `config.yaml` for defaults, use `config.local.yaml` for your custom settings.

## Using Logging

### Python Code

```python
from src.logger import get_logger

# Get logger for your module
logger = get_logger(__name__)

# Log at different levels
logger.debug("Detailed debugging info")
logger.info("Trade opened: BUY EURUSD @ 1.0850")
logger.warning("Low ML confidence: 0.35")
logger.error("Failed to fetch data")
logger.critical("Database connection lost")

# Log exceptions with full stack trace
try:
    risky_operation()
except Exception:
    logger.exception("Operation failed")
```

See [docs/LOGGING.md](docs/LOGGING.md) for complete logging guide.

## Development Workflow

### Daily Development

```powershell
# 1. Activate environment
poetry shell

# 2. Make changes to code

# 3. Run tests
pytest

# 4. Format code
black src/ tests/
isort src/ tests/

# 5. Check code quality
flake8 src/

# 6. Run your code
python run_all.py
```

### Adding New Features

```powershell
# 1. Create a new branch (if using git)
git checkout -b feature/my-new-feature

# 2. Write code in src/

# 3. Write tests in tests/

# 4. Run tests
pytest tests/

# 5. Commit changes
git add .
git commit -m "Add my new feature"
```

## Next Steps

### 1. Hyperparameter Optimization (HPO)
Currently your model uses hardcoded hyperparameters. Next priority:
- Create `src/models/hpo.py` with Optuna
- Optimize: n_estimators, max_depth, learning_rate, etc.
- Track experiments with MLflow

### 2. Backtesting Framework
Test your strategy on historical data:
- Create `src/backtesting/strategy.py`
- Use backtrader or custom backtester
- Calculate: win rate, Sharpe ratio, max drawdown

### 3. Multi-Asset Support
Expand beyond EUR/USD:
- Train models for multiple currency pairs
- Support stocks (via yfinance)
- Support crypto (via ccxt/Binance)

### 4. Production Deployment
When ready for live trading:
- Set `trading.mode` to "live" in config
- Use proper risk management
- Set up monitoring and alerts
- Start with paper trading first!

## Troubleshooting

### Poetry not found
```powershell
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### Python not found
- Install Python 3.10+ from python.org
- Make sure to check "Add Python to PATH" during installation

### Import errors
```powershell
# Make sure you're in the poetry environment
poetry shell

# Or prefix commands with poetry run
poetry run python script.py
```

### Tests failing
```powershell
# Install dev dependencies
poetry install --with dev

# Run specific test
poetry run pytest tests/unit/test_features.py -v
```

### Can't access MT5 data
- Make sure MetaTrader 5 is installed and logged in
- Run MT5 at least once before using data_fetch.py
- Check MT5 allows Python API access

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Getting Help

1. Check `docs/` folder for guides
2. Run `verify_setup.py` to diagnose issues
3. Check logs in `logs/trading_bot.log`
4. Review test output: `pytest -v`

## Summary

You now have a complete ML trading bot setup with:
- ✓ Poetry dependency management
- ✓ YAML-based configuration
- ✓ Professional logging system
- ✓ Comprehensive test suite
- ✓ XGBoost ML model
- ✓ FastAPI prediction server
- ✓ MT5 integration
- ✓ Organized project structure

**Status**: Ready for development! 🚀

**Next Priority**: Implement HPO with Optuna to optimize model performance.
