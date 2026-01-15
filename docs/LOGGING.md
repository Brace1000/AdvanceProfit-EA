# Logging Guide

This guide explains how to use logging in the AdvanceProfit-EA trading bot.

## Basic Usage

```python
from src.logger import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Log at different levels
logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("An error occurred")
logger.critical("A critical error - program may not be able to continue")
```

## Log Levels

### When to use each level:

**DEBUG** - Detailed information for diagnosing problems
```python
logger.debug(f"Feature values: {features}")
logger.debug(f"Model parameters: {params}")
```

**INFO** - Confirmation that things are working as expected
```python
logger.info("Starting model training")
logger.info("Trade opened: BUY EURUSD @ 1.0850")
logger.info("API server started on http://127.0.0.1:8000")
```

**WARNING** - Something unexpected happened, but program continues
```python
logger.warning("Low ML confidence: 0.35")
logger.warning("API rate limit approaching")
logger.warning("High volatility detected")
```

**ERROR** - A serious problem occurred, some function failed
```python
logger.error("Failed to fetch data from API")
logger.error("Model file not found")
logger.error("Trade execution failed")
```

**CRITICAL** - A very serious error, program may crash
```python
logger.critical("Database connection lost")
logger.critical("Out of memory")
logger.critical("Cannot connect to broker")
```

## Module-Specific Loggers

Use module names to organize logs:

```python
# In src/data_collection/fetch.py
from src.logger import get_logger
logger = get_logger("trading_bot.data.fetch")
logger.info("Fetching EURUSD data...")

# In src/models/train.py
from src.logger import get_logger
logger = get_logger("trading_bot.models.train")
logger.info("Training XGBoost model...")

# In src/api/main.py
from src.logger import get_logger
logger = get_logger("trading_bot.api")
logger.info("API endpoint called: /predict")
```

## Logging Exceptions

Use `logger.exception()` to log full stack traces:

```python
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Operation failed")
    # This automatically includes the full stack trace
```

## Configuration

Logging is configured in `config.yaml`:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/trading_bot.log"
  console: true
```

## Best Practices

### ✅ DO:

```python
# Use appropriate log levels
logger.info("Trade executed successfully")
logger.error("Failed to connect to API")

# Include context in log messages
logger.info(f"Training with {len(X_train)} samples")
logger.warning(f"Low confidence: {confidence:.2%}")

# Use __name__ for module loggers
logger = get_logger(__name__)

# Log exceptions with stack traces
logger.exception("Unexpected error during training")

# Use structured logging for important events
logger.info(
    "Trade opened",
    extra={
        "symbol": "EURUSD",
        "direction": "BUY",
        "price": 1.0850,
        "volume": 0.1
    }
)
```

### ❌ DON'T:

```python
# Don't use print statements
print("Starting bot")  # ❌

# Don't log sensitive information
logger.info(f"API key: {api_key}")  # ❌

# Don't log at wrong level
logger.error("User clicked button")  # ❌ Use info
logger.info("Database connection failed")  # ❌ Use error

# Don't log inside tight loops
for i in range(1000000):
    logger.debug(f"Processing {i}")  # ❌ Too verbose

# Don't concatenate strings in log calls
logger.info("Value: " + str(value))  # ❌
logger.info(f"Value: {value}")  # ✅
```

## Examples

### Training Script
```python
from src.logger import get_logger

logger = get_logger("trading_bot.train")

def train_model(X, y):
    logger.info("Starting model training")
    logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")

    try:
        model = XGBClassifier(**params)
        model.fit(X, y)

        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        logger.info(f"Training accuracy: {train_acc:.2%}")
        logger.info(f"Validation accuracy: {val_acc:.2%}")

        if val_acc < 0.35:
            logger.warning("Low validation accuracy - consider more data or different features")

        logger.info("Model training complete")
        return model

    except Exception as e:
        logger.exception("Model training failed")
        raise
```

### API Server
```python
from src.logger import get_logger
from fastapi import FastAPI

logger = get_logger("trading_bot.api")
app = FastAPI()

@app.on_event("startup")
async def startup():
    logger.info("API server starting")
    logger.info("Loading model...")
    # Load model
    logger.info("Model loaded successfully")

@app.post("/predict")
async def predict(features):
    logger.debug(f"Prediction request with {len(features)} features")

    try:
        prediction = model.predict(features)
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
        return {"prediction": prediction, "confidence": confidence}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
```

### Data Collection
```python
from src.logger import get_logger

logger = get_logger("trading_bot.data")

def fetch_data(symbol, timeframe, start_date, end_date):
    logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")

    try:
        df = download_data(symbol, timeframe, start_date, end_date)
        logger.info(f"Downloaded {len(df)} bars")

        if len(df) < 100:
            logger.warning(f"Very few bars downloaded: {len(df)}")

        logger.debug(f"Data shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logger.exception(f"Failed to fetch {symbol} data")
        return None
```

## Viewing Logs

### Console
Logs are printed to console with colors (if terminal supports it)

### Log Files
Logs are saved to `logs/trading_bot.log` with rotation:
- Maximum file size: 10MB
- Keeps 5 backup files
- Files are named: `trading_bot.log`, `trading_bot.log.1`, etc.

### Filtering Logs
```bash
# View all logs
cat logs/trading_bot.log

# View only errors
grep ERROR logs/trading_bot.log

# View logs from specific module
grep "trading_bot.api" logs/trading_bot.log

# Tail logs in real-time
tail -f logs/trading_bot.log
```

## Integration with MLflow

Logs can be automatically tracked with MLflow experiments:

```python
import mlflow
from src.logger import get_logger

logger = get_logger("trading_bot.experiment")

with mlflow.start_run():
    logger.info("Starting experiment")

    # Train model
    model = train_model(X, y)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)

    logger.info("Experiment complete")
```

## Troubleshooting

**Q: Logs not appearing in console**
A: Check log level in config.yaml - set to DEBUG to see everything

**Q: Log file not created**
A: Ensure `logs/` directory exists and is writable

**Q: Too much logging**
A: Increase log level to WARNING or ERROR in production

**Q: Can't see colors**
A: Some terminals don't support colors - logs will still work, just without colors
