# AdvanceProfit-EA

# EA + ML Integration Setup Guide

## Overview
This guide explains how to integrate your MQL5 EA with the Python ML prediction API.

---

## Step 1: Enable WebRequest in MT5

**CRITICAL**: You must allow MT5 to make HTTP requests to your Python API.

### In MetaTrader 5:
1. Go to **Tools** → **Options**
2. Click the **Expert Advisors** tab
3. Check **"Allow WebRequest for listed URL:"**
4. Add these URLs (one per line):
   ```
   http://127.0.0.1:8000
   http://localhost:8000
   ```
5. Click **OK**

**Without this step, the EA cannot communicate with the API!**

---

## Step 2: Start the Python API

Before running the EA, make sure your API is running:

```bash
# Terminal 1: Start the API
cd ~/Desktop/AdvanceProfit-EA
uvicorn main:app --reloaduvicorn main:app --reloaduvicorn main:app --reload
```

Verify it's running by opening: `http://127.0.0.1:8000/health`

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Step 3: Install the Updated EA

1. Copy the **AdvancedProfitEA_ML.mq5** file to:
   ```
   C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\[Your MT5 ID]\MQL5\Experts\
   ```
   Or use MT5's **File** → **Open Data Folder** → **MQL5** → **Experts**

2. In MT5 MetaEditor:
   - Open the file
   - Click **Compile** (F7)
   - Check for 0 errors

3. Drag the EA onto your chart

---

## Step 4: Configure EA Settings

### ML API Settings (NEW!)
- **UseMLPredictions**: `true` (enable ML)
- **API_URL**: `http://127.0.0.1:8000/predict`
- **ML_Confidence_Threshold**: `0.40` (40% minimum confidence)
- **CombineWithTechnical**: `true` (both ML and technical must agree)

### Keep your existing settings:
- Risk Management
- Strategy Settings
- Trade Filters
- Position Management

---

## Step 5: Test the Integration

### Watch the Logs
In MT5 Experts tab, you should see:
```
Advanced Profit EA v2.10 with ML
ML Integration: ENABLED
API URL: http://127.0.0.1:8000/predict
```

When a signal occurs:
```
ML Prediction: buy (Confidence: 65.3%)
  Sell: 15.2% | Range: 19.5% | Buy: 65.3%
✓ ML and Technical signals agree: BUY
BUY order opened: Lot=0.10 SL=1.0800 TP=1.0850
```

---

## How It Works

### 1. Feature Calculation
The EA calculates 11 features on **DAILY timeframe**:
- `close_ema50`: Current EMA(50)
- `ema50_ema200`: Difference between EMAs
- `rsi`: RSI value
- `rsi_slope`: RSI momentum
- `atr_ratio`: ATR relative to price
- `adx`: ADX value
- `body`: Candle body size
- `range`: Candle range
- `hour`: 0 (daily data)
- `session`: 0 (daily data)
- `prev_return`: Previous day return

### 2. API Call
The EA sends these features as JSON to your Python API:
```json
{
  "features": [1.08, 0.002, 50.5, 0.1, 0.0005, 25.3, 0.0001, 0.0003, 0, 0, 0.0002]
}
```

### 3. ML Prediction
The API returns probabilities:
```json
{
  "sell": 0.20,
  "range": 0.15,
  "buy": 0.65,
  "prediction": "buy",
  "confidence": 0.65
}
```

### 4. Decision Logic
**If `CombineWithTechnical = true`:**
- ✓ Trade only if ML **AND** technical signals agree
- ✗ Skip if they disagree

**If `CombineWithTechnical = false`:**
- Trade based on ML prediction only (if confidence > threshold)

---

## Trading Modes

### Mode 1: ML + Technical Confirmation (Recommended)
```cpp
UseMLPredictions = true
CombineWithTechnical = true
ML_Confidence_Threshold = 0.40
```
**Most conservative** - Both systems must agree

### Mode 2: ML Only
```cpp
UseMLPredictions = true
CombineWithTechnical = false
ML_Confidence_Threshold = 0.50
```
**Moderate** - Higher threshold, ML-driven

### Mode 3: Technical Only (Original)
```cpp
UseMLPredictions = false
```
**Original EA** - No ML integration

---

## Troubleshooting

### Error: "WebRequest error: 4014"
**Solution**: Add the API URL to allowed URLs in MT5 settings (see Step 1)

### Error: "WebRequest error: 4060"
**Solution**: 
- Check if Python API is running
- Try `http://localhost:8000/predict` instead of `127.0.0.1`

### "ML confidence too low"
**Solution**: 
- Lower `ML_Confidence_Threshold` (e.g., 0.35)
- Or wait for stronger signals

### No ML predictions appearing
**Solution**:
1. Check Python API logs: `uvicorn main:app --reload`
2. Test API manually: `python3 test_api.py`
3. Verify EA settings: `UseMLPredictions = true`

### "Failed to copy indicator data for ML"
**Solution**: 
- EA needs at least 200 bars of daily data
- Wait for more historical data to load

---

## Performance Monitoring

### In MT5 Experts Tab
Monitor these messages:
- ML prediction results
- Signal agreement/disagreement
- Trade execution

### In Python Terminal
Monitor API requests:
```
INFO: 127.0.0.1:56789 - "POST /predict HTTP/1.1" 200 OK
```

### Key Metrics to Track
- Win rate with ML vs without
- Average confidence of winning trades
- Signal agreement rate (ML vs Technical)

---

## Best Practices

1. **Start with Demo Account**
   - Test the integration thoroughly
   - Monitor for at least 1-2 weeks

2. **Use Conservative Settings Initially**
   ```cpp
   ML_Confidence_Threshold = 0.50  // Higher threshold
   CombineWithTechnical = true     // Require agreement
   ```

3. **Keep API Running**
   - Use `screen` or `tmux` on Linux
   - Or run as a service

4. **Monitor Daily**
   - Check EA logs
   - Check API logs
   - Track signal quality

5. **Retrain Model Periodically**
   - Add new data to CSV
   - Run `python3 run_all.py`
   - Restart API with new model

---

## Example Workflow

### Daily Routine:
```bash
# Morning: Start API
cd ~/Desktop/AdvanceProfit-EA
uvicorn main:app --reload &

# In MT5: Start EA on chart

# Evening: Check performance
python3 test_api.py  # Test API
# Review MT5 Experts log
```

### Weekly: Update Model
```bash
# Add new data to EURUSD_D1_raw.csv
# Retrain
python3 run_all.py

# Restart API
kill -9 $(lsof -ti:8000)
uvicorn main:app --reload &
```

---

## Summary of Changes

### What's New:
✅ ML prediction integration via HTTP API  
✅ Confidence threshold filtering  
✅ Combined ML + Technical signal validation  
✅ Daily timeframe feature calculation  
✅ Detailed logging of predictions  

### What's Unchanged:
✅ All original risk management  
✅ Position management (trailing stops, breakeven)  
✅ Trade filters (time, volatility, trend)  
✅ Lot sizing and money management  

---

## Need Help?

### Common Issues:
1. **API not responding**: Check `uvicorn` is running
2. **WebRequest errors**: Verify allowed URLs in MT5
3. **Low accuracy**: Retrain model with more data
4. **Too few signals**: Lower confidence threshold

### Testing:
```bash
# Test API manually
python3 test_api.py

# Check API health
curl http://127.0.0.1:8000/health
```

Good luck with your ML-enhanced trading! 🚀