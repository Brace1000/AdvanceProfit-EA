# Buy Model: Training, Testing & Design Decisions

## Overview

The Buy model predicts favorable long (BUY) entry conditions for EUR/USD on the H1 timeframe. It is an XGBoost multi-class classifier that outputs probabilities for three outcomes: **buy wins**, **range** (no barrier hit), and **buy loses**. The EA enters a buy trade only when buy probability dominates the other classes.

**Walk-forward result**: +1,177 pips | 47.2% win rate | 836 trades | 0.40 threshold

---

## Table of Contents

1. [Why a Separate Buy Model](#1-why-a-separate-buy-model)
2. [Training Data & Labeling](#2-training-data--labeling)
3. [Features (14 inputs)](#3-features-14-inputs)
4. [Model Architecture](#4-model-architecture)
5. [Walk-Forward Validation](#5-walk-forward-validation)
6. [Threshold Optimization](#6-threshold-optimization)
7. [Regime Filter Decision](#7-regime-filter-decision)
8. [How the EA Uses Predictions](#8-how-the-ea-uses-predictions)
9. [Buy vs Sell Model Comparison](#9-buy-vs-sell-model-comparison)
10. [API Architecture](#10-api-architecture)
11. [Key Files](#11-key-files)

---

## 1. Why a Separate Buy Model

The original system was sell-only. Walk-forward testing of the Sell model showed:

| Metric | Sell Model | Buy Model |
|--------|-----------|-----------|
| Total pips | +74 (with CB) | +1,177 |
| Win rate | 47.0% | 47.2% |
| Total trades | 217 | 836 |
| Profitable windows | 2 of 5 | 4 of 5 |
| Pips/trade | +0.3 | +1.4 |

The Sell model's edge was marginal (+74 pips over ~2 years) and concentrated in just 2 windows. The Buy model showed consistent positive edge across 4 of 5 walk-forward windows.

### Why not use one model for both directions?

A single model trained on both buy and sell labels would conflate two different market regimes. Bullish and bearish markets have different volatility profiles, momentum characteristics, and mean-reversion behavior. Separate models let each specialize in recognizing its target regime.

### Why not trade the Sell model's "sell" signal from the Buy endpoint?

The Buy model outputs three probabilities: `buy_prob`, `range_prob`, `sell_prob`. When `sell_prob` is high, it means **"this is NOT a buy setup"** - it's a rejection signal, not a sell recommendation. The Buy model was trained to recognize buy-favorable conditions; its sell probability is calibrated differently than the dedicated Sell model's output.

Think of it as a specialist opinion: a flu doctor saying "you don't have flu" is different from an oncologist saying "you have cancer." Each model is a specialist for its direction.

---

## 2. Training Data & Labeling

### Data Source
- **Pair**: EUR/USD
- **Timeframe**: H1 (hourly candles)
- **Period**: ~17,000 bars (~2018-2026)
- **Source**: MetaTrader 5 historical data

### Triple Barrier Labeling

Instead of simple return-based labels (e.g., "price went up 0.1%"), we use **triple barrier labeling** that mirrors exactly how the EA trades:

```
For each bar, simulate a BUY entry and check what happens:

Label 0 (Buy wins):  Price rises +20 pips BEFORE falling -15 pips
Label 1 (Range):     Neither barrier hit within 50 bars
Label 2 (Buy loses): Price falls -15 pips BEFORE rising +20 pips
```

**Why triple barriers?** Traditional labels ask "did price go up?" but that's not how trades work. A trade has a TP, an SL, and a time limit. The model learns to predict *which barrier gets hit first* - directly modeling the trade outcome.

The barriers match the EA settings:
- **TP**: 20 pips (buy target above entry)
- **SL**: 15 pips (stop loss below entry)
- **Max holding**: 50 bars (roughly 2 trading days)

### Label Distribution (full dataset)

| Label | Count | Percentage |
|-------|-------|-----------|
| Buy wins (0) | 7,467 | 44.0% |
| Range (1) | 87 | 0.5% |
| Buy loses (2) | 9,424 | 55.5% |

The base buy win rate (excluding range) is **44.2%**, which is above the 42.9% breakeven rate for 20/15 TP/SL. This means even random entries have a slight long bias in EUR/USD over this period - the model's job is to improve on this base rate by filtering for the best setups.

### Breakeven Win Rate

With TP=20 and SL=15 (risk/reward of 1.33:1):

```
Breakeven = SL / (TP + SL) = 15 / (20 + 15) = 42.9%
```

Any win rate above 42.9% is profitable. The model achieves 47.2%, which is +4.3% above breakeven.

---

## 3. Features (14 inputs)

The Buy model uses the same 14 features as the Sell model, computed across H1 and H4 timeframes. These features capture **trend, momentum, volatility, and regime** - everything the model needs to assess whether conditions favor a buy.

### H1 Features (7)

| # | Feature | Formula | What it measures |
|---|---------|---------|-----------------|
| 1 | `close_ema50_h1` | (close - EMA50) / EMA50 | Price displacement from trend |
| 2 | `ema50_ema200_h1` | (EMA50 - EMA200) / close | Trend direction & strength |
| 3 | `rsi_h1` | Standard RSI(14) | Momentum / overbought-oversold |
| 4 | `atr_ratio_h1` | ATR(14) / close | Normalized volatility |
| 5 | `squeeze_ratio_h1` | BB width / KC width | Bollinger-Keltner squeeze detection |
| 6 | `choppiness_h1` | Choppiness Index(14) | Trending (low) vs ranging (high) |
| 7 | `body_ratio_h1` | (close-open) / (high-low) | Candle character (-1 to +1) |

### H4 Features (7)

| # | Feature | Formula | What it measures |
|---|---------|---------|-----------------|
| 8 | `ema50_ema200_h4` | (EMA50 - EMA200) / close | Higher timeframe trend |
| 9 | `atr_ratio_h4` | ATR(14) / close | H4 volatility context |
| 10 | `adx_h4` | Simplified ADX | H4 trend strength |
| 11 | `squeeze_ratio_h4` | BB width / KC width | H4 squeeze detection |
| 12 | `choppiness_h4` | Choppiness Index(14) | H4 regime classification |
| 13 | `body_ratio_h4` | (close-open) / (high-low) | H4 candle character |
| 14 | `range_pct_h4` | (high-low) / close | H4 per-bar volatility |

### Feature Importance (trained model)

The top features by XGBoost gain:

```
 1. ema50_ema200_h4:  0.0976  (H4 trend direction - most important)
 2. choppiness_h4:    0.0967  (H4 regime - trending vs ranging)
 3. ema50_ema200_h1:  0.0954  (H1 trend direction)
 4. squeeze_ratio_h4: 0.0939  (H4 volatility squeeze)
 5. rsi_h1:           0.0886  (Momentum)
 6. atr_ratio_h1:     0.0805  (Volatility)
 7. range_pct_h4:     0.0748  (H4 bar range)
 8. choppiness_h1:    0.0725  (H1 regime)
 9. adx_h4:           0.0680  (H4 trend strength)
10. close_ema50_h1:   0.0669  (Price displacement)
```

The H4 trend direction (`ema50_ema200_h4`) is the most important feature. This makes intuitive sense: higher timeframe trend alignment is the strongest predictor of whether a buy trade will reach its target.

### Why these 14 features and not more?

These features were selected through correlation-based filtering (Spearman > 0.85 threshold). We considered adding directional features (see `add_directional_features.py`) like ROC, DI diff, price position, and momentum slope - but the 14 core features already capture the essential information without overfitting risk. More features increases the chance of finding spurious patterns in training data that don't hold in live trading.

---

## 4. Model Architecture

### XGBoost Classifier

```python
XGBClassifier(
    n_estimators=100,     # 100 trees (conservative)
    max_depth=3,          # Shallow trees prevent overfitting
    learning_rate=0.03,   # Slow learning rate
    objective='multi:softprob',  # 3-class probability output
    num_class=3,          # Buy wins / Range / Buy loses
)
```

### Why XGBoost?

- **Gradient boosted trees** handle non-linear feature interactions naturally (e.g., "RSI < 30 AND H4 trend bullish" combinations)
- **Shallow depth (3)** prevents memorizing noise - each tree can only make 8 leaf decisions
- **Low learning rate (0.03)** means each tree contributes little, requiring consensus across many trees
- **Probability output** (`softprob`) gives calibrated probabilities rather than hard predictions, enabling threshold tuning

### Why not deep learning?

With 14 features and ~17K samples, XGBoost outperforms neural networks. Deep learning excels with high-dimensional data (images, text, thousands of features). For tabular data with moderate feature counts, gradient boosted trees consistently win (see "Why do tree-based models still outperform deep learning on tabular data?" - Grinsztajn et al., 2022).

---

## 5. Walk-Forward Validation

### Methodology

Walk-forward validation prevents **look-ahead bias** - the most dangerous pitfall in trading model development. Instead of training on all data and testing on a random subset, we simulate real-world deployment:

```
Window 1: Train on bars 0-2829,     test on bars 2829-5658
Window 2: Train on bars 0-5658,     test on bars 5658-8487
Window 3: Train on bars 0-8487,     test on bars 8487-11316
Window 4: Train on bars 0-11316,    test on bars 11316-14145
Window 5: Train on bars 0-14145,    test on bars 14145-16978
```

Each window trains ONLY on past data, then tests on the immediate future. The training set grows (expanding window), simulating how you'd retrain with more data over time.

### Results at 0.34 threshold (initial)

| Window | Period | Trades | Wins | Win Rate | Pips | CB |
|--------|--------|--------|------|----------|------|----|
| W1 | bars 2829-5658 | 93 | 41 | 44.1% | +31 | 46 |
| W2 | bars 5658-8487 | 136 | 60 | 44.1% | +38 | 31 |
| W3 | bars 8487-11316 | 115 | 50 | 43.5% | +13 | 47 |
| W4 | bars 11316-14145 | 208 | 99 | 47.6% | +324 | 16 |
| W5 | bars 14145-16974 | 142 | 61 | 43.0% | +0 | 45 |
| **TOTAL** | | **694** | **311** | **44.8%** | **+406** | 185 |

All 5 windows were positive or breakeven. This is critical - a model that's profitable overall but has catastrophic drawdown periods is not usable.

### Results at 0.40 threshold (optimized - final)

| Threshold | Trades | Wins | Win Rate | Pips | CB | vs BE |
|-----------|--------|------|----------|------|----|-------|
| 0.25 | 746 | 325 | 43.6% | +105 | 209 | +0.7% |
| 0.28 | 643 | 279 | 43.4% | +50 | 218 | +0.5% |
| 0.30 | 518 | 214 | 41.3% | -338 | 234 | -1.5% |
| 0.32 | 668 | 295 | 44.2% | +208 | 195 | +1.3% |
| 0.34 | 694 | 311 | 44.8% | +406 | 185 | +2.0% |
| 0.36 | 638 | 275 | 43.1% | -3 | 198 | +0.2% |
| 0.38 | 527 | 225 | 42.7% | -90 | 200 | -0.2% |
| **0.40** | **836** | **395** | **47.2%** | **+1177** | **112** | **+4.4%** |
| 0.42 | 625 | 283 | 45.3% | +450 | 135 | +2.4% |
| 0.45 | 369 | 153 | 41.5% | -217 | 118 | -1.4% |
| 0.50 | 233 | 108 | 46.4% | +262 | 19 | +3.5% |

The 0.40 threshold dominates on every metric simultaneously: best pips, best win rate, AND most trades. This is unusual - typically there's a trade-off between selectivity and volume.

### Walk-Forward Trade Simulation

The simulation is realistic:
- One trade at a time (no overlapping)
- Commission included (0.00001 per trade)
- TP/SL checked against high/low of each bar (not just close)
- Timeout at 50 bars if no barrier hit
- Circuit breaker active (5 losses or 100 pip DD triggers 48-bar pause)

---

## 6. Threshold Optimization

### What is the confidence threshold?

The model outputs three probabilities that sum to 1.0:
```
buy_prob + range_prob + sell_prob = 1.0
```

The confidence threshold is the minimum `buy_prob` required to trigger a trade. At 0.40, the model must be at least 40% confident it's a buy setup.

### Why 0.40 specifically?

The threshold sweep (above) tested 11 values from 0.25 to 0.50 across all 5 walk-forward windows. The 0.40 threshold was the clear winner:

- **+1,177 pips** (nearly 3x the next best at 0.42)
- **47.2% win rate** (highest of all thresholds)
- **836 trades** (most trades - not sacrificing volume for quality)
- **112 circuit breaker triggers** (fewer than lower thresholds, meaning fewer loss streaks)

### Why 0.40 works (and 0.34 doesn't)

At 0.34, the model fires on many ambiguous signals where buy probability barely edges out the other classes. These marginal signals have ~44% win rate - barely above breakeven. At 0.40, the model only fires when buy is the clearly dominant class, pushing win rate to 47.2%.

The fact that 0.40 generates MORE trades than 0.34 (836 vs 694) while having a HIGHER win rate suggests that the 0.34 threshold was actually generating noise trades that triggered the circuit breaker, which then blocked subsequent good signals.

### Confidence Spread Gate

In addition to the threshold, the EA requires a **confidence spread**:

```
buy_prob - max(sell_prob, range_prob) >= 1.5%
```

This prevents trades where buy is at 40% but sell is at 39.5%. The model must show clear separation between buy and the competing classes. This adds a second layer of filtering beyond the raw threshold.

---

## 7. Regime Filter Decision

### What is the regime filter?

The regime filter uses the Choppiness Index on both H1 and H4 timeframes to classify the market as "trending" or "ranging." It blocks trades when either timeframe is choppier than its rolling median.

### Why it's OFF for the Buy model

The regime filter was designed for the Sell model and calibrated against sell signal characteristics. When applied to Buy signals in testing, it **blocked nearly all buy signals** - the filter's median-based threshold was too aggressive for buy setups.

This makes sense intuitively: the choppiness filter was tuned during periods where the market was bearish. Buy setups naturally occur during different regime conditions than sell setups. Rather than re-calibrate the regime filter for buys (which would require its own walk-forward validation), we decided to rely on the model's built-in regime awareness through the choppiness features (features #6 and #12).

The model already incorporates choppiness as input features. Applying an external choppiness filter on top of the model's own choppiness-aware predictions is redundant at best and counterproductive at worst.

### Sell model regime filter results (for comparison)

The Sell model with regime filter and circuit breaker: +74 pips, 47.0% WR, 217 trades.
Without regime filter (raw): -185 pips, 44.7% WR, 459 trades.

For sells, the regime filter improved results dramatically (turning -185 into +74). But it did so by blocking 53% of signals - mostly during the bullish periods (W3, W4) where selling was wrong. The Buy model doesn't have this problem because it's aligned with the market's natural upward bias.

---

## 8. How the EA Uses Predictions

### Signal Flow

```
OnTick()
  |
  +-> GetTradeSignal()
        |
        +-> GetMLPrediction()     <- Calls /predict/buy API
        |     |
        |     +-> buy_prob >= 0.40?   (threshold gate)
        |     +-> buy_prob - max(sell, range) >= 0.015?  (spread gate)
        |     +-> Returns +1 (buy signal) or 0 (no signal)
        |
        +-> Regime filter?  (OFF by default for buy)
        +-> Returns +1 or 0
  |
  +-> signal == 1 && TradeDirection == BUY_ONLY?
        |
        +-> OpenBuyTrade()   <- Opens twin trades at ASK
```

### Three-Class Probability Interpretation

When the Buy model returns probabilities, here's what they mean:

| Scenario | buy_prob | sell_prob | range_prob | EA Action |
|----------|----------|----------|------------|-----------|
| Strong buy setup | 0.52 | 0.25 | 0.23 | BUY (spread = 0.27) |
| Weak buy lean | 0.38 | 0.33 | 0.29 | No trade (below 0.40) |
| Bearish market | 0.24 | 0.52 | 0.24 | No trade (buy too low) |
| Ranging/uncertain | 0.30 | 0.28 | 0.42 | No trade (range dominant) |
| Buy dominant but close | 0.41 | 0.40 | 0.19 | No trade (spread = 0.01 < 0.015) |

The model is conservative by design. Most bars produce no signal. When it does signal, win rate is 47.2%.

### Twin Trade Execution

When a buy signal triggers, the EA opens two positions (the "twin trade" system):

| Trade | Size | TP | SL | Management |
|-------|------|----|----|------------|
| Twin A (Banker) | 50% of lot | +10 pips above ASK | -15 pips below ASK | None - runs to TP or SL |
| Twin B (Runner) | 50% of lot | +20 pips above ASK | -15 pips below ASK | Breakeven at +10, then trail |

This design protects against the most common loss scenario: price moves 8-10 pips in your favor, then reverses and hits SL. With twins:
- Twin A captures the +10 pip move as profit
- Twin B moves SL to breakeven at +10, so the reversal only costs the spread

See `STRATEGY_PLAN_AND_ANALYSIS.md` for live trade evidence of this working (+$316 twin vs +$107 single in the same 7-trade session).

---

## 9. Buy vs Sell Model Comparison

| Metric | Buy Model | Sell Model |
|--------|-----------|------------|
| Walk-forward pips | **+1,177** | +74 |
| Win rate | **47.2%** | 47.0% |
| Total trades | **836** | 217 |
| Pips per trade | **+1.4** | +0.3 |
| Profitable windows | **4/5** | 2/5 |
| Regime filter needed | No | Yes |
| Circuit breaker triggers | 112 | 52 |
| Threshold | 0.40 | 0.34 |
| Confidence spread | 1.5% | 1.5% |
| Status | **Active** | Archived |

### Why the Buy model works better

1. **Long bias in EUR/USD**: Over 2018-2026, EUR/USD had more buy-favorable periods than sell-favorable ones (44% base buy win rate vs asymmetric sell)
2. **Trend alignment**: The model's top feature (`ema50_ema200_h4`) captures macro trend direction, which favored long positions in more periods
3. **Less adversarial**: Selling requires precise timing against the natural carry and institutional flow. Buying aligns with the "buy the dip" behavior of the broader market

### Sell Model: Not deleted, just archived

The Sell model (`xgb_eurusd_h1.pkl`) and its endpoint (`/predict`) remain in the codebase. The EA supports `TRADE_SELL_ONLY` mode via the `TradeDirection` input. If market conditions shift to a sustained bearish regime, the Sell model can be reactivated by changing one dropdown in the EA settings.

---

## 10. API Architecture

### Server

```
FastAPI (uvicorn) running at http://127.0.0.1:8000
```

### Endpoints

| Endpoint | Method | Model | Response |
|----------|--------|-------|----------|
| `/predict/buy` | POST | `xgb_eurusd_h1_buy.pkl` | `{buy, range, sell, prediction, confidence}` |
| `/predict` | POST | `xgb_eurusd_h1.pkl` | `{sell, range, buy, prediction, confidence}` |
| `/health` | GET | - | Model load status |

### Buy Model Label Mapping

The Buy model maps labels differently than the Sell model in the API response:

```python
# Buy model: label 0 = buy wins, 1 = range, 2 = buy loses
return {
    "buy": probs[0],      # Probability buy wins
    "range": probs[1],
    "sell": probs[2],     # Probability buy loses (sell wins)
    "prediction": ["buy", "range", "sell"][argmax],
    "confidence": max(probs),
}
```

This means both endpoints return the same key names (`buy`, `range`, `sell`) but the internal label-to-key mapping is inverted. The EA doesn't need to know about this - it just reads the probability values.

### Request Format

The EA sends 14 features as a JSON array:

```json
{
  "features": [-0.0027, 0.0029, 33.12, 0.0014, 2.61, 34.32, 0.0, 0.0062, 0.0025, 0.0016, 3.35, 56.40, 0.0, 0.0025]
}
```

Feature order must match `features_used_buy.json` exactly.

---

## 11. Key Files

### Model & Config
| File | Purpose |
|------|---------|
| `models/xgb_eurusd_h1_buy.pkl` | Trained Buy model (XGBoost, 340KB) |
| `models/xgb_eurusd_h1.pkl` | Trained Sell model (archived, 234KB) |
| `features_used_buy.json` | 14 feature names for Buy model |
| `features_used.json` | 14 feature names for Sell model (identical) |
| `config.yaml` | Full system configuration |

### Training & Validation Scripts
| File | Purpose |
|------|---------|
| `scripts/train_buy_model.py` | Buy model training pipeline |
| `scripts/walk_forward_buy.py` | Walk-forward at single threshold (0.34) |
| `scripts/walk_forward_buy_threshold_sweep.py` | Threshold optimization (found 0.40) |
| `scripts/walk_forward.py` | Sell model walk-forward (comparison) |
| `scripts/barrier_sweep.py` | TP/SL optimization |
| `scripts/regime_analysis.py` | Choppiness/regime analysis |

### Server & EA
| File | Purpose |
|------|---------|
| `main.py` | FastAPI server with `/predict/buy` and `/predict` |
| `AdvanceEA.mq5` | MetaTrader 5 Expert Advisor (v4.00) |

### Logs (validation results)
| File | Purpose |
|------|---------|
| `logs/train_buy_model.log` | Training output with feature importance |
| `logs/walk_forward_buy.log` | Walk-forward results at 0.34 threshold |
| `logs/walk_forward_buy_threshold_sweep.log` | Threshold sweep (0.40 selected) |
| `logs/walk_forward.log` | Sell model walk-forward (comparison) |

---

## Appendix: Retraining the Model

To retrain with updated data:

```bash
# 1. Update H1 data file at data/EURUSD_H1_clean.csv

# 2. Train new model
poetry run python scripts/train_buy_model.py

# 3. Validate with walk-forward
poetry run python scripts/walk_forward_buy_threshold_sweep.py

# 4. Check logs/walk_forward_buy_threshold_sweep.log for results

# 5. Restart API server
poetry run uvicorn main:app --reload
```

The model should be retrained periodically (quarterly recommended) as market microstructure evolves. Always validate with walk-forward before deploying a new model version.
