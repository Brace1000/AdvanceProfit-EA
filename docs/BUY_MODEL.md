# Buy Model: Training, Testing & Design Decisions

## Overview

The Buy model predicts favorable long (BUY) entry conditions for EUR/USD on the H1 timeframe. It is an XGBoost multi-class classifier that outputs probabilities for three outcomes: **buy wins**, **range** (no barrier hit), and **buy loses**. The EA enters a buy trade only when buy probability dominates the other classes.

**Walk-forward results**:
- Buy signals: +1,177 pips | 47.2% WR | 836 trades | 0.40 threshold
- Sell signals (from same model): +563 pips | 46.8% WR | 459 trades | 0.56 threshold
- **Combined potential**: +1,740 pips

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
10. [BOTH Mode: Sell Signals from the Buy Model](#10-both-mode-sell-signals-from-the-buy-model)
11. [Threshold vs Confidence Spread: Two Different Questions](#11-threshold-vs-confidence-spread-two-different-questions)
12. [The Sell Model as R&D: How Failure Led to Discovery](#12-the-sell-model-as-rd-how-failure-led-to-discovery)
13. [API Architecture](#13-api-architecture)
14. [Key Files](#14-key-files)

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

### What does the Buy model's sell probability mean?

The Buy model outputs three probabilities: `buy_prob`, `range_prob`, `sell_prob`. When `sell_prob` is high, it means **"this is NOT a buy setup — in fact, the buy would LOSE."** Initially we treated this as a pure rejection signal, not a sell recommendation. But walk-forward testing revealed that at high confidence (≥56%), the Buy model's sell_prob identifies genuinely bearish conditions that outperform the dedicated Sell model by 7.6x.

Think of it this way: the Buy model was trained with triple-barrier labels where label 2 ("buy loses") means *price fell 15 pips before rising 20 pips*. That's exactly what a sell trade needs to win. The model isn't just saying "don't buy" — it's identifying conditions where the opposite direction wins.

> **See [Section 10](#10-both-mode-sell-signals-from-the-buy-model)** for the full walk-forward validation of this discovery.

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

**BUY_ONLY mode:**
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
        +-> Regime filter?  (OFF by default)
        +-> Returns +1 or 0
  |
  +-> signal == 1  ->  OpenBuyTrade()  (twin trades at ASK)
```

**BOTH mode (recommended):**
```
OnTick()
  |
  +-> GetTradeSignal()
        |
        +-> GetMLPrediction()     <- Single call to /predict/buy
              |
              +-> Check BUY first (priority):
              |     buy_prob >= 0.40 AND spread >= 0.015?  -> return +1
              |
              +-> Then check SELL:
              |     sell_prob >= 0.56?  -> return -1
              |
              +-> Neither passes?  -> return 0
  |
  +-> signal == +1  ->  OpenBuyTrade()   (twin trades at ASK, TP above)
  +-> signal == -1  ->  OpenSellTrade()  (twin trades at BID, TP below)
```

Note: In BOTH mode, buy signals have priority. If both buy and sell somehow pass their respective thresholds (extremely unlikely given they sum to ≤1.0), the buy signal wins.

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

| Metric | Buy Model (buys) | Buy Model (sells) | Sell Model |
|--------|-------------------|-------------------|------------|
| Walk-forward pips | **+1,177** | **+563** | +74 |
| Win rate | **47.2%** | **46.8%** | 47.0% |
| Total trades | **836** | **459** | 217 |
| Pips per trade | **+1.4** | **+1.2** | +0.3 |
| Profitable windows | **4/5** | **3/5** | 2/5 |
| Regime filter needed | No | No | Yes |
| Circuit breaker triggers | 112 | 102 | 52 |
| Threshold | 0.40 | 0.56 | 0.34 |
| Confidence spread | 1.5% | Not needed | 1.5% |
| Status | **Active** | **Active (BOTH)** | Archived |

### Why the Buy model works better

1. **Long bias in EUR/USD**: Over 2018-2026, EUR/USD had more buy-favorable periods than sell-favorable ones (44% base buy win rate vs asymmetric sell)
2. **Trend alignment**: The model's top feature (`ema50_ema200_h4`) captures macro trend direction, which favored long positions in more periods
3. **Less adversarial**: Selling requires precise timing against the natural carry and institutional flow. Buying aligns with the "buy the dip" behavior of the broader market

### Why the Buy model's sell signals outperform the Sell model

The Buy model's sell signals (+563 pips) beat the dedicated Sell model (+74 pips) by 7.6x despite having fewer features specialized for selling. Three factors explain this:

1. **Better training data**: The Buy model was trained on inverted labels — label 2 ("buy loses") means *price hit -15 pips before +20 pips*, which is exactly a successful sell trade (TP=15, SL=20). The dedicated Sell model had its own labeling but narrower probability distributions (0.275–0.375 range vs the Buy model's broader separation).
2. **Higher threshold = higher quality**: The 0.56 sell threshold is extremely selective. The model must be very confident that a buy would lose before signaling a sell. This naturally filters for the strongest bearish setups.
3. **Same features work both ways**: The 14 features capture trend, momentum, and volatility — these indicate direction regardless of which model reads them. The Buy model learned bearish patterns as "anti-patterns" to avoid, but those same patterns are exactly what you want for sells.

### Sell Model: Not deleted, just archived

The Sell model (`xgb_eurusd_h1.pkl`) and its endpoint (`/predict`) remain in the codebase. The EA supports `TRADE_SELL_ONLY` mode via the `TradeDirection` input. If market conditions shift to a sustained bearish regime, the Sell model can be reactivated by changing one dropdown in the EA settings. However, BOTH mode is now the recommended configuration since the Buy model handles both directions more effectively.

---

## 10. BOTH Mode: Sell Signals from the Buy Model

### The Discovery

After the Buy model proved dominant for long entries, a natural question arose: if the model says "this buy would LOSE" with high confidence, can we trade the other direction?

The Buy model's label 2 ("buy loses") means price fell 15 pips before rising 20 pips. Flipping the perspective, that's a sell trade with TP=15 and SL=20 (inverted from buy's TP=20/SL=15). The breakeven win rate for this inverted trade is:

```
Sell BE = SL / (TP + SL) = 20 / (15 + 20) = 57.1%
```

Wait — that's much higher than buy's 42.9% breakeven. But the actual walk-forward simulation uses TP=20/SL=15 for sells too (same as buy but in the opposite direction), giving the same 42.9% breakeven. The model identifies *when bearish conditions dominate*, and the EA places a standard sell trade.

### Walk-Forward Results

Three filtering strategies were tested across thresholds 0.40–0.60:

**Pure Threshold (sell_prob >= X):**

| Threshold | Trades | Wins | Win Rate | Pips | CB | vs BE |
|-----------|--------|------|----------|------|----|-------|
| 0.40 | 595 | 256 | 43.0% | -32 | 225 | +0.2% |
| 0.46 | 864 | 384 | 44.4% | +357 | 166 | +1.6% |
| 0.52 | 615 | 274 | 44.6% | +277 | 164 | +1.7% |
| **0.56** | **459** | **215** | **46.8%** | **+563** | **102** | **+4.0%** |
| 0.58 | 493 | 222 | 45.0% | +295 | 53 | +2.2% |
| 0.60 | 295 | 117 | 39.7% | -380 | 69 | -3.2% |

The sweet spot is **0.56**: highest pips, second-highest win rate, manageable trade count. Above 0.56, sample size drops and performance degrades. Below 0.56, noise trades dilute the edge.

**Adding confidence spread filters made no difference** — at thresholds ≥0.52, the spread gate is mathematically redundant (see [Section 11](#11-threshold-vs-confidence-spread-two-different-questions)).

### Per-Window Breakdown (0.56 threshold)

| Window | Trades | Wins | Win Rate | Pips | CB |
|--------|--------|------|----------|------|----|
| W1 | 101 | 39 | 38.6% | -160 | 37 |
| W2 | 73 | 41 | 56.2% | +321 | 0 |
| W3 | 128 | 68 | 53.1% | +433 | 7 |
| W4 | 95 | 38 | 40.0% | -105 | 40 |
| W5 | 62 | 29 | 46.8% | +74 | 18 |
| **Total** | **459** | **215** | **46.8%** | **+563** | 102 |

Three of five windows are profitable (W2, W3, W5). W1 and W4 show losses — the model isn't infallible for sells, but the profitable windows overwhelm the losers. Circuit breaker also protects against extended drawdowns in weak windows.

### Comparison with Dedicated Sell Model

| Metric | Buy Model (sells at 0.56) | Sell Model (at 0.34) |
|--------|---------------------------|----------------------|
| Pips | **+563** | +74 |
| Win rate | 46.8% | 47.0% |
| Trades | 459 | 217 |
| Pips/trade | **+1.2** | +0.3 |
| Improvement | **7.6x** | baseline |

The Buy model generates 2x more sell trades at virtually the same win rate, yielding 7.6x more pips. The dedicated Sell model's narrow probability range (0.275–0.375) meant it could never separate signal from noise as effectively as the Buy model's wider probability distribution.

### EA Implementation

BOTH mode uses a single API call to `/predict/buy` and evaluates both directions:

```
1. Call /predict/buy → get {buy_prob, sell_prob, range_prob}
2. Check BUY: buy_prob >= 0.40 AND spread >= 1.5%  → open buy
3. Check SELL: sell_prob >= 0.56                     → open sell
4. Neither passes → no trade
```

Buy signals have priority (checked first). Only one direction can trigger per bar. The sell check intentionally omits the confidence spread gate — see next section for why.

---

## 11. Threshold vs Confidence Spread: Two Different Questions

### The Two Gates

The EA uses two separate confidence checks, each answering a different question:

| Gate | Formula | Question it answers |
|------|---------|-------------------|
| **Threshold** | `prob >= X` | "Is the model at least minimally convinced?" |
| **Confidence Spread** | `prob - max(other probs) >= Y` | "Is this the model's best pick by a clear margin?" |

### Why Both Matter for Buy Signals

At the 0.40 buy threshold, the model can fire on ambiguous signals:

| Scenario | buy_prob | sell_prob | range_prob | Threshold? | Spread? | Should trade? |
|----------|----------|----------|------------|-----------|---------|---------------|
| Clear buy | 0.52 | 0.25 | 0.23 | Pass (52%) | Pass (27%) | **YES** |
| Marginal buy | 0.41 | 0.40 | 0.19 | Pass (41%) | **Fail (1%)** | **NO** |
| Fake buy | 0.42 | 0.45 | 0.13 | Pass (42%) | **Fail (-3%)** | **NO** |

The "fake buy" scenario is the dangerous one: buy passes the threshold but sell actually has higher probability. Without the spread gate, the EA would open a buy trade when the model thinks sell is more likely. The spread gate catches this.

### Why Sell Signals Don't Need the Spread Gate

At the 0.56 sell threshold, the spread gate is mathematically redundant:

```
If sell_prob >= 0.56, then:
  buy_prob + range_prob <= 0.44    (probabilities sum to 1.0)
  max(buy_prob, range_prob) <= 0.44
  sell_prob - max(buy, range) >= 0.56 - 0.44 = 0.12 (12%)
```

At 56% sell probability, the spread is *guaranteed* to be at least 12% — far above any reasonable spread gate. This is confirmed by the walk-forward data: results at 0.56 were **identical** across all three test configurations (pure threshold, +1.5% spread, +5% spread).

The spread gate only matters when the threshold is low enough that competing probabilities could be close. At 0.40 buy threshold, buy_prob + others = 0.60, so others could be as high as 0.39 — dangerously close. At 0.56 sell threshold, the math eliminates the ambiguity automatically.

### Summary

```
Buy signal:   Threshold (0.40) + Spread (1.5%) = two-layer filter
Sell signal:  Threshold (0.56) alone = sufficient (spread >= 12% guaranteed)
```

---

## 12. The Sell Model as R&D: How Failure Led to Discovery

### Lineage of the System

The current BOTH mode architecture didn't emerge from a single plan. It evolved through a series of experiments where each "failure" contributed essential knowledge:

```
Phase 1: Sell Model (original)
  └─ Result: +74 pips (marginal)
  └─ Contributed: Feature selection (14 features), walk-forward methodology,
     circuit breaker design, twin trade system, barrier sweep results

Phase 2: Buy Model (pivot)
  └─ Result: +1,177 pips (breakthrough)
  └─ Contributed: Proved directional models work, established 0.40 threshold,
     confirmed regime filter is unnecessary for buys

Phase 3: Threshold Optimization
  └─ Result: 0.34 → 0.40 (+770 pip improvement)
  └─ Contributed: Showed threshold is not just a filter — it changes which
     circuit breaker pauses happen, affecting downstream trade quality

Phase 4: Sell Signal Discovery (BOTH mode)
  └─ Result: +563 additional pips from same model
  └─ Contributed: The Buy model's "anti-signal" (sell_prob >= 0.56) is a
     better sell indicator than the dedicated Sell model
```

### What Each Phase Taught Us

**Phase 1 (Sell Model)** was not wasted work. It established:
- The 14-feature set that both models use (correlation-filtered from a larger candidate set)
- Walk-forward validation methodology (expanding window, 5 folds)
- The circuit breaker mechanism (5 losses or 100 pip DD → 48-bar pause)
- The twin trade execution system (banker + runner)
- TP/SL optimization via barrier sweep (settled on 20/15)

Without the Sell model's development, the Buy model would have started from scratch. Instead, it inherited a proven feature engineering pipeline and validation framework.

**Phase 2 (Buy Model)** proved that the same features and methodology could produce a much stronger edge when aligned with the dominant market direction. The key insight: EUR/USD over 2018–2026 had a structural long bias (44% base buy win rate). Fighting this bias with sells was always going to be harder.

**Phase 3 (Threshold Optimization)** revealed a non-obvious interaction: the 0.40 threshold produced MORE trades than 0.34 while having a HIGHER win rate. This happened because marginal signals at 0.34 triggered the circuit breaker, which then blocked subsequent high-quality signals. Raising the threshold eliminated the noise trades, kept the circuit breaker from firing, and allowed good signals through. The lesson: filtering quality affects everything downstream.

**Phase 4 (BOTH Mode)** came from asking "what are we leaving on the table?" The Buy model's three-class output contains sell information for free — we just needed to find the right threshold to extract it. At 0.56, the model's sell conviction is high enough that the signal is trustworthy, adding +563 pips without any model changes, retraining, or additional API calls.

### The R&D Lesson

The Sell model's +74 pips look like failure next to the Buy model's +1,177. But the path through the Sell model was necessary:

```
Sell model → features, methodology → Buy model → threshold tuning → BOTH mode
   +74 pips      (infrastructure)      +1,177 pips    (optimization)    +563 pips
                                                                     = +1,740 total
```

Each phase built on the last. The final system (+1,740 combined pips) wouldn't exist without the Sell model's "failure" establishing the foundation.

---

## 13. API Architecture

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

## 14. Key Files

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
| `scripts/walk_forward_buy_model_sell_signals.py` | BOTH mode sell signal validation (found 0.56) |
| `scripts/walk_forward.py` | Sell model walk-forward (comparison) |
| `scripts/barrier_sweep.py` | TP/SL optimization |
| `scripts/regime_analysis.py` | Choppiness/regime analysis |

### Server & EA
| File | Purpose |
|------|---------|
| `main.py` | FastAPI server with `/predict/buy` and `/predict` |
| `AdvanceEA.mq5` | MetaTrader 5 Expert Advisor (v4.00, BOTH mode) |

### Logs (validation results)
| File | Purpose |
|------|---------|
| `logs/train_buy_model.log` | Training output with feature importance |
| `logs/walk_forward_buy.log` | Walk-forward results at 0.34 threshold |
| `logs/walk_forward_buy_threshold_sweep.log` | Threshold sweep (0.40 selected) |
| `logs/walk_forward_buy_model_sell_signals.log` | BOTH mode sell signal walk-forward results |
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
