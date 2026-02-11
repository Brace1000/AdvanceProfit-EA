//+------------------------------------------------------------------+
//|                                     AdvancedProfitEA_ML.mq5      |
//|                        ML Trading System (14 features)            |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property version   "4.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

// Trade direction enum
enum ENUM_TRADE_DIRECTION
{
   TRADE_BUY_ONLY,   // BUY ONLY
   TRADE_SELL_ONLY,   // SELL ONLY
   TRADE_BOTH         // BOTH (Buy + Sell via Buy model)
};

// Input Parameters
input group "=== ML API Settings ==="
input ENUM_TRADE_DIRECTION TradeDirection = TRADE_BUY_ONLY; // Trade Direction
input string API_URL_Buy  = "http://127.0.0.1:8000/predict/buy"; // Buy API endpoint
input string API_URL_Sell = "http://127.0.0.1:8000/predict";     // Sell API endpoint
input double ML_Confidence_Threshold = 0.40; // Buy confidence threshold (40%)
input double ML_Sell_Threshold = 0.56;       // Sell confidence threshold (56%, BOTH mode)
input double MinConfidenceSpread = 0.015;    // Signal must beat others by 1.5%

input group "=== Risk Management ==="
input double RiskPercent = 1.0;              // Risk per trade (% of balance)
input double MaxDailyLoss = 3.0;             // Max daily loss (% of balance)
input double MaxDailyProfit = 5.0;           // Daily profit target (%)
input int    MaxSimultaneousTrades = 2;      // Max AT-RISK positions (protected positions don't count)
input double TP_Pips = 20.0;                // Take Profit (pips) - fixed barrier
input double SL_Pips = 15.0;                // Stop Loss (pips) - fixed barrier

input group "=== Technical Strategy ==="
input int    MA_Fast = 10;                   // Fast MA Period
input int    MA_Slow = 30;                   // Slow MA Period
input int    RSI_Period = 14;                // RSI Period
input int    RSI_Overbought = 70;            // RSI Overbought Level
input int    RSI_Oversold = 30;              // RSI Oversold Level
input int    TrendMA_Period = 50;            // Trend MA Period
input bool   UseTrendFilter = true;          // Only trade with trend
input bool   TechnicalOnly = false;           // Technical signals only (no ML)
input bool   CombineWithTechnical = false;   // Require both ML + Technical (ignored if TechnicalOnly)

input group "=== Trade Filters ==="
input bool   UseTimeFilter = true;           // Enable time filter
input int    StartHour = 8;                  // Trading start hour
input int    EndHour = 20;                   // Trading end hour

input group "=== Regime Filter ==="
input bool   UseRegimeFilter = false;        // Only trade in trending regimes
input int    RegimeLookback = 500;           // Bars for rolling median

input group "=== Circuit Breaker ==="
input bool   CB_Enabled = true;              // Enable circuit breaker
input int    CB_MaxConsecLosses = 5;         // Max consecutive losses before pause
input double CB_MaxDrawdownPips = 100.0;     // Max drawdown (pips) before pause
input int    CB_CooldownBars = 48;           // Bars to pause after trigger
input bool   CB_ResetOnWin = true;           // Reset loss counter on win

input group "=== Twin Trade System ==="
input bool   UseTwinTrades = true;           // Enable twin trade system
input double TwinA_TP_Pips = 10.0;           // Trade A (Banker): Take Profit pips
input double TwinB_TP_Pips = 20.0;           // Trade B (Runner): Take Profit pips
input double TwinB_BE_Trigger = 10.0;        // Trade B: Breakeven trigger pips
input double TwinB_BE_Offset = 2.0;          // Trade B: Lock in pips at breakeven
input double TwinB_Trail_Pips = 8.0;         // Trade B: Trailing stop distance
input double TwinB_Trail_Step = 2.0;         // Trade B: Trailing step size

input group "=== Position Management (Single Trade) ==="
input bool   UseTrailingStop = true;         // Enable trailing stop
input double TrailingStopPips = 8;           // Trailing stop distance (pips)
input double TrailingStepPips = 2;           // Trailing step (pips)
input bool   UseBreakeven = true;            // Move SL to breakeven
input double BreakevenTriggerPips = 10;      // Pips profit to trigger BE
input double BreakevenOffsetPips = 2;        // BE offset (lock in pips)
input bool   UsePartialClose = false;        // Partial close (disabled for twin)
input double PartialClosePercent = 50.0;     // % to close at first target

// Global Variables
CTrade trade;
CPositionInfo posInfo;
CAccountInfo accInfo;

// Technical indicator handles
int handleMA_Fast, handleMA_Slow, handleRSI, handleATR, handleTrendMA;

// ML indicator handles (H1 + H4)
int handleEMA50_H1, handleEMA200_H1, handleRSI_H1, handleATR_H1;
int handleEMA50_H4, handleEMA200_H4, handleATR_H4;

double dailyStartBalance;
datetime lastBarTime;
int totalTradesToday = 0;
double dailyProfitLoss = 0.0;

// Circuit breaker state
int    cb_consecutive_losses = 0;
double cb_peak_pnl = 0.0;
double cb_running_pnl = 0.0;
int    cb_bar_counter = 0;
int    cb_paused_until_bar = 0;
int    cb_triggers = 0;
int    cb_last_position_count = 0;

// Regime filter cached values (updated each bar)
double last_chop_h1 = 50.0;
double last_chop_h4 = 50.0;

// Forward declarations
string StringTrimCustom(const string str);
double StringToDoubleCustom(const string value);
double ComputeSqueezeRatio(ENUM_TIMEFRAMES tf, int shift);
double ComputeChoppiness(ENUM_TIMEFRAMES tf, int shift);
double ComputeSimplifiedADX(ENUM_TIMEFRAMES tf, int shift);
double ComputeRollingMedian(ENUM_TIMEFRAMES tf, int shift, int lookback);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("Advanced Profit EA v4.00 - ML Trading System (14 features)");
   Print("========================================");

   // Technical indicators
   handleMA_Fast = iMA(_Symbol, _Period, MA_Fast, 0, MODE_SMA, PRICE_CLOSE);
   handleMA_Slow = iMA(_Symbol, _Period, MA_Slow, 0, MODE_SMA, PRICE_CLOSE);
   handleRSI = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
   handleATR = iATR(_Symbol, _Period, 14);
   handleTrendMA = iMA(_Symbol, _Period, TrendMA_Period, 0, MODE_SMA, PRICE_CLOSE);

   // ML indicators: H1
   handleEMA50_H1  = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA200_H1 = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
   handleRSI_H1    = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   handleATR_H1    = iATR(_Symbol, PERIOD_H1, 14);

   // ML indicators: H4
   handleEMA50_H4  = iMA(_Symbol, PERIOD_H4, 50, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA200_H4 = iMA(_Symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);
   handleATR_H4    = iATR(_Symbol, PERIOD_H4, 14);

   if(handleMA_Fast == INVALID_HANDLE || handleMA_Slow == INVALID_HANDLE ||
      handleRSI == INVALID_HANDLE || handleATR == INVALID_HANDLE ||
      handleTrendMA == INVALID_HANDLE ||
      handleEMA50_H1 == INVALID_HANDLE || handleEMA200_H1 == INVALID_HANDLE ||
      handleRSI_H1 == INVALID_HANDLE || handleATR_H1 == INVALID_HANDLE ||
      handleEMA50_H4 == INVALID_HANDLE || handleEMA200_H4 == INVALID_HANDLE ||
      handleATR_H4 == INVALID_HANDLE)
   {
      Print("Error initializing indicators!");
      return(INIT_FAILED);
   }

   dailyStartBalance = accInfo.Balance();
   lastBarTime = 0;

   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   string dirStr = (TradeDirection == TRADE_BUY_ONLY) ? "BUY ONLY" :
                   (TradeDirection == TRADE_SELL_ONLY) ? "SELL ONLY" : "BOTH (Buy + Sell)";
   Print("Trade Direction: ", dirStr);
   Print("Mode: ", TechnicalOnly ? "TECHNICAL ONLY (no ML)" :
         (CombineWithTechnical ? "ML + Technical Confluence" : "ML-Only"));
   if(!TechnicalOnly)
   {
      Print("API URL: ", GetActiveAPIURL());
      Print("Buy Threshold: ", DoubleToString(ML_Confidence_Threshold * 100, 0), "%");
      if(TradeDirection == TRADE_BOTH)
         Print("Sell Threshold: ", DoubleToString(ML_Sell_Threshold * 100, 0), "%");
      Print("Min Confidence Spread: ", DoubleToString(MinConfidenceSpread * 100, 1), "% (signal must beat others)");
   }
   Print("TP: ", TP_Pips, " pips | SL: ", SL_Pips, " pips");
   if(UseTwinTrades)
      Print("Twin Trades: ON | A(Banker)=", TwinA_TP_Pips, "p | B(Runner)=", TwinB_TP_Pips,
            "p BE@", TwinB_BE_Trigger, " Trail@", TwinB_Trail_Pips);
   else
      Print("Twin Trades: OFF (single trade mode)");
   Print("Technical: MA(", MA_Fast, ",", MA_Slow, ") + RSI(", RSI_Period, ") + Trend(", TrendMA_Period, ")");
   Print("Regime Filter: ", UseRegimeFilter ? "ON" : "OFF");
   Print("Circuit Breaker: ", CB_Enabled ? "ON" : "OFF");
   Print("Risk per trade: ", RiskPercent, "%");
   Print("Starting balance: $", DoubleToString(dailyStartBalance, 2));

   cb_last_position_count = CountOpenPositions();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(handleMA_Fast);
   IndicatorRelease(handleMA_Slow);
   IndicatorRelease(handleRSI);
   IndicatorRelease(handleATR);
   IndicatorRelease(handleTrendMA);

   IndicatorRelease(handleEMA50_H1);
   IndicatorRelease(handleEMA200_H1);
   IndicatorRelease(handleRSI_H1);
   IndicatorRelease(handleATR_H1);

   IndicatorRelease(handleEMA50_H4);
   IndicatorRelease(handleEMA200_H4);
   IndicatorRelease(handleATR_H4);

   Print("EA Stopped. Total trades today: ", totalTradesToday);
   Print("Daily P&L: $", DoubleToString(dailyProfitLoss, 2));
   Print("Circuit breaker triggers: ", cb_triggers);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for closed trades (circuit breaker tracking)
   if(CB_Enabled)
      CheckClosedTrades();

   datetime currentBarTime = iTime(_Symbol, PERIOD_H1, 0);
   bool isNewBar = (currentBarTime != lastBarTime);

   if(isNewBar)
   {
      lastBarTime = currentBarTime;
      cb_bar_counter++;

      CheckDailyReset();

      if(!CheckDailyLimits())
      {
         CloseAllPositions("Daily limit reached");
         return;
      }

      if(UseTrailingStop)
         ManageTrailingStops();

      ManagePositions();

      // Circuit breaker check
      if(CB_Enabled && cb_bar_counter < cb_paused_until_bar)
      {
         Print("Circuit breaker active - paused until bar ", cb_paused_until_bar,
               " (current: ", cb_bar_counter, ")");
         return;
      }

      // Use at-risk count: protected positions (BE activated) don't block new trades
      int atRisk = CountAtRiskPositions();
      int totalOpen = CountOpenPositions();

      if(atRisk < MaxSimultaneousTrades)
      {
         int signal = GetTradeSignal();

         if(signal == 1 && (TradeDirection == TRADE_BUY_ONLY || TradeDirection == TRADE_BOTH))
         {
            if(totalOpen > atRisk)
               Print("Opening new trade: ", totalOpen, " open but only ", atRisk, " at risk");
            OpenBuyTrade();
         }
         else if(signal == -1 && (TradeDirection == TRADE_SELL_ONLY || TradeDirection == TRADE_BOTH))
         {
            if(totalOpen > atRisk)
               Print("Opening new trade: ", totalOpen, " open but only ", atRisk, " at risk");
            OpenSellTrade();
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Get technical signal (MA crossover + RSI)                         |
//+------------------------------------------------------------------+
int GetTechnicalSignal()
{
   double maFast[], maSlow[], rsi[], trendMA[];
   ArraySetAsSeries(maFast, true);
   ArraySetAsSeries(maSlow, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(trendMA, true);

   if(CopyBuffer(handleMA_Fast, 0, 0, 3, maFast) < 3 ||
      CopyBuffer(handleMA_Slow, 0, 0, 3, maSlow) < 3 ||
      CopyBuffer(handleRSI, 0, 0, 1, rsi) < 1 ||
      CopyBuffer(handleTrendMA, 0, 0, 1, trendMA) < 1)
      return 0;

   double close = iClose(_Symbol, _Period, 1);

   bool upTrend = close > trendMA[0];
   bool downTrend = close < trendMA[0];

   // Bullish crossover: fast MA crosses above slow MA
   bool bullishCross = (maFast[1] > maSlow[1] && maFast[2] <= maSlow[2]);

   // RSI in bullish zone
   bool rsiBullish = rsi[0] > 50 && rsi[0] < RSI_Overbought;

   if(bullishCross && rsiBullish)
   {
      if(!UseTrendFilter || upTrend)
      {
         Print("Technical BUY: Bullish cross + RSI=", DoubleToString(rsi[0], 1));
         return 1;
      }
   }

   // Bearish crossover: fast MA crosses below slow MA
   bool bearishCross = (maFast[1] < maSlow[1] && maFast[2] >= maSlow[2]);

   // RSI in bearish zone
   bool rsiBearish = rsi[0] < 50 && rsi[0] > RSI_Oversold;

   if(bearishCross && rsiBearish)
   {
      if(!UseTrendFilter || downTrend)
      {
         Print("Technical SELL: Bearish cross + RSI=", DoubleToString(rsi[0], 1));
         return -1;
      }
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Compute Squeeze Ratio (BB width / KC width)                      |
//| Matches Python: engineer.py _squeeze_ratio()                     |
//+------------------------------------------------------------------+
double ComputeSqueezeRatio(ENUM_TIMEFRAMES tf, int shift)
{
   // BB width = 4 * StdDev(close, 20)
   double closes[];
   ArraySetAsSeries(closes, true);
   if(CopyClose(_Symbol, tf, shift, 20, closes) < 20)
      return 1.0;

   double sum = 0, sum2 = 0;
   for(int i = 0; i < 20; i++)
   {
      sum += closes[i];
      sum2 += closes[i] * closes[i];
   }
   double mean = sum / 20.0;
   double variance = sum2 / 20.0 - mean * mean;
   double std_dev = MathSqrt(MathMax(0.0, variance));
   double bb_width = 4.0 * std_dev;

   // KC width = 2 * SMA(ATR(14), 20)
   double atr_vals[];
   ArraySetAsSeries(atr_vals, true);
   int atr_handle = (tf == PERIOD_H1) ? handleATR_H1 : handleATR_H4;
   if(CopyBuffer(atr_handle, 0, shift, 20, atr_vals) < 20)
      return 1.0;

   double atr_sum = 0;
   for(int i = 0; i < 20; i++)
      atr_sum += atr_vals[i];
   double avg_atr = atr_sum / 20.0;
   double kc_width = 2.0 * avg_atr;

   if(kc_width < 1e-10)
      return 1.0;

   return bb_width / kc_width;
}

//+------------------------------------------------------------------+
//| Compute Choppiness Index (14-period)                             |
//| Matches Python: engineer.py _choppiness()                        |
//| Values near 100 = choppy/ranging, near 0 = trending              |
//+------------------------------------------------------------------+
double ComputeChoppiness(ENUM_TIMEFRAMES tf, int shift)
{
   int period = 14;

   double highs[], lows[], closes[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);

   // Need period bars + 1 previous close for TR calculation
   if(CopyHigh(_Symbol, tf, shift, period, highs) < period ||
      CopyLow(_Symbol, tf, shift, period, lows) < period ||
      CopyClose(_Symbol, tf, shift, period + 1, closes) < period + 1)
      return 50.0;

   double tr_sum = 0;
   double high_max = -DBL_MAX;
   double low_min = DBL_MAX;

   for(int i = 0; i < period; i++)
   {
      double prev_close = closes[i + 1]; // previous close (series: i+1 is older)
      double h = highs[i];
      double l = lows[i];

      double tr = MathMax(h - l, MathMax(MathAbs(h - prev_close), MathAbs(l - prev_close)));
      tr_sum += tr;

      high_max = MathMax(high_max, h);
      low_min = MathMin(low_min, l);
   }

   double price_range = high_max - low_min;
   if(price_range < 1e-10)
      return 50.0;

   double chop = 100.0 * MathLog10(tr_sum / price_range) / MathLog10((double)period);
   return MathMax(0.0, MathMin(100.0, chop));
}

//+------------------------------------------------------------------+
//| Compute Simplified ADX (NOT standard iADX)                       |
//| Matches Python: engineer.py _adx_like()                          |
//| = SMA(14) of |plus_dm - minus_dm|                                |
//+------------------------------------------------------------------+
double ComputeSimplifiedADX(ENUM_TIMEFRAMES tf, int shift)
{
   int period = 14;

   double highs[], lows[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);

   // Need period + 1 bars for diff calculation
   if(CopyHigh(_Symbol, tf, shift, period + 1, highs) < period + 1 ||
      CopyLow(_Symbol, tf, shift, period + 1, lows) < period + 1)
      return 0.0;

   double sum = 0;
   for(int i = 0; i < period; i++)
   {
      // diff: highs[i] - highs[i+1] (i+1 is older in series mode)
      double plus_dm = MathMax(highs[i] - highs[i + 1], 0.0);
      double minus_dm = MathMax(lows[i + 1] - lows[i], 0.0);
      sum += MathAbs(plus_dm - minus_dm);
   }

   return sum / period;
}

//+------------------------------------------------------------------+
//| Compute rolling median of choppiness for regime filter            |
//+------------------------------------------------------------------+
double ComputeRollingMedian(ENUM_TIMEFRAMES tf, int shift, int lookback)
{
   double values[];
   ArrayResize(values, lookback);
   int count = 0;

   for(int i = shift; i < shift + lookback && count < lookback; i++)
   {
      double chop = ComputeChoppiness(tf, i);
      if(chop > 0 && chop < 100)
      {
         values[count] = chop;
         count++;
      }
   }

   if(count < 10)
      return 50.0; // Not enough data, use neutral value

   // Sort for median
   ArrayResize(values, count);
   ArraySort(values);

   if(count % 2 == 0)
      return (values[count / 2 - 1] + values[count / 2]) / 2.0;
   else
      return values[count / 2];
}

//+------------------------------------------------------------------+
//| Get active API URL based on trade direction                      |
//+------------------------------------------------------------------+
string GetActiveAPIURL()
{
   if(TradeDirection == TRADE_BUY_ONLY || TradeDirection == TRADE_BOTH)
      return API_URL_Buy;
   else
      return API_URL_Sell;
}

//+------------------------------------------------------------------+
//| Get ML prediction from Python API (14 features)                  |
//| Feature order matches features_used.json exactly                 |
//+------------------------------------------------------------------+
int GetMLPrediction(double &confidence, double &chop_h1_out, double &chop_h4_out)
{
   // --- H1 indicator buffers ---
   double ema50_h1[3], ema200_h1[3], rsi_h1[2], atr_h1[2];
   double close_h1[3], open_h1[2], high_h1[2], low_h1[2];

   // --- H4 indicator buffers ---
   double ema50_h4[3], ema200_h4[3], atr_h4[2];
   double close_h4[3], open_h4[2], high_h4[2], low_h4[2];

   ArraySetAsSeries(ema50_h1, true); ArraySetAsSeries(ema200_h1, true);
   ArraySetAsSeries(rsi_h1, true);   ArraySetAsSeries(atr_h1, true);
   ArraySetAsSeries(close_h1, true); ArraySetAsSeries(open_h1, true);
   ArraySetAsSeries(high_h1, true);  ArraySetAsSeries(low_h1, true);

   ArraySetAsSeries(ema50_h4, true); ArraySetAsSeries(ema200_h4, true);
   ArraySetAsSeries(atr_h4, true);
   ArraySetAsSeries(close_h4, true); ArraySetAsSeries(open_h4, true);
   ArraySetAsSeries(high_h4, true);  ArraySetAsSeries(low_h4, true);

   // Copy H1 indicators (need 3 bars for EMA shift)
   if(CopyBuffer(handleEMA50_H1, 0, 0, 3, ema50_h1) < 3 ||
      CopyBuffer(handleEMA200_H1, 0, 0, 3, ema200_h1) < 3 ||
      CopyBuffer(handleRSI_H1, 0, 0, 2, rsi_h1) < 2 ||
      CopyBuffer(handleATR_H1, 0, 0, 2, atr_h1) < 2)
   {
      Print("Failed to copy H1 indicator data");
      return 0;
   }

   // Copy H1 prices
   if(CopyClose(_Symbol, PERIOD_H1, 0, 3, close_h1) < 3 ||
      CopyOpen(_Symbol, PERIOD_H1, 0, 2, open_h1) < 2 ||
      CopyHigh(_Symbol, PERIOD_H1, 0, 2, high_h1) < 2 ||
      CopyLow(_Symbol, PERIOD_H1, 0, 2, low_h1) < 2)
   {
      Print("Failed to copy H1 price data");
      return 0;
   }

   // Copy H4 indicators (need 3 bars for EMA shift)
   if(CopyBuffer(handleEMA50_H4, 0, 0, 3, ema50_h4) < 3 ||
      CopyBuffer(handleEMA200_H4, 0, 0, 3, ema200_h4) < 3 ||
      CopyBuffer(handleATR_H4, 0, 0, 2, atr_h4) < 2)
   {
      Print("Failed to copy H4 indicator data");
      return 0;
   }

   // Copy H4 prices
   if(CopyClose(_Symbol, PERIOD_H4, 0, 3, close_h4) < 3 ||
      CopyOpen(_Symbol, PERIOD_H4, 0, 2, open_h4) < 2 ||
      CopyHigh(_Symbol, PERIOD_H4, 0, 2, high_h4) < 2 ||
      CopyLow(_Symbol, PERIOD_H4, 0, 2, low_h4) < 2)
   {
      Print("Failed to copy H4 price data");
      return 0;
   }

   // === Compute 14 features (exact order from features_used.json) ===

   // 1. close_ema50_h1: (close - ema50) / ema50, bar index 1 (shift 1)
   double f01_close_ema50_h1 = (close_h1[1] - ema50_h1[1]) / (ema50_h1[1] + 1e-10);

   // 2. ema50_ema200_h1: (ema50 - ema200) / close, bar index 2 (extra EMA lag)
   double f02_ema50_ema200_h1 = (ema50_h1[2] - ema200_h1[2]) / (close_h1[2] + 1e-10);

   // 3. rsi_h1: standard RSI, bar index 1
   double f03_rsi_h1 = rsi_h1[1];

   // 4. atr_ratio_h1: atr / close, bar index 1
   double f04_atr_ratio_h1 = atr_h1[1] / (close_h1[1] + 1e-10);

   // 5. squeeze_ratio_h1: BB_width / KC_width, bar index 1
   double f05_squeeze_h1 = ComputeSqueezeRatio(PERIOD_H1, 1);

   // 6. choppiness_h1: choppiness index, bar index 1
   double f06_chop_h1 = ComputeChoppiness(PERIOD_H1, 1);
   chop_h1_out = f06_chop_h1; // Output for regime filter

   // 7. body_ratio_h1: (close - open) / (high - low), bar index 1
   double range_h1 = high_h1[1] - low_h1[1];
   double f07_body_ratio_h1 = (range_h1 > 1e-8)
      ? MathMax(-1.0, MathMin(1.0, (close_h1[1] - open_h1[1]) / range_h1))
      : 0.0;  // Doji candle (no range)

   // 8. ema50_ema200_h4: (ema50 - ema200) / close, bar index 2 (H4 shift + EMA lag)
   double f08_ema50_ema200_h4 = (ema50_h4[2] - ema200_h4[2]) / (close_h4[2] + 1e-10);

   // 9. atr_ratio_h4: atr / close, bar index 1
   double f09_atr_ratio_h4 = atr_h4[1] / (close_h4[1] + 1e-10);

   // 10. adx_h4: simplified ADX (NOT standard iADX), bar index 1
   double f10_adx_h4 = ComputeSimplifiedADX(PERIOD_H4, 1);

   // 11. squeeze_ratio_h4: BB_width / KC_width, bar index 1
   double f11_squeeze_h4 = ComputeSqueezeRatio(PERIOD_H4, 1);

   // 12. choppiness_h4: choppiness index, bar index 1
   double f12_chop_h4 = ComputeChoppiness(PERIOD_H4, 1);
   chop_h4_out = f12_chop_h4; // Output for regime filter

   // 13. body_ratio_h4: (close - open) / (high - low), bar index 1
   double range_h4 = high_h4[1] - low_h4[1];
   double f13_body_ratio_h4 = (range_h4 > 1e-8)
      ? MathMax(-1.0, MathMin(1.0, (close_h4[1] - open_h4[1]) / range_h4))
      : 0.0;  // Doji candle (no range)

   // 14. range_pct_h4: (high - low) / close, bar index 1
   double f14_range_pct_h4 = range_h4 / (close_h4[1] + 1e-10);

   // Build JSON request with exactly 14 features
   string features = StringFormat(
      "[%.8f,%.8f,%.4f,%.8f,%.6f,%.4f,%.6f,%.8f,%.8f,%.8f,%.6f,%.4f,%.6f,%.8f]",
      f01_close_ema50_h1,
      f02_ema50_ema200_h1,
      f03_rsi_h1,
      f04_atr_ratio_h1,
      f05_squeeze_h1,
      f06_chop_h1,
      f07_body_ratio_h1,
      f08_ema50_ema200_h4,
      f09_atr_ratio_h4,
      f10_adx_h4,
      f11_squeeze_h4,
      f12_chop_h4,
      f13_body_ratio_h4,
      f14_range_pct_h4
   );

   string json_request = "{\"features\":" + features + "}";

   Print("Features: ", features);

   // Make HTTP request
   uchar post_data[];
   uchar result_data[];
   string result_headers;

   StringToCharArray(json_request, post_data, 0, StringLen(json_request));

   int timeout = 5000;
   ResetLastError();

   string headers = "Content-Type: application/json\r\n";
   string activeURL = GetActiveAPIURL();
   int res = WebRequest(
      "POST",
      activeURL,
      headers,
      timeout,
      post_data,
      result_data,
      result_headers
   );

   if(res == -1)
   {
      int last_error = GetLastError();
      string error_desc = "";

      switch(last_error)
      {
         case 4016: error_desc = "URL not allowed in WebRequest list"; break;
         case 4018: error_desc = "WebRequest is disabled"; break;
         case 4060: error_desc = "Network error"; break;
         case 4061: error_desc = "Timeout"; break;
         case 4062: error_desc = "Invalid URL"; break;
         default: error_desc = "Unknown error"; break;
      }

      Print("WebRequest error ", last_error, ": ", error_desc);
      if(last_error == 4016 || last_error == 4018)
      {
         Print("Add URL to: Tools -> Options -> Expert Advisors");
         Print("Add: http://127.0.0.1, http://localhost");
      }
      return 0;
   }

   string response = CharArrayToString(result_data);

   // Extract prediction and confidence
   double sell_prob = ExtractValue(response, "sell");
   double range_prob = ExtractValue(response, "range");
   double buy_prob = ExtractValue(response, "buy");
   confidence = ExtractValue(response, "confidence");

   string prediction = ExtractStringValue(response, "prediction");

   // Round confidence to avoid floating point precision issues
   confidence = NormalizeDouble(confidence, 4);

   // Direction-aware signal logic
   if(TradeDirection == TRADE_BOTH)
   {
      // BOTH MODE: Check buy first (priority), then sell — single API call
      Print("ML: ", prediction, " | Buy=", DoubleToString(buy_prob*100, 1),
            "% Range=", DoubleToString(range_prob*100, 1),
            "% Sell=", DoubleToString(sell_prob*100, 1), "%");

      // Check BUY signal (threshold 0.40)
      double buy_spread = NormalizeDouble(buy_prob - MathMax(sell_prob, range_prob), 4);
      if(buy_prob >= ML_Confidence_Threshold && buy_spread >= MinConfidenceSpread)
      {
         Print(">>> BUY signal: prob=", DoubleToString(buy_prob*100, 1),
               "% spread=", DoubleToString(buy_spread*100, 1), "%");
         return 1;
      }

      // Check SELL signal (threshold 0.56)
      double sell_spread = NormalizeDouble(sell_prob - MathMax(buy_prob, range_prob), 4);
      if(sell_prob >= ML_Sell_Threshold)
      {
         Print(">>> SELL signal: prob=", DoubleToString(sell_prob*100, 1),
               "% spread=", DoubleToString(sell_spread*100, 1), "%");
         return -1;
      }

      // No signal
      if(buy_prob >= ML_Confidence_Threshold)
         Print("Buy spread too low: ", DoubleToString(buy_spread*100, 1), "%");
      else if(sell_prob >= ML_Confidence_Threshold)
         Print("Sell below sell threshold: ", DoubleToString(sell_prob*100, 1),
               "% < ", DoubleToString(ML_Sell_Threshold*100, 0), "%");
      else
         Print("No signal: max prob=", DoubleToString(MathMax(buy_prob, sell_prob)*100, 1), "%");

      return 0;
   }
   else if(TradeDirection == TRADE_BUY_ONLY)
   {
      // BUY MODE: buy_prob must dominate max(sell, range)
      double max_other = MathMax(sell_prob, range_prob);
      double conf_spread = NormalizeDouble(buy_prob - max_other, 4);

      Print("ML: ", prediction, " | Buy=", DoubleToString(buy_prob*100, 1),
            "% Range=", DoubleToString(range_prob*100, 1),
            "% Sell=", DoubleToString(sell_prob*100, 1),
            "% | Spread=", DoubleToString(conf_spread*100, 1), "%");

      if(buy_prob >= ML_Confidence_Threshold)
      {
         if(conf_spread >= MinConfidenceSpread)
            return 1;  // Valid buy signal

         Print("Confidence spread too low: ", DoubleToString(conf_spread*100, 1),
               "% < ", DoubleToString(MinConfidenceSpread*100, 1), "% required");
         return 0;
      }

      if(buy_prob < ML_Confidence_Threshold)
         Print("Buy prob below threshold: ", DoubleToString(buy_prob*100, 1),
               "% < ", DoubleToString(ML_Confidence_Threshold*100, 1), "%");

      return 0;
   }
   else
   {
      // SELL MODE: sell_prob must dominate max(buy, range)
      double max_other = MathMax(buy_prob, range_prob);
      double conf_spread = NormalizeDouble(sell_prob - max_other, 4);

      Print("ML: ", prediction, " | Sell=", DoubleToString(sell_prob*100, 1),
            "% Range=", DoubleToString(range_prob*100, 1),
            "% Buy=", DoubleToString(buy_prob*100, 1),
            "% | Spread=", DoubleToString(conf_spread*100, 1), "%");

      if(sell_prob >= ML_Confidence_Threshold)
      {
         if(conf_spread >= MinConfidenceSpread)
            return -1;  // Valid sell signal

         Print("Confidence spread too low: ", DoubleToString(conf_spread*100, 1),
               "% < ", DoubleToString(MinConfidenceSpread*100, 1), "% required");
         return 0;
      }

      if(sell_prob < ML_Confidence_Threshold)
         Print("Sell prob below threshold: ", DoubleToString(sell_prob*100, 1),
               "% < ", DoubleToString(ML_Confidence_Threshold*100, 1), "%");

      return 0;
   }
}

//+------------------------------------------------------------------+
//| Get trade signal with regime filter and ML/Technical confluence  |
//+------------------------------------------------------------------+
int GetTradeSignal()
{
   // Time filter
   if(UseTimeFilter && !IsWithinTradingHours())
      return 0;

   // Get technical signal (always needed)
   int tech_signal = GetTechnicalSignal();

   // ─── TECHNICAL ONLY MODE ───
   if(TechnicalOnly)
   {
      if(TradeDirection == TRADE_BOTH)
      {
         // BOTH: accept any technical signal
         if(tech_signal != 0)
         {
            Print(">>> Technical ", (tech_signal == 1 ? "BUY" : "SELL"),
                  " signal: MA(", MA_Fast, ",", MA_Slow, ") cross + RSI");
            return tech_signal;
         }
         return 0;
      }

      int target_signal = (TradeDirection == TRADE_BUY_ONLY) ? 1 : -1;
      if(tech_signal != target_signal)
         return 0;

      Print(">>> Technical ", (target_signal == 1 ? "BUY" : "SELL"),
            " signal: MA(", MA_Fast, ",", MA_Slow, ") cross + RSI");
      return target_signal;
   }

   // ─── ML MODES (ML-only or ML+Technical) ───
   double ml_confidence = 0;
   double chop_h1 = 50.0, chop_h4 = 50.0;
   int ml_signal = GetMLPrediction(ml_confidence, chop_h1, chop_h4);

   // Combine logic
   if(!CombineWithTechnical)
   {
      // ML-only mode: pass through whatever ML returned (+1, -1, or 0)
      if(ml_signal == 0)
         return 0;
   }
   else
   {
      // Combined mode: ML signal must exist, technical can strengthen or veto
      if(ml_signal == 0)
         return 0;

      string dir_label = (ml_signal == 1) ? "BUY" : "SELL";

      if(tech_signal == ml_signal)
      {
         Print("+ ML + Technical AGREE: ", dir_label);
      }
      else if(tech_signal == 0)
      {
         Print("+ ML ", dir_label, " (Technical no signal)");
      }
      else
      {
         Print("x ML ", dir_label, " but Technical disagrees");
         return 0;
      }
   }

   // Regime filter: only trade when both timeframes are trending (low choppiness)
   if(UseRegimeFilter && ml_signal != 0)
   {
      double median_h1 = ComputeRollingMedian(PERIOD_H1, 1, RegimeLookback);
      double median_h4 = ComputeRollingMedian(PERIOD_H4, 1, MathMin(RegimeLookback / 4, 125));

      bool h1_trending = (chop_h1 < median_h1);
      bool h4_trending = (chop_h4 < median_h4);

      if(!h1_trending || !h4_trending)
      {
         Print("Regime filter blocked: chop_h1=", DoubleToString(chop_h1, 1),
               " (med=", DoubleToString(median_h1, 1), ") | chop_h4=",
               DoubleToString(chop_h4, 1), " (med=", DoubleToString(median_h4, 1), ")");
         return 0;
      }

      Print("Regime OK: chop_h1=", DoubleToString(chop_h1, 1),
            " < ", DoubleToString(median_h1, 1),
            " | chop_h4=", DoubleToString(chop_h4, 1),
            " < ", DoubleToString(median_h4, 1));
   }

   return ml_signal;
}

//+------------------------------------------------------------------+
//| Open SELL trade with fixed TP/SL                                 |
//+------------------------------------------------------------------+
void OpenSellTrade()
{
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double pipSize = point * 10;  // 1 pip = 10 points for 5-digit broker

   double slDistance = SL_Pips * pipSize;
   double sl = NormalizeDouble(price + slDistance, _Digits);

   // Calculate total lot size based on risk
   double totalLotSize = CalculateLotSize(slDistance);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(UseTwinTrades)
   {
      // ═══ TWIN TRADE SYSTEM ═══
      // Split lot size in half for each trade
      double halfLot = MathFloor((totalLotSize / 2.0) / lotStep) * lotStep;
      if(halfLot < minLot) halfLot = minLot;

      // Trade A: The Banker (quick TP, no trailing)
      double tpA = NormalizeDouble(price - TwinA_TP_Pips * pipSize, _Digits);
      string commentA = "TwinA TP=" + DoubleToString(TwinA_TP_Pips, 0);

      if(trade.Sell(halfLot, _Symbol, price, sl, tpA, commentA))
      {
         Print("TWIN-A (Banker) opened: Lot=", DoubleToString(halfLot, 2),
               " TP=", DoubleToString(TwinA_TP_Pips, 0), " pips");
      }
      else
      {
         Print("Failed to open TWIN-A: ", trade.ResultRetcodeDescription());
      }

      // Trade B: The Runner (full TP, with BE + trailing)
      double tpB = NormalizeDouble(price - TwinB_TP_Pips * pipSize, _Digits);
      string commentB = "TwinB TP=" + DoubleToString(TwinB_TP_Pips, 0);

      if(trade.Sell(halfLot, _Symbol, price, sl, tpB, commentB))
      {
         Print("TWIN-B (Runner) opened: Lot=", DoubleToString(halfLot, 2),
               " TP=", DoubleToString(TwinB_TP_Pips, 0), " pips (BE@",
               DoubleToString(TwinB_BE_Trigger, 0), ", Trail@",
               DoubleToString(TwinB_Trail_Pips, 0), ")");
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open TWIN-B: ", trade.ResultRetcodeDescription());
      }
   }
   else
   {
      // ═══ SINGLE TRADE (original behavior) ═══
      double tpDistance = TP_Pips * pipSize;
      double tp = NormalizeDouble(price - tpDistance, _Digits);

      if(totalLotSize < minLot)
      {
         Print("Lot size too small: ", DoubleToString(totalLotSize, 4), ", using minimum: ", DoubleToString(minLot, 2));
         totalLotSize = minLot;
      }

      string comment = "ML Sell TP=" + DoubleToString(TP_Pips, 0) + "/SL=" + DoubleToString(SL_Pips, 0);

      if(trade.Sell(totalLotSize, _Symbol, price, sl, tp, comment))
      {
         Print("SELL opened: Lot=", DoubleToString(totalLotSize, 2),
               " Price=", DoubleToString(price, _Digits),
               " SL=", DoubleToString(sl, _Digits),
               " TP=", DoubleToString(tp, _Digits));
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open SELL: ", trade.ResultRetcodeDescription());
      }
   }
}

//+------------------------------------------------------------------+
//| Open BUY trade with fixed TP/SL                                  |
//+------------------------------------------------------------------+
void OpenBuyTrade()
{
   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double pipSize = point * 10;  // 1 pip = 10 points for 5-digit broker

   double slDistance = SL_Pips * pipSize;
   double sl = NormalizeDouble(price - slDistance, _Digits);

   // Calculate total lot size based on risk
   double totalLotSize = CalculateLotSize(slDistance);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(UseTwinTrades)
   {
      // Twin Trade System
      double halfLot = MathFloor((totalLotSize / 2.0) / lotStep) * lotStep;
      if(halfLot < minLot) halfLot = minLot;

      // Trade A: The Banker (quick TP, no trailing)
      double tpA = NormalizeDouble(price + TwinA_TP_Pips * pipSize, _Digits);
      string commentA = "TwinA TP=" + DoubleToString(TwinA_TP_Pips, 0);

      if(trade.Buy(halfLot, _Symbol, price, sl, tpA, commentA))
      {
         Print("TWIN-A (Banker) BUY opened: Lot=", DoubleToString(halfLot, 2),
               " TP=", DoubleToString(TwinA_TP_Pips, 0), " pips");
      }
      else
      {
         Print("Failed to open TWIN-A BUY: ", trade.ResultRetcodeDescription());
      }

      // Trade B: The Runner (full TP, with BE + trailing)
      double tpB = NormalizeDouble(price + TwinB_TP_Pips * pipSize, _Digits);
      string commentB = "TwinB TP=" + DoubleToString(TwinB_TP_Pips, 0);

      if(trade.Buy(halfLot, _Symbol, price, sl, tpB, commentB))
      {
         Print("TWIN-B (Runner) BUY opened: Lot=", DoubleToString(halfLot, 2),
               " TP=", DoubleToString(TwinB_TP_Pips, 0), " pips (BE@",
               DoubleToString(TwinB_BE_Trigger, 0), ", Trail@",
               DoubleToString(TwinB_Trail_Pips, 0), ")");
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open TWIN-B BUY: ", trade.ResultRetcodeDescription());
      }
   }
   else
   {
      // Single trade mode
      double tpDistance = TP_Pips * pipSize;
      double tp = NormalizeDouble(price + tpDistance, _Digits);

      if(totalLotSize < minLot)
      {
         Print("Lot size too small: ", DoubleToString(totalLotSize, 4), ", using minimum: ", DoubleToString(minLot, 2));
         totalLotSize = minLot;
      }

      string comment = "ML Buy TP=" + DoubleToString(TP_Pips, 0) + "/SL=" + DoubleToString(SL_Pips, 0);

      if(trade.Buy(totalLotSize, _Symbol, price, sl, tp, comment))
      {
         Print("BUY opened: Lot=", DoubleToString(totalLotSize, 2),
               " Price=", DoubleToString(price, _Digits),
               " SL=", DoubleToString(sl, _Digits),
               " TP=", DoubleToString(tp, _Digits));
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open BUY: ", trade.ResultRetcodeDescription());
      }
   }
}

//+------------------------------------------------------------------+
//| Check for closed trades and update circuit breaker               |
//+------------------------------------------------------------------+
void CheckClosedTrades()
{
   int currentPositions = CountOpenPositions();

   // A position was closed
   if(currentPositions < cb_last_position_count)
   {
      // Check recent deal history
      datetime from = TimeCurrent() - 3600; // Last hour
      HistorySelect(from, TimeCurrent());

      int totalDeals = HistoryDealsTotal();
      for(int i = totalDeals - 1; i >= 0; i--)
      {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket == 0) continue;

         long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
         string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);

         if(magic != 123456 || symbol != _Symbol || entry != DEAL_ENTRY_OUT)
            continue;

         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         double netProfit = profit + commission + swap;

         // Update circuit breaker state
         // Convert dollar P&L to pips: profit / (lots * pip_value_per_lot)
         double lots = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         double pip_point = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10; // 1 pip in price
         double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
         double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
         double pnl_pips = (tick_size > 0 && tick_value > 0 && lots > 0)
            ? netProfit / (lots * tick_value / tick_size * pip_point)
            : 0.0;

         cb_running_pnl += pnl_pips;
         cb_peak_pnl = MathMax(cb_peak_pnl, cb_running_pnl);
         double dd_pips = cb_peak_pnl - cb_running_pnl;

         if(netProfit <= 0)
         {
            cb_consecutive_losses++;
            Print("CB: Loss #", cb_consecutive_losses, " ",
                  DoubleToString(pnl_pips, 1), " pips | Cumulative: ",
                  DoubleToString(cb_running_pnl, 1), " pips | DD: ",
                  DoubleToString(dd_pips, 1), " pips");
         }
         else
         {
            if(CB_ResetOnWin)
               cb_consecutive_losses = 0;
            Print("CB: Win +", DoubleToString(pnl_pips, 1), " pips | Streak reset | Cumulative: ",
                  DoubleToString(cb_running_pnl, 1), " pips");
         }

         // Check triggers
         if(cb_consecutive_losses >= CB_MaxConsecLosses || dd_pips >= CB_MaxDrawdownPips)
         {
            cb_paused_until_bar = cb_bar_counter + CB_CooldownBars;
            cb_triggers++;
            cb_consecutive_losses = 0;

            Print("*** CIRCUIT BREAKER TRIGGERED ***");
            Print("Pausing for ", CB_CooldownBars, " bars (until bar ", cb_paused_until_bar, ")");
            Print("Reason: ", (cb_consecutive_losses >= CB_MaxConsecLosses) ?
                  "consecutive losses" : "drawdown exceeded");
         }

         break; // Process only the most recent exit deal
      }
   }

   cb_last_position_count = currentPositions;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk percentage                      |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
   double balance = accInfo.Balance();
   double riskAmount = balance * RiskPercent / 100.0;
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickValue <= 0 || tickSize <= 0 || slDistance <= 0)
   {
      Print("Error getting symbol info for lot calculation");
      return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   }

   double lotSize = riskAmount / (slDistance / tickSize * tickValue);

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = NormalizeDouble(lotSize, 2);

   // Margin safety check
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double margin_per_lot = SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
   if(margin_per_lot > 0 && freeMargin > 0)
   {
      double marginRequired = lotSize * margin_per_lot;
      if(marginRequired > freeMargin * 0.5)
      {
         Print("Reducing lot size due to margin constraints");
         lotSize = MathMin(lotSize, freeMargin * 0.5 / margin_per_lot);
         lotSize = MathFloor(lotSize / lotStep) * lotStep;
         lotSize = NormalizeDouble(lotSize, 2);
      }
   }

   return lotSize;
}

//+------------------------------------------------------------------+
//| Manage open positions (breakeven, partial close)                 |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() != _Symbol || posInfo.Magic() != 123456)
            continue;

         string comment = posInfo.Comment();
         bool isTwinA = (StringFind(comment, "TwinA") >= 0);
         bool isTwinB = (StringFind(comment, "TwinB") >= 0);

         // TwinA positions: NO breakeven or trailing - just let them run to TP/SL
         if(isTwinA)
            continue;

         ulong ticket = posInfo.Ticket();
         double currentPrice = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);

         double openPrice = posInfo.PriceOpen();
         double sl = posInfo.StopLoss();

         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         double profitPips = 0;
         if(point > 0)
         {
            if(posInfo.PositionType() == POSITION_TYPE_BUY)
               profitPips = (currentPrice - openPrice) / (point * 10);
            else
               profitPips = (openPrice - currentPrice) / (point * 10);
         }

         // Determine which parameters to use
         double beTrigger = isTwinB ? TwinB_BE_Trigger : BreakevenTriggerPips;
         double beOffset = isTwinB ? TwinB_BE_Offset : BreakevenOffsetPips;

         // Breakeven
         if(UseBreakeven && profitPips >= beTrigger && sl != 0)
         {
            double newSL = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                           openPrice + beOffset * point * 10 :
                           openPrice - beOffset * point * 10;

            newSL = NormalizeDouble(newSL, _Digits);

            if((posInfo.PositionType() == POSITION_TYPE_BUY && newSL > sl) ||
               (posInfo.PositionType() == POSITION_TYPE_SELL && (newSL < sl || sl == 0)))
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print(isTwinB ? "TWIN-B" : "Position", " ", ticket,
                        " → breakeven +", DoubleToString(beOffset, 0), " pips locked");
            }
         }

         // Partial close (only for non-twin single trades)
         if(!isTwinB && UsePartialClose && profitPips >= BreakevenTriggerPips * 2)
         {
            double closeVolume = posInfo.Volume() * PartialClosePercent / 100.0;
            double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            closeVolume = MathMax(closeVolume, minLot);
            closeVolume = NormalizeDouble(closeVolume, 2);

            if(closeVolume >= minLot && closeVolume < posInfo.Volume())
            {
               if(trade.PositionClosePartial(ticket, closeVolume))
                  Print("Partial close: ", DoubleToString(PartialClosePercent, 0),
                        "% of position ", ticket);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stops for all positions                          |
//+------------------------------------------------------------------+
void ManageTrailingStops()
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() != _Symbol || posInfo.Magic() != 123456)
            continue;

         string comment = posInfo.Comment();
         bool isTwinA = (StringFind(comment, "TwinA") >= 0);
         bool isTwinB = (StringFind(comment, "TwinB") >= 0);

         // TwinA positions: NO trailing - just let them run to TP/SL
         if(isTwinA)
            continue;

         ulong ticket = posInfo.Ticket();
         double currentPrice = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);

         double sl = posInfo.StopLoss();
         double openPrice = posInfo.PriceOpen();

         // Determine which parameters to use
         double trailPips = isTwinB ? TwinB_Trail_Pips : TrailingStopPips;
         double stepPips = isTwinB ? TwinB_Trail_Step : TrailingStepPips;
         double trailDistance = trailPips * point * 10;
         double trailStep = stepPips * point * 10;

         // Only trail after breakeven has been activated (SL is better than entry)
         bool beActive = false;
         if(posInfo.PositionType() == POSITION_TYPE_SELL)
            beActive = (sl < openPrice);  // SL below entry = profit locked
         else
            beActive = (sl > openPrice);  // SL above entry = profit locked

         if(!beActive)
            continue;  // Don't trail until breakeven is activated

         if(posInfo.PositionType() == POSITION_TYPE_BUY && sl > 0)
         {
            double newSL = NormalizeDouble(currentPrice - trailDistance, _Digits);
            if(newSL > sl + trailStep)
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print(isTwinB ? "TWIN-B" : "Position", " ", ticket,
                        " trailing → SL=", DoubleToString(newSL, _Digits));
            }
         }
         else if(posInfo.PositionType() == POSITION_TYPE_SELL && sl > 0)
         {
            double newSL = NormalizeDouble(currentPrice + trailDistance, _Digits);
            if(newSL < sl - trailStep)
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print(isTwinB ? "TWIN-B" : "Position", " ", ticket,
                        " trailing → SL=", DoubleToString(newSL, _Digits));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Extract numeric value from JSON string                           |
//+------------------------------------------------------------------+
double ExtractValue(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start == -1)
   {
      search = "'" + key + "':";
      start = StringFind(json, search);
      if(start == -1) return 0.0;
   }

   start += StringLen(search);
   int end = start;

   while(end < StringLen(json))
   {
      string ch = StringSubstr(json, end, 1);
      if(ch == "," || ch == "}")
         break;
      end++;
   }

   string value_str = StringSubstr(json, start, end - start);
   value_str = StringTrimCustom(value_str);
   return StringToDoubleCustom(value_str);
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractStringValue(const string json, const string key)
{
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start == -1)
   {
      search = "'" + key + "':'";
      start = StringFind(json, search);
      if(start == -1) return "";
   }

   start += StringLen(search);
   int end = StringFind(json, "\"", start);

   if(end == -1)
      end = StringFind(json, "'", start);

   if(end == -1 || end <= start)
      return "";

   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Custom string trim                                               |
//+------------------------------------------------------------------+
string StringTrimCustom(const string str)
{
   string result = str;
   while(StringLen(result) > 0 && StringSubstr(result, 0, 1) == " ")
      result = StringSubstr(result, 1);
   while(StringLen(result) > 0 && StringSubstr(result, StringLen(result)-1, 1) == " ")
      result = StringSubstr(result, 0, StringLen(result)-1);
   return result;
}

//+------------------------------------------------------------------+
//| Custom string to double                                          |
//+------------------------------------------------------------------+
double StringToDoubleCustom(const string value)
{
   string clean_value = "";
   for(int i = 0; i < StringLen(value); i++)
   {
      string ch = StringSubstr(value, i, 1);
      if((ch >= "0" && ch <= "9") || ch == "." || ch == "-")
         clean_value += ch;
   }

   if(clean_value == "" || clean_value == "-")
      return 0.0;

   return ::StringToDouble(clean_value);
}

//+------------------------------------------------------------------+
//| Check daily limits                                               |
//+------------------------------------------------------------------+
bool CheckDailyLimits()
{
   double currentBalance = accInfo.Balance();
   dailyProfitLoss = currentBalance - dailyStartBalance;
   double dailyPL_Percent = 0;

   if(dailyStartBalance > 0)
      dailyPL_Percent = (dailyProfitLoss / dailyStartBalance) * 100.0;

   if(dailyPL_Percent <= -MaxDailyLoss)
   {
      Print("Daily loss limit reached: ", DoubleToString(dailyPL_Percent, 1), "%");
      return false;
   }

   if(dailyPL_Percent >= MaxDailyProfit)
   {
      Print("Daily profit target reached: ", DoubleToString(dailyPL_Percent, 1), "%");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Daily reset                                                      |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   MqlDateTime currentTime;
   TimeToStruct(TimeCurrent(), currentTime);

   static int lastDay = -1;

   if(lastDay == -1)
      lastDay = currentTime.day;

   if(currentTime.day != lastDay)
   {
      dailyStartBalance = accInfo.Balance();
      totalTradesToday = 0;
      dailyProfitLoss = 0.0;
      lastDay = currentTime.day;

      Print("=== New Trading Day ===");
      Print("Balance: $", DoubleToString(dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Check trading hours                                              |
//+------------------------------------------------------------------+
bool IsWithinTradingHours()
{
   MqlDateTime currentTime;
   TimeToStruct(TimeCurrent(), currentTime);
   return (currentTime.hour >= StartHour && currentTime.hour < EndHour);
}

//+------------------------------------------------------------------+
//| Count open positions                                             |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == 123456)
            count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Count at-risk positions (SL worse than entry)                    |
//| Protected positions (SL at breakeven or better) don't count      |
//+------------------------------------------------------------------+
int CountAtRiskPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() != _Symbol || posInfo.Magic() != 123456)
            continue;

         double sl = posInfo.StopLoss();
         double entry = posInfo.PriceOpen();

         // No SL = at risk
         if(sl == 0)
         {
            count++;
            continue;
         }

         // For SELL: SL > entry means at risk (loss if hit)
         // For BUY: SL < entry means at risk (loss if hit)
         if(posInfo.PositionType() == POSITION_TYPE_SELL)
         {
            if(sl > entry)  // SL above entry = would be a loss
               count++;
            // else: SL <= entry = protected (breakeven or profit locked)
         }
         else // BUY
         {
            if(sl < entry)  // SL below entry = would be a loss
               count++;
            // else: SL >= entry = protected
         }
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason)
{
   Print("Closing all positions: ", reason);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == 123456)
         {
            ulong ticket = posInfo.Ticket();
            if(trade.PositionClose(ticket))
               Print("Closed position: ", ticket);
         }
      }
   }
}
//+------------------------------------------------------------------+
