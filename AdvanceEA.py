//+------------------------------------------------------------------+
//|                                     AdvancedProfitEA_ML.mq5      |
//|                   Advanced Trading System with ML Integration     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property version   "2.20"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

// Input Parameters
input group "=== ML API Settings ==="
input bool   UseMLPredictions = true;        // Use ML predictions
input string API_URL = "http://127.0.0.1:8000/predict"; // API endpoint
input double ML_Confidence_Threshold = 0.40; // Minimum confidence (40%)
input bool   CombineWithTechnical = false;    // Combine ML with technical signals

input group "=== Risk Management ==="
input double RiskPercent = 1.0;              // Risk per trade (% of balance)
input double MaxDailyLoss = 3.0;             // Max daily loss (% of balance)
input double MaxDailyProfit = 5.0;           // Daily profit target (%)
input int    MaxSimultaneousTrades = 3;      // Maximum open positions
input bool   UseTrailingStop = true;         // Enable trailing stop
input double TrailingStopPips = 30;          // Trailing stop distance (pips)
input double TrailingStepPips = 10;          // Trailing step (pips)

input group "=== Strategy Settings ==="
input int    MA_Fast = 10;                   // Fast MA Period
input int    MA_Slow = 30;                   // Slow MA Period
input int    RSI_Period = 14;                // RSI Period
input int    RSI_Overbought = 70;            // RSI Overbought Level
input int    RSI_Oversold = 30;              // RSI Oversold Level
input int    ATR_Period = 14;                // ATR Period for volatility
input double ATR_Multiplier = 2.0;           // ATR multiplier for SL/TP

input group "=== Trade Filters ==="
input bool   UseTrendFilter = true;          // Only trade with trend
input int    TrendMA_Period = 50;            // Trend MA Period
input bool   UseTimeFilter = true;           // Enable time filter
input int    StartHour = 8;                  // Trading start hour
input int    EndHour = 20;                   // Trading end hour
input bool   UseVolatilityFilter = true;     // Filter low volatility
input double MinATR = 0.0001;                // Minimum ATR value

input group "=== Position Management ==="
input bool   UsePartialClose = true;         // Partial close at targets
input double PartialClosePercent = 50.0;     // % to close at first target
input bool   UseBreakeven = true;            // Move SL to breakeven
input double BreakevenTriggerPips = 20;      // Pips profit to trigger BE
input double BreakevenOffsetPips = 5;        // BE offset (pips)

// Global Variables
CTrade trade;
CPositionInfo posInfo;
CAccountInfo accInfo;

int handleMA_Fast, handleMA_Slow, handleRSI, handleATR, handleTrendMA;

// ML indicator handles (H1 + H4)
int handleEMA50_H1, handleEMA200_H1, handleADX_H1, handleRSI_H1, handleATR_H1;
int handleEMA50_H4, handleEMA200_H4, handleADX_H4, handleRSI_H4, handleATR_H4;

double dailyStartBalance;
datetime lastBarTime;
int totalTradesToday = 0;
double dailyProfitLoss = 0.0;

// Forward declarations for helper functions
string StringTrimCustom(const string str);
double StringToDoubleCustom(const string value);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("Advanced Profit EA v2.20 with ML (H1 + H4 features)");
   Print("========================================");
   
   // Initialize indicators (for technical strategy on current chart timeframe)
   handleMA_Fast = iMA(_Symbol, _Period, MA_Fast, 0, MODE_SMA, PRICE_CLOSE);
   handleMA_Slow = iMA(_Symbol, _Period, MA_Slow, 0, MODE_SMA, PRICE_CLOSE);
   handleRSI = iRSI(_Symbol, _Period, RSI_Period, PRICE_CLOSE);
   handleATR = iATR(_Symbol, _Period, ATR_Period);
   handleTrendMA = iMA(_Symbol, _Period, TrendMA_Period, 0, MODE_SMA, PRICE_CLOSE);
   
   // ML-specific indicators (H1 and H4 timeframes)
   handleEMA50_H1 = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA200_H1 = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
   handleADX_H1 = iADX(_Symbol, PERIOD_H1, 14);
   handleRSI_H1 = iRSI(_Symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE);
   handleATR_H1 = iATR(_Symbol, PERIOD_H1, ATR_Period);

   handleEMA50_H4 = iMA(_Symbol, PERIOD_H4, 50, 0, MODE_EMA, PRICE_CLOSE);
   handleEMA200_H4 = iMA(_Symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);
   handleADX_H4 = iADX(_Symbol, PERIOD_H4, 14);
   handleRSI_H4 = iRSI(_Symbol, PERIOD_H4, RSI_Period, PRICE_CLOSE);
   handleATR_H4 = iATR(_Symbol, PERIOD_H4, ATR_Period);
   
   if(handleMA_Fast == INVALID_HANDLE || handleMA_Slow == INVALID_HANDLE ||
      handleRSI == INVALID_HANDLE || handleATR == INVALID_HANDLE || 
      handleTrendMA == INVALID_HANDLE ||
      handleEMA50_H1 == INVALID_HANDLE || handleEMA200_H1 == INVALID_HANDLE ||
      handleADX_H1 == INVALID_HANDLE || handleRSI_H1 == INVALID_HANDLE || handleATR_H1 == INVALID_HANDLE ||
      handleEMA50_H4 == INVALID_HANDLE || handleEMA200_H4 == INVALID_HANDLE ||
      handleADX_H4 == INVALID_HANDLE || handleRSI_H4 == INVALID_HANDLE || handleATR_H4 == INVALID_HANDLE)
   {
      Print("Error initializing indicators!");
      return(INIT_FAILED);
   }
   
   dailyStartBalance = accInfo.Balance();
   lastBarTime = 0;
   
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   Print("ML Integration: ", UseMLPredictions ? "ENABLED" : "DISABLED");
   Print("API URL: ", API_URL);
   Print("ML Confidence Threshold: ", ML_Confidence_Threshold);
   Print("Risk per trade: ", RiskPercent, "%");
   Print("Starting balance: $", dailyStartBalance);
   
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
   IndicatorRelease(handleADX_H1);
   IndicatorRelease(handleRSI_H1);
   IndicatorRelease(handleATR_H1);

   IndicatorRelease(handleEMA50_H4);
   IndicatorRelease(handleEMA200_H4);
   IndicatorRelease(handleADX_H4);
   IndicatorRelease(handleRSI_H4);
   IndicatorRelease(handleATR_H4);
   
   Print("EA Stopped. Total trades today: ", totalTradesToday);
   Print("Daily P&L: $", dailyProfitLoss);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   bool isNewBar = (currentBarTime != lastBarTime);
   
   if(isNewBar)
   {
      lastBarTime = currentBarTime;
      
      CheckDailyReset();
      
      if(!CheckDailyLimits())
      {
         CloseAllPositions("Daily limit reached");
         return;
      }
      
      if(UseTrailingStop)
         ManageTrailingStops();
      
      ManagePositions();
      
      if(CountOpenPositions() < MaxSimultaneousTrades)
      {
         int signal = GetTradeSignal();
         
         if(signal == 1)
            OpenTrade(ORDER_TYPE_BUY);
         else if(signal == -1)
            OpenTrade(ORDER_TYPE_SELL);
      }
   }
}

//+------------------------------------------------------------------+
//| Get ML prediction from Python API (H1 + H4 features)             |
//+------------------------------------------------------------------+
int GetMLPrediction(double &confidence)
{
   if(!UseMLPredictions)
      return 0;
   
   // H1 indicator buffers
   double ema50_h1[2], ema200_h1[2], rsi_h1[2], atr_h1[2], adx_h1[2];
   double close_h1[3], open_h1[2], high_h1[2], low_h1[2];

   // H4 indicator buffers
   double ema50_h4[2], ema200_h4[2], rsi_h4[2], atr_h4[2], adx_h4[2];
   double close_h4[3], open_h4[2], high_h4[2], low_h4[2];
   
   ArraySetAsSeries(ema50_h1, true); ArraySetAsSeries(ema200_h1, true);
   ArraySetAsSeries(rsi_h1, true);   ArraySetAsSeries(atr_h1, true);
   ArraySetAsSeries(adx_h1, true);
   ArraySetAsSeries(close_h1, true); ArraySetAsSeries(open_h1, true);
   ArraySetAsSeries(high_h1, true);  ArraySetAsSeries(low_h1, true);

   ArraySetAsSeries(ema50_h4, true); ArraySetAsSeries(ema200_h4, true);
   ArraySetAsSeries(rsi_h4, true);   ArraySetAsSeries(atr_h4, true);
   ArraySetAsSeries(adx_h4, true);
   ArraySetAsSeries(close_h4, true); ArraySetAsSeries(open_h4, true);
   ArraySetAsSeries(high_h4, true);  ArraySetAsSeries(low_h4, true);
   
   // Copy H1 indicators
   if(CopyBuffer(handleEMA50_H1, 0, 0, 2, ema50_h1) < 2 ||
      CopyBuffer(handleEMA200_H1, 0, 0, 2, ema200_h1) < 2 ||
      CopyBuffer(handleRSI_H1, 0, 0, 2, rsi_h1) < 2 ||
      CopyBuffer(handleATR_H1, 0, 0, 2, atr_h1) < 2 ||
      CopyBuffer(handleADX_H1, 0, 0, 2, adx_h1) < 2)
   {
      Print("Failed to copy H1 indicator data for ML");
      return 0;
   }
   // Copy H1 prices
   if(CopyClose(_Symbol, PERIOD_H1, 0, 3, close_h1) < 3 ||
      CopyOpen(_Symbol, PERIOD_H1, 0, 2, open_h1) < 2 ||
      CopyHigh(_Symbol, PERIOD_H1, 0, 2, high_h1) < 2 ||
      CopyLow(_Symbol, PERIOD_H1, 0, 2, low_h1) < 2)
   {
      Print("Failed to copy H1 price data for ML");
      return 0;
   }

   // Copy H4 indicators
   if(CopyBuffer(handleEMA50_H4, 0, 0, 2, ema50_h4) < 2 ||
      CopyBuffer(handleEMA200_H4, 0, 0, 2, ema200_h4) < 2 ||
      CopyBuffer(handleRSI_H4, 0, 0, 2, rsi_h4) < 2 ||
      CopyBuffer(handleATR_H4, 0, 0, 2, atr_h4) < 2 ||
      CopyBuffer(handleADX_H4, 0, 0, 2, adx_h4) < 2)
   {
      Print("Failed to copy H4 indicator data for ML");
      return 0;
   }
   // Copy H4 prices
   if(CopyClose(_Symbol, PERIOD_H4, 0, 3, close_h4) < 3 ||
      CopyOpen(_Symbol, PERIOD_H4, 0, 2, open_h4) < 2 ||
      CopyHigh(_Symbol, PERIOD_H4, 0, 2, high_h4) < 2 ||
      CopyLow(_Symbol, PERIOD_H4, 0, 2, low_h4) < 2)
   {
      Print("Failed to copy H4 price data for ML");
      return 0;
   }

   // Compute H1 features
   double close_ema50_h1 = ema50_h1[0];
   double ema50_ema200_h1 = ema50_h1[0] - ema200_h1[0];
   double rsi_val_h1 = rsi_h1[0];
   double rsi_slope_h1 = rsi_h1[0] - rsi_h1[1];
   double atr_ratio_h1 = atr_h1[0] / close_h1[0];
   double adx_val_h1 = adx_h1[0];
   double body_h1 = close_h1[0] - open_h1[0];
   double range_h1 = high_h1[0] - low_h1[0];
   
   // Hour and session from server time
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   double hour = now.hour;
   double session = 0;
   if(now.hour >= 13 && now.hour <= 21)
      session = 2; // US
   else if(now.hour >= 7 && now.hour <= 15)
      session = 1; // Europe
   else
      session = 0; // Asia/other

   double prev_return_h1 = 0.0;
   if(close_h1[1] != 0.0)
      prev_return_h1 = (close_h1[0] - close_h1[1]) / close_h1[1];

   // Compute H4 features
   double close_ema50_h4 = ema50_h4[0];
   double ema50_ema200_h4 = ema50_h4[0] - ema200_h4[0];
   double rsi_val_h4 = rsi_h4[0];
   double rsi_slope_h4 = rsi_h4[0] - rsi_h4[1];
   double atr_ratio_h4 = atr_h4[0] / close_h4[0];
   double adx_val_h4 = adx_h4[0];
   double body_h4 = close_h4[0] - open_h4[0];
   double range_h4 = high_h4[0] - low_h4[0];

   double prev_return_h4 = 0.0;
   if(close_h4[1] != 0.0)
      prev_return_h4 = (close_h4[0] - close_h4[1]) / close_h4[1];

   // Build JSON request with 20 features in the exact order expected by Python
   string features = StringFormat(
      "[%.6f,%.6f,%.4f,%.6f,%.8f,%.4f,%.8f,%.8f,%.0f,%.0f,%.8f,%.6f,%.6f,%.4f,%.6f,%.8f,%.4f,%.8f,%.8f,%.8f]",
      // H1 features
      close_ema50_h1, ema50_ema200_h1, rsi_val_h1, rsi_slope_h1,
      atr_ratio_h1, adx_val_h1, body_h1, range_h1, hour, session, prev_return_h1,
      // H4 features
      close_ema50_h4, ema50_ema200_h4, rsi_val_h4, rsi_slope_h4,
      atr_ratio_h4, adx_val_h4, body_h4, range_h4, prev_return_h4
   );
   
   string json_request = "{\"features\":" + features + "}";
   
   // Make HTTP request
   uchar post_data[];
   uchar result_data[];
   string result_headers;
   
   StringToCharArray(json_request, post_data, 0, StringLen(json_request));
   
   int timeout = 5000; // 5 seconds
   ResetLastError();
   
   string headers = "Content-Type: application/json\r\n";
   int res = WebRequest(
      "POST",
      API_URL,
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
         Print("Please add URL to allowed list in: Tools->Options->Expert Advisors");
         Print("Add: http://127.0.0.1, http://localhost");
      }
      return 0;
   }
   
   string response = CharArrayToString(result_data);
   Print("API Response: ", response);
   
   // Extract prediction and confidence
   double sell_prob = ExtractValue(response, "sell");
   double range_prob = ExtractValue(response, "range");
   double buy_prob = ExtractValue(response, "buy");
   confidence = ExtractValue(response, "confidence");
   
   string prediction = ExtractStringValue(response, "prediction");
   
   Print("ML Prediction: ", prediction, " (Confidence: ", DoubleToString(confidence*100, 1), "%)");
   Print("  Sell: ", DoubleToString(sell_prob*100, 1), "% | Range: ", 
         DoubleToString(range_prob*100, 1), "% | Buy: ", DoubleToString(buy_prob*100, 1), "%");
   
   if(confidence < ML_Confidence_Threshold)
   {
      Print("ML confidence too low, skipping trade");
      return 0;
   }
   
   if(prediction == "buy")
      return 1;
   else if(prediction == "sell")
      return -1;
   else
      return 0;
}

//+------------------------------------------------------------------+
//| Get combined trade signal (ML + Technical)                       |
//+------------------------------------------------------------------+
int GetTradeSignal()
{
   // Apply filters
   if(UseTimeFilter && !IsWithinTradingHours())
      return 0;
   
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(handleATR, 0, 0, 1, atr) <= 0)
      return 0;
   
   if(UseVolatilityFilter && atr[0] < MinATR)
      return 0;
   
   // Get ML prediction
   double ml_confidence = 0;
   int ml_signal = GetMLPrediction(ml_confidence);
   
   // If not using ML at all, get technical signal
   if(!UseMLPredictions)
      return GetTechnicalSignal();
   
   // If using ML only (no technical combination)
   if(!CombineWithTechnical)
   {
      if(ml_signal != 0 && ml_confidence >= ML_Confidence_Threshold)
      {
         Print("✓ ML signal only: ", ml_signal == 1 ? "BUY" : "SELL", " (Confidence: ", DoubleToString(ml_confidence*100, 1), "%)");
         return ml_signal;
      }
      else
      {
         if(ml_signal == 0)
            Print("ML returned no signal");
         else if(ml_confidence < ML_Confidence_Threshold)
            Print("ML confidence too low: ", DoubleToString(ml_confidence*100, 1), "% < ", DoubleToString(ML_Confidence_Threshold*100, 1), "%");
         return 0;
      }
   }
   
   // If combining ML with technical (CombineWithTechnical = true)
   if(CombineWithTechnical)
   {
      int tech_signal = GetTechnicalSignal();
      
      if(ml_signal == tech_signal && ml_signal != 0 && ml_confidence >= ML_Confidence_Threshold)
      {
         Print("✓ ML and Technical signals agree: ", ml_signal == 1 ? "BUY" : "SELL");
         return ml_signal;
      }
      else if(tech_signal != 0 && ml_confidence < ML_Confidence_Threshold)
      {
         Print("✗ ML confidence too low, using technical signal: ", tech_signal == 1 ? "BUY" : "SELL");
         return tech_signal;
      }
      else
      {
         Print("✗ ML and Technical signals disagree - no trade");
         Print("  ML: ", ml_signal == 1 ? "BUY" : (ml_signal == -1 ? "SELL" : "NONE"), 
               " (Confidence: ", DoubleToString(ml_confidence*100, 1), "%)");
         Print("  Technical: ", tech_signal == 1 ? "BUY" : (tech_signal == -1 ? "SELL" : "NONE"));
         return 0;
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Get technical signal (original strategy)                         |
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
   
   bool bullishCross = (maFast[1] > maSlow[1] && maFast[2] <= maSlow[2]);
   bool bearishCross = (maFast[1] < maSlow[1] && maFast[2] >= maSlow[2]);
   
   bool rsiBullish = rsi[0] > 50 && rsi[0] < RSI_Overbought;
   bool rsiBearish = rsi[0] < 50 && rsi[0] > RSI_Oversold;
   
   if(bullishCross && rsiBullish)
   {
      if(!UseTrendFilter || upTrend)
         return 1;
   }
   
   if(bearishCross && rsiBearish)
   {
      if(!UseTrendFilter || downTrend)
         return -1;
   }
   
   return 0;
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
      // Try alternative format
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
      // Try alternative format
      search = "'" + key + "':'";
      start = StringFind(json, search);
      if(start == -1) return "";
   }
   
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   
   if(end == -1)
   {
      // Try single quotes
      end = StringFind(json, "'", start);
   }
   
   if(end == -1 || end <= start)
      return "";
   
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Custom string trim function                                      |
//+------------------------------------------------------------------+
string StringTrimCustom(const string str)
{
   string result = str;
   
   // Trim leading spaces
   while(StringSubstr(result, 0, 1) == " ")
      result = StringSubstr(result, 1);
   
   // Trim trailing spaces
   while(StringSubstr(result, StringLen(result)-1, 1) == " ")
      result = StringSubstr(result, 0, StringLen(result)-1);
   
   return result;
}

//+------------------------------------------------------------------+
//| Custom string to double conversion                               |
//+------------------------------------------------------------------+
double StringToDoubleCustom(const string value)
{
   // Remove any non-numeric characters except decimal point and minus
   string clean_value = "";
   for(int i = 0; i < StringLen(value); i++)
   {
      string ch = StringSubstr(value, i, 1);
      if((ch >= "0" && ch <= "9") || ch == "." || ch == "-")
         clean_value += ch;
   }
   
   if(clean_value == "" || clean_value == "-")
      return 0.0;
   
   // Use MQL5's built-in conversion
   return ::StringToDouble(clean_value);
}

//+------------------------------------------------------------------+
//| Open a new trade with dynamic risk management                    |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE orderType)
{
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(handleATR, 0, 0, 1, atr) <= 0)
   {
      Print("Failed to get ATR for position sizing");
      return;
   }
   
   double price = (orderType == ORDER_TYPE_BUY) ?
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                  SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double slDistance = atr[0] * ATR_Multiplier;
   double tpDistance = atr[0] * ATR_Multiplier * 2.0;
   
   double sl = (orderType == ORDER_TYPE_BUY) ? price - slDistance : price + slDistance;
   double tp = (orderType == ORDER_TYPE_BUY) ? price + tpDistance : price - tpDistance;
   
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   
   double lotSize = CalculateLotSize(MathAbs(price - sl));
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lotSize < minLot)
   {
      Print("Calculated lot size too small: ", lotSize, ", minimum: ", minLot);
      lotSize = minLot;
   }
   
   string comment = UseMLPredictions ? "ML-Enhanced EA " : "Advanced EA ";
   comment += (orderType == ORDER_TYPE_BUY) ? "Buy" : "Sell";
   
   if(orderType == ORDER_TYPE_BUY)
   {
      if(trade.Buy(lotSize, _Symbol, price, sl, tp, comment))
      {
         Print("BUY order opened: Lot=", lotSize, " Price=", price, " SL=", sl, " TP=", tp);
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open BUY order: ", trade.ResultRetcodeDescription());
      }
   }
   else
   {
      if(trade.Sell(lotSize, _Symbol, price, sl, tp, comment))
      {
         Print("SELL order opened: Lot=", lotSize, " Price=", price, " SL=", sl, " TP=", tp);
         totalTradesToday++;
      }
      else
      {
         Print("Failed to open SELL order: ", trade.ResultRetcodeDescription());
      }
   }
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
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(tickValue <= 0 || tickSize <= 0 || point <= 0)
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
   
   // Safety check for margin
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double marginRequired = 0.0;
   
   // Calculate approximate margin required for this lot size
   double margin_per_lot = SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
   if(margin_per_lot > 0)
      marginRequired = lotSize * margin_per_lot;
   
   // Use max 50% of free margin
   if(marginRequired > freeMargin * 0.5 && freeMargin > 0)
   {
      Print("Reducing lot size due to margin constraints");
      lotSize = MathMin(lotSize, freeMargin * 0.5 / marginRequired * lotSize);
      lotSize = MathFloor(lotSize / lotStep) * lotStep;
      lotSize = NormalizeDouble(lotSize, 2);
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
         
         ulong ticket = posInfo.Ticket();
         double currentPrice = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         
         double openPrice = posInfo.PriceOpen();
         double sl = posInfo.StopLoss();
         double profitPips = 0;
         
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         if(point > 0)
         {
            if(posInfo.PositionType() == POSITION_TYPE_BUY)
               profitPips = (currentPrice - openPrice) / (point * 10); // Convert to pips (10 points per pip for 5-digit brokers)
            else
               profitPips = (openPrice - currentPrice) / (point * 10);
         }
         
         if(UseBreakeven && profitPips >= BreakevenTriggerPips && sl != 0)
         {
            double newSL = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                           openPrice + BreakevenOffsetPips * point * 10 :
                           openPrice - BreakevenOffsetPips * point * 10;
            
            newSL = NormalizeDouble(newSL, _Digits);
            
            if((posInfo.PositionType() == POSITION_TYPE_BUY && newSL > sl) ||
               (posInfo.PositionType() == POSITION_TYPE_SELL && newSL < sl))
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print("Position ", ticket, " moved to breakeven+", BreakevenOffsetPips, " pips");
            }
         }
         
         if(UsePartialClose && profitPips >= BreakevenTriggerPips * 2)
         {
            double closeVolume = posInfo.Volume() * PartialClosePercent / 100.0;
            double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            closeVolume = MathMax(closeVolume, minLot);
            closeVolume = NormalizeDouble(closeVolume, 2);
            
            if(closeVolume >= minLot && closeVolume < posInfo.Volume())
            {
               if(trade.PositionClosePartial(ticket, closeVolume))
                  Print("Partial close: ", PartialClosePercent, "% of position ", ticket, " at profit");
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
         
         ulong ticket = posInfo.Ticket();
         double currentPrice = (posInfo.PositionType() == POSITION_TYPE_BUY) ?
                               SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                               SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         
         double sl = posInfo.StopLoss();
         double trailDistance = TrailingStopPips * point * 10; // Convert pips to price
         double trailStep = TrailingStepPips * point * 10;
         
         if(posInfo.PositionType() == POSITION_TYPE_BUY && sl > 0)
         {
            double newSL = currentPrice - trailDistance;
            newSL = NormalizeDouble(newSL, _Digits);
            if(newSL > sl + trailStep)
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print("Trailing stop updated for BUY position ", ticket);
            }
         }
         else if(posInfo.PositionType() == POSITION_TYPE_SELL && sl > 0)
         {
            double newSL = currentPrice + trailDistance;
            newSL = NormalizeDouble(newSL, _Digits);
            if(newSL < sl - trailStep)
            {
               if(trade.PositionModify(ticket, newSL, posInfo.TakeProfit()))
                  Print("Trailing stop updated for SELL position ", ticket);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check daily limits (max loss/profit)                             |
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
      Print("Daily loss limit reached: ", dailyPL_Percent, "%");
      return false;
   }
   
   if(dailyPL_Percent >= MaxDailyProfit)
   {
      Print("Daily profit target reached: ", dailyPL_Percent, "%");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if new day started and reset counters                      |
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
      Print("Starting balance: $", dailyStartBalance);
   }
}

//+------------------------------------------------------------------+
//| Check if within trading hours                                    |
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
