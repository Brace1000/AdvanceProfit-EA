//+------------------------------------------------------------------+
//|                                                    AdvanceEA.mq5 |
//|                        Copyright 2024, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "2.10"
#property strict

//--- Input parameters
input bool   UseMLPredictions = true;           // Enable ML predictions
input string API_URL = "http://127.0.0.1:8000/predict"; // ML API URL
input double ML_Confidence_Threshold = 0.40;   // Minimum ML confidence (0.0-1.0)
input bool   CombineWithTechnical = true;      // Require ML + Technical agreement

input double LotSize = 0.10;                   // Lot size
input int    StopLoss = 100;                   // Stop Loss in points
input int    TakeProfit = 200;                 // Take Profit in points
input bool   UseTrailingStop = true;           // Use trailing stop
input int    TrailingStopPoints = 50;          // Trailing stop points

input int    MagicNumber = 12345;              // Magic number for orders

//--- Global variables
int ticket = 0;
double lastPrediction = 0.0;
string lastPredictionStr = "none";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Check if WebRequest is allowed
   if(!TerminalInfoInteger(TERMINAL_WEBREQUEST_ENABLED))
   {
      Print("ERROR: WebRequest not enabled! Enable in Tools -> Options -> Expert Advisors");
      return(INIT_FAILED);
   }

   // Add API URL to allowed URLs
   if(!WebRequestCheck(API_URL))
   {
      Print("ERROR: API URL not in allowed list! Add to Tools -> Options -> Expert Advisors");
      return(INIT_FAILED);
   }

   Print("Advanced Profit EA v2.10 with ML");
   Print("ML Integration: ", UseMLPredictions ? "ENABLED" : "DISABLED");
   Print("API URL: ", API_URL);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Cleanup
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only trade on new bar
   static datetime lastBar = 0;
   if(lastBar == iTime(_Symbol, PERIOD_CURRENT, 0)) return;
   lastBar = iTime(_Symbol, PERIOD_CURRENT, 0);

   // Check for open positions
   if(PositionSelect(_Symbol))
   {
      ManageOpenPosition();
      return;
   }

   // Generate signal
   int signal = GenerateSignal();
   if(signal == 0) return;

   // Open position
   OpenPosition(signal);
}

//+------------------------------------------------------------------+
//| Generate trading signal                                          |
//+------------------------------------------------------------------+
int GenerateSignal()
{
   int technicalSignal = GetTechnicalSignal();
   int mlSignal = UseMLPredictions ? GetMLSignal() : technicalSignal;

   if(!UseMLPredictions)
   {
      return technicalSignal;
   }

   if(CombineWithTechnical)
   {
      if(technicalSignal == mlSignal && technicalSignal != 0)
      {
         Print("✓ ML and Technical signals agree: ", technicalSignal > 0 ? "BUY" : "SELL");
         return technicalSignal;
      }
      else
      {
         Print("✗ ML and Technical signals disagree - skipping");
         return 0;
      }
   }
   else
   {
      return mlSignal;
   }
}

//+------------------------------------------------------------------+
//| Get technical signal                                             |
//+------------------------------------------------------------------+
int GetTechnicalSignal()
{
   // Simple EMA crossover strategy
   double ema50 = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);

   if(ema50 > ema200) return 1;  // Buy
   if(ema50 < ema200) return -1; // Sell

   return 0;
}

//+------------------------------------------------------------------+
//| Get ML signal                                                    |
//+------------------------------------------------------------------+
int GetMLSignal()
{
   // Calculate features for daily timeframe
   double features[11];
   if(!CalculateFeatures(features))
   {
      Print("Failed to calculate ML features");
      return 0;
   }

   // Call API
   string json = CreateJSON(features);
   string response = CallAPI(json);

   if(response == "")
   {
      Print("Failed to get ML prediction");
      return 0;
   }

   // Parse response
   return ParsePrediction(response);
}

//+------------------------------------------------------------------+
//| Calculate ML features                                            |
//+------------------------------------------------------------------+
bool CalculateFeatures(double &features[])
{
   // Use daily timeframe for features
   ENUM_TIMEFRAMES tf = PERIOD_D1;

   // Need enough bars
   if(iBars(_Symbol, tf) < 200)
   {
      Print("Not enough bars for ML features");
      return false;
   }

   double close = iClose(_Symbol, tf, 0);
   double ema50 = iMA(_Symbol, tf, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, tf, 200, 0, MODE_EMA, PRICE_CLOSE);

   features[0] = close;                          // close
   features[1] = ema50 - ema200;                 // ema50_ema200
   features[2] = iRSI(_Symbol, tf, 14, PRICE_CLOSE, 0); // rsi
   features[3] = 0.0;                            // rsi_slope (simplified)
   features[4] = iATR(_Symbol, tf, 14, 0) / close; // atr_ratio
   features[5] = iADX(_Symbol, tf, 14, 0);       // adx
   features[6] = MathAbs(iOpen(_Symbol, tf, 0) - close); // body
   features[7] = iHigh(_Symbol, tf, 0) - iLow(_Symbol, tf, 0); // range
   features[8] = 0;                              // hour
   features[9] = 0;                              // session
   features[10] = (close - iClose(_Symbol, tf, 1)) / iClose(_Symbol, tf, 1); // prev_return

   return true;
}

//+------------------------------------------------------------------+
//| Create JSON for API call                                         |
//+------------------------------------------------------------------+
string CreateJSON(double &features[])
{
   string json = "{\"features\": [";
   for(int i = 0; i < ArraySize(features); i++)
   {
      json += DoubleToString(features[i], 6);
      if(i < ArraySize(features) - 1) json += ",";
   }
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Call ML API                                                      |
//+------------------------------------------------------------------+
string CallAPI(string json)
{
   string headers = "Content-Type: application/json";
   char post[];
   char result[];
   string result_headers;

   StringToCharArray(json, post, 0, StringLen(json));

   int res = WebRequest("POST", API_URL, headers, 5000, post, result, result_headers);

   if(res == -1)
   {
      Print("WebRequest error: ", GetLastError());
      return "";
   }

   if(res != 200)
   {
      Print("API returned status: ", res);
      return "";
   }

   return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| Parse ML prediction response                                     |
//+------------------------------------------------------------------+
int ParsePrediction(string response)
{
   // Simple JSON parsing (in real implementation, use proper JSON parser)
   string prediction = "";
   double confidence = 0.0;

   // Extract prediction
   int pos = StringFind(response, "\"prediction\":\"");
   if(pos != -1)
   {
      pos += 14;
      int end = StringFind(response, "\"", pos);
      if(end != -1)
      {
         prediction = StringSubstr(response, pos, end - pos);
      }
   }

   // Extract confidence
   pos = StringFind(response, "\"confidence\":");
   if(pos != -1)
   {
      pos += 13;
      int end = StringFind(response, "}", pos);
      if(end != -1)
      {
         confidence = StringToDouble(StringSubstr(response, pos, end - pos));
      }
   }

   Print("ML Prediction: ", prediction, " (Confidence: ", DoubleToString(confidence, 3), ")");

   if(confidence < ML_Confidence_Threshold)
   {
      Print("ML confidence too low: ", DoubleToString(confidence, 3), " < ", DoubleToString(ML_Confidence_Threshold, 2));
      return 0;
   }

   if(prediction == "buy") return 1;
   if(prediction == "sell") return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Open position                                                    |
//+------------------------------------------------------------------+
void OpenPosition(int signal)
{
   double price = (signal > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = (signal > 0) ? price - StopLoss * _Point : price + StopLoss * _Point;
   double tp = (signal > 0) ? price + TakeProfit * _Point : price - TakeProfit * _Point;

   ENUM_ORDER_TYPE type = (signal > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = type;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = MagicNumber;
   request.comment = "AdvanceEA ML";

   if(OrderSend(request, result))
   {
      Print((signal > 0 ? "BUY" : "SELL"), " order opened: Lot=", DoubleToString(LotSize, 2),
            " SL=", DoubleToString(sl, _Digits), " TP=", DoubleToString(tp, _Digits));
   }
   else
   {
      Print("Order failed: ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| Manage open position                                             |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   if(!UseTrailingStop) return;

   ulong ticket = PositionGetTicket(0);
   double currentSL = PositionGetDouble(POSITION_SL);
   double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);

   bool isBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;

   double newSL = currentSL;

   if(isBuy)
   {
      if(currentPrice > PositionGetDouble(POSITION_PRICE_OPEN) + TrailingStopPoints * _Point)
      {
         newSL = currentPrice - TrailingStopPoints * _Point;
         if(newSL > currentSL)
         {
            ModifyPosition(ticket, newSL, PositionGetDouble(POSITION_TP));
         }
      }
   }
   else
   {
      if(currentPrice < PositionGetDouble(POSITION_PRICE_OPEN) - TrailingStopPoints * _Point)
      {
         newSL = currentPrice + TrailingStopPoints * _Point;
         if(newSL < currentSL)
         {
            ModifyPosition(ticket, newSL, PositionGetDouble(POSITION_TP));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Modify position                                                  |
//+------------------------------------------------------------------+
void ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_SLTP;
   request.symbol = _Symbol;
   request.sl = sl;
   request.tp = tp;
   request.position = ticket;

   OrderSend(request, result);
}

//+------------------------------------------------------------------+
