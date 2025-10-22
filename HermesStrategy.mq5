//+------------------------------------------------------------------+
//|                                              HermesStrategy.mq5 |
//|                        Hermes Strategy - Expert Advisor          |
//|                         Converted from Pine Script Strategy      |
//+------------------------------------------------------------------+
#property copyright "Hermes Strategy"
#property link      ""
#property version   "1.00"

//--- Indicator handle for ALMA signals
int hermesIndicatorHandle = INVALID_HANDLE;

//============================================================================
// INPUTS
//============================================================================
input int    ShortPeriod = 30;              // Short Period (10-200)
input int    LongPeriod = 250;              // Long Period (50-400)
input double AlmaOffset = 0.95;             // ALMA Offset (0.0-1.0)
input double AlmaSigma = 4.0;               // ALMA Sigma (1.0-10.0)

input int    MomentumLookback = 3;          // Momentum Lookback (1-20)
input bool   UseMomentumFilters = true;     // Use Momentum Filters

input int    FastHmaPeriod = 30;            // Trend Fast Period (1-200)
input int    SlowEmaPeriod = 80;            // Slow EMA Period (2-400)

input double RiskPercent = 100.0;           // Portfolio Percent Per Trade
input int    MagicNumber = 123456;          // Magic Number

//--- Global state variables
bool   trendingRegimeLong = false;
bool   trendingRegimeShort = false;
double positionEntryPrice = 0.0;

//--- Arrays for calculations
double closeArray[];
double highArray[];
double lowArray[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Hermes Strategy EA initialized");

   // Reset state
   trendingRegimeLong = false;
   trendingRegimeShort = false;
   positionEntryPrice = 0.0;

   // Create custom indicator handle for Hermes ALMA signals
   // Try without path first (same directory), then with Indicators path
   hermesIndicatorHandle = iCustom(_Symbol, _Period, "HermesIndicator",
                                   ShortPeriod, LongPeriod, AlmaOffset, AlmaSigma);

   // If failed, try with full path to Indicators folder
   if(hermesIndicatorHandle == INVALID_HANDLE)
   {
      hermesIndicatorHandle = iCustom(_Symbol, _Period, "Indicators\\HermesIndicator",
                                      ShortPeriod, LongPeriod, AlmaOffset, AlmaSigma);
   }

   if(hermesIndicatorHandle == INVALID_HANDLE)
   {
      Print("Failed to create Hermes Indicator handle.");
      Print("Make sure HermesIndicator.mq5 is compiled and located in:");
      Print("1. MQL5/Indicators/ folder, OR");
      Print("2. Same folder as this EA");
      return(INIT_FAILED);
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Hermes Strategy EA deinitialized. Reason: ", reason);

   // Release indicator handle
   if(hermesIndicatorHandle != INVALID_HANDLE)
      IndicatorRelease(hermesIndicatorHandle);
}

//+------------------------------------------------------------------+
//| HMA (Hull Moving Average) Calculation using WMA                  |
//+------------------------------------------------------------------+
double CalculateSimpleHMA(int period, int shift)
{
   // Simplified HMA using double-smoothed WMA approximation
   int halfPeriod = (int)MathMax(1, period / 2);

   // Create temporary handles if needed
   static int localWmaHalfHandle = INVALID_HANDLE;
   static int localWmaFullHandle = INVALID_HANDLE;
   static int lastPeriod = 0;

   // Recreate handles if period changed
   if(lastPeriod != period)
   {
      if(localWmaHalfHandle != INVALID_HANDLE) IndicatorRelease(localWmaHalfHandle);
      if(localWmaFullHandle != INVALID_HANDLE) IndicatorRelease(localWmaFullHandle);

      localWmaHalfHandle = iMA(_Symbol, _Period, halfPeriod, 0, MODE_LWMA, PRICE_CLOSE);
      localWmaFullHandle = iMA(_Symbol, _Period, period, 0, MODE_LWMA, PRICE_CLOSE);
      lastPeriod = period;
   }

   if(localWmaHalfHandle == INVALID_HANDLE || localWmaFullHandle == INVALID_HANDLE)
      return 0.0;

   double wmaHalfBuffer[];
   double wmaFullBuffer[];
   ArraySetAsSeries(wmaHalfBuffer, true);
   ArraySetAsSeries(wmaFullBuffer, true);

   if(CopyBuffer(localWmaHalfHandle, 0, shift, 1, wmaHalfBuffer) <= 0) return 0.0;
   if(CopyBuffer(localWmaFullHandle, 0, shift, 1, wmaFullBuffer) <= 0) return 0.0;

   return 2.0 * wmaHalfBuffer[0] - wmaFullBuffer[0];
}

//+------------------------------------------------------------------+
//| Get Highest Value in Range                                       |
//+------------------------------------------------------------------+
double GetHighest(const double &array[], int count, int shift)
{
   double highest = -DBL_MAX;
   for(int i = 0; i < count; i++)
   {
      int idx = shift + i;
      if(idx >= ArraySize(array)) continue;
      if(array[idx] > highest) highest = array[idx];
   }
   return highest;
}

//+------------------------------------------------------------------+
//| Get Lowest Value in Range                                        |
//+------------------------------------------------------------------+
double GetLowest(const double &array[], int count, int shift)
{
   double lowest = DBL_MAX;
   for(int i = 0; i < count; i++)
   {
      int idx = shift + i;
      if(idx >= ArraySize(array)) continue;
      if(array[idx] < lowest) lowest = array[idx];
   }
   return lowest;
}

//+------------------------------------------------------------------+
//| Check if we're in a position                                     |
//+------------------------------------------------------------------+
bool IsInPosition()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            positionEntryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Get current position type (returns POSITION_TYPE_BUY, POSITION_TYPE_SELL, or -1) |
//+------------------------------------------------------------------+
int GetPositionType()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            return (int)PositionGetInteger(POSITION_TYPE);
         }
      }
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Calculate lot size                                                |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lotSize = (balance * RiskPercent / 100.0) / (SymbolInfoDouble(_Symbol, SYMBOL_BID) * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));

   // Normalize to broker's lot step
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

   return lotSize;
}

//+------------------------------------------------------------------+
//| Open Long Position                                                |
//+------------------------------------------------------------------+
bool OpenLongPosition()
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = CalculateLotSize();
   request.type = ORDER_TYPE_BUY;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Hermes Long";

   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
      {
         Print("Long position opened at ", request.price);
         positionEntryPrice = request.price;
         trendingRegimeLong = false;
         return true;
      }
   }

   Print("Failed to open long position. Error: ", GetLastError(), " Retcode: ", result.retcode);
   return false;
}

//+------------------------------------------------------------------+
//| Open Short Position                                               |
//+------------------------------------------------------------------+
bool OpenShortPosition()
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = CalculateLotSize();
   request.type = ORDER_TYPE_SELL;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Hermes Short";

   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
      {
         Print("Short position opened at ", request.price);
         positionEntryPrice = request.price;
         trendingRegimeShort = false;
         return true;
      }
   }

   Print("Failed to open short position. Error: ", GetLastError(), " Retcode: ", result.retcode);
   return false;
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
bool ClosePosition(string reason)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};

            // Get position type to determine close order type
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

            request.action = TRADE_ACTION_DEAL;
            request.symbol = _Symbol;
            request.position = ticket;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            request.price = (posType == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            request.deviation = 10;
            request.magic = MagicNumber;
            request.comment = reason;

            if(OrderSend(request, result))
            {
               if(result.retcode == TRADE_RETCODE_DONE)
               {
                  Print("Position closed. Reason: ", reason);
                  trendingRegimeLong = false;
                  trendingRegimeShort = false;
                  return true;
               }
            }

            Print("Failed to close position. Error: ", GetLastError());
            return false;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;

   // Ensure we have enough bars
   int bars = Bars(_Symbol, _Period);
   if(bars < LongPeriod + 10) return;

   // Prepare arrays for price data
   int arraySize = 100;
   ArrayResize(closeArray, arraySize);
   ArrayResize(highArray, arraySize);
   ArrayResize(lowArray, arraySize);

   ArraySetAsSeries(closeArray, true);
   ArraySetAsSeries(highArray, true);
   ArraySetAsSeries(lowArray, true);

   // Copy price data
   if(CopyClose(_Symbol, _Period, 0, arraySize, closeArray) < arraySize) return;
   if(CopyHigh(_Symbol, _Period, 0, arraySize, highArray) < arraySize) return;
   if(CopyLow(_Symbol, _Period, 0, arraySize, lowArray) < arraySize) return;

   // Get ALMA values from indicator
   double shortTermBuffer[];
   double baselineBuffer[];
   ArraySetAsSeries(shortTermBuffer, true);
   ArraySetAsSeries(baselineBuffer, true);

   double shortTerm = 0.0;
   double baseline = 0.0;

   // Buffer 0 = Short-Term Signal, Buffer 1 = Long-Term Baseline
   if(CopyBuffer(hermesIndicatorHandle, 0, 0, 1, shortTermBuffer) > 0)
      shortTerm = shortTermBuffer[0];
   if(CopyBuffer(hermesIndicatorHandle, 1, 0, 1, baselineBuffer) > 0)
      baseline = baselineBuffer[0];

   // Calculate HMA and EMA for trending regime
   double fastHma = CalculateSimpleHMA(FastHmaPeriod, 0);

   // Get slow EMA value
   static int slowEmaHandle = INVALID_HANDLE;
   if(slowEmaHandle == INVALID_HANDLE)
   {
      slowEmaHandle = iMA(_Symbol, _Period, SlowEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   }

   double slowEmaBuffer[];
   ArraySetAsSeries(slowEmaBuffer, true);
   double slowEma = 0.0;
   if(CopyBuffer(slowEmaHandle, 0, 0, 1, slowEmaBuffer) > 0)
      slowEma = slowEmaBuffer[0];

   // Get current prices
   double currentClose = closeArray[0];
   double currentHigh = highArray[0];
   double currentLow = lowArray[0];

   //============================================================================
   // SIGNAL LOGIC
   //============================================================================
   bool bullishState = (shortTerm > baseline);
   bool bearishState = (shortTerm < baseline);

   // Momentum filters
   bool isHighestClose = true;
   bool isLowestLow = true;

   if(UseMomentumFilters && MomentumLookback > 0)
   {
      double highestClose = GetHighest(closeArray, MomentumLookback, 1);
      double highestHigh = GetHighest(highArray, MomentumLookback, 1);
      isHighestClose = (currentClose >= highestClose && currentHigh >= highestHigh);

      double lowestLow = GetLowest(lowArray, MomentumLookback, 1);
      double lowestClose = GetLowest(closeArray, MomentumLookback, 1);
      isLowestLow = (currentLow <= lowestLow && currentClose <= lowestClose);
   }

   // Build long signals
   bool buySignal = bullishState;
   if(UseMomentumFilters) buySignal = buySignal && isHighestClose;

   // Build short signals (reversed logic)
   bool sellSignal = bearishState;
   if(UseMomentumFilters) sellSignal = sellSignal && isLowestLow;

   //============================================================================
   // REGIME DETECTION & EXIT LOGIC
   //============================================================================
   bool inPosition = IsInPosition();
   int posType = GetPositionType();

   // Reset regimes when not in position
   if(!inPosition)
   {
      trendingRegimeLong = false;
      trendingRegimeShort = false;
   }

   // Entry logic - only enter if no position exists
   if(!inPosition)
   {
      if(buySignal)
      {
         OpenLongPosition();
         return;
      }
      else if(sellSignal)
      {
         OpenShortPosition();
         return;
      }
   }

   // Exit logic for both long and short positions
   if(inPosition)
   {
      // Store previous values for cross detection
      static double prevFastHma = 0;
      static double prevSlowEma = 0;
      bool trendCrossUnder = (prevFastHma > prevSlowEma && fastHma < slowEma);
      bool trendCrossOver = (prevFastHma < prevSlowEma && fastHma > slowEma);
      prevFastHma = fastHma;
      prevSlowEma = slowEma;

      //========================================================================
      // LONG POSITION EXIT LOGIC
      //========================================================================
      if(posType == POSITION_TYPE_BUY)
      {
         // Trending regime detection for longs
         bool trendingSetupLong = (slowEma > positionEntryPrice &&
                                   fastHma > positionEntryPrice &&
                                   fastHma > slowEma);

         if(trendingSetupLong)
            trendingRegimeLong = true;
         else if(!trendCrossUnder)
            trendingRegimeLong = false;

         // Trending regime exit
         bool buyMomentumOk = !UseMomentumFilters || isHighestClose;
         bool closeBelowEntry = (currentClose < positionEntryPrice);
         bool normalTrendingExit = trendCrossUnder && (!UseMomentumFilters || isLowestLow);

         bool trendingExit = trendingRegimeLong && (closeBelowEntry || normalTrendingExit);

         // Ranging regime exit
         bool rangingExit = !trendingRegimeLong && bearishState;
         if(UseMomentumFilters) rangingExit = rangingExit && isLowestLow;

         // Execute exit
         if(trendingExit)
         {
            ClosePosition(closeBelowEntry ? "Long Trending: Below Entry" : "Long Trending: HMA Cross");
         }
         else if(rangingExit)
         {
            ClosePosition("Long Ranging: ALMA Signal");
         }
      }

      //========================================================================
      // SHORT POSITION EXIT LOGIC (REVERSED)
      //========================================================================
      else if(posType == POSITION_TYPE_SELL)
      {
         // Trending regime detection for shorts (reversed)
         bool trendingSetupShort = (slowEma < positionEntryPrice &&
                                    fastHma < positionEntryPrice &&
                                    fastHma < slowEma);

         if(trendingSetupShort)
            trendingRegimeShort = true;
         else if(!trendCrossOver)
            trendingRegimeShort = false;

         // Trending regime exit (reversed)
         bool sellMomentumOk = !UseMomentumFilters || isLowestLow;
         bool closeAboveEntry = (currentClose > positionEntryPrice);
         bool normalTrendingExit = trendCrossOver && (!UseMomentumFilters || isHighestClose);

         bool trendingExit = trendingRegimeShort && (closeAboveEntry || normalTrendingExit);

         // Ranging regime exit (reversed)
         bool rangingExit = !trendingRegimeShort && bullishState;
         if(UseMomentumFilters) rangingExit = rangingExit && isHighestClose;

         // Execute exit
         if(trendingExit)
         {
            ClosePosition(closeAboveEntry ? "Short Trending: Above Entry" : "Short Trending: HMA Cross");
         }
         else if(rangingExit)
         {
            ClosePosition("Short Ranging: ALMA Signal");
         }
      }
   }
}
//+------------------------------------------------------------------+
