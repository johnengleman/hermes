//+------------------------------------------------------------------+
//|                                            HermesIndicator.mq5   |
//|                        Hermes Strategy - Indicator Only          |
//|                         Shows ALMA signals in separate window    |
//+------------------------------------------------------------------+
#property copyright "Hermes Indicator"
#property link      ""
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

//--- plot Short-Term Signal
#property indicator_label1  "Short-Term Signal"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot Long-Term Baseline
#property indicator_label2  "Long-Term Baseline"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrBlack
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- plot Zero Line
#property indicator_label3  "Zero"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGray
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

//============================================================================
// INPUTS
//============================================================================
input int    ShortPeriod = 30;              // Short Period (10-200)
input int    LongPeriod = 250;              // Long Period (50-400)
input double AlmaOffset = 0.95;             // ALMA Offset (0.0-1.0)
input double AlmaSigma = 4.0;               // ALMA Sigma (1.0-10.0)

//--- Indicator buffers
double ShortTermBuffer[];
double BaselineBuffer[];
double ZeroBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- indicator buffers mapping
   SetIndexBuffer(0, ShortTermBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, BaselineBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ZeroBuffer, INDICATOR_DATA);

   //--- set precision
   IndicatorSetInteger(INDICATOR_DIGITS, 5);

   //--- set label name
   IndicatorSetString(INDICATOR_SHORTNAME, "Hermes ALMA Signals");

   //--- initialization done
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| ALMA (Arnaud Legoux Moving Average) Calculation                  |
//| Calculates ALMA looking back from 'shift' position               |
//+------------------------------------------------------------------+
double CalculateALMA(const double &array[], int period, double offset, double sigma, int shift)
{
   if(period <= 0 || shift < period - 1) return 0.0;

   double m = offset * (period - 1);
   double s = period / sigma;
   double sum = 0.0;
   double weightSum = 0.0;

   // Look back from shift position
   // i=0 is the oldest bar in the lookback window
   // i=period-1 is the most recent bar (at shift position)
   for(int i = 0; i < period; i++)
   {
      int idx = shift - (period - 1 - i);  // Oldest to newest
      if(idx < 0 || idx >= ArraySize(array)) continue;

      double weight = MathExp(-((i - m) * (i - m)) / (2 * s * s));
      sum += array[idx] * weight;
      weightSum += weight;
   }

   return (weightSum > 0) ? sum / weightSum : 0.0;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   //--- Don't set price arrays as series - work with normal indexing
   ArraySetAsSeries(ShortTermBuffer, false);
   ArraySetAsSeries(BaselineBuffer, false);
   ArraySetAsSeries(ZeroBuffer, false);

   int start = prev_calculated - 1;
   if(start < LongPeriod) start = LongPeriod;

   //--- Arrays for calculations
   double logReturnArray[];

   ArrayResize(logReturnArray, rates_total);
   ArraySetAsSeries(logReturnArray, false);

   //--- Calculate log returns for all bars first
   for(int i = 1; i < rates_total; i++)
   {
      double dailyReturn = (close[i - 1] > 0) ? close[i] / close[i - 1] : 1.0;
      logReturnArray[i] = MathLog(dailyReturn);
   }
   logReturnArray[0] = 0.0;

   //--- Main calculation loop
   for(int i = start; i < rates_total; i++)
   {
      //--- Calculate ALMA for short and long term
      double shortTerm = CalculateALMA(logReturnArray, ShortPeriod, AlmaOffset, AlmaSigma, i);
      double longTerm = CalculateALMA(logReturnArray, LongPeriod, AlmaOffset, AlmaSigma, i);

      //--- Store in buffers
      ShortTermBuffer[i] = shortTerm;
      BaselineBuffer[i] = longTerm;
      ZeroBuffer[i] = 0.0;
   }

   //--- Return value of prev_calculated for next call
   return(rates_total);
}
//+------------------------------------------------------------------+
