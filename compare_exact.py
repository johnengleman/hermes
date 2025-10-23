"""
Compare Python with exact TradingView parameters and date range
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# OPTIMIZER'S BEST PARAMETERS (from user):
TRADINGVIEW_PARAMS = {
    "short_period": 12,
    "long_period": 377,
    "alma_offset": 0.97,
    "alma_sigma": 9.0,
    "momentum_lookback": 1,
    "use_macro_filter": 1,  # ENABLED!
    "macro_ema_period": 121,
    "fast_hma_period": 52,
    "slow_ema_period": 59,
    "slow_ema_rising_lookback": 7,
}

# Load BTC data
print("Loading BTC data...")
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

# Use ALL data for indicator calculation (need history for 377-day ALMA!)
close = df['Close']
high = df['High']
low = df['Low']

print(f"Data range: {close.index[0]} to {close.index[-1]}")
print(f"Total days: {len(close)}\n")

# Run strategy
print("Running strategy with TradingView parameters...")
print("Parameters:")
for key, value in TRADINGVIEW_PARAMS.items():
    print(f"  {key:30s}: {value}")
print()

entries, exits, position_size = run_strategy_simple(close, high, low, **TRADINGVIEW_PARAMS)

# Create portfolio (execute at OPEN prices to match TradingView)
open_prices = df['Open']
portfolio = vbt.Portfolio.from_signals(
    close=close,
    price=open_prices,  # Execute at OPEN prices (not 'open' parameter!)
    entries=entries,
    exits=exits,
    size=1.0,
    size_type=SizeType.Percent,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq='1D'
)

# Get trades
trades = portfolio.trades.records_readable

print("="*80)
print("COMPARISON")
print("="*80)
print(f"Python Trades:          {len(trades)}")
print(f"TradingView Trades:     124")
print(f"Difference:             {len(trades) - 124}")
print()
print(f"Python Return:          {portfolio.total_return()*100:,.1f}%")
print(f"TradingView Return:     114,720%")
print(f"Difference:             {portfolio.total_return()*100 - 114720:.1f}%")
print()

# Show first 10 trades for comparison
print("="*80)
print("FIRST 10 PYTHON TRADES (compare with TradingView)")
print("="*80)
for i in range(min(10, len(trades))):
    trade = trades.iloc[i]
    entry_date = trade['Entry Timestamp'].strftime('%Y-%m-%d')
    exit_date = trade['Exit Timestamp'].strftime('%Y-%m-%d')
    entry_price = trade['Avg Entry Price']
    exit_price = trade['Avg Exit Price']
    pnl = trade['Return'] * 100
    print(f"Trade {i+1}: Entry {entry_date} @ ${entry_price:.2f}  →  Exit {exit_date} @ ${exit_price:.2f}  |  {pnl:+.2f}%")

print("\n" + "="*80)
print("FIRST 10 TRADINGVIEW TRADES (from your CSV)")
print("="*80)
print("Trade 1: Entry 2016-02-18 @ $415.18  →  Exit 2016-02-24 @ $420.32  |  +1.24%")
print("Trade 2: Entry 2016-02-27 @ $430.84  →  Exit 2016-03-02 @ $433.03  |  +0.51%")
print("Trade 3: Entry 2016-03-08 @ $412.21  →  Exit 2016-03-13 @ $410.56  |  -0.40%")
print("Trade 4: Entry 2016-03-16 @ $415.41  →  Exit 2016-03-19 @ $409.66  |  -1.38%")
print("Trade 5: Entry 2016-03-23 @ $416.66  →  Exit 2016-03-25 @ $412.86  |  -0.91%")
print("Trade 6: Entry 2016-03-27 @ $416.96  →  Exit 2016-03-30 @ $416.40  |  -0.13%")
print("Trade 7: Entry 2016-04-02 @ $417.68  →  Exit 2016-04-09 @ $419.48  |  +0.43%")
print("Trade 8: Entry 2016-04-11 @ $423.29  →  Exit 2016-04-19 @ $430.90  |  +1.80%")
print("Trade 9: Entry 2016-04-20 @ $437.06  →  Exit 2016-04-28 @ $445.30  |  +1.89%")
print("Trade 10: Entry 2016-05-07 @ $462.31  →  Exit 2016-05-11 @ $452.77  |  -2.06%")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
if len(trades) > 124:
    print("❌ Python has MORE trades than TradingView")
    print("   Possible causes:")
    print("   - Entry logic difference")
    print("   - Pyramiding settings")
    print("   - Signal calculation difference")
elif len(trades) < 124:
    print("❌ Python has FEWER trades than TradingView")
    print("   Possible causes:")
    print("   - Entry logic is more restrictive")
    print("   - Different signal interpretation")
    print("   - Data alignment issues")
else:
    print("✅ Trade count matches!")
    print("   Now checking if returns match...")
