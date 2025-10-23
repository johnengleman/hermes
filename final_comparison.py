"""
Final comparison - match TradingView's exact date range
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# Optimizer's best parameters
PARAMS = {
    "short_period": 12,
    "long_period": 377,
    "alma_offset": 0.97,
    "alma_sigma": 9.0,
    "momentum_lookback": 1,
    "use_macro_filter": 1,
    "macro_ema_period": 121,
    "fast_hma_period": 52,
    "slow_ema_period": 59,
    "slow_ema_rising_lookback": 7,
}

# Load ALL data (needed for indicator calculation with 377-day ALMA!)
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

close = df['Close']
high = df['High']
low = df['Low']
open_prices = df['Open']

print(f"Full data range: {close.index[0]} to {close.index[-1]}")
print(f"Total days: {len(close)}\n")

# Run strategy on ALL data
entries, exits, position_size = run_strategy_simple(close, high, low, **PARAMS)

# Create portfolio starting from TradingView's first trade date
TRADINGVIEW_START = '2016-02-18'
print(f"Filtering portfolio to match TradingView start date: {TRADINGVIEW_START}\n")

# Filter signals to start at TradingView date
entries_filtered = entries.copy()
exits_filtered = exits.copy()
entries_filtered.loc[:TRADINGVIEW_START] = False
exits_filtered.loc[:TRADINGVIEW_START] = False

# Create portfolio with filtered signals
portfolio = vbt.Portfolio.from_signals(
    close=close,
    price=open_prices,
    entries=entries_filtered,
    exits=exits_filtered,
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
print("FINAL COMPARISON")
print("="*80)
print(f"Python Trades:          {len(trades)}")
print(f"TradingView Trades:     124")
print(f"Match:                  {'✅ YES' if len(trades) == 124 else '❌ NO'}")
print()
print(f"Python Return:          {portfolio.total_return()*100:,.1f}%")
print(f"TradingView Return:     114,720%")
print(f"Difference:             {portfolio.total_return()*100 - 114720:.1f}%")
print()

# Show first 10 trades
print("="*80)
print("FIRST 10 TRADES COMPARISON")
print("="*80)
for i in range(min(10, len(trades))):
    p_trade = trades.iloc[i]
    p_entry_date = p_trade['Entry Timestamp'].strftime('%Y-%m-%d')
    p_exit_date = p_trade['Exit Timestamp'].strftime('%Y-%m-%d')
    p_entry_price = p_trade['Avg Entry Price']
    p_exit_price = p_trade['Avg Exit Price']

    print(f"\nTrade {i+1}:")
    print(f"  Python:      Entry {p_entry_date} @ ${p_entry_price:.2f}, Exit {p_exit_date} @ ${p_exit_price:.2f}")

    # Expected TradingView values (from CSV)
    if i == 0:
        print(f"  TradingView: Entry 2016-02-18 @ $415.18, Exit 2016-02-24 @ $420.32")
    elif i == 1:
        print(f"  TradingView: Entry 2016-02-27 @ $430.84, Exit 2016-03-02 @ $433.03")
    elif i == 2:
        print(f"  TradingView: Entry 2016-03-08 @ $412.21, Exit 2016-03-13 @ $410.56")

print("\n" + "="*80)
if len(trades) == 124 and abs(portfolio.total_return()*100 - 114720) < 1000:
    print("✅ SUCCESS! Python now matches TradingView!")
else:
    print("⚠️  Close, but still some differences to investigate")
print("="*80)
