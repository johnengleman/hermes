"""
Test if vectorbt is actually using OPEN prices for execution
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# Load data
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

# Filter to small date range for testing
df_test = df[(df.index >= '2016-02-17') & (df.index <= '2016-02-25')].copy()

print("Test Data:")
print(df_test[['Open', 'High', 'Low', 'Close']])
print()

# Create simple signals: buy on 2016-02-18, sell on 2016-02-24
entries = pd.Series(False, index=df_test.index)
exits = pd.Series(False, index=df_test.index)
entries.loc['2016-02-18'] = True
exits.loc['2016-02-24'] = True

print("Signals:")
print(f"Entry signal: {entries[entries].index[0]} (should execute at OPEN of next bar)")
print(f"Exit signal:  {exits[exits].index[0]} (should execute at OPEN of next bar)")
print()

# Test 1: Portfolio with CLOSE only
print("="*80)
print("TEST 1: Using CLOSE prices only")
print("="*80)
port_close = vbt.Portfolio.from_signals(
    close=df_test['Close'],
    entries=entries,
    exits=exits,
    size=1.0,
    size_type=SizeType.Percent,
    init_cash=10000,
    fees=0.0,
    freq='1D'
)

trades_close = port_close.trades.records_readable
if len(trades_close) > 0:
    trade = trades_close.iloc[0]
    print(f"Entry Price:  ${trade['Avg Entry Price']:.2f}")
    print(f"Exit Price:   ${trade['Avg Exit Price']:.2f}")
    print(f"Expected:     Entry at CLOSE of 2016-02-18 ($421.19), Exit at CLOSE of 2016-02-24 ($423.94)")
print()

# Test 2: Portfolio with OPEN parameter
print("="*80)
print("TEST 2: Using OPEN parameter")
print("="*80)
port_open = vbt.Portfolio.from_signals(
    close=df_test['Close'],
    open=df_test['Open'],
    entries=entries,
    exits=exits,
    size=1.0,
    size_type=SizeType.Percent,
    init_cash=10000,
    fees=0.0,
    freq='1D'
)

trades_open = port_open.trades.records_readable
if len(trades_open) > 0:
    trade = trades_open.iloc[0]
    print(f"Entry Price:  ${trade['Avg Entry Price']:.2f}")
    print(f"Exit Price:   ${trade['Avg Exit Price']:.2f}")
    print(f"Expected:     Entry at OPEN of 2016-02-18 ($415.18), Exit at OPEN of 2016-02-24 ($423.13)")
    print()
    print(f"TradingView:  Entry $415.18, Exit $420.32")
print()

# Test 3: Check what prices vectorbt is actually seeing
print("="*80)
print("TEST 3: Check actual data alignment")
print("="*80)
print("2016-02-18 (entry bar):")
print(f"  Open:  ${df_test.loc['2016-02-18', 'Open']:.2f}")
print(f"  Close: ${df_test.loc['2016-02-18', 'Close']:.2f}")
print()
print("2016-02-24 (exit bar):")
print(f"  Open:  ${df_test.loc['2016-02-24', 'Open']:.2f}")
print(f"  Close: ${df_test.loc['2016-02-24', 'Close']:.2f}")
print()

# Check if TradingView might be using next day's open
if '2016-02-19' in df_test.index:
    print("2016-02-19 (day after entry):")
    print(f"  Open:  ${df_test.loc['2016-02-19', 'Open']:.2f}")
    print()
if '2016-02-25' in df_test.index:
    print("2016-02-25 (day after exit):")
    print(f"  Open:  ${df_test.loc['2016-02-25', 'Open']:.2f}")
