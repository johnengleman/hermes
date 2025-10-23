"""
Debug script to output detailed trade log for comparison with TradingView
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# Your MANUAL parameters that should match TradingView
MANUAL_PARAMS = {
    "short_period": 30,
    "long_period": 250,
    "alma_offset": 0.95,
    "alma_sigma": 4.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,
    "macro_ema_period": 100,
    "fast_hma_period": 30,
    "slow_ema_period": 80,
    "slow_ema_rising_lookback": 3,
}

# Load BTC data
print("Loading BTC data...")
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

close = df['Close']
high = df['High']
low = df['Low']

print(f"Data range: {close.index[0]} to {close.index[-1]}")
print(f"Total days: {len(close)}\n")

# Run strategy
print("Running strategy...")
entries, exits, position_size = run_strategy_simple(close, high, low, **MANUAL_PARAMS)

# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=1.0,
    size_type=SizeType.Percent,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq='1D'
)

# Get detailed trade log
trades = portfolio.trades.records_readable

print("="*80)
print("DETAILED TRADE LOG")
print("="*80)
print(f"Total Trades: {len(trades)}\n")

# Format and display trades
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# First, let's see what columns we have
print("Available columns:", trades.columns.tolist())
print()

# Select key columns (use actual column names)
trade_summary = trades

print(trade_summary.to_string())

# Summary statistics
print("\n" + "="*80)
print("TRADE STATISTICS")
print("="*80)
print(f"Total Return:        {portfolio.total_return()*100:,.1f}%")
print(f"Final Value:         ${portfolio.final_value():,.2f}")
print(f"Total Trades:        {len(trades)}")

# Export to CSV for easy comparison
trade_summary.to_csv('python_trades_detailed.csv', index=False)
print(f"\n✅ Trade log exported to 'python_trades_detailed.csv'")
print(f"   Compare this with TradingView's trade list!")
