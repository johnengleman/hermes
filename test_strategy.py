"""
Quick test to debug position sizing issue
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
from pathlib import Path

from v3.hermes import run_strategy_simple
from v3.strategy_config import MANUAL_DEFAULTS

# Load data
data_path = Path("data/eur_30min.csv")
df = pd.read_csv(data_path)

# Process data
if "time" in df.columns:
    df["date"] = pd.to_datetime(df["time"], unit="s")
elif "timestamp" in df.columns:
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
else:
    df["date"] = pd.to_datetime(df["date"])

df.set_index("date", inplace=True)
df.sort_index(inplace=True)

close = df["close"]
high = df["high"]
low = df["low"]
open_price = df["open"]

print(f"Data loaded: {len(df)} bars")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Run strategy with default parameters
params = MANUAL_DEFAULTS.copy()
print(f"\nParameters:")
print(f"  broker_leverage: {params['broker_leverage']}")
print(f"  risk_per_trade_pct: {params['risk_per_trade_pct']}")

entries, exits, position_target = run_strategy_simple(close, high, low, open_price, **params)

print(f"\nStrategy output:")
print(f"  Entries: {entries.sum()} ({(entries > 0).sum()} longs, {(entries < 0).sum()} shorts)")
print(f"  Exits: {exits.sum()}")
print(f"  Position target (first 10): {position_target.head(10).values}")
print(f"  Position target unique values: {position_target.unique()}")

# Calculate expected value
expected_pct = params['risk_per_trade_pct'] * params['broker_leverage']
print(f"\nExpected position_target: {expected_pct}")
print(f"After /100 for VectorBT: {expected_pct / 100.0}")

# Try running portfolio
long_entries = entries > 0
short_entries = entries < 0
position_size_vbt = position_target / 100.0

print(f"\nVectorBT inputs:")
print(f"  Long entries: {long_entries.sum()}")
print(f"  Short entries: {short_entries.sum()}")
print(f"  Position size (first 10): {position_size_vbt.head(10).values}")

try:
    portfolio = vbt.Portfolio.from_signals(
        close, 
        entries=long_entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=exits,
        size=position_size_vbt,
        size_type=SizeType.Percent,
        init_cash=2000,
        fees=0.0,
        slippage=0.0,
        freq="30min",
        accumulate=False
    )
    
    stats = portfolio.stats()
    print(f"\nPortfolio stats:")
    print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"  Total Trades: {portfolio.trades.count()}")
    print(f"  Win Rate: {portfolio.trades.win_rate():.2%}")
    print(f"  Final Value: ${portfolio.final_value():.2f}")
    
except Exception as e:
    print(f"\n❌ Error creating portfolio: {e}")
    import traceback
    traceback.print_exc()
