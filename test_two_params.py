"""
Test both parameter sets to see what Python returns vs TradingView
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# Optimizer's "best" parameters (23,000% in TradingView)
OPTIMIZER_BEST = {
    "short_period": 10,
    "long_period": 218,
    "alma_offset": 0.97,
    "alma_sigma": 6.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,  # DISABLED
    "macro_ema_period": 143,
    "fast_hma_period": 38,
    "slow_ema_period": 147,  # <-- Different!
    "slow_ema_rising_lookback": 13,  # <-- Different!
}

# User's better parameters (213,000% in TradingView!!!)
USER_BETTER = {
    "short_period": 10,
    "long_period": 218,
    "alma_offset": 0.97,
    "alma_sigma": 6.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,  # DISABLED
    "macro_ema_period": 143,
    "fast_hma_period": 38,
    "slow_ema_period": 59,  # <-- Different!
    "slow_ema_rising_lookback": 1,  # <-- Different!
}

# Load data
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

close = df['Close']
high = df['High']
low = df['Low']
open_prices = df['Open']

def test_params(params, name, expected_tv_return):
    """Test a parameter set"""
    print("\n" + "="*80)
    print(f"{name}")
    print("="*80)
    print("Parameters:")
    print(f"  slow_ema_period:           {params['slow_ema_period']}")
    print(f"  slow_ema_rising_lookback:  {params['slow_ema_rising_lookback']}")
    print()

    # Run strategy
    entries, exits, position_size = run_strategy_simple(close, high, low, **params)

    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        price=open_prices,
        entries=entries,
        exits=exits,
        size=1.0,
        size_type=SizeType.Percent,
        init_cash=10000,
        fees=0.0,
        slippage=0.0,
        freq='1D'
    )

    trades = portfolio.trades.records_readable
    python_return = portfolio.total_return() * 100

    print(f"RESULTS:")
    print(f"  Python Return:      {python_return:>12,.1f}%")
    print(f"  TradingView Return: {expected_tv_return:>12,.1f}%")
    print(f"  Difference:         {python_return - expected_tv_return:>12,.1f}%")
    print(f"  Total Trades:       {len(trades):>12}")

    return python_return, len(trades)

# Test both
opt_return, opt_trades = test_params(OPTIMIZER_BEST, "OPTIMIZER'S BEST PARAMS", 23000)
user_return, user_trades = test_params(USER_BETTER, "USER'S BETTER PARAMS", 213000)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nOptimizer found EMA={OPTIMIZER_BEST['slow_ema_period']}, Rising={OPTIMIZER_BEST['slow_ema_rising_lookback']}")
print(f"  Python:      {opt_return:,.1f}%")
print(f"  TradingView: 23,000%")
print(f"  Ratio:       {opt_return/23000:.2f}x")

print(f"\nUser found EMA={USER_BETTER['slow_ema_period']}, Rising={USER_BETTER['slow_ema_rising_lookback']}")
print(f"  Python:      {user_return:,.1f}%")
print(f"  TradingView: 213,000%")
print(f"  Ratio:       {user_return/213000:.2f}x")

print(f"\n🔥 User's params are {user_return/opt_return:.2f}x better in Python")
print(f"🔥 User's params are {213000/23000:.2f}x better in TradingView")

if user_return > opt_return:
    print(f"\n⚠️  CRITICAL: Python optimizer should have found user's params!")
    print(f"   The optimizer is NOT finding the global optimum.")
else:
    print(f"\n✅ Python correctly prefers optimizer params")
    print(f"   The issue is that TradingView prefers different params")
    print(f"   This suggests an implementation difference!")
