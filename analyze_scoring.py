"""
Analyze why the optimizer prefers worse parameters
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple, compute_composite_score
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# Optimizer's "best" parameters
OPTIMIZER_BEST = {
    "short_period": 10,
    "long_period": 218,
    "alma_offset": 0.97,
    "alma_sigma": 6.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,
    "macro_ema_period": 143,
    "fast_hma_period": 38,
    "slow_ema_period": 147,
    "slow_ema_rising_lookback": 13,
}

# User's better parameters
USER_BETTER = {
    "short_period": 10,
    "long_period": 218,
    "alma_offset": 0.97,
    "alma_sigma": 6.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,
    "macro_ema_period": 143,
    "fast_hma_period": 38,
    "slow_ema_period": 59,
    "slow_ema_rising_lookback": 1,
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

def analyze_params(params, name):
    """Analyze a parameter set"""
    print("\n" + "="*80)
    print(f"{name}")
    print("="*80)

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

    stats = portfolio.stats()

    # Compute optimizer's scoring
    training_days = len(close)
    score, components = compute_composite_score(portfolio, stats, params, training_days)

    # Display metrics
    print(f"\nRaw Metrics:")
    print(f"  Total Return:       {portfolio.total_return()*100:>12,.1f}%")
    print(f"  Sortino Ratio:      {components['sortino_raw']:>12.2f}")
    print(f"  Calmar Ratio:       {components.get('calmar_ratio', 0):>12.2f}")
    print(f"  Max Drawdown:       {stats.get('Max Drawdown [%]', 0):>12.1f}%")
    print(f"  Total Trades:       {portfolio.trades.count():>12}")
    print(f"  Trades/Year:        {components['trades_per_year']:>12.1f}")
    print(f"  Win Rate:           {components['win_rate']*100:>12.1f}%")

    print(f"\nOptimizer Scoring:")
    print(f"  Composite Score:    {score:>12.4f}")
    print(f"  Sortino Component:  {components['sortino_raw'] * 0.50:>12.4f} (50% weight)")
    print(f"  Calmar Component:   {components.get('calmar_ratio', 0) * 0.20:>12.4f} (20% weight)")

    return score

# Analyze both
opt_score = analyze_params(OPTIMIZER_BEST, "OPTIMIZER'S BEST")
user_score = analyze_params(USER_BETTER, "USER'S BETTER")

# Summary
print("\n" + "="*80)
print("WHY IS THE OPTIMIZER CHOOSING WORSE PARAMETERS?")
print("="*80)
print(f"\nOptimizer's Score: {opt_score:.4f}")
print(f"User's Score:      {user_score:.4f}")
print()

if opt_score > user_score:
    print(f"🚨 PROBLEM: Optimizer scores its params HIGHER ({opt_score:.4f} > {user_score:.4f})")
    print(f"   The scoring function prefers worse absolute returns!")
    print(f"   Solution: Reduce weight on risk-adjusted metrics, increase weight on absolute returns")
else:
    print(f"✅ Optimizer correctly scores user's params higher")
    print(f"   The search algorithm just didn't explore this region")
    print(f"   Solution: Better initialization or wider search")
