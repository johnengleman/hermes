"""
Test script to compare optimizer's best parameters vs manual parameters
"""
import pandas as pd
import numpy as np
from optimize_hermes import run_strategy_simple
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# ============================================================================
# PARAMETER SETS TO COMPARE
# ============================================================================

# User's MANUAL parameters (reported 150,000% return in TradingView)
MANUAL_PARAMS = {
    "short_period": 30,
    "long_period": 250,
    "alma_offset": 0.95,
    "alma_sigma": 4.0,
    "momentum_lookback": 1,
    "use_macro_filter": 0,  # DISABLED - KEY TO HIGH RETURNS
    "macro_ema_period": 100,
    "fast_hma_period": 30,
    "slow_ema_period": 80,
    "slow_ema_rising_lookback": 3,
}

# Optimizer's BEST parameters (reported 47,429% return in Python)
OPTIMIZER_BEST = {
    "short_period": 33,
    "long_period": 294,
    "alma_offset": 0.86,
    "alma_sigma": 6.0,
    "momentum_lookback": 3,
    "use_macro_filter": 0,  # DISABLED
    "macro_ema_period": 100,
    "fast_hma_period": 16,
    "slow_ema_period": 97,
    "slow_ema_rising_lookback": 11,
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading BTC data...")
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Capitalize column names to match expected format
df.columns = [col.capitalize() for col in df.columns]

close = df['Close']
high = df['High']
low = df['Low']

print(f"Data range: {close.index[0]} to {close.index[-1]}")
print(f"Total days: {len(close)}")

# ============================================================================
# RUN BACKTESTS
# ============================================================================

def run_backtest(params, name):
    """Run backtest with given parameters and print results"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    # Print parameters
    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key:30s}: {value}")

    # Run strategy
    entries, exits, position_size = run_strategy_simple(close, high, low, **params)

    # Create portfolio (ZERO fees to match TradingView defaults)
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        size=1.0,  # 100% of equity
        size_type=SizeType.Percent,
        init_cash=10000,
        fees=0.0,  # TradingView default: no commission
        slippage=0.0,  # TradingView default: no slippage
        freq='1D'
    )

    # Get statistics
    stats = portfolio.stats()
    total_return = portfolio.total_return()

    # Calculate additional metrics
    num_trades = portfolio.trades.count()
    years = len(close) / 365
    trades_per_year = num_trades / years

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"{'='*70}")
    print(f"Total Return:        {total_return*100:>12.1f}%")
    print(f"Final Value:         ${portfolio.final_value():>12,.2f}")
    print(f"Total Trades:        {num_trades:>12.0f}")
    print(f"Trades/Year:         {trades_per_year:>12.1f}")

    if 'Sharpe Ratio' in stats:
        print(f"Sharpe Ratio:        {stats['Sharpe Ratio']:>12.2f}")
    if 'Sortino Ratio' in stats:
        print(f"Sortino Ratio:       {stats['Sortino Ratio']:>12.2f}")
    if 'Max Drawdown [%]' in stats:
        print(f"Max Drawdown:        {stats['Max Drawdown [%]']:>12.1f}%")
    if 'Win Rate [%]' in stats:
        print(f"Win Rate:            {stats['Win Rate [%]']:>12.1f}%")

    return portfolio, stats, total_return

# ============================================================================
# RUN COMPARISONS
# ============================================================================

print("\n" + "="*70)
print("PARAMETER COMPARISON TEST")
print("="*70)

# Test manual parameters
manual_portfolio, manual_stats, manual_return = run_backtest(MANUAL_PARAMS, "MANUAL (User's 150k% TradingView)")

# Test optimizer's best
optimizer_portfolio, optimizer_stats, optimizer_return = run_backtest(OPTIMIZER_BEST, "OPTIMIZER BEST (47k% Python)")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\nManual Return:     {manual_return*100:>12.1f}%")
print(f"Optimizer Return:  {optimizer_return*100:>12.1f}%")
print(f"Difference:        {(manual_return - optimizer_return)*100:>12.1f}%")

if manual_return > optimizer_return:
    ratio = manual_return / optimizer_return
    print(f"\n🎯 Manual parameters are {ratio:.2f}x better than optimizer's best!")
    print(f"\n💡 This suggests the optimizer may be:")
    print(f"   1. Stuck in a local optimum")
    print(f"   2. Over-optimizing for risk-adjusted metrics vs absolute returns")
    print(f"   3. Not exploring the right parameter region")
else:
    ratio = optimizer_return / manual_return
    print(f"\n🤔 Optimizer found {ratio:.2f}x better parameters!")
    print(f"\n💡 This suggests:")
    print(f"   1. TradingView and Python implementations may differ")
    print(f"   2. Different date ranges or data sources")
    print(f"   3. Different fee/slippage calculations")

print("\n" + "="*70)
print("Key Parameter Differences:")
print("="*70)
print(f"{'Parameter':<30} {'Manual':>15} {'Optimizer':>15} {'Diff':>10}")
print("-"*70)
for key in MANUAL_PARAMS.keys():
    manual_val = MANUAL_PARAMS[key]
    opt_val = OPTIMIZER_BEST[key]
    if isinstance(manual_val, (int, float)) and isinstance(opt_val, (int, float)):
        if isinstance(manual_val, float):
            diff = opt_val - manual_val
            print(f"{key:<30} {manual_val:>15.2f} {opt_val:>15.2f} {diff:>10.2f}")
        else:
            diff = opt_val - manual_val
            print(f"{key:<30} {manual_val:>15} {opt_val:>15} {diff:>10}")
    else:
        print(f"{key:<30} {str(manual_val):>15} {str(opt_val):>15} {'':>10}")

print("\n" + "="*70)
