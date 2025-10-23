"""
Create detailed trade-by-trade comparison CSV
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

# Load data
df = pd.read_csv("btc_daily.csv")
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df.columns = [col.capitalize() for col in df.columns]

close = df['Close']
high = df['High']
low = df['Low']
open_price = df['Open']

# Run strategy
entries, exits, position_size = run_strategy_simple(close, high, low, **PARAMS)

# Create portfolio
portfolio = vbt.Portfolio.from_signals(
    close=close,
    open=open_price,
    entries=entries,
    exits=exits,
    size=1.0,
    size_type=SizeType.Percent,
    init_cash=10000,
    fees=0.0,
    slippage=0.0,
    freq='1D'
)

# Get Python trades
python_trades = portfolio.trades.records_readable

# Filter to 2016 onwards for easier comparison
python_trades_2016 = python_trades[python_trades['Entry Timestamp'] >= '2016-01-01'].reset_index(drop=True)

# Export for comparison
python_trades_2016.to_csv('python_trades_comparison.csv', index=False)
print(f"✅ Exported {len(python_trades_2016)} Python trades to python_trades_comparison.csv")

# Load TradingView trades
tv_trades = pd.read_csv('Hermes_SIMPLE_Strategy_COINBASE_BTCUSD_2025-10-22.csv')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Python trades (2016+):    {len(python_trades_2016)}")
print(f"TradingView trades:       {len(tv_trades)//2}")  # Each trade has entry + exit row
print(f"Python total return:      {portfolio.total_return()*100:,.1f}%")
print(f"TradingView return:       114,720%")

print("\n" + "="*80)
print("SIDE-BY-SIDE COMPARISON (first 20 trades)")
print("="*80)

# Show first 20 trades side by side
for i in range(min(20, len(python_trades_2016))):
    p_trade = python_trades_2016.iloc[i]

    p_entry_date = p_trade['Entry Timestamp'].strftime('%Y-%m-%d')
    p_exit_date = p_trade['Exit Timestamp'].strftime('%Y-%m-%d')
    p_entry_price = p_trade['Avg Entry Price']
    p_exit_price = p_trade['Avg Exit Price']

    print(f"\nPython Trade {i+1}:")
    print(f"  Entry: {p_entry_date} @ ${p_entry_price:.2f}")
    print(f"  Exit:  {p_exit_date} @ ${p_exit_price:.2f}")

    if i < len(tv_trades)//2:
        tv_entry_idx = i * 2 + 1  # Entry rows are odd (1, 3, 5...)
        tv_exit_idx = i * 2       # Exit rows are even (0, 2, 4...)

        tv_entry = tv_trades.iloc[tv_entry_idx]
        tv_exit = tv_trades.iloc[tv_exit_idx]

        tv_entry_date = tv_entry['Date/Time']
        tv_exit_date = tv_exit['Date/Time']
        tv_entry_price = tv_entry['Price USD']
        tv_exit_price = tv_exit['Price USD']

        print(f"TradingView Trade {i+1}:")
        print(f"  Entry: {tv_entry_date} @ ${tv_entry_price:.2f}")
        print(f"  Exit:  {tv_exit_date} @ ${tv_exit_price:.2f}")

        # Check if dates match
        if p_entry_date == tv_entry_date and p_exit_date == tv_exit_date:
            print(f"  ✅ Dates MATCH")
        else:
            print(f"  ❌ Dates DIFFER")
