"""
Generic Backtesting Tool
Runs a single backtest with default parameters to verify strategy implementation

USAGE:
    python backtest.py

CONFIGURATION:
    Edit the top of this file to configure which strategy to test:
    
    STRATEGY_MODULE = "v1.hermes"              # Python module path
    STRATEGY_FUNCTION = "run_strategy_simple"  # Function name
    STRATEGY_CONFIG = "v1.strategy_config"     # Config module with MANUAL_DEFAULTS
    
    ASSET_NAME = "BTC"                         # Asset name for output file
    DATA_FILE = Path("data/btc_daily.csv")     # Data file path
    CAPITAL_BASE = 10000                       # Initial capital
    START_DATE = "2013-01-01"                  # Optional: filter from this date

PURPOSE:
    This tool verifies that the Python implementation produces the same results
    as the TradingView Pine Script implementation. It's completely generic and
    works with any strategy that follows the pattern:
    
        run_strategy(close, high, low, **params) -> (entries, exits, position_target)

OUTPUT:
    - Detailed performance metrics printed to console
    - CSV file with results: {ASSET_NAME}_backtest_results.csv
    - Comparison checklist for TradingView verification
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Strategy to test (change this to test different strategies)
STRATEGY_MODULE = "v1.hermes"          # Python module path (e.g., "v1.hermes", "v2.hermes_complex")
STRATEGY_FUNCTION = "run_strategy_simple"  # Function name in the module
STRATEGY_CONFIG = "v1.strategy_config"     # Config module with MANUAL_DEFAULTS

# Backtest settings
CAPITAL_BASE = 10000  # Match TradingView default initial_capital
START_DATE = "2013-01-01"  # Optional: filter data from this date

# Asset data source
ASSET_NAME = "BTC"
DATA_FILE = Path("data/btc_daily.csv")


# ============================================================================
# DATA LOADING
# ============================================================================

def _standardize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize price data to common format"""
    if df.empty:
        raise ValueError("price dataframe is empty")

    cols = {col.lower(): col for col in df.columns}

    def _get_column(possible):
        for key in possible:
            if key in cols:
                return df[cols[key]]
        return None

    time_col = _get_column(["time", "timestamp", "date"])
    if time_col is None:
        raise ValueError("price file missing time column")

    if np.issubdtype(time_col.dtype, np.number):
        raw_dt = pd.to_datetime(time_col, unit="s", utc=True, errors="coerce")
    else:
        raw_dt = pd.to_datetime(time_col, utc=True, errors="coerce")

    try:
        dt_index = pd.DatetimeIndex(raw_dt)
    except Exception as err:
        raise ValueError("unable to create datetime index from time column") from err

    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_convert(None)

    close_col = _get_column(["close", "settle", "price"])
    if close_col is None:
        raise ValueError("price file missing close column")

    open_col = _get_column(["open"])
    high_col = _get_column(["high"])
    low_col = _get_column(["low"])
    volume_col = _get_column(["volume", "vol"])

    open_series = open_col if open_col is not None else close_col
    high_series = high_col if high_col is not None else close_col
    low_series = low_col if low_col is not None else close_col

    standardized = pd.DataFrame(
        {
            "time": (dt_index.view("int64") // 10 ** 9).astype(np.int64),
            "open": open_series.astype(float).to_numpy(),
            "high": high_series.astype(float).to_numpy(),
            "low": low_series.astype(float).to_numpy(),
            "close": close_col.astype(float).to_numpy(),
        }
    )

    if volume_col is not None:
        standardized["volume"] = volume_col.astype(float).fillna(0.0).to_numpy()

    standardized = standardized.sort_values("time")
    standardized = standardized.drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    return standardized


def load_data(file_path: Path) -> pd.DataFrame:
    """Load and prepare data for backtesting"""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = _standardize_price_frame(pd.read_csv(file_path))
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    if START_DATE:
        df = df[df["time"] >= START_DATE].copy()
    
    df.set_index("time", inplace=True)
    df = df.sort_index()
    
    return df


# ============================================================================
# BACKTEST EXECUTION
# ============================================================================

def run_backtest():
    """
    Run a single backtest with default parameters.
    Completely generic - works with any strategy that follows the pattern:
    
    run_strategy_function(close, high, low, **params) -> (entries, exits, position_target)
    """
    
    print("\n" + "="*70)
    print("GENERIC STRATEGY BACKTEST")
    print("="*70)
    
    # Dynamically import strategy components
    print(f"\n📦 Loading strategy...")
    print(f"   Module:   {STRATEGY_MODULE}")
    print(f"   Function: {STRATEGY_FUNCTION}")
    print(f"   Config:   {STRATEGY_CONFIG}")
    
    try:
        # Import strategy function
        strategy_module = __import__(STRATEGY_MODULE, fromlist=[STRATEGY_FUNCTION])
        run_strategy = getattr(strategy_module, STRATEGY_FUNCTION)
        
        # Import configuration
        config_module = __import__(STRATEGY_CONFIG, fromlist=["MANUAL_DEFAULTS", "STRATEGY_NAME"])
        MANUAL_DEFAULTS = config_module.MANUAL_DEFAULTS
        STRATEGY_NAME = getattr(config_module, "STRATEGY_NAME", "Unknown Strategy")
        
    except Exception as e:
        print(f"\n❌ Error loading strategy: {e}")
        sys.exit(1)
    
    print(f"   ✓ Strategy loaded: {STRATEGY_NAME}")
    
    # Load data
    print(f"\n📊 Loading data...")
    print(f"   Asset: {ASSET_NAME}")
    print(f"   File:  {DATA_FILE}")
    
    try:
        data = load_data(DATA_FILE)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)
    
    start_date = data.index[0]
    end_date = data.index[-1]
    print(f"   ✓ Loaded {len(data)} bars")
    print(f"   Range: {start_date.date()} → {end_date.date()}")
    
    if len(data) < 150:
        print(f"\n⚠️  Insufficient data: only {len(data)} bars")
        sys.exit(1)
    
    # Display parameters being used
    print(f"\n📋 Default Parameters:")
    print("   " + "-"*66)
    for key, value in sorted(MANUAL_DEFAULTS.items()):
        # Format percentages nicely
        if isinstance(value, float) and 0 < value < 1 and "rate" in key.lower():
            print(f"   {key:30s} {value*100:6.3f}%")
        else:
            print(f"   {key:30s} {value}")
    print("   " + "-"*66)
    
    # Extract price data
    close = data["close"]
    high = data["high"]
    low = data["low"]
    
    # Run strategy (completely generic!)
    print(f"\n🔄 Running strategy...")
    try:
        entries, exits, position_target = run_strategy(close, high, low, **MANUAL_DEFAULTS)
    except Exception as e:
        print(f"\n❌ Error running strategy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    num_entries = entries.sum()
    num_exits = exits.sum()
    print(f"   ✓ Strategy executed")
    print(f"   Entry signals:  {num_entries}")
    print(f"   Exit signals:   {num_exits}")
    
    # Create portfolio
    print(f"\n💼 Creating portfolio...")
    print(f"   Initial capital: ${CAPITAL_BASE:,.0f}")
    print(f"   Commission:      {MANUAL_DEFAULTS.get('commission_rate', 0)*100:.3f}%")
    print(f"   Slippage:        {MANUAL_DEFAULTS.get('slippage_rate', 0)*100:.3f}%")
    
    # Split entries into long and short signals for VectorBT
    long_entries = entries > 0
    short_entries = entries < 0
    
    portfolio = vbt.Portfolio.from_signals(
        close, 
        long_entries, 
        short_entries,
        size=position_target,
        size_type=SizeType.Percent,  # 1.0 = 100% of cash
        init_cash=CAPITAL_BASE,
        fees=MANUAL_DEFAULTS.get("commission_rate", 0.0),
        slippage=MANUAL_DEFAULTS.get("slippage_rate", 0.0),
        freq="1D",
        accumulate=False  # Match Pine Script pyramiding=1 (no position stacking)
    )
    
    # Calculate statistics
    stats = portfolio.stats()
    
    # Extract metrics
    total_return = stats.get("Total Return [%]", 0)
    max_dd = stats.get("Max Drawdown [%]", 0) / 100
    sharpe = stats.get("Sharpe Ratio", np.nan)
    sortino = stats.get("Sortino Ratio", np.nan)
    num_trades = portfolio.trades.count()
    win_rate = portfolio.trades.win_rate() if num_trades > 0 else 0
    profit_factor = stats.get("Profit Factor", 0)
    
    # Calculate additional metrics
    years = len(close) / 365
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    calmar = (annualized_return / 100) / max_dd if max_dd > 0 else 0
    trades_per_year = num_trades / years if years > 0 else 0
    
    # Get trade details
    trades = portfolio.trades.records_readable
    
    # Display results
    print(f"\n{'='*70}")
    print("✅ BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"\n📈 RETURNS:")
    print(f"   Total Return:          {total_return:+.2f}%")
    print(f"   Annualized Return:     {annualized_return:+.2f}%")
    print(f"   Max Drawdown:          {max_dd*100:.2f}%")
    print(f"\n📊 RISK-ADJUSTED METRICS:")
    print(f"   Sharpe Ratio:          {sharpe:.3f}")
    print(f"   Sortino Ratio:         {sortino:.3f}")
    print(f"   Calmar Ratio:          {calmar:.3f}")
    print(f"\n💼 TRADING STATISTICS:")
    print(f"   Total Trades:          {num_trades}")
    print(f"   Trades per Year:       {trades_per_year:.1f}")
    print(f"   Win Rate:              {win_rate*100:.2f}%")
    print(f"   Profit Factor:         {profit_factor:.3f}")
    
    if num_trades > 0:
        avg_win = stats.get("Avg Winning Trade [%]", 0)
        avg_loss = stats.get("Avg Losing Trade [%]", 0)
        print(f"   Avg Winning Trade:     {avg_win:.2f}%")
        print(f"   Avg Losing Trade:      {avg_loss:.2f}%")
    
    print(f"\n⏱️  PERIOD:")
    print(f"   Start Date:            {start_date.date()}")
    print(f"   End Date:              {end_date.date()}")
    print(f"   Days:                  {len(close)}")
    print(f"   Years:                 {years:.2f}")
    print(f"{'='*70}")
    
    # Show sample trades
    if not trades.empty and len(trades) > 0:
        print(f"\n📝 FIRST 10 TRADES (for verification):")
        print(f"{'='*70}")
        for idx, trade in trades.head(10).iterrows():
            entry_date = pd.to_datetime(trade['Entry Timestamp']).date()
            exit_date = pd.to_datetime(trade['Exit Timestamp']).date()
            pnl_pct = trade['Return'] * 100
            entry_price = trade['Avg Entry Price']
            exit_price = trade['Avg Exit Price']
            print(f"   #{idx+1:2d}: {entry_date} → {exit_date} | "
                  f"${entry_price:7.2f} → ${exit_price:7.2f} | "
                  f"Return: {pnl_pct:+6.2f}%")
        
        if len(trades) > 10:
            print(f"   ... and {len(trades) - 10} more trades")
        
        if len(trades) > 10:
            print(f"   ... and {len(trades) - 10} more trades")
    
    # Save detailed results
    output_dir = Path("v1")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{ASSET_NAME.lower()}_backtest_results.csv"
    
    result = pd.DataFrame([{
        "asset": ASSET_NAME,
        "strategy": STRATEGY_NAME,
        "start_date": start_date,
        "end_date": end_date,
        "days": len(close),
        "years": years,
        "total_return": total_return / 100,
        "annualized_return": annualized_return / 100,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "num_trades": num_trades,
        "trades_per_year": trades_per_year,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "initial_capital": CAPITAL_BASE,
    }])
    
    result.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("COMPARISON CHECKLIST FOR TRADINGVIEW:")
    print(f"{'='*70}")
    print(f"1. Total Return:       {total_return:+.2f}%")
    print(f"2. Number of Trades:   {num_trades}")
    print(f"3. Win Rate:           {win_rate*100:.2f}%")
    print(f"4. Profit Factor:      {profit_factor:.3f}")
    print(f"5. Max Drawdown:       {max_dd*100:.2f}%")
    print(f"\nIf these match TradingView, the implementations are equivalent!")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_backtest()
