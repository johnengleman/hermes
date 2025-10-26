"""
Hermes Strategy Optimizer - Generic Optimizer
Genetic algorithm optimization that works with any strategy
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
from scipy.optimize import differential_evolution
from pathlib import Path
import warnings
import datetime

# Import strategy-specific components
from v1.hermes import run_strategy_simple
from v1.strategy_config import (
    MANUAL_DEFAULTS,
    NUM_PARAMETERS,
    get_optimization_bounds,
    decode_parameters,
    validate_parameters,
)

warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

# Create output directories
Path("reports").mkdir(exist_ok=True)
Path("reports/quantstats").mkdir(exist_ok=True)
Path("reports/heatmaps").mkdir(exist_ok=True)

# ============================================================================
# GENERIC OPTIMIZER CONFIGURATION
# ============================================================================

# Optimization Settings
QUICK_MODE = True
GENETIC_POPULATION_SIZE = 25  # Increased from 20 for more diversity
OPTIMIZATION_METHOD = "genetic"
GENETIC_QUICK_MAX_ITER = 100   # Quick mode
GENETIC_MAX_ITERATIONS = 150   # Full mode

CAPITAL_BASE = 150000

# Asset data sources
ASSET_DATA_SOURCES = {
    "BTC": {
        "primary": Path("data/btc_daily.csv"),
        "proxies": [
            Path("data/blx_daily.csv"),
            Path("data/cme_btc_daily.csv"),
        ],
    },
    "ETH": {
        "primary": Path("data/eth_daily.csv"),
        "proxies": [
            Path("data/eth_daily_proxy.csv"),
        ],
    },
}


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def compute_composite_score(portfolio, stats, params, training_days):
    """
    Return-focused composite score with risk modifiers.
    
    Philosophy: Prioritize absolute returns while penalizing excessive risk.
    A strategy with 2x the returns should ALWAYS score higher, even with more risk.
    
    Returns:
        Tuple of (score, components_dict)
    """
    sortino = stats.get("Sortino Ratio", 0)
    if np.isnan(sortino) or np.isinf(sortino):
        sortino = 0

    total_return = portfolio.total_return()
    if np.isnan(total_return) or np.isinf(total_return):
        total_return = 0

    max_dd = stats.get("Max Drawdown [%]", 0) / 100
    if max_dd == 0:
        max_dd = 0.01

    num_trades = portfolio.trades.count()
    win_rate = portfolio.trades.win_rate() if num_trades > 0 else 0
    if np.isnan(win_rate):
        win_rate = 0

    trades_per_year = (num_trades / training_days) * 365
    years = training_days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Hard Constraints (return 0 if violated)
    if trades_per_year < 3:
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, 
                     "constraint_violation": "too_few_trades",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}
    
    if annualized_return < 0.10:  # Minimum 10% annualized return
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, 
                     "constraint_violation": "insufficient_return",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}
    
    if trades_per_year > 200:  # Avoid over-trading
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, 
                     "constraint_violation": "excessive_trading",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}

    # === RETURN-FIRST SCORING ===
    
    # 1. Annualized Return (60% weight) - PRIMARY OBJECTIVE
    #    Log scale to handle wide range (10% to 1000%+)
    #    Score = log(1 + return) / log(11) to normalize 10x return = 1.0
    return_score = min(np.log(1 + annualized_return) / np.log(11), 1.0) * 0.60
    
    # 2. Sortino Ratio (25% weight) - Risk-adjusted quality
    #    Measures how efficiently we generate returns vs downside risk
    sortino_score = min(sortino / 3.0, 1.0) * 0.25
    
    # 3. Drawdown Modifier (10% weight) - Soft penalty, not hard constraint
    #    We accept drawdown for high returns, but reward lower DD
    calmar = annualized_return / max_dd
    calmar_score = min(calmar / 3.0, 1.0) * 0.10
    
    # 4. Consistency Bonus (5% weight) - Minor bonus for good habits
    #    Trade frequency and win rate bonuses
    if 5 <= trades_per_year <= 50:
        freq_score = 1.0
    elif 3 <= trades_per_year < 5:
        freq_score = 0.7
    elif 50 < trades_per_year <= 100:
        freq_score = 0.8
    else:
        freq_score = 0.5
    
    if 0.30 <= win_rate <= 0.65:
        winrate_score = 1.0
    elif 0.20 <= win_rate < 0.30:
        winrate_score = 0.7
    elif 0.65 < win_rate <= 0.80:
        winrate_score = 0.8
    else:
        winrate_score = 0.6
    
    consistency_score = (freq_score * 0.6 + winrate_score * 0.4) * 0.05
    
    # === FINAL COMPOSITE SCORE ===
    # Returns dominate (60%), risk-adjustment is secondary (35%), consistency is bonus (5%)
    composite = return_score + sortino_score + calmar_score + consistency_score

    return composite, {
        "sortino_raw": sortino,
        "composite_score": composite,
        "return_score": return_score,
        "sortino_score": sortino_score,
        "calmar_score": calmar_score,
        "consistency_score": consistency_score,
        "calmar_ratio": calmar,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "trades_per_year": trades_per_year,
        "win_rate": win_rate,
        "constraint_violation": None,
    }


def optimize_parameters_genetic(data, start_date, end_date, max_iterations=60):
    """
    Generic genetic algorithm optimization using Differential Evolution.
    
    Works with any strategy that provides:
    - get_optimization_bounds(): Returns bounds array
    - decode_parameters(x): Converts optimizer array to params dict
    - validate_parameters(params): Checks strategy-specific constraints
    
    This is a TRUE genetic algorithm that runs population-based search.
    With popsize=15 and 9 parameters: 15×9 = 135 individuals per generation.
    60 iterations = 8,100 total evaluations (~10-15 minutes)
    150 iterations = 20,250 total evaluations (~25-40 minutes)
    """
    close = data.loc[start_date:end_date, "close"]
    high = data.loc[start_date:end_date, "high"]
    low = data.loc[start_date:end_date, "low"]

    if len(close) < 150:
        print(f"  ⚠ Insufficient data: only {len(close)} bars — skipping")
        return None

    training_days = len(close)
    
    # Get bounds from strategy config (completely generic!)
    bounds = get_optimization_bounds()
    
    population_size = GENETIC_POPULATION_SIZE * len(bounds)
    start_time = datetime.datetime.now()
    
    print(f"  ▶ [{start_time.strftime('%H:%M:%S')}] Genetic optimization on {start_date.date()}–{end_date.date()}")
    print(f"     {training_days} bars | pop={population_size} | max_iter={max_iterations}")
    print(f"     Expected evaluations: ~{population_size * max_iterations:,}")

    call_count = [0]
    best_score = [float("-inf")]
    best_sortino = [0.0]
    best_return = [0.0]
    best_trades_per_year = [0.0]
    best_x = [None]  # Track the actual best parameters array
    generation = [0]
    last_logged_gen = [-1]

    def objective(x):
        """Objective function for differential evolution."""
        call_count[0] += 1
        current_gen = call_count[0] // population_size
        
        # Enforce max iterations manually
        if current_gen > max_iterations:
            return 999.0
        
        # Log only once per generation (at the end)
        if current_gen > generation[0] and generation[0] > last_logged_gen[0]:
            last_logged_gen[0] = generation[0]
            now = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"    [{now}] Gen {generation[0]}/{max_iterations} complete | "
                  f"Best Score: {best_score[0]:.2f} | Sortino: {best_sortino[0]:.2f} | "
                  f"Return: {best_return[0]:.1%} | Trades/yr: {best_trades_per_year[0]:.1f}")
        
        generation[0] = current_gen

        try:
            # Decode parameters (strategy-specific)
            params = decode_parameters(x)

            # Validate parameters (strategy-specific constraints)
            is_valid, penalty = validate_parameters(params)
            if not is_valid:
                return penalty

            # Run strategy
            entries, exits, position_target = run_strategy_simple(close, high, low, **params)

            if entries.sum() < 3:
                return 10.0

            portfolio = vbt.Portfolio.from_signals(
                close, entries, exits,
                size=position_target,
                size_type=SizeType.Percent,
                init_cash=CAPITAL_BASE,
                fees=MANUAL_DEFAULTS["commission_rate"],
                slippage=MANUAL_DEFAULTS["slippage_rate"],
                freq="1D",
                accumulate=False  # Match Pine Script pyramiding=1
            )
            stats = portfolio.stats()

            score, components = compute_composite_score(portfolio, stats, params, training_days)
            sortino = components["sortino_raw"]
            total_return = stats["Total Return [%]"] / 100.0
            trades_per_year = components["trades_per_year"]

            if score > best_score[0]:
                best_score[0] = score
                best_sortino[0] = sortino
                best_return[0] = total_return
                best_trades_per_year[0] = trades_per_year
                best_x[0] = x.copy()  # Save the actual best parameter array!

            return -score  # Minimize negative score

        except Exception as e:
            return 10.0

    # Run differential evolution with MAXIMUM diversity to avoid local maxima
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        popsize=GENETIC_POPULATION_SIZE,
        strategy='rand2bin',        # More exploratory than best1bin - avoids local maxima
        mutation=(0.8, 1.99),       # Very high mutation for aggressive exploration
        recombination=0.9,          # High recombination for gene mixing
        seed=None,                  # Random seed for different results each run
        workers=1,                  # Single-threaded for reproducibility
        polish=False,               # Don't refine solution (can reduce diversity)
        updating='deferred',        # Evaluate full generation before updating
        atol=0,                     # Disable absolute tolerance
        tol=1e-100,                 # Effectively disable relative tolerance
        init='sobol',               # Use Sobol sequence for better initial coverage
        disp=False,                 # Don't print convergence messages
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    # Use OUR tracked best parameters, not differential_evolution's result!
    # (DE may return a different solution due to numerical precision/noise)
    if best_x[0] is not None:
        best_params = decode_parameters(best_x[0])
        best_params["score"] = best_score[0]
        best_params["train_sortino"] = best_sortino[0]
        best_params["train_return"] = best_return[0]
    else:
        # Fallback to DE's result if we somehow didn't track anything
        best_params = decode_parameters(result.x)
        best_params["score"] = -result.fun
        best_params["train_sortino"] = best_sortino[0]
        best_params["train_return"] = best_return[0]

    print(f"  ✔ [{end_time.strftime('%H:%M:%S')}] Optimization complete in {duration}")
    print(f"     Total evaluations: {call_count[0]:,} | Best Score: {best_params['score']:5.2f} "
          f"(Sortino: {best_params['train_sortino']:.2f}, Return: {best_params['train_return']:.1%})")
    
    # Print parameter summary (generic - just show all params)
    print(f"     Optimized parameters:")
    for key, value in best_params.items():
        if key not in ["score", "train_sortino", "train_return"]:
            print(f"       {key}: {value}")

    return best_params


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


def load_asset_data(asset_name: str, config: dict) -> pd.DataFrame | None:
    """Load and standardize asset price data"""
    primary_path = config.get("primary", Path())

    if not primary_path.exists():
        print(f"✗ {asset_name}: primary file {primary_path} not found")
        return None

    try:
        asset_df = _standardize_price_frame(pd.read_csv(primary_path))
        print(f"✓ {asset_name}: loaded data ({len(asset_df)} rows) from {primary_path}")
        asset_df["asset"] = asset_name
        return asset_df
    except Exception as err:
        print(f"✗ {asset_name}: failed to load data ({primary_path}): {err}")
        return None


# ============================================================================
# OPTIMIZATION WORKFLOW
# ============================================================================

def quick_optimize(data, max_iterations, stage_name=""):
    """Run quick optimization on full dataset"""
    print(f"\n{'='*70}")
    print(f"⚡ QUICK OPTIMIZATION: {stage_name}")
    print(f"{'='*70}")

    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data[data["time"] >= "2013-01-01"].copy()
    data.set_index("time", inplace=True)
    data = data.sort_index()

    start_date, end_date = data.index[0], data.index[-1]
    print(f"\nData range: {start_date.date()} – {end_date.date()} | {len(data)} days")

    if len(data) < 150:
        print(f"  ⚠️ Insufficient data: only {len(data)} bars — aborting")
        return pd.DataFrame()

    print(f"\n🧬 Genetic optimization: {max_iterations} generations (pop: {GENETIC_POPULATION_SIZE * NUM_PARAMETERS})")
    print(f"   Expected total evaluations: ~{max_iterations * GENETIC_POPULATION_SIZE * NUM_PARAMETERS:,}")

    best = optimize_parameters_genetic(
        data,
        start_date,
        end_date,
        max_iterations=max_iterations,
    )

    if best is None:
        print("❌ Optimization failed")
        return pd.DataFrame()

    # Evaluate best parameters
    close = data["close"]
    high = data["high"]
    low = data["low"]
    entries, exits, position_target = run_strategy_simple(close, high, low, **best)
    
    portfolio = vbt.Portfolio.from_signals(
        close, entries, exits,
        size=position_target,
        size_type=SizeType.Percent,
        init_cash=CAPITAL_BASE,
        fees=MANUAL_DEFAULTS["commission_rate"],
        slippage=MANUAL_DEFAULTS["slippage_rate"],
        freq="1D",
        accumulate=False  # Match Pine Script pyramiding=1
    )
    
    stats = portfolio.stats()
    composite, components = compute_composite_score(portfolio, stats, best, len(close))
    sortino = components["sortino_raw"]
    calmar = components["calmar_ratio"]
    sharpe = stats.get("Sharpe Ratio", np.nan)
    max_dd = stats.get("Max Drawdown [%]", 0) / 100
    total_return = stats.get("Total Return [%]", 0)
    num_trades = portfolio.trades.count()
    win_rate = portfolio.trades.win_rate() if num_trades > 0 else 0

    print(f"\n{'='*70}")
    print("✅ OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"  Composite Score:   {composite:.2f}")
    print(f"  Sortino Ratio:     {sortino:.2f}")
    print(f"  Calmar Ratio:      {calmar:.2f}")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Total Return:      {total_return:+.1f}%")
    print(f"  Max Drawdown:      {max_dd*100:.1f}%")
    print(f"  Win Rate:          {win_rate*100:.1f}%")
    print(f"  Number of Trades:  {num_trades}")
    print(f"{'='*70}")

    # Build result dictionary - generic approach
    result = {
        "start_date": start_date,
        "end_date": end_date,
        "days": len(close),
        "composite_score": composite,
        "sortino": sortino,
        "calmar": calmar,
        "sharpe": sharpe,
        "total_return": total_return / 100,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }
    
    # Add all optimized parameters (generic!)
    for key, value in best.items():
        if key not in ["score", "train_sortino"]:
            result[key] = value

    return pd.DataFrame([result])


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main optimization workflow"""
    from v1.strategy_config import STRATEGY_NAME, NUM_PARAMETERS
    
    print("\n==========================================")
    print(f"STRATEGY OPTIMIZER - {STRATEGY_NAME}")
    print("==========================================\n")
    
    print(f"📊 Strategy: {STRATEGY_NAME}")
    print(f"   Parameters to optimize: {NUM_PARAMETERS}\n")
    
    if QUICK_MODE:
        print("⚡ QUICK MODE ENABLED - Fast single-period optimization")
        print("   (Set QUICK_MODE=False for full walk-forward testing)\n")
    
    # Load asset data
    asset_data_map = {}
    for asset_name, asset_cfg in ASSET_DATA_SOURCES.items():
        df = load_asset_data(asset_name, asset_cfg)
        if df is not None:
            asset_data_map[asset_name] = df
    
    if not asset_data_map:
        print("✗ No valid asset data found. Please check your data files.")
        return
    
    # Run optimization for each asset
    all_results = []
    max_iter = GENETIC_QUICK_MAX_ITER if QUICK_MODE else GENETIC_MAX_ITERATIONS
    
    for asset_name, asset_df in asset_data_map.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING: {asset_name}")
        print(f"{'='*70}")
        
        result = quick_optimize(
            asset_df.copy(),
            max_iterations=max_iter,
            stage_name=f"{asset_name}"
        )
        
        if not result.empty:
            result["asset"] = asset_name
            all_results.append(result)
    
    # Save results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        output_file = "hermes_simple_optimization_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
