"""
Hermes Strategy Optimizer - Trend-Adaptive ALMA with Genetic/Bayesian Optimization
====================================================================================
✅ Efficiency Ratio + R-Squared trend detection (2-factor model)
✅ Dynamic ALMA parameters that interpolate based on market regime
✅ Regime-based exits (trending vs ranging logic)
✅ Numba JIT acceleration
✅ QuantStats HTML reports + Parameter heatmaps

OPTIMIZATION METHODS:
---------------------
1. GENETIC (Differential Evolution) - RECOMMENDED for 15+ parameters
   - Better global exploration for high-dimensional spaces
   - Population-based search (12 multiplier × 18 params = 216 individuals)
   - Robust to local optima
   - Quick: 40 gens = 8,640 evaluations (~10-20 min)
   - Full: 100 gens = 21,600 evaluations per window (~30-60 min)
   - Set: OPTIMIZATION_METHOD = "genetic"

2. BAYESIAN (Gaussian Process)
   - Better for <15 dimensions with expensive evaluations
   - Sequential model-based optimization
   - 150-500 evaluations depending on mode
   - Set: OPTIMIZATION_METHOD = "bayesian"

MODES:
------
1. QUICK MODE (QUICK_MODE = True):
   - Single-period optimization on entire dataset (~10-20 min per asset)
   - Tests from 2013 onwards (full history)
   - No walk-forward validation
   - Use for rapid parameter testing and viability checks
   - Genetic: 40 generations = 8,640 evaluations
   - Bayesian: 150 calls with 50 random starts

2. FULL MODE (QUICK_MODE = False):
   - Complete walk-forward optimization with bull market detection
   - Multiple train/test windows with purging
   - Full validation and robustness testing
   - Generates QuantStats reports and heatmaps
   - Genetic: 100 generations = 21,600 evaluations per window (~30-60 min)
   - Bayesian: 300-500 calls per window

HOW TO USE:
-----------
1. Set OPTIMIZATION_METHOD = "genetic" (line 140) - recommended for 18 parameters
2. Set QUICK_MODE = True (line 130) for fast testing
3. Set QUICK_MODE = False for production-ready walk-forward validation
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from scipy.optimize import differential_evolution
from numba import njit
import datetime
import warnings
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("⚠️  QuantStats not installed. Run: pip install quantstats")
from pathlib import Path

warnings.filterwarnings("ignore", message="The objective has been evaluated")
pd.set_option("future.no_silent_downcasting", True)

# Create output directories for reports
Path("reports").mkdir(exist_ok=True)
Path("reports/quantstats").mkdir(exist_ok=True)
Path("reports/heatmaps").mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Manual defaults matching hermes.pine
MANUAL_DEFAULTS = {
    "short_period": 30,
    "long_period": 250,
    "alma_offset": 0.95,
    "alma_sigma": 4.0,
    "trending_alma_offset": 0.75,
    "ranging_alma_offset": 0.95,
    "trending_alma_sigma": 8.0,
    "ranging_alma_sigma": 4.0,
    "trend_analysis_period": 50,
    "trend_threshold": 0.60,
    "weight_efficiency": 0.70,
    "weight_rsquared": 0.30,
    "fast_hma_period": 30,
    "slow_ema_period": 80,
    "momentum_lookback": 3,
    "slow_ema_rising_lookback": 3,
    "macro_ema_period": 150,
    "profit_period_scale_factor": 0.02,
    "commission_rate": 0.0,  # TradingView default: no commission
    "slippage_rate": 0.0,  # TradingView default: no slippage (daily timeframe)
}

# SIMPLE parameter space - only 10 parameters (matches hermes_old.pine)
SIMPLE_SPACE = [
    Integer(10, 150, name="short_period"),
    Integer(100, 400, name="long_period"),
    Integer(80, 99, name="alma_offset_int"),  # 0.80-0.99
    Categorical([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], name="alma_sigma"),
    Integer(1, 10, name="momentum_lookback"),
    Categorical([0, 1], name="use_macro_filter"),  # NEW: 0=disabled, 1=enabled
    Integer(50, 250, name="macro_ema_period"),
    Integer(10, 100, name="fast_hma_period"),
    Integer(30, 200, name="slow_ema_period"),
    Integer(1, 15, name="slow_ema_rising_lookback"),
]

# Stage 1: Global search space (MODERATELY WIDENED - Better balance)
STAGE1_SPACE = [
    # ALMA base periods - Moderate expansion
    Integer(5, 150, name="short_period"),  # Original: 10-100, Wide: 5-200, Now: 5-150
    Integer(80, 450, name="long_period"),   # Original: 100-400, Wide: 50-500, Now: 80-450

    # ALMA base parameters - Moderate expansion
    Integer(70, 99, name="alma_offset_int"),  # Original: 85-99, Wide: 50-99, Now: 0.70-0.99
    Categorical([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], name="alma_sigma"),  # Original: 3-10, Now: 2-12

    # Dynamic ALMA parameters - Moderate expansion
    Integer(60, 92, name="trending_alma_offset_int"),  # Original: 70-90, Wide: 50-95, Now: 0.60-0.92
    Integer(85, 99, name="ranging_alma_offset_int"),   # Original: 90-99, Wide: 80-99, Now: 0.85-0.99
    Categorical([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], name="trending_alma_sigma"),  # Original: 4-12, Now: 3-14
    Categorical([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0], name="ranging_alma_sigma"),  # Original: 2-7, Now: 1.5-8

    # Trend detection parameters - Moderate expansion
    Integer(15, 150, name="trend_analysis_period"),  # Original: 30-100, Wide: 10-200, Now: 15-150
    Integer(30, 85, name="trend_threshold_int"),  # Original: 40-80, Wide: 20-90, Now: 0.30-0.85

    # Trend component weights - Moderate expansion
    Integer(30, 95, name="weight_efficiency_int"),  # Original: 50-90, Wide: 10-95, Now: 0.30-0.95
    Integer(5, 70, name="weight_rsquared_int"),    # Original: 10-50, Wide: 5-90, Now: 0.05-0.70

    # Price structure parameters - Moderate expansion
    Integer(10, 150, name="fast_hma_period"),  # Original: 20-100, Wide: 5-200, Now: 10-150
    Integer(30, 250, name="slow_ema_period"),  # Original: 50-150, Wide: 20-300, Now: 30-250

    # Momentum filters - Moderate expansion
    Integer(1, 15, name="momentum_lookback"),  # Original: 1-10, Wide: 1-20, Now: 1-15
    Integer(1, 15, name="slow_ema_rising_lookback"),  # Original: 1-10, Wide: 1-20, Now: 1-15

    # Macro filter - Moderate expansion
    Integer(75, 350, name="macro_ema_period"),  # Original: 100-250, Wide: 50-400, Now: 75-350

    # Profit scaling - Moderate expansion
    Integer(0, 15, name="profit_scale_int"),  # Original: 0-10, Wide: 0-20, Now: 0.00-0.15
]

# ============================================================================
# SIMPLE MODE: Use old simple strategy (matches hermes_old.pine)
# ============================================================================
# Set to True to use the SIMPLE strategy (8 params, no dynamic ALMA)
# Set to False to use the COMPLEX strategy (18 params, full features)
SIMPLE_MODE = True  # 🔥 SET THIS TO TRUE FOR BETTER PERFORMANCE

# ============================================================================
# QUICK MODE: Fast single-period optimization (no walk-forward)
# ============================================================================
# Set to True for quick parameter testing without walk-forward validation
QUICK_MODE = False  # Change to True for fast testing
QUICK_MODE_CALLS = 150  # Fewer iterations for speed (not used with genetic)
QUICK_MODE_RANDOM_STARTS = 50  # Not used with genetic

# ============================================================================
# OPTIMIZATION METHOD
# ============================================================================
# Choose optimization algorithm:
# - "bayesian": Gaussian Process (good for <15 dimensions, expensive evaluations)
# - "genetic": Differential Evolution (better for 15+ dimensions, more robust)
OPTIMIZATION_METHOD = "genetic"  # Recommended for large parameter spaces

# Genetic algorithm settings (only used when OPTIMIZATION_METHOD = "genetic")
# Population size = popsize × num_dimensions (18 params × multiplier = total individuals)
GENETIC_POPULATION_SIZE = 15  # Multiplier: 15 × 18 = 270 individuals per generation
GENETIC_MAX_ITERATIONS = 300  # Full mode: 300 generations = 81,000 evaluations (wider space needs more)
GENETIC_QUICK_MAX_ITER = 100   # Quick mode: 100 generations = 27,000 evaluations

STAGE1_CALLS = 300
STAGE1_RANDOM_STARTS = 100

STAGE2_CALLS = 500
STAGE2_RANDOM_STARTS = 50

# Bull market detection
MIN_BULL_PERIOD_DAYS = 250
BULL_DETECTION_EMA_PERIOD = 200
BULL_SLOPE_LOOKBACK = 20
TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.20
MIN_TEST_DAYS = 90
PURGE_DAYS = 10

CAPITAL_BASE = 150000

BOOTSTRAP_BLOCK_SIZE = 30
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 42

ASSET_DATA_SOURCES = {
    "BTC": {
        "primary": Path("btc_daily.csv"),
        "proxies": [
            Path("data/blx_daily.csv"),
            Path("data/cme_btc_daily.csv"),
        ],
    },
    "ETH": {
        "primary": Path("eth_daily.csv"),
        "proxies": [
            Path("data/eth_daily_proxy.csv"),
        ],
    },
    "SOL": {
        "primary": Path("sol_daily.csv"),
        "proxies": [],
    },
}

# Objective penalties
PENALTY_EXCESSIVE_TRADES = 0.02
PENALTY_EXTREME_GAIN = 0.05
PENALTY_HIGH_DRAWDOWN = 0.1

# ============================================================================
# NUMBA-ACCELERATED TREND DETECTION FUNCTIONS
# ============================================================================

@njit(cache=True, fastmath=True)
def efficiency_ratio_numba(close, period):
    """
    Calculate Efficiency Ratio: directional movement / total movement.
    Returns: 0-1 where 1 = perfectly efficient trend
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i < period:
            result[i] = 0.0
            continue

        # Directional movement (net change)
        change = abs(close[i] - close[i - period])

        # Total movement (sum of absolute changes)
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])

        if volatility > 0:
            result[i] = min(1.0, change / volatility)
        else:
            result[i] = 0.0

    return result


@njit(cache=True, fastmath=True)
def r_squared_numba(close, period):
    """
    Calculate linear regression R-Squared.
    Returns: 0-1 where higher = more linear = trending
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
            result[i] = 0.0
            continue

        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0

        for j in range(period):
            idx = i - period + 1 + j
            x = float(j)
            y = close[idx]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y

        n_period = float(period)

        # Pearson correlation
        numerator = (n_period * sum_xy) - (sum_x * sum_y)
        denom_part1 = (n_period * sum_x2) - (sum_x * sum_x)
        denom_part2 = (n_period * sum_y2) - (sum_y * sum_y)

        if denom_part1 <= 0 or denom_part2 <= 0:
            result[i] = 0.0
            continue

        denominator = np.sqrt(denom_part1 * denom_part2)

        if denominator > 0:
            correlation = numerator / denominator
            result[i] = correlation * correlation  # R-squared
            result[i] = min(1.0, max(0.0, result[i]))
        else:
            result[i] = 0.0

    return result


@njit(cache=True, fastmath=True)
def alma_numba(src, period, offset, sigma):
    """
    ALMA (Arnaud Legoux Moving Average) - Gaussian-weighted smoothing.
    """
    n = len(src)
    result = np.empty(n, dtype=np.float64)

    # Calculate Gaussian weights
    m = offset * (period - 1)
    s = period / sigma

    for i in range(n):
        if i < period - 1:
            # Not enough data yet, use simple average
            result[i] = np.mean(src[:i+1])
        else:
            # Apply Gaussian weighting
            wtd_sum = 0.0
            cum_wt = 0.0

            for j in range(period):
                idx = i - period + 1 + j
                diff = j - m
                wt = np.exp(-(diff * diff) / (2 * s * s))
                wtd_sum += src[idx] * wt
                cum_wt += wt

            result[i] = wtd_sum / cum_wt if cum_wt != 0 else src[i]

    return result


@njit(cache=True, fastmath=True)
def hma_numba(close, period):
    """
    Hull Moving Average (HMA).
    """
    n = len(close)
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))

    # WMA helper function
    def wma(data, length, end_idx):
        if end_idx < length - 1:
            length = end_idx + 1

        weighted_sum = 0.0
        weight_sum = 0.0
        for i in range(length):
            idx = end_idx - length + 1 + i
            weight = float(i + 1)
            weighted_sum += data[idx] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else data[end_idx]

    # Calculate WMA(2*WMA(n/2) - WMA(n))
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
            result[i] = close[i]
            continue

        wma_half = wma(close, half_period, i)
        wma_full = wma(close, period, i)
        raw = 2 * wma_half - wma_full

        # Need to build array for final smoothing
        # Store intermediate values
        result[i] = raw

    # Second pass: smooth with sqrt period
    final_result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < period - 1:
            final_result[i] = close[i]
        else:
            # WMA of the raw values
            weighted_sum = 0.0
            weight_sum = 0.0
            length = min(sqrt_period, i + 1)

            for j in range(length):
                idx = i - length + 1 + j
                weight = float(j + 1)
                weighted_sum += result[idx] * weight
                weight_sum += weight

            final_result[i] = weighted_sum / weight_sum if weight_sum > 0 else result[i]

    return final_result


# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def run_strategy(close, high, low, **params):
    """
    Hermes strategy with trend-adaptive ALMA and regime-based exits.

    Matches hermes.pine logic:
    1. Calculate Efficiency Ratio + R-Squared
    2. Compute composite trend strength (weighted)
    3. Interpolate ALMA parameters based on trend strength
    4. Entry: bullish ALMA cross + momentum + macro filter
    5. Exit: regime-dependent (trending vs ranging)
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)

    # Extract parameters
    short_period = int(params["short_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]

    trending_alma_offset = params["trending_alma_offset"]
    ranging_alma_offset = params["ranging_alma_offset"]
    trending_alma_sigma = params["trending_alma_sigma"]
    ranging_alma_sigma = params["ranging_alma_sigma"]

    trend_analysis_period = int(params["trend_analysis_period"])
    trend_threshold = params["trend_threshold"]
    weight_efficiency = params["weight_efficiency"]
    weight_rsquared = params["weight_rsquared"]

    fast_hma_period = int(params["fast_hma_period"])
    slow_ema_period = int(params["slow_ema_period"])
    momentum_lookback = int(params["momentum_lookback"])
    slow_ema_rising_lookback = int(params["slow_ema_rising_lookback"])
    macro_ema_period = int(params["macro_ema_period"])
    profit_scale_factor = params["profit_period_scale_factor"]

    # === TREND DETECTION ===
    efficiency_ratio = efficiency_ratio_numba(close_np, trend_analysis_period)
    r_squared = r_squared_numba(close_np, trend_analysis_period)

    # Normalize weights
    weight_sum = weight_efficiency + weight_rsquared
    w_eff = weight_efficiency / weight_sum if weight_sum > 0 else 0.70
    w_rsq = weight_rsquared / weight_sum if weight_sum > 0 else 0.30

    # Composite trend strength
    trend_strength = (w_eff * efficiency_ratio) + (w_rsq * r_squared)
    trend_strength = np.clip(trend_strength, 0.0, 1.0)

    # Smooth trend strength (EMA with span=5)
    smoothed_trend_strength = pd.Series(trend_strength).ewm(span=5, adjust=False).mean().to_numpy()

    is_trending_market = smoothed_trend_strength > trend_threshold

    # === LOG RETURNS FOR ALMA ===
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # === DYNAMIC ALMA (4 variants: ranging/trending × short/long) ===
    long_term_ranging = alma_numba(returns, long_period, ranging_alma_offset, ranging_alma_sigma)
    long_term_trending = alma_numba(returns, long_period, trending_alma_offset, trending_alma_sigma)
    short_term_ranging = alma_numba(returns, short_period, ranging_alma_offset, ranging_alma_sigma)
    short_term_trending = alma_numba(returns, short_period, trending_alma_offset, trending_alma_sigma)

    # Interpolate based on trend strength
    long_term = long_term_ranging + (smoothed_trend_strength * (long_term_trending - long_term_ranging))
    short_term = short_term_ranging + (smoothed_trend_strength * (short_term_trending - short_term_ranging))
    baseline = long_term

    # === PRICE STRUCTURE ===
    fast_hma = hma_numba(close_np, fast_hma_period)
    slow_ema = pd.Series(close_np).ewm(span=slow_ema_period, adjust=False).mean().to_numpy()
    macro_ema = pd.Series(close_np).ewm(span=macro_ema_period, adjust=False).mean().to_numpy()

    # === ENTRY CONDITIONS ===
    bullish_state = short_term > baseline
    in_bull_market = close_np > macro_ema

    # Momentum filter: close and high at N-bar high
    highest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    is_highest_close = (close_np >= np.nan_to_num(highest_close_prev, nan=0)) & \
                       (high_np >= np.nan_to_num(highest_high_prev, nan=0))

    # Slow EMA rising filter
    slow_ema_rising = np.zeros(len(slow_ema), dtype=bool)
    for i in range(slow_ema_rising_lookback, len(slow_ema)):
        slow_ema_rising[i] = slow_ema[i] > slow_ema[i - slow_ema_rising_lookback]

    # Combined buy signal
    buy_signal = bullish_state & is_highest_close & in_bull_market & slow_ema_rising

    # === EXIT CONDITIONS (REGIME-BASED) ===
    # This is simplified - full implementation would track positions and regime state
    # For backtesting, we use simple exit logic

    bearish_state = short_term < baseline

    # Sell momentum filter: low and close at N-bar low
    lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    lowest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    is_lowest_low = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                    (close_np <= np.nan_to_num(lowest_close_prev, nan=np.inf))

    # Trending exit: HMA crosses under slow EMA
    trend_cross_under = (fast_hma < slow_ema) & (np.roll(fast_hma, 1) >= np.roll(slow_ema, 1))

    # Ranging exit: bearish state with momentum
    ranging_exit = bearish_state & is_lowest_low

    # Combined sell signal (simplified - doesn't track regime state per position)
    sell_signal = ranging_exit | trend_cross_under

    # Convert to entry/exit events
    buy_prev = np.roll(buy_signal, 1)
    sell_prev = np.roll(sell_signal, 1)
    buy_prev[0] = False
    sell_prev[0] = False

    entries = buy_signal & (~buy_prev)
    exits = sell_signal & (~sell_prev)

    # Full allocation (no volatility scaling in this version)
    position_target = np.ones(len(close_np), dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)


def run_strategy_simple(close, high, low, **params):
    """
    SIMPLE Hermes strategy - matches hermes_old.pine

    - Only 9 parameters
    - Fixed ALMA (no dynamic interpolation)
    - Simple regime detection
    - No Efficiency Ratio / R-Squared
    - No room-to-breathe system
    - No profit scaling
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)

    # Extract parameters (now 10 with macro filter toggle!)
    short_period = int(params["short_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]
    momentum_lookback = int(params["momentum_lookback"])
    use_macro_filter = bool(params.get("use_macro_filter", 1))  # Default to enabled for backwards compat
    macro_ema_period = int(params["macro_ema_period"])
    fast_hma_period = int(params["fast_hma_period"])
    slow_ema_period = int(params["slow_ema_period"])
    slow_ema_rising_lookback = int(params["slow_ema_rising_lookback"])

    # === LOG RETURNS FOR ALMA ===
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # === SIMPLE FIXED ALMA (no dynamic parameters) ===
    long_term = alma_numba(returns, long_period, alma_offset, alma_sigma)
    short_term = alma_numba(returns, short_period, alma_offset, alma_sigma)
    baseline = long_term

    # === PRICE STRUCTURE ===
    fast_hma = hma_numba(close_np, fast_hma_period)
    slow_ema = pd.Series(close_np).ewm(span=slow_ema_period, adjust=False).mean().to_numpy()
    macro_ema = pd.Series(close_np).ewm(span=macro_ema_period, adjust=False).mean().to_numpy()

    # === ENTRY CONDITIONS ===
    bullish_state = short_term > baseline

    # Momentum filter
    highest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    is_highest_close = (close_np >= np.nan_to_num(highest_close_prev, nan=0)) & \
                       (high_np >= np.nan_to_num(highest_high_prev, nan=0))

    # Slow EMA rising
    slow_ema_rising = np.zeros(len(slow_ema), dtype=bool)
    for i in range(slow_ema_rising_lookback, len(slow_ema)):
        slow_ema_rising[i] = slow_ema[i] > slow_ema[i - slow_ema_rising_lookback]

    # Buy signal (conditionally apply macro filter)
    if use_macro_filter:
        in_bull_market = close_np > macro_ema
        buy_signal = bullish_state & is_highest_close & in_bull_market & slow_ema_rising
    else:
        # No macro filter - this is what gets 150,000%!
        buy_signal = bullish_state & is_highest_close & slow_ema_rising

    # === REGIME-BASED EXIT CONDITIONS (MATCHES hermes_old.pine) ===
    bearish_state = short_term < baseline

    # Sell momentum
    lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    lowest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    is_lowest_low = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                    (close_np <= np.nan_to_num(lowest_close_prev, nan=np.inf))

    # Trend cross under detection
    trend_cross_under = (fast_hma < slow_ema) & (np.roll(fast_hma, 1) >= np.roll(slow_ema, 1))

    # Basic ranging exit signal
    ranging_sell_signal = bearish_state & is_lowest_low

    # === STATE MACHINE: Simulate Pine Script's regime tracking ===
    # Track: in_position, trending_regime, entry_price for each bar
    n = len(close_np)
    in_position = np.zeros(n, dtype=bool)
    trending_regime = np.zeros(n, dtype=bool)
    entry_price = np.zeros(n, dtype=np.float64)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Carry forward state from previous bar
        in_position[i] = in_position[i-1]
        trending_regime[i] = trending_regime[i-1]
        entry_price[i] = entry_price[i-1]

        # Entry detection (Pine: enter whenever buy_signal is true and not in position)
        if buy_signal[i] and not in_position[i]:
            entries[i] = True
            in_position[i] = True
            trending_regime[i] = False  # Start in ranging mode
            entry_price[i] = close_np[i]  # Record entry price

        # Exit logic (only when in position)
        elif in_position[i]:
            # Detect trending setup (Pine: slow_ema > entry AND fast_hma > entry AND fast_hma > slow_ema)
            trending_setup = (slow_ema[i] > entry_price[i] and
                            fast_hma[i] > entry_price[i] and
                            fast_hma[i] > slow_ema[i])

            # Update regime state
            if trending_setup:
                trending_regime[i] = True
            elif not trend_cross_under[i]:
                # Pine: "else if not trend_cross_under" - only reset if not crossing
                trending_regime[i] = False

            # Calculate exit conditions
            close_below_entry = close_np[i] < entry_price[i]
            sell_momentum_ok = is_lowest_low[i]  # use_momentum_filters is always True in simple mode
            normal_trending_exit = trend_cross_under[i] and sell_momentum_ok

            # Trending exit: Only when in trending regime
            trending_exit = trending_regime[i] and (close_below_entry or normal_trending_exit)

            # Ranging exit: Only when NOT in trending regime
            ranging_exit = (not trending_regime[i]) and ranging_sell_signal[i]

            # Execute exit
            if trending_exit or ranging_exit:
                exits[i] = True
                in_position[i] = False
                trending_regime[i] = False
                entry_price[i] = 0.0

    # CRITICAL FIX: Shift signals forward by 1 bar to match TradingView
    # TradingView executes orders on the bar AFTER the signal is generated
    # Without this shift, Python executes on the same bar (look-ahead bias!)
    entries_series = pd.Series(entries, index=close.index).shift(1).fillna(False).astype(bool)
    exits_series = pd.Series(exits, index=close.index).shift(1).fillna(False).astype(bool)

    position_target = np.ones(len(close_np), dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (entries_series, exits_series, position_series)


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

def get_strategy_function():
    """Return the appropriate strategy function based on SIMPLE_MODE"""
    return run_strategy_simple if SIMPLE_MODE else run_strategy


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def compute_composite_score(portfolio, stats, params, training_days):
    """
    Compute composite objective score using crypto-adapted risk-adjusted metrics.
    Same as before - using Sortino + Calmar + stability bonus.
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

    # Hard constraints
    if trades_per_year < 3:
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "too_few_trades",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }

    if max_dd > 0.60:
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "excessive_drawdown",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }

    years = training_days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1
    if annualized_return < 0.10:
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "insufficient_return",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }

    if trades_per_year > 200:
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "excessive_trading",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }

    # SIMPLIFIED SCORING: MAXIMIZE ABSOLUTE RETURNS
    # Primary metric: Direct annualized return (80%)
    # Cap at 10.0 (1000% per year) to avoid infinities
    return_score = min(annualized_return, 10.0) * 0.80

    # Secondary 1: Drawdown penalty (5%)
    # 30% DD = full score, 70% DD = 0 score
    dd_penalty = max(0.0, 1.0 - (max_dd - 0.30) / 0.40) if max_dd > 0.30 else 1.0
    dd_score = dd_penalty * 0.05

    # Secondary 2: Trade frequency penalty (15% - INCREASED FROM 2.5%)
    # This is critical for walk-forward validation - need enough trades to test!
    # Prefer 5-50 trades/year, with HARSH penalty below 3 trades/year
    if trades_per_year >= 5:
        if trades_per_year <= 50:
            freq_score = 1.0
        else:  # > 50
            freq_score = max(0.0, 1.0 - (trades_per_year - 50) / 150)
    elif trades_per_year >= 3:
        # Linear penalty: 3-5 trades/year gets 0.4-1.0
        freq_score = 0.4 + (trades_per_year - 3) * 0.3
    else:
        # HARSH penalty below 3 trades/year: 0-3 gets 0.0-0.4
        # Even 1 trade/year only gets 0.13 (vs old 0.2)
        freq_score = (trades_per_year / 3.0) * 0.4

    freq_penalty = freq_score * 0.15

    # Win rate: prefer 30-70% (weight reduced to keep total at 100%)
    if 0.30 <= win_rate <= 0.70:
        wr_score = 1.0
    elif win_rate < 0.30:
        wr_score = win_rate / 0.30
    else:
        wr_score = max(0.0, 1.0 - (win_rate - 0.70) / 0.20)

    wr_penalty = wr_score * 0.00  # Disabled for now - focus on trade frequency

    composite = return_score + dd_score + freq_penalty + wr_penalty

    # For backwards compatibility
    primary_score = return_score
    calmar = annualized_return / max_dd if max_dd > 0 else 0

    return composite, {
        "sortino_raw": sortino,
        "composite_score": composite,
        "return_score": return_score,
        "dd_score": dd_score,
        "freq_penalty": freq_penalty,
        "stability_score": freq_penalty,  # For backwards compatibility
        "calmar_ratio": calmar,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "trades_per_year": trades_per_year,
        "win_rate": win_rate,
        "constraint_violation": None,
    }


# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def optimize_parameters_bayesian(data, start_date, end_date,
                                 param_space, n_calls, n_random_starts, n_jobs=12):
    close = data.loc[start_date:end_date, "close"]
    high = data.loc[start_date:end_date, "high"]
    low = data.loc[start_date:end_date, "low"]
    open_prices = data.loc[start_date:end_date, "open"]

    if len(close) < 150:
        print(f"  ⚠ Insufficient data: only {len(close)} bars — skipping")
        return None

    start_time = datetime.datetime.now()
    training_days = len(close)
    print(f"  ▶ [{start_time.strftime('%H:%M:%S')}] "
          f"Optimizing on {start_date.date()}–{end_date.date()} "
          f"({training_days} bars, {n_calls} Bayesian calls)")

    call_count = [0]
    best_score = [float("-inf")]
    best_sortino_raw = [0.0]

    @use_named_args(param_space)
    def objective(**raw_params):
        call_count[0] += 1

        try:
            # Convert integer parameters to float based on mode
            if SIMPLE_MODE:
                params = {
                    "short_period": raw_params["short_period"],
                    "long_period": raw_params["long_period"],
                    "alma_offset": raw_params["alma_offset_int"] / 100.0,
                    "alma_sigma": raw_params["alma_sigma"],
                    "momentum_lookback": raw_params["momentum_lookback"],
                    "use_macro_filter": raw_params["use_macro_filter"],
                    "macro_ema_period": raw_params["macro_ema_period"],
                    "fast_hma_period": raw_params["fast_hma_period"],
                    "slow_ema_period": raw_params["slow_ema_period"],
                    "slow_ema_rising_lookback": raw_params["slow_ema_rising_lookback"],
                }
            else:
                params = {
                    "short_period": raw_params["short_period"],
                    "long_period": raw_params["long_period"],
                    "alma_offset": raw_params["alma_offset_int"] / 100.0,
                    "alma_sigma": raw_params["alma_sigma"],
                    "trending_alma_offset": raw_params["trending_alma_offset_int"] / 100.0,
                    "ranging_alma_offset": raw_params["ranging_alma_offset_int"] / 100.0,
                    "trending_alma_sigma": raw_params["trending_alma_sigma"],
                    "ranging_alma_sigma": raw_params["ranging_alma_sigma"],
                    "trend_analysis_period": raw_params["trend_analysis_period"],
                    "trend_threshold": raw_params["trend_threshold_int"] / 100.0,
                    "weight_efficiency": raw_params["weight_efficiency_int"] / 100.0,
                    "weight_rsquared": raw_params["weight_rsquared_int"] / 100.0,
                    "fast_hma_period": raw_params["fast_hma_period"],
                    "slow_ema_period": raw_params["slow_ema_period"],
                    "momentum_lookback": raw_params["momentum_lookback"],
                    "slow_ema_rising_lookback": raw_params["slow_ema_rising_lookback"],
                    "macro_ema_period": raw_params["macro_ema_period"],
                    "profit_period_scale_factor": raw_params["profit_scale_int"] / 100.0,
                }

            # Parameter constraints
            if params["short_period"] >= params["long_period"]:
                return 999.0

            if params["long_period"] - params["short_period"] < 20:
                return 999.0

            if params["fast_hma_period"] >= params["slow_ema_period"]:
                return 999.0

            # Complex mode specific constraints
            if not SIMPLE_MODE:
                if params["trending_alma_offset"] > params["ranging_alma_offset"]:
                    return 999.0  # Trending should be more responsive (lower offset)

                if params["trending_alma_sigma"] < params["ranging_alma_sigma"]:
                    return 999.0  # Trending should be smoother (higher sigma)

            strategy_func = get_strategy_function()
            entries, exits, position_target = strategy_func(close, high, low, **params)

            if entries.sum() < 3:
                return 10.0

            portfolio = vbt.Portfolio.from_signals(
                close, entries, exits,
                price=open_prices,  # Execute at OPEN prices to match TradingView
                size=position_target,
                size_type=SizeType.Percent,
                init_cash=CAPITAL_BASE,
                fees=MANUAL_DEFAULTS["commission_rate"],
                slippage=MANUAL_DEFAULTS["slippage_rate"],
                freq="1D"
            )
            stats = portfolio.stats()

            score, components = compute_composite_score(portfolio, stats, params, training_days)
            sortino = components["sortino_raw"]

            now = datetime.datetime.now().strftime("[%H:%M:%S]")

            if components.get("constraint_violation"):
                if call_count[0] % 100 == 0:
                    violation = components["constraint_violation"]
                    print(f"    {now} Call {call_count[0]:3d}/{n_calls} - Rejected: {violation}")

            if score > best_score[0]:
                best_score[0] = score
                best_sortino_raw[0] = sortino
                if call_count[0] % 10 == 0 or call_count[0] <= 3:
                    trades_yr = components["trades_per_year"]
                    calmar = components["calmar_ratio"]
                    total_return = stats.get("Total Return [%]", 0.0)
                    eff_wt = params["weight_efficiency"]
                    rsq_wt = params["weight_rsquared"]
                    print(f"    {now} ✓ Call {call_count[0]:3d}/{n_calls} "
                          f"Score: {score:5.2f} (Sortino: {sortino:.2f}, "
                          f"Calmar: {calmar:.2f}, Return: {total_return:+.1f}%, Trades/yr: {trades_yr:.0f}, "
                          f"Weights: E={eff_wt:.2f}/R²={rsq_wt:.2f})")
            elif call_count[0] % 50 == 0:
                print(f"    {now} Progress {call_count[0]:3d}/{n_calls} "
                      f"| Best score: {best_score[0]:.2f}")

            return -score
        except Exception as e:
            if call_count[0] % 100 == 0:
                print(f"    Error in call {call_count[0]}: {e}")
            return 10.0

    result = gp_minimize(
        objective,
        param_space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=42,
        verbose=False,
        n_jobs=n_jobs,
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    dim_names = [dim.name for dim in param_space]
    raw_best = dict(zip(dim_names, result.x))

    # Build best_params based on mode
    if SIMPLE_MODE:
        best_params = {
            "short_period": int(raw_best["short_period"]),
            "long_period": int(raw_best["long_period"]),
            "alma_offset": raw_best["alma_offset_int"] / 100.0,
            "alma_sigma": float(raw_best["alma_sigma"]),
            "momentum_lookback": int(raw_best["momentum_lookback"]),
            "use_macro_filter": int(raw_best["use_macro_filter"]),
            "macro_ema_period": int(raw_best["macro_ema_period"]),
            "fast_hma_period": int(raw_best["fast_hma_period"]),
            "slow_ema_period": int(raw_best["slow_ema_period"]),
            "slow_ema_rising_lookback": int(raw_best["slow_ema_rising_lookback"]),
            "score": -result.fun,
            "train_sortino": best_sortino_raw[0],
        }
    else:
        best_params = {
            "short_period": int(raw_best["short_period"]),
            "long_period": int(raw_best["long_period"]),
            "alma_offset": raw_best["alma_offset_int"] / 100.0,
            "alma_sigma": float(raw_best["alma_sigma"]),
            "trending_alma_offset": raw_best["trending_alma_offset_int"] / 100.0,
            "ranging_alma_offset": raw_best["ranging_alma_offset_int"] / 100.0,
            "trending_alma_sigma": float(raw_best["trending_alma_sigma"]),
            "ranging_alma_sigma": float(raw_best["ranging_alma_sigma"]),
            "trend_analysis_period": int(raw_best["trend_analysis_period"]),
            "trend_threshold": raw_best["trend_threshold_int"] / 100.0,
            "weight_efficiency": raw_best["weight_efficiency_int"] / 100.0,
            "weight_rsquared": raw_best["weight_rsquared_int"] / 100.0,
            "fast_hma_period": int(raw_best["fast_hma_period"]),
            "slow_ema_period": int(raw_best["slow_ema_period"]),
            "momentum_lookback": int(raw_best["momentum_lookback"]),
            "slow_ema_rising_lookback": int(raw_best["slow_ema_rising_lookback"]),
            "macro_ema_period": int(raw_best["macro_ema_period"]),
            "profit_period_scale_factor": raw_best["profit_scale_int"] / 100.0,
            "score": -result.fun,
            "train_sortino": best_sortino_raw[0],
        }

    print(f"  ✔ [{end_time.strftime('%H:%M:%S')}] "
          f"Done in {duration} | Best Score {best_params['score']:5.2f} "
          f"(Train Sortino: {best_params['train_sortino']:.2f})")
    print(f"     ALMA Periods: short={best_params['short_period']}, long={best_params['long_period']}")
    print(f"     ALMA Fixed: offset={best_params['alma_offset']:.2f}, sigma={best_params['alma_sigma']:.1f}")

    if SIMPLE_MODE:
        print(f"     Price Structure: HMA={best_params['fast_hma_period']}, EMA={best_params['slow_ema_period']}")
        macro_status = "ENABLED" if best_params['use_macro_filter'] else "DISABLED"
        print(f"     Filters: Momentum={best_params['momentum_lookback']}, EMA Rising={best_params['slow_ema_rising_lookback']}")
        print(f"     Macro Filter: {macro_status} (period={best_params['macro_ema_period']})")
    else:
        print(f"     Dynamic ALMA: trending={best_params['trending_alma_offset']:.2f}/{best_params['trending_alma_sigma']:.1f}, "
              f"ranging={best_params['ranging_alma_offset']:.2f}/{best_params['ranging_alma_sigma']:.1f}")
        print(f"     Trend Detection: period={best_params['trend_analysis_period']}, "
              f"threshold={best_params['trend_threshold']:.2f}")
        print(f"     Weights: Efficiency={best_params['weight_efficiency']:.2f}, R²={best_params['weight_rsquared']:.2f}")
        print(f"     Price Structure: HMA={best_params['fast_hma_period']}, EMA={best_params['slow_ema_period']}")

    return best_params


def optimize_parameters_genetic(data, start_date, end_date,
                                param_space, max_iterations, workers=12):
    """
    Genetic algorithm optimization using Differential Evolution.
    Better for high-dimensional parameter spaces (15+ parameters).

    Args:
        data: Price data DataFrame
        start_date, end_date: Date range for optimization
        param_space: skopt-style parameter space (will be converted to bounds)
        max_iterations: Maximum generations to run
        workers: Number of parallel workers

    Returns:
        Dictionary of best parameters and scores
    """
    close = data.loc[start_date:end_date, "close"]
    high = data.loc[start_date:end_date, "high"]
    low = data.loc[start_date:end_date, "low"]
    open_prices = data.loc[start_date:end_date, "open"]

    if len(close) < 150:
        print(f"  ⚠ Insufficient data: only {len(close)} bars — skipping")
        return None

    start_time = datetime.datetime.now()
    training_days = len(close)

    # Convert skopt space to scipy bounds
    bounds = []
    param_names = []
    param_types = []  # Track if parameter is continuous or discrete

    for dim in param_space:
        param_names.append(dim.name)
        if hasattr(dim, 'low') and hasattr(dim, 'high'):  # Integer or Real
            bounds.append((dim.low, dim.high))
            param_types.append('continuous' if 'int' not in dim.name else 'integer')
        elif hasattr(dim, 'categories'):  # Categorical
            # Map categories to indices
            bounds.append((0, len(dim.categories) - 1))
            param_types.append(('categorical', dim.categories))
        else:
            raise ValueError(f"Unknown dimension type for {dim.name}")

    population_size = GENETIC_POPULATION_SIZE * len(param_names)
    print(f"  ▶ [{start_time.strftime('%H:%M:%S')}] "
          f"Genetic optimization on {start_date.date()}–{end_date.date()} "
          f"({training_days} bars, pop={population_size}, max_iter={max_iterations})")

    call_count = [0]
    best_score = [float("-inf")]
    best_sortino_raw = [0.0]

    def objective(x):
        """Objective function for differential evolution."""
        call_count[0] += 1

        try:
            # Convert array to parameters
            raw_params = {}
            for i, (name, ptype) in enumerate(zip(param_names, param_types)):
                if isinstance(ptype, tuple) and ptype[0] == 'categorical':
                    # Categorical: round to nearest index and map to category
                    idx = int(round(x[i]))
                    idx = max(0, min(idx, len(ptype[1]) - 1))
                    raw_params[name] = ptype[1][idx]
                elif ptype == 'integer':
                    raw_params[name] = int(round(x[i]))
                else:
                    raw_params[name] = x[i]

            # Convert to strategy parameters based on mode
            if SIMPLE_MODE:
                # Simple mode: now 10 parameters (added macro filter toggle)
                params = {
                    "short_period": raw_params["short_period"],
                    "long_period": raw_params["long_period"],
                    "alma_offset": raw_params["alma_offset_int"] / 100.0,
                    "alma_sigma": raw_params["alma_sigma"],
                    "momentum_lookback": raw_params["momentum_lookback"],
                    "use_macro_filter": raw_params["use_macro_filter"],
                    "macro_ema_period": raw_params["macro_ema_period"],
                    "fast_hma_period": raw_params["fast_hma_period"],
                    "slow_ema_period": raw_params["slow_ema_period"],
                    "slow_ema_rising_lookback": raw_params["slow_ema_rising_lookback"],
                }
            else:
                # Complex mode: all 18 parameters
                params = {
                    "short_period": raw_params["short_period"],
                    "long_period": raw_params["long_period"],
                    "alma_offset": raw_params["alma_offset_int"] / 100.0,
                    "alma_sigma": raw_params["alma_sigma"],
                    "trending_alma_offset": raw_params["trending_alma_offset_int"] / 100.0,
                    "ranging_alma_offset": raw_params["ranging_alma_offset_int"] / 100.0,
                    "trending_alma_sigma": raw_params["trending_alma_sigma"],
                    "ranging_alma_sigma": raw_params["ranging_alma_sigma"],
                    "trend_analysis_period": raw_params["trend_analysis_period"],
                    "trend_threshold": raw_params["trend_threshold_int"] / 100.0,
                    "weight_efficiency": raw_params["weight_efficiency_int"] / 100.0,
                    "weight_rsquared": raw_params["weight_rsquared_int"] / 100.0,
                    "fast_hma_period": raw_params["fast_hma_period"],
                    "slow_ema_period": raw_params["slow_ema_period"],
                    "momentum_lookback": raw_params["momentum_lookback"],
                    "slow_ema_rising_lookback": raw_params["slow_ema_rising_lookback"],
                    "macro_ema_period": raw_params["macro_ema_period"],
                    "profit_period_scale_factor": raw_params["profit_scale_int"] / 100.0,
                }

            # Parameter constraints
            if params["short_period"] >= params["long_period"]:
                return 999.0
            if params["long_period"] - params["short_period"] < 20:
                return 999.0
            if params["fast_hma_period"] >= params["slow_ema_period"]:
                return 999.0

            # Complex mode specific constraints
            if not SIMPLE_MODE:
                if params["trending_alma_offset"] > params["ranging_alma_offset"]:
                    return 999.0
                if params["trending_alma_sigma"] < params["ranging_alma_sigma"]:
                    return 999.0

            strategy_func = get_strategy_function()
            entries, exits, position_target = strategy_func(close, high, low, **params)

            if entries.sum() < 3:
                return 10.0

            portfolio = vbt.Portfolio.from_signals(
                close, entries, exits,
                price=open_prices,  # Execute at OPEN prices to match TradingView
                size=position_target,
                size_type=SizeType.Percent,
                init_cash=CAPITAL_BASE,
                fees=MANUAL_DEFAULTS["commission_rate"],
                slippage=MANUAL_DEFAULTS["slippage_rate"],
                freq="1D"
            )
            stats = portfolio.stats()

            score, components = compute_composite_score(portfolio, stats, params, training_days)
            sortino = components.get("sortino_raw", 0.0)

            # Track best
            if score > best_score[0]:
                best_score[0] = score
                best_sortino_raw[0] = sortino

                now = datetime.datetime.now().strftime('%H:%M:%S')
                trades_yr = components["trades_per_year"]
                calmar = components["calmar_ratio"]
                total_return = stats.get("Total Return [%]", 0.0)
                print(f"    {now} ★ Gen {call_count[0]//population_size} "
                      f"Score: {score:5.2f} (Sortino: {sortino:.2f}, "
                      f"Calmar: {calmar:.2f}, Return: {total_return:+.1f}%, Trades/yr: {trades_yr:.0f})")

            return -score  # Minimize negative score

        except Exception as e:
            return 10.0

    # Run differential evolution
    # Note: workers=1 to avoid multiprocessing pickling issues with nested function
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        popsize=GENETIC_POPULATION_SIZE,
        strategy='best1bin',
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        workers=1,  # Single-threaded to avoid pickling issues
        updating='immediate',  # Update best solution immediately
        polish=False,  # Don't use L-BFGS-B polish (unnecessary)
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    # Extract best parameters
    raw_best = {}
    for i, (name, ptype) in enumerate(zip(param_names, param_types)):
        if isinstance(ptype, tuple) and ptype[0] == 'categorical':
            idx = int(round(result.x[i]))
            idx = max(0, min(idx, len(ptype[1]) - 1))
            raw_best[name] = ptype[1][idx]
        elif ptype == 'integer':
            raw_best[name] = int(round(result.x[i]))
        else:
            raw_best[name] = result.x[i]

    # Build best_params based on mode
    if SIMPLE_MODE:
        best_params = {
            "short_period": int(raw_best["short_period"]),
            "long_period": int(raw_best["long_period"]),
            "alma_offset": raw_best["alma_offset_int"] / 100.0,
            "alma_sigma": float(raw_best["alma_sigma"]),
            "momentum_lookback": int(raw_best["momentum_lookback"]),
            "use_macro_filter": int(raw_best["use_macro_filter"]),
            "macro_ema_period": int(raw_best["macro_ema_period"]),
            "fast_hma_period": int(raw_best["fast_hma_period"]),
            "slow_ema_period": int(raw_best["slow_ema_period"]),
            "slow_ema_rising_lookback": int(raw_best["slow_ema_rising_lookback"]),
            "score": -result.fun,
            "train_sortino": best_sortino_raw[0],
        }
    else:
        best_params = {
            "short_period": int(raw_best["short_period"]),
            "long_period": int(raw_best["long_period"]),
            "alma_offset": raw_best["alma_offset_int"] / 100.0,
            "alma_sigma": float(raw_best["alma_sigma"]),
            "trending_alma_offset": raw_best["trending_alma_offset_int"] / 100.0,
            "ranging_alma_offset": raw_best["ranging_alma_offset_int"] / 100.0,
            "trending_alma_sigma": float(raw_best["trending_alma_sigma"]),
            "ranging_alma_sigma": float(raw_best["ranging_alma_sigma"]),
            "trend_analysis_period": int(raw_best["trend_analysis_period"]),
            "trend_threshold": raw_best["trend_threshold_int"] / 100.0,
            "weight_efficiency": raw_best["weight_efficiency_int"] / 100.0,
            "weight_rsquared": raw_best["weight_rsquared_int"] / 100.0,
            "fast_hma_period": int(raw_best["fast_hma_period"]),
            "slow_ema_period": int(raw_best["slow_ema_period"]),
            "momentum_lookback": int(raw_best["momentum_lookback"]),
            "slow_ema_rising_lookback": int(raw_best["slow_ema_rising_lookback"]),
            "macro_ema_period": int(raw_best["macro_ema_period"]),
            "profit_period_scale_factor": raw_best["profit_scale_int"] / 100.0,
            "score": -result.fun,
            "train_sortino": best_sortino_raw[0],
        }

    print(f"  ✔ [{end_time.strftime('%H:%M:%S')}] "
          f"Done in {duration} | Best Score {best_params['score']:5.2f} "
          f"(Train Sortino: {best_params['train_sortino']:.2f})")
    print(f"     ALMA Periods: short={best_params['short_period']}, long={best_params['long_period']}")
    print(f"     ALMA Fixed: offset={best_params['alma_offset']:.2f}, sigma={best_params['alma_sigma']:.1f}")

    if SIMPLE_MODE:
        print(f"     Price Structure: HMA={best_params['fast_hma_period']}, EMA={best_params['slow_ema_period']}")
        macro_status = "ENABLED" if best_params['use_macro_filter'] else "DISABLED"
        print(f"     Filters: Momentum={best_params['momentum_lookback']}, EMA Rising={best_params['slow_ema_rising_lookback']}")
        print(f"     Macro Filter: {macro_status} (period={best_params['macro_ema_period']})")
    else:
        print(f"     Dynamic ALMA: trending={best_params['trending_alma_offset']:.2f}/{best_params['trending_alma_sigma']:.1f}, "
              f"ranging={best_params['ranging_alma_offset']:.2f}/{best_params['ranging_alma_sigma']:.1f}")
        print(f"     Trend Detection: period={best_params['trend_analysis_period']}, "
              f"threshold={best_params['trend_threshold']:.2f}")
        print(f"     Weights: Efficiency={best_params['weight_efficiency']:.2f}, R²={best_params['weight_rsquared']:.2f}")
        print(f"     Price Structure: HMA={best_params['fast_hma_period']}, EMA={best_params['slow_ema_period']}")

    return best_params


def optimize_parameters(data, start_date, end_date, param_space, n_calls_or_iter, n_random_starts=None, n_jobs=12):
    """
    Unified optimization interface that routes to Bayesian or Genetic optimizer.

    Args:
        data: Price data DataFrame
        start_date, end_date: Date range
        param_space: Parameter search space
        n_calls_or_iter: Number of calls (Bayesian) or max iterations (Genetic)
        n_random_starts: Random starts for Bayesian (ignored for Genetic)
        n_jobs: Number of parallel workers

    Returns:
        Dictionary of best parameters
    """
    if OPTIMIZATION_METHOD == "genetic":
        return optimize_parameters_genetic(data, start_date, end_date, param_space, n_calls_or_iter, n_jobs)
    elif OPTIMIZATION_METHOD == "bayesian":
        return optimize_parameters_bayesian(data, start_date, end_date, param_space, n_calls_or_iter, n_random_starts or 50, n_jobs)
    else:
        raise ValueError(f"Unknown optimization method: {OPTIMIZATION_METHOD}. Use 'bayesian' or 'genetic'")


# ============================================================================
# REPORTING & VISUALIZATION (reuse from original)
# ============================================================================

def generate_quantstats_report(portfolio, benchmark_returns, iteration_name, stage_name):
    if not QUANTSTATS_AVAILABLE:
        print("  ⚠️  Skipping QuantStats report (not installed)")
        return None

    strategy_returns = portfolio.returns()
    qs.extend_pandas()
    report_path = f"reports/quantstats/{stage_name}_{iteration_name}.html"

    print(f"\n  📊 Generating QuantStats report: {report_path}")

    qs.reports.html(
        strategy_returns,
        benchmark=benchmark_returns,
        output=report_path,
        title=f"Hermes Strategy - {iteration_name}",
        download_filename=report_path,
    )

    print(f"     ✓ Report saved: {report_path}")
    return report_path


def generate_parameter_heatmaps(data, test_start, test_end,
                                best_params, iteration_name, stage_name):
    """Generate parameter robustness heatmaps."""
    print(f"\n  🔥 Generating parameter heatmaps (robustness analysis)...")

    test_close = data.loc[test_start:test_end, "close"]
    test_high = data.loc[test_start:test_end, "high"]
    test_low = data.loc[test_start:test_end, "low"]
    test_open = data.loc[test_start:test_end, "open"]

    # Heatmap 1: Short vs Long Period
    print("     - Short vs Long Period heatmap...")

    short_range = np.arange(
        max(15, best_params["short_period"] - 20),
        min(150, best_params["short_period"] + 20),
        5
    )
    long_range = np.arange(
        max(100, best_params["long_period"] - 50),
        min(350, best_params["long_period"] + 50),
        10
    )

    results_period = np.zeros((len(short_range), len(long_range)))

    for i, short in enumerate(short_range):
        for j, long in enumerate(long_range):
            if short >= long:
                results_period[i, j] = np.nan
                continue

            try:
                params = best_params.copy()
                params["short_period"] = int(short)
                params["long_period"] = int(long)

                strategy_func = get_strategy_function()
                entries, exits, position_target = strategy_func(
                    test_close, test_high, test_low, **params
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    price=test_open,  # Execute at OPEN prices to match TradingView
                    size=position_target,
                    size_type=SizeType.Percent,
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_period[i, j] = port.total_return()
            except Exception:
                results_period[i, j] = np.nan

    heatmap_df = pd.DataFrame(results_period, index=short_range, columns=long_range)
    heatmap_path = f"reports/heatmaps/{stage_name}_{iteration_name}_short_vs_long.csv"
    heatmap_df.to_csv(heatmap_path)
    print(f"       ✓ Saved: {heatmap_path}")

    # Heatmap 2: Trend Weights (Efficiency vs R-Squared)
    print("     - Efficiency vs R-Squared weight heatmap...")

    eff_range = np.arange(0.50, 0.91, 0.05)
    rsq_range = np.arange(0.10, 0.51, 0.05)

    results_weights = np.zeros((len(eff_range), len(rsq_range)))

    for i, eff_wt in enumerate(eff_range):
        for j, rsq_wt in enumerate(rsq_range):
            try:
                params = best_params.copy()
                params["weight_efficiency"] = eff_wt
                params["weight_rsquared"] = rsq_wt

                strategy_func = get_strategy_function()
                entries, exits, position_target = strategy_func(
                    test_close, test_high, test_low, **params
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    price=test_open,  # Execute at OPEN prices to match TradingView
                    size=position_target,
                    size_type=SizeType.Percent,
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_weights[i, j] = port.total_return()
            except Exception:
                results_weights[i, j] = np.nan

    heatmap_df2 = pd.DataFrame(
        results_weights,
        index=[f"{x:.2f}" for x in eff_range],
        columns=[f"{x:.2f}" for x in rsq_range]
    )
    heatmap_path2 = f"reports/heatmaps/{stage_name}_{iteration_name}_efficiency_vs_rsquared_weights.csv"
    heatmap_df2.to_csv(heatmap_path2)
    print(f"       ✓ Saved: {heatmap_path2}")

    # Heatmap 3: Dynamic ALMA offsets
    print("     - Trending vs Ranging ALMA Offset heatmap...")

    trend_offset_range = np.arange(0.70, 0.91, 0.02)
    range_offset_range = np.arange(0.90, 1.00, 0.01)

    results_offsets = np.zeros((len(trend_offset_range), len(range_offset_range)))

    for i, trend_off in enumerate(trend_offset_range):
        for j, range_off in enumerate(range_offset_range):
            if trend_off > range_off:  # Invalid: trending should be more responsive (lower)
                results_offsets[i, j] = np.nan
                continue

            try:
                params = best_params.copy()
                params["trending_alma_offset"] = trend_off
                params["ranging_alma_offset"] = range_off

                strategy_func = get_strategy_function()
                entries, exits, position_target = strategy_func(
                    test_close, test_high, test_low, **params
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    price=test_open,  # Execute at OPEN prices to match TradingView
                    size=position_target,
                    size_type=SizeType.Percent,
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_offsets[i, j] = port.total_return()
            except Exception:
                results_offsets[i, j] = np.nan

    heatmap_df3 = pd.DataFrame(
        results_offsets,
        index=[f"{x:.2f}" for x in trend_offset_range],
        columns=[f"{x:.2f}" for x in range_offset_range]
    )
    heatmap_path3 = f"reports/heatmaps/{stage_name}_{iteration_name}_trending_vs_ranging_offset.csv"
    heatmap_df3.to_csv(heatmap_path3)
    print(f"       ✓ Saved: {heatmap_path3}")

    print(f"     ✓ All heatmaps generated for {iteration_name}")

    return {
        "period_heatmap": heatmap_path,
        "weights_heatmap": heatmap_path2,
        "offsets_heatmap": heatmap_path3,
    }


# ============================================================================
# DATA LOADING (reuse from original)
# ============================================================================

def _standardize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure OHLC structure with epoch-second timestamps."""
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


def _merge_primary_with_proxies(primary: pd.DataFrame, proxies: list[pd.DataFrame]) -> pd.DataFrame:
    """Append proxy history that predates the primary dataset."""
    if not proxies:
        return primary

    merged = primary.copy()
    merged["datetime"] = pd.to_datetime(merged["time"], unit="s")
    merged.set_index("datetime", inplace=True)

    for proxy in proxies:
        proxy_df = proxy.copy()
        proxy_df["datetime"] = pd.to_datetime(proxy_df["time"], unit="s")
        proxy_df.set_index("datetime", inplace=True)
        early_history = proxy_df.loc[proxy_df.index < merged.index.min()]
        if early_history.empty:
            continue
        merged = pd.concat([early_history, merged]).sort_index()

    merged = merged[~merged.index.duplicated(keep="last")]
    merged["time"] = (merged.index.view("int64") // 10 ** 9).astype(np.int64)
    merged.reset_index(drop=True, inplace=True)
    merged = merged.drop(columns=["datetime"], errors="ignore")

    cols = ["time", "open", "high", "low", "close"]
    if "volume" in merged.columns:
        cols.append("volume")
    return merged[cols]


def load_asset_data(asset_name: str, config: dict) -> pd.DataFrame | None:
    """Load primary price history plus optional proxies for a given asset."""
    primary_path: Path = config.get("primary", Path())
    proxies_paths = config.get("proxies", [])

    if not primary_path.exists():
        print(f"✗ {asset_name}: primary file {primary_path} not found")
        return None

    try:
        primary_df = _standardize_price_frame(pd.read_csv(primary_path))
        print(f"✓ {asset_name}: loaded primary data ({len(primary_df)} rows) from {primary_path}")
    except Exception as err:
        print(f"✗ {asset_name}: failed to load primary data ({primary_path}): {err}")
        return None

    proxy_frames = []
    for proxy_path in proxies_paths:
        if not proxy_path.exists():
            continue
        try:
            proxy_df = _standardize_price_frame(pd.read_csv(proxy_path))
            proxy_frames.append(proxy_df)
            print(f"  ↳ attached proxy history ({len(proxy_df)} rows) from {proxy_path}")
        except Exception as err:
            print(f"  ⚠️  {asset_name}: failed to load proxy {proxy_path}: {err}")

    merged = _merge_primary_with_proxies(primary_df, proxy_frames)
    merged["asset"] = asset_name
    return merged


def load_all_asset_data(config: dict) -> dict[str, pd.DataFrame]:
    """Load available assets defined in ASSET_DATA_SOURCES."""
    assets: dict[str, pd.DataFrame] = {}
    for asset_name, asset_cfg in config.items():
        df = load_asset_data(asset_name, asset_cfg)
        if df is None or df.empty:
            continue
        assets[asset_name] = df
    return assets


def block_bootstrap_metrics(
    returns: pd.Series,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int | None = None,
) -> dict:
    """Estimate tail risk using block bootstrap on strategy returns."""
    returns_array = returns.dropna().to_numpy(dtype=np.float64)
    if len(returns_array) == 0 or block_size <= 0 or samples <= 0:
        return {}
    if len(returns_array) < block_size:
        return {}

    rng = np.random.default_rng(seed)
    n = len(returns_array)
    num_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size + 1, 1)

    total_returns = np.empty(samples, dtype=np.float64)
    max_drawdowns = np.empty(samples, dtype=np.float64)

    for i in range(samples):
        sample_blocks = []
        for _ in range(num_blocks):
            start = int(rng.integers(0, max_start))
            block = returns_array[start:start + block_size]
            sample_blocks.append(block)
        sample_path = np.concatenate(sample_blocks)[:n]
        equity_curve = np.cumprod(1.0 + sample_path)
        total_returns[i] = equity_curve[-1] - 1.0
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_drawdowns[i] = drawdowns.min()

    return {
        "bootstrap_return_p05": float(np.percentile(total_returns, 5)),
        "bootstrap_return_p50": float(np.percentile(total_returns, 50)),
        "bootstrap_return_p95": float(np.percentile(total_returns, 95)),
        "bootstrap_max_dd_p50": float(np.percentile(max_drawdowns, 50)),
        "bootstrap_max_dd_p95": float(np.percentile(max_drawdowns, 95)),
    }


# ============================================================================
# WALK-FORWARD ANALYSIS (reuse structure from original)
# ============================================================================

def detect_bull_market_periods(
    data,
    ema_period=BULL_DETECTION_EMA_PERIOD,
    slope_lookback=BULL_SLOPE_LOOKBACK,
    min_days=MIN_BULL_PERIOD_DAYS,
):
    """Detect bull-market segments using a rising EMA filter."""
    close = data["close"]
    macro_ema = close.ewm(span=ema_period, adjust=False).mean()
    ema_slope = macro_ema - macro_ema.shift(slope_lookback)

    in_bull = (close > macro_ema) & (ema_slope > 0)
    in_bull = in_bull.fillna(False)

    periods = []
    current_start = None
    last_true_idx = None
    period_counter = 1

    for idx, flag in in_bull.items():
        if flag:
            if current_start is None:
                current_start = idx
            last_true_idx = idx
        else:
            if current_start is not None and last_true_idx is not None:
                segment = data.loc[current_start:last_true_idx]
                if len(segment) >= min_days:
                    periods.append({
                        "name": f"Bull Period {period_counter}",
                        "start": segment.index[0],
                        "end": segment.index[-1],
                        "days": len(segment),
                        "return": (segment["close"].iloc[-1] / segment["close"].iloc[0] - 1),
                    })
                    period_counter += 1
                current_start = None
                last_true_idx = None
            else:
                current_start = None
                last_true_idx = None

    if current_start is not None and last_true_idx is not None:
        segment = data.loc[current_start:last_true_idx]
        if len(segment) >= min_days:
            periods.append({
                "name": f"Bull Period {period_counter}",
                "start": segment.index[0],
                "end": segment.index[-1],
                "days": len(segment),
                "return": (segment["close"].iloc[-1] / segment["close"].iloc[0] - 1),
            })

    return pd.DataFrame(periods)


def build_walk_forward_windows(
    segment_index,
    min_train_days=MIN_BULL_PERIOD_DAYS,
    min_test_days=MIN_TEST_DAYS,
    purge_days=PURGE_DAYS,
    train_fraction=TRAIN_FRACTION,
    test_fraction=TEST_FRACTION,
):
    """Generate anchored, purged walk-forward windows for a bull-market segment."""
    total_days = len(segment_index)
    if total_days < (min_train_days + min_test_days + purge_days):
        return []

    train_len = max(int(total_days * train_fraction), min_train_days)
    test_len = max(int(total_days * test_fraction), min_test_days)

    if train_len + purge_days + test_len > total_days:
        train_len = total_days - (purge_days + test_len)

    if train_len < min_train_days:
        return []

    windows = []
    train_end_pos = train_len - 1

    while True:
        test_start_pos = train_end_pos + purge_days + 1
        test_end_pos = test_start_pos + test_len - 1

        if test_start_pos >= total_days:
            break

        if test_end_pos >= total_days:
            test_end_pos = total_days - 1

        if test_start_pos > test_end_pos or test_end_pos <= train_end_pos:
            break

        windows.append((
            segment_index[0],
            segment_index[train_end_pos],
            segment_index[test_start_pos],
            segment_index[test_end_pos],
        ))

        if test_end_pos >= total_days - 1:
            break

        train_end_pos = test_end_pos

    return windows


def quick_optimize(data, param_space, n_calls, n_random_starts, stage_name=""):
    """
    QUICK MODE: Fast single-period optimization without walk-forward validation.

    Use this to quickly test if parameters can achieve good performance on a dataset
    before committing to full walk-forward testing. Runs on most recent large chunk
    or entire dataset if available.

    Returns: DataFrame with optimization results
    """

    print(f"\n{'='*70}\n⚡ QUICK MODE: {stage_name}\n{'='*70}")
    print("📌 Running FAST optimization (single period, no walk-forward)")
    print("   Use this to test parameter viability before full walk-forward testing\n")

    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data[data["time"] >= "2013-01-01"].copy()
    data.set_index("time", inplace=True)
    data = data.sort_index()

    # Use entire dataset for quick optimization
    print(f"📊 Using entire dataset ({len(data)} days)")

    start_date, end_date = data.index[0], data.index[-1]
    print(f"   Date range: {start_date.date()} – {end_date.date()}")

    close = data["close"]
    high = data["high"]
    low = data["low"]

    if len(close) < 150:
        print(f"  ⚠️ Insufficient data: only {len(close)} bars — aborting")
        return pd.DataFrame()

    if OPTIMIZATION_METHOD == "genetic":
        max_iter = GENETIC_QUICK_MAX_ITER if QUICK_MODE else GENETIC_MAX_ITERATIONS
        print(f"\n🧬 Genetic optimization: {max_iter} generations (pop size: {GENETIC_POPULATION_SIZE * len(param_space)})")
    else:
        print(f"\n🔍 Bayesian optimization: {n_calls} calls ({n_random_starts} random starts)")

    best = optimize_parameters(
        data,
        start_date,
        end_date,
        param_space,
        GENETIC_QUICK_MAX_ITER if (OPTIMIZATION_METHOD == "genetic" and QUICK_MODE) else (GENETIC_MAX_ITERATIONS if OPTIMIZATION_METHOD == "genetic" else n_calls),
        n_random_starts,
    )

    if best is None:
        print("❌ Optimization failed")
        return pd.DataFrame()

    # Run strategy with best parameters
    strategy_func = get_strategy_function()
    entries, exits, position_target = strategy_func(close, high, low, **best)
    port = vbt.Portfolio.from_signals(
        close, entries, exits,
        size=position_target,
        size_type=SizeType.Percent,
        init_cash=CAPITAL_BASE,
        fees=MANUAL_DEFAULTS["commission_rate"],
        slippage=MANUAL_DEFAULTS["slippage_rate"],
        freq="1D"
    )
    stats = port.stats()

    composite, components = compute_composite_score(
        port, stats,
        {"short_period": best["short_period"], "long_period": best["long_period"]},
        len(close)
    )

    sortino = components.get("sortino_raw", np.nan)
    calmar = components.get("calmar_ratio", np.nan)
    sharpe = stats.get("Sharpe Ratio", np.nan)
    max_dd = stats.get("Max Drawdown [%]", 0) / 100
    total_return = stats.get("Total Return [%]", 0) / 100
    num_trades = port.trades.count()
    win_rate = port.trades.win_rate()

    print(f"\n{'='*70}")
    print(f"✅ QUICK OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"  Composite Score:   {composite:.2f}")
    print(f"  Sortino Ratio:     {sortino:.2f}")
    print(f"  Calmar Ratio:      {calmar:.2f}")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Total Return:      {total_return*100:+.1f}%")
    print(f"  Max Drawdown:      {max_dd*100:.1f}%")
    print(f"  Win Rate:          {win_rate*100:.1f}%")
    print(f"  Number of Trades:  {num_trades}")
    print(f"{'='*70}")

    # Extract parameter values for DataFrame
    result = {
        "start_date": start_date,
        "end_date": end_date,
        "days": len(close),
        "composite_score": composite,
        "sortino": sortino,
        "calmar": calmar,
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
        # Parameters (conditional based on SIMPLE_MODE)
        "Short Period": best["short_period"],
        "Long Period": best["long_period"],
        "ALMA Offset": best["alma_offset"],
        "ALMA Sigma": best["alma_sigma"],
        "Fast HMA Period": best["fast_hma_period"],
        "Slow EMA Period": best["slow_ema_period"],
        "Momentum Lookback": best["momentum_lookback"],
        "Slow EMA Rising Lookback": best["slow_ema_rising_lookback"],
    }

    # Add SIMPLE_MODE specific parameters
    if SIMPLE_MODE:
        result["Use Macro Filter"] = best.get("use_macro_filter", "N/A")
        result["Macro EMA Period"] = best.get("macro_ema_period", "N/A")
    else:
        # Add COMPLEX_MODE specific parameters
        result.update({
            "Trending ALMA Offset": best["trending_alma_offset"],
            "Ranging ALMA Offset": best["ranging_alma_offset"],
            "Trending ALMA Sigma": best["trending_alma_sigma"],
            "Ranging ALMA Sigma": best["ranging_alma_sigma"],
            "Trend Analysis Period": best["trend_analysis_period"],
            "Trend Threshold": best["trend_threshold"],
            "Weight Efficiency": best["weight_efficiency"],
            "Weight R-Squared": best["weight_rsquared"],
            "Macro EMA Period": best["macro_ema_period"],
            "Profit Period Scale": best["profit_period_scale_factor"],
        })

    print(f"\n💡 If these results look promising, set QUICK_MODE=False for full walk-forward testing")
    print(f"   (Walk-forward validation is required for production use)\n")

    return pd.DataFrame([result])


def walk_forward_analysis(data, param_space, n_calls, n_random_starts, stage_name=""):
    """Perform walk-forward optimization with data-driven bull-market detection."""

    print(f"\n{'='*70}\n{stage_name}\n{'='*70}")

    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data[data["time"] >= "2013-01-01"].copy()
    data.set_index("time", inplace=True)
    data = data.sort_index()

    start_date, end_date = data.index[0], data.index[-1]
    print(f"\nData range: {start_date.date()} – {end_date.date()} | {len(data)} days")

    print(f"\n📊 Detecting bull market segments (200 EMA + slope filter)...")
    bull_periods = detect_bull_market_periods(data)

    if len(bull_periods) == 0:
        print("❌ Error: No qualifying bull periods detected with current settings")
        return pd.DataFrame()

    print("\n   Bull market periods (detected):")
    for _, period in bull_periods.iterrows():
        print(f"   - {period['name']}: {period['start'].date()} to {period['end'].date()} "
              f"({period['days']} days, {period['return']*100:+.1f}% return)")

    iteration = 1
    wf_results = []

    for _, period in bull_periods.iterrows():
        segment = data.loc[period["start"]:period["end"]]
        windows = build_walk_forward_windows(
            segment.index,
            min_train_days=MIN_BULL_PERIOD_DAYS,
            min_test_days=MIN_TEST_DAYS,
            purge_days=PURGE_DAYS,
        )

        if len(windows) == 0:
            print(f"\n  ⚠️  Skipping {period['name']} (insufficient length for walk-forward windows)")
            continue

        print(f"\n🔄 {period['name']}: generated {len(windows)} anchored walk-forward window(s)")

        for window_id, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
            train_slice = data.loc[train_start:train_end]
            test_slice = data.loc[test_start:test_end]
            train_days = len(train_slice)
            test_days = len(test_slice)

            print(f"\n─ {period['name']} | Window {window_id}")
            print(f"   Train: {train_start.date()}–{train_end.date()} ({train_days} days)")
            print(f"   Test:  {test_start.date()}–{test_end.date()} ({test_days} days, purge={PURGE_DAYS} days)")

            best = optimize_parameters(
                data,
                train_start,
                train_end,
                param_space,
                GENETIC_MAX_ITERATIONS if OPTIMIZATION_METHOD == "genetic" else n_calls,
                n_random_starts,
            )
            if best is None:
                continue

            test_close = test_slice["close"]
            test_high = test_slice["high"]
            test_low = test_slice["low"]
            test_open = test_slice["open"]
            train_close = train_slice["close"]

            strategy_func = get_strategy_function()
            entries, exits, position_target = strategy_func(
                test_close, test_high, test_low, **best
            )
            port = vbt.Portfolio.from_signals(
                test_close, entries, exits,
                size=position_target,
                size_type=SizeType.Percent,
                init_cash=CAPITAL_BASE,
                fees=MANUAL_DEFAULTS["commission_rate"],
                slippage=MANUAL_DEFAULTS["slippage_rate"],
                freq="1D"
            )
            stats = port.stats()

            test_composite, test_components = compute_composite_score(
                port, stats,
                {"short_period": best["short_period"], "long_period": best["long_period"]},
                test_days
            )

            test_sortino = test_components.get("sortino_raw", np.nan)
            test_calmar = test_components.get("calmar_ratio", np.nan)
            train_sortino = best["train_sortino"]
            train_score = best["score"]

            num_test_trades = port.trades.count()
            constraint_violation = test_components.get("constraint_violation")
            if constraint_violation:
                print(f"  ⚠️  CONSTRAINT VIOLATION: {constraint_violation}")
                print("     Test period failed validation - metrics may be unreliable")

            if num_test_trades < 5:
                print(f"  ⚠️  WARNING: Only {num_test_trades} test trades - metrics may be noisy")

            if not np.isnan(test_sortino) and not np.isnan(train_sortino):
                if test_sortino > train_sortino * 1.5:
                    print(f"  ⚠️  WARNING: Test Sortino ({test_sortino:.2f}) >> Train ({train_sortino:.2f})")
                    print("     Suspiciously high - possible lucky stretch or data issue")
                elif test_sortino < train_sortino * 0.3:
                    print(f"  ⚠️  WARNING: Test Sortino ({test_sortino:.2f}) << Train ({train_sortino:.2f})")
                    print("     Severe degradation - likely overfitting")

            train_returns = np.log(train_close / train_close.shift(1)).dropna()
            test_returns = np.log(test_close / test_close.shift(1)).dropna()
            train_vol = train_returns.std()
            test_vol = test_returns.std()
            vol_regime = 'high' if test_vol > train_vol * 1.5 else ('low' if test_vol < train_vol * 0.67 else 'normal')

            bootstrap = block_bootstrap_metrics(
                port.returns(),
                block_size=BOOTSTRAP_BLOCK_SIZE,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED + iteration + window_id,
            )

            iter_name = f"Iter{iteration}_{period['name'].replace(' ', '')}_W{window_id}"
            stage_short = stage_name.replace(" ", "_").replace(":", "")

            benchmark_returns = test_close.pct_change().fillna(0)
            try:
                generate_quantstats_report(port, benchmark_returns, iter_name, stage_short)
            except Exception as e:
                print(f"  ⚠️  QuantStats report failed: {e}")

            try:
                generate_parameter_heatmaps(
                    data, test_start, test_end,
                    best, iter_name, stage_short
                )
            except Exception as e:
                print(f"  ⚠️  Heatmap generation failed: {e}")

            wf_record = {
                "iteration": iteration,
                "bull_period": period["name"],
                "window_id": window_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_days": train_days,
                "test_days": test_days,
                "train_buyhold_return": train_close.iloc[-1] / train_close.iloc[0] - 1,
                "test_buyhold_return": test_close.iloc[-1] / test_close.iloc[0] - 1,
                # Parameters (common to both modes)
                "Short Period": best["short_period"],
                "Long Period": best["long_period"],
                "ALMA Offset": best["alma_offset"],
                "ALMA Sigma": best["alma_sigma"],
                "Fast HMA Period": best["fast_hma_period"],
                "Slow EMA Period": best["slow_ema_period"],
                "Momentum Lookback": best["momentum_lookback"],
                "Slow EMA Rising Lookback": best["slow_ema_rising_lookback"],
            }

            # Add mode-specific parameters
            if SIMPLE_MODE:
                wf_record.update({
                    "Use Macro Filter": best.get("use_macro_filter", "N/A"),
                    "Macro EMA Period": best.get("macro_ema_period", "N/A"),
                })
            else:
                wf_record.update({
                    "Trending ALMA Offset": best["trending_alma_offset"],
                    "Ranging ALMA Offset": best["ranging_alma_offset"],
                    "Trending ALMA Sigma": best["trending_alma_sigma"],
                    "Ranging ALMA Sigma": best["ranging_alma_sigma"],
                    "Trend Analysis Period": best["trend_analysis_period"],
                    "Trend Threshold": best["trend_threshold"],
                    "Weight Efficiency": best["weight_efficiency"],
                    "Weight R-Squared": best["weight_rsquared"],
                    "Macro EMA Period": best["macro_ema_period"],
                    "Profit Scale Factor": best["profit_period_scale_factor"],
                })

            # Add metrics
            wf_record.update({
                # Metrics
                "train_composite": train_score,
                "train_sortino": train_sortino,
                "test_composite": test_composite,
                "test_sortino": test_sortino,
                "test_calmar": test_calmar,
                "test_return": port.total_return(),
                "test_sharpe": stats.get("Sharpe Ratio", np.nan),
                "test_max_dd": stats.get("Max Drawdown [%]", 0) / 100,
                "num_trades": num_test_trades,
                "win_rate": port.trades.win_rate(),
                "train_vol": train_vol,
                "test_vol": test_vol,
                "vol_regime": vol_regime,
            })
            wf_record.update(bootstrap)

            wf_results.append(wf_record)

            iteration += 1

    df = pd.DataFrame(wf_results)
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n==========================================")
    print("HERMES STRATEGY - TREND-ADAPTIVE OPTIMIZER")
    print("==========================================\n")

    # Select parameter space based on mode
    param_space = SIMPLE_SPACE if SIMPLE_MODE else STAGE1_SPACE
    mode_name = "SIMPLE (10 params)" if SIMPLE_MODE else "COMPLEX (18 params)"

    if SIMPLE_MODE:
        print("🔥 SIMPLE MODE ENABLED - Using old simple strategy (10 parameters)")
        print("   This matches hermes_old.pine with MACRO FILTER TOGGLE")
        print("   The optimizer can now test with macro filter ON or OFF!\n")
    else:
        print("🧪 COMPLEX MODE - Using advanced trend-adaptive strategy (18 parameters)\n")

    if QUICK_MODE:
        print("⚡ QUICK MODE ENABLED - Fast single-period optimization")
        print("   (Set QUICK_MODE=False for full walk-forward testing)\n")
    else:
        print("🔄 FULL MODE - Walk-forward optimization with bull market detection\n")

    asset_data_map = load_all_asset_data(ASSET_DATA_SOURCES)
    if not asset_data_map:
        print("✗ No asset datasets found. Provide CSV files defined in ASSET_DATA_SOURCES.")
        return

    aggregated_stage1 = []
    aggregated_stage2 = []

    for asset_name, asset_df in asset_data_map.items():
        print(f"\n==================== {asset_name} ====================")
        print(f"Rows available: {len(asset_df)}")

        if QUICK_MODE:
            # Quick mode: single optimization, no walk-forward
            stage1 = quick_optimize(
                asset_df.copy(),
                param_space,
                QUICK_MODE_CALLS,
                QUICK_MODE_RANDOM_STARTS,
                f"{asset_name} | {mode_name} QUICK TEST"
            )
        else:
            # Full mode: walk-forward with bull market detection
            stage1 = walk_forward_analysis(
                asset_df.copy(),
                param_space,
                STAGE1_CALLS,
                STAGE1_RANDOM_STARTS,
                f"{asset_name} | {mode_name} STAGE 1"
            )
        if len(stage1) == 0:
            print(f"✗ {asset_name}: Stage 1 produced no valid windows.")
            continue

        stage1["asset"] = asset_name
        if QUICK_MODE:
            stage1_file = Path(f"hermes_quick_{asset_name}.csv")
        else:
            stage1_file = Path(f"hermes_stage1_{asset_name}.csv")
        stage1.to_csv(stage1_file, index=False)
        aggregated_stage1.append(stage1)

        # Skip Stage 2 in quick mode or simple mode (Stage 2 is for complex strategy only)
        if QUICK_MODE:
            print(f"\n✅ {asset_name} Quick Test Complete")
            print(f"   Results saved to: {stage1_file}")
            continue

        if SIMPLE_MODE:
            print(f"\n✅ {asset_name} SIMPLE MODE Complete - Stage 2 not applicable")
            print(f"   Results saved to: {stage1_file}")
            continue

        best_region = stage1.nlargest(5, "test_composite").mean(numeric_only=True).to_dict()

        # Build focused Stage 2 space
        stage2_space = [
            Integer(max(15, int(best_region["Short Period"]) - 20),
                    min(100, int(best_region["Short Period"]) + 20),
                    name="short_period"),
            Integer(max(100, int(best_region["Long Period"]) - 50),
                    min(400, int(best_region["Long Period"]) + 50),
                    name="long_period"),
            Integer(max(85, int(best_region["ALMA Offset"] * 100) - 5),
                    min(99, int(best_region["ALMA Offset"] * 100) + 5),
                    name="alma_offset_int"),
            Categorical([v for v in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                         if max(3.0, best_region["ALMA Sigma"] - 2.0) <= v <= min(10.0, best_region["ALMA Sigma"] + 2.0)],
                        name="alma_sigma"),
            Integer(max(70, int(best_region["Trending ALMA Offset"] * 100) - 5),
                    min(90, int(best_region["Trending ALMA Offset"] * 100) + 5),
                    name="trending_alma_offset_int"),
            Integer(max(90, int(best_region["Ranging ALMA Offset"] * 100) - 3),
                    min(99, int(best_region["Ranging ALMA Offset"] * 100) + 3),
                    name="ranging_alma_offset_int"),
            Categorical([v for v in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                         if max(4.0, best_region["Trending ALMA Sigma"] - 2.0) <= v <= min(12.0, best_region["Trending ALMA Sigma"] + 2.0)],
                        name="trending_alma_sigma"),
            Categorical([v for v in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
                         if max(2.0, best_region["Ranging ALMA Sigma"] - 2.0) <= v <= min(7.0, best_region["Ranging ALMA Sigma"] + 2.0)],
                        name="ranging_alma_sigma"),
            Integer(max(30, int(best_region["Trend Analysis Period"]) - 20),
                    min(100, int(best_region["Trend Analysis Period"]) + 20),
                    name="trend_analysis_period"),
            Integer(max(40, int(best_region["Trend Threshold"] * 100) - 10),
                    min(80, int(best_region["Trend Threshold"] * 100) + 10),
                    name="trend_threshold_int"),
            Integer(max(50, int(best_region["Weight Efficiency"] * 100) - 10),
                    min(90, int(best_region["Weight Efficiency"] * 100) + 10),
                    name="weight_efficiency_int"),
            Integer(max(10, int(best_region["Weight R-Squared"] * 100) - 10),
                    min(50, int(best_region["Weight R-Squared"] * 100) + 10),
                    name="weight_rsquared_int"),
            Integer(max(20, int(best_region["Fast HMA Period"]) - 10),
                    min(100, int(best_region["Fast HMA Period"]) + 10),
                    name="fast_hma_period"),
            Integer(max(50, int(best_region["Slow EMA Period"]) - 20),
                    min(150, int(best_region["Slow EMA Period"]) + 20),
                    name="slow_ema_period"),
            Integer(max(1, int(best_region["Momentum Lookback"]) - 2),
                    min(10, int(best_region["Momentum Lookback"]) + 2),
                    name="momentum_lookback"),
            Integer(max(1, int(best_region["Slow EMA Rising Lookback"]) - 2),
                    min(10, int(best_region["Slow EMA Rising Lookback"]) + 2),
                    name="slow_ema_rising_lookback"),
            Integer(max(100, int(best_region["Macro EMA Period"]) - 30),
                    min(250, int(best_region["Macro EMA Period"]) + 30),
                    name="macro_ema_period"),
            Integer(max(0, int(best_region["Profit Scale Factor"] * 100) - 2),
                    min(10, int(best_region["Profit Scale Factor"] * 100) + 2),
                    name="profit_scale_int"),
        ]

        stage2 = walk_forward_analysis(
            asset_df.copy(),
            stage2_space,
            STAGE2_CALLS,
            STAGE2_RANDOM_STARTS,
            f"{asset_name} | STAGE 2: FOCUSED SEARCH"
        )

        if len(stage2) == 0:
            print(f"⚠️  {asset_name}: Stage 2 produced no valid windows.")
        else:
            stage2["asset"] = asset_name
            stage2_file = Path(f"hermes_stage2_{asset_name}.csv")
            stage2.to_csv(stage2_file, index=False)
            aggregated_stage2.append(stage2)

        if not QUICK_MODE:  # Only print summary for full mode
            print(f"\n📊 {asset_name} Summary:")
            print(f"  Stage 1 avg test composite: {stage1['test_composite'].mean():.2f} "
                  f"(Sortino: {stage1['test_sortino'].mean():.2f})")
            if len(stage2) > 0:
                print(f"  Stage 2 avg test composite: {stage2['test_composite'].mean():.2f} "
                      f"(Sortino: {stage2['test_sortino'].mean():.2f})")
                print(f"  Stage 2 avg Calmar ratio: {stage2['test_calmar'].mean():.2f}")
            print("  Reports: reports/quantstats & reports/heatmaps")

    if len(aggregated_stage1) > 1:
        combined_stage1 = pd.concat(aggregated_stage1, ignore_index=True)
        if QUICK_MODE:
            combined_stage1.to_csv("hermes_quick_all_assets.csv", index=False)
            print(f"\n✅ Combined quick results saved to: hermes_quick_all_assets.csv")
        else:
            combined_stage1.to_csv("hermes_stage1_all_assets.csv", index=False)
    if len(aggregated_stage2) > 1:
        combined_stage2 = pd.concat(aggregated_stage2, ignore_index=True)
        combined_stage2.to_csv("hermes_stage2_all_assets.csv", index=False)


if __name__ == "__main__":
    main()
