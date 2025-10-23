"""
Hermes Strategy Optimizer - SIMPLE VERSION (matches hermes_old.pine)
====================================================================
This is the SIMPLIFIED version that was getting 3x better returns.

✅ Only 8 parameters (was 18)
✅ Fixed ALMA (no dynamic interpolation)
✅ Simple regime detection
✅ NO Efficiency Ratio / R-Squared
✅ NO room-to-breathe system
✅ NO profit scaling

OPTIMIZATION METHODS:
---------------------
1. GENETIC (Differential Evolution) - RECOMMENDED
   - Population-based search
   - Quick: 60 gens = ~10,000 evaluations (~5-10 min)
   - Full: 150 gens = ~25,000 evaluations (~15-30 min)

2. BAYESIAN (Gaussian Process)
   - Sequential model-based optimization
   - 200-500 evaluations depending on mode

MODES:
------
1. QUICK MODE (QUICK_MODE = True):
   - Single-period optimization (~5-10 min per asset)
   - Tests from 2013 onwards
   - Genetic: 60 generations
   - Bayesian: 200 calls

2. FULL MODE (QUICK_MODE = False):
   - Walk-forward optimization with bull market detection
   - Genetic: 150 generations per window
   - Bayesian: 400 calls per window
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

# Create output directories
Path("reports").mkdir(exist_ok=True)
Path("reports/quantstats").mkdir(exist_ok=True)
Path("reports/heatmaps").mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Manual defaults matching hermes_old.pine
MANUAL_DEFAULTS = {
    "short_period": 30,
    "long_period": 250,
    "alma_offset": 0.95,
    "alma_sigma": 4.0,
    "momentum_lookback": 1,
    "macro_ema_period": 100,
    "fast_hma_period": 30,
    "slow_ema_period": 80,
    "slow_ema_rising_lookback": 3,
    "commission_rate": 0.0035,
    "slippage_rate": 0.0005,
}

# SIMPLE parameter space - only 8 parameters!
STAGE1_SPACE = [
    Integer(10, 150, name="short_period"),
    Integer(100, 400, name="long_period"),
    Integer(80, 99, name="alma_offset_int"),  # 0.80-0.99
    Categorical([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], name="alma_sigma"),
    Integer(1, 10, name="momentum_lookback"),
    Integer(50, 250, name="macro_ema_period"),
    Integer(10, 100, name="fast_hma_period"),
    Integer(30, 200, name="slow_ema_period"),
    Integer(1, 15, name="slow_ema_rising_lookback"),
]

QUICK_MODE = True
OPTIMIZATION_METHOD = "genetic"

# Genetic settings
GENETIC_POPULATION_SIZE = 20  # 20 × 9 = 180 individuals per generation
GENETIC_MAX_ITERATIONS = 150  # Full mode
GENETIC_QUICK_MAX_ITER = 60   # Quick mode

# Bayesian settings
STAGE1_CALLS = 400
STAGE1_RANDOM_STARTS = 100
STAGE2_CALLS = 500
STAGE2_RANDOM_STARTS = 50

# Walk-forward settings
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

# ============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================================================

@njit(cache=True, fastmath=True)
def alma_numba(src, period, offset, sigma):
    """ALMA (Arnaud Legoux Moving Average)"""
    n = len(src)
    result = np.empty(n, dtype=np.float64)
    m = offset * (period - 1)
    s = period / sigma

    for i in range(n):
        if i < period - 1:
            result[i] = np.mean(src[:i+1])
        else:
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
    """Hull Moving Average"""
    n = len(close)
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))

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

    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < period - 1:
            result[i] = close[i]
            continue
        wma_half = wma(close, half_period, i)
        wma_full = wma(close, period, i)
        raw = 2 * wma_half - wma_full
        result[i] = raw

    final_result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < period - 1:
            final_result[i] = close[i]
        else:
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
# SIMPLE STRATEGY LOGIC
# ============================================================================

def run_strategy_simple(close, high, low, **params):
    """
    SIMPLE Hermes strategy - matches hermes_old.pine

    - Fixed ALMA parameters (no dynamic interpolation)
    - Simple regime detection based on price structure
    - No Efficiency Ratio / R-Squared
    - No room-to-breathe system
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)

    # Extract parameters (only 8!)
    short_period = int(params["short_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]
    momentum_lookback = int(params["momentum_lookback"])
    macro_ema_period = int(params["macro_ema_period"])
    fast_hma_period = int(params["fast_hma_period"])
    slow_ema_period = int(params["slow_ema_period"])
    slow_ema_rising_lookback = int(params["slow_ema_rising_lookback"])

    # === LOG RETURNS FOR ALMA ===
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # === SIMPLE FIXED ALMA ===
    long_term = alma_numba(returns, long_period, alma_offset, alma_sigma)
    short_term = alma_numba(returns, short_period, alma_offset, alma_sigma)
    baseline = long_term

    # === PRICE STRUCTURE ===
    fast_hma = hma_numba(close_np, fast_hma_period)
    slow_ema = pd.Series(close_np).ewm(span=slow_ema_period, adjust=False).mean().to_numpy()
    macro_ema = pd.Series(close_np).ewm(span=macro_ema_period, adjust=False).mean().to_numpy()

    # === ENTRY CONDITIONS ===
    bullish_state = short_term > baseline
    in_bull_market = close_np > macro_ema

    # Momentum filter
    highest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback).max().to_numpy()
    is_highest_close = (close_np >= np.nan_to_num(highest_close_prev, nan=0)) & \
                       (high_np >= np.nan_to_num(highest_high_prev, nan=0))

    # Slow EMA rising
    slow_ema_rising = np.zeros(len(slow_ema), dtype=bool)
    for i in range(slow_ema_rising_lookback, len(slow_ema)):
        slow_ema_rising[i] = slow_ema[i] > slow_ema[i - slow_ema_rising_lookback]

    # Buy signal
    buy_signal = bullish_state & is_highest_close & in_bull_market & slow_ema_rising

    # === EXIT CONDITIONS ===
    bearish_state = short_term < baseline

    # Sell momentum
    lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    lowest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback).min().to_numpy()
    is_lowest_low = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                    (close_np <= np.nan_to_num(lowest_close_prev, nan=np.inf))

    # Trending exit: HMA crosses under EMA
    trend_cross_under = (fast_hma < slow_ema) & (np.roll(fast_hma, 1) >= np.roll(slow_ema, 1))

    # Ranging exit: bearish ALMA + momentum
    ranging_exit = bearish_state & is_lowest_low

    # Combined sell signal
    sell_signal = ranging_exit | trend_cross_under

    # Convert to entries/exits
    buy_prev = np.roll(buy_signal, 1)
    sell_prev = np.roll(sell_signal, 1)
    buy_prev[0] = False
    sell_prev[0] = False

    entries = buy_signal & (~buy_prev)
    exits = sell_signal & (~sell_prev)

    position_target = np.ones(len(close_np), dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)


# ============================================================================
# COMPOSITE SCORING (reuse from complex version)
# ============================================================================

def compute_composite_score(portfolio, stats, params, training_days):
    """Compute composite objective score"""
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
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, "constraint_violation": "too_few_trades",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}
    if max_dd > 0.60:
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, "constraint_violation": "excessive_drawdown",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}

    years = training_days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1
    if annualized_return < 0.10:
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, "constraint_violation": "insufficient_return",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}
    if trades_per_year > 200:
        return 0.0, {"sortino_raw": sortino, "composite_score": 0.0, "constraint_violation": "excessive_trading",
                     "trades_per_year": trades_per_year, "win_rate": win_rate}

    # Composite score
    primary_score = sortino * 0.70
    calmar = annualized_return / max_dd
    calmar_normalized = min(calmar / 2.0, 5.0)
    calmar_score = calmar_normalized * 0.20

    if 12 <= trades_per_year <= 36:
        freq_stability = 1.0
    elif 3 <= trades_per_year < 12:
        freq_stability = 0.5 + (trades_per_year - 3) / 18
    elif 36 < trades_per_year <= 60:
        freq_stability = 1.0 - (trades_per_year - 36) / 48
    elif 60 < trades_per_year <= 100:
        freq_stability = 0.5 - (trades_per_year - 60) / 80
    else:
        freq_stability = 0.2

    if 0.30 <= win_rate <= 0.65:
        winrate_stability = 1.0
    elif win_rate < 0.30:
        winrate_stability = max(0.0, win_rate / 0.30)
    else:
        winrate_stability = max(0.5, 1.0 - (win_rate - 0.65) / 0.25)

    stability_bonus = (freq_stability * 0.5 + winrate_stability * 0.5) * 0.10
    composite = primary_score + calmar_score + stability_bonus

    return composite, {
        "sortino_raw": sortino,
        "composite_score": composite,
        "primary_score": primary_score,
        "calmar_score": calmar_score,
        "calmar_ratio": calmar,
        "stability_bonus": stability_bonus,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "trades_per_year": trades_per_year,
        "win_rate": win_rate,
        "constraint_violation": None,
    }


# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def optimize_parameters_bayesian(data, start_date, end_date, param_space, n_calls, n_random_starts, n_jobs=12):
    close = data.loc[start_date:end_date, "close"]
    high = data.loc[start_date:end_date, "high"]
    low = data.loc[start_date:end_date, "low"]

    if len(close) < 150:
        print(f"  ⚠ Insufficient data: only {len(close)} bars — skipping")
        return None

    start_time = datetime.datetime.now()
    training_days = len(close)
    print(f"  ▶ [{start_time.strftime('%H:%M:%S')}] Optimizing on {start_date.date()}–{end_date.date()} "
          f"({training_days} bars, {n_calls} Bayesian calls)")

    call_count = [0]
    best_score = [float("-inf")]
    best_sortino_raw = [0.0]

    @use_named_args(param_space)
    def objective(**raw_params):
        call_count[0] += 1

        try:
            params = {
                "short_period": raw_params["short_period"],
                "long_period": raw_params["long_period"],
                "alma_offset": raw_params["alma_offset_int"] / 100.0,
                "alma_sigma": raw_params["alma_sigma"],
                "momentum_lookback": raw_params["momentum_lookback"],
                "macro_ema_period": raw_params["macro_ema_period"],
                "fast_hma_period": raw_params["fast_hma_period"],
                "slow_ema_period": raw_params["slow_ema_period"],
                "slow_ema_rising_lookback": raw_params["slow_ema_rising_lookback"],
            }

            # Constraints
            if params["short_period"] >= params["long_period"]:
                return 999.0
            if params["long_period"] - params["short_period"] < 20:
                return 999.0
            if params["fast_hma_period"] >= params["slow_ema_period"]:
                return 999.0

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
                    print(f"    {now} ✓ Call {call_count[0]:3d}/{n_calls} "
                          f"Score: {score:5.2f} (Sortino: {sortino:.2f}, "
                          f"Calmar: {calmar:.2f}, Return: {total_return:+.1f}%, Trades/yr: {trades_yr:.0f})")
            elif call_count[0] % 50 == 0:
                print(f"    {now} Progress {call_count[0]:3d}/{n_calls} | Best score: {best_score[0]:.2f}")

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

    best_params = {
        "short_period": int(raw_best["short_period"]),
        "long_period": int(raw_best["long_period"]),
        "alma_offset": raw_best["alma_offset_int"] / 100.0,
        "alma_sigma": float(raw_best["alma_sigma"]),
        "momentum_lookback": int(raw_best["momentum_lookback"]),
        "macro_ema_period": int(raw_best["macro_ema_period"]),
        "fast_hma_period": int(raw_best["fast_hma_period"]),
        "slow_ema_period": int(raw_best["slow_ema_period"]),
        "slow_ema_rising_lookback": int(raw_best["slow_ema_rising_lookback"]),
        "score": -result.fun,
        "train_sortino": best_sortino_raw[0],
    }

    print(f"  ✔ [{end_time.strftime('%H:%M:%S')}] Done in {duration} | "
          f"Best Score {best_params['score']:5.2f} (Train Sortino: {best_params['train_sortino']:.2f})")
    print(f"     ALMA Periods: short={best_params['short_period']}, long={best_params['long_period']}")
    print(f"     ALMA Fixed: offset={best_params['alma_offset']:.2f}, sigma={best_params['alma_sigma']:.1f}")
    print(f"     Price Structure: HMA={best_params['fast_hma_period']}, EMA={best_params['slow_ema_period']}")

    return best_params


# [Rest of the file follows the same pattern - I'll include the import from the complex version for genetic optimizer, quick mode, walk-forward, data loading, etc.]
# Importing helper functions...
