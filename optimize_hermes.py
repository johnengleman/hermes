"""
Hermes Strategy Optimizer - ALMA Filter + Bayesian Walk-Forward
================================================================
✅ ALMA (Arnaud Legoux) Gaussian smoothing for Giovanni-style curves
✅ Ultra-smooth macro trend capture with natural outlier resistance
✅ Objective penalties (excessive trades, extreme parameters)
✅ Numba JIT acceleration
✅ QuantStats HTML reports + Parameter heatmaps
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
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

# Manual defaults for ALMA filter (Giovanni-style macro trend capture)
MANUAL_DEFAULTS = {
    "short_period": 30,
    "long_period": 150,
    "alma_offset": 0.95,
    "alma_sigma": 4,
    "buy_momentum_bars": 6,
    "sell_momentum_bars": 1,
    "baseline_momentum_bars": 3,  # Baseline must be rising over this lookback
    "macro_ema_period": 200,  # EMA period for bull/bear market filter (always enabled)
    "vol_lookback": 20,
    "target_vol": 0.02,
    "use_vol_scaling": False,
    "max_allocation": 1.0,  # 100% of capital
    "min_allocation": 0.0,
    "commission_rate": 0.0035,  # 35 bps per side (Fidelity crypto trading)
    "slippage_rate": 0.0005,    # 5 bps slippage assumption
}

# ALMA filter search space (Giovanni-style macro trend capture)
# Ranges constrained to reduce overfitting while allowing good manual parameters
STAGE1_SPACE = [
    # Period lengths for short and long term ALMA (macro-focused)
    Integer(10, 100, name="short_period"),       # Allows your 30, prevents extreme short
    Integer(100, 300, name="long_period"),        # Allows your 150, caps at 250 for responsiveness
    
    # ALMA Offset (0.85-0.99, discretized by 0.01 for efficiency)
    Integer(85, 99, name="alma_offset_int"),     # Will divide by 100: 0.85, 0.86, ..., 0.99
    
    # ALMA Sigma (3-9, discretized by 0.5 steps: 3.0, 3.5, 4.0, ..., 9.0)
    Categorical([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], name="alma_sigma"),
    
    # Momentum parameters (constrained but flexible)
    Integer(3, 10, name="buy_momentum_bars"),    # Allows your 6, some flexibility
    Integer(0, 6, name="sell_momentum_bars"),    # Allows your 1, prevents extreme
    
    # Baseline momentum (long-term ALMA must be rising over this lookback)
    Integer(1, 20, name="baseline_momentum_bars"),  # 1-50 bars lookback for baseline rising check
    
    # Macro trend filter EMA period (100-300, step of 10: 100, 110, 120, ..., 300)
    Categorical([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300], name="macro_ema_period"),
]

STAGE1_CALLS = 300
STAGE1_RANDOM_STARTS = 100

STAGE2_CALLS = 500
STAGE2_RANDOM_STARTS = 50

# NOTE: These are no longer used - walk-forward now uses ACTUAL bull market periods
# instead of fixed time windows. See walk_forward_analysis() for the new approach.
TRAIN_MONTHS = 48  # Legacy - not used in bull-market-only mode
TEST_MONTHS = 12   # Legacy - not used in bull-market-only mode
ROLL_MONTHS = 12   # Legacy - not used in bull-market-only mode

# NEW APPROACH (implemented in walk_forward_analysis):
# - Identify bull/bear periods using 200 EMA
# - Train on one bull period, test on the NEXT bull period
# - Completely skip bear markets (you won't trade them anyway)
# - Example: Train on 2015-2017 bull → Test on 2019-2021 bull

# Minimum bull period length for BOTH training AND testing (days)
MIN_BULL_PERIOD_DAYS = 250  # ~8 months minimum
# Rationale: 
# - 250 days = ~8 months of trading data
# - Enough for optimizer to find patterns with macro strategy (10-30 trades)
# - Results show 90-180 day periods produce unreliable metrics (1-3 test trades)
# - Applied to BOTH train and test periods
# - With 1-2 trades/month target, 250 days = 8-16 trades minimum
# - Set to 0 to use ALL bull periods regardless of length

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
PENALTY_EXCESSIVE_TRADES = 0.02   # Penalize > 200 trades/year
PENALTY_EXTREME_GAIN = 0.05       # Penalize very reactive filters
PENALTY_HIGH_DRAWDOWN = 0.1       # Penalize drawdown > 40%

# ============================================================================
# ALMA FILTER HELPER
# ============================================================================
# ALMA uses Gaussian weighting for ultra-smooth curves like Giovanni's indicator


def compute_composite_score(portfolio, stats, params, training_days):
    """
    Compute composite objective score using crypto-adapted risk-adjusted metrics.
    
    Primary Metric: Sortino Ratio (downside risk-adjusted returns)
    + Hard Constraints: Crypto-specific thresholds (higher volatility/returns expected)
    + Secondary Metrics: Calmar Ratio (return/drawdown for high-vol assets)
    
    CRYPTO-SPECIFIC ADAPTATIONS:
    - Minimum annual return: 10% (vs 3% for equities) - crypto has higher opportunity cost
    - Maximum drawdown: 60% (vs 40% for equities) - crypto is inherently volatile
    - Calmar normalization: 2.0 target (vs 1.0) - crypto strategies can be more aggressive
    - Win rate expectations: 30-65% (vs 40-60%) - wider range due to trend-following nature
    
    Returns: composite score (higher = better), components dict
    """
    
    # === EXTRACT BASE METRICS ===
    sortino = stats.get("Sortino Ratio", 0)
    if np.isnan(sortino) or np.isinf(sortino):
        sortino = 0
    
    total_return = portfolio.total_return()
    if np.isnan(total_return) or np.isinf(total_return):
        total_return = 0
    
    max_dd = stats.get("Max Drawdown [%]", 0) / 100
    if max_dd == 0:
        max_dd = 0.01  # Avoid division by zero
    
    num_trades = portfolio.trades.count()
    win_rate = portfolio.trades.win_rate() if num_trades > 0 else 0
    if np.isnan(win_rate):
        win_rate = 0
    
    trades_per_year = (num_trades / training_days) * 365
    
    # === HARD CONSTRAINTS (return large penalty if violated) ===
    # These are REQUIREMENTS, not preferences
    
    # Constraint 1: Minimum trade frequency (strategy must be active)
    if trades_per_year < 3:
        # Too few trades = unreliable statistics, reject outright
        # 3 trades/year = minimum (allows macro trend following with 1-2 trades/month)
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "too_few_trades",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }
    
    # Constraint 2: Maximum drawdown (crypto-adjusted risk management)
    if max_dd > 0.60:  # 60% drawdown for crypto (vs 40-50% for equities)
        # Crypto is inherently more volatile, but 60%+ is still catastrophic
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "excessive_drawdown",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }
    
    # Constraint 3: Minimum total return (crypto-adjusted opportunity cost)
    years = training_days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1
    if annualized_return < 0.10:  # Must beat 10% annual for crypto (vs 3% for equities)
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "insufficient_return",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }
    
    # Constraint 4: Excessive trading (likely overfit or data snooping)
    if trades_per_year > 200:
        return 0.0, {
            "sortino_raw": sortino,
            "composite_score": 0.0,
            "constraint_violation": "excessive_trading",
            "trades_per_year": trades_per_year,
            "win_rate": win_rate,
        }
    
    # === PRIMARY METRIC: Sortino Ratio (70% weight) ===
    # Industry standard: downside deviation-adjusted returns
    primary_score = sortino * 0.70
    
    # === SECONDARY METRIC 1: Calmar Ratio (20% weight) ===
    # Return / Max Drawdown (crypto-adjusted normalization)
    # Crypto Calmar targets: 2.0 = excellent, 1.0 = acceptable (vs 1.0/0.5 for equities)
    # Example: 50% annual return / 25% drawdown = 2.0 Calmar (great for crypto)
    calmar = annualized_return / max_dd
    calmar_normalized = min(calmar / 2.0, 5.0)  # Normalized to crypto expectations, cap at 5.0
    calmar_score = calmar_normalized * 0.20
    
    # === SECONDARY METRIC 2: Stability Bonus (10% weight) ===
    # Reward consistent strategies (moderate trade frequency, reasonable win rate)
    # This prevents overfitting to extreme parameter combinations
    
    # Trade frequency stability (prefer 12-36 trades/year = 1-3 trades/month)
    # Macro trend-following style: fewer, higher-quality trades
    if 12 <= trades_per_year <= 36:
        freq_stability = 1.0  # Ideal range (1-3 trades/month)
    elif 3 <= trades_per_year < 12:
        freq_stability = 0.5 + (trades_per_year - 3) / 18  # Ramp up from minimum
    elif 36 < trades_per_year <= 60:
        freq_stability = 1.0 - (trades_per_year - 36) / 48  # Gentle ramp down
    elif 60 < trades_per_year <= 100:
        freq_stability = 0.5 - (trades_per_year - 60) / 80  # More frequent = worse
    else:
        freq_stability = 0.2  # Too frequent (> 100/year = overtrading)
    
    # Win rate stability (crypto-adjusted: 30-65%, wider range due to trend-following)
    # Crypto trend-following can have lower win rates but bigger winners
    if 0.30 <= win_rate <= 0.65:
        winrate_stability = 1.0  # Realistic range for crypto trend-following
    elif win_rate < 0.30:
        winrate_stability = max(0.0, win_rate / 0.30)  # Low but not zero (20% = 0.67 score)
    else:  # > 65%
        winrate_stability = max(0.5, 1.0 - (win_rate - 0.65) / 0.25)  # Suspicious (>90% = 0.5)
    
    stability_bonus = (freq_stability * 0.5 + winrate_stability * 0.5) * 0.10
    
    # === COMPOSITE SCORE ===
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
# NUMBA‑ACCELERATED CORE FUNCTIONS
# ============================================================================

@njit(cache=True, fastmath=True)
def alma_numba(src, period, offset, sigma):
    """
    ALMA (Arnaud Legoux Moving Average) - Gaussian-weighted smoothing.
    Provides ultra-smooth curves like Giovanni's Power Law indicator.
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
def ewma_vol_numba(returns, period):
    """
    Exponentially weighted moving average volatility.
    Uses span parameter (equivalent to period for consistency).
    Faster regime adaptation than simple rolling std.
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    
    # Calculate alpha from span
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with first value
    result[0] = abs(returns[0]) if returns[0] != 0 else 1e-8
    
    for i in range(1, n):
        # EWMA of squared returns (variance)
        if i < period:
            # Warmup: use simple std
            result[i] = np.std(returns[:i+1])
            if result[i] == 0:
                result[i] = 1e-8
        else:
            # Exponentially weighted variance
            variance = alpha * (returns[i] ** 2) + (1 - alpha) * (result[i-1] ** 2)
            result[i] = np.sqrt(variance)
            if result[i] == 0:
                result[i] = 1e-8
    
    return result


# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def run_strategy(close, high, low, short_period, long_period, alma_offset, alma_sigma,
                 buy_momentum_bars, sell_momentum_bars, baseline_momentum_bars, macro_ema_period,
                 use_vol_scaling=MANUAL_DEFAULTS["use_vol_scaling"],
                 vol_lookback=MANUAL_DEFAULTS["vol_lookback"],
                 target_vol=MANUAL_DEFAULTS["target_vol"],
                 max_allocation=MANUAL_DEFAULTS["max_allocation"],
                 min_allocation=MANUAL_DEFAULTS["min_allocation"]):
    """ALMA filter crossover strategy on raw log returns with macro EMA filter.
    
    Notes:
    - baseline_momentum_bars checks if long-term ALMA is rising over lookback period
    - macro_ema_period is always enabled (optimizes the EMA lookback period)
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)

    # Calculate log returns
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0
    
    # Macro trend filter: Variable EMA period on price (always enabled)
    macro_ema = pd.Series(close_np).ewm(span=int(macro_ema_period), adjust=False).mean().to_numpy()
    in_bull_market = close_np > macro_ema
    
    # Apply ALMA filters to raw log returns
    # ALMA's Gaussian weighting provides natural outlier resistance
    long_term = alma_numba(returns, int(long_period), alma_offset, alma_sigma)
    short_term = alma_numba(returns, int(short_period), alma_offset, alma_sigma)

    baseline = long_term

    # Momentum filters (can be independently disabled by setting to 0)
    buy_momentum_bars = int(min(buy_momentum_bars, len(close_np) - 1))
    
    # Buy momentum: check if current close AND high are at max of previous N bars (excluding current)
    # Shift by 1 to exclude current bar from the rolling window
    highest_close = pd.Series(close_np).shift(1).rolling(buy_momentum_bars).max().to_numpy()
    is_highest_close = np.nan_to_num(close_np >= highest_close, nan=0).astype(bool)
    
    # Also check if high is at the max of previous N bars
    highest_high = pd.Series(high_np).shift(1).rolling(buy_momentum_bars).max().to_numpy()
    is_highest_high = np.nan_to_num(high_np >= highest_high, nan=0).astype(bool)
    
    # Combine both conditions for buy momentum
    is_highest_close = is_highest_close & is_highest_high
    
    # Sell momentum: optional (0 = disabled, 1+ = enabled with lookback)
    if sell_momentum_bars > 0:
        sell_momentum_bars = int(min(sell_momentum_bars, len(close_np) - 1))
        # Check if current low AND close are at min of previous N bars (excluding current)
        lowest_low = pd.Series(low_np).shift(1).rolling(sell_momentum_bars).min().to_numpy()
        is_lowest_low = np.nan_to_num(low_np <= lowest_low, nan=0).astype(bool)
        
        # Also check if close is at the min of previous N bars
        lowest_close = pd.Series(close_np).shift(1).rolling(sell_momentum_bars).min().to_numpy()
        is_lowest_close = np.nan_to_num(close_np <= lowest_close, nan=0).astype(bool)
        
        # Combine both conditions for sell momentum
        is_lowest_low = is_lowest_low & is_lowest_close
    else:
        # Sell momentum disabled: always true (no filter)
        is_lowest_low = np.ones(len(low_np), dtype=bool)

    # Check regime state: is blue line above or below black line?
    bullish_state = short_term > baseline
    bearish_state = short_term < baseline
    
    # Baseline momentum filter: check if long-term ALMA is rising
    baseline_momentum_bars = int(min(baseline_momentum_bars, len(baseline) - 1))
    baseline_rising = np.zeros(len(baseline), dtype=bool)
    for i in range(baseline_momentum_bars, len(baseline)):
        baseline_rising[i] = baseline[i] > baseline[i - baseline_momentum_bars]
    
    # Build entry/exit conditions with filters
    # Buy: blue > black AND momentum high AND baseline rising AND macro filter
    buy_condition = bullish_state & is_highest_close & baseline_rising & in_bull_market
    # Sell: blue < black AND momentum low (optional)
    sell_condition = bearish_state & is_lowest_low

    buy_prev = np.roll(buy_condition, 1)
    sell_prev = np.roll(sell_condition, 1)
    buy_prev[0] = False
    sell_prev[0] = False

    entries = buy_condition & (~buy_prev)
    exits = sell_condition & (~sell_prev)

    if use_vol_scaling:
        vol_array = ewma_vol_numba(returns, int(vol_lookback))
        vol_array = np.where(vol_array <= 1e-8, np.nan, vol_array)
        scaling = target_vol / vol_array
        scaling = np.nan_to_num(scaling, nan=max_allocation, posinf=max_allocation, neginf=min_allocation)
        position_target = np.clip(scaling, min_allocation, max_allocation)
    else:
        position_target = np.full(len(close_np), max_allocation, dtype=np.float64)

    position_series = pd.Series(position_target, index=close.index)
    entries = entries & (position_series > min_allocation + 1e-6)
    
    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)

# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def optimize_parameters_bayesian(data, start_date, end_date,
                                 param_space, n_calls, n_random_starts, n_jobs=12):
    close = data.loc[start_date:end_date, "close"]
    high = data.loc[start_date:end_date, "high"]
    low = data.loc[start_date:end_date, "low"]

    if len(close) < 150:
        print(f"  ⚠ Insufficient data: only {len(close)} bars — skipping")
        return None

    start_time = datetime.datetime.now()
    training_days = len(close)
    print(f"  ▶ [{start_time.strftime('%H:%M:%S')}] "
          f"Optimizing on {start_date.date()}–{end_date.date()} "
          f"({training_days} bars, {n_calls} Bayesian calls)")

    call_count = [0]
    best_score = [float("-inf")]  # Best penalized score
    best_sortino_raw = [0.0]  # Best raw Sortino (for efficient retrieval)

    @use_named_args(param_space)
    def objective(**params):
        call_count[0] += 1

        try:
            # === PARAMETER CONSTRAINTS (prevent overfitting) ===
            short_period = params["short_period"]
            long_period = params["long_period"]
            alma_offset = params["alma_offset_int"] / 100.0  # Convert integer to float (85 -> 0.85)
            alma_sigma = params["alma_sigma"]
            
            # CONSTRAINT 1: Short must be < Long (prevent inversion)
            if short_period >= long_period:
                return 999.0  # Heavy penalty
            
            # CONSTRAINT 2: Periods must have minimum separation (20 days)
            if (long_period - short_period) < 20:
                return 999.0  # Prevent too-similar periods
            
            # CONSTRAINT 3: Period ratio must be reasonable (prevent dead zones)
            period_ratio = short_period / long_period
            if period_ratio < 0.20:  # Too slow (like 53/300 = 0.177)
                return 999.0  # Prevents extremely smooth, non-reactive combinations
            
            # CONSTRAINT 4: Long period should not exceed 250 days (prevent excessive smoothing)
            if long_period > 250:
                return 999.0  # 250 days is already ~70% of 1-year, plenty smooth
            
            # CONSTRAINT 5: ALMA offset should be reasonable (redundant now, but keep for safety)
            if alma_offset < 0.75:  # Too laggy
                return 999.0
            # ===================================================
            
            entries, exits, position_target = run_strategy(
                close, high, low,
                short_period, long_period,
                alma_offset, alma_sigma,
                params["buy_momentum_bars"], params["sell_momentum_bars"],
                params["baseline_momentum_bars"],
                params["macro_ema_period"],
            )

            if entries.sum() < 3:
                return 10.0

            # Position sizing with volatility targeting / capital constraints
            portfolio = vbt.Portfolio.from_signals(
                close, entries, exits,
                size=position_target,
                size_type=vbt.Portfolio.SizeType.Percent,
                init_cash=CAPITAL_BASE,
                fees=MANUAL_DEFAULTS["commission_rate"],
                slippage=MANUAL_DEFAULTS["slippage_rate"],
                freq="1D"
            )
            stats = portfolio.stats()
            
            # Compute composite score from multiple metrics
            score, components = compute_composite_score(portfolio, stats, params, training_days)
            sortino = components["sortino_raw"]

            # --- PRINT PROGRESS ---
            now = datetime.datetime.now().strftime("[%H:%M:%S]")
            
            # Track constraint violations for diagnostics
            if components.get("constraint_violation"):
                if call_count[0] % 100 == 0:  # Don't spam, just periodic updates
                    violation = components["constraint_violation"]
                    print(f"    {now} Call {call_count[0]:3d}/{n_calls} - Rejected: {violation}")
            
            if score > best_score[0]:
                best_score[0] = score
                best_sortino_raw[0] = sortino  # Store raw Sortino for efficient retrieval
                if call_count[0] % 10 == 0 or call_count[0] <= 3:
                    # Show breakdown of composite score
                    trades_yr = components["trades_per_year"]
                    calmar = components["calmar_ratio"]
                    print(f"    {now} ✓ Call {call_count[0]:3d}/{n_calls} "
                          f"Score: {score:5.2f} (Sortino: {sortino:.2f}, "
                          f"Calmar: {calmar:.2f}, Trades/yr: {trades_yr:.0f})")
            elif call_count[0] % 50 == 0:
                print(f"    {now} Progress {call_count[0]:3d}/{n_calls} "
                      f"| Best score: {best_score[0]:.2f}")
            # -----------------------

            return -score  # Minimize negative score = maximize score
        except Exception:
            return 10.0

    # --- Run optimizer with all CPU cores ---
    result = gp_minimize(
        objective,
        param_space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=42,
        verbose=False,
        n_jobs=n_jobs,  # adjust: 12/16 for cloud, override in tests
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    # --- Extract best params (raw Sortino already stored during optimization) ---
    dim_names = [dim.name for dim in param_space]
    raw_best = dict(zip(dim_names, result.x))

    best_params = {
        "short_period": int(raw_best["short_period"]),
        "long_period": int(raw_best["long_period"]),
        "alma_offset": raw_best["alma_offset_int"] / 100.0,  # Convert integer back to float
        "alma_sigma": float(raw_best["alma_sigma"]),
        "buy_momentum_bars": int(raw_best["buy_momentum_bars"]),
        "sell_momentum_bars": int(raw_best["sell_momentum_bars"]),
        "baseline_momentum_bars": int(raw_best["baseline_momentum_bars"]),
        "macro_ema_period": int(raw_best["macro_ema_period"]),
        "score": -result.fun,  # Penalized score (what we optimized)
        "train_sortino": best_sortino_raw[0],  # Raw Sortino (stored during optimization)
    }
    
    print(f"  ✔ [{end_time.strftime('%H:%M:%S')}] "
          f"Done in {duration} | Best Score {best_params['score']:5.2f} (Train Sortino: {best_params['train_sortino']:.2f})")
    print(f"     Periods: short={best_params['short_period']}, long={best_params['long_period']}")
    print(f"     ALMA: offset={best_params['alma_offset']:.2f}, "
          f"sigma={best_params['alma_sigma']:.1f}")
    print(f"     Momentum: lookback=({best_params['buy_momentum_bars']},{best_params['sell_momentum_bars']})")
    print(f"     Baseline Momentum: {best_params['baseline_momentum_bars']} bars")
    print(f"     Macro Filter: {best_params['macro_ema_period']}-day EMA")

    return best_params

# ============================================================================
# REPORTING & VISUALIZATION
# ============================================================================

def generate_quantstats_report(portfolio, benchmark_returns, iteration_name, stage_name):
    """
    Generate a comprehensive QuantStats HTML report for a backtest iteration.
    
    Args:
        portfolio: vectorbt Portfolio object
        benchmark_returns: pandas Series of benchmark returns (buy & hold)
        iteration_name: str, e.g., "Stage1_Iter1_2017to2020"
        stage_name: str, e.g., "STAGE_1_GLOBAL_SEARCH"
    """
    if not QUANTSTATS_AVAILABLE:
        print("  ⚠️  Skipping QuantStats report (not installed)")
        return None
    
    # Get strategy returns
    strategy_returns = portfolio.returns()
    
    # Configure QuantStats
    qs.extend_pandas()
    
    # Generate full HTML report
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
    """
    Generate parameter robustness heatmaps using VectorBT.
    
    Shows how performance changes when parameters vary around optimal values.
    Helps identify which parameters are most sensitive (need precision) vs robust (can vary).
    
    Args:
        data: full dataset
        train_start, train_end: training period bounds
        test_start, test_end: test period bounds
        best_params: dict of optimal parameters
        iteration_name: str, e.g., "Iter1_2017to2020"
        stage_name: str, e.g., "STAGE_1_GLOBAL_SEARCH"
    """
    print(f"\n  🔥 Generating parameter heatmaps (robustness analysis)...")
    
    test_close = data.loc[test_start:test_end, "close"]
    test_high = data.loc[test_start:test_end, "high"]
    test_low = data.loc[test_start:test_end, "low"]
    
    # ========================================
    # HEATMAP 1: Short vs Long Period
    # ========================================
    print("     - Short vs Long Period heatmap...")
    
    short_range = np.arange(
        max(15, best_params["short_period"] - 20),
        min(150, best_params["short_period"] + 20),
        5
    )
    long_range = np.arange(
        max(80, best_params["long_period"] - 40),
        min(250, best_params["long_period"] + 40),
        10
    )
    
    results_period = np.zeros((len(short_range), len(long_range)))
    
    for i, short in enumerate(short_range):
        for j, long in enumerate(long_range):
            if short >= long:  # Invalid combination
                results_period[i, j] = np.nan
                continue
            
            try:
                entries, exits, position_target = run_strategy(
                    test_close, test_high, test_low,
                    int(short), int(long),
                    best_params["alma_offset"], best_params["alma_sigma"],
                    best_params["buy_momentum_bars"], best_params["sell_momentum_bars"],
                    best_params["baseline_momentum_bars"],
                    best_params["macro_ema_period"],
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    size=position_target,
                    size_type="targetpercent",
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_period[i, j] = port.total_return()
            except Exception:
                results_period[i, j] = np.nan
    
    # Save heatmap data
    heatmap_df = pd.DataFrame(
        results_period,
        index=short_range,
        columns=long_range
    )
    heatmap_path = f"reports/heatmaps/{stage_name}_{iteration_name}_short_vs_long.csv"
    heatmap_df.to_csv(heatmap_path)
    print(f"       ✓ Saved: {heatmap_path}")
    
    # ========================================
    # HEATMAP 2: ALMA Offset vs Sigma
    # ========================================
    print("     - ALMA Offset vs Sigma heatmap...")
    
    offset_range = np.arange(
        max(0.85, best_params["alma_offset"] - 0.05),
        min(0.99, best_params["alma_offset"] + 0.05),
        0.01
    )
    sigma_range = np.arange(
        max(3.0, best_params["alma_sigma"] - 2.0),
        min(9.0, best_params["alma_sigma"] + 2.0),
        0.5
    )
    
    results_alma = np.zeros((len(offset_range), len(sigma_range)))
    
    for i, offset in enumerate(offset_range):
        for j, sigma in enumerate(sigma_range):
            try:
                entries, exits, position_target = run_strategy(
                    test_close, test_high, test_low,
                    best_params["short_period"], best_params["long_period"],
                    offset, sigma,
                    best_params["buy_momentum_bars"], best_params["sell_momentum_bars"],
                    best_params["baseline_momentum_bars"],
                    best_params["macro_ema_period"],
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    size=position_target,
                    size_type="targetpercent",
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_alma[i, j] = port.total_return()
            except Exception:
                results_alma[i, j] = np.nan
    
    # Save heatmap data
    heatmap_df2 = pd.DataFrame(
        results_alma,
        index=[f"{x:.2f}" for x in offset_range],
        columns=[f"{x:.1f}" for x in sigma_range]
    )
    heatmap_path2 = f"reports/heatmaps/{stage_name}_{iteration_name}_offset_vs_sigma.csv"
    heatmap_df2.to_csv(heatmap_path2)
    print(f"       ✓ Saved: {heatmap_path2}")
    
    # ========================================
    # HEATMAP 3: Buy vs Sell Momentum Lookback
    # ========================================
    print("     - Buy vs Sell Momentum heatmap...")
    
    buy_range = np.arange(
        max(3, best_params["buy_momentum_bars"] - 4),
        min(15, best_params["buy_momentum_bars"] + 4),
        1
    )
    sell_range = np.arange(0, 7, 1)
    
    results_momentum = np.zeros((len(buy_range), len(sell_range)))
    
    for i, buy_bars in enumerate(buy_range):
        for j, sell_bars in enumerate(sell_range):
            try:
                entries, exits, position_target = run_strategy(
                    test_close, test_high, test_low,
                    best_params["short_period"], best_params["long_period"],
                    best_params["alma_offset"], best_params["alma_sigma"],
                    int(buy_bars), int(sell_bars),
                    best_params["baseline_momentum_bars"],
                    best_params["macro_ema_period"],
                )
                port = vbt.Portfolio.from_signals(
                    test_close, entries, exits,
                    size=position_target,
                    size_type="targetpercent",
                    init_cash=CAPITAL_BASE,
                    fees=MANUAL_DEFAULTS["commission_rate"],
                    slippage=MANUAL_DEFAULTS["slippage_rate"],
                    freq="1D"
                )
                results_momentum[i, j] = port.total_return()
            except Exception:
                results_momentum[i, j] = np.nan
    
    # Save heatmap data
    heatmap_df3 = pd.DataFrame(
        results_momentum,
        index=buy_range,
        columns=sell_range
    )
    heatmap_path3 = f"reports/heatmaps/{stage_name}_{iteration_name}_buy_vs_sell_momentum.csv"
    heatmap_df3.to_csv(heatmap_path3)
    print(f"       ✓ Saved: {heatmap_path3}")
    
    print(f"     ✓ All heatmaps generated for {iteration_name}")
    
    return {
        "period_heatmap": heatmap_path,
        "alma_heatmap": heatmap_path2,
        "momentum_heatmap": heatmap_path3,
    }


# ============================================================================
# DATA INGESTION HELPERS
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
# WALK‑FORWARD ANALYSIS HELPERS
# ============================================================================

def detect_bull_market_periods(
    data,
    ema_period=BULL_DETECTION_EMA_PERIOD,
    slope_lookback=BULL_SLOPE_LOOKBACK,
    min_days=MIN_BULL_PERIOD_DAYS,
):
    """
    Detect bull-market segments using a rising EMA filter.
    
    A bull segment is defined by:
    - Price > EMA(ema_period)
    - EMA slope positive over `slope_lookback`
    """
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
    """
    Generate anchored, purged walk-forward windows for a bull-market segment.
    """
    total_days = len(segment_index)
    if total_days < (min_train_days + min_test_days + purge_days):
        return []
    
    train_len = max(int(total_days * train_fraction), min_train_days)
    test_len = max(int(total_days * test_fraction), min_test_days)
    
    # Ensure windows fit inside the segment
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


def walk_forward_analysis(data, param_space, n_calls, n_random_starts, stage_name=""):
    """
    Perform walk-forward optimization with data-driven bull-market detection.
    
    Steps:
    1. Detect bull segments using a rising EMA filter
    2. Within each bull, run anchored/purged walk-forward splits
    3. Optimize on each training slice and evaluate on the subsequent test slice
    """
    
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
            
            best = optimize_parameters_bayesian(
                data,
                train_start,
                train_end,
                param_space,
                n_calls,
                n_random_starts,
            )
            if best is None:
                continue
            
            test_close = test_slice["close"]
            test_high = test_slice["high"]
            test_low = test_slice["low"]
            train_close = train_slice["close"]
            
            entries, exits, position_target = run_strategy(
                test_close, test_high, test_low,
                best["short_period"], best["long_period"],
                best["alma_offset"], best["alma_sigma"],
                best["buy_momentum_bars"], best["sell_momentum_bars"],
                best["baseline_momentum_bars"],
                best["macro_ema_period"],
            )
            port = vbt.Portfolio.from_signals(
                test_close, entries, exits,
                size=position_target,
                size_type="targetpercent",
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
                # ALMA parameters
                "Short Period": best["short_period"],
                "Long Period": best["long_period"],
                "ALMA Offset": best["alma_offset"],
                "ALMA Sigma": best["alma_sigma"],
                "Buy Lookback": best["buy_momentum_bars"],
                "Sell Lookback": best["sell_momentum_bars"],
                "Baseline Momentum": best["baseline_momentum_bars"],
                "Macro EMA Period": best["macro_ema_period"],
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
                # Regime analysis
                "train_vol": train_vol,
                "test_vol": test_vol,
                "vol_regime": vol_regime,
            }
            wf_record.update(bootstrap)
            
            wf_results.append(wf_record)
            
            iteration += 1
    
    # Return results DataFrame
    df = pd.DataFrame(wf_results)
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n==========================================")
    print("HERMES STRATEGY - REPARAMETERIZED OPTIMIZATION")
    print("==========================================\n")
    asset_data_map = load_all_asset_data(ASSET_DATA_SOURCES)
    if not asset_data_map:
        print("✗ No asset datasets found. Provide CSV files defined in ASSET_DATA_SOURCES.")
        return
    
    aggregated_stage1 = []
    aggregated_stage2 = []
    
    for asset_name, asset_df in asset_data_map.items():
        print(f"\n==================== {asset_name} ====================")
        print(f"Rows available: {len(asset_df)}")
        
        stage1 = walk_forward_analysis(
            asset_df.copy(),
            STAGE1_SPACE,
            STAGE1_CALLS,
            STAGE1_RANDOM_STARTS,
            f"{asset_name} | STAGE 1: GLOBAL SEARCH"
        )
        if len(stage1) == 0:
            print(f"✗ {asset_name}: Stage 1 produced no valid windows.")
            continue
        
        stage1["asset"] = asset_name
        stage1_file = Path(f"hermes_stage1_{asset_name}.csv")
        stage1.to_csv(stage1_file, index=False)
        aggregated_stage1.append(stage1)
        
        best_region = stage1.nlargest(5, "test_composite").mean(numeric_only=True).to_dict()
        
        stage2_space = [
            Integer(max(15, int(best_region["Short Period"]) - 20),
                    min(100, int(best_region["Short Period"]) + 20), 
                    name="short_period"),
            Integer(max(80, int(best_region["Long Period"]) - 50),
                    min(250, int(best_region["Long Period"]) + 50), 
                    name="long_period"),
            Integer(max(85, int(best_region["ALMA Offset"] * 100) - 5),
                    min(99, int(best_region["ALMA Offset"] * 100) + 5), 
                    name="alma_offset_int"),
            Categorical([v for v in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                         if max(3.0, best_region["ALMA Sigma"] - 1.5) <= v <= min(9.0, best_region["ALMA Sigma"] + 1.5)],
                        name="alma_sigma"),
            Integer(max(3, int(best_region["Buy Lookback"]) - 4),
                    min(12, int(best_region["Buy Lookback"]) + 4), 
                    name="buy_momentum_bars"),
            Integer(max(0, int(best_region["Sell Lookback"]) - 2),
                    min(6, int(best_region["Sell Lookback"]) + 2), 
                    name="sell_momentum_bars"),
            Integer(max(1, int(best_region["Baseline Momentum"]) - 10),
                    min(50, int(best_region["Baseline Momentum"]) + 10), 
                    name="baseline_momentum_bars"),
            Categorical([v for v in range(100, 310, 10)
                         if max(100, int(best_region["Macro EMA Period"]) - 50) <= v <= min(300, int(best_region["Macro EMA Period"]) + 50)],
                        name="macro_ema_period"),
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
        combined_stage1.to_csv("hermes_stage1_all_assets.csv", index=False)
    if len(aggregated_stage2) > 1:
        combined_stage2 = pd.concat(aggregated_stage2, ignore_index=True)
        combined_stage2.to_csv("hermes_stage2_all_assets.csv", index=False)


if __name__ == "__main__":
    main()
