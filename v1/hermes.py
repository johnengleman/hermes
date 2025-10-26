"""
Hermes Strategy
ALMA-based trend following strategy with momentum filters
"""

import pandas as pd
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def alma_numba(src, period, offset, sigma):
    """
    Arnaud Legoux Moving Average (ALMA) - Numba accelerated
    
    Args:
        src: Source data array
        period: Lookback period
        offset: Gaussian offset (0-1)
        sigma: Gaussian sigma parameter
    
    Returns:
        ALMA values as numpy array
    """
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
    """
    Hull Moving Average (HMA) - Numba accelerated
    
    Args:
        close: Close price array
        period: Lookback period
    
    Returns:
        HMA values as numpy array
    """
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


def run_strategy_simple(close, high, low, **params):
    """
    Hermes Simple Strategy - ALMA-based trend following
    REWRITTEN to exactly match hermes.pine logic with stateful position tracking
    
    Args:
        close: Close price series (pandas Series)
        high: High price series (pandas Series)
        low: Low price series (pandas Series)
        **params: Strategy parameters dict containing:
            - short_period: Short ALMA period
            - long_period: Long ALMA period
            - alma_offset: ALMA offset parameter (0-1)
            - alma_sigma: ALMA sigma parameter
            - momentum_lookback_long: Momentum breakout lookback for long entries
            - momentum_lookback_short: Momentum breakout lookback for exits
            - macro_ema_period: Macro trend EMA period
            - fast_hma_period: Fast HMA period
            - slow_ema_period: Slow EMA period
            - slow_ema_rising_lookback: Lookback for EMA slope
    
    Returns:
        Tuple of (entries, exits, position_target) as pandas Series
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)
    n = len(close_np)

    # Extract parameters
    short_period = int(params["short_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]
    momentum_lookback_long = int(params["momentum_lookback_long"])
    momentum_lookback_short = int(params["momentum_lookback_short"])
    macro_ema_period = int(params["macro_ema_period"])
    fast_hma_period = int(params["fast_hma_period"])
    slow_ema_period = int(params["slow_ema_period"])
    slow_ema_rising_lookback = int(params["slow_ema_rising_lookback"])
    
    # Check if optional filters are enabled (0 = disabled)
    use_momentum_long = momentum_lookback_long > 0
    use_momentum_short = momentum_lookback_short > 0
    use_macro_filter = macro_ema_period > 0
    use_slow_ema_rising = slow_ema_rising_lookback > 0

    # Calculate log returns
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # Calculate ALMA filters on returns
    long_term = alma_numba(returns, long_period, alma_offset, alma_sigma)
    short_term = alma_numba(returns, short_period, alma_offset, alma_sigma)
    baseline = long_term

    # Calculate price-based indicators
    fast_hma = hma_numba(close_np, fast_hma_period)
    slow_ema = pd.Series(close_np).ewm(span=slow_ema_period, adjust=False).mean().to_numpy()
    
    # Macro filter (optional - only calculate if enabled)
    if use_macro_filter:
        macro_ema = pd.Series(close_np).ewm(span=macro_ema_period, adjust=False).mean().to_numpy()
        in_bull_market = close_np > macro_ema
    else:
        in_bull_market = np.ones(n, dtype=bool)  # Always true when disabled

    # ALMA trend states
    bullish_state = short_term > baseline
    bearish_state = short_term < baseline

    # Momentum filters for long entries (optional - only calculate if enabled)
    if use_momentum_long:
        highest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        is_highest_close = (close_np >= np.nan_to_num(highest_close_prev, nan=0)) & \
                           (high_np >= np.nan_to_num(highest_high_prev, nan=0))
    else:
        is_highest_close = np.ones(n, dtype=bool)  # Always true when disabled
    
    # Momentum filters for short/exit signals (optional - only calculate if enabled)
    if use_momentum_short:
        lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback_short).min().to_numpy()
        lowest_close_prev = pd.Series(close_np).shift(1).rolling(momentum_lookback_short).min().to_numpy()
        is_lowest_low = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                        (close_np <= np.nan_to_num(lowest_close_prev, nan=np.inf))
    else:
        is_lowest_low = np.ones(n, dtype=bool)  # Always true when disabled

    # Slow EMA rising filter (optional - only calculate if enabled)
    if use_slow_ema_rising:
        slow_ema_rising = np.zeros(n, dtype=bool)
        for i in range(slow_ema_rising_lookback, n):
            slow_ema_rising[i] = slow_ema[i] > slow_ema[i - slow_ema_rising_lookback]
    else:
        slow_ema_rising = np.ones(n, dtype=bool)  # Always true when disabled

    # Build buy signal (matches Pine Script exactly)
    buy_signal_base = bullish_state.copy()
    if use_momentum_long:
        buy_signal_base = buy_signal_base & is_highest_close
    buy_signal_base = buy_signal_base & in_bull_market & slow_ema_rising

    # Build sell signal base (bearish state + momentum)
    sell_signal_base = bearish_state.copy()
    if use_momentum_short:
        sell_signal_base = sell_signal_base & is_lowest_low

    # STATEFUL POSITION TRACKING (matching Pine Script logic)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    
    # Minimum bars required for indicators to be valid (warm-up period)
    min_bars = max(long_period, slow_ema_period, fast_hma_period)
    
    in_position = False
    trending_regime = False
    position_entry_price = 0.0
    
    for i in range(n):
        # Skip trading until we have enough bars for indicator warm-up
        if i < min_bars:
            continue
            
        # Entry logic (only when not in position)
        if not in_position and buy_signal_base[i]:
            entries[i] = True
            in_position = True
            trending_regime = False
            position_entry_price = close_np[i]
        
        # Exit logic (only when in position)
        elif in_position:
            # Trending regime detection
            trending_setup = (slow_ema[i] > position_entry_price and 
                            fast_hma[i] > position_entry_price and 
                            fast_hma[i] > slow_ema[i])
            
            # Check if HMA is below EMA (persistent state, not just crossunder)
            hma_below_ema = fast_hma[i] < slow_ema[i]
            
            # Calculate exit conditions BEFORE updating regime state
            sell_momentum_ok = not use_momentum_short or is_lowest_low[i]
            close_below_entry = close_np[i] < position_entry_price
            normal_trending_exit = hma_below_ema and sell_momentum_ok
            trending_exit = trending_regime and (close_below_entry or normal_trending_exit)
            
            # Update trending regime state (after exit calculation)
            if trending_setup:
                trending_regime = True
            if hma_below_ema:
                trending_regime = False
            
            # Ranging exit
            ranging_exit = not trending_regime and sell_signal_base[i]
            
            # Execute exit
            if trending_exit or ranging_exit:
                exits[i] = True
                in_position = False
                trending_regime = False
                position_entry_price = 0.0

    # Position sizing (100% allocation)
    position_target = np.ones(n, dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)
