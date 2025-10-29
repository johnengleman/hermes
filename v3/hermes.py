"""
Hermes Strategy - Long-Only Bitcoin
ALMA-based trend following strategy with momentum filters
Optimized for Bitcoin on daily timeframe
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


def run_strategy_simple(close, high, low, open, **params):
    """
    Hermes Simple Strategy - ALMA-based trend following (Long-Only)
    Rewritten to match hermes.pine logic for Bitcoin daily timeframe
    
    Args:
        close: Close price series (pandas Series)
        high: High price series (pandas Series)
        low: Low price series (pandas Series)
        open: Open price series (pandas Series)
        **params: Strategy parameters dict containing:
            - short_period: Short ALMA period
            - long_period: Long ALMA period
            - alma_offset: ALMA offset parameter (0-1)
            - alma_sigma: ALMA sigma parameter
            - alma_min_separation: Minimum ALMA separation to prevent whipsaw
            - momentum_lookback_long: Momentum breakout lookback for long entries (0 = disabled)
            - fast_ema_period: Fast EMA period for trending regime
            - slow_ema_period: Slow EMA period (trend filter)
            - trending_regime_min_distance: Min % distance for slow_ema above entry to activate trending regime
    
    Returns:
        Tuple of (entries, exits) as pandas Series
        - entries: 1 for long entry, 0 for no entry
        - exits: True when position should be closed
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)
    open_np = open.to_numpy(dtype=np.float64, copy=False)
    n = len(close_np)

    # Extract parameters
    short_period = int(params["short_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]
    alma_min_separation = params["alma_min_separation"]
    momentum_lookback_long = int(params["momentum_lookback_long"])
    fast_ema_period = int(params["fast_ema_period"])
    slow_ema_period = int(params["slow_ema_period"])
    trending_regime_min_distance = params["trending_regime_min_distance"]  # Percentage (e.g., 0.035 = 3.5%)
    
    # Check if optional filters are enabled (0 = disabled)
    use_momentum_long = momentum_lookback_long > 0

    # Calculate log returns
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # Calculate ALMA filters on returns
    long_term = alma_numba(returns, long_period, alma_offset, alma_sigma)
    short_term = alma_numba(returns, short_period, alma_offset, alma_sigma)
    baseline = long_term

    # Calculate price-based indicators
    fast_ema = pd.Series(close_np).ewm(span=fast_ema_period, adjust=False).mean().to_numpy()
    slow_ema = pd.Series(close_np).ewm(span=slow_ema_period, adjust=False).mean().to_numpy()
    
    # Price trend filter
    price_above_slow_ema = close_np > slow_ema

    # ALMA trend states
    bullish_state = short_term > baseline
    bearish_state = short_term < baseline
    
    # ALMA separation check - ensure meaningful cross with minimum distance
    alma_separation = np.abs(short_term - baseline)
    valid_separation = alma_separation >= alma_min_separation

    # Momentum filter for long entries (only calculate if enabled)
    if use_momentum_long:
        highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        prev_body_top = np.maximum(open_np, close_np)
        highest_body_top_prev = pd.Series(prev_body_top).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        is_highest_close = (high_np >= np.nan_to_num(highest_high_prev, nan=0)) & \
                           (close_np >= np.nan_to_num(highest_body_top_prev, nan=0))
    else:
        is_highest_close = np.ones(n, dtype=bool)

    # Build buy signal (long entry)
    buy_signal = bullish_state & valid_separation & price_above_slow_ema
    if use_momentum_long:
        buy_signal = buy_signal & is_highest_close

    # Build sell signal (ALMA bearish for ranging exits)
    sell_signal = bearish_state & valid_separation

    # STATEFUL POSITION TRACKING (matching Pine Script logic)
    entries = np.zeros(n, dtype=np.float64)  # 1 for long, 0 for no entry
    exits = np.zeros(n, dtype=bool)
    
    # Minimum bars required for indicators to be valid (warm-up period)
    min_bars = max(long_period, slow_ema_period, fast_ema_period)
    
    in_position = False
    trending_regime = False
    position_entry_price = 0.0
    just_exited = False
    
    for i in range(n):
        # Skip trading until we have enough bars for indicator warm-up
        if i < min_bars:
            continue
        
        # EXIT LOGIC (Process BEFORE entries)
        if in_position:
            # Check if we should activate trending regime (only check if not already in trending regime)
            if not trending_regime:
                # Trending regime: activated when slow_ema rises above entry by minimum distance
                # At entry: price > slow_ema, so slow_ema < entry (distance is negative)
                # As slow_ema rises above entry, distance becomes positive
                # When distance >= min_distance, trending regime activates (strong trend confirmed)
                ema_distance_from_entry = (slow_ema[i] - position_entry_price) / position_entry_price
                in_trending_regime = ema_distance_from_entry >= trending_regime_min_distance
                
                # Update persistent trending state (once activated, stays until exit)
                if in_trending_regime:
                    trending_regime = True
            
            # Exit conditions based on regime
            should_exit = False
            if trending_regime:
                # Trending exits: Fast EMA below Slow EMA (trend break) OR price below entry (emergency stop)
                ema_cross_down = fast_ema[i] < slow_ema[i]
                close_below_entry = close_np[i] < position_entry_price
                should_exit = ema_cross_down or close_below_entry
            else:
                # Ranging exits: ALMA bearish signal
                should_exit = sell_signal[i]
            
            # Execute exit
            if should_exit:
                exits[i] = True
                in_position = False
                trending_regime = False
                position_entry_price = 0.0
                just_exited = True
        else:
            # Reset just_exited flag when not in position (matches Pine Script)
            just_exited = False
        
        # ENTRY LOGIC (Process AFTER exits, skip if we just exited)
        if buy_signal[i] and not in_position and not just_exited:
            entries[i] = 1.0
            in_position = True
            trending_regime = False
            position_entry_price = close_np[i]

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index))
