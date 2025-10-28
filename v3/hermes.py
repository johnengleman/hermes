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


def run_strategy_simple(close, high, low, open, **params):
    """
    Hermes Simple Strategy - ALMA-based trend following with symmetric long/short
    REWRITTEN to exactly match hermes.pine logic with stateful position tracking
    
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
            - momentum_lookback: Momentum breakout lookback (used for both long/short)
            - fast_ema_period: Fast EMA period for trending exits
            - slow_ema_period: Slow EMA period (trend filter and exit reference)
            - trending_regime_min_distance: Minimum distance in pips slow_ema must move from entry to activate trending regime
            - broker_leverage: Broker leverage (e.g., 50 for EUR/USD, 20 for USD/JPY) - optional, defaults to 50
            - risk_per_trade_pct: Risk % per trade (e.g., 2.0 for 2%) - optional, defaults to 2.0
    
    Returns:
        Tuple of (entries, exits, position_target) as pandas Series
        - entries: 1 for long, -1 for short, 0 for no entry
        - exits: True when position should be closed
        - position_target: Leverage-adjusted position size (broker_leverage * risk_per_trade_pct / 100)
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
    momentum_lookback = int(params["momentum_lookback"])
    fast_ema_period = int(params["fast_ema_period"])
    slow_ema_period = int(params["slow_ema_period"])
    trending_regime_min_distance = params["trending_regime_min_distance"]  # In pips (e.g., 5 = 5 pips)
    
    # Leverage and risk parameters (with defaults for forex)
    broker_leverage = params.get("broker_leverage", 50.0)  # Default 50:1 for EUR/USD  
    risk_per_trade_pct = params.get("risk_per_trade_pct", 2.0)  # Default 2% risk
    
    # Pip size for EUR/USD (hardcoded)
    pip_size = 0.0001
    
    # Calculate effective position size for leveraged trading
    # With 50:1 leverage and 2% risk: position = 2% * 50 = 100% of account value
    # For VectorBT: we express this as leverage multiplier
    effective_position_pct = risk_per_trade_pct * broker_leverage
    
    # Check if optional filters are enabled (0 = disabled)
    use_momentum = momentum_lookback > 0

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
    
    # Price trend filters
    price_above_slow_ema = close_np > slow_ema
    price_below_slow_ema = close_np < slow_ema
    
    # EMA trend filters
    fast_above_slow = fast_ema > slow_ema
    fast_below_slow = fast_ema < slow_ema

    # ALMA trend states
    bullish_state = short_term > baseline
    bearish_state = short_term < baseline

    # Momentum filters (symmetric for long/short entries, only calculate if enabled)
    if use_momentum:
        # Long momentum: is_highest_close
        highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback).max().to_numpy()
        prev_body_top = np.maximum(open_np, close_np)
        highest_body_top_prev = pd.Series(prev_body_top).shift(1).rolling(momentum_lookback).max().to_numpy()
        is_highest_close = (high_np >= np.nan_to_num(highest_high_prev, nan=0)) & \
                           (close_np >= np.nan_to_num(highest_body_top_prev, nan=0))
        
        # Short momentum: is_lowest_close (symmetric)
        lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback).min().to_numpy()
        prev_body_bottom = np.minimum(open_np, close_np)
        lowest_body_bottom_prev = pd.Series(prev_body_bottom).shift(1).rolling(momentum_lookback).min().to_numpy()
        is_lowest_close = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                          (close_np <= np.nan_to_num(lowest_body_bottom_prev, nan=np.inf))
    else:
        is_highest_close = np.ones(n, dtype=bool)
        is_lowest_close = np.ones(n, dtype=bool)

    # Build buy signal (long entry - matches Pine Script exactly)
    buy_signal = bullish_state
    if use_momentum:
        buy_signal = buy_signal & is_highest_close
    buy_signal = buy_signal & price_above_slow_ema & fast_above_slow

    # Build sell signal (short entry - symmetric to buy)
    sell_signal = bearish_state
    if use_momentum:
        sell_signal = sell_signal & is_lowest_close
    sell_signal = sell_signal & price_below_slow_ema & fast_below_slow

    # ALMA reversal signals (for ranging exits)
    alma_bullish_reversal = bullish_state
    alma_bearish_reversal = bearish_state

    # STATEFUL POSITION TRACKING (matching Pine Script logic)
    entries = np.zeros(n, dtype=np.float64)  # 1 for long, -1 for short, 0 for no entry
    exits = np.zeros(n, dtype=bool)
    
    # Minimum bars required for indicators to be valid (warm-up period)
    min_bars = max(long_period, slow_ema_period, fast_ema_period)
    
    position_type = 0  # 0 = no position, 1 = long, -1 = short
    trending_regime = False
    position_entry_price = 0.0
    just_exited = False
    
    for i in range(n):
        # Skip trading until we have enough bars for indicator warm-up
        if i < min_bars:
            continue
        
        # EXIT LOGIC (Process BEFORE entries)
        if position_type != 0:
            
            if position_type == 1:  # LONG POSITION
                # Check if we should activate trending regime (only check if not already in trending regime)
                if not trending_regime:
                    min_distance_price = trending_regime_min_distance * pip_size
                    ema_distance_abs = abs(slow_ema[i] - position_entry_price)
                    in_trending_regime = (slow_ema[i] > position_entry_price) and (ema_distance_abs >= min_distance_price)
                    
                    # Update persistent trending state (once activated, stays until exit)
                    if in_trending_regime:
                        trending_regime = True
                
                # Exit conditions based on regime
                if trending_regime:
                    # Trending exits: Fast EMA below Slow EMA (trend break) OR price below entry
                    ema_cross_down = fast_ema[i] < slow_ema[i]
                    close_below_entry = close_np[i] < position_entry_price
                    should_exit = ema_cross_down or close_below_entry
                else:
                    # Ranging exits: ALMA bearish reversal
                    should_exit = alma_bearish_reversal[i]
                
                if should_exit:
                    exits[i] = True
                    position_type = 0
                    trending_regime = False
                    position_entry_price = 0.0
                    just_exited = True
            
            elif position_type == -1:  # SHORT POSITION
                # Check if we should activate trending regime (only check if not already in trending regime)
                if not trending_regime:
                    min_distance_price = trending_regime_min_distance * pip_size
                    ema_distance_abs = abs(slow_ema[i] - position_entry_price)
                    in_trending_regime = (slow_ema[i] < position_entry_price) and (ema_distance_abs >= min_distance_price)
                    
                    # Update persistent trending state (once activated, stays until exit)
                    if in_trending_regime:
                        trending_regime = True
                
                # Exit conditions based on regime
                if trending_regime:
                    # Trending exits: Fast EMA above Slow EMA (downtrend break) OR price above entry
                    ema_cross_up = fast_ema[i] > slow_ema[i]
                    close_above_entry = close_np[i] > position_entry_price
                    should_exit = ema_cross_up or close_above_entry
                else:
                    # Ranging exits: ALMA bullish reversal
                    should_exit = alma_bullish_reversal[i]
                
                if should_exit:
                    exits[i] = True
                    position_type = 0
                    trending_regime = False
                    position_entry_price = 0.0
                    just_exited = True
        else:
            # Reset just_exited flag when not in position (matches Pine Script)
            just_exited = False
        
        # ENTRY LOGIC (Process AFTER exits)
        # No just_exited check - allow position reversals on same bar
        if buy_signal[i]:
            if position_type == -1:  # Close short before going long
                exits[i] = True
                position_type = 0
                trending_regime = False
            elif position_type == 0:  # Open new long
                entries[i] = 1.0
                position_type = 1
                trending_regime = False
                position_entry_price = close_np[i]
        
        elif sell_signal[i]:
            if position_type == 1:  # Close long before going short
                exits[i] = True
                position_type = 0
                trending_regime = False
            elif position_type == 0:  # Open new short
                entries[i] = -1.0
                position_type = -1
                trending_regime = False
                position_entry_price = close_np[i]

    # Position sizing - leverage-adjusted (matches Pine Script)
    # effective_position_pct is already calculated above
    position_target = np.full(n, effective_position_pct, dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)
