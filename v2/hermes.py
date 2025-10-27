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
def calculate_adx(high, low, close, period):
    """
    Calculate ADX (Average Directional Index)
    
    Args:
        high: High price array
        low: Low price array
        close: Close price array
        period: ADX calculation period
    
    Returns:
        ADX values as numpy array
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        # True Range
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
        
        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smooth the TR and DM using Wilder's smoothing (similar to EMA)
    atr = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    # Initialize with SMA for first period
    atr[period] = np.mean(tr[1:period+1])
    smoothed_plus = np.mean(plus_dm[1:period+1])
    smoothed_minus = np.mean(minus_dm[1:period+1])
    
    if atr[period] != 0:
        plus_di[period] = 100 * smoothed_plus / atr[period]
        minus_di[period] = 100 * smoothed_minus / atr[period]
    
    # Smooth subsequent values
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        smoothed_plus = (smoothed_plus * (period - 1) + plus_dm[i]) / period
        smoothed_minus = (smoothed_minus * (period - 1) + minus_dm[i]) / period
        
        if atr[i] != 0:
            plus_di[i] = 100 * smoothed_plus / atr[i]
            minus_di[i] = 100 * smoothed_minus / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx[2 * period - 1] = np.mean(dx[period:2*period])
    for i in range(2 * period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx


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
    Hermes Simple Strategy - ALMA-based trend following with dynamic ADX-based short period
    
    Args:
        close: Close price series (pandas Series)
        high: High price series (pandas Series)
        low: Low price series (pandas Series)
        open: Open price series (pandas Series)
        **params: Strategy parameters dict containing:
            - min_short_period: Minimum short ALMA period (when ADX is low)
            - max_short_period: Maximum short ALMA period (when ADX is high)
            - adx_period: ADX calculation period
            - long_period: Long ALMA period
            - alma_offset: ALMA offset parameter (0-1)
            - alma_sigma: ALMA sigma parameter
            - alma_min_separation: Minimum ALMA separation for valid signal
            - momentum_lookback_long: Momentum breakout lookback for long entries
            - momentum_lookback_short: Momentum breakout lookback for exits
            - macro_ema_period: Macro trend EMA period
            - slow_ema_rising_lookback: Lookback for macro EMA slope
    
    Returns:
        Tuple of (entries, exits, position_target) as pandas Series
    """
    close_np = close.to_numpy(dtype=np.float64, copy=False)
    high_np = high.to_numpy(dtype=np.float64, copy=False)
    low_np = low.to_numpy(dtype=np.float64, copy=False)
    open_np = open.to_numpy(dtype=np.float64, copy=False)
    n = len(close_np)

    # Extract parameters
    min_short_period = int(params["min_short_period"])
    max_short_period = int(params["max_short_period"])
    adx_period = int(params["adx_period"])
    long_period = int(params["long_period"])
    alma_offset = params["alma_offset"]
    alma_sigma = params["alma_sigma"]
    alma_min_separation = params["alma_min_separation"]
    momentum_lookback_long = int(params["momentum_lookback_long"])
    momentum_lookback_short = int(params["momentum_lookback_short"])
    macro_ema_period = int(params["macro_ema_period"])
    slow_ema_rising_lookback = int(params["slow_ema_rising_lookback"])
    
    # Check if optional filters are enabled (0 = disabled)
    use_momentum_long = momentum_lookback_long > 0
    use_momentum_short = momentum_lookback_short > 0
    use_macro_filter = macro_ema_period > 0
    use_slow_ema_rising = slow_ema_rising_lookback > 0

    # Calculate log returns
    returns = np.log(close_np / np.roll(close_np, 1))
    returns[0] = 0.0

    # Calculate ADX for dynamic short period
    adx_values = calculate_adx(high_np, low_np, close_np, adx_period)
    
    # Map ADX to dynamic short period (linear relationship: higher ADX = longer period)
    # Normalize ADX (cap at 50, typical range is 0-50)
    adx_normalized = np.minimum(adx_values / 50.0, 1.0)
    dynamic_short_periods = np.round(min_short_period + (max_short_period - min_short_period) * adx_normalized).astype(np.int32)
    
    # Calculate long-term ALMA (static period)
    long_term = alma_numba(returns, long_period, alma_offset, alma_sigma)
    baseline = long_term
    
    # Calculate short-term ALMA with dynamic period (need to handle varying periods)
    short_term = np.zeros(n, dtype=np.float64)
    for i in range(n):
        period = dynamic_short_periods[i]
        if period < 1:
            period = min_short_period  # Safety fallback
        short_term[i] = alma_numba(returns[:i+1], min(period, i+1), alma_offset, alma_sigma)[i]

    # Macro filter (optional - only calculate if enabled)
    if use_macro_filter:
        macro_ema = pd.Series(close_np).ewm(span=macro_ema_period, adjust=False).mean().to_numpy()
        in_bull_market = close_np > macro_ema
    else:
        macro_ema = close_np  # Dummy value
        in_bull_market = np.ones(n, dtype=bool)  # Always true when disabled

    # ALMA trend states
    bullish_state = short_term > baseline
    bearish_state = short_term < baseline

    # ALMA separation check - ensure meaningful cross with minimum distance
    alma_separation = np.abs(short_term - baseline)
    valid_separation = alma_separation >= alma_min_separation

    # Momentum filters for long entries (optional - only calculate if enabled)
    if use_momentum_long:
        highest_high_prev = pd.Series(high_np).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        # Close must be above the highest of (previous open or close)
        prev_body_top = np.maximum(open_np, close_np)
        highest_body_top_prev = pd.Series(prev_body_top).shift(1).rolling(momentum_lookback_long).max().to_numpy()
        is_highest_close = (high_np >= np.nan_to_num(highest_high_prev, nan=0)) & \
                           (close_np >= np.nan_to_num(highest_body_top_prev, nan=0))
    else:
        is_highest_close = np.ones(n, dtype=bool)  # Always true when disabled
    
    # Momentum filters for short/exit signals (optional - only calculate if enabled)
    if use_momentum_short:
        lowest_low_prev = pd.Series(low_np).shift(1).rolling(momentum_lookback_short).min().to_numpy()
        # Close must be below the lowest of (previous open or close)
        prev_body_bottom = np.minimum(open_np, close_np)
        lowest_body_bottom_prev = pd.Series(prev_body_bottom).shift(1).rolling(momentum_lookback_short).min().to_numpy()
        is_lowest_low = (low_np <= np.nan_to_num(lowest_low_prev, nan=np.inf)) & \
                        (close_np <= np.nan_to_num(lowest_body_bottom_prev, nan=np.inf))
    else:
        is_lowest_low = np.ones(n, dtype=bool)  # Always true when disabled

    # Macro EMA rising filter (optional - only calculate if enabled)
    if use_slow_ema_rising:
        macro_ema_rising = np.zeros(n, dtype=bool)
        for i in range(slow_ema_rising_lookback, n):
            macro_ema_rising[i] = macro_ema[i] > macro_ema[i - slow_ema_rising_lookback]
    else:
        macro_ema_rising = np.ones(n, dtype=bool)  # Always true when disabled

    # Build buy signal (matches Pine Script exactly)
    buy_signal_base = bullish_state & valid_separation
    if use_momentum_long:
        buy_signal_base = buy_signal_base & is_highest_close
    if use_macro_filter:
        buy_signal_base = buy_signal_base & in_bull_market
    if use_slow_ema_rising:
        buy_signal_base = buy_signal_base & macro_ema_rising

    # Build sell signal base (bearish state + momentum)
    sell_signal_base = bearish_state & valid_separation
    if use_momentum_short:
        sell_signal_base = sell_signal_base & is_lowest_low

    # STATEFUL POSITION TRACKING (matching Pine Script logic)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    
    # Minimum bars required for indicators to be valid (warm-up period)
    # Account for ADX needing 2*period bars and max_short_period
    min_bars = max(long_period, max_short_period, 2 * adx_period)
    if use_macro_filter:
        min_bars = max(min_bars, macro_ema_period)
    
    in_position = False
    
    for i in range(n):
        # Skip trading until we have enough bars for indicator warm-up
        if i < min_bars:
            continue
            
        # Entry logic (only when not in position)
        if not in_position and buy_signal_base[i]:
            entries[i] = True
            in_position = True
        
        # Exit logic (only when in position)
        elif in_position and sell_signal_base[i]:
            exits[i] = True
            in_position = False

    # Position sizing (100% allocation)
    position_target = np.ones(n, dtype=np.float64)
    position_series = pd.Series(position_target, index=close.index)

    return (pd.Series(entries, index=close.index),
            pd.Series(exits, index=close.index),
            position_series)
