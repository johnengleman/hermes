"""
Hermes Strategy Configuration
Strategy-specific parameters, bounds, and defaults
"""

# Default parameter values (FOREX-TUNED for EUR/USD on 30min timeframe)
MANUAL_DEFAULTS = {
    "short_period": 10,
    "long_period": 80,
    "alma_offset": 0.75,
    "alma_sigma": 4.0,
    "momentum_lookback": 5,
    "fast_ema_period": 20,
    "slow_ema_period": 60,
    "trending_regime_min_distance": 5,  # 5 pips for EUR/USD (pip_size = 0.0001)
    "broker_leverage": 50.0,  # Oanda EUR/USD leverage (change to 20 for USD/JPY)
    "risk_per_trade_pct": 2.0,  # 2% risk per trade
    "commission_rate": 0.0,  # Set to 0 to match TradingView for pure strategy comparison
    "slippage_rate": 0.0,  # Set to 0 to match TradingView (process_orders_on_close=true)
}

# Parameter optimization ranges with explicit specifications
PARAM_RANGES = {
    "short_period": {
        "type": "categorical",
        "values": [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
        "description": "ALMA short period (bars) - widened for 30min timeframe"
    },
    "long_period": {
        "type": "categorical",
        "values": [30, 40, 50, 60, 80, 100, 120, 150, 175, 200, 225, 250, 300, 350, 400],
        "description": "ALMA long period (bars) - widened range"
    },
    "alma_offset": {
        "type": "categorical",
        "values": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "description": "ALMA offset (Gaussian center) - added lower values for more responsiveness"
    },
    "alma_sigma": {
        "type": "categorical",
        "values": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0],
        "description": "ALMA sigma (Gaussian width) - added lower values for sharper response"
    },
    "momentum_lookback": {
        "type": "categorical",
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40, 50],
        "description": "Momentum lookback (0 = disabled) - extended range for slower confirmation"
    },
    "fast_ema_period": {
        "type": "categorical",
        "values": [3, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 125, 150],
        "description": "Fast EMA period - much wider range from very fast to slow"
    },
    "slow_ema_period": {
        "type": "categorical",
        "values": [20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400],
        "description": "Slow EMA period - widened to allow both fast and slow trend filters"
    },
    "trending_regime_min_distance": {
        "type": "categorical",
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "description": "Trending regime threshold in pips (1 pip = 0.0001 for EUR/USD)"
    },
}

# Strategy metadata
STRATEGY_NAME = "Hermes Simple"
STRATEGY_VERSION = "1.0"
NUM_PARAMETERS = 8  # Used for genetic algorithm population sizing

# Parameter order for optimization (must match decode_parameters function)
PARAM_ORDER = [
    "short_period",
    "long_period", 
    "alma_offset",
    "alma_sigma",
    "momentum_lookback",
    "fast_ema_period",
    "slow_ema_period",
    "trending_regime_min_distance",
]


def get_optimization_bounds():
    """
    Build bounds array for scipy's differential_evolution.
    All parameters are categorical, so bounds are (0, num_values-1) for each.
    
    Returns:
        list: Bounds tuples for each parameter in PARAM_ORDER
    """
    bounds = []
    for param_name in PARAM_ORDER:
        param_spec = PARAM_RANGES[param_name]
        # All parameters are categorical - use index bounds
        bounds.append((0, len(param_spec["values"]) - 1))
    
    return bounds


def decode_parameters(x):
    """
    Convert optimization array to strategy parameters dictionary.
    All parameters are categorical - round to index and lookup value.
    
    Args:
        x: Array of continuous values from optimizer
        
    Returns:
        dict: Strategy parameters ready for run_strategy_simple()
    """
    params = {}
    
    for i, param_name in enumerate(PARAM_ORDER):
        param_spec = PARAM_RANGES[param_name]
        # Round to nearest index and clamp to valid range
        idx = int(round(x[i]))
        idx = max(0, min(idx, len(param_spec["values"]) - 1))
        params[param_name] = param_spec["values"][idx]
    
    # Add fixed parameters from MANUAL_DEFAULTS that aren't being optimized
    params["broker_leverage"] = MANUAL_DEFAULTS["broker_leverage"]
    params["risk_per_trade_pct"] = MANUAL_DEFAULTS["risk_per_trade_pct"]
    params["commission_rate"] = MANUAL_DEFAULTS["commission_rate"]
    params["slippage_rate"] = MANUAL_DEFAULTS["slippage_rate"]
    
    return params


def validate_parameters(params):
    """
    Check parameter constraints specific to this strategy.
    
    Args:
        params: Dictionary of strategy parameters
        
    Returns:
        tuple: (is_valid, penalty_score)
            - is_valid: False if constraints violated
            - penalty_score: High value to return if invalid (999.0)
    """
    # Constraint: short period must be less than long period
    if params["short_period"] >= params["long_period"]:
        return False, 999.0
    
    # Constraint: adequate separation between short and long
    if params["long_period"] - params["short_period"] < 20:
        return False, 999.0
    
    # Constraint: fast EMA must be faster than slow EMA
    if params["fast_ema_period"] >= params["slow_ema_period"]:
        return False, 999.0
    
    return True, 0.0
