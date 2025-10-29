"""
Hermes Strategy Configuration - Long-Only Bitcoin
Strategy-specific parameters, bounds, and defaults
Optimized for Bitcoin on daily timeframe
"""

# Default parameter values (Bitcoin daily timeframe)
MANUAL_DEFAULTS = {
    "short_period": 20,
    "long_period": 200,
    "alma_offset": 0.85,
    "alma_sigma": 6.0,
    "alma_min_separation": 0.0001,
    "momentum_lookback_long": 11,
    "fast_ema_period": 25,
    "slow_ema_period": 60,
    "trending_regime_min_distance": 0.035,  # 3.5% distance for trending regime activation
    "commission_rate": 0.000,  # 0.1% commission for crypto exchanges
    "slippage_rate": 0.0000,  # 0.05% slippage for Bitcoin
}

# Parameter optimization ranges with explicit specifications
PARAM_RANGES = {
    "short_period": {
        "type": "categorical",
        "values": [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80],
        "description": "ALMA short period (bars) - for daily Bitcoin timeframe"
    },
    "long_period": {
        "type": "categorical",
        "values": [100, 120, 150, 175, 200, 225, 250, 300, 350, 400],
        "description": "ALMA long period (bars) - wider range for Bitcoin trends"
    },
    "alma_offset": {
        "type": "categorical",
        "values": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "description": "ALMA offset (Gaussian center) - full range from 0.1 to 1.0"
    },
    "alma_sigma": {
        "type": "categorical",
        "values": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "description": "ALMA sigma (Gaussian width) - 2.0 to 10.0"
    },
    "alma_min_separation": {
        "type": "categorical",
        "values": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        "description": "Minimum ALMA separation to prevent whipsaw (0 = disabled)"
    },
    "momentum_lookback_long": {
        "type": "categorical",
        "values": [0, 3, 5, 7, 9, 11, 13, 15],
        "description": "Momentum lookback for long entries (0 = disabled)"
    },
    "fast_ema_period": {
        "type": "categorical",
        "values": [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
        "description": "Fast EMA period for trending regime detection"
    },
    "slow_ema_period": {
        "type": "categorical",
        "values": [100, 125, 150, 175, 200, 225, 250, 300],
        "description": "Slow EMA period for trend filter"
    },
    "trending_regime_min_distance": {
        "type": "categorical",
        "values": [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "description": "Min % distance for slow_ema above entry to activate trending regime (0 = disabled)"
    },
}

# Strategy metadata
STRATEGY_NAME = "Hermes Long-Only Bitcoin"
STRATEGY_VERSION = "3.0"
NUM_PARAMETERS = 9  # Used for genetic algorithm population sizing

# Parameter order for optimization (must match decode_parameters function)
PARAM_ORDER = [
    "short_period",
    "long_period", 
    "alma_offset",
    "alma_sigma",
    "alma_min_separation",
    "momentum_lookback_long",
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
