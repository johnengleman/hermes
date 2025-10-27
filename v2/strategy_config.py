"""
Hermes Strategy Configuration
Strategy-specific parameters, bounds, and defaults
"""

# Default parameter values
MANUAL_DEFAULTS = {
    "min_short_period": 10,
    "max_short_period": 50,
    "adx_period": 14,
    "long_period": 250,
    "alma_offset": 0.95,
    "alma_sigma": 4.0,
    "alma_min_separation": 0.0001,
    "momentum_lookback_long": 1,
    "momentum_lookback_short": 1,
    "macro_ema_period": 100,
    "slow_ema_rising_lookback": 3,
    "commission_rate": 0.0,  # Set to 0 to match TradingView for pure strategy comparison
    "slippage_rate": 0.0,  # Set to 0 to match TradingView (process_orders_on_close=true)
}

# Parameter optimization ranges with explicit specifications
PARAM_RANGES = {
    "min_short_period": {
        "type": "categorical",
        "values": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "description": "Minimum ALMA short period when ADX is low"
    },
    "max_short_period": {
        "type": "categorical",
        "values": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150],
        "description": "Maximum ALMA short period when ADX is high"
    },
    "adx_period": {
        "type": "categorical",
        "values": [5, 7, 10, 14, 20, 25, 30],
        "description": "ADX period for dynamic short period calculation"
    },
    "long_period": {
        "type": "categorical",
        "values": [100, 120, 150, 175, 200, 225, 250],
        "description": "ALMA long period (bars)"
    },
    "alma_offset": {
        "type": "categorical",
        "values": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1],
        "description": "ALMA offset (Gaussian center)"
    },
    "alma_sigma": {
        "type": "categorical",
        "values": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        "description": "ALMA sigma (Gaussian width)"
    },
    "momentum_lookback_long": {
        "type": "categorical",
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "description": "Bars to check for momentum confirmation when opening long positions (0 = disabled)"
    },
    "momentum_lookback_short": {
        "type": "categorical",
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "description": "Bars to check for momentum confirmation when closing/selling (0 = disabled)"
    },
    "macro_ema_period": {
        "type": "categorical",
        "values": [0, 150, 200, 250, 300, 350, 400, 450, 500],
        "description": "Macro trend EMA period (0 = disabled)"
    },
    "slow_ema_rising_lookback": {
        "type": "categorical",
        "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
        "description": "Bars to confirm macro EMA rising trend (0 = disabled)"
    },
}

# Strategy metadata
STRATEGY_NAME = "Hermes Simple"
STRATEGY_VERSION = "2.0"
NUM_PARAMETERS = 10  # Used for genetic algorithm population sizing

# Parameter order for optimization (must match decode_parameters function)
PARAM_ORDER = [
    "min_short_period",
    "max_short_period",
    "adx_period",
    "long_period", 
    "alma_offset",
    "alma_sigma",
    "momentum_lookback_long",
    "momentum_lookback_short",
    "macro_ema_period",
    "slow_ema_rising_lookback",
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
    params["alma_min_separation"] = MANUAL_DEFAULTS["alma_min_separation"]
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
    # Constraint: min_short_period must be less than max_short_period
    if params["min_short_period"] >= params["max_short_period"]:
        return False, 999.0
    
    # Constraint: max short period must be less than long period
    if params["max_short_period"] >= params["long_period"]:
        return False, 999.0
    
    # Constraint: adequate separation between max short and long
    if params["long_period"] - params["max_short_period"] < 20:
        return False, 999.0
    
    return True, 0.0
