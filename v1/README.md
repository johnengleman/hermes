# Hermes Strategy v1

This folder contains the strategy implementation and configuration for the Hermes ALMA-based trend following strategy.

## Files

- **`hermes.py`**: Core strategy implementation with `run_strategy_simple()` function
- **`strategy_config.py`**: Strategy-specific parameters, bounds, and defaults

## Creating a New Strategy

To create a new strategy version (e.g., v2), follow these steps:

### 1. Create Strategy Directory

```bash
mkdir v2
```

### 2. Implement Strategy Function

Create `v2/your_strategy.py`:

```python
"""
Your Strategy Name
Description of your strategy
"""

import pandas as pd
import numpy as np


def run_strategy(close, high, low, **params):
    """
    Your strategy implementation.
    
    Args:
        close: pd.Series of close prices (indexed by datetime)
        high: pd.Series of high prices
        low: pd.Series of low prices
        **params: Strategy parameters (from strategy_config.py)
    
    Returns:
        tuple: (entries, exits, position_target)
            - entries: pd.Series of boolean entry signals
            - exits: pd.Series of boolean exit signals
            - position_target: pd.Series of position sizing (typically 1.0 for 100%)
    """
    # Your strategy logic here
    
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    position_target = pd.Series(1.0, index=close.index)
    
    return entries, exits, position_target
```

### 3. Create Strategy Configuration

Create `v2/strategy_config.py`:

```python
"""
Your Strategy Configuration
Strategy-specific parameters, bounds, and defaults
"""

# Default parameter values
MANUAL_DEFAULTS = {
    "param1": 10,
    "param2": 0.5,
    # ... add all your parameters with default values
    "commission_rate": 0.0035,
    "slippage_rate": 0.0005,
}

# Parameter optimization ranges with explicit specifications
PARAM_RANGES = {
    "param1": {
        "type": "integer",
        "min": 5,
        "max": 50,
        "description": "Description of param1"
    },
    "param2": {
        "type": "float",
        "min": 0.1,
        "max": 0.9,
        "precision": 2,  # 2 decimal places
        "description": "Description of param2"
    },
    "param3": {
        "type": "categorical",
        "values": [1.0, 2.0, 3.0, 5.0, 10.0],
        "description": "Description of param3"
    },
    # ... define ranges for each parameter
}

# Strategy metadata
STRATEGY_NAME = "Your Strategy Name"
STRATEGY_VERSION = "2.0"
NUM_PARAMETERS = 3  # Total number of optimizable parameters

# Parameter order for optimization (must match decode_parameters)
PARAM_ORDER = ["param1", "param2", "param3"]


def get_optimization_bounds():
    """Build bounds array for scipy's differential_evolution."""
    bounds = []
    for param_name in PARAM_ORDER:
        param_spec = PARAM_RANGES[param_name]
        param_type = param_spec["type"]
        
        if param_type == "categorical":
            bounds.append((0, len(param_spec["values"]) - 1))
        elif param_type == "float":
            precision = param_spec.get("precision", 2)
            bounds.append((int(param_spec["min"] * (10 ** precision)), 
                          int(param_spec["max"] * (10 ** precision))))
        else:  # integer
            bounds.append((param_spec["min"], param_spec["max"]))
    
    return bounds


def decode_parameters(x):
    """Convert optimization array to strategy parameters dictionary."""
    params = {}
    for i, param_name in enumerate(PARAM_ORDER):
        param_spec = PARAM_RANGES[param_name]
        param_type = param_spec["type"]
        
        if param_type == "categorical":
            idx = int(round(x[i]))
            idx = max(0, min(idx, len(param_spec["values"]) - 1))
            params[param_name] = param_spec["values"][idx]
        elif param_type == "float":
            precision = param_spec.get("precision", 2)
            params[param_name] = round(x[i] / (10 ** precision), precision)
        else:  # integer
            params[param_name] = int(round(x[i]))
    
    return params


def validate_parameters(params):
    """Check parameter constraints (return (is_valid, penalty_score))."""
    # Add your parameter constraints here
    return True, 0.0
```

### 4. Update Optimizer Imports

Modify `optimize.py` to import from your new strategy:

```python
# Import strategy-specific components
from v2.your_strategy import run_strategy
from v2.strategy_config import MANUAL_DEFAULTS, PARAM_RANGES, NUM_PARAMETERS
```

### 5. Update Optimizer Bounds (if needed)

If your parameters have different types (categorical, integer ranges, float ranges), update the `bounds` list in `optimize_parameters_genetic()` to match your parameter structure.

## Parameter Types

The optimizer supports three parameter types with explicit specifications:

1. **Integer ranges**: 
   ```python
   "param_name": {
       "type": "integer",
       "min": 10,
       "max": 100,
       "description": "Parameter description"
   }
   ```

2. **Float ranges**: 
   ```python
   "param_name": {
       "type": "float",
       "min": 0.1,
       "max": 0.99,
       "precision": 2,  # Number of decimal places
       "description": "Parameter description"
   }
   ```

3. **Categorical**: 
   ```python
   "param_name": {
       "type": "categorical",
       "values": [2.0, 3.0, 4.0, 5.0],
       "description": "Parameter description"
   }
   ```

## Example: Hermes v1 Parameters

Current implementation (9 parameters):

1. `short_period` - Integer (10-150)
2. `long_period` - Integer (100-400)
3. `alma_offset` - Float (0.80-0.99, precision=2)
4. `alma_sigma` - Categorical [2.0, 3.0, ..., 10.0]
5. `momentum_lookback` - Integer (1-10)
6. `macro_ema_period` - Integer (50-250)
7. `fast_hma_period` - Integer (10-100)
8. `slow_ema_period` - Integer (30-200)
9. `slow_ema_rising_lookback` - Integer (1-15)

## Testing Your Strategy

1. Update imports in `optimize.py`
2. Run quick optimization: `python optimize.py` (with `QUICK_MODE = True`)
3. Review results in `hermes_simple_optimization_results.csv`
4. If satisfied, set `QUICK_MODE = False` for full walk-forward testing
