# Hermes Strategy - AI Assistant Guidelines

## Project Overview

Hermes is a sophisticated trading strategy system that uses ALMA (Arnaud Legoux Moving Average) filtering to capture trend signals across any asset class. The codebase consists of:

- `optimize_hermes_SIMPLE.py`: Base version (9 parameters)
- `optimize_hermes_complex.py`: Advanced version (18 parameters)
- `hermes_simple.pine`: TradingView Pine Script implementation (simple)
- `hermes_complex.pine`: TradingView Pine Script implementation (complex)

## Key Architectural Concepts

### 1. Returns-Based Foundation

- Strategy operates on **log returns** (`r_t = log(P_t / P_{t-1})`), not raw prices
- This provides scale invariance and asset-agnostic behavior
- See `alma_numba()` implementation for core signal processing

### 2. Dual Strategy Versions

```python
# Simple Mode (9 parameters)
SIMPLE_MODE = True  # in optimize_hermes_SIMPLE.py
run_strategy_simple()  # Basic ALMA with fixed parameters

# Complex Mode (18 parameters)
SIMPLE_MODE = False  # in optimize_hermes_complex.py
run_strategy()  # Dynamic ALMA with trend detection
```

### 3. Core Components

- **ALMA Filtering**: Uses Numba-accelerated implementations for performance
- **Trend Detection**: Efficiency ratio + R-squared in complex mode
- **Room-to-Breathe System**: Dynamic parameter interpolation (complex mode only)

## Development Workflows

### Parameter Optimization

```python
# Quick Testing
QUICK_MODE = True
OPTIMIZATION_METHOD = "genetic"  # or "bayesian"

# Full Walk-Forward
QUICK_MODE = False
```

### Running Optimizations

1. Configure Python environment:

```python
configure_python_environment()
install_python_packages(['numpy', 'pandas', 'vectorbt', 'numba', 'skopt'])
```

2. Optimize strategy:

```python
python optimize_hermes_SIMPLE.py  # For simple mode
python optimize_hermes_complex.py  # For complex mode
```

### Testing & Validation

- Use `run_strategy_simple()` or `run_strategy()` directly for strategy testing
- Results appear in `reports/` directory
- Check optimization output in `hermes_*_optimization_results.csv`

## Code Conventions

### 1. Numba-Accelerated Functions

- All core mathematical functions use `@njit(cache=True, fastmath=True)`
- Input arrays should be `numpy.float64` type
- Preallocate output arrays for performance

```python
@njit(cache=True, fastmath=True)
def alma_numba(src, period, offset, sigma):
    result = np.empty(n, dtype=np.float64)
    # ... implementation
```

### 2. Parameter Space Definition

```python
STAGE1_SPACE = [
    Integer(10, 150, name="short_period"),
    Integer(100, 400, name="long_period"),
    # ... other parameters
]
```

### 3. Error Handling

- Use explicit numpy array types
- Handle edge cases in numerical functions
- Validate parameters before optimization

## Integration Points

### 1. Data Pipeline

- Input: CSV files with OHLCV data
- Required columns: time/timestamp/date, open, high, low, close
- Optional: volume

### 2. Performance Metrics

- Primary: Sortino Ratio, Calmar Ratio
- Secondary: Win rate, trades per year
- Constraints: Max drawdown, minimum trades

## Common Tasks

### Adding New Features

1. Implement in both Python optimizer and Pine Script
2. Add parameters to appropriate parameter space
3. Update optimization objective if needed
4. Test in QUICK_MODE first

### Debugging Tips

- Use `showDebugInfo = true` in Pine Script
- Check `reports/quantstats/` for detailed performance breakdowns
- Validate parameter relationships in `reports/heatmaps/`

## Project Structure

```
hermes/
├── optimize_hermes_SIMPLE.py   # Simple strategy optimizer
├── optimize_hermes_complex.py  # Complex strategy optimizer
├── hermes_simple.pine         # TradingView simple implementation
├── hermes_complex.pine        # TradingView complex implementation
├── docs/                      # Documentation
└── reports/                   # Generated reports
```

## Best Practices

1. Always test parameter changes in QUICK_MODE first
2. Maintain parity between Python and Pine Script implementations
3. Use Numba acceleration for computationally intensive functions
4. Document parameter relationships and constraints
5. Validate changes with walk-forward optimization

## References

- See `docs/context.md` for detailed technical background
- Review optimization logs in `reports/` for parameter tuning
- Check Pine Script code for real-time implementation details
