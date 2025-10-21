# Installation & Setup Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run optimization
python optimize_hermes.py
```

## What Gets Generated

### 📊 QuantStats HTML Reports

**Location**: `reports/quantstats/`

Beautiful HTML tearsheet reports for each iteration with:

- **Returns Analysis**: Cumulative, monthly, rolling metrics
- **Risk Metrics**: Sharpe, Sortino, Calmar, Max Drawdown
- **Trade Analysis**: Win rate, avg win/loss, profit factor
- **Visualizations**: Equity curve, drawdown plot, monthly heatmap
- **vs Buy & Hold**: Direct comparison with benchmark

**Example**: `reports/quantstats/STAGE_1_GLOBAL_SEARCH_Iter1_2017BullMarketto2020BullMarket.html`

### 🔥 Parameter Heatmaps

**Location**: `reports/heatmaps/`

CSV files showing how performance changes when parameters vary:

1. **Short vs Long Period** - Which period combinations are robust?
2. **ALMA Offset vs Sigma** - How sensitive is smoothing?
3. **Buy vs Sell Momentum** - Which lookback values are stable?

**Use these to**:

- Identify which parameters need precision vs can vary
- Find "flat" regions (robust parameters)
- Avoid "cliff edges" (sensitive parameters)

**Example**: Open CSVs in Excel and create heatmaps with conditional formatting.

---

## Installation Details

### Dependencies

**Core**:

- `numpy`, `pandas` - Data manipulation
- `vectorbt` - Backtesting engine
- `scikit-optimize` - Bayesian optimization
- `numba` - JIT acceleration

**Reporting**:

- `quantstats` - HTML tearsheet reports
- `matplotlib`, `plotly` - Heatmap generation

### Optional: Install QuantStats

If you skip QuantStats initially, the script will still run but won't generate HTML reports:

```bash
pip install quantstats
```

---

## Understanding Output

### CSV Results

**hermes_stage1_results.csv** and **hermes_stage2_results.csv** contain:

```csv
iteration,train_start,train_end,test_start,test_end,
Short Period,Long Period,ALMA Offset,ALMA Sigma,
Buy Lookback,Sell Lookback,Macro EMA Period,
train_composite,train_sortino,test_composite,test_sortino,
test_calmar,test_return,test_sharpe,test_max_dd,
num_trades,win_rate,train_vol,test_vol,vol_regime
```

**Key Columns**:

- `test_composite` - Overall quality score (higher = better)
- `test_sortino` - Downside risk-adjusted returns
- `test_calmar` - Return / Max Drawdown
- `train_composite - test_composite` - Overfitting indicator (lower = better)

---

## Interpreting Results

### Good Signs ✅

- Test composite > 1.5
- Train/test degradation < 50%
- Heatmaps show "flat" regions around optimal params
- 10-30 trades per test period
- Win rate 30-65%

### Warning Signs ⚠️

- Test composite < 1.0
- Train/test degradation > 70%
- Heatmaps show "cliffs" (sharp dropoffs)
- < 5 trades (unreliable) or > 100 trades (overtrading)
- Win rate < 20% or > 90% (suspicious)

---

## Next Steps

1. **Open QuantStats HTML reports** - Beautiful, interactive analysis
2. **Review parameter heatmaps** - Understand robustness
3. **Compare Stage 1 vs Stage 2** - Did refinement help?
4. **Check best iteration** - Which bull market transition worked best?

**Recommended**: Use parameters from the iteration with **highest test_composite** score.
