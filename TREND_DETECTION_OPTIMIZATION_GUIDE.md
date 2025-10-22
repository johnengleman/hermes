# Advanced Trend Detection System - Optimization Guide

## Overview

The Hermes strategy now includes a **sophisticated multi-factor trend detection system** that replaces the simple "slowEma > entryPrice" logic with a composite indicator built from 7 advanced metrics.

### What Changed

**Before:** Simple check if `slowEma > positionEntryPrice` to determine trending vs ranging regime

**After:** Multi-dimensional trend analysis using:
1. **Hurst Exponent** (R/S Analysis) - Measures long-term memory and trend persistence
2. **Efficiency Ratio** - Directional movement / total movement
3. **R-Squared** (Linear Regression) - How linear is the price movement
4. **Fractal Dimension** (Higuchi Method) - Market smoothness vs choppiness
5. **Autocorrelation** - Momentum persistence at lag-1
6. **Volatility Clustering** (GARCH-like) - Regime detection via vol-of-vol
7. **Directional Consistency** - Ratio of consistent directional moves

## Dynamic ALMA Parameters

The strategy now **automatically adjusts ALMA settings** based on the detected market regime:

### Ranging Market (Trend Strength < Threshold)
- Uses `rangingAlmaOffset` (default: 0.95) - More Gaussian, smoother
- Uses `rangingAlmaSigma` (default: 4.0) - More reactive to changes

### Trending Market (Trend Strength > Threshold)
- Uses `trendingAlmaOffset` (default: 0.75) - Less Gaussian, more responsive
- Uses `trendingAlmaSigma` (default: 8.0) - Smoother to avoid whipsaws

### Interpolation
The ALMA parameters **smoothly interpolate** between ranging and trending values based on the composite trend strength score (0-1 scale).

```pine
dynamicAlmaOffset = rangingAlmaOffset + (trendStrength * (trendingAlmaOffset - rangingAlmaOffset))
dynamicAlmaSigma = rangingAlmaSigma + (trendStrength * (trendingAlmaSigma - rangingAlmaSigma))
```

## Optimizable Parameters

All of these parameters are now **exposed as inputs** and can be optimized:

### 1. Trend Detection Periods
- `trendAnalysisPeriod` (20-200): Lookback for most metrics
- `hurstPeriod` (50-300): Longer = more reliable Hurst calculation
- `fractalPeriod` (10-100): Lookback for fractal dimension

### 2. Trend Threshold
- `trendThreshold` (0.0-1.0): Score above which market is "trending"
  - Higher = more conservative (only trade strong trends)
  - Lower = more aggressive (use trending logic more often)

### 3. ALMA Parameter Ranges
- `trendingAlmaOffset` (0.0-1.0): Offset for trending markets
- `rangingAlmaOffset` (0.0-1.0): Offset for ranging markets
- `trendingAlmaSigma` (1.0-15.0): Sigma for trending markets
- `rangingAlmaSigma` (1.0-15.0): Sigma for ranging markets

### 4. Component Weights (OPTIMIZABLE!)
These weights determine how much each metric contributes to the final trend strength score:

- `weightHurst` (0.0-1.0, default: 0.25)
- `weightEfficiency` (0.0-1.0, default: 0.20)
- `weightRSquared` (0.0-1.0, default: 0.15)
- `weightFractal` (0.0-1.0, default: 0.15)
- `weightAutocorr` (0.0-1.0, default: 0.10)
- `weightVolCluster` (0.0-1.0, default: 0.10)
- `weightDirConsistency` (0.0-1.0, default: 0.05)

**Note:** Weights are automatically normalized to sum to 1.0 internally.

## How to Optimize with Your Existing Optimizer

Your `optimize_hermes.py` is already set up perfectly! Here's how to add the new parameters:

### Option 1: Simple - Just Optimize ALMA Ranges

Add to your search space:

```python
# In STAGE1_SPACE or STAGE2_SPACE:
Integer(75, 95, name="trending_alma_offset_int"),  # 0.75-0.95
Integer(85, 99, name="ranging_alma_offset_int"),   # 0.85-0.99
Categorical([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], name="trending_alma_sigma"),
Categorical([3.0, 4.0, 5.0, 6.0, 7.0], name="ranging_alma_sigma"),
Integer(40, 80, name="trend_threshold_int"),  # 0.40-0.80 (divide by 100)
```

### Option 2: Advanced - Optimize Component Weights

This is the **most powerful** approach:

```python
# Add to search space:
Integer(0, 50, name="weight_hurst"),       # Will divide by 100
Integer(0, 50, name="weight_efficiency"),
Integer(0, 50, name="weight_rsquared"),
Integer(0, 50, name="weight_fractal"),
Integer(0, 50, name="weight_autocorr"),
Integer(0, 50, name="weight_volcluster"),
Integer(0, 30, name="weight_dirconsist"),  # Smaller range (less important)
```

**Why this works:** The optimizer will discover which metrics are most predictive for your specific asset and time period!

### Option 3: Full Optimization (Recommended for Stage 2)

Combine both approaches for maximum flexibility:

```python
STAGE2_SPACE_ADVANCED = [
    # ... existing ALMA parameters ...

    # Trend detection periods
    Integer(30, 100, name="trend_analysis_period"),
    Integer(60, 200, name="hurst_period"),
    Integer(20, 60, name="fractal_period"),

    # Trend threshold
    Integer(40, 80, name="trend_threshold_int"),  # 0.40-0.80

    # ALMA dynamic ranges
    Integer(70, 85, name="trending_alma_offset_int"),  # 0.70-0.85
    Integer(90, 99, name="ranging_alma_offset_int"),   # 0.90-0.99
    Categorical([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], name="trending_alma_sigma"),
    Categorical([3.0, 4.0, 5.0, 6.0], name="ranging_alma_sigma"),

    # Component weights (most powerful!)
    Integer(10, 40, name="weight_hurst"),
    Integer(10, 40, name="weight_efficiency"),
    Integer(5, 30, name="weight_rsquared"),
    Integer(5, 30, name="weight_fractal"),
    Integer(0, 25, name="weight_autocorr"),
    Integer(0, 25, name="weight_volcluster"),
    Integer(0, 15, name="weight_dirconsist"),
]
```

## Implementation Notes

### TradingView Compatibility
The Pine Script code uses only standard TradingView functions - no external libraries needed.

### Performance Considerations
The trend detection functions use loops for calculations (Hurst, Fractal, R²). This is computationally intensive but provides institutional-grade analysis.

**Tip:** Start with longer timeframes (daily) to reduce computation. The indicators work on any timeframe.

### Interpretability
When `showDebugInfo = true`, you'll see:
- Trend strength line overlaid on the ALMA indicator
- Background color: green = trending, red = ranging
- Threshold line showing the cutoff

## Expected Benefits

1. **More Accurate Regime Detection**: Multi-factor analysis is far more robust than single indicators
2. **Better Exit Timing**: Trending markets use different exit logic (HMA crossunder) vs ranging (ALMA crossunder)
3. **Adaptive Smoothing**: ALMA automatically becomes more/less responsive based on market conditions
4. **Optimization Potential**: The component weights can be tuned per asset (BTC might favor Hurst, ETH might favor Efficiency Ratio)

## Validation Strategy

To validate the new system:

1. **Run Stage 1** with default weights (current values)
2. **Analyze results** - which periods had best performance?
3. **Run Stage 2** with weight optimization on best-performing periods
4. **Compare metrics**:
   - Does optimized weight ensemble beat uniform weights?
   - Which components got highest weights?
   - Is performance stable across walk-forward windows?

## Quick Start

1. Load `hermes.pine` in TradingView
2. Apply to any chart
3. Toggle `Debug Info` to see trend strength visualization
4. Experiment with manual parameter adjustments
5. When satisfied, use the optimizer to find optimal weights for your specific asset

## Next Steps

Consider adding these advanced features:
- **Multi-timeframe trend detection** - Check higher timeframes for macro trend
- **Regime-specific position sizing** - Larger positions in trending markets
- **Adaptive thresholds** - Trend threshold varies with volatility
- **Machine learning** - Use historical data to train optimal weights per regime

---

**Remember:** This is a sophisticated system. Start with defaults, understand the behavior, then optimize systematically using your walk-forward framework.
