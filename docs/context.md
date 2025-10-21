# Hermes Strategy - Technical Context

## Overview

The **Hermes Strategy** is an adaptive trading system that uses **ALMA (Arnaud Legoux Moving Average)** filtering to capture smooth, low-lag trend signals across any asset class. It's designed for macro trend following with minimal noise and fast signal generation.

> **🔑 KEY FOUNDATION:** Unlike traditional price-based indicators, Hermes operates **entirely on price returns** (log returns), not absolute price levels. This makes it naturally scale-invariant and comparable across any asset or time period.

---

## Inspiration: Giovanni Santostasi's Power Law Indicator

The strategy was inspired by Giovanni Santostasi's Power Law Volatility Indicator for Bitcoin, which visualizes expected vs. actual returns over time. However, Hermes generalizes this concept:

**Original Concept:**

- Bitcoin-specific power law model: `log(P) = m·log(t) + c`
- Expected diminishing returns over time
- Visual comparison of theoretical vs. actual performance

**Hermes Generalization:**

- Asset-agnostic adaptive filtering
- No time-based decay assumptions
- Dynamic volatility-adjusted signals
- Works on any timeframe, any asset

---

## Core Architecture

### 1. Data Transformation (Returns-Based Foundation)

**⚠️ CRITICAL CONCEPT:** Hermes operates on **log returns**, not raw prices. This is fundamental to how the strategy works.

**Log Returns:**

```
r_t = log(P_t / P_{t-1})
```

**Why Returns Instead of Price?**

1. **Scale Invariance:** Bitcoin at $100 vs $100,000 uses same parameters
2. **Stationarity:** Returns are more stationary than prices (no trending to infinity)
3. **Comparability:** Can optimize once and apply to BTC, ETH, SPY, etc.
4. **Statistical Properties:** Returns approximate normal distribution better than prices
5. **Additive Behavior:** Log returns can be summed across time periods

**Scaling:**

- Multiply by 5000 for numeric stability and visualization
- Maintains precision while keeping values in readable range
- Doesn't change the relative behavior, just makes numbers easier to work with

### 2. ALMA Filtering (Primary Innovation)

**What is ALMA?**

ALMA (Arnaud Legoux Moving Average) is a Gaussian-weighted moving average designed to minimize lag while maximizing smoothness.

**Applied to Returns:** ALMA filters are applied to **raw log returns** (not prices). This smooths the return stream to identify trend changes in momentum, rather than smoothing price levels. No z-score normalization or volatility scaling is used - signals are based purely on return momentum.

**Mathematical Foundation:**

```
ALMA_t = Σ(w_i · r_{t-i}) / Σ(w_i)

where: r_{t-i} = scaled log returns at time t-i
       w_i = exp(-(i - μ)² / (2σ²))
       μ = offset × (period - 1)
       σ = period / sigma
```

**Key Parameters:**

- **Period**: Window length (80 for short-term, 250 for long-term)
- **Offset** (0.90): Where the Gaussian peak sits (higher = more recent weight)
- **Sigma** (7.5): Width of Gaussian curve (higher = smoother)

**Why ALMA?**

- ✅ Ultra-smooth curves (like Giovanni's indicator)
- ✅ Minimal lag compared to SMA/EMA
- ✅ Natural outlier resistance via Gaussian weighting
- ✅ No phase distortion
- ✅ Tunable lag/smoothness tradeoff

### 3. Dual Signal Structure

**Long-Term (Baseline):**

- ALMA with 250-day period
- Represents macro trend direction
- Slow-moving, stable reference

**Short-Term (Signal):**

- ALMA with 80-day period
- Captures intermediate trend changes
- Faster response to regime shifts

**State-Based Logic:**

The strategy uses **regime states**, not crossover events:

```
Bullish Regime:  Short-term > Long-term (blue line above black)
Bearish Regime:  Short-term < Long-term (blue line below black)

Buy:  Enter when in bullish regime AND filters pass
Sell: Exit when in bearish regime AND filters pass
```

This allows entries at any time during a bullish regime (not just at crossover moments), enabling faster reaction to favorable conditions.

### 4. Entry/Exit Filters

**Buy Filters (all must be true):**

1. **Bullish Regime**: Short-term ALMA > Long-term ALMA (always required)
2. **Buy Momentum**: Current close ≥ highest close of previous N bars (default: 7, always required)
3. **Crossover Strength**: Distance between ALMA lines ≥ threshold (0.0 = disabled, optional)
4. **Macro Trend Filter**: Price > 200 EMA (optional, prevents bear market trades)

**Sell Filters:**

1. **Bearish Regime**: Short-term ALMA < Long-term ALMA (always required)
2. **Sell Momentum**: Current low ≤ lowest low of previous N bars (0 = disabled, optional)

**Key Design:**

- Buy filters are stricter (requires momentum + optional strength/macro filters)
- Sell filters are more flexible (momentum can be disabled for faster exits)
- Crossover strength at 0.0 effectively disables that filter
- Macro filter only affects entries, not exits (allows normal exit in bear markets)

### 5. Macro Trend Filter (200 EMA)

**Purpose:** Avoid trading against the major market trend by only allowing entries during bull markets.

**How it works:**

- Calculate 200-period Exponential Moving Average on price
- Bull market: `close > 200 EMA` → entries allowed
- Bear market: `close < 200 EMA` → no new entries (stay flat)
- Exits work normally regardless of macro trend

**Why 200 EMA?**

- Industry standard for macro trend identification
- Well-tested on Bitcoin and traditional markets
- Smooths out medium-term noise while capturing major trends
- Used by institutional traders and algorithms

**Benefits:**

- **Reduces drawdowns** by avoiding prolonged bear markets
- **Improves risk-adjusted returns** (higher Sortino ratio)
- **Fewer losing trades** in unfavorable market conditions
- **Psychological benefit** of trading with the macro tide

**Trade-offs:**

- May miss early bull market entries (whipsaw near 200 EMA)
- Reduces total number of trades
- Adds slight complexity to strategy

**Optimizer Testing:**
The optimizer tests both `use_macro_filter=True` and `False` to determine if the filter improves risk-adjusted performance on your specific asset and time period.

---

## Design Philosophy

### Returns-Based Processing

**Why Not Price-Based Indicators?**

Traditional indicators (moving averages, RSI, MACD) operate on price levels:

- ❌ Parameters must be retuned for different price ranges
- ❌ $100 → $110 treated differently than $10,000 → $11,000 (same 10% move!)
- ❌ Can't compare signals across different assets
- ❌ Susceptible to price scale biases

**Hermes operates on returns:**

- ✅ 10% return is 10% everywhere - scale invariant
- ✅ Same parameters work on penny stocks and Bitcoin
- ✅ Directly captures what traders care about: gains/losses
- ✅ Natural stationarity (prices trend to infinity, returns don't)

### Power Law Independence

Unlike Giovanni's Bitcoin-specific power law model, Hermes:

- Makes no assumptions about long-term decay
- Adapts to any asset's natural volatility characteristics
- Uses observed data, not theoretical curves

### Asset Agnostic

Works on:

- Cryptocurrencies (Bitcoin, Ethereum, etc.)
- Equities (stocks, indices)
- Commodities (gold, oil)
- Forex pairs
- Any liquid, continuous time series

### Self-Normalizing

- **Log returns normalize price scale differences** - $1 → $2 is same signal as $10,000 → $20,000 (both 100% returns)
- **ALMA's Gaussian weighting handles outliers naturally** - No need for separate outlier detection or z-score normalization
- **Returns-based = asset-agnostic** - Same parameters work across BTC, ETH, stocks, forex
- **No manual tuning needed for different assets** - Optimize once, apply everywhere
- **Raw returns (no normalization)** - Simpler, more interpretable signals

---

## Signal Interpretation

### Regime States

**Bullish Regime (Buy Opportunity):**

- Short-term ALMA > long-term ALMA (blue line above black)
- Indicates positive momentum regime
- Entry available any time conditions are met (not just at crossover)
- Additional filters prevent weak entries

**Bearish Regime (Sell Signal):**

- Short-term ALMA < long-term ALMA (blue line below black)
- Indicates negative momentum regime
- Exit when regime shifts and optional filters confirm

### Visual Indicators

**Blue Line:** Short-term signal (fast ALMA)
**Black Line:** Long-term baseline (slow ALMA)

**Trade Markers:**

- 🟢 Green triangle up = Buy executed
- 🔴 Red triangle down = Sell executed
- 🟠 Orange "M" = Blocked by momentum (in bullish regime but not at N-bar high)
- 🟣 Purple "W" = Blocked by weak crossover strength (distance too small)

---

## Parameter Guide

### ALMA Parameters

| Parameter    | Default | Range    | Purpose                    |
| ------------ | ------- | -------- | -------------------------- |
| Short Period | 80      | 10-200   | Fast signal responsiveness |
| Long Period  | 250     | 50-400   | Baseline stability         |
| ALMA Offset  | 0.90    | 0.0-1.0  | Lag vs smoothness balance  |
| ALMA Sigma   | 7.5     | 1.0-10.0 | Gaussian curve width       |

**Tuning Tips:**

- **Shorter periods** → More trades, faster signals, more noise
- **Longer periods** → Fewer trades, slower signals, smoother curves
- **Higher offset** → Less lag but less smooth
- **Higher sigma** → Smoother but more lag

### Strategy Parameters

| Parameter              | Default | Range      | Purpose                             |
| ---------------------- | ------- | ---------- | ----------------------------------- |
| Buy Lookback           | 7       | 1-15       | Momentum confirmation bars (always) |
| Sell Lookback          | 0       | 0-10       | Exit momentum bars (0=disabled)     |
| Min Crossover Strength | 0.0     | 0.0-0.002  | ALMA separation (0=disabled)        |
| Use Macro Filter       | True    | True/False | 200 EMA bull/bear filter            |

**Key Insights:**

- Buy momentum is always required (prevents weak entries)
- Sell momentum can be disabled (0) for faster exits
- Crossover strength at 0.0 disables the filter (trades all regime states)
- Macro filter reduces bear market exposure (recommended for crypto)

---

## Comparison to Giovanni's Indicator

| Aspect        | Power Law Indicator        | Hermes Strategy               |
| ------------- | -------------------------- | ----------------------------- |
| **Theory**    | Power law decay model      | Adaptive Gaussian filtering   |
| **Assets**    | Bitcoin only               | Any asset                     |
| **Signals**   | Visual over/undervaluation | Crossover entry/exit          |
| **Smoothing** | SMA/EMA                    | ALMA (superior)               |
| **Lag**       | Standard MA lag            | Minimal lag (Gaussian)        |
| **Math**      | log(P) = m·log(t) + c      | Gaussian-weighted convolution |
| **Use Case**  | Macro cycle analysis       | Tactical trend trading        |

**Conceptual Link:**

- Both compare "expected" vs "actual" behavior
- Giovanni: Expected = power law curve
- Hermes: Expected = long-term ALMA baseline

---

## Technical Advantages

### 1. Superior Smoothing

- ALMA produces smoother curves than SMA/EMA for same period
- Gaussian weighting prevents overshoot and ringing
- Clean signals with minimal whipsaws

### 2. Low Lag

- Traditional MA lag = period/2
- ALMA lag ≈ period × (1 - offset)
- With offset=0.90: ~25 days lag vs ~125 for SMA(250)

### 3. Outlier Resistance

- Extreme values get exponentially lower weights
- No need for explicit outlier filtering
- Robust to flash crashes and data spikes

### 4. Interpretability

- Two lines, clear crossover logic
- Debug markers show filter states

---

## Mathematical Properties

### Gaussian Weighting Function

```
Weight(i) = exp(-(i - μ)² / (2σ²))
```

Where:

- `i` = distance from current bar
- `μ` = offset × (period - 1) [peak location]
- `σ` = period / sigma [curve width]

**Properties:**

- Symmetric around μ (no phase shift)
- Decays exponentially from peak
- Total weight sums to 1.0 (normalized)
- Smooth, differentiable function

### Optimization Objective

Maximize out-of-sample Sortino Ratio:

```
Sortino = (Return - RFR) / Downside_Deviation

Objective = Sortino - Penalties

Penalties:
- Excessive trades (> 200/year)
- Extreme parameters (at bounds)
- High drawdown (> 40%)
```

### Parameter Relationships (W&B Metrics)

The optimizer logs derived ratios to W&B for analyzing parameter interactions:

**`ratios/period_ratio`** = `short_period / long_period`

- Measures responsiveness (0.2-0.7 typical range)
- Lower ratio = wider separation = slower trading
- Higher ratio = tighter separation = faster trading

**`ratios/offset_sigma_ratio`** = `alma_offset / alma_sigma`

- Measures lag vs smoothness balance (0.10-0.15 typical)
- Lower = smoother but laggier
- Higher = faster but noisier

**`ratios/effective_bandwidth_short`** = `short_period / alma_sigma`

- ALMA's effective averaging window for short-term (5-30)
- Higher = more data points averaged

**`ratios/effective_bandwidth_long`** = `long_period / alma_sigma`

- ALMA's effective averaging window for long-term (20-60)

**`ratios/period_spread`** = `long_period - short_period`

- Absolute separation between ALMA lines (50-300)

**Why ratios matter:** Performance may cluster around specific parameter relationships rather than absolute values. For example, `period_ratio=0.33` might work well regardless of whether it's (50,150) or (100,300).

---

## Implementation Notes

### TradingView (hermes.pine)

- Pure Pine Script v5
- Uses native `ta.alma()` function for ALMA smoothing
- Uses `ta.ema(close, 200)` for macro trend filter
- Real-time calculation on every bar
- Visual indicators and strategy execution
- Debug markers show filter states and regime

### Python Optimizer (optimize_hermes.py)

- Numba-accelerated ALMA implementation
- 200 EMA calculated via pandas `ewm(span=200)`
- Bayesian optimization (scikit-optimize)
- Walk-forward validation
- Weights & Biases logging
- Tests both macro filter ON/OFF configurations

### Performance

- **TradingView:** Instant execution (< 1ms per bar)
- **Python:** ~500ms per strategy backtest
- **Optimization:** 6-12 hours for full walk-forward

---

## Intended Use Cases

✅ **Good For:**

- Macro trend following (months to years)
- Low-frequency trading (days to weeks)
- Portfolio-level positioning
- Swing trading crypto/equities
- Capturing major regime changes

❌ **Not Suitable For:**

- High-frequency trading (seconds to minutes)
- Scalping strategies
- Market making
- Mean reversion in ranging markets
- Ultra-short timeframes

---

## Future Enhancements (Next Research Cycle)

- **Crypto Microstructure Signals:** Integrate funding rate carry, perpetual basis spreads, and stablecoin issuance as regime qualifiers once the core price engine is proven.
- **Macro Liquidity Overlays:** Test DXY momentum, US 10y yield trends, and global liquidity composites as filters for throttling risk-on exposure.
- **Cross-Sectional Consensus:** Evaluate whether ETH/SOL confirmation (e.g., dual bullish regimes) improves BTC entry quality.
- **Execution Layer:** Prototype trade slicing and VWAP slippage modelling specific to Fidelity's venue before allocating significant capital.

These are intentionally deferred until the validation pipeline (anchored walk-forward + bootstrap stress) shows stable behaviour with pure price data.

---

## Summary

Hermes takes the conceptual insight from Giovanni Santostasi's Power Law Indicator — comparing expected vs. observed behavior — and generalizes it into a robust, asset-agnostic trend-following system using state-of-the-art ALMA filtering.

**Core Innovation:** Replace theoretical power law curves with data-adaptive Gaussian smoothing that works universally across all assets and timeframes.

**Result:** Clean, smooth signals with minimal lag, natural outlier resistance, and intuitive crossover logic suitable for systematic trading.
