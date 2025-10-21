# Hermes Strategy - Quick Start Guide

## 🚀 30-Second Start

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# 2. Install W&B (if not already installed)
pip install wandb

# 3. Login to W&B
wandb login

# 4. Drop your CSVs into the repo root (btc_daily.csv, eth_daily.csv, ...)

# 5. Run optimization
python optimize_hermes.py

# 6. Open W&B dashboard (URL printed in console)
```

Done! The optimizer will run for 6-12 hours and save results to CSV files.

---

## 📋 What You Need

### Required Files:

- ✅ `optimize_hermes.py` - Main optimizer with ALMA filtering
- ✅ `hermes.pine` - TradingView strategy
- ✅ `btc_daily.csv` - Core BTC data (daily candles, Unix epoch seconds)
- ✅ Optional BTC backfill:
  - `data/blx_daily.csv` (BraveNewCoin index for pre-2015)
  - `data/cme_btc_daily.csv` (CME futures settlement series)
- ✅ Optional cross-asset datasets: `eth_daily.csv`, `sol_daily.csv`
- ✅ `requirements.txt` - Python dependencies

### System Requirements:

- Python 3.8+
- 8GB+ RAM recommended
- All packages from `requirements.txt` installed
- W&B account (free at wandb.ai)

---

## 📊 Understanding the Strategy

### ALMA Filtering

Hermes uses **ALMA (Arnaud Legoux Moving Average)** - a Gaussian-weighted moving average that provides:

- Ultra-smooth curves (like Giovanni's Power Law indicator)
- Minimal lag compared to traditional moving averages
- Natural outlier resistance
- Clean crossover signals

### Key Parameters

**ALMA Settings:**

- **Short Period** (80): Fast signal line
- **Long Period** (250): Slow baseline
- **ALMA Offset** (0.90): Lag vs smoothness balance
- **ALMA Sigma** (7.5): Gaussian curve width

**Strategy Settings:**

- **Buy/Sell Lookback**: Momentum confirmation (optional)
- **Min Entry Distance**: Separation requirement (optional)
- **Volatility Targeting**: Daily vol scaled toward 2% with a 100% capital cap
- **Execution Costs**: 35 bps commission + 5 bps slippage baked into every trade

---

## 📦 Multi-Asset & Data Handling

- The optimizer auto-detects BTC, ETH, and SOL CSVs in the project root.
- Proxy files under `data/` are stitched in *before* the primary series to extend history without double counting.
- Stage results are saved per asset (`hermes_stage1_BTC.csv`, `hermes_stage2_ETH.csv`, …) plus aggregated `hermes_stage*_all_assets.csv` when multiple markets are available.

---

## ⚙️ Configuration

### Quick Tweaks

**Change CPU cores:**

```python
# Line ~350 in optimize_hermes.py
n_jobs=12,  # Use 4 for M4 Air, 12-16 for cloud
```

**Reduce iterations for testing:**

```python
# Lines ~64-67
STAGE1_CALLS = 50      # Default: 300
STAGE1_RANDOM_STARTS = 20  # Default: 100
STAGE2_CALLS = 100     # Default: 500
STAGE2_RANDOM_STARTS = 40  # Default: 150
```

**Adjust penalties:**

```python
# Lines ~75-77
PENALTY_EXCESSIVE_TRADES = 0.02   # Higher = fewer trades
PENALTY_EXTREME_GAIN = 0.05       # Higher = more stable params
PENALTY_HIGH_DRAWDOWN = 0.1       # Higher = lower drawdown
```

**Adjust volatility targeting / costs:**

```python
# MANUAL_DEFAULTS in optimize_hermes.py
"target_vol": 0.02,        # Daily volatility objective
"use_vol_scaling": True,   # Toggle volatility targeting
"commission_rate": 0.0035, # 35 bps per side (Fidelity crypto desk)
"slippage_rate": 0.0005,   # 5 bps assumed slippage
"max_allocation": 1.0      # Cap position size at 100% of equity
```

---

## 📈 Understanding Output

### Console Output:

```
═══════════════════════════════════════════════════════════════════
STAGE 1: GLOBAL SEARCH
═══════════════════════════════════════════════════════════════════

─ Iteration 1: Train 2015-01-01–2016-12-31 | Test 2017-01-01–2017-12-31
  ▶ [14:23:45] Optimizing on 2015-01-01–2016-12-31 (730 bars, 300 calls)
    [14:23:52] ✓ Call  10/300 Score: 2.34
    [14:24:15] ✓ Call  50/300 Score: 2.67
    [14:25:42] ✓ Call 100/300 Score: 2.89
    ...
  ✔ [14:45:12] Done in 0:21:27 | Best Score  3.12
     Periods: short=85, long=245
     ALMA: offset=0.88, sigma=7.2
     Other: adapt=142, lookback=(18,8), minDist=0.0
```

### CSV Exports:

- `hermes_stage1_<ASSET>.csv` – Stage 1 walk-forward results per asset
- `hermes_stage2_<ASSET>.csv` – Stage 2 refinement per asset
- `hermes_stage1_all_assets.csv` / `hermes_stage2_all_assets.csv` – only created when ≥2 assets run in one sweep

**Key Columns:**

- `Short Period`, `Long Period`, `ALMA Offset`, `ALMA Sigma`
- `Buy Lookback`, `Sell Lookback`, `Baseline Momentum`, `Macro EMA Period`
- `train_composite`, `train_sortino`, `test_composite`, `test_sortino`, `test_calmar`
- `test_return`, `test_sharpe`, `test_max_dd`, `num_trades`, `win_rate`
- `bootstrap_return_p05`, `bootstrap_return_p50`, `bootstrap_max_dd_p95`

### W&B Dashboard:

Open the URL printed in console. Key visualizations:

1. **Parameter Stability** (`params/*`)

   - How Short Period, Long Period, etc. evolve
   - Stable = good (CoV < 0.5)

2. **Performance** (`metrics/*`)

   - Train vs test Sortino
   - Score degradation (overfitting indicator)
   - Consistency across windows

3. **Summary Stats**
   - Average test Sortino
   - Parameter coefficient of variation
   - Overall robustness metrics

---

## 🎯 Typical Workflow

### 1. Stage 1: Global Search

**Duration:** 2-4 hours (300 calls × anchored bull windows per asset)

**Goal:** Locate robust parameter basins under bull-only, purged walk-forward splits

**Search Ranges:**

- Short Period: 40-120 days
- Long Period: 150-350 days
- ALMA Offset: 0.80-0.95
- ALMA Sigma: 5.0-9.0

**Output:** Top 5 windows averaged to center Stage 2

### 2. Stage 2: Focused Search

**Duration:** 4-8 hours (500 calls × anchored windows)

**Goal:** Fine-tune around Stage 1 regimes while retesting on unseen bull segments

**Search Ranges:**

- Narrowed by ±20 around Stage 1 averages

**Output:** Stable, well-tested final parameters

### 3. Analysis

- Open W&B dashboard
- Run `python post_run_checks.py` for degradation & bootstrap stress checks
- Check parameter stability (low CoV = good)
- Verify train/test consistency (degradation < 0.5)
- Ensure no bounds camping (params not at edges)
- Look for consistent test Sortino across windows

### 4. Transfer to TradingView

- Pick best window (highest test Sortino)
- Copy parameter values from CSV
- Paste into `hermes.pine` inputs
- Backtest in TradingView to verify
- Paper trade before going live!

---

## ✅ Post-Run Verification

```bash
python post_run_checks.py
```

- Scans every `hermes_stage*.csv` file for degradation and stress metrics
- Flags windows where test Sortino collapses (<0.3× training) or drawdowns exceed tolerance
- A quick triage before promoting settings into execution or TradingView

---

## 🔍 Key Metrics to Watch

### ✅ Good Strategy Signs:

```
✓ Test Sortino: 1.5-3.0 (consistent across windows)
✓ Score degradation: < 0.5 (minimal overfitting)
✓ Param CoV: < 0.5 (stable parameters)
✓ Penalties: < 0.2 (normal behavior)
✓ ALMA parameters: in middle of search range (not at bounds)
✓ Win rate: 40-60% (reasonable)
✓ Trades/year: 10-50 (not excessive)
```

### ❌ Red Flags:

```
✗ Test Sortino: huge variance across windows (unstable)
✗ Score degradation: > 1.0 (severe overfitting)
✗ Param CoV: > 1.0 (parameters jumping around)
✗ Penalties: > 0.5 (pathological behavior)
✗ ALMA parameters: consistently at bounds (search range too narrow)
✗ Win rate: < 30% or > 80% (suspicious)
✗ Trades/year: > 200 (likely overtrading)
```

---

## 🐛 Troubleshooting

### "wandb: ERROR Unable to import wandb"

```bash
pip install wandb
```

### "wandb: ERROR Login required"

```bash
wandb login
# Paste your API key from wandb.ai/authorize
```

### "btc_daily.csv not found"

Make sure CSV is in same folder as script with columns: `time`, `close`, `low`

### Optimizer hits parameter bounds repeatedly

Expand search ranges in `STAGE1_SPACE` (lines ~45-63)

### Takes too long

- Reduce `STAGE1_CALLS` and `STAGE2_CALLS`
- Use cloud compute (5-7× faster than laptop)
- Increase `n_jobs` if you have more cores

### Results seem overfit

- Increase penalty coefficients (lines ~75-77)
- Add more walk-forward windows (increase `TEST_MONTHS`)
- Use longer training periods

### Poor performance on recent data

- Strategy may be regime-dependent
- Consider different assets or timeframes
- Check if market structure has changed

---

## 📚 Additional Documentation

**Core Docs:**

- `QUICKSTART.md` (this file) - Get running fast
- `context.md` - Technical details and theory
- `WANDB_SETUP.md` - W&B integration guide

**Code:**

- `hermes.pine` - TradingView strategy (Pine Script v5)
- `optimize_hermes.py` - Python optimizer with ALMA

---

## ☁️ Cloud Deployment (Optional)

For faster optimization, use cloud computing:

### RunPod / Vast.ai Setup:

```bash
# 1. SSH into instance
ssh -p PORT user@INSTANCE_IP

# 2. Upload files
scp -P PORT *.py *.csv requirements.txt user@INSTANCE_IP:~/

# 3. Setup environment
pip install -r requirements.txt
wandb login  # paste API key

# 4. Run optimization in background
nohup python optimize_hermes.py > run.log 2>&1 &

# 5. Monitor progress
tail -f run.log

# 6. Download results when done
scp -P PORT user@INSTANCE_IP:~/hermes_*.csv .
```

**Recommended Specs:**

- 12-16 vCPU
- 32GB RAM
- Any OS (Linux preferred)
- $0.30-0.50/hour typical

---

## 💡 Pro Tips

1. **Run Stage 1 first, analyze results, then decide if Stage 2 is needed**
   - If Stage 1 shows unstable params, adjust penalties first
2. **Use W&B runs to compare different configurations**

   - Try different penalty settings
   - Compare performance across assets

3. **Export CSVs and analyze in Excel/Python**

   - Create custom visualizations
   - Find optimal parameter regions

4. **Save promising parameter sets**

   - Document what works for different market regimes
   - Build a library of configurations

5. **Test on multiple assets before committing**

   - What works on BTC may not work on ETH
   - Diversify to reduce strategy risk

6. **Paper trade extensively before live trading**
   - Verify execution matches backtest
   - Check slippage and fees impact

---

## 🎓 Understanding ALMA

### Why ALMA over Traditional MAs?

| Feature            | SMA       | EMA    | ALMA           |
| ------------------ | --------- | ------ | -------------- |
| Smoothness         | ⭐⭐      | ⭐⭐⭐ | ⭐⭐⭐⭐⭐     |
| Lag                | ⭐ (high) | ⭐⭐   | ⭐⭐⭐⭐ (low) |
| Outlier Resistance | ⭐⭐      | ⭐⭐   | ⭐⭐⭐⭐       |
| Whipsaw Reduction  | ⭐⭐      | ⭐⭐⭐ | ⭐⭐⭐⭐⭐     |

### Tuning ALMA:

**For more responsiveness:**

- Decrease Short Period (80 → 60)
- Increase ALMA Offset (0.90 → 0.93)

**For more smoothness:**

- Increase ALMA Sigma (7.5 → 8.5)
- Increase Long Period (250 → 300)

**For more trades:**

- Decrease both periods
- Disable distance filter

**For fewer trades:**

- Increase both periods
- Enable distance filter

---

## 🚦 Next Steps

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Login to W&B:** `wandb login`
3. **Run optimizer:** `python optimize_hermes.py`
4. **Monitor W&B dashboard:** Check URL in console
5. **Analyze results:** Review CSV files and dashboard
6. **Transfer to TradingView:** Copy best parameters
7. **Paper trade:** Test before live deployment

---

**Ready to optimize?**

Run `python optimize_hermes.py` and let it cook! 🍳

The optimizer will run walk-forward analysis, find optimal ALMA parameters, and log everything to Weights & Biases for easy analysis.

**Questions?** Check `context.md` for technical details or `WANDB_SETUP.md` for dashboard help.

Good luck! 🚀
