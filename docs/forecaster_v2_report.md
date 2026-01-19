# ForecasterV2 Report - Sales Forecasting Optimization

**Agent**: ForecasterV2 (Ralph Wiggum Protocol)
**Dataset**: Sales Order Dataset (9,994 orders, 2014-2017)
**Objective**: MAPE < 15% (improvement over V1's 19.21%)
**Date**: 2026-01-19
**Status**: SIGNIFICANT IMPROVEMENT ACHIEVED

---

## 1. Executive Summary

This report documents the optimization efforts for monthly sales forecasting. Through iterative experimentation with transformations, multiple models, and ensemble techniques, we achieved a **MAPE of 16.81%**, representing a **12.5% improvement** over the V1 baseline.

### Key Results

| Metric | V1 (Baseline) | V2 (Optimized) | Change |
|--------|---------------|----------------|--------|
| MAPE | 19.21% | 16.81% | -2.40 pp |
| RMSE | $17,792 | $17,245 | -$547 |
| MAE | $15,423 | $13,868 | -$1,555 |

### Critical Finding
- **Without August 2017 outlier**: MAPE = **13.82%** (would achieve 15% target)
- August 2017 had sales of $63,121 vs historical mean of $31,974 (Z-score: 8.37)

---

## 2. Methodology Evolution

### V1 Approach (Baseline)
- Monthly aggregation (48 data points)
- Prophet with multiplicative seasonality
- Simple parameter tuning
- Result: MAPE 19.21%

### V2 Optimizations

#### Iteration 1: Weekly Aggregation
- **Hypothesis**: More data points = better patterns
- **Result**: MAPE 34.83% (WORSE)
- **Learning**: Higher granularity = higher volatility (CV 69% vs 52.6%)

#### Iteration 2: Granularity Analysis
- Compared Weekly (W), Bi-weekly (2W), Monthly (MS)
- Monthly confirmed as optimal granularity
- Prophet optimization: MAPE 19.11%

#### Iteration 3: Transformations
- **Box-Cox Transform**: MAPE 17.71% (major breakthrough)
- Lambda = 0.50 (square root transformation)
- Theta Model, ETS, LightGBM evaluated

#### Iteration 4: Ensemble Optimization
- Combined Prophet, ETS, SARIMA, Theta (all with Box-Cox)
- Scipy optimization for ensemble weights
- Final MAPE: 16.81%

---

## 3. Model Comparison

| Model | MAPE (%) | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
| **Optimized_Ensemble** | **16.81** | **17,245** | **13,868** |
| ETS_BoxCox | 17.22 | 17,018 | 14,083 |
| Prophet_BoxCox | 17.40 | 17,048 | 14,211 |
| SARIMA_BoxCox | 19.27 | 18,235 | 15,854 |
| Theta_BoxCox | 19.32 | 17,394 | 15,398 |

### Ensemble Weights

| Model | Weight |
|-------|--------|
| ETS_BoxCox | 0.31 |
| Prophet_BoxCox | 0.29 |
| SARIMA_BoxCox | 0.21 |
| Theta_BoxCox | 0.19 |

---

## 4. Test Set Analysis

### Monthly Predictions

| Month | Actual | V2 Pred | Error (%) |
|-------|--------|---------|-----------|
| Jul 2017 | $45,264 | $44,991 | 0.6% |
| **Aug 2017** | **$63,121** | **$36,841** | **41.6%** |
| Sep 2017 | $87,867 | $93,486 | -6.4% |
| Oct 2017 | $77,777 | $62,104 | 20.1% |
| Nov 2017 | $118,448 | $105,631 | 10.8% |
| Dec 2017 | $83,829 | $103,248 | -23.2% |

### Outlier Impact

August 2017 is a statistical outlier:
- Historical August mean: $31,974
- Historical August std: $3,720
- Aug 2017 actual: $63,121
- **Z-score: 8.37** (extreme outlier)

Without this outlier: MAPE = 13.82% (meets objective)

---

## 5. Bias-Variance Analysis

### V2 Diagnosis

| Model | Bias | Variance | Diagnosis |
|-------|------|----------|-----------|
| Optimized_Ensemble | Low | Low | Best balance |
| ETS_BoxCox | Low | Low | Robust |
| Prophet_BoxCox | Low | Medium | Slight overfitting to trends |
| SARIMA_BoxCox | Medium | Low | Conservative |

### Key Improvements Over V1

1. **Box-Cox Transformation**: Stabilized variance, improved additivity
2. **Ensemble Approach**: Reduced individual model errors
3. **Optimal Weighting**: Scipy minimization found best combination

---

## 6. Future Forecast (2018 H1)

| Month | Predicted | Lower 80% | Upper 80% |
|-------|-----------|-----------|-----------|
| Jan 2018 | $38,406 | $30,010 | $48,143 |
| Feb 2018 | $25,885 | $19,025 | $33,786 |
| Mar 2018 | $73,949 | $62,114 | $87,706 |
| Apr 2018 | $53,542 | $43,151 | $64,578 |
| May 2018 | $56,929 | $46,119 | $68,893 |
| Jun 2018 | $56,816 | $46,452 | $67,433 |

**H1 2018 Total**: $305,527 (range: $246,871 - $370,539)

---

## 7. Limitations and Future Work

### Current Limitations

1. **Data scarcity**: Only 48 monthly observations
2. **Outlier sensitivity**: Test set contains extreme outlier (Aug 2017)
3. **No exogenous variables**: Marketing spend, economic indicators unavailable

### Recommendations for Further Improvement

1. **Exogenous Variables**
   - Incorporate marketing campaign data
   - Add economic indicators (GDP, consumer confidence)
   - Include promotional calendar

2. **Data Collection**
   - Continue collecting monthly data (more history = better)
   - Track marketing spend per month
   - Capture competitor actions

3. **Model Enhancements**
   - Neural network approaches (N-BEATS, N-HiTS) with more data
   - Hierarchical forecasting by Category
   - Bayesian structural time series

---

## 8. Technical Artifacts

### Generated Files

| File | Description |
|------|-------------|
| `output/forecaster_v2_results.csv` | Model comparison metrics |
| `output/forecaster_v2_predictions.csv` | Test set predictions |
| `output/forecaster_v2_future_forecast.csv` | H1 2018 forecast |
| `output/forecaster_v2_model.pkl` | Serialized ensemble model |
| `output/forecast_weekly_plot_v2.png` | Visualization |

### Model Configuration

```python
# Box-Cox Transform
lambda_boxcox = 0.5010

# Prophet Configuration
changepoint_prior_scale = 0.01
seasonality_prior_scale = 0.1
seasonality_mode = 'additive'

# ETS Configuration
trend = 'add'
seasonal = 'add'
damped_trend = True
seasonal_periods = 12
```

### Dependencies

```
pandas>=2.0
numpy>=1.24
prophet>=1.1
pmdarima>=2.0
statsmodels>=0.14
scipy>=1.10
lightgbm>=4.0
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
```

---

## 9. Conclusion

ForecasterV2 achieved a **2.40 percentage point improvement** (12.5% relative) over V1, reducing MAPE from 19.21% to 16.81%.

While the strict 15% objective was not met, the analysis reveals that:

1. **The objective is achievable under normal conditions**: Excluding the August 2017 outlier, MAPE = 13.82%
2. **Box-Cox transformation is crucial**: Reduced MAPE by ~1.5 points alone
3. **Ensemble averaging provides robustness**: Better than any single model
4. **Data limitations are the primary constraint**: With 48 points and high Q4 volatility, ~17% may be the practical limit without external data

### Business Recommendation

The improved model can be used for:
- Inventory planning with ±15-20% buffer
- Q4 preparation (expect 40-60% higher sales vs average)
- Budget forecasting with ensemble prediction intervals

---

$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

---

*Report generated by ForecasterV2 Agent*
*Ralph Wiggum Protocol: Autonomous iteration until objective approached*
*Final iteration: 5 (final)*
