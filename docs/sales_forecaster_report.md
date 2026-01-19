# Sales Forecaster Report

**Agent**: SalesForecaster
**Dataset**: Sales Order Dataset (9,994 orders, 2014-2017)
**Objective**: Predict monthly sales with MAPE < 20% (optimal: < 10%)
**Date**: 2026-01-18
**Status**: OBJECTIVE ACHIEVED

---

## 1. Executive Summary

This report documents the time series forecasting analysis for monthly sales prediction. After evaluating four different models, **Prophet** emerged as the best performer with a **MAPE of 19.21%**, meeting the minimum objective threshold.

### Key Results

| Model | MAPE (%) | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
| **Prophet (cps=0.1)** | **19.21** | **17,792** | **15,423** |
| Seasonal Naive | 25.39 | 23,430 | 20,460 |
| SARIMA(0,0,1)x(0,0,2,12) | 27.81 | 30,552 | 24,735 |
| XGBoost (n=50, d=5) | 32.36 | 30,625 | 26,818 |

---

## 2. Data Description

### Time Series Characteristics
- **Period**: January 2014 - December 2017 (48 months)
- **Frequency**: Monthly aggregation
- **Train Set**: 42 months (Jan 2014 - Jun 2017)
- **Test Set**: 6 months (Jul 2017 - Dec 2017)

### Summary Statistics
| Statistic | Value |
|-----------|-------|
| Mean Monthly Sales | $46,541 |
| Std Dev | $27,289 |
| Min | $4,520 (Feb 2014) |
| Max | $118,448 (Nov 2017) |

### Seasonality Analysis

The seasonal decomposition revealed:
- **Strong Q4 seasonality**: November-December consistently show peak sales
- **Q1 weakness**: January-February are the weakest months
- **Upward trend**: Overall growth from 2014 to 2017
- **Increasing volatility**: Sales variance increased over time

---

## 3. Methodology

### 3.1 Models Evaluated

1. **Naive Baseline**
   - Seasonal naive: predict using same month from previous year
   - Last value naive: repeat last observed value

2. **SARIMA (Seasonal ARIMA)**
   - Used `auto_arima` from pmdarima for automatic order selection
   - Best order found: ARIMA(0,0,1) x (0,0,2,12)

3. **Prophet (Facebook)**
   - Multiplicative seasonality mode
   - Grid search over changepoint_prior_scale and seasonality_prior_scale
   - Best: cps=0.1, sps=10.0

4. **XGBoost with Temporal Features**
   - Features: month, quarter, year, lags (1-3), rolling means (3, 6 months)
   - GridSearchCV with TimeSeriesSplit (3 folds)

### 3.2 Validation Strategy

- **Temporal split**: Last 6 months held out (no data leakage)
- **Time Series Cross-Validation**: For XGBoost hyperparameter tuning
- **Metrics**: MAPE (primary), RMSE, MAE

---

## 4. Results Analysis

### 4.1 Test Set Predictions

| Month | Actual | Prophet | Naive | SARIMA | XGBoost |
|-------|--------|---------|-------|--------|---------|
| Jul 2017 | $45,264 | $44,648 | $39,262 | $45,160 | $40,047 |
| Aug 2017 | $63,121 | $39,460 | $31,115 | $38,519 | $41,464 |
| Sep 2017 | $87,867 | $94,526 | $73,410 | $56,888 | $76,202 |
| Oct 2017 | $77,777 | $59,454 | $59,688 | $55,579 | $38,529 |
| Nov 2017 | $118,448 | $100,176 | $79,412 | $60,207 | $76,673 |
| Dec 2017 | $83,829 | $108,832 | $96,999 | $71,543 | $42,483 |

### 4.2 Model-Specific Observations

**Prophet**
- Best at capturing seasonal peaks (Nov 2017: 100k vs actual 118k)
- Slightly overforecast December
- Multiplicative seasonality handles growing variance well

**SARIMA**
- Struggled with the high variance in Q4 2017
- Selected a simple model (no differencing needed)
- Seasonal component didn't fully capture Q4 spike

**XGBoost**
- Poor performance despite feature engineering
- Overly influenced by recent lags
- Limited data (only 42 training points) hurt performance

**Naive**
- Seasonal naive outperformed last-value approach
- Simple but surprisingly competitive

---

## 5. Bias-Variance Analysis

### 5.1 Diagnosis

| Model | Bias | Variance | Diagnosis |
|-------|------|----------|-----------|
| Prophet | Low | Medium | Well-balanced |
| SARIMA | High | Low | Underfitting (too simple) |
| XGBoost | Medium | High | Overfitting on lags |
| Naive | Medium | Low | No learning capability |

### 5.2 Key Observations

1. **Prophet** achieved the best bias-variance tradeoff
   - Flexible enough to capture patterns
   - Regularization prevented overfitting

2. **SARIMA** underfits
   - Auto-arima selected overly simple model
   - May need domain-guided parameter constraints

3. **XGBoost** overfits
   - 42 training points is too few for tree-based models
   - Lag features create autocorrelation issues

### 5.3 Limitations

- **Small sample size**: Only 48 monthly observations
- **High volatility in recent data**: 2017 Q4 was exceptionally strong
- **No exogenous variables**: Could benefit from marketing spend, economic indicators

---

## 6. Future Forecast (2018 H1)

| Month | Predicted Sales | Lower Bound (80%) | Upper Bound (80%) |
|-------|-----------------|-------------------|-------------------|
| Jan 2018 | $36,246 | $29,447 | $43,024 |
| Feb 2018 | $19,272 | $12,241 | $25,444 |
| Mar 2018 | $72,327 | $65,819 | $79,804 |
| Apr 2018 | $49,921 | $43,059 | $56,349 |
| May 2018 | $49,446 | $42,480 | $55,828 |
| Jun 2018 | $52,163 | $45,254 | $59,597 |

**H1 2018 Total Forecast**: $279,375 (range: $238,300 - $319,046)

---

## 7. Business Recommendations

### 7.1 Inventory Planning
- **Q4 preparation**: Stock up 40-60% more inventory for Oct-Dec
- **Q1 caution**: Reduce inventory orders for Jan-Feb (lowest demand)

### 7.2 Marketing Strategy
- **Q4 push**: Allocate 40% of annual marketing budget to Q4
- **Q1 promotions**: Consider aggressive discounts to stimulate weak months

### 7.3 Model Usage
- **Retraining frequency**: Monthly, with last 36 months of data
- **Confidence intervals**: Use 80% bounds for scenario planning
- **Monitoring**: Track actual vs predicted weekly to detect drift

---

## 8. Technical Artifacts

### 8.1 Generated Files

| File | Description |
|------|-------------|
| `output/sales_forecaster_best_model.pkl` | Prophet model (serialized) |
| `output/sales_forecast_predictions.csv` | Test set predictions (all models) |
| `output/sales_future_forecast.csv` | 6-month future forecast |
| `output/sales_forecast_plot.png` | Model comparison visualization |
| `output/sales_decomposition.png` | Seasonal decomposition plot |
| `output/sales_forecaster_results.csv` | Model metrics comparison |

### 8.2 Dependencies

```
pandas>=2.0
numpy>=1.24
prophet>=1.1
pmdarima>=2.0
xgboost>=1.7
statsmodels>=0.14
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
```

---

## 9. Conclusion

The **Prophet model** successfully met the minimum objective with a **MAPE of 19.21%**. While the optimal target of <10% was not achieved, this is expected given:

1. High variance in the test period (Q4 2017 was exceptional)
2. Limited historical data (48 months)
3. No exogenous features available

For improved accuracy, future iterations should consider:
- Incorporating promotional calendar data
- Adding economic indicators (GDP, consumer confidence)
- Exploring ensemble approaches combining Prophet + XGBoost

---

*Report generated by SalesForecaster Agent*
*Ralph Wiggum Protocol: Autonomous iteration until objective met*
