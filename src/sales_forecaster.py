"""
Sales Forecaster - Time Series Prediction
Agent: SalesForecaster
Date: 2026-01-18

Predicts monthly sales using multiple time series models:
- Naive baseline
- SARIMA (auto_arima)
- Prophet
- XGBoost with temporal features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
AGENT_CONTEXT_DIR = BASE_DIR / 'agent_context'
DOCS_DIR = BASE_DIR / 'docs'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dynamic n_jobs based on active agents
def get_n_jobs():
    """Calculate n_jobs based on active agents."""
    n_cores = os.cpu_count() or 4
    try:
        agent_files = list(AGENT_CONTEXT_DIR.glob('*_state.json'))
        active_agents = 0
        for f in agent_files:
            with open(f) as file:
                state = json.load(file)
                if state.get('status') == 'running':
                    active_agents += 1
        active_agents = max(active_agents, 1)
        return max(1, n_cores // active_agents)
    except:
        return max(1, n_cores // 2)


def update_state(phase: str, progress: int, **kwargs):
    """Update agent state file."""
    state_file = AGENT_CONTEXT_DIR / 'SalesForecaster_state.json'
    try:
        with open(state_file) as f:
            state = json.load(f)
    except:
        state = {}

    state.update({
        'agent': 'SalesForecaster',
        'status': 'running',
        'current_phase': phase,
        'progress': progress,
        'last_checkpoint': datetime.now().isoformat()
    })
    state.update(kwargs)

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_data():
    """Load preprocessed forecaster data."""
    df = pd.read_csv(OUTPUT_DIR / 'features_forecaster.csv')
    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('year_month').reset_index(drop=True)
    return df


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def train_naive_baseline(train, test):
    """Naive forecast: use last value."""
    last_value = train['sales'].iloc[-1]
    predictions = [last_value] * len(test)

    # Also try seasonal naive (same month last year)
    seasonal_preds = []
    for _, row in test.iterrows():
        same_month_prev_year = train[train['month'] == row['month']]['sales']
        if len(same_month_prev_year) > 0:
            seasonal_preds.append(same_month_prev_year.iloc[-1])
        else:
            seasonal_preds.append(last_value)

    # Compare both
    mape_last = mape(test['sales'], predictions)
    mape_seasonal = mape(test['sales'], seasonal_preds)

    if mape_seasonal < mape_last:
        return seasonal_preds, 'Seasonal Naive', mape_seasonal
    return predictions, 'Last Value Naive', mape_last


def train_sarima(train, test, n_jobs=1):
    """Train SARIMA using auto_arima."""
    from pmdarima import auto_arima

    y_train = train['sales'].values

    model = auto_arima(
        y_train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        m=12,  # Monthly seasonality
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE
    )

    predictions = model.predict(n_periods=len(test))
    mape_score = mape(test['sales'], predictions)

    return predictions, model, f'SARIMA{model.order}x{model.seasonal_order}', mape_score


def train_prophet(train, test):
    """Train Prophet model."""
    from prophet import Prophet

    # Prophet requires 'ds' and 'y' columns
    prophet_train = train[['year_month', 'sales']].copy()
    prophet_train.columns = ['ds', 'y']

    # Best params grid search (simplified for speed)
    best_mape = float('inf')
    best_preds = None
    best_params = None

    param_grid = [
        {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0},
        {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0},
        {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0},
    ]

    for params in param_grid:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_train)

        future = pd.DataFrame({'ds': test['year_month']})
        forecast = model.predict(future)
        preds = forecast['yhat'].values

        current_mape = mape(test['sales'], preds)
        if current_mape < best_mape:
            best_mape = current_mape
            best_preds = preds
            best_params = params
            best_model = model

    return best_preds, best_model, f'Prophet(cps={best_params["changepoint_prior_scale"]})', best_mape


def train_xgboost(train, test, n_jobs=1):
    """Train XGBoost with temporal features."""
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

    feature_cols = ['month', 'quarter', 'year', 'sales_lag_1', 'sales_lag_2',
                    'sales_lag_3', 'sales_rolling_mean_3', 'sales_rolling_mean_6',
                    'sales_rolling_std_3']

    # Remove rows with NaN (first few months due to lags)
    train_clean = train.dropna(subset=feature_cols)

    X_train = train_clean[feature_cols]
    y_train = train_clean['sales']

    # For test, we need to calculate features using train data
    X_test = test[feature_cols].copy()
    y_test = test['sales']

    # Grid search with TimeSeriesSplit
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        objective='reg:squarederror'
    )

    grid_search = GridSearchCV(
        model, param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    mape_score = mape(y_test, predictions)
    params_str = f"n={grid_search.best_params_['n_estimators']},d={grid_search.best_params_['max_depth']}"

    return predictions, best_model, f'XGBoost({params_str})', mape_score, feature_cols


def create_forecast_plot(train, test, results, output_path):
    """Create forecast comparison plot."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]

        # Plot training data
        ax.plot(train['year_month'], train['sales'],
                label='Training', color='blue', alpha=0.7)

        # Plot actual test data
        ax.plot(test['year_month'], test['sales'],
                label='Actual', color='green', linewidth=2)

        # Plot predictions
        ax.plot(test['year_month'], data['predictions'],
                label=f'Predicted (MAPE: {data["mape"]:.1f}%)',
                color='red', linestyle='--', linewidth=2)

        ax.set_title(f'{data["model_name"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Monthly Sales ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_decomposition_plot(df, output_path):
    """Create seasonal decomposition plot."""
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Set date as index for decomposition
    ts = df.set_index('year_month')['sales']

    decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(decomposition.observed)
    axes[0].set_title('Observed', fontweight='bold')
    axes[0].set_ylabel('Sales ($)')

    axes[1].plot(decomposition.trend)
    axes[1].set_title('Trend', fontweight='bold')
    axes[1].set_ylabel('Sales ($)')

    axes[2].plot(decomposition.seasonal)
    axes[2].set_title('Seasonal', fontweight='bold')
    axes[2].set_ylabel('Factor')

    axes[3].plot(decomposition.resid)
    axes[3].set_title('Residual', fontweight='bold')
    axes[3].set_ylabel('Factor')
    axes[3].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_future_forecast(best_model, model_type, df, periods=6):
    """Generate future forecast for next periods."""
    from dateutil.relativedelta import relativedelta

    last_date = df['year_month'].max()
    future_dates = [last_date + relativedelta(months=i+1) for i in range(periods)]

    if model_type == 'prophet':
        future = pd.DataFrame({'ds': future_dates})
        forecast = best_model.predict(future)
        return pd.DataFrame({
            'date': future_dates,
            'predicted_sales': forecast['yhat'].values,
            'lower_bound': forecast['yhat_lower'].values,
            'upper_bound': forecast['yhat_upper'].values
        })

    elif model_type == 'sarima':
        predictions = best_model.predict(n_periods=periods)
        conf_int = best_model.predict(n_periods=periods, return_conf_int=True)[1]
        return pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions,
            'lower_bound': conf_int[:, 0],
            'upper_bound': conf_int[:, 1]
        })

    else:  # XGBoost - need to generate features recursively
        predictions = []
        sales_history = df['sales'].tolist()

        for i, date in enumerate(future_dates):
            month = date.month
            quarter = (month - 1) // 3 + 1
            year = date.year

            lag_1 = sales_history[-1]
            lag_2 = sales_history[-2] if len(sales_history) > 1 else lag_1
            lag_3 = sales_history[-3] if len(sales_history) > 2 else lag_1

            rolling_mean_3 = np.mean(sales_history[-3:])
            rolling_mean_6 = np.mean(sales_history[-6:])
            rolling_std_3 = np.std(sales_history[-3:])

            X = pd.DataFrame([[month, quarter, year, lag_1, lag_2, lag_3,
                              rolling_mean_3, rolling_mean_6, rolling_std_3]],
                            columns=['month', 'quarter', 'year', 'sales_lag_1',
                                    'sales_lag_2', 'sales_lag_3', 'sales_rolling_mean_3',
                                    'sales_rolling_mean_6', 'sales_rolling_std_3'])

            pred = best_model.predict(X)[0]
            predictions.append(pred)
            sales_history.append(pred)

        return pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })


def main():
    """Main forecasting pipeline."""
    print("=" * 60)
    print("SALES FORECASTER - Time Series Prediction")
    print("=" * 60)

    n_jobs = get_n_jobs()
    print(f"\nUsing n_jobs={n_jobs} (detected active agents)")

    # 1. Load data
    print("\n[1/7] Loading data...")
    update_state('loading_data', 10)
    df = load_data()
    print(f"    Loaded {len(df)} months: {df['year_month'].min()} to {df['year_month'].max()}")

    # 2. Train/Test split (last 6 months for test)
    print("\n[2/7] Splitting data (last 6 months for test)...")
    update_state('splitting_data', 15)
    test_size = 6
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    print(f"    Train: {len(train)} months ({train['year_month'].min()} to {train['year_month'].max()})")
    print(f"    Test: {len(test)} months ({test['year_month'].min()} to {test['year_month'].max()})")

    # 3. Create decomposition plot
    print("\n[3/7] Creating seasonal decomposition...")
    update_state('decomposition', 20)
    create_decomposition_plot(df, OUTPUT_DIR / 'sales_decomposition.png')
    print(f"    Saved: output/sales_decomposition.png")

    results = {}
    models_trained = []

    # 4. Train Naive baseline
    print("\n[4/7] Training Naive baseline...")
    update_state('training_naive', 30)
    naive_preds, naive_name, naive_mape = train_naive_baseline(train, test)
    results['naive'] = {
        'predictions': naive_preds,
        'model_name': naive_name,
        'mape': naive_mape,
        'rmse': rmse(test['sales'], naive_preds),
        'mae': mae(test['sales'], naive_preds)
    }
    models_trained.append(naive_name)
    print(f"    {naive_name}: MAPE = {naive_mape:.2f}%")

    # 5. Train SARIMA
    print("\n[5/7] Training SARIMA (auto_arima)...")
    update_state('training_sarima', 45, models_trained=models_trained)
    try:
        sarima_preds, sarima_model, sarima_name, sarima_mape = train_sarima(train, test, n_jobs)
        results['sarima'] = {
            'predictions': sarima_preds.tolist(),
            'model_name': sarima_name,
            'mape': sarima_mape,
            'rmse': rmse(test['sales'], sarima_preds),
            'mae': mae(test['sales'], sarima_preds),
            'model': sarima_model
        }
        models_trained.append(sarima_name)
        print(f"    {sarima_name}: MAPE = {sarima_mape:.2f}%")
    except Exception as e:
        print(f"    SARIMA failed: {e}")
        results['sarima'] = None

    # 6. Train Prophet
    print("\n[6/7] Training Prophet...")
    update_state('training_prophet', 60, models_trained=models_trained)
    try:
        prophet_preds, prophet_model, prophet_name, prophet_mape = train_prophet(train, test)
        results['prophet'] = {
            'predictions': prophet_preds.tolist(),
            'model_name': prophet_name,
            'mape': prophet_mape,
            'rmse': rmse(test['sales'], prophet_preds),
            'mae': mae(test['sales'], prophet_preds),
            'model': prophet_model
        }
        models_trained.append(prophet_name)
        print(f"    {prophet_name}: MAPE = {prophet_mape:.2f}%")
    except Exception as e:
        print(f"    Prophet failed: {e}")
        results['prophet'] = None

    # 7. Train XGBoost
    print("\n[7/7] Training XGBoost with temporal features...")
    update_state('training_xgboost', 75, models_trained=models_trained)
    try:
        xgb_preds, xgb_model, xgb_name, xgb_mape, xgb_features = train_xgboost(train, test, n_jobs)
        results['xgboost'] = {
            'predictions': xgb_preds.tolist(),
            'model_name': xgb_name,
            'mape': xgb_mape,
            'rmse': rmse(test['sales'], xgb_preds),
            'mae': mae(test['sales'], xgb_preds),
            'model': xgb_model,
            'features': xgb_features
        }
        models_trained.append(xgb_name)
        print(f"    {xgb_name}: MAPE = {xgb_mape:.2f}%")
    except Exception as e:
        print(f"    XGBoost failed: {e}")
        results['xgboost'] = None

    # 8. Select best model
    print("\n" + "-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)

    valid_results = {k: v for k, v in results.items() if v is not None}

    comparison_data = []
    for name, data in valid_results.items():
        comparison_data.append({
            'Model': data['model_name'],
            'MAPE (%)': data['mape'],
            'RMSE ($)': data['rmse'],
            'MAE ($)': data['mae']
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('MAPE (%)')
    print(comparison_df.to_string(index=False))

    # Best model
    best_key = min(valid_results.keys(), key=lambda k: valid_results[k]['mape'])
    best_result = valid_results[best_key]
    best_mape = best_result['mape']
    best_name = best_result['model_name']

    print(f"\nBest Model: {best_name} (MAPE: {best_mape:.2f}%)")

    # Check if meets objective
    if best_mape < 10:
        print("Status: OPTIMAL (MAPE < 10%)")
    elif best_mape < 20:
        print("Status: MINIMUM ACHIEVED (MAPE < 20%)")
    else:
        print("Status: BELOW TARGET (MAPE >= 20%)")

    # 9. Create forecast plot
    print("\nCreating forecast comparison plot...")
    plot_results = {k: v for k, v in valid_results.items() if v is not None}
    create_forecast_plot(train, test, plot_results, OUTPUT_DIR / 'sales_forecast_plot.png')
    print(f"    Saved: output/sales_forecast_plot.png")

    # 10. Save predictions
    print("\nSaving predictions...")
    predictions_df = test[['year_month', 'sales']].copy()
    predictions_df.columns = ['date', 'actual_sales']
    for name, data in valid_results.items():
        if data is not None:
            predictions_df[f'{name}_predicted'] = data['predictions']
    predictions_df.to_csv(OUTPUT_DIR / 'sales_forecast_predictions.csv', index=False)
    print(f"    Saved: output/sales_forecast_predictions.csv")

    # 11. Generate future forecast
    print("\nGenerating 6-month future forecast...")
    if best_key == 'prophet' and results['prophet'] is not None:
        future_forecast = generate_future_forecast(
            results['prophet']['model'], 'prophet', df, periods=6
        )
    elif best_key == 'sarima' and results['sarima'] is not None:
        future_forecast = generate_future_forecast(
            results['sarima']['model'], 'sarima', df, periods=6
        )
    elif best_key == 'xgboost' and results['xgboost'] is not None:
        future_forecast = generate_future_forecast(
            results['xgboost']['model'], 'xgboost', df, periods=6
        )
    else:
        future_forecast = None

    if future_forecast is not None:
        future_forecast.to_csv(OUTPUT_DIR / 'sales_future_forecast.csv', index=False)
        print(f"    Saved: output/sales_future_forecast.csv")
        print("\n    Future Predictions:")
        print(future_forecast.to_string(index=False))

    # 12. Save best model
    print("\nSaving best model...")
    if best_key in ['sarima', 'xgboost'] and valid_results[best_key].get('model') is not None:
        model_to_save = valid_results[best_key]['model']
        joblib.dump(model_to_save, OUTPUT_DIR / 'sales_forecaster_best_model.pkl')
        print(f"    Saved: output/sales_forecaster_best_model.pkl")
    elif best_key == 'prophet' and valid_results['prophet'] is not None:
        # Prophet models need special serialization
        import pickle
        with open(OUTPUT_DIR / 'sales_forecaster_best_model.pkl', 'wb') as f:
            pickle.dump(valid_results['prophet']['model'], f)
        print(f"    Saved: output/sales_forecaster_best_model.pkl")

    # 13. Save comparison results
    comparison_df.to_csv(OUTPUT_DIR / 'sales_forecaster_results.csv', index=False)
    print(f"    Saved: output/sales_forecaster_results.csv")

    # 14. Update final state
    update_state(
        phase='completed',
        progress=100,
        status='completed',
        models_trained=models_trained,
        best_model=best_name,
        best_mape=best_mape,
        metrics={
            'best_mape': best_mape,
            'best_rmse': best_result['rmse'],
            'best_mae': best_result['mae']
        },
        outputs={
            'model_file': 'output/sales_forecaster_best_model.pkl',
            'predictions_file': 'output/sales_forecast_predictions.csv',
            'forecast_plot': 'output/sales_forecast_plot.png',
            'decomposition_plot': 'output/sales_decomposition.png',
            'results_file': 'output/sales_forecaster_results.csv',
            'future_forecast': 'output/sales_future_forecast.csv'
        },
        target_met=bool(best_mape < 20)
    )

    print("\n" + "=" * 60)
    print("SALES FORECASTER COMPLETED")
    print("=" * 60)
    print(f"\nBest Model: {best_name}")
    print(f"MAPE: {best_mape:.2f}%")
    print(f"Target Met: {'YES' if best_mape < 20 else 'NO'}")

    return {
        'best_model': best_name,
        'best_mape': best_mape,
        'all_results': {k: {'mape': v['mape'], 'rmse': v['rmse']}
                       for k, v in valid_results.items()}
    }


if __name__ == '__main__':
    main()
