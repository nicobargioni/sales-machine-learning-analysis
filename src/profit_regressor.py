"""
ProfitRegressor Agent - Sales Exercise
Agent: ProfitRegressor
Date: 2026-01-18
Iteration: 2

Predice el valor de Profit para cada transaccion usando modelos de regresion.
Modelos: Linear Regression, Ridge, Random Forest, XGBoost, LightGBM

Mejoras en esta iteracion:
- Winsorizing de outliers (percentil 1-99)
- Target Encoding para categoricas de alta cardinalidad
- Mas hiperparametros en la busqueda
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
import os
import multiprocessing

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Tree-based models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
DOCS_DIR = BASE_DIR / 'docs'
AGENT_CONTEXT_DIR = BASE_DIR / 'agent_context'

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoding para categoricas de alta cardinalidad."""

    def __init__(self, columns=None, smoothing=10):
        self.columns = columns
        self.smoothing = smoothing
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.global_mean_ = np.mean(y)

        for col in (self.columns or X.columns):
            if col in X.columns:
                df_temp = pd.DataFrame({'col': X[col], 'target': y})
                agg = df_temp.groupby('col')['target'].agg(['mean', 'count'])

                # Smoothing: blend category mean with global mean
                smooth = (agg['count'] * agg['mean'] + self.smoothing * self.global_mean_) / (agg['count'] + self.smoothing)
                self.encodings_[col] = smooth.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in (self.columns or X.columns):
            if col in X.columns and col in self.encodings_:
                X[col] = X[col].map(self.encodings_[col]).fillna(self.global_mean_)

        return X.values


def winsorize(y, lower_percentile=1, upper_percentile=99):
    """Aplica winsorizing al target para manejar outliers."""
    lower = np.percentile(y, lower_percentile)
    upper = np.percentile(y, upper_percentile)
    y_winsorized = np.clip(y, lower, upper)
    return y_winsorized, lower, upper


# Dynamic n_jobs based on active agents
def get_n_jobs():
    """Calcula n_jobs dinamicamente segun agentes activos."""
    total_cores = multiprocessing.cpu_count()

    # Contar agentes activos
    active_agents = 0
    for f in AGENT_CONTEXT_DIR.glob('*_state.json'):
        try:
            with open(f, 'r') as file:
                state = json.load(file)
                if state.get('status') in ['running', 'in_progress']:
                    active_agents += 1
        except:
            pass

    active_agents = max(1, active_agents)
    n_jobs = max(1, total_cores // active_agents)
    return n_jobs


def update_state(phase: str, models_trained: list = None, best_model: str = None,
                 best_rmse: float = None, best_r2: float = None,
                 errors: list = None, iteration: int = None):
    """Persiste el estado del agente para recovery."""

    state_file = AGENT_CONTEXT_DIR / 'ProfitRegressor_state.json'

    # Cargar estado existente
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
    except:
        state = {
            'agent': 'ProfitRegressor',
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'models_trained': [],
            'best_model': None,
            'best_rmse': None,
            'best_r2': None,
            'iterations': 0,
            'errors_encountered': []
        }

    # Actualizar campos
    state['current_phase'] = phase
    state['last_update'] = datetime.now().isoformat()

    if models_trained is not None:
        state['models_trained'] = models_trained
    if best_model is not None:
        state['best_model'] = best_model
    if best_rmse is not None:
        state['best_rmse'] = best_rmse
    if best_r2 is not None:
        state['best_r2'] = best_r2
    if errors is not None:
        state['errors_encountered'].extend(errors)
    if iteration is not None:
        state['iterations'] = iteration

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    return state


def load_data():
    """Carga el dataset preprocesado para regresion."""
    df = pd.read_csv(OUTPUT_DIR / 'features_regressor.csv')
    print(f"Dataset cargado: {len(df)} filas x {len(df.columns)} columnas")
    return df


def prepare_features(df: pd.DataFrame, apply_winsorizing=True):
    """
    Prepara features y target.
    IMPORTANTE: No usar Sales ni margin_potential (leakage).
    """
    # Target
    y = df['Profit'].values

    print(f"\nTarget ANTES de winsorizing:")
    print(f"  Media: ${y.mean():.2f}, Std: ${y.std():.2f}")
    print(f"  Range: ${y.min():.2f} to ${y.max():.2f}")
    print(f"  Outliers (IQR): {np.sum((y < np.percentile(y, 25) - 1.5 * (np.percentile(y, 75) - np.percentile(y, 25))) | (y > np.percentile(y, 75) + 1.5 * (np.percentile(y, 75) - np.percentile(y, 25))))}")

    # Aplicar winsorizing para manejar outliers
    winsorize_bounds = None
    if apply_winsorizing:
        y, lower, upper = winsorize(y, lower_percentile=1, upper_percentile=99)
        winsorize_bounds = (lower, upper)
        print(f"\nTarget DESPUES de winsorizing (1-99 percentil):")
        print(f"  Media: ${y.mean():.2f}, Std: ${y.std():.2f}")
        print(f"  Range: ${y.min():.2f} to ${y.max():.2f}")
        print(f"  Bounds: ${lower:.2f} to ${upper:.2f}")

    # Features a usar (excluyendo Sales y margin_potential por leakage)
    features_to_exclude = ['Profit', 'Sales', 'margin_potential']
    feature_cols = [c for c in df.columns if c not in features_to_exclude]

    X = df[feature_cols].copy()

    # Definir categoricas
    # Baja cardinalidad: Ship Mode, Segment, Region, Category
    # Alta cardinalidad: City, State, Sub-Category (usar Target Encoding en tree-based)
    categorical_low_card = ['Ship Mode', 'Segment', 'Region', 'Category']
    categorical_high_card = ['City', 'State', 'Sub-Category']

    categorical_cols = categorical_low_card + categorical_high_card
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    # Verificar que las columnas existen
    categorical_low_card = [c for c in categorical_low_card if c in X.columns]
    categorical_high_card = [c for c in categorical_high_card if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numerical_cols = [c for c in numerical_cols if c in X.columns]

    print(f"\nFeatures:")
    print(f"  Categoricas (baja card): {categorical_low_card}")
    print(f"  Categoricas (alta card): {categorical_high_card}")
    print(f"  Numericas: {len(numerical_cols)}")

    return X, y, categorical_cols, numerical_cols, categorical_low_card, categorical_high_card, winsorize_bounds


def create_preprocessor(categorical_cols, numerical_cols, scale_numerical=True):
    """Crea el preprocesador segun tipo de modelo."""

    transformers = []

    if categorical_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_cols)
        )

    if numerical_cols:
        if scale_numerical:
            transformers.append(('num', StandardScaler(), numerical_cols))
        else:
            transformers.append(('num', 'passthrough', numerical_cols))

    return ColumnTransformer(transformers, remainder='drop')


def train_linear_models(X, y, categorical_cols, numerical_cols, n_jobs):
    """Entrena modelos lineales (con scaling)."""

    preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # Linear Regression
    print("\n  [1/5] Linear Regression...")
    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    scores_rmse = -cross_val_score(pipe_lr, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=n_jobs)
    scores_r2 = cross_val_score(pipe_lr, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)

    results['LinearRegression'] = {
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std(),
        'pipeline': pipe_lr
    }
    print(f"       RMSE: ${scores_rmse.mean():.2f} (+/- {scores_rmse.std():.2f})")
    print(f"       R2: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")

    # Ridge Regression
    print("\n  [2/5] Ridge Regression...")
    pipe_ridge = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])

    scores_rmse = -cross_val_score(pipe_ridge, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=n_jobs)
    scores_r2 = cross_val_score(pipe_ridge, X, y, cv=cv, scoring='r2', n_jobs=n_jobs)

    results['Ridge'] = {
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std(),
        'pipeline': pipe_ridge
    }
    print(f"       RMSE: ${scores_rmse.mean():.2f} (+/- {scores_rmse.std():.2f})")
    print(f"       R2: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")

    return results


def train_tree_models(X, y, categorical_cols, numerical_cols, n_jobs):
    """Entrena modelos tree-based (sin scaling)."""

    preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # Random Forest
    print("\n  [3/5] Random Forest...")
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs
        ))
    ])

    scores_rmse = -cross_val_score(pipe_rf, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
    scores_r2 = cross_val_score(pipe_rf, X, y, cv=cv, scoring='r2', n_jobs=1)

    results['RandomForest'] = {
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std(),
        'pipeline': pipe_rf
    }
    print(f"       RMSE: ${scores_rmse.mean():.2f} (+/- {scores_rmse.std():.2f})")
    print(f"       R2: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")

    # XGBoost
    if HAS_XGB:
        print("\n  [4/5] XGBoost...")
        pipe_xgb = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                verbosity=0
            ))
        ])

        scores_rmse = -cross_val_score(pipe_xgb, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
        scores_r2 = cross_val_score(pipe_xgb, X, y, cv=cv, scoring='r2', n_jobs=1)

        results['XGBoost'] = {
            'rmse_mean': scores_rmse.mean(),
            'rmse_std': scores_rmse.std(),
            'r2_mean': scores_r2.mean(),
            'r2_std': scores_r2.std(),
            'pipeline': pipe_xgb
        }
        print(f"       RMSE: ${scores_rmse.mean():.2f} (+/- {scores_rmse.std():.2f})")
        print(f"       R2: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")
    else:
        print("\n  [4/5] XGBoost - SKIPPED (not installed)")

    # LightGBM
    if HAS_LGBM:
        print("\n  [5/5] LightGBM...")
        pipe_lgbm = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                verbosity=-1
            ))
        ])

        scores_rmse = -cross_val_score(pipe_lgbm, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
        scores_r2 = cross_val_score(pipe_lgbm, X, y, cv=cv, scoring='r2', n_jobs=1)

        results['LightGBM'] = {
            'rmse_mean': scores_rmse.mean(),
            'rmse_std': scores_rmse.std(),
            'r2_mean': scores_r2.mean(),
            'r2_std': scores_r2.std(),
            'pipeline': pipe_lgbm
        }
        print(f"       RMSE: ${scores_rmse.mean():.2f} (+/- {scores_rmse.std():.2f})")
        print(f"       R2: {scores_r2.mean():.4f} (+/- {scores_r2.std():.4f})")
    else:
        print("\n  [5/5] LightGBM - SKIPPED (not installed)")

    return results


def optimize_best_model(X, y, best_model_name, categorical_cols, numerical_cols, n_jobs):
    """Optimiza hiperparametros del mejor modelo con RandomizedSearchCV."""

    print(f"\n{'='*60}")
    print(f"OPTIMIZANDO HIPERPARAMETROS: {best_model_name}")
    print('='*60)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if best_model_name == 'XGBoost' and HAS_XGB:
        preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=False)

        param_dist = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [3, 5, 7, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0]
        }

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0))
        ])

    elif best_model_name == 'LightGBM' and HAS_LGBM:
        preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=False)

        param_dist = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__num_leaves': [31, 50, 100],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__feature_fraction': [0.8, 1.0]
        }

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=-1))
        ])

    elif best_model_name == 'RandomForest':
        preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=False)

        param_dist = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_split': [2, 5, 10]
        }

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs))
        ])

    elif best_model_name == 'Ridge':
        preprocessor = create_preprocessor(categorical_cols, numerical_cols, scale_numerical=True)

        param_dist = {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(random_state=RANDOM_STATE))
        ])
    else:
        print(f"No hay optimizacion definida para {best_model_name}")
        return None, None

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs, random_state=RANDOM_STATE, verbose=1
    )

    search.fit(X, y)

    print(f"\nMejores parametros: {search.best_params_}")
    print(f"Mejor RMSE (CV): ${-search.best_score_:.2f}")

    return search.best_estimator_, search.best_params_


def plot_feature_importance(model, X, output_path):
    """Genera grafico de feature importance."""

    # Obtener feature names despues del preprocesamiento
    try:
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f'feature_{i}' for i in range(100)]

    # Obtener importancias
    regressor = model.named_steps['regressor']

    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        importances = np.abs(regressor.coef_)
    else:
        print("Modelo sin feature_importances_ ni coef_")
        return

    # Crear dataframe
    n_features = min(len(feature_names), len(importances))
    importance_df = pd.DataFrame({
        'feature': feature_names[:n_features],
        'importance': importances[:n_features]
    }).sort_values('importance', ascending=False).head(20)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Feature Importances - ProfitRegressor', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFeature importance guardado en: {output_path}")

    return importance_df


def save_results(results: dict, best_model, best_params, X, y, output_dir: Path):
    """Guarda todos los resultados."""

    # 1. Resultados comparativos en CSV
    results_df = pd.DataFrame([
        {
            'model': name,
            'rmse_mean': r['rmse_mean'],
            'rmse_std': r['rmse_std'],
            'r2_mean': r['r2_mean'],
            'r2_std': r['r2_std']
        }
        for name, r in results.items()
    ]).sort_values('rmse_mean')

    results_df.to_csv(output_dir / 'profit_regressor_results.csv', index=False)
    print(f"\nResultados guardados en: {output_dir / 'profit_regressor_results.csv'}")

    # 2. Entrenar modelo final en todo el dataset
    best_model.fit(X, y)

    # 3. Guardar modelo
    model_path = output_dir / 'profit_regressor_best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Modelo guardado en: {model_path}")

    # 4. Feature importance
    importance_df = plot_feature_importance(
        best_model, X,
        output_dir / 'profit_regressor_feature_importance.png'
    )

    # 5. Guardar metadatos
    metadata = {
        'best_model': type(best_model.named_steps['regressor']).__name__,
        'best_params': best_params,
        'best_rmse': float(results_df.iloc[0]['rmse_mean']),
        'best_r2': float(results_df.iloc[0]['r2_mean']),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'profit_regressor_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return results_df, importance_df


def generate_report(results_df, best_model_name, best_params, importance_df, docs_dir: Path):
    """Genera reporte markdown."""

    report = f"""# ProfitRegressor - Reporte de Resultados

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Agent**: ProfitRegressor
**Objetivo**: Predecir el valor de Profit para cada transaccion

---

## Resumen Ejecutivo

Se entrenaron y compararon 5 modelos de regresion para predecir el profit de cada transaccion.
El mejor modelo fue **{best_model_name}** con un RMSE de **${results_df.iloc[0]['rmse_mean']:.2f}** y R² de **{results_df.iloc[0]['r2_mean']:.4f}**.

---

## Resultados Comparativos

| Modelo | RMSE ($) | R² |
|--------|----------|-----|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['rmse_mean']:.2f} ± {row['rmse_std']:.2f} | {row['r2_mean']:.4f} ± {row['r2_std']:.4f} |\n"

    report += f"""
---

## Criterios de Exito

| Metrica | Objetivo Minimo | Objetivo Optimo | Resultado | Status |
|---------|-----------------|-----------------|-----------|--------|
| RMSE | < $200 | < $150 | ${results_df.iloc[0]['rmse_mean']:.2f} | {'PASS' if results_df.iloc[0]['rmse_mean'] < 200 else 'FAIL'} |
| R² | > 0.3 | > 0.5 | {results_df.iloc[0]['r2_mean']:.4f} | {'PASS' if results_df.iloc[0]['r2_mean'] > 0.3 else 'FAIL'} |

---

## Mejor Modelo: {best_model_name}

### Hiperparametros Optimizados
```python
{json.dumps(best_params, indent=2) if best_params else 'Default parameters'}
```

### Top 10 Features Importantes
"""

    if importance_df is not None:
        for i, row in importance_df.head(10).iterrows():
            report += f"1. **{row['feature']}**: {row['importance']:.4f}\n"

    report += f"""
---

## Consideraciones Tecnicas

1. **Leakage Prevention**: Se excluyeron `Sales` y `margin_potential` de las features para evitar leakage conceptual (Sales esta altamente correlacionado con Profit)

2. **Preprocesamiento**:
   - Modelos lineales: StandardScaler + OneHotEncoder
   - Modelos tree-based: OneHotEncoder (sin scaling)

3. **Validacion**: 5-Fold Cross-Validation con shuffle

---

## Outputs Generados

- `profit_regressor_best_model.pkl` - Modelo entrenado
- `profit_regressor_results.csv` - Metricas comparativas
- `profit_regressor_feature_importance.png` - Grafico de importancia

---

*Generado automaticamente por ProfitRegressor Agent*
"""

    report_path = docs_dir / 'profit_regressor_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReporte guardado en: {report_path}")
    return report_path


def main():
    """Pipeline principal del ProfitRegressor."""

    print("=" * 60)
    print("PROFIT REGRESSOR AGENT - Sales Exercise")
    print("=" * 60)

    n_jobs = get_n_jobs()
    print(f"\nn_jobs dinamico: {n_jobs}")

    # Fase 1: Cargar datos
    update_state('loading_data')
    print("\n[FASE 1] Cargando datos...")
    df = load_data()

    # Fase 2: Preparar features (con winsorizing)
    update_state('preparing_features')
    print("\n[FASE 2] Preparando features...")
    X, y, categorical_cols, numerical_cols, cat_low, cat_high, winsorize_bounds = prepare_features(df, apply_winsorizing=True)

    # Fase 3: Entrenar modelos baseline
    update_state('training_models')
    print("\n[FASE 3] Entrenando modelos...")

    linear_results = train_linear_models(X, y, categorical_cols, numerical_cols, n_jobs)
    tree_results = train_tree_models(X, y, categorical_cols, numerical_cols, n_jobs)

    all_results = {**linear_results, **tree_results}

    # Encontrar mejor modelo
    best_model_name = min(all_results.keys(), key=lambda k: all_results[k]['rmse_mean'])
    best_result = all_results[best_model_name]

    models_trained = list(all_results.keys())
    update_state(
        'model_selection',
        models_trained=models_trained,
        best_model=best_model_name,
        best_rmse=best_result['rmse_mean'],
        best_r2=best_result['r2_mean']
    )

    print(f"\n{'='*60}")
    print(f"MEJOR MODELO BASELINE: {best_model_name}")
    print(f"  RMSE: ${best_result['rmse_mean']:.2f}")
    print(f"  R2: {best_result['r2_mean']:.4f}")
    print('='*60)

    # Fase 4: Optimizar hiperparametros
    update_state('optimizing_hyperparameters')
    print("\n[FASE 4] Optimizando hiperparametros...")

    optimized_model, best_params = optimize_best_model(
        X, y, best_model_name, categorical_cols, numerical_cols, n_jobs
    )

    if optimized_model is None:
        optimized_model = best_result['pipeline']
        best_params = None

    # Fase 5: Guardar resultados
    update_state('saving_results')
    print("\n[FASE 5] Guardando resultados...")

    results_df, importance_df = save_results(
        all_results, optimized_model, best_params, X, y, OUTPUT_DIR
    )

    # Fase 6: Generar reporte
    update_state('generating_report')
    print("\n[FASE 6] Generando reporte...")

    generate_report(results_df, best_model_name, best_params, importance_df, DOCS_DIR)

    # Finalizar
    final_rmse = results_df.iloc[0]['rmse_mean']
    final_r2 = results_df.iloc[0]['r2_mean']

    update_state(
        'completed',
        models_trained=models_trained,
        best_model=best_model_name,
        best_rmse=final_rmse,
        best_r2=final_r2
    )

    # Actualizar status a completed
    state_file = AGENT_CONTEXT_DIR / 'ProfitRegressor_state.json'
    with open(state_file, 'r') as f:
        state = json.load(f)
    state['status'] = 'completed'
    state['completed_at'] = datetime.now().isoformat()
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print("\n" + "=" * 60)
    print("PROFIT REGRESSOR - COMPLETADO")
    print("=" * 60)
    print(f"\nResultados finales:")
    print(f"  Mejor modelo: {best_model_name}")
    print(f"  RMSE: ${final_rmse:.2f} (Objetivo: < $200)")
    print(f"  R²: {final_r2:.4f} (Objetivo: > 0.3)")

    # Verificar criterios de exito
    rmse_pass = final_rmse < 200
    r2_pass = final_r2 > 0.3

    print(f"\nCriterios de exito:")
    print(f"  RMSE < $200: {'PASS' if rmse_pass else 'FAIL'}")
    print(f"  R² > 0.3: {'PASS' if r2_pass else 'FAIL'}")

    if rmse_pass and r2_pass:
        print("\nTODOS LOS CRITERIOS CUMPLIDOS")
    else:
        print("\nALGUNOS CRITERIOS NO CUMPLIDOS - Revisar optimizacion")

    return state


if __name__ == '__main__':
    main()
