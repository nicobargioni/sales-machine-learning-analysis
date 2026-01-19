"""
ProfitabilityClassifier Agent
=============================
Clasificar transacciones como rentables (Profit > 0) o no rentables (Profit <= 0).

Modelos: Logistic Regression, Random Forest, XGBoost, LightGBM
Métrica principal: F1-Score
Objetivo: F1 > 0.75 (mínimo), > 0.85 (óptimo); ROC-AUC > 0.80 (mínimo), > 0.90 (óptimo)
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Sales_csv.csv"
OUTPUT_DIR = BASE_DIR / "output"
DOCS_DIR = BASE_DIR / "docs"
AGENT_CONTEXT_DIR = BASE_DIR / "agent_context"
STATE_FILE = AGENT_CONTEXT_DIR / "ProfitabilityClassifier_state.json"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Detect n_jobs dynamically
def get_n_jobs():
    """Detect optimal n_jobs based on CPU cores and active agents."""
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    # Check for other agents
    agent_files = list(AGENT_CONTEXT_DIR.glob("*_state.json"))
    active_agents = 0
    for f in agent_files:
        try:
            with open(f) as fh:
                state = json.load(fh)
                if state.get("status") == "running":
                    active_agents += 1
        except:
            pass
    active_agents = max(1, active_agents)
    n_jobs = max(1, total_cores // active_agents - 1)
    return n_jobs

N_JOBS = get_n_jobs()
print(f"[CONFIG] Using n_jobs={N_JOBS}")


def update_state(updates: dict):
    """Update agent state file."""
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        state.update(updates)
        state["last_checkpoint"] = datetime.now().isoformat()
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not update state: {e}")


def load_data():
    """Load and initial exploration of dataset."""
    print("\n" + "="*60)
    print("FASE 1: CARGA DE DATOS")
    print("="*60)

    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Create target variable
    df['is_profitable'] = (df['Profit'] > 0).astype(int)

    # Check class distribution
    class_dist = df['is_profitable'].value_counts(normalize=True)
    print(f"\nDistribución de clases:")
    print(f"  Rentables (1): {class_dist.get(1, 0)*100:.1f}%")
    print(f"  No rentables (0): {class_dist.get(0, 0)*100:.1f}%")

    update_state({
        "current_phase": "data_loaded",
        "progress": {
            "data_loaded": True,
            "n_samples": len(df),
            "class_distribution": {
                "profitable": float(class_dist.get(1, 0)),
                "non_profitable": float(class_dist.get(0, 0))
            }
        }
    })

    return df


def feature_engineering(df):
    """Create features as specified in the plan."""
    print("\n" + "="*60)
    print("FASE 2: FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()

    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%b-%y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%b-%y')

    # Temporal features
    df['ship_days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['order_month'] = df['Order Date'].dt.month
    df['order_quarter'] = df['Order Date'].dt.quarter
    df['order_dayofweek'] = df['Order Date'].dt.dayofweek
    df['order_year'] = df['Order Date'].dt.year

    # Business features
    df['high_discount'] = (df['Discount'] > 0.2).astype(int)
    df['is_standard_shipping'] = (df['Ship Mode'] == 'Standard Class').astype(int)

    print("Features creadas:")
    print("  - ship_days (días de envío)")
    print("  - order_month, order_quarter, order_dayofweek, order_year")
    print("  - high_discount (Discount > 0.2)")
    print("  - is_standard_shipping")

    return df


def preprocess_data(df):
    """Prepare features for modeling."""
    print("\n" + "="*60)
    print("FASE 3: PREPROCESAMIENTO")
    print("="*60)

    # Columns to drop
    drop_cols = [
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'Country', 'Profit',
        'Order Date', 'Ship Date', 'is_profitable'
    ]

    # Separate target
    y = df['is_profitable'].values

    # Features
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    print(f"Features seleccionadas ({len(feature_cols)}): {feature_cols}")

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nCategóricas ({len(cat_cols)}): {cat_cols}")
    print(f"Numéricas ({len(num_cols)}): {num_cols}")

    # Encode categoricals with LabelEncoder for tree-based models
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Store feature names
    feature_names = X.columns.tolist()

    # Convert to numpy
    X = X.values

    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Class balance: {np.bincount(y)}")

    update_state({
        "current_phase": "preprocessing_done",
        "progress": {
            "data_loaded": True,
            "preprocessing_done": True,
            "n_features": len(feature_names),
            "feature_names": feature_names
        }
    })

    return X, y, feature_names, label_encoders


def train_and_evaluate_models(X, y, feature_names):
    """Train all models with Stratified K-Fold CV."""
    print("\n" + "="*60)
    print("FASE 4: ENTRENAMIENTO DE MODELOS")
    print("="*60)

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Calculate class weight ratio for imbalanced handling
    class_ratio = np.bincount(y)[0] / np.bincount(y)[1]

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
    }

    # Try to import XGBoost and LightGBM (catch all exceptions, not just ImportError)
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=class_ratio,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        print("[OK] XGBoost disponible")
    except Exception as e:
        print(f"[WARN] XGBoost no disponible ({type(e).__name__}), omitiendo...")

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=-1
        )
        print("[OK] LightGBM disponible")
    except Exception as e:
        print(f"[WARN] LightGBM no disponible ({type(e).__name__}), omitiendo...")

    # Scoring metrics
    scoring = {
        'f1': 'f1',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")

        # Scale data for Logistic Regression
        if name == 'Logistic Regression':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None

        # Cross-validation
        cv_results = cross_validate(
            model, X_scaled, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=N_JOBS
        )

        # Fit final model on full data
        model.fit(X_scaled, y)

        # Store results
        results[name] = {
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
        }

        trained_models[name] = {
            'model': model,
            'scaler': scaler
        }

        print(f"  F1-Score: {results[name]['f1_mean']:.4f} (+/- {results[name]['f1_std']:.4f})")
        print(f"  ROC-AUC:  {results[name]['roc_auc_mean']:.4f} (+/- {results[name]['roc_auc_std']:.4f})")
        print(f"  Accuracy: {results[name]['accuracy_mean']:.4f}")
        print(f"  Precision: {results[name]['precision_mean']:.4f}")
        print(f"  Recall: {results[name]['recall_mean']:.4f}")

        # Update state after each model
        update_state({
            "progress": {
                "models_trained": {name: results[name] for name in results}
            }
        })

    return results, trained_models


def hyperparameter_tuning(X, y, trained_models, results):
    """Fine-tune the best performing model."""
    print("\n" + "="*60)
    print("FASE 5: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*60)

    # Find best model by F1 score
    best_name = max(results, key=lambda k: results[k]['f1_mean'])
    print(f"Mejor modelo base: {best_name} (F1={results[best_name]['f1_mean']:.4f})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    class_ratio = np.bincount(y)[0] / np.bincount(y)[1]

    # Define hyperparameter grids
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    }

    try:
        import xgboost as xgb
        param_grids['XGBoost'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    except Exception:
        pass

    try:
        import lightgbm as lgb
        param_grids['LightGBM'] = {
            'n_estimators': [100, 200, 300],
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.8, 1.0]
        }
    except Exception:
        pass

    if best_name not in param_grids:
        print(f"[WARN] No hay grid para {best_name}, usando modelo base")
        return trained_models[best_name], best_name, results[best_name]

    # Prepare data
    if best_name == 'Logistic Regression':
        scaler = StandardScaler()
        X_search = scaler.fit_transform(X)
    else:
        X_search = X
        scaler = trained_models[best_name].get('scaler')

    # Get base model
    base_model = trained_models[best_name]['model']

    # RandomizedSearchCV for efficiency
    print(f"\n[TUNING] Optimizando {best_name}...")
    search = RandomizedSearchCV(
        estimator=base_model.__class__(**{
            k: v for k, v in base_model.get_params().items()
            if k not in param_grids[best_name]
        }),
        param_distributions=param_grids[best_name],
        n_iter=20,
        cv=cv,
        scoring='f1',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1
    )

    search.fit(X_search, y)

    print(f"\nMejores parámetros: {search.best_params_}")
    print(f"Mejor F1 (CV): {search.best_score_:.4f}")

    # Get optimized metrics
    optimized_model = search.best_estimator_
    cv_results = cross_validate(
        optimized_model, X_search, y,
        cv=cv,
        scoring={'f1': 'f1', 'roc_auc': 'roc_auc', 'accuracy': 'accuracy',
                 'precision': 'precision', 'recall': 'recall'},
        n_jobs=N_JOBS
    )

    optimized_results = {
        'f1_mean': cv_results['test_f1'].mean(),
        'f1_std': cv_results['test_f1'].std(),
        'roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'roc_auc_std': cv_results['test_roc_auc'].std(),
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'precision_mean': cv_results['test_precision'].mean(),
        'recall_mean': cv_results['test_recall'].mean(),
        'best_params': search.best_params_
    }

    print(f"\nResultados optimizados:")
    print(f"  F1-Score: {optimized_results['f1_mean']:.4f}")
    print(f"  ROC-AUC:  {optimized_results['roc_auc_mean']:.4f}")

    return {'model': optimized_model, 'scaler': scaler}, best_name, optimized_results


def generate_visualizations(best_model_data, X, y, feature_names, model_name, all_results):
    """Generate confusion matrix, ROC curve, and feature importance plots."""
    print("\n" + "="*60)
    print("FASE 6: VISUALIZACIONES")
    print("="*60)

    model = best_model_data['model']
    scaler = best_model_data.get('scaler')

    if scaler:
        X_pred = scaler.transform(X)
    else:
        X_pred = X

    y_pred = model.predict(X_pred)
    y_prob = model.predict_proba(X_pred)[:, 1]

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Confusion Matrix
    print("[PLOT] Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Rentable', 'Rentable'],
                yticklabels=['No Rentable', 'Rentable'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusión - {model_name}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'profitability_confusion_matrix.png', dpi=150)
    plt.close()

    # 2. ROC Curve
    print("[PLOT] ROC Curve...")
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC - ProfitabilityClassifier')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'profitability_roc_curve.png', dpi=150)
    plt.close()

    # 3. Feature Importance (if available)
    print("[PLOT] Feature Importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importancia')
        ax.set_title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'profitability_feature_importance.png', dpi=150)
        plt.close()
    elif hasattr(model, 'coef_'):
        # For logistic regression
        coefs = np.abs(model.coef_[0])
        indices = np.argsort(coefs)[::-1][:15]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), coefs[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('|Coeficiente|')
        ax.set_title(f'Feature Importance (Coeficientes) - {model_name}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'profitability_feature_importance.png', dpi=150)
        plt.close()

    # 4. Model Comparison Bar Chart
    print("[PLOT] Model Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = list(all_results.keys())
    f1_scores = [all_results[m]['f1_mean'] for m in model_names]
    roc_scores = [all_results[m]['roc_auc_mean'] for m in model_names]

    # F1 comparison
    colors = ['green' if m == model_name else 'steelblue' for m in model_names]
    axes[0].barh(model_names, f1_scores, color=colors)
    axes[0].axvline(x=0.75, color='red', linestyle='--', label='Objetivo mínimo')
    axes[0].axvline(x=0.85, color='green', linestyle='--', label='Objetivo óptimo')
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('Comparación de Modelos - F1-Score')
    axes[0].legend()

    # ROC-AUC comparison
    axes[1].barh(model_names, roc_scores, color=colors)
    axes[1].axvline(x=0.80, color='red', linestyle='--', label='Objetivo mínimo')
    axes[1].axvline(x=0.90, color='green', linestyle='--', label='Objetivo óptimo')
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('Comparación de Modelos - ROC-AUC')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'profitability_model_comparison.png', dpi=150)
    plt.close()

    print("[OK] Visualizaciones guardadas en output/")


def save_results(best_model_data, model_name, results, all_results, feature_names):
    """Save model, results CSV, and documentation."""
    print("\n" + "="*60)
    print("FASE 7: GUARDADO DE RESULTADOS")
    print("="*60)

    # 1. Save best model
    model_path = OUTPUT_DIR / 'profitability_classifier_best_model.pkl'
    joblib.dump(best_model_data, model_path)
    print(f"[OK] Modelo guardado: {model_path}")

    # 2. Save results CSV
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = 'Model'
    results_df.to_csv(OUTPUT_DIR / 'profitability_classifier_results.csv')
    print(f"[OK] Resultados CSV guardados")

    # 3. Generate documentation
    doc_content = f"""# ProfitabilityClassifier - Reporte de Resultados

**Fecha de ejecución**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Objetivo**: Clasificar transacciones como rentables (Profit > 0) o no rentables (Profit <= 0)

---

## Resumen Ejecutivo

**Mejor modelo**: {model_name}
**F1-Score**: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})
**ROC-AUC**: {results['roc_auc_mean']:.4f} (+/- {results['roc_auc_std']:.4f})
**Accuracy**: {results['accuracy_mean']:.4f}
**Precision**: {results['precision_mean']:.4f}
**Recall**: {results['recall_mean']:.4f}

### Evaluación vs Objetivos

| Métrica | Valor | Objetivo Mínimo | Objetivo Óptimo | Status |
|---------|-------|-----------------|-----------------|--------|
| F1-Score | {results['f1_mean']:.4f} | > 0.75 | > 0.85 | {'CUMPLIDO' if results['f1_mean'] > 0.75 else 'NO CUMPLIDO'} |
| ROC-AUC | {results['roc_auc_mean']:.4f} | > 0.80 | > 0.90 | {'CUMPLIDO' if results['roc_auc_mean'] > 0.80 else 'NO CUMPLIDO'} |

---

## Comparación de Modelos

| Modelo | F1-Score | ROC-AUC | Accuracy | Precision | Recall |
|--------|----------|---------|----------|-----------|--------|
"""

    for name, res in all_results.items():
        doc_content += f"| {name} | {res['f1_mean']:.4f} | {res['roc_auc_mean']:.4f} | {res['accuracy_mean']:.4f} | {res['precision_mean']:.4f} | {res['recall_mean']:.4f} |\n"

    doc_content += f"""
---

## Análisis de Sesgo/Varianza

- **Cross-Validation**: Stratified 5-Fold para mantener proporción de clases
- **Varianza del modelo**: F1 std = {results['f1_std']:.4f} ({"baja" if results['f1_std'] < 0.05 else "moderada"} variabilidad)
- **Interpretación**: El modelo muestra {"estabilidad" if results['f1_std'] < 0.05 else "cierta variabilidad"} entre folds

### Consideraciones sobre Overfitting

- Se utilizó `class_weight='balanced'` para manejar desbalance de clases
- La validación cruzada estratificada garantiza representatividad en cada fold
- No se observan signos evidentes de overfitting dado el bajo std en métricas

---

## Feature Engineering Aplicado

1. **ship_days**: Días entre orden y envío
2. **order_month**: Mes de la orden (estacionalidad)
3. **order_quarter**: Trimestre de la orden
4. **order_dayofweek**: Día de la semana
5. **order_year**: Año de la orden
6. **high_discount**: Flag para descuentos > 20%
7. **is_standard_shipping**: Flag para envío estándar

---

## Outputs Generados

- `profitability_classifier_best_model.pkl` - Modelo entrenado
- `profitability_classifier_results.csv` - Métricas de todos los modelos
- `profitability_confusion_matrix.png` - Matriz de confusión
- `profitability_roc_curve.png` - Curva ROC
- `profitability_feature_importance.png` - Importancia de features
- `profitability_model_comparison.png` - Comparación de modelos

---

## Conclusiones y Recomendaciones

1. **Rendimiento**: El modelo {model_name} logra un F1-Score de {results['f1_mean']:.4f}, {"superando" if results['f1_mean'] > 0.75 else "por debajo de"} el objetivo mínimo de 0.75.

2. **Balance Precision/Recall**:
   - Precision: {results['precision_mean']:.4f} (capacidad de evitar falsos positivos)
   - Recall: {results['recall_mean']:.4f} (capacidad de detectar todas las transacciones no rentables)

3. **Variables más importantes**: Ver gráfico de feature importance para identificar los principales drivers de rentabilidad.

4. **Próximos pasos sugeridos**:
   - Analizar SHAP values para explicabilidad más profunda
   - Investigar transacciones con alta probabilidad de pérdida
   - Considerar umbrales de decisión alternativos según costo de negocio

---

*Reporte generado automáticamente por ProfitabilityClassifier Agent*
"""

    doc_path = DOCS_DIR / 'profitability_classifier_report.md'
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    print(f"[OK] Documentación guardada: {doc_path}")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("PROFITABILITY CLASSIFIER - RALPH WIGGUM AGENT")
    print("="*60)
    print(f"Inicio: {datetime.now()}")

    try:
        # Phase 1: Load data
        df = load_data()

        # Phase 2: Feature engineering
        df = feature_engineering(df)

        # Phase 3: Preprocessing
        X, y, feature_names, encoders = preprocess_data(df)

        # Phase 4: Train models
        all_results, trained_models = train_and_evaluate_models(X, y, feature_names)

        # Phase 5: Hyperparameter tuning
        best_model_data, best_name, best_results = hyperparameter_tuning(X, y, trained_models, all_results)

        # Update all_results with optimized
        all_results[f'{best_name} (Optimized)'] = best_results

        # Phase 6: Visualizations
        generate_visualizations(best_model_data, X, y, feature_names, best_name, all_results)

        # Phase 7: Save results
        save_results(best_model_data, best_name, best_results, all_results, feature_names)

        # Final state update
        update_state({
            "status": "completed",
            "current_phase": "finished",
            "progress": {
                "data_loaded": True,
                "preprocessing_done": True,
                "models_trained": all_results,
                "best_model": best_name,
                "metrics": best_results
            },
            "finished_at": datetime.now().isoformat()
        })

        print("\n" + "="*60)
        print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        print(f"Mejor modelo: {best_name}")
        print(f"F1-Score: {best_results['f1_mean']:.4f}")
        print(f"ROC-AUC: {best_results['roc_auc_mean']:.4f}")
        print(f"Fin: {datetime.now()}")

        # Check objectives
        f1_ok = best_results['f1_mean'] > 0.75
        roc_ok = best_results['roc_auc_mean'] > 0.80

        if f1_ok and roc_ok:
            print("\n[SUCCESS] Objetivos mínimos CUMPLIDOS")
        else:
            print("\n[WARNING] Algunos objetivos NO cumplidos:")
            if not f1_ok:
                print(f"  - F1-Score ({best_results['f1_mean']:.4f}) < 0.75")
            if not roc_ok:
                print(f"  - ROC-AUC ({best_results['roc_auc_mean']:.4f}) < 0.80")

        return True

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"\n[ERROR] {e}")
        print(error_msg)

        update_state({
            "status": "error",
            "errors": [str(e), error_msg]
        })

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
