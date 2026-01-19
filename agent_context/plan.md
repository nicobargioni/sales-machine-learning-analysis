# Plan: Sales Exercise

**Proyecto**: sales_exercise  
**Dataset**: datawitharyan/sales-order-dataset (Kaggle)  
**Fecha**: 2026-01-18  
**Estado**: Borrador

---

## Resumen Ejecutivo

Proyecto de Data Science sobre datos de ventas retail (9,994 órdenes, 4 años). Se ejecutarán 4 ejercicios de ML en paralelo:
1. **Regresión**: Predecir el profit de cada transacción
2. **Clasificación**: Detectar transacciones rentables vs no rentables
3. **Forecasting**: Proyectar ventas mensuales futuras
4. **Clustering**: Segmentar clientes usando análisis RFM

Cada ejercicio será ejecutado por un agente autónomo que comparará múltiples modelos y documentará resultados.

---

## Dataset

| Métrica | Valor |
|---------|-------|
| Registros | 9,994 |
| Features | 21 |
| Período | 2014-01-03 → 2017-12-30 |
| País | United States |
| Clientes únicos | 793 |
| Productos únicos | 1,862 |

### Features Principales
- **Numéricas**: Sales, Quantity, Discount, Profit, Postal Code
- **Categóricas**: Ship Mode, Segment, Region, Category, Sub-Category, State, City
- **Temporales**: Order Date, Ship Date
- **IDs**: Order ID, Customer ID, Product ID

---

## Agentes

### ProfitRegressor

- **Responsabilidad**: Predecir el valor de Profit para cada transacción usando modelos de regresión supervisada.

- **Modelos a comparar**:
  - Linear Regression (baseline)
  - Ridge Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor

- **Preprocesamiento**:
  - **Categóricas**: 
    - One-Hot Encoding para Ship Mode, Segment, Region (baja cardinalidad)
    - Target Encoding para Category, Sub-Category, State, City (alta cardinalidad, para tree-based)
    - One-Hot para todos en modelos lineales
  - **Numéricas**: 
    - StandardScaler para modelos lineales (Linear, Ridge)
    - Sin scaling para tree-based (RF, XGB, LGBM)
  - **Target**: 
    - Considerar log-transform si mejora distribución (probar con y sin)
  - **Drop**: Row ID, Order ID, Customer ID, Customer Name, Product ID, Product Name, Country (constante)

- **Feature Engineering**:
  - `ship_days`: Ship Date - Order Date (tiempo de envío)
  - `order_month`: Mes de la orden (1-12)
  - `order_quarter`: Trimestre (1-4)
  - `order_dayofweek`: Día de la semana (0-6)
  - `order_year`: Año de la orden
  - `margin_potential`: Sales * (1 - Discount) - proxy de margen

- **Validación**:
  - 5-Fold Cross-Validation (KFold, shuffle=True)
  - Métricas: RMSE (principal), MAE, R², MAPE

- **Consideraciones especiales**:
  - **Outliers**: Profit tiene 18.8% outliers. Probar con y sin tratamiento (winsorizing al percentil 1-99)
  - **Valores negativos**: El target puede ser negativo, no usar log-transform directo
  - **Leakage**: NO usar Sales como feature (altamente correlacionado con target, sería leakage conceptual)

- **Hiperparámetros**:
  ```python
  # XGBoost
  {
      'n_estimators': [100, 200, 500],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.05, 0.1],
      'subsample': [0.8, 1.0],
      'colsample_bytree': [0.8, 1.0]
  }
  # LightGBM
  {
      'n_estimators': [100, 200, 500],
      'num_leaves': [31, 50, 100],
      'learning_rate': [0.01, 0.05, 0.1],
      'feature_fraction': [0.8, 1.0]
  }
  # Random Forest
  {
      'n_estimators': [100, 200, 500],
      'max_depth': [10, 20, None],
      'min_samples_split': [2, 5, 10]
  }
  ```

- **Dependencias**: Ninguna

- **Output**:
  - `output/profit_regressor_best_model.pkl`
  - `output/profit_regressor_results.csv`
  - `output/profit_regressor_feature_importance.png`
  - `docs/profit_regressor_report.md`

---

### ProfitabilityClassifier

- **Responsabilidad**: Clasificar transacciones como rentables (`Profit > 0`) o no rentables (`Profit <= 0`).

- **Modelos a comparar**:
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - XGBoost Classifier
  - LightGBM Classifier
  - SVM (si el dataset no es muy grande para entrenamiento)

- **Preprocesamiento**:
  - **Categóricas**: 
    - One-Hot Encoding para baja cardinalidad
    - Target Encoding para alta cardinalidad (tree-based)
  - **Numéricas**: 
    - StandardScaler para Logistic Regression y SVM
    - Sin scaling para tree-based
  - **Target**: 
    - `is_profitable = 1 if Profit > 0 else 0`
  - **Drop**: Row ID, Order ID, Customer ID, Customer Name, Product ID, Product Name, Country, Profit (es el target)

- **Feature Engineering**:
  - `ship_days`: Ship Date - Order Date
  - `order_month`, `order_quarter`, `order_dayofweek`, `order_year`
  - `high_discount`: 1 if Discount > 0.2 else 0
  - `is_standard_shipping`: 1 if Ship Mode == 'Standard Class' else 0

- **Validación**:
  - Stratified 5-Fold Cross-Validation (mantener proporción de clases)
  - Métricas: F1-Score (principal), Accuracy, Precision, Recall, ROC-AUC

- **Consideraciones especiales**:
  - **Desbalance**: Verificar distribución de clases. Si hay desbalance >70/30, aplicar:
    - `class_weight='balanced'` en modelos que lo soporten
    - SMOTE si es severo
  - **Interpretabilidad**: Calcular feature importance y SHAP values para explicar qué causa pérdidas

- **Hiperparámetros**:
  ```python
  # XGBoost
  {
      'n_estimators': [100, 200],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.1],
      'scale_pos_weight': [1, ratio_clases]
  }
  # Logistic Regression
  {
      'C': [0.01, 0.1, 1, 10],
      'penalty': ['l1', 'l2'],
      'solver': ['liblinear', 'saga']
  }
  ```

- **Dependencias**: Ninguna

- **Output**:
  - `output/profitability_classifier_best_model.pkl`
  - `output/profitability_classifier_results.csv`
  - `output/profitability_confusion_matrix.png`
  - `output/profitability_roc_curve.png`
  - `docs/profitability_classifier_report.md`

---

### SalesForecaster

- **Responsabilidad**: Predecir ventas agregadas mensuales para los próximos 3-6 meses usando series temporales.

- **Modelos a comparar**:
  - Naive (baseline: último valor / promedio móvil)
  - ARIMA / SARIMA
  - Prophet (Facebook)
  - XGBoost con features temporales
  - (Opcional) LSTM si hay suficientes datos

- **Preprocesamiento**:
  - **Agregación**: Sumar Sales por mes (Order Date)
  - **Frecuencia**: Mensual (MS)
  - **Período**: ~48 meses de datos
  - **Train/Test Split**: Últimos 6 meses como test (temporal split, NO random)

- **Feature Engineering** (para XGBoost temporal):
  - `month`: Mes del año (1-12)
  - `quarter`: Trimestre
  - `year`: Año
  - `lag_1`, `lag_2`, `lag_3`: Ventas de meses anteriores
  - `rolling_mean_3`, `rolling_mean_6`: Promedios móviles
  - `rolling_std_3`: Volatilidad

- **Validación**:
  - Time Series Split (expanding window) - NO usar KFold normal
  - Métricas: RMSE (principal), MAE, MAPE

- **Consideraciones especiales**:
  - **Estacionalidad**: Detectar y modelar patrones estacionales (fin de año, Q4)
  - **Tendencia**: Verificar si hay tendencia creciente/decreciente
  - **Pocos datos**: Solo ~48 puntos mensuales, modelos simples pueden funcionar mejor
  - **Backtesting**: Validar predicciones en múltiples ventanas temporales

- **Hiperparámetros**:
  ```python
  # SARIMA (usar auto_arima para búsqueda)
  {
      'p': range(0, 3),
      'd': range(0, 2),
      'q': range(0, 3),
      'P': range(0, 2),
      'D': range(0, 2),
      'Q': range(0, 2),
      'm': 12  # estacionalidad mensual
  }
  # Prophet
  {
      'changepoint_prior_scale': [0.01, 0.1, 0.5],
      'seasonality_prior_scale': [0.1, 1, 10],
      'seasonality_mode': ['additive', 'multiplicative']
  }
  ```

- **Dependencias**: Ninguna

- **Output**:
  - `output/sales_forecaster_best_model.pkl`
  - `output/sales_forecast_predictions.csv`
  - `output/sales_forecast_plot.png`
  - `output/sales_decomposition.png`
  - `docs/sales_forecaster_report.md`

---

### CustomerSegmenter

- **Responsabilidad**: Segmentar la base de clientes usando análisis RFM (Recency, Frequency, Monetary) y clustering no supervisado.

- **Modelos a comparar**:
  - K-Means (baseline)
  - K-Means con PCA
  - DBSCAN
  - Hierarchical Clustering (Agglomerative)
  - Gaussian Mixture Models (GMM)

- **Preprocesamiento**:
  - **Agregación por Customer ID**:
    - `Recency`: Días desde última compra (referencia: última fecha del dataset)
    - `Frequency`: Número total de órdenes
    - `Monetary`: Suma total de Sales
  - **Scaling**: StandardScaler obligatorio para todos los modelos de clustering
  - **Outliers**: Considerar log-transform o winsorizing para RFM (distribuciones sesgadas)

- **Feature Engineering**:
  - `avg_order_value`: Monetary / Frequency
  - `avg_profit`: Suma de Profit / Frequency
  - `days_as_customer`: Última orden - Primera orden
  - `preferred_category`: Categoría más comprada (modo)
  - `preferred_segment`: Segment del cliente

- **Validación**:
  - **Elbow Method**: Para K-Means, graficar inertia vs k
  - **Silhouette Score**: Métrica principal para comparar modelos
  - **Davies-Bouldin Index**: Métrica secundaria
  - **Análisis cualitativo**: Perfilar clusters resultantes

- **Consideraciones especiales**:
  - **Número de clusters**: Probar k=3 a k=8, elegir por silhouette + interpretabilidad
  - **Interpretabilidad**: Cada cluster debe tener un perfil claro y accionable
  - **Business alignment**: Nombrar clusters (ej: "Champions", "At Risk", "Lost")

- **Hiperparámetros**:
  ```python
  # K-Means
  {
      'n_clusters': range(3, 9),
      'init': ['k-means++', 'random'],
      'n_init': 10
  }
  # DBSCAN
  {
      'eps': [0.3, 0.5, 0.7, 1.0],
      'min_samples': [3, 5, 10]
  }
  # Hierarchical
  {
      'n_clusters': range(3, 9),
      'linkage': ['ward', 'complete', 'average']
  }
  ```

- **Dependencias**: Ninguna

- **Output**:
  - `output/customer_segmenter_model.pkl`
  - `output/customer_segments.csv`
  - `output/rfm_distribution.png`
  - `output/cluster_profiles.png`
  - `output/elbow_silhouette_plot.png`
  - `docs/customer_segmenter_report.md`

---

### Reporter

- **Responsabilidad**: Generar paper final en LaTeX consolidando resultados de todos los agentes.

- **Estructura del Paper**:
  1. **Abstract**: Resumen ejecutivo de metodología y hallazgos
  2. **Introducción**: Contexto del problema de negocio
  3. **Dataset y EDA**: Descripción de datos, hallazgos exploratorios
  4. **Metodología**: 
     - 4.1 Regresión de Profit
     - 4.2 Clasificación de Rentabilidad
     - 4.3 Forecasting de Ventas
     - 4.4 Segmentación de Clientes
  5. **Resultados**: Tablas comparativas, métricas, gráficos
  6. **Discusión**: Interpretación de resultados, limitaciones
  7. **Conclusiones**: Recomendaciones de negocio
  8. **Anexos**: Hiperparámetros, código relevante

- **Dependencias**: ProfitRegressor, ProfitabilityClassifier, SalesForecaster, CustomerSegmenter (TODOS)

- **Output**:
  - `output/paper.tex`
  - `output/paper.pdf`

---

## Orden de Ejecución

```
┌─────────────────────┐   ┌─────────────────────────┐
│  ProfitRegressor    │   │  ProfitabilityClassifier│
└─────────┬───────────┘   └───────────┬─────────────┘
          │                           │
          │   ┌───────────────────┐   │
          │   │  SalesForecaster  │   │
          │   └─────────┬─────────┘   │
          │             │             │
          │   ┌─────────────────────┐ │
          │   │ CustomerSegmenter  │  │
          │   └─────────┬───────────┘ │
          │             │             │
          ▼             ▼             ▼
       ┌──────────────────────────────────┐
       │            Reporter              │
       └──────────────────────────────────┘

Paralelismo: Los 4 agentes de ML pueden ejecutarse en PARALELO
Reporter: Espera a que TODOS terminen
```

---

## Criterios de Éxito

| Ejercicio | Métrica Principal | Objetivo Mínimo | Objetivo Óptimo |
|-----------|-------------------|-----------------|-----------------|
| ProfitRegressor | RMSE | < $200 | < $150 |
| ProfitRegressor | R² | > 0.3 | > 0.5 |
| ProfitabilityClassifier | F1-Score | > 0.75 | > 0.85 |
| ProfitabilityClassifier | ROC-AUC | > 0.80 | > 0.90 |
| SalesForecaster | MAPE | < 20% | < 10% |
| CustomerSegmenter | Silhouette | > 0.3 | > 0.5 |

---

## Notas Técnicas

- **Entorno**: Python 3.x, usar `/data_science/.venv`
- **GPU**: Usar MPS (Apple Silicon) si disponible para modelos de Deep Learning
- **Reproducibilidad**: `random_state=42` en todos los modelos
- **n_jobs**: Ajustar dinámicamente según agentes activos (ver `CLAUDE.md`)
- **Logging**: Cada agente debe escribir progreso en `agent_context/[nombre]_status.json`

### Librerías Requeridas
```
pandas
numpy
scikit-learn
xgboost
lightgbm
prophet
matplotlib
seaborn
joblib
shap (opcional)
```

---

## Checklist Pre-Aprobación

- [x] EDA completado y documentado (`docs/EDA.md`)
- [x] Ejercicios de ML definidos con justificación
- [x] Preprocesamiento especificado por tipo de modelo
- [x] Métricas de validación definidas
- [x] Criterios de éxito establecidos
- [x] Dependencias entre agentes claras
- [x] Feature engineering especificado
- [x] Hiperparámetros a explorar definidos

---

*Plan generado: 2026-01-18 por Ralph Discovery Agent*
