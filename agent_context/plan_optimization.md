# Plan: Optimización de Modelos - Sales Exercise

**Proyecto**: sales_exercise (optimización)
**Fecha**: 2026-01-18
**Estado**: Aprobado

---

## Resumen

Re-entrenamiento de 3 modelos que no alcanzaron objetivos óptimos:
1. Forecasting: MAPE 19% → objetivo <15%
2. Regresión: R² 0.43 → objetivo >0.5
3. Clustering: 2 segmentos → objetivo 4-5 segmentos accionables

---

## Agentes

### ForecasterV2

- **Responsabilidad**: Mejorar forecasting usando agregación semanal en lugar de mensual

- **Cambios vs versión anterior**:
  - Agregación: **semanal** (W) en lugar de mensual (~200 puntos vs 48)
  - Más datos = mejor captura de patrones

- **Modelos a comparar**:
  - Prophet (con seasonality semanal)
  - SARIMA con más granularidad
  - XGBoost con lags semanales

- **Preprocesamiento**:
  - Agregar Sales por semana: `df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum()`
  - Train: primeras 180 semanas, Test: últimas 20 semanas

- **Feature Engineering** (XGBoost):
  - `week_of_year` (1-52)
  - `month`, `quarter`
  - `lag_1`, `lag_2`, `lag_4`, `lag_52` (año anterior)
  - `rolling_mean_4`, `rolling_mean_12`

- **Validación**:
  - Time Series Split
  - Métricas: MAPE (principal), RMSE, MAE
  - **Objetivo**: MAPE < 15%

- **Dependencias**: Ninguna

- **Output**:
  - `output/forecaster_v2_model.pkl`
  - `output/forecaster_v2_results.csv`
  - `output/forecast_weekly_plot.png`
  - `docs/forecaster_v2_report.md`

---

### RegressorV2

- **Responsabilidad**: Mejorar predicción de Profit con feature engineering avanzado y segmentación

- **Cambios vs versión anterior**:
  1. Feature engineering con interacciones
  2. Modelos segmentados por Category
  3. Target encoding para alta cardinalidad

- **Estrategia de Segmentación**:
  ```python
  # Entrenar un modelo por Category
  for category in ['Furniture', 'Office Supplies', 'Technology']:
      df_cat = df[df['Category'] == category]
      model_cat = train_model(df_cat)
  ```

- **Feature Engineering Adicional**:
  - `discount_x_quantity`: Discount * Quantity
  - `segment_region`: Segment + '_' + Region (interacción)
  - `subcategory_discount`: Sub-Category + nivel de descuento
  - `is_high_quantity`: Quantity > 5
  - `shipping_speed`: días de envío categorizados

- **Modelos**:
  - LightGBM con más hiperparámetros
  - CatBoost (maneja categóricas nativamente)
  - Stacking ensemble

- **Validación**:
  - 5-Fold CV
  - **Objetivo**: R² > 0.5, RMSE < $70

- **Dependencias**: Ninguna

- **Output**:
  - `output/regressor_v2_model.pkl` (o dict de modelos por categoría)
  - `output/regressor_v2_results.csv`
  - `output/regressor_v2_comparison.png`
  - `docs/regressor_v2_report.md`

---

### ClustererV2

- **Responsabilidad**: Generar segmentación con 4-5 clusters accionables para marketing

- **Cambios vs versión anterior**:
  - Forzar K-Means con k=4 o k=5 (evaluar ambos)
  - Priorizar interpretabilidad sobre Silhouette
  - Nombrar segmentos con perfiles de negocio

- **Segmentos Objetivo** (estilo RFM clásico):
  1. **Champions**: Alta F, Alta M, Baja R
  2. **Loyal Customers**: Alta F, Media M
  3. **At Risk**: Baja F reciente, Histórico bueno
  4. **Lost**: Alta R, Baja F
  5. (Opcional) **New Customers**: Baja F, Baja R

- **Modelos**:
  - K-Means con k=4
  - K-Means con k=5
  - Elegir el que tenga mejor balance Silhouette + interpretabilidad

- **Análisis Adicional**:
  - Perfil detallado de cada cluster
  - Recomendaciones de marketing por segmento
  - Matriz de transición (si hubiera datos temporales)

- **Validación**:
  - Silhouette Score (aceptable >0.25 si hay buena interpretabilidad)
  - Tamaño de clusters (ninguno <5% del total)

- **Dependencias**: Ninguna

- **Output**:
  - `output/clusterer_v2_model.pkl`
  - `output/customer_segments_v2.csv`
  - `output/cluster_profiles_v2.png`
  - `output/segment_recommendations.md`
  - `docs/clusterer_v2_report.md`

---

## Orden de Ejecución

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ForecasterV2 │  │ RegressorV2  │  │ ClustererV2  │
└──────────────┘  └──────────────┘  └──────────────┘
       │                │                  │
       └────────────────┴──────────────────┘
                        │
              (Todos en paralelo)
```

---

## Criterios de Éxito

| Modelo | Métrica | Objetivo |
|--------|---------|----------|
| ForecasterV2 | MAPE | < 15% |
| RegressorV2 | R² | > 0.50 |
| RegressorV2 | RMSE | < $70 |
| ClustererV2 | Silhouette | > 0.25 |
| ClustererV2 | Clusters | 4-5 interpretables |

---

*Plan de optimización generado: 2026-01-18*
