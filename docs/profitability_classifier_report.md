# ProfitabilityClassifier - Reporte de Resultados

**Fecha de ejecución**: 2026-01-18 13:49:39
**Objetivo**: Clasificar transacciones como rentables (Profit > 0) o no rentables (Profit <= 0)

---

## Resumen Ejecutivo

**Mejor modelo**: Random Forest
**F1-Score**: 0.9665 (+/- 0.0014)
**ROC-AUC**: 0.9791 (+/- 0.0017)
**Accuracy**: 0.9451
**Precision**: 0.9518
**Recall**: 0.9816

### Evaluación vs Objetivos

| Métrica | Valor | Objetivo Mínimo | Objetivo Óptimo | Status |
|---------|-------|-----------------|-----------------|--------|
| F1-Score | 0.9665 | > 0.75 | > 0.85 | CUMPLIDO |
| ROC-AUC | 0.9791 | > 0.80 | > 0.90 | CUMPLIDO |

---

## Comparación de Modelos

| Modelo | F1-Score | ROC-AUC | Accuracy | Precision | Recall |
|--------|----------|---------|----------|-----------|--------|
| Logistic Regression | 0.9433 | 0.9493 | 0.9097 | 0.9552 | 0.9319 |
| Random Forest | 0.9648 | 0.9796 | 0.9426 | 0.9533 | 0.9767 |
| XGBoost | 0.9551 | 0.9816 | 0.9291 | 0.9754 | 0.9356 |
| LightGBM | 0.9586 | 0.9810 | 0.9341 | 0.9717 | 0.9458 |
| Random Forest (Optimized) | 0.9665 | 0.9791 | 0.9451 | 0.9518 | 0.9816 |

---

## Análisis de Sesgo/Varianza

- **Cross-Validation**: Stratified 5-Fold para mantener proporción de clases
- **Varianza del modelo**: F1 std = 0.0014 (baja variabilidad)
- **Interpretación**: El modelo muestra estabilidad entre folds

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

1. **Rendimiento**: El modelo Random Forest logra un F1-Score de 0.9665, superando el objetivo mínimo de 0.75.

2. **Balance Precision/Recall**:
   - Precision: 0.9518 (capacidad de evitar falsos positivos)
   - Recall: 0.9816 (capacidad de detectar todas las transacciones no rentables)

3. **Variables más importantes**: Ver gráfico de feature importance para identificar los principales drivers de rentabilidad.

4. **Próximos pasos sugeridos**:
   - Analizar SHAP values para explicabilidad más profunda
   - Investigar transacciones con alta probabilidad de pérdida
   - Considerar umbrales de decisión alternativos según costo de negocio

---

*Reporte generado automáticamente por ProfitabilityClassifier Agent*
