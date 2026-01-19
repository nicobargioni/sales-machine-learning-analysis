# Customer Segmentation V2 Report

## Executive Summary

Este análisis mejora la segmentación anterior pasando de **2 clusters** a **5 segmentos accionables** para marketing, priorizando interpretabilidad sobre métricas puras.

**Modelo**: K-Means (k=5)
**Silhouette Score**: 0.2513
**Objetivo cumplido**: Sí (objetivo: >0.25)

---

## Metodología

### Cambios vs Versión Anterior

| Aspecto | V1 | V2 |
|---------|----|----|
| Modelo | DBSCAN | K-Means |
| Clusters | 2 | 5 |
| Silhouette | 0.51 | 0.2513 |
| Enfoque | Métricas | Interpretabilidad |

### Preprocesamiento

1. **Log-transform**: Aplicado a monetary, frequency, avg_order_value
2. **Winsorizing**: Percentiles 1-99
3. **StandardScaler**: Normalización Z-score
4. **Features**: recency, frequency_log, monetary_log, avg_order_value_log, avg_profit_scaled

### Selección de K

Se evaluaron k=4 y k=5 con criterios:
- Silhouette Score >= 0.25
- Ningún cluster < 5% del total
- Preferencia por mayor granularidad si ambos válidos

---

## Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Silhouette Score | 0.2513 |
| Davies-Bouldin Index | 1.1722 |
| Calinski-Harabasz Index | 287.69 |
| Min Cluster Size | 7.7% |

---

## Perfiles de Segmentos

### Cluster 0: Low Value

- **Clientes**: 251 (31.7%)
- **Recency promedio**: 102 días
- **Frequency promedio**: 5.6 órdenes
- **Monetary promedio**: $1,269.99
- **Valor por orden**: $250.78
- **Profit promedio**: $27.37

### Cluster 1: Lost

- **Clientes**: 73 (9.2%)
- **Recency promedio**: 578 días
- **Frequency promedio**: 3.7 órdenes
- **Monetary promedio**: $1,671.31
- **Valor por orden**: $467.16
- **Profit promedio**: $17.17

### Cluster 2: Hibernating

- **Clientes**: 61 (7.7%)
- **Recency promedio**: 232 días
- **Frequency promedio**: 3.3 órdenes
- **Monetary promedio**: $230.02
- **Valor por orden**: $70.23
- **Profit promedio**: $8.24

### Cluster 3: Big Spenders

- **Clientes**: 104 (13.1%)
- **Recency promedio**: 124 días
- **Frequency promedio**: 6.4 órdenes
- **Monetary promedio**: $6,598.87
- **Valor por orden**: $1,077.45
- **Profit promedio**: $283.03

### Cluster 4: Loyal Customers

- **Clientes**: 304 (38.3%)
- **Recency promedio**: 74 días
- **Frequency promedio**: 8.1 órdenes
- **Monetary promedio**: $3,803.01
- **Valor por orden**: $498.39
- **Profit promedio**: $18.05

---

## Comparación con V1

La versión anterior usando DBSCAN logró mayor Silhouette (0.51) pero solo generó 2 clusters:
- Cluster 0: 784 clientes (prácticamente todos)
- Cluster 1: 3 clientes (outliers de alto valor)
- Noise: 6 clientes

Esta granularidad era **insuficiente para segmentación de marketing**. La V2 sacrifica algo de cohesión estadística por **mayor accionabilidad comercial**.

---

## Recomendaciones

1. **Usar estos segmentos para campañas diferenciadas** según el archivo `segment_recommendations.md`
2. **Monitorear migración entre segmentos** trimestralmente
3. **Validar con equipo de marketing** que los perfiles coinciden con su conocimiento del negocio
4. **Considerar segmentación adicional** dentro de cada cluster por categoría de producto

---

## Archivos Generados

- `customer_segments_v2.csv`: Dataset con cluster y nombre de segmento
- `clusterer_v2_model.pkl`: Modelo serializado
- `cluster_profiles_v2.png`: Visualización de perfiles
- `segment_recommendations.md`: Recomendaciones de marketing
- `clusterer_v2_results.csv`: Métricas del modelo

---

*Generado por ClustererV2 Agent | 2026-01-19*
