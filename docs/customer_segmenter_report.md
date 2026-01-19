# Customer Segmentation Report

## Executive Summary

Este análisis segmenta la base de **793 clientes** utilizando análisis RFM (Recency, Frequency, Monetary) y técnicas de clustering no supervisado.

**Mejor modelo**: DBSCAN
**Silhouette Score**: 0.5064
**Número de segmentos**: 2

---

## Metodología

### 1. Construcción de Features RFM

Se agregaron las transacciones por cliente para crear:

| Feature | Descripción |
|---------|-------------|
| Recency | Días desde la última compra |
| Frequency | Número de órdenes únicas |
| Monetary | Suma total de ventas ($) |
| avg_order_value | Valor promedio por orden |
| avg_profit | Profit promedio por orden |

### 2. Preprocesamiento

- **Log-transform**: Aplicado a Monetary, Frequency y avg_order_value para reducir skewness
- **Winsorizing**: Percentiles 1-99 para mitigar outliers extremos
- **StandardScaler**: Normalización Z-score para clustering

### 3. Modelos Evaluados

| Modelo | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|------------|----------------|-------------------|
| DBSCAN | 0.5064 | 0.5026 | 18.27 |
| KMeans | 0.2878 | 1.1704 | 295.81 |
| KMeans_PCA | 0.2857 | 1.1766 | 296.16 |
| GMM | 0.2458 | 2.8867 | 97.26 |
| Hierarchical | 0.2314 | 1.2994 | 246.66 |


---

## Perfiles de Segmentos

### Cluster 0: Loyal Customers

- **Clientes**: 784
- **Recency promedio**: 143.0 días
- **Frequency promedio**: 6.4 órdenes
- **Monetary promedio**: $2856.75
- **Valor por orden**: $448.60
- **Profit promedio**: $52.63

### Cluster 1: VIP Whales (Outliers)

- **Clientes**: 3
- **Recency promedio**: 436.3 días
- **Frequency promedio**: 6.3 órdenes
- **Monetary promedio**: $14565.40
- **Valor por orden**: $2494.05
- **Profit promedio**: $986.40

> **Nota**: Estos 3 clientes representan outliers de alto valor (5x el monetary promedio). Aunque DBSCAN los clasificó correctamente como diferentes del grupo principal, su alta recency sugiere que son cuentas corporativas o compradores ocasionales de alto volumen.

### Noise Points (6 clientes)

DBSCAN identificó 6 clientes como "ruido" que no pertenecen claramente a ningún cluster. Estos deberían analizarse individualmente.

---

## Interpretación y Limitaciones

### Sobre la elección de DBSCAN

- **Ventaja**: Mayor Silhouette Score (0.51) indica clusters más cohesivos y separados
- **Limitación**: Solo identifica 2 clusters + noise, reduciendo granularidad para segmentación de marketing
- **Alternativa recomendada**: Para campañas de marketing, considerar K-Means (k=4) que genera 4 segmentos accionables aunque con menor Silhouette (0.29)

---

## Recomendaciones de Negocio

Basado en los segmentos identificados:

1. **Loyal Customers (784 clientes)**:
   - Programa de retención
   - Incentivos de referidos
   - Ofertas personalizadas según preferred_category

2. **VIP Whales (3 clientes)**:
   - Atención premium personalizada
   - Campañas de reactivación urgentes (alta recency)
   - Análisis individual de necesidades

---

## Archivos Generados

- `customer_segments.csv`: Dataset con cluster asignado por cliente
- `customer_segmenter_model.pkl`: Modelo de clustering serializado
- `rfm_distribution.png`: Distribución de variables RFM
- `elbow_silhouette_plot.png`: Análisis de K óptimo
- `cluster_profiles.png`: Visualización de perfiles

---

*Generado por CustomerSegmenter Agent | 2026-01-18 13:48:21*
