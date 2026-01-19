# 📊 Exploratory Data Analysis - Sales Order Dataset

**Proyecto:** sales_exercise  
**Fecha:** 2026-01-18  
**Dataset:** datawitharyan/sales-order-dataset (Kaggle)

---

## 1. Información General

| Métrica | Valor |
|---------|-------|
| **Filas** | 9,994 |
| **Columnas** | 21 |
| **Memoria** | 9.21 MB |
| **Período** | 2014-01-03 → 2017-12-30 (≈4 años) |
| **País** | United States (único) |
| **Valores Nulos** | 0 ✅ |
| **Duplicados** | 0 ✅ |

---

## 2. Esquema de Datos

### Variables de Identificación
| Columna | Tipo | Cardinalidad |
|---------|------|--------------|
| Row ID | int64 | 9,994 (único) |
| Order ID | object | 5,009 |
| Customer ID | object | 793 |
| Product ID | object | 1,862 |

### Variables Temporales
| Columna | Tipo | Rango |
|---------|------|-------|
| Order Date | datetime | 2014-01-03 → 2017-12-30 |
| Ship Date | datetime | 2014-01-07 → 2018-01-05 |

### Variables Categóricas
| Columna | Cardinalidad | Valores |
|---------|--------------|---------|
| Ship Mode | 4 | Standard Class (60%), Second Class, First Class, Same Day |
| Segment | 3 | Consumer (52%), Corporate (30%), Home Office (18%) |
| Region | 4 | West (32%), East (28%), Central (23%), South (16%) |
| Category | 3 | Office Supplies (60%), Furniture (21%), Technology (18%) |
| Sub-Category | 17 | Binders, Paper, Furnishings, Phones, Storage... |
| State | 49 | California (#1), New York, Texas... |
| City | 531 | New York City (#1), Los Angeles, Philadelphia... |

### Variables Numéricas (Targets Potenciales)
| Columna | Mean | Std | Min | Max | Outliers |
|---------|------|-----|-----|-----|----------|
| **Sales** | $229.86 | $623.25 | $0.44 | $22,638.48 | 11.7% |
| **Profit** | $28.66 | $234.26 | -$6,599.98 | $8,399.98 | 18.8% |
| Quantity | 3.79 | 2.23 | 1 | 14 | 1.7% |
| Discount | 0.16 | 0.21 | 0 | 0.8 | 8.6% |

---

## 3. Análisis de Correlaciones

```
             Sales  Quantity  Discount  Profit
Sales        1.000    0.201    -0.028   0.479
Quantity     0.201    1.000     0.009   0.066
Discount    -0.028    0.009     1.000  -0.219
Profit       0.479    0.066    -0.219   1.000
```

### Hallazgos:
- **Sales ↔ Profit**: Correlación moderada positiva (0.479) - mayor venta, mayor ganancia
- **Discount ↔ Profit**: Correlación negativa (-0.219) - ⚠️ descuentos erosionan ganancia
- **Quantity ↔ Profit**: Correlación débil (0.066) - cantidad no garantiza ganancia

---

## 4. Distribución Temporal

### Órdenes por Año (estimado por fechas)
- **2014**: ~25% de datos
- **2015**: ~25% de datos
- **2016**: ~25% de datos
- **2017**: ~25% de datos

### Ship Modes
| Modo | Cantidad | % |
|------|----------|---|
| Standard Class | 5,968 | 59.7% |
| Second Class | 1,945 | 19.5% |
| First Class | 1,538 | 15.4% |
| Same Day | 543 | 5.4% |

---

## 5. Hallazgos Clave

### ✅ Fortalezas del Dataset
1. **Datos limpios**: Sin nulos ni duplicados
2. **Buena granularidad temporal**: 4 años de datos
3. **Variables ricas**: Mezcla de categóricas, numéricas y temporales
4. **Targets claros**: Sales y Profit son candidatos naturales

### ⚠️ Consideraciones
1. **Outliers significativos**: Profit tiene 18.8% de outliers (valores extremos positivos y negativos)
2. **Profit negativo**: Existen transacciones con pérdida (min = -$6,599.98)
3. **País único**: Solo datos de USA, no hay variabilidad geográfica internacional
4. **Desbalance de categorías**: Office Supplies domina (60%)

### 💡 Oportunidades
1. **Feature Engineering potencial**:
   - Tiempo de envío (Ship Date - Order Date)
   - Margen de ganancia (Profit / Sales)
   - Estacionalidad (mes, trimestre, día de semana)
   - Agregaciones por cliente (CLV, frecuencia)
   
2. **Target Engineering**:
   - Clasificación binaria: `is_profitable = Profit > 0`
   - Clasificación multiclase: `profit_tier` (pérdida, bajo, medio, alto)

---

## 6. Variables Candidatas por Tipo de Problema

### Para Regresión
| Target | Justificación |
|--------|---------------|
| **Profit** | Variable continua de alto interés de negocio |
| Sales | Variable continua, más simple de predecir |

### Para Clasificación
| Target | Construcción | Justificación |
|--------|--------------|---------------|
| **is_profitable** | `Profit > 0` → 1, else 0 | Binario, detectar transacciones rentables |
| profit_tier | Bins de Profit | Multiclase, segmentar niveles de rentabilidad |

### Para Forecasting
| Target | Agregación | Justificación |
|--------|------------|---------------|
| **Sales diarias/mensuales** | Suma por fecha | Predicción de demanda |
| Profit temporal | Suma por fecha | Proyección de rentabilidad |

### Para Clustering
| Enfoque | Variables | Justificación |
|---------|-----------|---------------|
| **Segmentación de clientes** | RFM (Recency, Frequency, Monetary) | Marketing dirigido |
| Segmentación de productos | Sales, Profit, Quantity | Portfolio analysis |

---

## 7. Recomendaciones para Ejercicios ML

### 🎯 Ejercicios Recomendados (por complejidad)

| # | Tipo | Target | Complejidad | Valor de Negocio |
|---|------|--------|-------------|------------------|
| 1 | **Regresión** | Profit | Media | ⭐⭐⭐⭐⭐ |
| 2 | **Clasificación** | is_profitable | Baja | ⭐⭐⭐⭐ |
| 3 | **Forecasting** | Sales mensuales | Alta | ⭐⭐⭐⭐⭐ |
| 4 | **Clustering** | Clientes (RFM) | Media | ⭐⭐⭐⭐ |

---

*Generado automáticamente por Ralph Discovery Agent*
