"""
Feature Engineering Pipeline - Sales Exercise
Agent: Features
Date: 2026-01-18

Genera datasets preprocesados para todos los agentes de ML:
- ProfitRegressor
- ProfitabilityClassifier
- SalesForecaster
- CustomerSegmenter
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
AGENT_CONTEXT_DIR = BASE_DIR / 'agent_context'

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Carga el dataset crudo."""
    df = pd.read_csv(DATA_DIR / 'Sales_csv.csv')

    # Parsear fechas (formato: DD-Mon-YY, ej: 17-Jul-16)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%b-%y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%b-%y')

    return df


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features base compartidas por todos los agentes."""
    df = df.copy()

    # Features temporales
    df['ship_days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['order_month'] = df['Order Date'].dt.month
    df['order_quarter'] = df['Order Date'].dt.quarter
    df['order_dayofweek'] = df['Order Date'].dt.dayofweek
    df['order_year'] = df['Order Date'].dt.year
    df['order_day'] = df['Order Date'].dt.day

    return df


def create_regressor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features específicas para ProfitRegressor."""
    df = df.copy()

    # Feature engineering según plan
    df['margin_potential'] = df['Sales'] * (1 - df['Discount'])

    # Columnas a eliminar según plan (evitar leakage)
    cols_to_drop = ['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
                    'Product ID', 'Product Name', 'Country',
                    'Order Date', 'Ship Date']

    df_regressor = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df_regressor


def create_classifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features específicas para ProfitabilityClassifier."""
    df = df.copy()

    # Target binario
    df['is_profitable'] = (df['Profit'] > 0).astype(int)

    # Features adicionales
    df['high_discount'] = (df['Discount'] > 0.2).astype(int)
    df['is_standard_shipping'] = (df['Ship Mode'] == 'Standard Class').astype(int)

    # Columnas a eliminar (Profit es el target original, no puede ser feature)
    cols_to_drop = ['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
                    'Product ID', 'Product Name', 'Country',
                    'Order Date', 'Ship Date', 'Profit']

    df_classifier = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df_classifier


def create_forecaster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features específicas para SalesForecaster (serie temporal mensual)."""
    df = df.copy()

    # Agregar por mes
    df['year_month'] = df['Order Date'].dt.to_period('M')
    monthly = df.groupby('year_month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': 'nunique'  # Número de órdenes
    }).reset_index()

    monthly.columns = ['year_month', 'sales', 'profit', 'quantity', 'n_orders']
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values('year_month').reset_index(drop=True)

    # Features temporales
    monthly['month'] = monthly['year_month'].dt.month
    monthly['quarter'] = monthly['year_month'].dt.quarter
    monthly['year'] = monthly['year_month'].dt.year

    # Lags
    for lag in [1, 2, 3]:
        monthly[f'sales_lag_{lag}'] = monthly['sales'].shift(lag)

    # Rolling features
    monthly['sales_rolling_mean_3'] = monthly['sales'].shift(1).rolling(3).mean()
    monthly['sales_rolling_mean_6'] = monthly['sales'].shift(1).rolling(6).mean()
    monthly['sales_rolling_std_3'] = monthly['sales'].shift(1).rolling(3).std()

    return monthly


def create_segmenter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features RFM para CustomerSegmenter."""
    df = df.copy()

    # Fecha de referencia (última fecha del dataset + 1 día)
    reference_date = df['Order Date'].max() + pd.Timedelta(days=1)

    # Agregación por cliente
    customer_agg = df.groupby('Customer ID').agg({
        'Order Date': ['max', 'min'],
        'Order ID': 'nunique',
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'Segment': 'first',
        'Region': 'first',
        'State': 'first'
    })

    # Aplanar columnas
    customer_agg.columns = ['last_order', 'first_order', 'frequency',
                           'monetary', 'total_profit', 'total_quantity',
                           'preferred_category', 'segment', 'region', 'state']
    customer_agg = customer_agg.reset_index()

    # RFM
    customer_agg['recency'] = (reference_date - customer_agg['last_order']).dt.days

    # Features derivadas
    customer_agg['avg_order_value'] = customer_agg['monetary'] / customer_agg['frequency']
    customer_agg['avg_profit'] = customer_agg['total_profit'] / customer_agg['frequency']
    customer_agg['days_as_customer'] = (customer_agg['last_order'] - customer_agg['first_order']).dt.days
    customer_agg['avg_quantity'] = customer_agg['total_quantity'] / customer_agg['frequency']

    # Eliminar columnas de fecha
    customer_agg = customer_agg.drop(columns=['last_order', 'first_order'])

    return customer_agg


def save_datasets(df_base: pd.DataFrame,
                  df_regressor: pd.DataFrame,
                  df_classifier: pd.DataFrame,
                  df_forecaster: pd.DataFrame,
                  df_segmenter: pd.DataFrame) -> dict:
    """Guarda todos los datasets procesados."""

    stats = {}

    # Dataset base con todas las features
    df_base.to_csv(OUTPUT_DIR / 'features_base.csv', index=False)
    stats['base'] = {'rows': len(df_base), 'cols': len(df_base.columns)}

    # Dataset para regresión
    df_regressor.to_csv(OUTPUT_DIR / 'features_regressor.csv', index=False)
    stats['regressor'] = {'rows': len(df_regressor), 'cols': len(df_regressor.columns)}

    # Dataset para clasificación
    df_classifier.to_csv(OUTPUT_DIR / 'features_classifier.csv', index=False)
    stats['classifier'] = {'rows': len(df_classifier), 'cols': len(df_classifier.columns)}

    # Dataset para forecasting
    df_forecaster.to_csv(OUTPUT_DIR / 'features_forecaster.csv', index=False)
    stats['forecaster'] = {'rows': len(df_forecaster), 'cols': len(df_forecaster.columns)}

    # Dataset para segmentación
    df_segmenter.to_csv(OUTPUT_DIR / 'features_segmenter.csv', index=False)
    stats['segmenter'] = {'rows': len(df_segmenter), 'cols': len(df_segmenter.columns)}

    return stats


def save_feature_metadata(df_regressor: pd.DataFrame,
                          df_classifier: pd.DataFrame,
                          df_forecaster: pd.DataFrame,
                          df_segmenter: pd.DataFrame) -> dict:
    """Guarda metadatos de features para cada agente."""

    metadata = {
        'regressor': {
            'features': list(df_regressor.columns),
            'target': 'Profit',
            'categorical': ['Ship Mode', 'Segment', 'Region', 'Category',
                          'Sub-Category', 'State', 'City'],
            'numerical': ['Sales', 'Quantity', 'Discount', 'Postal Code',
                         'ship_days', 'order_month', 'order_quarter',
                         'order_dayofweek', 'order_year', 'order_day',
                         'margin_potential']
        },
        'classifier': {
            'features': list(df_classifier.columns),
            'target': 'is_profitable',
            'categorical': ['Ship Mode', 'Segment', 'Region', 'Category',
                          'Sub-Category', 'State', 'City'],
            'numerical': ['Sales', 'Quantity', 'Discount', 'Postal Code',
                         'ship_days', 'order_month', 'order_quarter',
                         'order_dayofweek', 'order_year', 'order_day',
                         'high_discount', 'is_standard_shipping']
        },
        'forecaster': {
            'features': list(df_forecaster.columns),
            'target': 'sales',
            'date_column': 'year_month'
        },
        'segmenter': {
            'features': list(df_segmenter.columns),
            'rfm_features': ['recency', 'frequency', 'monetary'],
            'customer_id': 'Customer ID'
        }
    }

    with open(OUTPUT_DIR / 'features_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def update_agent_state(stats: dict, metadata: dict):
    """Actualiza el estado del agente Features."""

    state = {
        'agent': 'Features',
        'status': 'completed',
        'last_run': datetime.now().isoformat(),
        'datasets_generated': list(stats.keys()),
        'stats': stats,
        'outputs': {
            'base': 'output/features_base.csv',
            'regressor': 'output/features_regressor.csv',
            'classifier': 'output/features_classifier.csv',
            'forecaster': 'output/features_forecaster.csv',
            'segmenter': 'output/features_segmenter.csv',
            'metadata': 'output/features_metadata.json'
        },
        'features_created': {
            'base': ['ship_days', 'order_month', 'order_quarter',
                    'order_dayofweek', 'order_year', 'order_day'],
            'regressor': ['margin_potential'],
            'classifier': ['is_profitable', 'high_discount', 'is_standard_shipping'],
            'forecaster': ['sales_lag_1', 'sales_lag_2', 'sales_lag_3',
                          'sales_rolling_mean_3', 'sales_rolling_mean_6',
                          'sales_rolling_std_3'],
            'segmenter': ['recency', 'frequency', 'monetary', 'avg_order_value',
                         'avg_profit', 'days_as_customer', 'avg_quantity']
        }
    }

    with open(AGENT_CONTEXT_DIR / 'Features_state.json', 'w') as f:
        json.dump(state, f, indent=2)

    return state


def main():
    """Pipeline principal de Feature Engineering."""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE - Sales Exercise")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/6] Cargando dataset...")
    df = load_data()
    print(f"    Filas: {len(df):,}")
    print(f"    Columnas: {len(df.columns)}")
    print(f"    Período: {df['Order Date'].min()} → {df['Order Date'].max()}")

    # 2. Features base
    print("\n[2/6] Creando features base...")
    df_base = create_base_features(df)
    print(f"    Features temporales agregadas: ship_days, order_month, etc.")

    # 3. Features por agente
    print("\n[3/6] Creando features para ProfitRegressor...")
    df_regressor = create_regressor_features(df_base)
    print(f"    Dataset: {len(df_regressor)} filas x {len(df_regressor.columns)} cols")

    print("\n[4/6] Creando features para ProfitabilityClassifier...")
    df_classifier = create_classifier_features(df_base)
    print(f"    Dataset: {len(df_classifier)} filas x {len(df_classifier.columns)} cols")
    profitable_pct = df_classifier['is_profitable'].mean() * 100
    print(f"    Balance de clases: {profitable_pct:.1f}% rentables")

    print("\n[5/6] Creando features para SalesForecaster...")
    df_forecaster = create_forecaster_features(df)
    print(f"    Dataset: {len(df_forecaster)} filas x {len(df_forecaster.columns)} cols")
    print(f"    Período mensual: {df_forecaster['year_month'].min()} → {df_forecaster['year_month'].max()}")

    print("\n[6/6] Creando features para CustomerSegmenter...")
    df_segmenter = create_segmenter_features(df)
    print(f"    Dataset: {len(df_segmenter)} clientes x {len(df_segmenter.columns)} cols")

    # 4. Guardar datasets
    print("\n" + "-" * 60)
    print("Guardando datasets...")
    stats = save_datasets(df_base, df_regressor, df_classifier,
                         df_forecaster, df_segmenter)

    # 5. Guardar metadatos
    print("Guardando metadatos...")
    metadata = save_feature_metadata(df_regressor, df_classifier,
                                     df_forecaster, df_segmenter)

    # 6. Actualizar estado del agente
    print("Actualizando estado del agente...")
    state = update_agent_state(stats, metadata)

    # Resumen
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print("\nDatasets generados:")
    for name, s in stats.items():
        print(f"  - {name}: {s['rows']:,} filas x {s['cols']} columnas")

    print(f"\nArchivos en: {OUTPUT_DIR}")
    print(f"Estado en: {AGENT_CONTEXT_DIR / 'Features_state.json'}")

    return state


if __name__ == '__main__':
    main()
