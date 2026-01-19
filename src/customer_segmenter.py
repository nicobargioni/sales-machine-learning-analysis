"""
CustomerSegmenter Agent - RFM Analysis & Clustering
Sales Exercise Project

Este agente segmenta clientes usando análisis RFM y clustering no supervisado.
Compara: K-Means, K-Means+PCA, DBSCAN, Hierarchical, GMM

Autor: CustomerSegmenter Agent (Ralph Wiggum Protocol)
Fecha: 2026-01-18
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

warnings.filterwarnings('ignore')

# Configuración de paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "Sales_csv.csv"
OUTPUT_DIR = BASE_DIR / "output"
DOCS_DIR = BASE_DIR / "docs"
AGENT_CONTEXT_DIR = BASE_DIR / "agent_context"
STATE_FILE = AGENT_CONTEXT_DIR / "CustomerSegmenter_state.json"

# Configuración de recursos (3 agentes activos)
N_AGENTS = 3
N_CORES = os.cpu_count() or 4
N_JOBS = max(1, N_CORES // N_AGENTS)

RANDOM_STATE = 42

# Paleta de colores para visualización
CLUSTER_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']


def update_state(state_dict: dict):
    """Persiste el estado del agente para recuperación."""
    state_dict['last_update'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state_dict, f, indent=2)


def load_state() -> dict:
    """Carga el estado previo del agente."""
    default_state = {
        "agent": "CustomerSegmenter",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "current_phase": "initialization",
        "completed_phases": [],
        "best_model": None,
        "best_silhouette": None,
        "n_clusters": None,
        "errors": []
    }
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            loaded = json.load(f)
            # Merge with defaults to ensure all keys exist
            for key, value in default_state.items():
                if key not in loaded:
                    loaded[key] = value
            return loaded
    return default_state


def load_data() -> pd.DataFrame:
    """Carga el dataset de ventas."""
    print(f"[INFO] Cargando dataset desde {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%b-%y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%b-%y')
    print(f"[INFO] Dataset cargado: {len(df)} registros, {df['Customer ID'].nunique()} clientes únicos")
    return df


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features RFM (Recency, Frequency, Monetary) agregando por cliente.
    También genera features adicionales: avg_order_value, avg_profit, days_as_customer
    """
    print("[INFO] Creando features RFM...")

    # Fecha de referencia: última fecha del dataset + 1 día
    reference_date = df['Order Date'].max() + timedelta(days=1)
    print(f"[INFO] Fecha de referencia para Recency: {reference_date}")

    # Agregación por cliente
    rfm = df.groupby('Customer ID').agg({
        'Order Date': ['max', 'min', 'nunique'],  # última compra, primera compra, n_orders
        'Order ID': 'nunique',                     # Frequency (órdenes únicas)
        'Sales': 'sum',                            # Monetary
        'Profit': 'sum',                           # Para avg_profit
        'Quantity': 'sum',                         # Total items comprados
        'Discount': 'mean',                        # Descuento promedio
        'Segment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Segmento predominante
        'Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0], # Categoría preferida
        'Region': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],   # Región
    }).reset_index()

    # Aplanar columnas multi-nivel
    rfm.columns = [
        'Customer ID', 'last_purchase', 'first_purchase', 'purchase_dates_count',
        'Frequency', 'Monetary', 'total_profit', 'total_quantity',
        'avg_discount', 'preferred_segment', 'preferred_category', 'region'
    ]

    # Calcular Recency (días desde última compra)
    rfm['Recency'] = (reference_date - rfm['last_purchase']).dt.days

    # Features derivadas
    rfm['avg_order_value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['avg_profit'] = rfm['total_profit'] / rfm['Frequency']
    rfm['days_as_customer'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
    rfm['avg_items_per_order'] = rfm['total_quantity'] / rfm['Frequency']
    rfm['profit_margin'] = rfm['total_profit'] / rfm['Monetary'].replace(0, np.nan)

    # Limpiar valores infinitos o NaN
    rfm['profit_margin'] = rfm['profit_margin'].fillna(0)
    rfm['avg_profit'] = rfm['avg_profit'].fillna(0)

    print(f"[INFO] Features RFM creadas para {len(rfm)} clientes")
    print(f"[INFO] Columnas: {list(rfm.columns)}")

    return rfm


def analyze_rfm_distribution(rfm: pd.DataFrame, output_dir: Path):
    """Analiza y visualiza la distribución de variables RFM."""
    print("[INFO] Analizando distribución de RFM...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Variables principales RFM
    main_vars = ['Recency', 'Frequency', 'Monetary', 'avg_order_value', 'avg_profit', 'days_as_customer']

    for i, var in enumerate(main_vars):
        ax = axes[i // 3, i % 3]
        data = rfm[var].dropna()

        # Histograma con KDE
        ax.hist(data, bins=30, edgecolor='white', alpha=0.7, color='#3498db')
        ax.set_title(f'{var}\n(skew={stats.skew(data):.2f})', fontsize=12)
        ax.set_xlabel(var)
        ax.set_ylabel('Frecuencia')

        # Añadir estadísticas
        stats_text = f'μ={data.mean():.1f}\nσ={data.std():.1f}\nMed={data.median():.1f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Distribución de Variables RFM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rfm_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Gráfico guardado: {output_dir / 'rfm_distribution.png'}")

    return {var: {'mean': rfm[var].mean(), 'std': rfm[var].std(), 'skew': stats.skew(rfm[var].dropna())}
            for var in main_vars}


def preprocess_for_clustering(rfm: pd.DataFrame) -> tuple:
    """
    Preprocesa datos para clustering:
    1. Selecciona features numéricas relevantes
    2. Aplica log-transform para reducir skewness
    3. Escala con StandardScaler
    """
    print("[INFO] Preprocesando datos para clustering...")

    # Features para clustering (solo numéricas continuas)
    clustering_features = ['Recency', 'Frequency', 'Monetary', 'avg_order_value', 'avg_profit']

    X_raw = rfm[clustering_features].copy()

    # Log-transform para variables con alta skewness (Monetary, avg_order_value típicamente)
    # Usar log1p para manejar valores cercanos a 0
    for col in ['Monetary', 'avg_order_value', 'Frequency']:
        if X_raw[col].min() >= 0:
            X_raw[f'{col}_log'] = np.log1p(X_raw[col])
        else:
            # Para valores negativos, usamos signed log
            X_raw[f'{col}_log'] = np.sign(X_raw[col]) * np.log1p(np.abs(X_raw[col]))

    # Features finales para clustering
    final_features = ['Recency', 'Frequency_log', 'Monetary_log', 'avg_order_value_log', 'avg_profit']
    X = X_raw[final_features].copy()

    # Manejar outliers con winsorizing (percentil 1-99)
    for col in X.columns:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[INFO] Features para clustering: {final_features}")
    print(f"[INFO] Shape final: {X_scaled.shape}")

    return X_scaled, scaler, final_features


def find_optimal_k_kmeans(X: np.ndarray, k_range: range) -> dict:
    """
    Encuentra el número óptimo de clusters usando Elbow Method y Silhouette.
    """
    print(f"[INFO] Buscando K óptimo en rango {list(k_range)}...")

    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X)

        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))

        print(f"  k={k}: Silhouette={results['silhouette'][-1]:.4f}, DB={results['davies_bouldin'][-1]:.4f}")

    return results


def plot_elbow_silhouette(results: dict, output_dir: Path):
    """Grafica Elbow Method y Silhouette Score."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow (Inertia)
    axes[0].plot(results['k'], results['inertia'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Número de Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)

    # Silhouette Score
    axes[1].plot(results['k'], results['silhouette'], 'go-', linewidth=2)
    axes[1].set_xlabel('Número de Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)

    # Marcar el mejor k
    best_k_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_k_idx]
    axes[1].axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    axes[1].legend()

    # Davies-Bouldin (menor es mejor)
    axes[2].plot(results['k'], results['davies_bouldin'], 'ro-', linewidth=2)
    axes[2].set_xlabel('Número de Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin Index (menor = mejor)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Análisis de Número Óptimo de Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'elbow_silhouette_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Gráfico guardado: {output_dir / 'elbow_silhouette_plot.png'}")


def train_and_compare_models(X: np.ndarray, k_optimal: int) -> dict:
    """
    Entrena y compara múltiples modelos de clustering.
    Retorna diccionario con resultados de cada modelo.
    """
    print(f"\n[INFO] Entrenando modelos con k={k_optimal}...")

    models_results = {}

    # 1. K-Means (baseline)
    print("  Training K-Means...")
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    labels_kmeans = kmeans.fit_predict(X)
    models_results['KMeans'] = {
        'model': kmeans,
        'labels': labels_kmeans,
        'silhouette': silhouette_score(X, labels_kmeans),
        'davies_bouldin': davies_bouldin_score(X, labels_kmeans),
        'calinski_harabasz': calinski_harabasz_score(X, labels_kmeans)
    }
    print(f"    Silhouette: {models_results['KMeans']['silhouette']:.4f}")

    # 2. K-Means + PCA (2 componentes para visualización, más para clustering)
    print("  Training K-Means + PCA...")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # 95% varianza
    X_pca = pca.fit_transform(X)
    print(f"    PCA components: {pca.n_components_} (explained variance: {pca.explained_variance_ratio_.sum():.2%})")

    kmeans_pca = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    labels_kmeans_pca = kmeans_pca.fit_predict(X_pca)
    models_results['KMeans_PCA'] = {
        'model': kmeans_pca,
        'pca': pca,
        'labels': labels_kmeans_pca,
        'silhouette': silhouette_score(X_pca, labels_kmeans_pca),
        'davies_bouldin': davies_bouldin_score(X_pca, labels_kmeans_pca),
        'calinski_harabasz': calinski_harabasz_score(X_pca, labels_kmeans_pca)
    }
    print(f"    Silhouette: {models_results['KMeans_PCA']['silhouette']:.4f}")

    # 3. DBSCAN - buscar eps óptimo
    print("  Training DBSCAN...")
    best_dbscan = None
    best_dbscan_score = -1
    best_eps = None

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_samples in [3, 5, 10]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=N_JOBS)
            labels_dbscan = dbscan.fit_predict(X)
            n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

            if n_clusters >= 2 and n_clusters <= 10:
                # Excluir noise points para silhouette
                mask = labels_dbscan != -1
                if mask.sum() > n_clusters:
                    score = silhouette_score(X[mask], labels_dbscan[mask])
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan = dbscan
                        best_eps = eps

    if best_dbscan is not None:
        labels_dbscan = best_dbscan.fit_predict(X)
        mask = labels_dbscan != -1
        models_results['DBSCAN'] = {
            'model': best_dbscan,
            'labels': labels_dbscan,
            'silhouette': best_dbscan_score,
            'davies_bouldin': davies_bouldin_score(X[mask], labels_dbscan[mask]) if mask.sum() > 2 else np.nan,
            'calinski_harabasz': calinski_harabasz_score(X[mask], labels_dbscan[mask]) if mask.sum() > 2 else np.nan,
            'n_noise': (labels_dbscan == -1).sum(),
            'eps': best_eps
        }
        print(f"    Silhouette: {models_results['DBSCAN']['silhouette']:.4f}, Noise: {models_results['DBSCAN']['n_noise']}")
    else:
        print("    DBSCAN: No valid clustering found")

    # 4. Hierarchical Clustering
    print("  Training Hierarchical (Agglomerative)...")
    hierarchical = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
    labels_hier = hierarchical.fit_predict(X)
    models_results['Hierarchical'] = {
        'model': hierarchical,
        'labels': labels_hier,
        'silhouette': silhouette_score(X, labels_hier),
        'davies_bouldin': davies_bouldin_score(X, labels_hier),
        'calinski_harabasz': calinski_harabasz_score(X, labels_hier)
    }
    print(f"    Silhouette: {models_results['Hierarchical']['silhouette']:.4f}")

    # 5. Gaussian Mixture Model
    print("  Training GMM...")
    gmm = GaussianMixture(n_components=k_optimal, covariance_type='full', random_state=RANDOM_STATE, n_init=5)
    labels_gmm = gmm.fit_predict(X)
    models_results['GMM'] = {
        'model': gmm,
        'labels': labels_gmm,
        'silhouette': silhouette_score(X, labels_gmm),
        'davies_bouldin': davies_bouldin_score(X, labels_gmm),
        'calinski_harabasz': calinski_harabasz_score(X, labels_gmm),
        'bic': gmm.bic(X),
        'aic': gmm.aic(X)
    }
    print(f"    Silhouette: {models_results['GMM']['silhouette']:.4f}")

    return models_results


def select_best_model(models_results: dict) -> tuple:
    """Selecciona el mejor modelo basado en Silhouette Score."""
    print("\n[INFO] Comparando modelos...")

    comparison = []
    for name, result in models_results.items():
        comparison.append({
            'Model': name,
            'Silhouette': result['silhouette'],
            'Davies_Bouldin': result.get('davies_bouldin', np.nan),
            'Calinski_Harabasz': result.get('calinski_harabasz', np.nan)
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('Silhouette', ascending=False)
    print("\n" + comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models_results[best_model_name]

    print(f"\n[INFO] Mejor modelo: {best_model_name} (Silhouette={best_model['silhouette']:.4f})")

    return best_model_name, best_model, comparison_df


def create_cluster_profiles(rfm: pd.DataFrame, labels: np.ndarray, model_name: str) -> pd.DataFrame:
    """Crea perfiles descriptivos de cada cluster."""
    print("\n[INFO] Creando perfiles de clusters...")

    rfm_clustered = rfm.copy()
    rfm_clustered['Cluster'] = labels

    # Excluir noise points de DBSCAN si existen
    if -1 in labels:
        rfm_valid = rfm_clustered[rfm_clustered['Cluster'] != -1]
    else:
        rfm_valid = rfm_clustered

    # Métricas por cluster
    profile_metrics = ['Recency', 'Frequency', 'Monetary', 'avg_order_value', 'avg_profit', 'days_as_customer']

    profiles = rfm_valid.groupby('Cluster')[profile_metrics].agg(['mean', 'median', 'std', 'count'])

    # Simplificar para visualización
    profile_summary = rfm_valid.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'avg_order_value': 'mean',
        'avg_profit': 'mean',
        'days_as_customer': 'mean',
        'Customer ID': 'count'
    }).rename(columns={'Customer ID': 'n_customers'})

    # Normalizar para comparación
    profile_normalized = profile_summary.copy()
    for col in profile_metrics:
        col_min = profile_summary[col].min()
        col_max = profile_summary[col].max()
        if col_max > col_min:
            profile_normalized[col] = (profile_summary[col] - col_min) / (col_max - col_min)

    # Asignar nombres de negocio basados en características
    cluster_names = []
    for idx, row in profile_summary.iterrows():
        # Lógica de naming basada en RFM
        r_score = 1 if row['Recency'] < profile_summary['Recency'].median() else 0
        f_score = 1 if row['Frequency'] > profile_summary['Frequency'].median() else 0
        m_score = 1 if row['Monetary'] > profile_summary['Monetary'].median() else 0

        if r_score and f_score and m_score:
            name = "Champions"
        elif r_score and f_score:
            name = "Loyal Customers"
        elif r_score and m_score:
            name = "Big Spenders"
        elif not r_score and f_score and m_score:
            name = "At Risk - High Value"
        elif not r_score and not f_score:
            name = "Lost/Inactive"
        elif r_score:
            name = "Recent Customers"
        else:
            name = f"Segment {idx}"

        cluster_names.append(name)

    profile_summary['Cluster_Name'] = cluster_names

    print("\n--- Perfiles de Clusters ---")
    print(profile_summary.round(2).to_string())

    return rfm_clustered, profile_summary


def plot_cluster_profiles(profile_summary: pd.DataFrame, rfm_clustered: pd.DataFrame, output_dir: Path):
    """Visualiza los perfiles de clusters."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Radar chart de perfiles (usando métricas normalizadas)
    ax1 = fig.add_subplot(2, 2, 1, polar=True)

    metrics = ['Recency', 'Frequency', 'Monetary', 'avg_order_value', 'avg_profit']
    n_metrics = len(metrics)
    angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]  # Cerrar el polígono

    # Normalizar para el radar
    profile_norm = profile_summary.copy()
    for col in metrics:
        col_min = profile_summary[col].min()
        col_max = profile_summary[col].max()
        if col_max > col_min:
            profile_norm[col] = (profile_summary[col] - col_min) / (col_max - col_min)
        else:
            profile_norm[col] = 0.5

    for idx, (cluster, row) in enumerate(profile_norm.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=f"C{cluster}: {row['Cluster_Name']}",
                 color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
        ax1.fill(angles, values, alpha=0.1, color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_title('Perfiles de Clusters (Normalizado)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    # 2. Distribución de clientes por cluster
    ax2 = fig.add_subplot(2, 2, 2)
    valid_clusters = rfm_clustered[rfm_clustered['Cluster'] != -1]['Cluster']
    cluster_counts = valid_clusters.value_counts().sort_index()
    bars = ax2.bar(cluster_counts.index, cluster_counts.values,
                   color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in cluster_counts.index])
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Número de Clientes')
    ax2.set_title('Distribución de Clientes por Cluster')

    # Añadir labels con nombres
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        name = profile_summary.loc[cluster, 'Cluster_Name'] if cluster in profile_summary.index else f"C{cluster}"
        ax2.text(cluster, count + 5, f'{name}\n({count})', ha='center', fontsize=8)

    # 3. Scatter RFM (Recency vs Monetary, coloreado por cluster)
    ax3 = fig.add_subplot(2, 2, 3)
    valid_data = rfm_clustered[rfm_clustered['Cluster'] != -1]
    scatter = ax3.scatter(valid_data['Recency'], valid_data['Monetary'],
                          c=valid_data['Cluster'], cmap='Set1', alpha=0.6, s=30)
    ax3.set_xlabel('Recency (días)')
    ax3.set_ylabel('Monetary ($)')
    ax3.set_title('Recency vs Monetary por Cluster')
    plt.colorbar(scatter, ax=ax3, label='Cluster')

    # 4. Box plots por cluster
    ax4 = fig.add_subplot(2, 2, 4)
    valid_data.boxplot(column='Monetary', by='Cluster', ax=ax4)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Monetary ($)')
    ax4.set_title('Distribución de Monetary por Cluster')
    plt.suptitle('')  # Remover título automático de boxplot

    plt.suptitle('Análisis de Segmentos de Clientes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Gráfico guardado: {output_dir / 'cluster_profiles.png'}")


def generate_report(rfm_clustered: pd.DataFrame, profile_summary: pd.DataFrame,
                    comparison_df: pd.DataFrame, best_model_name: str,
                    best_silhouette: float, docs_dir: Path):
    """Genera el reporte técnico en Markdown."""
    print("\n[INFO] Generando reporte técnico...")

    report = f"""# Customer Segmentation Report

## Executive Summary

Este análisis segmenta la base de **{len(rfm_clustered)} clientes** utilizando análisis RFM (Recency, Frequency, Monetary) y técnicas de clustering no supervisado.

**Mejor modelo**: {best_model_name}
**Silhouette Score**: {best_silhouette:.4f}
**Número de segmentos**: {profile_summary.shape[0]}

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
"""

    for _, row in comparison_df.iterrows():
        report += f"| {row['Model']} | {row['Silhouette']:.4f} | {row['Davies_Bouldin']:.4f} | {row['Calinski_Harabasz']:.2f} |\n"

    report += f"""

---

## Perfiles de Segmentos

"""

    for cluster, row in profile_summary.iterrows():
        report += f"""### Cluster {cluster}: {row['Cluster_Name']}

- **Clientes**: {int(row['n_customers'])}
- **Recency promedio**: {row['Recency']:.1f} días
- **Frequency promedio**: {row['Frequency']:.1f} órdenes
- **Monetary promedio**: ${row['Monetary']:.2f}
- **Valor por orden**: ${row['avg_order_value']:.2f}
- **Profit promedio**: ${row['avg_profit']:.2f}

"""

    report += f"""---

## Recomendaciones de Negocio

Basado en los segmentos identificados:

1. **Champions**: Programa de lealtad premium, acceso anticipado a nuevos productos
2. **Loyal Customers**: Incentivos de upselling, referidos
3. **Big Spenders**: Atención personalizada, ofertas exclusivas
4. **At Risk - High Value**: Campañas de reactivación urgentes
5. **Lost/Inactive**: Win-back campaigns, descuentos especiales

---

## Archivos Generados

- `customer_segments.csv`: Dataset con cluster asignado por cliente
- `customer_segmenter_model.pkl`: Modelo de clustering serializado
- `rfm_distribution.png`: Distribución de variables RFM
- `elbow_silhouette_plot.png`: Análisis de K óptimo
- `cluster_profiles.png`: Visualización de perfiles

---

*Generado por CustomerSegmenter Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    report_path = docs_dir / 'customer_segmenter_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"[INFO] Reporte guardado: {report_path}")


def main():
    """Pipeline principal de segmentación de clientes."""
    print("=" * 60)
    print("CustomerSegmenter Agent - Starting")
    print(f"n_jobs = {N_JOBS} (compartiendo con {N_AGENTS} agentes)")
    print("=" * 60)

    # Cargar estado
    state = load_state()
    state['status'] = 'running'
    state['current_phase'] = 'loading_data'
    update_state(state)

    try:
        # 1. Cargar datos
        df = load_data()
        state['completed_phases'].append('data_loaded')
        state['current_phase'] = 'rfm_creation'
        update_state(state)

        # 2. Crear features RFM
        rfm = create_rfm_features(df)
        state['completed_phases'].append('rfm_created')
        state['current_phase'] = 'distribution_analysis'
        update_state(state)

        # 3. Analizar distribución
        rfm_stats = analyze_rfm_distribution(rfm, OUTPUT_DIR)
        state['completed_phases'].append('distribution_analyzed')
        state['current_phase'] = 'preprocessing'
        update_state(state)

        # 4. Preprocesar
        X_scaled, scaler, feature_names = preprocess_for_clustering(rfm)
        state['completed_phases'].append('preprocessing_done')
        state['current_phase'] = 'finding_optimal_k'
        update_state(state)

        # 5. Encontrar K óptimo
        k_results = find_optimal_k_kmeans(X_scaled, range(3, 9))
        plot_elbow_silhouette(k_results, OUTPUT_DIR)

        # Seleccionar K por máximo silhouette
        best_k_idx = np.argmax(k_results['silhouette'])
        k_optimal = k_results['k'][best_k_idx]
        print(f"\n[INFO] K óptimo seleccionado: {k_optimal} (Silhouette={k_results['silhouette'][best_k_idx]:.4f})")

        state['completed_phases'].append('optimal_k_found')
        state['current_phase'] = 'model_training'
        update_state(state)

        # 6. Entrenar y comparar modelos
        models_results = train_and_compare_models(X_scaled, k_optimal)
        state['completed_phases'].append('models_trained')
        state['current_phase'] = 'model_selection'
        update_state(state)

        # 7. Seleccionar mejor modelo
        best_model_name, best_model, comparison_df = select_best_model(models_results)
        state['best_model'] = best_model_name
        state['best_silhouette'] = float(best_model['silhouette'])
        state['n_clusters'] = k_optimal
        state['completed_phases'].append('best_model_selected')
        state['current_phase'] = 'profiling'
        update_state(state)

        # 8. Crear perfiles de clusters
        rfm_clustered, profile_summary = create_cluster_profiles(rfm, best_model['labels'], best_model_name)
        plot_cluster_profiles(profile_summary, rfm_clustered, OUTPUT_DIR)
        state['completed_phases'].append('profiles_created')
        state['current_phase'] = 'saving_outputs'
        update_state(state)

        # 9. Guardar outputs
        # Modelo
        model_data = {
            'model': best_model['model'],
            'scaler': scaler,
            'feature_names': feature_names,
            'best_model_name': best_model_name,
            'k_optimal': k_optimal,
            'silhouette': best_model['silhouette'],
            'profile_summary': profile_summary
        }
        if 'pca' in best_model:
            model_data['pca'] = best_model['pca']

        joblib.dump(model_data, OUTPUT_DIR / 'customer_segmenter_model.pkl')
        print(f"[INFO] Modelo guardado: {OUTPUT_DIR / 'customer_segmenter_model.pkl'}")

        # Customer segments CSV
        output_cols = ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'avg_order_value',
                       'avg_profit', 'preferred_segment', 'preferred_category', 'Cluster']
        rfm_clustered[output_cols].to_csv(OUTPUT_DIR / 'customer_segments.csv', index=False)
        print(f"[INFO] Segmentos guardados: {OUTPUT_DIR / 'customer_segments.csv'}")

        # Model comparison
        comparison_df.to_csv(OUTPUT_DIR / 'customer_segmenter_results.csv', index=False)
        print(f"[INFO] Resultados guardados: {OUTPUT_DIR / 'customer_segmenter_results.csv'}")

        state['completed_phases'].append('outputs_saved')
        state['current_phase'] = 'generating_report'
        update_state(state)

        # 10. Generar reporte
        generate_report(rfm_clustered, profile_summary, comparison_df,
                        best_model_name, best_model['silhouette'], DOCS_DIR)

        # Finalizar
        state['status'] = 'completed'
        state['current_phase'] = 'done'
        state['completed_phases'].append('report_generated')
        update_state(state)

        print("\n" + "=" * 60)
        print("CustomerSegmenter Agent - COMPLETED")
        print(f"Best Model: {best_model_name}")
        print(f"Silhouette Score: {best_model['silhouette']:.4f}")
        print(f"Clusters: {k_optimal}")

        # Verificar objetivo
        if best_model['silhouette'] >= 0.5:
            print("TARGET ACHIEVED: Silhouette >= 0.5 (Optimal)")
        elif best_model['silhouette'] >= 0.3:
            print("TARGET ACHIEVED: Silhouette >= 0.3 (Minimum)")
        else:
            print("WARNING: Silhouette < 0.3 - Below minimum target")

        print("=" * 60)

        return state

    except Exception as e:
        state['status'] = 'error'
        state['errors'].append(str(e))
        update_state(state)
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
