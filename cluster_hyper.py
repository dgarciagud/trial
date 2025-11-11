 # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# 1) CARGA TU DATAFRAME
#    df = pd.read_csv("resultados_lightgbm.csv")
#    df = pd.read_excel("resultados_lightgbm.xlsx", engine="openpyxl")
# ============================================================

# --- EJEMPLO mínimo (borra en producción) ---
# df = pd.read_csv("tus_resultados.csv")
# ------------------------------------------------------------

# ============================================================
# 2) DEFINIR ESPACIO DE CLUSTERING (solo hiperparámetros)
# ============================================================
param_cols = ["learning_rate","num_leaves","feature_fraction","min_data_in_leaf","boost_rounds"]
# Si tienes más hiperparámetros válidos, añádelos aquí:
# param_cols += ["lambda_l1","lambda_l2","max_depth","min_gain_to_split", ...]

# Asegurar tipos numéricos y filtrar filas válidas
X = df[param_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
idx_kept = X.index  # mapear al df original

if X.shape[0] < 2:
    raise ValueError("Se necesitan al menos 2 filas válidas para hacer clustering.")

# ============================================================
# 3) ESCALADO
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

# ============================================================
# 4) KMEANS (k=10, ajustado a tamaño del dataset)
# ============================================================
K = 10
K = min(K, X_scaled.shape[0])  # no más clústers que filas
if K <= 1:
    raise ValueError("No hay suficientes filas para generar 10 clústers. Aumenta el dataset o reduce K.")

kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Silhouette (si hay al menos k+1 muestras, de lo contrario no es válido)
sil = np.nan
if X_scaled.shape[0] > K:
    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = np.nan

# Añadir etiquetas al df original
df_clustered = df.copy()
df_clustered.loc[idx_kept, "cluster_id"] = labels

# ============================================================
# 5) REPRESENTANTE POR CLÚSTER (medoide aprox)
# ============================================================
from numpy.linalg import norm

X_scaled_df = pd.DataFrame(X_scaled, index=idx_kept, columns=param_cols)
representantes = []
for c in range(K):
    members_idx = X_scaled_df.index[df_clustered.loc[idx_kept, "cluster_id"] == c]
    members_scaled = X_scaled_df.loc[members_idx].values
    centroid = centroids[c]

    dists = norm(members_scaled - centroid, axis=1)
    tmp = pd.DataFrame({"idx": members_idx, "dist": dists}).set_index("idx")

    # (Opcional) si quieres preferir menor complejidad, calcula y desempata:
    # comp = (df.loc[members_idx, "num_leaves"] * df.loc[members_idx, "boost_rounds"]).rename("complexity")
    # tmp = tmp.join(comp)
    # tmp = tmp.sort_values(by=["dist","complexity"], ascending=[True, True])

    tmp = tmp.sort_values(by="dist", ascending=True)
    best_idx = tmp.index[0]
    representantes.append(best_idx)

representantes_df = df.loc[representantes].copy()
representantes_df["cluster_id"] = df_clustered.loc[representantes, "cluster_id"].values
representantes_df = representantes_df.sort_values("cluster_id").reset_index(drop=True)

print(f"Silhouette score (k={K}): {sil if not np.isnan(sil) else 'N/A'}")
print("\n=== REPRESENTANTES (1 por clúster) ===")
print(representantes_df[["cluster_id"] + param_cols])

# ============================================================
# 6) (Opcional) Visualización 2D con PCA
# ============================================================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, index=idx_kept, columns=["PC1","PC2"])
    pca_df["cluster_id"] = labels
    pca_df["is_rep"] = pca_df.index.isin(representantes)

    plt.figure(figsize=(9,7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster_id", style="is_rep",
                    palette="tab10", s=90)
    plt.title(f"KMeans (k={K}) sobre hiperparámetros (sin IC)" + (f" | Silhouette={sil:.3f}" if not np.isnan(sil) else ""))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("No se pudo generar el gráfico PCA:", e)

# ============================================================
# 7) (Opcional) Guardar a Excel
# ============================================================
# with pd.ExcelWriter("clusters_sin_ic.xlsx", engine="openpyxl") as writer:
#     df_clustered.to_excel(writer, sheet_name="todas_las_filas", index=False)
#     representantes_df.to_excel(writer, sheet_name="representantes", index=False)
# print("Archivo 'clusters_sin_ic.xlsx' guardado.")
