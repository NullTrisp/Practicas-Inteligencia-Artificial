# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Definir centroides iniciales
centroides = [[1, 1], [-1, -1], [1, -1]]

# Generar un conjunto de datos X utilizando 'make_blobs'
X, labels_true = make_blobs(
    n_samples=750, centers=centroides, cluster_std=0.4, random_state=0)

# Visualizar los datos generados de forma aleatoria
plt.scatter(X[:, 0], X[:, 1], s=5)
plt.show()

# Ejecutar DBSCAN  (eps => radio, min_samples => mínimo número de puntos para considerar un cluster). Ajustar a los datos generados X
radio = 0.3
print("Algoritmo DBScan con radio (eps) = ", radio)

clustering = DBSCAN(eps=radio, min_samples=10).fit(X)

core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
# core_sample_indices_   (índices de los núcleos)
core_samples_mask[clustering.core_sample_indices_] = True

labels = clustering.labels_
# Una etiqueta con valor -1 se corresponde con ruido. No se clasifica en ningún cluste

# Número de clusters en las etiquetas. Ignorando el ruido (label=-1)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Visualizar diferentes métricas
print('Número estimado de clusters: %d' % n_clusters_)

print("Homogeneidad: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# Visualizar resultados
unique_labels = set(labels)

# Definir los colores. [x,x,x,x]
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Color negro utilizado para representar el ruido. Puntos fuera de un cluster
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
plt.title('Número estimado de clusters: %d' % n_clusters_)
plt.show()
