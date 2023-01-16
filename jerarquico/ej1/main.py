# Importar librerias
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el conjunto de datos
dataset = pd.read_csv('Clientes_Tienda.csv')

# Seleccionar por ingresos y puntuación
X = dataset.iloc[:, [2, 3]].values

# Creamos el dendrograma para encontrar el número óptimo de clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# Visualizar el dendrograma
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancias Euclidianas')
plt.show()

# Realizar el clustering jerárquico ajustando al conjunto de datos
hc = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')

# Predicción del cluster
y_hc = hc.fit_predict(X)

# Visualizar los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1],
            s=100, c='magenta', label='Cluster 5')

plt.title('Clusters de clientes')
plt.xlabel('Ingresos')
plt.ylabel('Puntuación')
plt.legend()
plt.show()
