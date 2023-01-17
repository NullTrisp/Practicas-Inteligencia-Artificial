import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Leemos el csv de data2.csv y lo convertimos en un dataframe
df = pd.read_csv('data2.csv', names=["x", "y", "r", "g", "b"], header=None)

# tomamos x como los datos de input desde la fila 210591 a la 228069
X = df[["x", "y", "r", "g", "b"]].values[210591:228069]

dbscan = DBSCAN(eps=2, min_samples=5)

# Entrenamos el modelo
dbscan.fit(X)

# Obtenemos los labels
labels = dbscan.labels_

# Obtenemos los centroides
centroids = dbscan.components_

# Obtenemos el número de clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Obtenemos el número de outliers
n_outliers = list(labels).count(-1)

# Imprimimos los resultados
print("Número de clusters: ", n_clusters)
print("Número de outliers: ", n_outliers)

# graficamos los clusters en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')
plt.show()
