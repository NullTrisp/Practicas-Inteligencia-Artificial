import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# leemos el csv
df = pd.read_csv('data12.csv', names=["x", "y", "r", "g", "b", "valido"])

# seleccionamos de la columna 0 a la 5 como características
x = df.iloc[:, 0:5].values
# seleccionamos la columna 6 como etiquetas
y = df.iloc[:, 5].values

# dividimos los datos en datos de entrenamiento y de prueba con un 80% para entrenamiento y 20% para prueba
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# instanciamos el modelo de clasificación con k igual a 10
knn = KNeighborsClassifier(n_neighbors=10)

# fit the model with the training data
knn.fit(x_train, y_train)

# predict for the test data
y_pred = knn.predict(x_test)

# calculamos la precisión del modelo
accuracy = knn.score(x_test, y_test)
print("Precisión del modelo: {}".format(accuracy))

# calculamos la precisión del modelo en base a las predicciones
accuracy = np.mean(y_pred == y_test)
print("Precisión del modelo en base a los test: {}".format(accuracy))

# calculamos la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Verdaderos positivos (VP):", cm[1][1])
print("Verdaderos negativos (VN):", cm[0][0])
print("Falsos positivos (FP):", cm[0][1])
print("Falsos negativos (FN):", cm[1][0])

# realizamos una predicción para un nuevo dato
prediction = knn.predict([[380, 260, 200, 130, 116]])
print("Predicción para un nuevo dato: {}".format(prediction))
if (prediction == 1):
    print("Producto valido")
else:
    print("Producto no valido")

# graficamos la correlación entre las características
sns.heatmap(df.corr(), annot=True)
plt.show()

# graficamos las relaciones entre las características
sns.pairplot(df, hue="valido", vars=["x", "y", "r", "g", "b"])
plt.show()

"""
    Precisión del modelo: 0.9966777408637874
    Precisión del modelo en base a los test: 0.9966777408637874
    Verdaderos positivos (VP): 320
    Verdaderos negativos (VN): 280
    Falsos positivos (FP): 2
    Falsos negativos (FN): 0
    
    Conclusiones:
    - Para este caso se utilizó el algoritmo KNN con k igual a 10
    - El porque de está decisión es debido a que al investigar online se menciona mucho que para 
        datasets pequeños y que cuyos datos sean numericos es mejor utilizar este algoritmo.
    - El numero de k se determinó en base a distintos valores y se eligió el que mejor resultado dio,
        en este caso al utilizar un k=15 da un resultado identico al de utilizar k=10, mientras que el usar 
        un k=20 la precision del modelo disminuye por un overtraining.
"""
