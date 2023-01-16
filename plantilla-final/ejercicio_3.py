import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import sklearn.tree as tree
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt

# leemos el csv
df = pd.read_csv('data3.csv', names=["x", "y", "r", "g", "b", "valido", "a"])

# graficamos las relaciones entre las características
sns.pairplot(df, hue="valido", vars=["x", "y", "r", "g", "b", "a"])
plt.show()

# seleccionamos de la columna 0, 1,2,3,4 y 6 como características
x = df.iloc[:, [0, 1, 2, 3, 4, 6]].values
# seleccionamos la columna 5 como etiquetas
y = df.iloc[:, 5].values

# dividimos los datos en datos de entrenamiento y de prueba con un 80% para entrenamiento y 20% para prueba
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# instanciamos el modelo de arbol de decisión
arbol = tree.DecisionTreeClassifier()

# ajustamos el modelo con los datos de entrenamiento
arbol.fit(x_train, y_train)

# predecimos para los datos de prueba
y_pred = arbol.predict(x_test)

# calculamos la precisión del modelo
accuracy = arbol.score(x_test, y_test)
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
prediction = arbol.predict([[380, 260, 200, 130, 116, 0]])
print("Predicción para un nuevo dato: {}".format(prediction))
if (prediction == 1):
    print("Es válido")
else:
    print("No es válido")

# generamos el archivo .dot
dot_data = tree.export_graphviz(arbol, out_file=None, feature_names=["x", "y", "r", "g", "b", "a"], class_names=[
                                "no válido", "válido"], filled=True, rounded=True, special_characters=True)

# generamos el archivo .dot
graph = graphviz.Source(dot_data)

# renderizamos el archivo .dot
graph.render("arbol_decision")

# mostramos el árbol de decisión
graph.view()

"""
    Precisión del modelo: 1.0
    Precisión del modelo en base a los test: 1.0
    Verdaderos positivos (VP): 320
    Verdaderos negativos (VN): 282
    Falsos positivos (FP): 0
    Falsos negativos (FN): 0
    Predicción para un nuevo dato: [1]
    Es válido

    Conclusiones:
    - En este caso, incialmente realicé el modelo con el algoritmo knn, 
        pero despues de realizar pruebas exhaustivas, me dí cuenta que el nuevo valor agregado
        definia directamente la clase a la que pertenecía, por lo que el modelo no era necesario.
    - Por lo tanto, realicé el modelo con el algoritmo de árbol de decisión, el cual permite predecir valores no solamente por su cuantia
        sino tambien por su relevancia en el modelo, dando así que el nuevo valor agregado definia directamente la clase a la que pertenecía.
    - Al renderizar el arbol se evidencia como unicamente se necesita saber la ultima caracteristica para definir la clase a la que pertenece.
"""
