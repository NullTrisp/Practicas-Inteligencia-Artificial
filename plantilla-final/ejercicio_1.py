from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
    Ejercicio 1
"""
# realizamos la lectura del csv
df = pd.read_csv('data12.csv')

# selecciones la primera columna como variable independiente
x = df.iloc[:, 0].values

# selecciones la segunda columna como variable dependiente
y = df.iloc[:, 1].values

# convertimos las variables a un array de numpy
x = np.array(x).reshape(-1, 1).astype(np.float64).ravel()
y = np.array(y).reshape(-1, 1).ravel()

"""
    Apartado a
"""
# obtenemos datos de entrenamiento y de prueba tomando el primer 80% de los datos para entrenamiento y el 20% restante para prueba
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, shuffle=False)

# obtenemos los coeficientes de la regresión
coeffs = np.polyfit(x_train, y_train, 4)
# instanciamos el modelo de regresión polinomial
modelo = np.poly1d(coeffs)

# predecimos los valores de y para los datos de prueba
y_pred = modelo(x_test)

# calculamos el error cuadrático medio
ecm = mean_squared_error(y_test, y_pred)
print('ECM: {}'.format(round(ecm)))

# calculamos el coeficiente de correlación
corr = np.corrcoef(y_test, y_pred)[0, 1]
print('Correlación: {}'.format(corr))

# graficamos los datos de entrenamiento y de prueba
plt.scatter(x_train, y_train, color='blue')
plt.scatter(x_test, y_test, color='red')

# agregamos labels a los ejes
plt.xlabel('columna 1')
plt.ylabel('columna 2')

# agregamos leyendas
plt.legend(['datos de entrenamiento', 'datos de prueba'])

# graficamos los valores de predicción
plt.plot(x_test, y_pred, color='black')

# agregamos el ecm a la gráfica
plt.text(0.5, 0.5, 'ECM: {}'.format(round(ecm)), fontsize=12)

# mostramos la gráfica
plt.show()
print()
"""
    Conclusiones apartado a:
    - El modelo de regresión polinomial utilizando el primer 80% de los datos no es el adecuado para este problema 
        ya que no es un representativo de la mayoría de los datos.
    - Independientemente del grado del polinomio, el error cuadrático medio es muy alto para los datos de prueba.
    - El coeficiente de correlación es muy bajo, lo que indica que no existe una relación de los datos de prueba y los de test.
"""

"""
    Apartado b
"""
# obtenemos datos de entrenamiento y de prueba tomando aleatoriamente el 80% de los datos para entrenamiento y el 20% restante para prueba
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# graficamos los datos de entrenamiento y de prueba
plt.scatter(x_train, y_train, color='blue')
plt.scatter(x_test, y_test, color='red')

# agregamos labels a los ejes
plt.xlabel('columna 1')
plt.ylabel('columna 2')

# agregamos leyendas
plt.legend(['datos de entrenamiento', 'datos de prueba'])

# obtenemos los coeficientes de la regresión
coeffs = np.polyfit(x_train, y_train, 4)
# instanciamos el modelo de regresión polinomial
modelo = np.poly1d(coeffs)

x_test = np.sort(x_test)
y_test = np.sort(y_test)

# predecimos los valores de y para los datos de prueba
y_pred = modelo(x_test)

# calculamos el error cuadrático medio
ecm = mean_squared_error(y_test, y_pred)
print('ECM: {}'.format(round(ecm)))

# calculamos el coeficiente de correlación
corr = np.corrcoef(y_test, y_pred)[0, 1]
print('Correlación: {}'.format(corr))

# graficamos los valores de predicción
xx = np.linspace(x_test.min(), x_test.max(), 500)
plt.plot(xx, modelo(xx), color='black')

# agregamos el ecm a la gráfica
plt.text(0.5, 0.5, 'ECM: {}'.format(round(ecm)), fontsize=12)

# mostramos la gráfica
plt.show()
print()

"""
    Conclusiones apartado b:
    - El modelo de regresión polinomial utilizando el 80% de los datos de forma aleatoria es el adecuado para este problema
        ya que es un representativo de la mayoría de los datos.
    - Para este caso el grado de polinomio 4 es el adecuado ya que el error cuadrático medio es el más bajo.
    - El coeficiente de correlación es alto, lo que indica que existe una relación de los datos de prueba y los de test.
"""

"""
    Apartado a
    ECM: 8218
    Correlación: -0.49046145511361594

    Apartado b
    ECM: 84
    Correlación: 0.995878875477147
    
    Conclusiones apartado c:
    - Indudablemente para este problema de predicción se requiere de un modelo polinomial de grado 4. 
        Por otro lado para esta situación no es viable utilizar el primer 80% de los datos para entrenamiento, 
        por el contrario se requiere de cierta aleatoriedad para así representar de una mejor manera los datos.
    - Se evidencia una gran diferencia entre el ECM y la correlación entre modelo y modelo teniendo muy poca correlacion
        y un elevado ECM para el apartado a y lo contario en el b.
    - Con estas conclusiones se puede decir que el modelo del apartado b es el adecuado para este problema.
"""
