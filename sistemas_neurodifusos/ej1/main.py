import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importar los datos
dataframe = pd.read_csv(r"comprar_alquilar.csv")
# Mostrar las 10 primeras filas
dataframe.head(10)
# ingresos y gastos (comunes, pago_coche,gastos_otros) son mensuales (de 1 personas o 2 si están casados)
# trabajo: 0-sin trabajo 1-autonomo 2-asalariado 3-empresario 4-Autonomos 5-Asalariados 6-Autonomo y Asalariado 7-Empresario y Autonomo 8 Empresarios o empresario y autónomo
# estado_civil: 0-soltero 1-casado 2-divorciado
# hijos: Cantidad de hijos menores (que no trabajan)
# comprar: 0-mejor alquilar 1-Comprar casa

# Agrupar datos según si es compra o alquiler
print(dataframe.groupby('comprar').size())

# Mostrar histogramas
dataframe.drop(['comprar'], axis=1).hist()
plt.show()

# Preprocesar los datos. Crear 2 columnas nuevas ('gastos' y 'financiar')
# En una agrupamos los gastos mensuales ('gastos')
# En la otra el presupuesto a financiar para comprar la casa ('financiar')
dataframe['gastos'] = (dataframe['gastos_comunes']+dataframe['gastos_otros']+dataframe['pago_coche'
                                                                                       ])
dataframe['financiar'] = dataframe['vivienda']-dataframe['ahorros']

# Graficar los ingresos frente al importe a financiar
atributos_sel = ["ingresos", "financiar"]
X = dataframe[atributos_sel].values
y = dataframe["comprar"]

# Construir el gráfico
fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Modelo Naive Bayes', size=14)

xlim = (2000, 9000)
ylim = (100000, 700000)

xg = np.linspace(xlim[0], xlim[1], 40)
yg = np.linspace(ylim[0], ylim[1], 30)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)

ax.set(xlim=xlim, ylim=ylim)

plt.show()

# Crear el modelo de Gaussian Naive Bayes
# Dividimos los datos en entrenamiento y test (20 %)
X_train, X_test = train_test_split(dataframe, test_size=0.2, random_state=6)
y_train = X_train["comprar"]
y_test = X_test["comprar"]
# Definir el clasificador
gnb = GaussianNB()

# Entrenar el modelo
atributos = ['ingresos', 'ahorros', 'hijos', 'trabajo', 'financiar']
gnb.fit(X_train[atributos].values, y_train)

# Predecir para los datos de test
y_pred = gnb.predict(X_test[atributos])

# Mostrar resultados
print('Precisión en el set de Entrenamiento: {:.2f}'
      .format(gnb.score(X_train[atributos], y_train)))
print('Precisión en el set de Test: {:.2f}'.format(
    gnb.score(X_test[atributos], y_test)))

# Resultados sobre los datos de test
print("Total de Muestras en Test: {}\nFallos: {}".format(X_test.shape[0], (y_test !=
                                                                           y_pred).sum()))

# Mostrar la matriz de confusión y el informe de clasificación
print("Matriz de Confusión")
print(confusion_matrix(y_test, y_pred))
print("Informe de clasificación")
print(classification_report(y_test, y_pred))


# Predicción sobre nuevos valores
#                 ['ingresos', 'ahorros', 'hijos', 'trabajo', 'financiar']
print(gnb.predict([[2000,        5000,     0,       5,         200000],
                   [6000,        34000,    2,       5,         320000]]))
# Resultado esperado 0-Alquilar, 1-Comprar casa
