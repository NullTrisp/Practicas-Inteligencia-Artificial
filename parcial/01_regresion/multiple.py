import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def f(x):
    """ función para aproximar mediante interpolación polinómica """
    return x * np.sin(x)


def generar_modelo_poly(x, y, color='rojo', grado=1):
   # Ajuste del polinomio de grado 'degree' a los datos de entrenamiento x,y
    coeffs = np.polyfit(x, y, deg=grado)
    # Determinar y escribir la forma del polinomio
    p = np.poly1d(coeffs, variable='X')
    print("Polinomio de grado ", grado, " : ")
    print(p)
    print("")

    y_pred = np.polyval(p, x)
    print('Error cuadrático medio: %.2f\n' % mean_squared_error(y, y_pred))

    plt.plot(x, y_pred, color=color,
             linewidth=2, label="grado% d" % grado)


n_data = 30

# generate random data for poly regression
x = np.linspace(0, 10, n_data)
np.random.RandomState(0).shuffle(x)
x = np.sort(x)
y = f(x)

# divide date into training and test sets
n_train = int(len(x) * 0.7)

x_train = np.array(x[:n_train])
y_train = np.array(y[:n_train])

x_test = np.array(x[n_train:])
y_test = np.array(y[n_train:])

generar_modelo_poly(x_train, y_train, color='blue', grado=1)
generar_modelo_poly(x_train, y_train, color='grey', grado=4)
# generar_modelo_poly(x_train, y_train, color='red', grado=10)

# scatter training data
plt.scatter(x_train, y_train, color='blue', s=30,
            marker='o', label="training points")

# scatter test data
plt.scatter(x_test, y_test, color='red')

# show graph
plt.show()
