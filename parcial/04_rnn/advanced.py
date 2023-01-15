import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = np.loadtxt('Datos.txt')
X = data[:, 1:]
y = data[:, 0]

print('X.shape =', X.shape)
print('y.shape =', y.shape)

colors = ['red', 'blue']
plt.scatter(X[:, 0], X[:, 1], c='g')
plt.scatter(X[y > 0, 0], X[y > 0, 1], c=colors[0])
plt.scatter(X[y == 0, 0], X[y == 0, 1], c=colors[1])
plt.show()

ntrain = int(3*len(y)/4)
X_train = X[:ntrain, :]
y_train = y[:ntrain]
X_test = X[ntrain:, :]
y_test = y[ntrain:]
print("Entrenamiento:   X ", np.shape(X_train), "   y ", np.shape(y_train))
print("Test:   X ", np.shape(X_test), "   y ", np.shape(y_test))

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                    activation='relu', max_iter=500)

mlp.fit(X_train, y_train)

plt.plot(mlp.loss_curve_)

plt.xlabel('Iteracción')
plt.ylabel('Función de coste')
plt.show()

y_pred = mlp.predict(X_test)

print("Matriz de confusión")
print(confusion_matrix(y_test, y_pred))

print()
print("Falsos positivos (0): ", confusion_matrix(y_test, y_pred)[0][1])
print("Falsos positivos (1): ", confusion_matrix(y_test, y_pred)[1][0])
print()

print("Clasificación de los resultados de la Validación(test)")
print("precision = num detecciones correctas / numero detecciones")
print("recall=num detecciones correctas / numero total de objetos en esa clase")
print()
print(classification_report(y_test, y_pred))

plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='r')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='b')
plt.scatter(X_test[(y_test == 1) & (y_pred == 0), 0],
            X_test[(y_test == 1) & (y_pred == 0), 1], c='orange')

plt.scatter(X_test[(y_test == 0) & (y_pred == 1), 0],
            X_test[(y_test == 0) & (y_pred == 1), 1], c='orange')
plt.show()

print(mlp.predict([[-4, 5]]))