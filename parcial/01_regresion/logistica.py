import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb

dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")
dataframe.head()

dataframe.describe()

print(dataframe.groupby('clase').size())

dataframe.drop(['clase'], 1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase', size=4, vars=["duracion",
                                                           "paginas", "acciones", "valor"], kind='reg')

X = np.array(dataframe.drop(['clase'], 1))
y = np.array(dataframe['clase'])
X.shape

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.2, random_state=7)

model = linear_model.LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

print("Verdaderos positivos (VP) (Windows -> Windows):", cm[0][0])
print("Falsos positivos (FP) (Windows -> Macintosh):", cm[1][0])
print("Falsos positivos (FP) (Windows -> Linux):", cm[2][0])

print("Verdaderos positivos (VP) (Macintosh -> Macintosh):", cm[1][1])
print("Falsos positivos (FP) (Macintosh -> Windows):", cm[0][1])
print("Falsos positivos (FP) (Macintosh -> Linux):", cm[2][1])

print("Verdaderos positivos (VP) (Linux -> Linux):", cm[2][2])
print("Falsos positivos (FP) (Linux -> Windows):", cm[0][2])
print("Falsos positivos (FP) (Linux -> Macintosh):", cm[1][2])

print()
print(" Informe de clasificación (sobre datos de test")
print(classification_report(y_test, y_pred))

# PREDECIR PARA UN CASO DETERMINADO
# Suponer un usuario con valores: Tiempo Duración: 5, Paginas visitadas: 2, Acciones al navegar: 3, Valoración: 5
X_nuevo = pd.DataFrame({'duracion': [5], 'paginas': [
                       2], 'acciones': [3], 'valor': [5]})
y_pred_nuevo = model.predict(X_nuevo)
print("Nuevo usuario: ", y_pred_nuevo)

print(model.predict([[5, 2, 3, 5]]))
