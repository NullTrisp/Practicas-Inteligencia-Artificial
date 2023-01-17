from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import pandas as pd

# Leemos el archivo csv y lo convertimos en un dataframe
df = pd.read_csv('data1.csv', names=["x", "y", "r", "g", "b"], header=None)

# tomamos x como x e y como y
x = df[["x", "y"]].values
# tomamos r, g y b como y
y = df[["r", "g", "b"]].values

""" APARTADO A """
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, random_state=42, shuffle=False)

# Creamos una red neuronal con x e y como entradas y r, g y b como salidas
# Crear un modelo secuencial
model = Sequential()

# Añadir una capa de entrada con 2 neuronas (para x e y)
model.add(Dense(2, input_dim=2, activation='relu'))

# Añadir una capa oculta con 4 neuronas
model.add(Dense(4, activation='relu'))

# Añadir una capa de salida con 3 neuronas (para r g b)
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con datos de entrada x e y y datos de salida r g b
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluar el modelo con datos de prueba
metrics = model.evaluate(x_test, y_test, batch_size=32)

# Imprimir la precisión del modelo
print("Accuracy:", metrics[1])


# Realizamos una predicción con los siguientes datos 1,93
print(model.predict([[1, 93]]))

""" APARTADO B """
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, random_state=42, shuffle=True)

# Creamos una red neuronal con x e y como entradas y r, g y b como salidas
# Crear un modelo secuencial
model = Sequential()

# Añadir una capa de entrada con 2 neuronas (para x e y)
model.add(Dense(2, input_dim=2, activation='relu'))

# Añadir una capa oculta con 4 neuronas
model.add(Dense(4, activation='relu'))

# Añadir una capa de salida con 3 neuronas (para r g b)
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con datos de entrada x e y y datos de salida r g b
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluar el modelo con datos de prueba
metrics = model.evaluate(x_test, y_test, batch_size=32)

# Imprimir la precisión del modelo
print("Accuracy:", metrics[1])


# Realizamos una predicción con los siguientes datos 1,93
print(model.predict([[1, 93]]))
