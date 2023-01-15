# Librerías
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
target_data = np.array([[1], [0], [0], [1]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)

scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict([[0, 0]]).round())
print(model.predict([[1, 0]]).round())
print(model.predict([[0, 1]]).round())
