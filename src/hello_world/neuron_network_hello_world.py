from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([[0, 1, 0], [0, 0, 1], [1, 3, 2], [3, 2, 1]])
y = np.array([0, 0, 1, 1]).T

# setup network
simple_model = Sequential()
simple_model.add(Dense(10, input_shape=(x.shape[1],), activation='relu', name='layer1'))

simple_model.add(Dense(8, activation='relu', name='layer2'))

simple_model.add(Dense(8, activation='relu', name='layer3'))

simple_model.add(Dense(4, activation='relu', name='layer4'))

simple_model.add(Dense(1, activation='sigmoid', name='layer5'))

# set optimizer and loss function
simple_model.compile(optimizer="sgd", loss="mse")

# set training epochs
simple_model.fit(x, y, epochs=2000)

print("result")
result = simple_model.predict(x[1:2])
print(result)

