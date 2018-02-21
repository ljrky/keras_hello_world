import numpy as np
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# generate data
x = np.linspace(-1, 1, 2000)
np.random.shuffle(x)
y = 0.5 * x + 2 + np.random.normal(0, 0.5, (2000,))

# prepare test data and training data
x_train, y_train = x[:1600], y[:1600]
x_test, y_test = x[1600:], y[1600:]

# setup network
model = Sequential()
model.add(Dense(1,input_dim=1))
model.compile(loss='mse', optimizer="sgd")

# training model
print("Training")
for step in range(3001):
    cost = model.train_on_batch(x_train, y_train)
    if step % 100 == 0:
        print("training cost : ", cost)

print("Testing")
cost = model.evaluate(x_test, y_test, batch_size=40)

print("test cost", cost)
w, b = model.get_weights()
print("weight = ", w, "bias", b)

Y_pred = model.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,Y_pred)
plt.show()