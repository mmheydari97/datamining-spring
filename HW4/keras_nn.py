from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np


class KerasNet:
    def __init__(self, activation='sigmoid', lr=1, epochs=3000):
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.model = Sequential()
        self.model.add(Dense(3, activation=self.activation, input_dim=3))
        self.model.add(Dense(4, activation=self.activation))
        self.model.add(Dense(1, activation=self.activation))
        self.loss = None
        self.acc = None
        self.y_pred = None

    def run(self, x, y):
        self.model.compile(optimizer=SGD(lr=self.lr),
                           loss='mse',
                           metrics=['acc'])
        self.model.fit(x, y, epochs=self.epochs, verbose=False)
        self.loss, self.acc = self.model.evaluate(x, y, verbose=False)
        self.y_pred = self.model.predict(x)
        return self
