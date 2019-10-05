import numpy as np


class Perceptron:

	@staticmethod
	def step(z):
		return 1 if z >= 0 else 0

	def __init__(self, lr=0.01, epochs=100):
		self.lr = lr
		self.epochs = epochs
		self.W = None
		self.errors = None

	@staticmethod
	def weight_init(x):
		a = 1 + x.shape[1]
		sigma = np.sqrt(2/(a+1))
		return np.random.normal(0, sigma, size=a)

	def feed_forward(self, x):
		z = np.dot(x, self.W[1:]) + self.W[0]
		return self.step(z)

	def loss(self, x, y):
		error = 0
		for j in range(len(y)):
			y_hat = self.feed_forward(x[j])
			err = (y[j] - y_hat) * self.lr
			self.W[1:] += err * x[j]
			self.W[0] += err
			error += int(err != 0.0)
		return error/np.shape(x)[0]

	def fit(self, x, y):
		self.W = self.weight_init(x)
		self.errors = []

		for _ in range(self.epochs):
			self.errors.append(self.loss(x, y))

		return self

	def predict(self, x_test):
		y_pred = []
		for j in range(len(x_test)):
			y_pred.append(self.step(np.dot(x_test[j], self.W[1:]) + self.W[0]))
		return y_pred
