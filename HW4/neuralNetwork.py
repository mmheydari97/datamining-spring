import numpy as np


def sigmoid(z):
	return 1.0/(1 + np.exp(-z))


def d_sigmoid(z):
	return z * (1.0 - z)


class NeuralNetwork:
	def __init__(self, n_i=3, n_h=4, n_o=1, lr=1, epochs=2000):
		self.w0 = np.random.normal(scale=np.sqrt(2/(n_i+n_h)), size=(n_i, n_h))
		self.w1 = np.random.normal(scale=np.sqrt(2/(n_h+n_o)), size=(n_h, n_o))
		self.x = None
		self.h = None
		self.y = None
		self.y_pred = None
		self.learning_rate = lr
		self.epochs = epochs
		self.errors = []

	def feedforward(self):
		self.h = sigmoid(np.dot(self.x, self.w0))
		self.y_pred = sigmoid(np.dot(self.h, self.w1))

	def backprop(self):
		# application of the chain rule to find derivative of the loss function with respect to w1 and w0
		d_w1 = np.dot(self.h.T, (2 * (self.y - self.y_pred) * d_sigmoid(self.y_pred)))
		d_w0 = np.dot(self.x.T, (np.dot(
			2*(self.y - self.y_pred)*d_sigmoid(self.y_pred), self.w1.T) * d_sigmoid(self.h)))

		# update the weights with the derivative (slope) of the loss function
		self.w0 += d_w0*self.learning_rate
		self.w1 += d_w1*self.learning_rate

	def loss(self):
		return np.mean((self.y-self.y_pred)**2)

	def fit(self, x, y):
		self.x = x
		self.y = y
		for i in range(self.epochs):
			self.feedforward()
			self.errors.append(self.loss())
			self.backprop()
		return self

	def predict(self, x_t):
		self.h = sigmoid(np.dot(x_t, self.w0))
		return sigmoid(np.dot(self.h, self.w1))
