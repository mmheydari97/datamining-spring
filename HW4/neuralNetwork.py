import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

sess = tf.InteractiveSession()


class NeuralNetwork:
	@staticmethod
	def sigmoid(z):
		return np.exp(z)

	def __init__(self, n_h=4, lr=0.01, epochs=50, activation=tf.nn.sigmoid):
		self.n_cls = None
		self.n_f = None
		self.n_out = None
		self.n_hidden = n_h
		self.learning_rate = lr
		self.epochs = epochs
		self.activation = activation
		self.input_X = tf.placeholder(
			'float32', shape=(None, self.n_f), name="input_X")
		self.input_y = tf.placeholder(
			'float32', shape=(None, self.n_cls), name="input_y")

		self.weights_0 = None
		self.weights_1 = None
		self.bias_0 = None
		self.bias_1 = None

		self.y_pred = None
		self.loss = None
		self.optimizer = None

	def init_weights(self):
		self.weights_0 = tf.Variable(
			tf.random_normal([self.n_f, self.n_hidden], stddev=(1 / tf.sqrt(float(self.n_f)))))
		self.bias_0 = tf.Variable(tf.random_normal([self.n_hidden]))

		self.weights_1 = tf.Variable(
			tf.random_normal([self.n_hidden, self.n_out], stddev=(1 / tf.sqrt(float(self.n_hidden)))))
		self.bias_1 = tf.Variable(tf.random_normal([self.n_out]))

	def forward_pass(self):
		hidden_output_0 = self.activation(tf.matmul(self.input_X, self.weights_0) + self.bias_0)
		y_pred = self.activation(tf.matmul(hidden_output_0, self.weights_1)+self.bias_1)
		return y_pred

	def loss(self, y):
		loss = tf.reduce_mean(tf.square(self.y_pred - y)) * 0.5
		return loss

	def optimize(self):
		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
			self.loss, var_list=[self.weights_0, self.weights_1, self.bias_0, self.bias_1])

	def accuracy(self, y):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_pred, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def fit(self, x, y):
		sess.run(tf.global_variables_initializer())
		sess.run(self.init_weights())

		training_acc = []

		for ep in range(self.epochs):
			self.y_pred = sess.run(self.forward_pass())
			training_acc.append(sess.run(self.accuracy, feed_dict={input_X: X_train,
																input_y: y_train, keep_prob: 1}))

