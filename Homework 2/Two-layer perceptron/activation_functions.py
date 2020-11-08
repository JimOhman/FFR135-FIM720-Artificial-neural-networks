import numpy as np


class Tanh():

	def __call__(self, input):
		self.output = np.tanh(input)
		return self.output

	def grad(self):
		return (1 - self.output ** 2)


class ScaledTanh():

	def __call__(self, input):
		self.output = np.tanh(0.5*input)
		return self.output

	def grad(self):
		return 0.5 * (1 - self.output ** 2)