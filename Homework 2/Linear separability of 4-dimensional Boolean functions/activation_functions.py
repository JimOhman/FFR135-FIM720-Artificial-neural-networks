import numpy as np


class ScaledTanh():

	def __call__(self, input):
		self.output = np.tanh(0.5*input)
		return self.output

	def grad(self):
		return 0.5 * (1 - self.output ** 2)
