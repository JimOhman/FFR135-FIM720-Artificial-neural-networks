import numpy as np


class MSE():

	def __call__(self, target, output):
		self.error = target - output
		return 0.5 * (self.error**2).mean()

	def grad(self):
		return -self.error