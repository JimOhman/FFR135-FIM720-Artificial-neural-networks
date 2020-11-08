import numpy as np


class RestrictedBoltzmanMachine():

	def __init__(self, num_hidden_units, num_visible_units, dataset=None):
		self.num_hidden_units = num_hidden_units
		self.num_visible_units = num_visible_units

		self.hidden_units = None
		self.hidden_bias_units = np.zeros(num_hidden_units)

		self.visible_units = None
		self.visible_bias_units = np.zeros(num_visible_units)

		if dataset is not None:
			probs_is_on = np.zeros_like(self.visible_bias_units)
			for pattern in dataset:
				probs_is_on += pattern > 0
			probs_is_on = probs_is_on / len(dataset)
			self.visible_bias_units = np.log(probs_is_on/(1-probs_is_on))

		size = (num_hidden_units, num_visible_units)
		self.weights = np.random.normal(scale=0.1, size=size)

	def reconstruction_step(self):
		hidden_probs = 1 / (1+np.exp(-2*self.hidden_field()))
		r = np.random.rand(*hidden_probs.shape)
		self.hidden_units = np.where(r < hidden_probs, 1, -1)

		visible_probs = 1 / (1+np.exp(-2*self.visible_field()))
		q = np.random.rand(*visible_probs.shape)
		self.visible_units = np.where(q < visible_probs, 1, -1)

	def hidden_field(self):
		hidden_field = np.dot(self.weights, self.visible_units)
		hidden_field -= self.hidden_bias_units
		return hidden_field

	def visible_field(self):
		visible_field = np.dot(self.hidden_units, self.weights)
		visible_field -= self.visible_bias_units
		return visible_field

	def update_weights(self, state, learning_rate, recon_steps=100):
		self.visible_units = state.copy()
		
		init_visible = state
		init_hidden_average = np.tanh(self.hidden_field())
		init_correlations = np.outer(init_hidden_average, init_visible)

		for _ in range(recon_steps):
			self.reconstruction_step()

		recon_visible = self.visible_units
		recon_hidden_average = np.tanh(self.hidden_field())
		recon_correlations = np.outer(recon_hidden_average, recon_visible)

		weights_grad = (init_correlations - recon_correlations)
		self.weights += learning_rate * weights_grad

		hidden_bias_grad = (init_hidden_average - recon_hidden_average)
		self.hidden_bias_units -= learning_rate * hidden_bias_grad

		visible_bias_grad = (init_visible - recon_visible)
		self.visible_bias_units -= learning_rate * visible_bias_grad

	def energy(self):
		energy = np.dot(self.hidden_units, self.weights)
		energy = -np.dot(energy, self.visible_units)
		energy += np.dot(self.visible_units, self.visible_bias_units)
		energy += np.dot(self.hidden_units, self.hidden_bias_units)
		return energy
