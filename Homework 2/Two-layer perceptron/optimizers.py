import numpy as np


class SGD():

  def __init__(self, layers, learning_rate, momentum=0.):
    self.layers = layers
    self.learning_rate = learning_rate
    self.momentum = momentum

    self.gradients = [{'weights': 0, 'threshold': 0} for _ in range(len(self.layers))]

  def step(self):
    for idx, layer in enumerate(self.layers):
      weights_grad = self.momentum * self.gradients[idx]['weights']
      weights_grad += (1-self.momentum) * layer.gradients['weights']
      self.gradients[idx]['weights'] = weights_grad

      layer.weights -= self.learning_rate * weights_grad

      if layer.threshold is not None:
        threshold_grad = (self.momentum * self.gradients[idx]['threshold'])
        threshold_grad += (1-self.momentum) * layer.gradients['threshold']
        self.gradients[idx]['threshold'] = threshold_grad

        layer.threshold -= self.learning_rate * threshold_grad


class RMSProp():

  def __init__(self, layers, learning_rate, beta=0., epsilon=10**(-8)):
    self.layers = layers
    self.learning_rate = learning_rate
    self.beta = beta
    self.epsilon = epsilon

    self.second_moment = [{'weights': 0, 'threshold': 0} for _ in range(len(self.layers))]

  def step(self):
    for idx, layer in enumerate(self.layers):
      second_moment = self.beta * self.second_moment[idx]['weights']
      second_moment += (1-self.beta) * layer.gradients['weights']**2
      self.second_moment[idx]['weights'] = second_moment

      weight_change = self.learning_rate * layer.gradients['weights']
      weight_change /= np.sqrt(second_moment) + self.epsilon
      layer.weights -= weight_change

      if layer.threshold is not None:
        second_moment_t = (self.beta * self.second_moment[idx]['threshold'])
        second_moment_t += (1-self.beta) * layer.gradients['threshold']**2
        self.second_moment[idx]['threshold'] = second_moment_t

        threshold_change = self.learning_rate * layer.gradients['threshold']
        threshold_change /= np.sqrt(second_moment_t) + self.epsilon
        layer.threshold -= threshold_change


class Adam():

  def __init__(self, layers, learning_rate, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
    self.layers = layers
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.first_moment = [{'weights': 0, 'threshold': 0} for _ in range(len(self.layers))]
    self.second_moment = [{'weights': 0, 'threshold': 0} for _ in range(len(self.layers))]
    self.t = 0

  def step(self):
    self.t += 1
    for idx, layer in enumerate(self.layers):
      first_moment = self.beta1 * self.first_moment[idx]['weights']
      first_moment += (1-self.beta1) * layer.gradients['weights']
      self.first_moment[idx]['weights'] = first_moment

      first_moment_corr = first_moment / (1-self.beta1**self.t)

      second_moment = self.beta2 * self.second_moment[idx]['weights']
      second_moment += (1-self.beta2) * layer.gradients['weights']**2
      self.second_moment[idx]['weights'] = second_moment

      second_moment_corr = second_moment / (1-self.beta2**self.t)

      weight_change = self.learning_rate * first_moment_corr
      weight_change /= (np.sqrt(second_moment_corr) + self.epsilon)
      layer.weights -= weight_change

      if layer.threshold is not None:
        first_moment = self.beta1 * self.first_moment[idx]['threshold']
        first_moment += (1-self.beta1) * layer.gradients['threshold']
        self.first_moment[idx]['threshold'] = first_moment

        first_moment_corr = first_moment / (1-self.beta1**self.t)

        second_moment = self.beta2 * self.second_moment[idx]['threshold']
        second_moment += (1-self.beta2) * layer.gradients['threshold']**2
        self.second_moment[idx]['threshold'] = second_moment

        second_moment_corr = second_moment / (1-self.beta2**self.t)

        weight_change = self.learning_rate * first_moment_corr
        weight_change /= (np.sqrt(second_moment_corr) + self.epsilon)
        layer.threshold -= weight_change

