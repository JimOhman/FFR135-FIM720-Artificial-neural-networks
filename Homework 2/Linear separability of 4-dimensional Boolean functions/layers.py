import numpy as np


class FullyConnected():

  def __init__(self, input_dim, output_dim, activation, threshold=True):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.activation = activation

    self.weights = np.random.uniform(low=-0.2, high=0.2, size=(output_dim, input_dim))
    self.threshold = np.random.uniform(low=-1, high=1, size=(output_dim)) if threshold else None

    self.gradients = {'weights': 0, 'threshold': 0}
    self.output = None

  def __call__(self, input):
    field = np.dot(input, self.weights.T)
    if self.threshold is not None:
      field -= self.threshold
    self.output = self.activation(field)
    return self.output

  def __repr__(self):
    msg = "({}, {})".format(self.input_dim, self.output_dim)
    return msg

  def __str__(self):
    msg = "input dimension: {}\n".format(self.input_dim)
    msg += "output dimension: {}\n".format(self.output_dim)
    msg += "threshold: {}\n".format(True if self.threshold is not None else False)
    msg += "activation: {}\n".format(self.activation)
    return msg




