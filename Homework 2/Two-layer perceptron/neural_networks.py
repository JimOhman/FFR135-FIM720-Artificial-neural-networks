import numpy as np


class StandardNetwork():

  def __init__(self, layers):
    self.layers = layers

  def forward(self, input):
    self.input = input
    output = input
    for layer in self.layers:
      output = layer(output)
    return output

  def backward(self, error):
    batch_size = error.shape[0]

    delta = (error * self.layers[-1].activation.grad())
    previous_output = self.layers[-2].output if len(self.layers) > 1 else self.input
    weights_gradient = np.tensordot(delta.T, previous_output, axes=1) / batch_size
    self.layers[-1].gradients['weights'] = weights_gradient
    self.layers[-1].gradients['threshold'] = -delta.mean(0)

    for idx in reversed(range(len(self.layers)-1)):
      delta = np.dot(delta, self.layers[idx+1].weights)
      delta *= self.layers[idx].activation.grad()
      previous_output = self.layers[idx-1].output if idx > 0 else self.input
      weights_gradient = np.tensordot(delta.T, previous_output, axes=1) / batch_size
      self.layers[idx].gradients['weights'] = weights_gradient
      self.layers[idx].gradients['threshold'] = -delta.mean(0)
