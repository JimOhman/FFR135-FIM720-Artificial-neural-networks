

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

      layer.weights -= self.learning_rate * weights_grad
      self.gradients[idx]['weights'] = weights_grad

      if layer.threshold is not None:
        threshold_grad = (self.momentum * self.gradients[idx]['threshold'])
        threshold_grad += (1-self.momentum) * layer.gradients['threshold']

        layer.threshold -= self.learning_rate * threshold_grad
        self.gradients[idx]['threshold'] = threshold_grad
