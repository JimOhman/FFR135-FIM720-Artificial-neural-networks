import numpy as np


class HopfieldNetwork():

  def __init__(self, size):
    self.size = size
    self.weights = np.random.rand(size, size)
    self.state = np.random.choice([-1, 1], size=size)

  def store_patterns(self, patterns, zero_diagonal=True):
    self.weights = (1/self.size) * np.tensordot(patterns.T, patterns, axes=1)
    if zero_diagonal:
      np.fill_diagonal(self.weights, 0)

  def insert_state(self, state):
    assert len(state) == self.size
    self.state = state

  def update_neuron_state(self, neuron_index, beta=None):
    field = np.dot(self.weights[neuron_index, :], self.state)
    if beta is None:
      self.state[neuron_index] = np.where(field >= 0, 1, -1)
      # self.state[neuron_index] = 1 if field >= 0 else -1
    else:
      prob = 1 / (1 + np.exp(-2*beta*field))
      self.state[neuron_index] = np.where(np.random.rand() <= prob, 1, -1)
      # self.state[neuron_index] = 1 if np.random.rand() <= prob else -1

  def run_asynchronous_updates(self, times, beta=None, beta_increment=0, return_states=False):
    if return_states:
      states = [self.state.copy()]
    for t in range(times):
      neuron_index = np.random.randint(self.size)
      self.update_neuron_state(neuron_index, beta=beta)
      if beta is not None:
        beta += beta_increment
      if return_states:
        states.append(self.state.copy())
    if return_states:
      return states
