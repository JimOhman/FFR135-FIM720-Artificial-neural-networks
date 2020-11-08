import numpy as np
from activation_functions import ScaledTanh
from error_functions import MSE
from optimizers import SGD
from data_handler import DataHandler
from layers import FullyConnected
from neural_networks import StandardNetwork
import csv


with open('input_data_numeric.csv') as f:
  data = csv.reader(f, delimiter=',', quotechar='|')
  inputs = []
  for row in data:
    inputs.append([int(i) for i in row[1:]])
  inputs = np.array(inputs)

with open('boolean_functions.csv') as f:
  data = csv.reader(f, delimiter=',', quotechar='|')
  all_targets = []
  for row in data:
    all_targets.append([[int(i)] for i in row])
  all_targets = np.array(all_targets)


learning_rate = 0.02
batch_size = 1
updates = 10000

layers = [FullyConnected(4, 1, activation=ScaledTanh(), threshold=True)]

for layer in layers:
  size = (layer.output_dim, layer.input_dim)
  layer.weights = np.random.uniform(low=-0.2, high=0.2, size=size)
  size = layer.output_dim
  layer.threshold = np.random.uniform(low=-1, high=1, size=size)

network = StandardNetwork(layers)

optimizer = SGD(layers, learning_rate)
error_function = MSE()

repeats = 10
is_linearly_separable = []
for targets in all_targets:
  for idx in range(1, repeats+1):
    data = DataHandler(inputs, targets)

    linearly_separable = False
    for jdx in range(updates):
      input, target = data.sample(batch_size)

      output = network.forward(input)
      train_error = error_function(target, output)
      network.backward(error=error_function.grad())
      optimizer.step()

      full_input, full_target = data.full
      full_output = network.forward(full_input)
      accuracy = (np.sign(full_output) == full_target).mean()
      print("repeat: [{}/{}]".format(idx, repeats))
      print("update: ({}/{})".format(jdx, updates))
      print("   error: {:.6f}".format(train_error))
      print("   accuracy: {}%\n".format(int(100*accuracy)))
      if accuracy == 1:
        linearly_separable = True
        break
    
    if linearly_separable:
      break
  is_linearly_separable.append(linearly_separable)

print(is_linearly_separable)
