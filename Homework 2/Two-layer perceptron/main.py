from neural_networks import StandardNetwork
from activation_functions import Tanh
from data_handler import DataHandler
from layers import FullyConnected
from error_functions import MSE
from optimizers import SGD, RMSProp, Adam
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import numpy as np
import csv


with open('data/training_set.csv') as f:
  data = csv.reader(f, delimiter=',', quotechar='|')
  train_inputs = []
  train_targets = []
  for row in data:
    train_inputs.append([float(i) for i in row[:-1]])
    train_targets.append([float(row[-1])])
  train_inputs = np.array(train_inputs)
  train_targets = np.array(train_targets)

with open('data/validation_set.csv') as f:
  data = csv.reader(f, delimiter=',', quotechar='|')
  val_inputs = []
  val_targets = []
  for row in data:
    val_inputs.append([float(i) for i in row[:-1]])
    val_targets.append([float(row[-1])])
  val_inputs = np.array(val_inputs)
  val_targets = np.array(val_targets)

data = DataHandler(train_inputs, train_targets)
data.validation = [val_inputs, val_targets]
data.val_size = val_inputs.shape[0]


plot_updates = True
plot_decision_boundary = True
plot_parameter_changes = True
save_parameters = False

learning_rate = 0.01
batch_size = 1
updates = 2000000

M1 = 25
M2 = 10

layers = [FullyConnected(2, M1, activation=Tanh(), threshold=True),
          FullyConnected(M1, M2, activation=Tanh(), threshold=True),
          FullyConnected(M2, 1, activation=Tanh(), threshold=True)]

if plot_parameter_changes:
  old_layers = deepcopy(layers)

network = StandardNetwork(layers)
optimizer = SGD(layers, learning_rate)
error_function = MSE()

errors = {'training': [], 'validation': []}
for i in range(updates):
  input, target = data.sample(batch_size)

  output = network.forward(input)
  train_error = error_function(target, output)
  errors['training'].append(train_error)
  network.backward(error_function.grad())
  optimizer.step()

  if i % 1000 == 0:
    val_input, val_target = data.validation
    val_output = network.forward(val_input)
    class_error = 0.5 * np.abs((val_target - np.sign(val_output))).mean()
    errors['validation'].append(100*class_error)
    print("update: {}".format(i))
    print("   error: {:.6f}".format(train_error))
    print("   class error: {:.2f}%\n".format(100*class_error))


if plot_updates:
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
  x_range_train = range(len(errors['training']))
  ax[0].plot(x_range_train, errors['training'])

  x_range_val = range(len(errors['validation']))
  ax[1].plot(x_range_val, errors['validation'])
  plt.show()


if plot_parameter_changes:
  new_layers = network.layers
  num_layers = len(new_layers)

  fig, axes = plt.subplots(nrows=1, ncols=num_layers, figsize=(15, 5))
  axes = [axes] if num_layers == 1 else axes
  for idx, ax in enumerate(axes):
    sns.kdeplot(old_layers[idx].weights.flatten(), ax=ax, shade=True, label='old')
    sns.kdeplot(new_layers[idx].weights.flatten(), ax=ax, shade=True, label='new')
    ax.set_title('weights layer: {}'.format(idx+1))
  plt.show()

  fig, axes = plt.subplots(nrows=1, ncols=num_layers, figsize=(15, 5))
  axes = [axes] if num_layers == 1 else axes
  for idx, ax in enumerate(axes):
    sns.kdeplot(old_layers[idx].threshold.flatten(), ax=ax, shade=True, label='old')
    sns.kdeplot(new_layers[idx].threshold.flatten(), ax=ax, shade=True, label='new')
    ax.set_title('threshold layer: {}'.format(idx+1))
  plt.legend()
  plt.show()


if plot_decision_boundary:
  fig, ax = plt.subplots(figsize=(12, 12))

  x_min, x_max = train_inputs[:, 0].min(), train_inputs[:, 0].max()
  y_min, y_max = train_inputs[:, 1].min(), train_inputs[:, 1].max()
  xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                       np.arange(y_min, y_max, .02))
  Z = np.sign(network.forward(np.c_[xx.ravel(), yy.ravel()]))
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

  plt.scatter(train_inputs[:, 0], train_inputs[:, 1], c=train_targets, cmap=plt.cm.coolwarm)
  plt.show()


if save_parameters:
  for idx, layer in enumerate(network.layers):
    weights = layer.weights
    threshold = layer.threshold
    with open('parameters/w{}.csv'.format(idx+1), 'w', newline='') as file:
      writer = csv.writer(file)
      if idx != 2:
        for row in weights:
          writer.writerow(row)
      else:
        for row in weights[0, :]:
          writer.writerow([row])
    with open('parameters/t{}.csv'.format(idx+1), 'w', newline='') as file:
      writer = csv.writer(file)
      for row in threshold:
        writer.writerow([row])
