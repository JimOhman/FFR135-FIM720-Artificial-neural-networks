from torchvision import datasets, transforms
from models import ModelOne, ModelTwo
import matplotlib.pyplot as plt
from torch import nn, optim
from time import time
import pandas as pd
import numpy as np
import pickle
import torch


def get_mnist_dataset():
  transform = transforms.Compose([transforms.ToTensor()])
  mnist_params = {'download': True, 'train': True, 'transform': transform}
  dataset = datasets.MNIST('PATH_TO_STORE_TRAINSET', **mnist_params)
  training_set, validation_set = torch.utils.data.random_split(dataset, [50000, 10000])
  test_set = datasets.MNIST('PATH_TO_STORE_TESTSET', **mnist_params)
  return training_set, validation_set, test_set


def get_metrics(model, loss_function, batch, labels, detailed=False):
  output = model(batch)
  loss = loss_function(output, labels)

  _, predicted_labels = torch.max(output, 1)
  is_correct = (predicted_labels == labels).squeeze()
  accuracy = 100 * (is_correct.sum().item() / is_correct.size()[0])

  if detailed:
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for i, label in enumerate(labels):
      class_correct[label] += is_correct[i].item()
      class_total[label] += 1
    class_correct = np.array(class_correct)
    class_total = np.array(class_total)

    class_accuracy = 100 * class_correct / class_total
    return loss, accuracy, class_accuracy

  return loss, accuracy


def evaluate_model(model, loss_function, batch_generator, device):
  sum_loss, sum_accuracy, tot_samples = 0, 0, 0
  with torch.no_grad():
    for batch, labels in batch_generator:
      batch, labels = batch.to(device), labels.to(device)

      num_samples = batch.size()[0]
      tot_samples += num_samples

      loss, accuracy = get_metrics(model, loss_function, batch, labels)

      sum_loss += loss.item() * num_samples
      sum_accuracy += accuracy * num_samples
  loss = sum_loss / tot_samples
  accuracy = sum_accuracy / tot_samples
  return loss, accuracy


def run_training(model, max_epochs=60,
                        batch_size=8192,
                        learning_rate=0.01,
                        momentum=0.9,
                        validation_patience=5,
                        validation_frequency=30,
                        verbose=True):
  
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  model.to(device)
  
  results =  {'training': {'loss': {}, 'accuracy': {}},
              'validation': {'loss': {}, 'accuracy': {}},
              'test': {'loss': 0, 'accuracy': 0}}

  params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}
  training_generator = torch.utils.data.DataLoader(training_set, **params)
  validation_generator = torch.utils.data.DataLoader(validation_set, **params)
  test_generator = torch.utils.data.DataLoader(test_set, **params)

  params = {'lr': learning_rate, 'momentum': momentum}
  optimizer = torch.optim.SGD(model.parameters(), **params)
  loss_function = nn.NLLLoss()
  
  patience, best_accuracy = 0, 0
  stop_training = False

  initial_time = time()

  print("\nStarting training with:")
  print("   max_epochs = {}".format(max_epochs))
  print("   batch_size = {}".format(batch_size))
  print("   learning_rate = {}".format(learning_rate))
  print("   momentum = {}".format(momentum))
  print("   validation_patience = {}".format(validation_patience))
  print("   validation_frequency = {}".format(validation_frequency))
  print("   verbose = {}\n".format(verbose))

  iteration = 0
  for epoch in range(max_epochs):
    sum_loss, sum_accuracy, tot_samples = 0, 0, 0
    for batch, labels in training_generator:
      batch, labels = batch.to(device), labels.to(device)

      num_samples = batch.size()[0]
      tot_samples += num_samples
      
      loss, accuracy = get_metrics(model, loss_function, batch, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      sum_loss += loss.item() * num_samples
      sum_accuracy += accuracy * num_samples

      if (iteration+1) % validation_frequency == 0:
        inputs = [model, loss_function, validation_generator, device]
        loss, accuracy = evaluate_model(*inputs)
        results['validation']['loss'][iteration] = loss
        results['validation']['accuracy'][iteration] = accuracy

        if verbose:
          print('Epoch {}, iteration {}'.format(epoch, iteration))
          print('   validation loss: {:.4f}'.format(loss))
          print('   validation accuracy: {:.2f} %'.format(accuracy))

        if accuracy > best_accuracy:
          best_accuracy = accuracy
          patience = 0
        else:
          patience += 1
          if patience == validation_patience:
            stop_training = True
            break

      iteration += 1

    training_loss = sum_loss / tot_samples
    training_accuracy = sum_accuracy / tot_samples
    
    results['training']['loss'][iteration] = training_loss
    results['training']['accuracy'][iteration] = training_accuracy

    if verbose:
      print('Epoch {}, iteration {}'.format(epoch, iteration))
      print('   training loss: {:.4f}'.format(training_loss))
      print('   training accuracy: {:.2f} %'.format(training_accuracy))

    if stop_training:
      msg = "early stop at epoch: {}, iteration: {}!"
      print(msg.format(epoch, iteration))
      break

  inputs = [model, loss_function, test_generator, device]
  loss, accuracy = evaluate_model(*inputs)
  results['test']['loss'] = loss
  results['test']['accuracy'] = accuracy

  final_time = (time() - initial_time) / 60
  print("\nTraining finished in {:.2f} min".format(final_time))
  print('   test loss: {:.4f}'.format(loss))
  print('   test accuracy: {:.2f} %'.format(accuracy))
  return results


def plot_results(results, name='', fig=None, ax=None, figsize=(10, 5)):
  if fig is None:
    fig, ax = plt.subplots(1, 2, figsize=figsize)
  ax[0].set_ylabel('loss')
  ax[1].set_ylabel('accuracy')
  for i in range(2):
    ax[i].set_xlabel('batch')
  for key, result in results.items():
    for idx, values in enumerate(result.values()):
      label = "[{}] - {}".format(name, key) if name else key
      if key != 'test':
        ax[idx].plot(list(values.keys()), list(values.values()), label=label)
      else:
        ax[idx].axhline(values, linestyle='--', label=label)
      ax[idx].legend()
  return fig, ax


def save_results(results, name):
  with open('results/' + name + '.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def load_results(name):
  with open('results/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)


if __name__ == '__main__':

  training_params = {'max_epochs': 60,
                     'validation_patience': np.float('inf'),
                     'validation_frequency': 1000,
                     'learning_rate': 0.01,
                     'momentum': 0.9,
                     'batch_size': 32,
                     'verbose': True}

  save = True
  plot = True

  training_set, validation_set, test_set = get_mnist_dataset()
  input_channels, *frame_size = training_set[0][0].size()
  models = {'network 1': ModelOne(input_channels, frame_size),
            'network 2': ModelTwo(input_channels, frame_size)}

  if plot:
    fig, ax = None, None

  for run_tag, model in models.items():
    results = run_training(model, **training_params)

    if save:
      torch.save(model.state_dict(), 'models/{}'.format(run_tag))
      save_results(results, run_tag)

    if plot:
      fig, ax = plot_results(results, name=run_tag, fig=fig, ax=ax)

  if plot:
    plt.tight_layout()
    plt.savefig('images/{}_{}'.format(*list(models.keys())))
