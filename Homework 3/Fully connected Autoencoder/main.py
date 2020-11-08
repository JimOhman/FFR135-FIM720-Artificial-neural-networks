from networks import AutoEncoderNetwork, SmallAutoEncoderNetwork
from time import time
from plotting import plot_bottleneck_outputs, plot_decoder_outputs
from plotting import plot_results, plot_montage
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import torch


def get_mnist_dataset():
  transform = transforms.Compose([transforms.ToTensor()])
  mnist_params = {'download': True, 'train': True, 'transform': transform}
  dataset = datasets.MNIST('PATH_TO_STORE_TRAINSET', **mnist_params)
  training_set, validation_set = torch.utils.data.random_split(dataset, [50000, 10000])
  test_set = datasets.MNIST('PATH_TO_STORE_TESTSET', **mnist_params)
  return training_set, validation_set, test_set


def evaluate_model(model, loss_function, batch_generator, device):
  sum_loss, tot_samples = 0, 0
  with torch.no_grad():
    for batch, labels in batch_generator:
      batch, labels = batch.to(device), labels.to(device)
      batch = batch.view(-1, 784)

      num_samples = batch.size()[0]
      tot_samples += num_samples

      output = model(batch)
      loss = loss_function(output, batch)

      sum_loss += loss.item() * num_samples
  loss = sum_loss / tot_samples
  return loss


def run_training(model, max_epochs=400,
                        batch_size=32,
                        learning_rate=0.001,
                        validation_patience=np.float('inf'),
                        validation_frequency=5000,
                        save_frequency=np.float('inf'),
                        verbose=True):
  
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  model.to(device)
  
  results =  {'training': {'loss': {}},
              'validation': {'loss': {}},
              'test': {'loss': 0}}

  params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}
  training_generator = torch.utils.data.DataLoader(training_set, **params)
  validation_generator = torch.utils.data.DataLoader(validation_set, **params)
  test_generator = torch.utils.data.DataLoader(test_set, **params)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_function = torch.nn.MSELoss()
  
  patience, best_loss = 0, 0
  stop_training = False

  initial_time = time()

  print("\nStarting training with:")
  print("   max_epochs = {}".format(max_epochs))
  print("   batch_size = {}".format(batch_size))
  print("   learning_rate = {}".format(learning_rate))
  print("   validation_patience = {}".format(validation_patience))
  print("   validation_frequency = {}".format(validation_frequency))
  print("   save_frequency = {}".format(save_frequency))
  print("   verbose = {}\n".format(verbose))

  iteration = 0
  for epoch in range(max_epochs):
    sum_loss, tot_samples = 0, 0
    for batch, labels in training_generator:
      batch, labels = batch.to(device), labels.to(device)
      batch = batch.view(-1, 784)

      num_samples = batch.size()[0]
      tot_samples += num_samples
      
      output = model(batch)
      loss = loss_function(output, batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      sum_loss += loss.item() * num_samples

      if (iteration+1) % validation_frequency == 0:
        inputs = [model, loss_function, validation_generator, device]
        loss = evaluate_model(*inputs)
        results['validation']['loss'][iteration] = loss

        if verbose:
          print('Epoch {} - validation loss: {:.4f}'.format(epoch, loss))

        if loss > best_loss:
          best_loss = loss
          patience = 0
        else:
          patience += 1
          if patience == validation_patience:
            stop_training = True
            break

      iteration += 1

    training_loss = sum_loss / tot_samples
    results['training']['loss'][iteration] = training_loss

    if verbose:
      print('Epoch {} - training loss: {:.4f}'.format(epoch, training_loss))

    if stop_training:
      msg = "early stop at epoch: {}!"
      print(msg.format(epoch))
      break

  inputs = [model, loss_function, test_generator, device]
  loss = evaluate_model(*inputs)
  results['test']['loss'] = loss

  final_time = (time() - initial_time) / 60
  print("\nTraining finished in {:.2f} min".format(final_time))
  print('   test loss: {:.4f}'.format(loss))
  return results


def save_results(results, name):
  with open('results/' + name + '.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def load_results(name):
  with open('results/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)


if __name__ == '__main__':

  training_params = {'max_epochs': 2,
                     'validation_patience': np.float('inf'),
                     'validation_frequency': 5000,
                     'learning_rate': 0.001,
                     'batch_size': 64,
                     'verbose': True}

  plot = True
  save = True

  training_set, validation_set, test_set = get_mnist_dataset()

  all_models = {'normal': [AutoEncoderNetwork(784, bn_dim) for bn_dim in [2, 4]]}

  for label, models in all_models.items():
    for model in models:
      name = label + '_bottleneck_dim_{}'.format(model.bottleneck_dim)
      results = run_training(model, **training_params)

      if plot:
        plot_results(results, save=True, name=name)
        plot_montage(model, test_set, save=True, name=name)
        plot_decoder_outputs(model, test_set, save=True, name=name)
        if model.bottleneck_dim == 2:
          plot_bottleneck_outputs(model, test_set, save=True, name=name)

      if save:
       torch.save(model.state_dict(), 'models/{}'.format(name))
