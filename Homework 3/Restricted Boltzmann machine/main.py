from network import RestrictedBoltzmanMachine
from datasets import get_bar_stripes_dataset
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import ray


def get_probs_model(rbm, dataset, time_steps=20, num_samples=500):
  state_counts = np.zeros(len(dataset))
  size = (num_samples, rbm.num_visible_units)
  random_states = np.random.choice([-1, 1], size)
  for n in range(num_samples):
    rbm.visible_units = random_states[n]
    for _ in range(time_steps):
      rbm.reconstruction_step()
      for idx, state in enumerate(dataset):
        recon_state = rbm.visible_units
        is_equal = np.array_equal(recon_state, state)
        if is_equal:
          state_counts[idx] += 1
          break
      if is_equal:
        break
  probs_model = state_counts / num_samples
  return probs_model


def estimate_kl_div(rbm, probs_data, dataset, time_steps=20, num_samples=500):
  probs_model = get_probs_model(rbm, dataset, time_steps, num_samples)
  kl_div = np.sum(probs_data * np.log(probs_data / (probs_model+10**(-8))))
  return kl_div


@ray.remote
def run_cdk(rbm, dataset, epochs=100, learning_rate=0.01, k=100, metrics=[], verbose=False):
  rbm = deepcopy(rbm)
  metrics_data = {key: [] for key in metrics}
  for epoch in range(epochs):
    for state in dataset:
      rbm.update_weights(state, learning_rate, recon_steps=k)

    if metrics:
      if metrics_data.get('kl-divergence') is not None:
        kl_div = estimate_kl_div(rbm, probs_data, dataset)
        metrics_data['kl-divergence'].append(kl_div)

      if metrics_data.get('energy') is not None:
        energy = rbm.energy()
        metrics_data['energy'].append(energy)

      if verbose:
        print("-----epoch {}-----".format(epoch))
        for key, values in metrics_data.items():
          print("   {}: {:.3f}".format(key, values[-1]))
        print()
  return rbm, metrics_data


def plot_results(results, figsize=(10, 5)):
  fig, axes = plt.figure(figsize=figsize), []
  num_metrics = len(results[0][1])
  for idx in range(num_metrics):
    axes.append(plt.subplot(1, num_metrics, idx+1))
  for rbm, metrics_data in results:
    label = 'hidden neurons: {}'.format(rbm.num_hidden_units)
    for idx, (key, values) in enumerate(metrics_data.items()):
      axes[idx].plot(values, label=label)
      axes[idx].set_ylabel('{}'.format(key))
      axes[idx].set_xlabel('epochs')
      axes[idx].legend()
  plt.tight_layout()
  plt.show()


def print_results(results, dataset):
  print("\n-----final results-----\n")
  for rbm, metrics_data in results:
    probs_model = get_probs_model(rbm, dataset)
    print("{} hidden units".format(rbm.num_hidden_units))
    for key, values in metrics_data.items():
      print("  {}: {:.5f}".format(key, values[-1]))
    print()


def show_pattern_completion(rbm, dataset, pattern_idx, figsize=(10, 5)):
  fig, ax = plt.subplots(nrows=2, ncols=11, figsize=figsize)
  ax[0, 0].set_ylabel('untrained', fontsize=12)
  ax[1, 0].set_ylabel('trained', fontsize=12)
  imshow_params = {'cmap': 'gray_r', 'vmin':-1, 'vmax':1}

  size = (rbm.num_hidden_units, rbm.num_visible_units)
  untrained_rbm = RestrictedBoltzmanMachine(*size, dataset)

  pattern = dataset[pattern_idx]
  distorted_pattern = pattern.copy().reshape(3,3)
  distorted_pattern[:, 1:] = 0
  distorted_pattern = distorted_pattern.ravel()

  rbm.visible_units = distorted_pattern.copy()
  untrained_rbm.visible_units = distorted_pattern.copy()
  for t in range(11):
    ax[0, t].imshow(untrained_rbm.visible_units.reshape(3,3), **imshow_params)
    ax[1, t].imshow(rbm.visible_units.reshape(3,3), **imshow_params)
    rbm.reconstruction_step()
    untrained_rbm.reconstruction_step()
    for k in range(2):
      ax[k, t].axhline(y=0.5, c='white')
      ax[k, t].axhline(y=1.5, c='white')
      ax[k, t].axvline(x=0.5, c='white')
      ax[k, t].axvline(x=1.5, c='white')
      ax[k, t].set_xticks([])
      ax[k, t].set_yticks([])
      ax[k, t].set_title('t = {}'.format(t))
  plt.subplots_adjust(hspace=-0.5)
  plt.show()


if __name__ == '__main__':
  ray.init()

  dataset, probs_data = get_bar_stripes_dataset()

  cdk_params = {'k': 100,
                'epochs': 500,
                'learning_rate': 0.001,
                'metrics': ['kl-divergence'],
                'verbose': True}

  num_visible_units = dataset[0].size
  all_num_hidden_units = [8]

  all_rbms = []
  for num_hidden_units in all_num_hidden_units:
    size = (num_hidden_units, num_visible_units)
    all_rbms.append(RestrictedBoltzmanMachine(*size, dataset))

  results = ray.get([run_cdk.remote(rbm, dataset, **cdk_params) for rbm in all_rbms])

  plot_results(results, figsize=(10, 5))
  print_results(results, dataset)

  rbm = results[-1][0]
  show_pattern_completion(rbm, dataset, pattern_idx=-1)
