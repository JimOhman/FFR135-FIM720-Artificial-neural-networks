from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import random
import ray


@ray.remote
def run_trials(hopfield_network, p, args):
  np.random.seed(args.seed)
  random.seed(args.seed)

  state_size = len(hopfield_network.state)
  neuron_indices = list(range(state_size))

  errors = 0
  for n in range(0, args.num_trials, state_size):
    patterns = np.random.choice([-1, 1], size=(p, args.N))

    hopfield_network.store_patterns(patterns, zero_diagonal=args.zero_diagonal)
    hopfield_network.insert_state(random.choice(patterns))

    previous_neuron_states = hopfield_network.state[neuron_indices]
    hopfield_network.update_neuron_state(neuron_indices)
    updated_neuron_states = hopfield_network.state[neuron_indices]

    errors += np.sum(previous_neuron_states != updated_neuron_states)

  return errors

def print_results(all_errors, args):
  print("\033[92m[Number of errors after {} trails]\033[0m".format(args.num_trials))
  for p, errors in zip(args.all_p, all_errors):
    print('(p={}, N={}): errors={} ({:.4f}%)'.format(p, args.N, errors, (errors/args.num_trials)))
  print()

def plot_results(all_errors, args):
  error_probs = np.array([(errors/args.num_trials) for errors in all_errors])
  alphas = np.array([(p/args.N) for p in args.all_p])

  def qubic_poly(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
  params, paramscov = curve_fit(qubic_poly, alphas, error_probs)
 
  fig, ax = plt.subplots()
  ax.plot(alphas, qubic_poly(alphas, *params), color='red', linewidth=3)
  ax.plot(alphas, error_probs, 'bo', linewidth=2)
  ax.set_xlabel(r'$\alpha$', fontsize=20)
  ax.set_ylabel(r'$P^{t=1}_{error}$', fontsize=20)
  fig.suptitle('{} trails'.format(args.num_trials), fontsize=15)
  plt.grid()
  plt.show()

if __name__ == '__main__':
  from hopfield_network import HopfieldNetwork
  import argparse
  ray.init()

  parser = argparse.ArgumentParser()

  parser.add_argument('--num_trials', type=int, default=100000)
  parser.add_argument('--N', type=int, default=120)
  parser.add_argument('--all_p', nargs='+', type=int, default=[12, 24, 48, 70, 100, 120])
  parser.add_argument('--zero_diagonal', action='store_true')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--print_results', action='store_true')
  parser.add_argument('--plot_results', action='store_true')

  args = parser.parse_args()

  hopfield_network = HopfieldNetwork(args.N)

  all_errors = ray.get([run_trials.remote(hopfield_network, p, args) for p in args.all_p])

  if args.print_results:
    print_results(all_errors, args)
  if args.plot_results:
    plot_results(all_errors, args)
  
