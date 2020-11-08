import numpy as np
import ray


@ray.remote
def run_trial(hopfield_network, args, seed):
  np.random.seed(seed)

  patterns = np.random.choice([-1, 1], size=(args.p, args.N))

  hopfield_network.store_patterns(patterns)
  hopfield_network.insert_state(patterns[0].copy())

  hopfield_network.run_asynchronous_updates(times=args.times, beta=args.beta)

  order_parameter = (1/N) * np.dot(patterns[0], state)

  return order_parameter


if __name__ == '__main__':
  from hopfield_network import HopfieldNetwork
  import argparse
  ray.init()

  parser = argparse.ArgumentParser()

  parser.add_argument('--times', type=int, default=2*10**5)
  parser.add_argument('--repeat', type=int, default=100)
  parser.add_argument('--beta', type=float, default=2)
  parser.add_argument('--N', type=int, default=200)
  parser.add_argument('--p', nargs='+', type=int, default=[7, 45])
  parser.add_argument('--print_result', action='store_true')

  args = parser.parse_args()

  hopfield_network = HopfieldNetwork(args.N)
  average_order_parameters = {}

  for p in args.p:
    order_parameters = ray.get([run_trial.remote(hopfield_network, args, seed=n) for n in range(args.repeat)])

    average_order_parameter = np.mean(order_parameters)
    average_order_parameters[p] = average_order_parameter

    if args.print_result:
      print(average_order_parameter)
