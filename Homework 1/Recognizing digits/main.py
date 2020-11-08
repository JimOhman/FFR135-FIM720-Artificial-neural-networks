import matplotlib.pyplot as plt
from matplotlib import animation


def plot_state(state, shape, ax=None, title='', show=False):
  if ax is None:
    fig, ax = plt.subplots()
  image = ax.imshow(state.reshape(shape), animated=True, cmap='gray')
  title = ax.text(0.5, 1.05, title, fontsize=15, ha="center", transform=ax.transAxes)
  if show:
    plt.show()
  return [image, title]

def animate_states(states, shape, interval=1):
  fig, axes = plt.subplots(nrows=1, ncols=2)
  container = []

  plot_state(states[0], shape, ax=axes[0], title='T=0')
  for t in range(0, len(states), interval):
    artists = plot_state(states[t], shape, ax=axes[1], title='T={}'.format(t))
    container.append(artists)

  ani = animation.ArtistAnimation(fig, container, repeat=False, interval=5, blit=False)
  plt.show()


if __name__ == '__main__':
  from hopfield_network import HopfieldNetwork
  import numpy as np
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('--times', type=int, default=1000)
  parser.add_argument('--beta', type=float, default=None)
  parser.add_argument('--animate', action='store_true')
  parser.add_argument('--interval', type=int, default=10)
  parser.add_argument('--shape', nargs='+', type=int, default=(16, 10))
  parser.add_argument('--print_final_states', action='store_true')

  args = parser.parse_args()

  with open('distorted_digits.txt') as distorted_digits_txt:
    distorted_digits = []
    for digit in distorted_digits_txt.read().split('\n\n'):
      exec('distorted_digits.append(np.array({}).flatten())'.format(digit))

  with open('digits.txt') as digits_txt:
    digits = []
    for digit in digits_txt.read().split('\n\n'):
      exec('digits.append(np.array({}).flatten())'.format(digit))
    digits = np.array(digits)

  size = len(digits[0])
  hopfield_network = HopfieldNetwork(size)

  hopfield_network.store_patterns(digits)

  for distorted_digit in distorted_digits:
    hopfield_network.insert_state(distorted_digit)

    states = hopfield_network.run_asynchronous_updates(times=args.times, beta=args.beta, return_states=True)

    if args.animate:
      animate_states(states, interval=args.interval, shape=args.shape)

    if args.print_final_states:
      print(repr(hopfield_network.state.reshape(args.shape)))
      print()

