from tic_tac_toe import Game, Player
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import time


def train(game, players, games=1000, 
                         verbose=False,
                         plot=False,
                         window_size=1000,
                         seed=None):
  if seed is not None:
    np.random.seed(seed)
  
  if plot:
    fig, ax = plt.subplots()
    data = []
    x_data = range(games)

  wins = deque([], maxlen=window_size)
  wins_count = np.zeros(3)
  start_time = time.time()

  for _ in range(games):
    game.play(players, verbose)

    for marker, player in players.items():
      if player.train:
        result = marker * game.winner
        player.learn(game.history[marker], result)

    if len(wins) == window_size:
      wins_count[wins[0]] -= 1
    wins_count[game.winner] += 1
    wins.append(game.winner)

    if plot:
      results = 100*(wins_count/sum(wins_count))
      data.append(results)

    game.reset()

  duration = (time.time()-start_time)
  results = 100*(wins_count/sum(wins_count))

  print("Training finished in {:.2f}s!\n".format(duration))
  print("Results of last {} games".format(window_size))
  print("  Player 1 wins: {:.0f}%".format(results[1]))
  print("  Player 2 wins: {:.0f}%".format(results[-1]))
  print("  Draws: {:.0f}%".format(results[0]))

  if plot:
    labels = ["draws", "player 1 wins", "player 2 wins"]
    for label, y_data in zip(labels, zip(*data)):
      ax.plot(x_data, y_data, label=label)
    ax.set_ylabel("%")
    ax.set_xlabel("game")
    plt.legend()
    plt.show()


def evaluate(game, players, noise=0.,
                            games=1000, 
                            verbose=False,
                            seed=None):
  if seed is not None:
    np.random.seed(seed)

  for player in players.values():
    player.train = False

  msg = "\nEvaluating {} games against {:.2f} noise level."
  print(msg.format(games, noise))
  for marker, player in players.items():
    win_count = np.zeros(3)

    players[marker].epsilon = 0.
    players[-marker].epsilon = noise

    for _ in range(games):
      game.play(players, verbose)
      win_count[game.winner] += 1
      game.reset()

    results = 100*(win_count/games)
    wins = results[marker]
    draws = results[0]
    losses = results[-marker]

    player_id = 2 if marker == -1 else 1
    msg = " Player {} - W:{:.2f}%, D:{:.2f}%, L:{:.2f}%"
    print(msg.format(player_id, wins, draws, losses))


def play_against(game, players, control=0,
                                show_values=False):

  np.set_printoptions(precision=1, suppress=True)
  
  def get_user_input(board):
    legal_moves, = np.where(board==0)
    msg = "Select a valid move: "
    move = (int(input("\n" + msg)) - 1)
    while move not in legal_moves:
      move = (int(input(msg)) - 1)
    return move

  for player in players.values():
    player.train = False
    player.verbose = show_values

  if control:
    if control == 'both':
      for player in players.values():
        player.find_move = get_user_input
    else:
      players[control].find_move = get_user_input
    print("\nUser controls {}.".format(control))
  else:
    print("\nNo user control.")

  game.play(players, verbose=True)


def save_q_table(path, q_table):
  board_func = lambda b: np.frombuffer(b).reshape(3,3)
  value_func = lambda v: v.reshape(3,3)
  boards = [board_func(b) for b in q_table.keys()]
  values = [value_func(v) for v in q_table.values()]
  q_table = np.vstack((np.hstack(boards), np.hstack(values)))
  np.savetxt(path, q_table, delimiter=',')


if __name__ == '__main__':

  first_player_params = {"train": True,
                         "epsilon": 0.7, 
                         "epsilon_decay": 0.9999,
                         "min_epsilon": 0.,
                         "gamma": 1., 
                         "alpha": 0.4,
                         "initial_value": 0.1,
                         "verbose": False}

  second_player_params = {"train": True,
                          "epsilon": 0.7, 
                          "epsilon_decay": 0.9999,
                          "min_epsilon": 0.,
                          "gamma": 1., 
                          "alpha": 0.4,
                          "initial_value": 0.1,
                          "verbose": False}

  first_player = Player(**first_player_params)
  second_player = Player(**second_player_params)

  players = {1: first_player, -1: second_player}
  game = Game()

  train(game, players, games=20000,
                       verbose=False,
                       plot=True,
                       window_size=1000,
                       seed=None)

  first_player.save_state_dict('states/first_player')
  second_player.save_state_dict('states/second_player')

  save_q_table('q_tables/player1.csv', first_player.q_table)
  save_q_table('q_tables/player2.csv', second_player.q_table)

  noise_levels = [0.1, 1.]
  for noise in noise_levels:
    evaluate(game, players, noise=noise, 
                            games=10000, 
                            verbose=False, 
                            seed=None)

  first_player = Player()
  first_player.load_state_dict('states/first_player')

  second_player = Player()
  second_player.load_state_dict('states/second_player')

  players = {1: first_player, -1: second_player}
  game = Game()

  play_against(game, players, control='both', 
                              show_values=True)
