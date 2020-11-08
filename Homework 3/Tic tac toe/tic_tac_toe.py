import numpy as np
import pickle


class Player():

  def __init__(self, 
               train=False, 
               epsilon=0., 
               epsilon_decay=1.,
               min_epsilon=0.,
               gamma=1., 
               alpha=0.1,
               initial_value=0.,
               verbose=False):
    self.q_table = {}
    self.train = train
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.min_epsilon = min_epsilon
    self.gamma = gamma
    self.alpha = alpha
    self.initial_value = initial_value
    self.verbose = verbose

  def find_move(self, board):
    board_key = board.tobytes()
    values = self.q_table.get(board_key)

    if values is None:
      legal_moves, = np.where(board==0)
      values = np.full(9, np.nan)
      values[legal_moves] = self.initial_value
      self.q_table[board_key] = values
      move = np.random.choice(legal_moves, 1)
    else:
      if np.random.rand() < self.epsilon:
        legal_moves, = np.where(board==0)
        move = np.random.choice(legal_moves, 1)
      else:
        move = np.nanargmax(values)

    if self.train:
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.epsilon, self.min_epsilon)

    if self.verbose:
      print(values.reshape(3,3), "\n")
    return move

  def learn(self, history, result):
    reward = result
    max_next_value = 0
    for board_key, move in reversed(history):
      value_estimate = reward + self.gamma * max_next_value
      values = self.q_table.get(board_key)
      values[move] += self.alpha * (value_estimate - values[move])
      max_next_value = np.nanmax(values)
      reward = 0

  def state_dict(self):
    state_dict = {'q_table': self.q_table,
                  'epsilon': self.epsilon,
                  'epsilon_decay': self.epsilon_decay,
                  'min_epsilon': self.min_epsilon,
                  'gamma': self.gamma,
                  'alpha': self.alpha,
                  'initial_value': self.initial_value}
    return state_dict

  def save_state_dict(self, path):
    with open(path, 'wb') as handle:
      protocol = pickle.HIGHEST_PROTOCOL
      pickle.dump(self.state_dict(), handle, protocol=protocol)

  def load_state_dict(self, path):
    with open(path, 'rb') as handle:
      state_dict = pickle.load(handle)
    for k, v in state_dict.items():
      setattr(self, k, v)


class Game():

  def __init__(self):
    self.board = np.zeros(9)
    self.history = {1:[], -1:[]}
    self.winner = 0
    self.turn = 1

  def play(self, players, verbose=False):
    for move_number in range(9):
      board_key = self.board.tobytes()
      player = players[self.turn]

      if verbose:
        print(self.board.reshape(3,3), "\n")

      move = player.find_move(self.board)
      self.board[move] = self.turn

      self.history[self.turn].append([board_key, move])

      if move_number >= 4:
        if self.did_win(move):
          self.winner = self.turn
          break
      self.turn *= -1

    if verbose:
      print(self.board.reshape(3,3), "\n")
      if self.winner:
        winner = 2 if self.winner == -1 else 1
        print("Player {} won!".format(winner))
      else:
        print("Draw!")

  def did_win(self, move):
    win = False
    board = self.board.reshape(3,3)
    i, j = np.unravel_index(move, (3,3))
    if abs(np.sum(board[i,:])) == 3:
      win = True
    elif abs(np.sum(board[:,j])) == 3:
      win = True
    elif (i, j) in [(0,0), (1,1), (2,2)]:
      if abs(np.sum(board.diagonal())) == 3:
        win = True
    elif (i, j) in [(0,2), (1,1), (2,0)]:
      if abs(np.sum(np.rot90(board).diagonal())) == 3:
        win = True
    return win

  def reset(self):
    self.board = np.zeros(9)
    self.history = {-1:[], 1:[]}
    self.winner = 0
    self.turn = 1
