import numpy as np
import random


class DataHandler():

  def __init__(self, inputs, targets, p_train=1., shuffle=True):
    self.full = (inputs.copy(), targets.copy())
    self.size = inputs.shape[0]

    if shuffle:
      indices = list(range(self.size))
      random.shuffle(indices)
      inputs, targets = inputs[indices], targets[indices]

    train_idx = int(self.size * p_train)
    self.train = [inputs[:train_idx], targets[:train_idx]]
    self.train_size = self.train[0].shape[0]
    self.train_indices = list(range(self.train_size))

    val_idx = self.size - (self.size - train_idx)//2
    self.validation = [inputs[train_idx:val_idx], targets[train_idx:val_idx]]
    self.val_size = self.validation[0].shape[0]

    self.test = [inputs[val_idx:], targets[val_idx:]]
    self.test_size = self.test[0].shape[0]

    self.sampling_index = 0
    self.sampled = 0

  def normalize(self):
    train_inputs, _ = self.train
    val_inputs, _ = self.validation
    test_inputs, _ = self.test

    self.mean = np.mean(train_inputs, axis=0)
    self.std = np.std(train_inputs)

    self.train[0] = (train_inputs - self.mean) / self.std
    self.validation[0] = (val_inputs - self.mean) / self.std
    self.test[0] = (test_inputs - self.mean) / self.std

  def sample(self, batch_size, shuffle_each_epoch=True):
    inputs, targets = self.train
    start = self.sampling_index
    end = self.sampling_index + batch_size
    batch = [inputs[start:end], targets[start:end]]
    self.sampling_index = (self.sampling_index + batch_size) % self.train_size

    if shuffle_each_epoch:
      self.sampled += len(batch[0])
      if self.sampled % self.train_size == 0:
        random.shuffle(self.train_indices)
        self.train[0] = inputs[self.train_indices]
        self.train[1] = targets[self.train_indices]
    return batch
