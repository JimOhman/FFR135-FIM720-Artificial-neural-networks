import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoderNetwork(nn.Module):

  def __init__(self, input_dim, bottleneck_dim):
    super(AutoEncoderNetwork, self).__init__()

    self.bottleneck_dim = bottleneck_dim

    self.fc1 = torch.nn.Linear(input_dim, 150)
    self.fc2 = torch.nn.Linear(150, 50)

    self.bottleneck = torch.nn.Linear(50, bottleneck_dim)

    self.fc4 = torch.nn.Linear(bottleneck_dim, 50)
    self.fc5 = torch.nn.Linear(50, 150)
    self.fc6 = torch.nn.Linear(150, 784)

    self.leaky_relu = torch.nn.LeakyReLU(0.01)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.bottleneck(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    return self.leaky_relu(self.fc6(x))


class SmallAutoEncoderNetwork(nn.Module):

  def __init__(self, input_dim, bottleneck_dim):
    super(SmallAutoEncoderNetwork, self).__init__()

    self.bottleneck_dim = bottleneck_dim

    self.fc1 = torch.nn.Linear(input_dim, 50)
    torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)

    self.bottleneck = torch.nn.Linear(50, bottleneck_dim)
    torch.nn.init.xavier_uniform_(self.bottleneck.weight, gain=1.0)

    self.fc3 = torch.nn.Linear(bottleneck_dim, 784)
    torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)

    self.leaky_relu = torch.nn.LeakyReLU(0.01)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.bottleneck(x))
    return self.leaky_relu(self.fc3(x))
