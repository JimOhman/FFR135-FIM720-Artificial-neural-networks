import torch
import torch.nn as nn
import torch.nn.functional as F


def get_output_conv_dims(frame_size, kernel_sizes, strides, padding):
  conv_out_w = frame_size[0]
  conv_out_h = frame_size[1]
  for kernel_size, stride, pad in zip(kernel_sizes, strides, padding):
    conv_out_w = int((conv_out_w - (kernel_size - 1) - 1) / stride + 1)
    conv_out_h = int((conv_out_h - (kernel_size - 1) - 1) / stride + 1)
    if pad:
      conv_out_w += 2
      conv_out_h += 2
  return conv_out_w, conv_out_h


class ModelOne(nn.Module):

  def __init__(self, input_channels, frame_size):
    super(ModelOne, self).__init__()
    self.conv = torch.nn.Conv2d(input_channels, 20, kernel_size=5, stride=1, padding=1)

    self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    kernel_sizes = [5, 2]
    strides = [1, 2]
    padding = [1, 0]

    dims = [frame_size, kernel_sizes, strides, padding]
    conv_out_w, conv_out_h = get_output_conv_dims(*dims)

    conv_out_dim = conv_out_w * conv_out_h * 20

    self.fc1 = torch.nn.Linear(conv_out_dim, 100)

    self.fc2 = torch.nn.Linear(100, 10)

    self.log_softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.conv(x))
    x = self.max_pool(x)
    x = F.relu(self.fc1(x.view(batch_size, -1)))
    return self.log_softmax(self.fc2(x))


class ModelTwo(nn.Module):

  def __init__(self, input_channels, frame_size):
    super(ModelTwo, self).__init__()
    self.conv1 = torch.nn.Conv2d(input_channels, 20, kernel_size=3, stride=1, padding=1, bias=False)
    
    self.batch_norm1 = nn.BatchNorm2d(20)

    self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    self.conv2 = torch.nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1, bias=False)

    self.batch_norm2 = nn.BatchNorm2d(30)

    self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    self.conv3 = torch.nn.Conv2d(30, 50, kernel_size=3, stride=1, padding=1, bias=False)

    self.batch_norm3 = nn.BatchNorm2d(50)

    kernel_sizes = [3, 2, 3, 2, 3]
    strides = [1, 2, 1, 2, 1]
    padding = [1, 0, 1, 0, 1]

    dims = [frame_size, kernel_sizes, strides, padding]
    conv_out_w, conv_out_h = get_output_conv_dims(*dims)

    conv_out_dim = conv_out_w * conv_out_h * 50

    self.fc = torch.nn.Linear(conv_out_dim, 10)

    self.log_softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    batch_size = x.size(0)
    x = self.conv1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)

    x = self.max_pool1(x)

    x = self.conv2(x)
    x = self.batch_norm2(x)
    x = F.relu(x)

    x = self.max_pool2(x)

    x = self.conv3(x)
    x = self.batch_norm3(x)
    x = F.relu(x)
    return self.log_softmax(self.fc(x.view(batch_size, -1)))
