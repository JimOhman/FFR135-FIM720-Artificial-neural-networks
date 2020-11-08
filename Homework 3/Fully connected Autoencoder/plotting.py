from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torch


def plot_results(results, save=False, name='default'):
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_ylabel('loss', fontsize=12)
  ax.set_xlabel('batch', fontsize=12)
  for label, result in results.items():
    for values in result.values():
      if values:
        if label != 'test':
          x, y = zip(*values.items())
          ax.plot(x, y, label=label, linewidth=2)
        else:
          ax.axhline(values, linestyle='--', label=label, linewidth=2)
  ax.legend()
  if save:
    plt.savefig('images/loss/{}'.format(name))
  return fig, ax


def plot_bottleneck_outputs(model, dataset, save=False, name='default'):
  device = torch.device('cpu')
  model = model.to(device)

  activation = {}
  def get_activation(name):
    def hook(model, input, output):
      activation[name] = output.detach()
    return hook

  hook_func = get_activation('bottleneck')
  model.bottleneck.register_forward_hook(hook_func)

  images = {}
  for i in range(10):
    idx = dataset.targets == i
    images[i] = dataset.data[idx]

  num_images = 1000
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.set_ylabel('bottleneck output 1', fontsize=12)
  ax.set_xlabel('bottleneck output 0', fontsize=12)
  with torch.no_grad():
    for i in range(10):
      inputs = images[i][:num_images].view(-1, 784).float() / 255
      output = model(inputs)
      bottleneck_output = activation['bottleneck'].numpy()
      x, y = zip(*bottleneck_output)
      ax.scatter(x, y, label=i)
  plt.legend()

  if save:
    plt.savefig('images/scatter/{}'.format(name))


def plot_decoder_outputs(model, dataset, save=False, name='default'):
  device = torch.device('cpu')
  model = model.to(device)

  activation = {}
  def get_activation(name):
    def hook(model, input, output):
      activation[name] = output.detach()
    return hook

  hook_func = get_activation('bottleneck')
  model.bottleneck.register_forward_hook(hook_func)

  images = {}
  decoder_inputs = {}
  with torch.no_grad():
    for i in range(10):
      idx = dataset.targets == i
      inputs = dataset.data[idx].view(-1, 784).float() / 255
      output = model(inputs)

      images[i] = inputs
      average_activation = activation['bottleneck'].mean(dim=0)
      decoder_inputs[i] = average_activation.unsqueeze(0)

  def change_decoder_input():
    def hook(model, input, output):
      output = decoder_inputs[i]
      return output
    return hook

  hook_func = change_decoder_input()
  model.bottleneck.register_forward_hook(hook_func)

  fig, ax = plt.subplots(nrows=10, ncols=1)
  with torch.no_grad():
    for i in range(10):
      dummy_input = images[0][0].unsqueeze(0)
      decoder_output = model(dummy_input)
      image = decoder_output.squeeze().view(28, 28)
      ax[i].imshow(image.numpy(), cmap='gray')
      ax[i].axis('off')

  plt.subplots_adjust(wspace=0, hspace=0)

  if save:
    plt.savefig('images/decoder/{}'.format(name))


def plot_montage(model, dataset, save=False, name='default'):
  device = torch.device('cpu')
  model = model.to(device)

  images = {}
  for i in range(10):
    idx = dataset.targets == i
    images[i] = dataset.data[idx]

  ncols = 8
  input_grid = GridSpec(10, ncols//2, wspace=0, hspace=-0.01, 
                                      left=0.1, right=0.475)
  output_grid = GridSpec(10, ncols//2, wspace=0, hspace=-0.01, 
                                       left=0.525, right=0.9)

  fig = plt.figure(figsize=(25, 25))
  for k in range(0, ncols//2):
    for i in range(10):
      input_image = images[i][k].float() / 255

      with torch.no_grad():
        inputs = input_image.unsqueeze(0).view(-1, 784)
        outputs = model(inputs)
        outputs = outputs.squeeze().view(28, 28)

      input_ax = fig.add_subplot(input_grid[i, k])
      input_ax.imshow(input_image.numpy(), cmap='gray')
      input_ax.set_aspect("auto")
      input_ax.axis('off')

      output_ax = fig.add_subplot(output_grid[i, k])
      output_ax.imshow(outputs.numpy(), cmap='gray')
      output_ax.set_aspect("auto")
      output_ax.axis('off')

  if save:
    plt.savefig('images/montage/{}'.format(name))
