import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import torchvision

# 初始化一个MNIST数据测试集，
test_dataset = torchvision.datasets.MNIST(
    root='/home/lz/Documents/Programs-Released-Codes/Programming-Codes-Python/data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 展示第一张图片
img, label = test_dataset[0]
img = np.reshape(img, (28,28))
plt.imshow(img)
plt.axis('off')

# 超参数T时间常量
T = 100
spikes_array = np.load("s_t_array.npy")
voltage_array = np.load("v_t_array.npy")

spikes_list = spikes_array.tolist()
voltage_list = voltage_array.tolist()


visualizing.plot_2d_heatmap(array=np.asarray(voltage_list), title='Membrane Potentials', xlabel='Simulating Step',
                            ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
visualizing.plot_1d_spikes(spikes=np.asarray(spikes_list), title='Membrane Potentials', xlabel='Simulating Step',
                           ylabel='Neuron Index', dpi=200)

plt.show()