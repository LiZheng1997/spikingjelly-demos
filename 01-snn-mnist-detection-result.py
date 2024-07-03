import torch
import torch.nn as nn
import numpy as np
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import torchvision

# 初始化一个MNIST数据测试集，
test_dataset = torchvision.datasets.MNIST(
    root='/home/lz/Documents', # 更改为你自己的MNIST数据集的绝对路径
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True # 如果你不需要自动帮你下载数据，则改为False
)

# 展示第一张图片
img, label = test_dataset[0]
img = np.reshape(img, (28,28))
plt.imshow(img)
plt.axis('off')


T = 100 # 超参数T时间常量
#下面两个数组的数据来源于lif_fc_mnist.py脚本训练的模型推理得到的结果
spikes_array = np.load("s_t_array.npy") # 读取spikes array的数据
voltage_array = np.load("v_t_array.npy") # 读取voltage array的数据

# 可视化数据结果，heatmap和spikes
visualizing.plot_2d_heatmap(array=voltage_array, title='Membrane Potentials', xlabel='Simulating Step',
                            ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
visualizing.plot_1d_spikes(spikes=spikes_array, title='Membrane Potentials', xlabel='Simulating Step',
                           ylabel='Neuron Index', dpi=200)

plt.show()