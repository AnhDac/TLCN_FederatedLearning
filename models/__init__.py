import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

# Download and Save MNIST 
data_train = MNIST('~/mnist_data', train=True, download=True)

# Print Data
# print(data_train)
# print(data_train[12])


import matplotlib.pyplot as plt

random_image = data_train[0][0]
random_image_label = data_train[0][1]

# Print the Image using Matplotlib
plt.imshow(random_image)
# print("The label of the image is:", random_image_label)


