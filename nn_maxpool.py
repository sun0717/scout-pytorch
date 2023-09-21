import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset=dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Ceil_model，卷积核没取满，没有值，继续向下取
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
"""
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Ceil_model，卷积核没取满，没有值，继续向下取
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

model = Model()
output = model(input)
# RuntimeError: "max_pool2d" not implemented for 'Long' 最大池化没法对’Long‘实现，解决：dtype=torch.float32
print(output)
"""

model = Model()
writer = SummaryWriter("./logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

