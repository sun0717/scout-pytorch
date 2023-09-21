import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=False)
# drop_last 丢弃多余的
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = Model()

writer = SummaryWriter("./logs")
step = 0
# Shape: input (N:batch_size, C:输入通道, H_in, W_in)
# Shape: onput (N:batch_size, C:输出通道, H_out, W_out)
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])

    # 彩色图像是三通道,reshape不算严谨,如果不知道第一个值是多少就写-1，会根据后面的值计算
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1