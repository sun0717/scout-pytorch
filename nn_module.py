import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, input):
        # 丐版nn
        output = input + 1
        return output
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
# torch.nn 是对 torch


network = Model()
x = torch.tensor(1.0)
output = network(x)
print(output)