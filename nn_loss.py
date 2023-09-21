import torch
from torch.nn import L1Loss
from torch import nn
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 2 / 3 = 0.6667 process: (1-1+2-2+3-5) / 3
loss = L1Loss()
result = loss(inputs, targets)

# MSELoss 均方差
# (0 + 0 + 2^2) / 3 = 1.3333
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

# 交叉熵损失 CrossEntropyLoss() 训练分类任务，10个类别

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y  =torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

## 总结： 损失函数：注意根据需求使用，什么时候使用什么损失函数.注意loss_function，输入的情况是什么，输出的情况是什么