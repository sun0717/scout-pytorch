from tensorboardX import SummaryWriter
import time
from model import *
import torchvision.datasets
from torch.utils.data import DataLoader
# 准备数据集
# 训练数据集
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor(), download=True)
# 测试数据集
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(), download=True)
# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集的长度为:{}'.format(train_data_size))
print('测试数据集的长度为:{}'.format(test_data_size))

# 加载数据dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 模型
model = Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网路的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的论述
epoch = 10

# 添加tensorboard
writer = SummaryWriter('./logs_train')
start_time = time.time()
# 训练步骤开始
for i in range(epoch):
    print('第 {} 轮训练开始'.format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        # 损失值, 预测的输出和真实的 target 放进入
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        # 反向传播，得到每一个参数节点的梯度
        loss.backward()
        # 对其中的参数进行优化
        optimizer.step()

        total_train_step = total_train_step + 1
        # loss加.item，去掉tensor数据类型
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print("整体测试集上的Loss: {}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")

writer.close()