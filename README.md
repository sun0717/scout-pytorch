# scout-pytorch
b站土堆pytorch入门课程
    output        target
    
    选择 (10)      选择 (30)
    
    填空 (10)      填空 (20)
    
    解答 (10)      解答 (50)

Loss: 输出相较于真实的差距

作用： 1. 计算实际输出和目标之间的差距 
      2. 为我们更新更新输出提供一定的依据 (反向传播)

L1loss = (0+0+2) / 3 = 0.6


### 网络训练步骤：
- 准备数据集
```python
# 准备数据集
# 训练数据集
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor(), download=True)
# 测试数据集
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(), download=True)
```
- 准备dataloader
```python
# 加载数据dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
```
- 创建网络模型、损失函数、优化器、训练中的参数
```python
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
```
- 设置训练轮数
```python
for i in range(epoch):
    print('第 {} 轮训练开始'.format(i+1))
    ……
```
- 进入训练状态，去dataloader取数据
```python
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
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

```
- 展示输出
```python
        total_train_step = total_train_step + 1
        # loss加.item，去掉tensor数据类型
        if total_train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
```
- 每轮训练好，进行一次测试,从测试数据集中取数据，计算误差
```python
    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
```

- 保存模型
```python
    torch.save(model, "model_{}.pth".format(i))
    # 方式二是官方推荐的
    torch.save(model.state_dict(), "model_{}.pth".format(i))
    print("模型已保存")
```
