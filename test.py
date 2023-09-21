import torch
import torchvision.transforms
from PIL import Image
from torch import nn

device = torch.device("cuda")
# 定义
image_path = "./imgs/airplane.jpg"
image = Image.open(image_path)

# 测试数据集
# png 格式是四个通道。除了RGB三通道外，还有一个透明度通道
# image = image.convert('RBG')
# print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()
                                            ])

image = transform(image)
print(image.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        input = self.model1(input)
        return input

model = torch.load("model_9.pth", map_location=torch.device('cpu'))
# model = torch.load("model_9.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))