import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# train_data = torchvision.datasets.ImageNet('./data_image_net', split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),download=True)

# (1) vgg16_true.add_module('add_linear', nn.Linear(1000, 10))

# Expected output:
# (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
# (1): ReLU(inplace=True)
# (2): Dropout(p=0.5, inplace=False)
# (3): Linear(in_features=4096, out_features=4096, bias=True)
# (4): ReLU(inplace=True)
# (5): Dropout(p=0.5, inplace=False)
# (6): Linear(in_features=4096, out_features=1000, bias=True)
# )
# (add_linear): Linear(in_features=1000, out_features=10, bias=True)

# 如果不想让加的层在外面，加到 classifier 中
# (2) vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))

# Expected output:
# (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
# (1): ReLU(inplace=True)
# (2): Dropout(p=0.5, inplace=False)
# (3): Linear(in_features=4096, out_features=4096, bias=True)
# (4): ReLU(inplace=True)
# (5): Dropout(p=0.5, inplace=False)
# (6): Linear(in_features=4096, out_features=1000, bias=True)
# (add_linear): Linear(in_features=1000, out_features=10, bias=True)
# )

# 如果不想要序号为add_linear
vgg16_false.classifier[6] == nn.Linear(4096, 10)
print(vgg16_true)